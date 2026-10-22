"""Measure how novel each TDC benchmark's scaffold-split test set actually is.

No training and no network inference: every number is computed from the dataset's
own molecules. A dataset the default loader cannot construct is recorded as a
skip with its reason, never silently dropped.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from ..chemistry import MORGAN_BITS, MORGAN_RADIUS, nearest_distances

# shortcut: novelty is sampled, not exhaustive, so one dataset stays under a
# minute. Raise both caps for the final published table.
MAX_QUERIES = 2000
MAX_POOL = 20000
SEED = 42

FIELDS: list[str] = [
    "dataset", "loader", "label_name", "status", "error",
    "n_molecules", "n_skeletons", "skeleton_overcount",
    "n_train", "n_test", "n_test_novel_by_smiles", "n_skeleton_leak",
    "n_queries", "n_pool", "n_parse_fail",
    "median_novelty", "frac_ge_05", "frac_ge_07", "frac_ge_09",
]

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def _parsed(smiles):
    """(fingerprint, skeleton key) for one structure, or (None, None)."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None, None
        try:
            skeleton = Chem.MolToInchiKey(mol)[:14]
        except Exception:
            skeleton = None
    return _MORGAN.GetFingerprint(mol), skeleton


def catalog():
    """Every single-prediction molecular dataset, paired with its loader class."""
    from tdc.metadata import (
        adme_dataset_names, hts_dataset_names, toxicity_dataset_names,
    )
    groups = (("ADME", adme_dataset_names), ("Tox", toxicity_dataset_names),
              ("HTS", hts_dataset_names))
    return [(name, loader) for loader, names in groups for name in names]


def audit_dataset(name, loader_name, path, structure_column="Drug", label_name=None):
    """One row of the novelty spectrum. Loader failures become a skip row.

    label_name selects one endpoint of a multi-label dataset; the default loader
    call fails on those, which is what the agent resolves.
    """
    from tdc import single_pred

    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update(dataset=name, loader=loader_name, label_name=label_name or "",
               status="ok")
    try:
        kwargs = {"label_name": label_name} if label_name else {}
        data = getattr(single_pred, loader_name)(name=name, path=path, **kwargs)
        split = data.get_split(method="scaffold", seed=SEED)
    except Exception as error:
        # shortcut: a multi-label set is audited on the first endpoint TDC lists,
        # not all of them. Widen this to sweep every endpoint of tox21/toxcast.
        if label_name is None and "label name" in str(error).lower():
            from tdc.utils import retrieve_label_name_list

            labels = retrieve_label_name_list(name)
            if labels:
                return audit_dataset(name, loader_name, path, structure_column,
                                     label_name=labels[0])
        row.update(status="skipped", error=f"{type(error).__name__}: {error}"[:300])
        return row

    train = set(split["train"][structure_column])
    test = set(split["test"][structure_column])
    rng = np.random.default_rng(SEED)

    parsed = {s: _parsed(s) for s in train | test}
    row["n_parse_fail"] = sum(1 for fp, _ in parsed.values() if fp is None)
    skeletons = {s: k for s, (_, k) in parsed.items() if k}
    row["n_molecules"] = len(parsed)
    row["n_skeletons"] = len(set(skeletons.values()))
    row["skeleton_overcount"] = len(skeletons) - row["n_skeletons"]
    row["n_train"], row["n_test"] = len(train), len(test)

    train_skeletons = {skeletons[s] for s in train if s in skeletons}
    novel = [s for s in sorted(test - train) if parsed[s][0] is not None]
    row["n_test_novel_by_smiles"] = len(novel)
    row["n_skeleton_leak"] = sum(
        1 for s in novel if skeletons.get(s) in train_skeletons
    )

    pool = [parsed[s][0] for s in sorted(train) if parsed[s][0] is not None]
    if not pool or not novel:
        row.update(status="skipped", error="no comparable train/test structures")
        return row
    if len(pool) > MAX_POOL:
        pool = [pool[i] for i in rng.choice(len(pool), MAX_POOL, replace=False)]
    if len(novel) > MAX_QUERIES:
        novel = [novel[i] for i in rng.choice(len(novel), MAX_QUERIES, replace=False)]
    row["n_pool"], row["n_queries"] = len(pool), len(novel)

    novelty = 1.0 - nearest_distances([parsed[s][0] for s in novel], pool)
    row["median_novelty"] = round(float(np.median(novelty)), 4)
    for cut in (0.5, 0.7, 0.9):
        row[f"frac_ge_0{int(cut * 10)}"] = round(float((novelty >= cut).mean()), 4)
    return row


def sweep(output, path, only=None):
    """Append one row per dataset, skipping datasets already in the file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with output.open() as stream:
            done = {row["dataset"] for row in csv.DictReader(stream)}
    targets = [(n, l) for n, l in catalog() if not only or n in only]

    with output.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for name, loader_name in targets:
            if name in done:
                continue
            row = audit_dataset(name, loader_name, path)
            writer.writerow(row)
            stream.flush()
            print(f"{row['status']:8} {name:45} "
                  f"median={row['median_novelty']} leak={row['n_skeleton_leak']}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="audit/novelty_spectrum.csv")
    parser.add_argument("--path", default="notebooks/data")
    parser.add_argument("--only", nargs="*", help="audit just these dataset names")
    args = parser.parse_args()
    sweep(args.output, args.path, only=set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
