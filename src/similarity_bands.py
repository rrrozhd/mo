"""Split each reported CYP metric into novelty bands.

A single test-set AUC averages over molecules that look nothing like the training
set and molecules that nearly duplicate it. This re-scores the frozen predictions
inside bands of nearest-training similarity, reusing the same scaffold-bootstrap
evaluation, so the reported number can be read as the average it is.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .chemistry import fingerprints, nearest_distances
from .benchmark import verify_run
from .evaluate import ARMS, LLM_ARMS, evaluate

BANDS = (("<0.4", 0.0, 0.4), ("0.4-0.6", 0.4, 0.6), (">=0.6", 0.6, 1.01))


def novelty(query_smiles, train_smiles):
    """Max Tanimoto from each query to the training pool."""
    return 1.0 - nearest_distances(fingerprints(query_smiles), fingerprints(train_smiles))


def decompose(run_dir, artifacts, bootstrap=1000, seed=42, arms=ARMS):
    run_dir, artifacts = Path(run_dir), Path(artifacts)
    cohort = verify_run(artifacts, run_dir)
    cohort["molecule_id"] = cohort.molecule_id.astype(str)
    with (run_dir / "predictions.jsonl").open() as source:
        records = [json.loads(line) for line in source if line.strip()]

    manifest = pd.read_csv(artifacts / "manifest.csv", dtype={"molecule_id": str})
    train = manifest.loc[manifest.split == "train", "smiles"].tolist()
    cohort["novelty"] = novelty(cohort.smiles.tolist(), train)

    frames, contrast_frames = [], []
    for label, low, high in BANDS:
        band = cohort[(cohort.novelty >= low) & (cohort.novelty < high)]
        if band.empty:
            continue
        ids = set(band.molecule_id)
        metrics, contrasts, _ = evaluate(band.drop(columns="novelty"),
                                         [r for r in records if str(r["molecule_id"]) in ids],
                                         bootstrap, seed, arms)
        for table in (metrics, contrasts):
            if not table.empty:
                table.insert(0, "band", label)
                table.insert(1, "band_median_novelty", round(float(band.novelty.median()), 4))
        frames.append(metrics)
        contrast_frames.append(contrasts)

    whole, whole_contrasts, _ = evaluate(cohort.drop(columns="novelty"), records,
                                         bootstrap, seed, arms)
    for table in (whole, whole_contrasts):
        if not table.empty:
            table.insert(0, "band", "all")
            table.insert(1, "band_median_novelty", round(float(cohort.novelty.median()), 4))

    # the bands must partition the cohort, or a band filter has dropped molecules
    # With more than one arm there is a row per arm, so the key needs that dimension.
    key = ["arm", "enzyme", "metric"]
    banded = pd.concat(frames, ignore_index=True).groupby(key).n_evaluable.sum()
    if not banded.equals(whole.set_index(key).n_evaluable.reindex(banded.index)):
        raise ValueError("Novelty bands do not partition the evaluated cohort")
    contrasts = pd.concat([whole_contrasts, *contrast_frames], ignore_index=True)
    return cohort, pd.concat([whole, *frames], ignore_index=True), contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/full"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("audit/novelty_decomposition.csv"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=LLM_ARMS)
    args = parser.parse_args()

    cohort, metrics, contrasts = decompose(args.run_dir, args.artifacts,
                                           args.bootstrap, args.seed, tuple(args.arms))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    if not contrasts.empty:
        contrasts.to_csv(args.output.with_name("novelty_contrasts.csv"), index=False)
    cohort[["molecule_id", "novelty"]].to_csv(
        args.output.with_name("test_novelty.csv"), index=False)

    table = contrasts if not contrasts.empty else metrics
    columns = ["band", "enzyme", "n_evaluable", "value", "ci_low", "ci_high"]
    if not contrasts.empty:
        columns.insert(1, "contrast")
        columns.append("excludes_zero")
    shown = table.loc[table.metric == "auc", columns]
    print(f"Test-set novelty: median {cohort.novelty.median():.3f}, "
          f"{(cohort.novelty >= 0.6).mean():.1%} at or above 0.6")
    print(shown.to_string(index=False))


if __name__ == "__main__":
    main()
