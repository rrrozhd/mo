"""Train five frozen CYP inhibition baselines on one global scaffold split."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from ..chemistry.features import (
    DESCRIPTOR_NAMES, ENZYMES, MORGAN_BITS, MORGAN_RADIUS, canonical_smiles,
    featurize, fingerprints, nearest_distances,
)

MODEL_MAPPING = {
    "CYP1A2": "lightgbm", "CYP2C9": "xgboost", "CYP2C19": "random_forest",
    "CYP2D6": "random_forest", "CYP3A4": "lightgbm",
}


def merge_datasets(datasets):
    """Merge canonical identities; absent/conflicting labels remain unknown."""
    molecules = {}
    observations = {}
    report = {"input_rows": {}, "invalid_smiles": [], "invalid_labels": [],
              "duplicate_observations": 0, "conflicts": []}
    for enzyme in ENZYMES:
        frame = datasets[enzyme]
        report["input_rows"][enzyme] = len(frame)
        for row_number, (raw_smiles, label) in enumerate(zip(frame["Drug"], frame["Y"])):
            try:
                smiles = canonical_smiles(raw_smiles)
            except ValueError:
                report["invalid_smiles"].append(
                    {"enzyme": enzyme, "row": row_number, "smiles": str(raw_smiles)}
                )
                continue
            molecules.setdefault(smiles, {key: np.nan for key in ENZYMES})
            if pd.isna(label):
                continue
            if label not in (0, 1):
                report["invalid_labels"].append(
                    {"enzyme": enzyme, "row": row_number, "label": str(label)}
                )
                continue
            key = (smiles, enzyme)
            if key in observations:
                report["duplicate_observations"] += 1
            observations.setdefault(key, set()).add(int(label))
    for (smiles, enzyme), labels in observations.items():
        if len(labels) == 1:
            molecules[smiles][enzyme] = labels.pop()
        else:
            report["conflicts"].append({"smiles": smiles, "enzyme": enzyme})
    rows = [{"smiles": smiles, **molecules[smiles]} for smiles in sorted(molecules)]
    union = pd.DataFrame(rows, columns=["smiles", *ENZYMES])
    report["canonical_molecules"] = len(union)
    report["observed_labels"] = {enzyme: int(union[enzyme].notna().sum()) for enzyme in ENZYMES}
    report["identity_policy"] = "RDKit canonical isomeric SMILES; salts/tautomers not normalized"
    report["conflict_policy"] = "Set conflicting endpoint labels to missing; retain other labels"
    return union, report


def scaffold_manifest(union, seed=42):
    from tdc.utils import create_scaffold_split

    union = union.sort_values("smiles").reset_index(drop=True).copy()
    if union.empty or union["smiles"].duplicated().any():
        raise ValueError("Scaffold splitting needs a nonempty union with unique identities")
    union["molecule_id"] = union["smiles"].map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    union["scaffold"] = union["smiles"].map(
        lambda value: MurckoScaffold.MurckoScaffoldSmiles(
            mol=Chem.MolFromSmiles(value), includeChirality=False
        )
    )
    split = create_scaffold_split(union, seed=seed, frac=[0.7, 0.1, 0.2], entity="smiles")
    manifest = pd.concat([frame.assign(split=name) for name, frame in split.items()], ignore_index=True)
    manifest = manifest.sort_values("smiles").reset_index(drop=True)
    assert len(manifest) == len(union), "TDC omitted molecules during scaffold splitting"
    assert not manifest["molecule_id"].duplicated().any(), "Identity overlap across split"
    assert manifest.groupby("scaffold")["split"].nunique().max() == 1, "Scaffold leakage"
    return manifest[["molecule_id", "smiles", "scaffold", "split", *ENZYMES]]


def classifier(enzyme, seed, jobs, trees=None):
    family = MODEL_MAPPING[enzyme]
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees or 300, min_samples_leaf=2, n_jobs=jobs, random_state=seed
        )
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=trees or 600, learning_rate=0.05, num_leaves=63,
            n_jobs=jobs, random_state=seed, verbosity=-1,
        )
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=trees or 600, learning_rate=0.05, max_depth=7,
        n_jobs=jobs, random_state=seed, tree_method="hist", eval_metric="logloss",
    )


def train(data_dir, output_dir, seed=42, jobs=4, trees=None):
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output path is not empty: {output_dir}. Use a new output directory for retraining.")

    from tdc.single_pred import ADME

    data_dir = Path(data_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        enzyme: ADME(name=f"{enzyme}_Veith", path=str(data_dir)).get_data()
        for enzyme in ENZYMES
    }
    dataset_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for enzyme in ENZYMES
        for path in [data_dir / f"{enzyme.lower()}_veith.tab"]
    }
    union, cleaning = merge_datasets(datasets)
    manifest = scaffold_manifest(union, seed)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    manifest.loc[manifest["split"] == "test"].to_csv(output_dir / "test.csv", index=False)
    print(f"Merged {len(manifest)} molecules; splits: {manifest['split'].value_counts().to_dict()}", flush=True)
    print(f"Featurizing Morgan {MORGAN_BITS} + {len(DESCRIPTOR_NAMES)} RDKit descriptors", flush=True)
    features = featurize(manifest["smiles"])
    all_fps = fingerprints(manifest["smiles"])
    metrics = []
    estimator_parameters = {}
    for enzyme in ENZYMES:
        observed = manifest[enzyme].notna()
        train_mask = observed & manifest["split"].eq("train")
        train_labels = manifest.loc[train_mask, enzyme].astype(int).to_numpy()
        if len(np.unique(train_labels)) != 2:
            raise ValueError(f"{enzyme}: training split must contain both label classes")
        print(f"Training {enzyme}: {MODEL_MAPPING[enzyme]}, n={int(train_mask.sum())}", flush=True)
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            classifier(enzyme, seed, jobs, trees),
        )
        model.fit(features[train_mask], train_labels)
        estimator_parameters[enzyme] = {
            key: "NaN" if isinstance(value, float) and np.isnan(value) else value
            for key, value in model[-1].get_params().items()
        }
        validation_scores = []
        for split in ("valid", "test"):
            mask = observed & manifest["split"].eq(split)
            labels = manifest.loc[mask, enzyme].astype(int).to_numpy()
            if not len(labels):
                raise ValueError(f"{enzyme}: {split} split has no observed labels")
            scores = model.predict_proba(features[mask])[:, 1]
            if split == "valid":
                validation_scores = scores.tolist()
            has_both_classes = len(np.unique(labels)) == 2
            metrics.append({
                "enzyme": enzyme, "model": MODEL_MAPPING[enzyme], "split": split,
                "n": len(labels), "positive_count": int(labels.sum()),
                "auc": float(roc_auc_score(labels, scores)) if has_both_classes else np.nan,
                "ap": float(average_precision_score(labels, scores)) if has_both_classes else np.nan,
            })
        train_fps = [fp for fp, keep in zip(all_fps, train_mask) if keep]
        distances = nearest_distances(train_fps, train_fps, exclude_self=True)
        bundle = {
            "model": model, "descriptor_names": list(DESCRIPTOR_NAMES),
            "rdkit_version": rdBase.rdkitVersion,
            "train_smiles": manifest.loc[train_mask, "smiles"].tolist(),
            "domain_threshold": float(np.quantile(distances, 0.95)),
            "validation_scores": validation_scores, "enzyme": enzyme,
        }
        joblib.dump(bundle, model_dir / f"{enzyme}.joblib", compress=3)
        pd.DataFrame(metrics).to_csv(output_dir / "train_metrics.csv", index=False)
        print(f"Saved {enzyme}; test AP={metrics[-1]['ap']:.4f}", flush=True)
    feature_names = [f"morgan_{index}" for index in range(MORGAN_BITS)] + list(DESCRIPTOR_NAMES)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "datasets": {enzyme: f"{enzyme}_Veith" for enzyme in ENZYMES},
        "dataset_sha256": dataset_hashes,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ("chemistry/features.py", "prediction/train.py")
        },
        "split": {"method": "TDC global union scaffold split", "fractions": [0.7, 0.1, 0.2],
                  "counts": manifest["split"].value_counts().to_dict(),
                  "identity_overlap": 0, "scaffold_overlap": 0},
        "cleaning": cleaning, "model_mapping": MODEL_MAPPING,
        "estimator_parameters": estimator_parameters,
        "estimator_parameters_encoding": "get_params values; IEEE NaN represented by the string NaN",
        "trees_override": trees,
        "feature_schema": {"morgan_radius": MORGAN_RADIUS, "morgan_bits": MORGAN_BITS,
                           "descriptor_names": list(DESCRIPTOR_NAMES),
                           "descriptor_count": len(DESCRIPTOR_NAMES), "feature_names": feature_names,
                           "feature_count": len(feature_names), "rdkit_version": rdBase.rdkitVersion},
        "descriptor_expectation": {"handoff_count": 210, "matches": len(DESCRIPTOR_NAMES) == 210},
        "imputation": "Per-endpoint training-only median; entirely missing columns filled with zero",
        "applicability_domain": "95th percentile of leave-one-out nearest training Tanimoto distances",
        "ranking_reference": "Per-endpoint observed validation set scores; not calibrated probabilities",
        "versions": {name: version(name) for name in
                     ("PyTDC", "rdkit", "numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "joblib")},
        "manifest_sha256": hashlib.sha256((output_dir / "manifest.csv").read_bytes()).hexdigest(),
        "caveats": [
            "Exploratory result: model families were selected using previous feasibility test performance.",
            "Frozen per-enzyme model mapping; no new search on validation or test data.",
            "Canonical isomeric identities and Murcko scaffolds are disjoint across global splits.",
            "Single seed, uncalibrated ranking scores, inhibition labels only; no clinical claims.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--trees", type=int, help="Override tree count for a genuine small training smoke run")
    args = parser.parse_args()
    if args.trees is not None and args.trees < 1:
        parser.error("--trees must be positive")
    train(args.data_dir, args.output_dir, args.seed, args.jobs, args.trees)


if __name__ == "__main__":
    main()
