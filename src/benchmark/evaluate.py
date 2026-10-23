"""Score frozen classifier predictions with scaffold-bootstrap confidence intervals."""

import argparse
import gzip
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


from ..chemistry.features import ENZYMES
ARMS = ("classifier",)
LLM_ARMS = ("classifier", "smiles", "name")
METRICS = {"auc": roc_auc_score, "ap": average_precision_score}


def _inputs(cohort, records, arms=ARMS):
    required = {"molecule_id", "scaffold", *ENZYMES}
    if not required.issubset(cohort.columns):
        raise ValueError(f"Cohort missing columns: {sorted(required - set(cohort.columns))}")
    cohort = cohort.copy()
    if cohort.molecule_id.isna().any():
        raise ValueError("Missing molecule_id in cohort")
    cohort["molecule_id"] = cohort.molecule_id.astype(str)
    if cohort.molecule_id.duplicated().any() or (cohort.molecule_id.str.strip() == "").any():
        raise ValueError("Cohort molecule_id values must be unique and nonempty")
    cohort["scaffold"] = cohort.scaffold.fillna("").astype(str)
    for enzyme in ENZYMES:
        cohort[enzyme] = pd.to_numeric(cohort[enzyme], errors="raise")
        if not cohort[enzyme].dropna().isin([0, 1]).all():
            raise ValueError(f"Invalid binary labels for {enzyme}")
    ids = set(cohort.molecule_id)
    indexed = {}
    for record in records:
        molecule_id = str(record["molecule_id"])
        arm, status = record["arm"], record["status"]
        if molecule_id not in ids or arm not in arms:
            raise ValueError(f"Unexpected prediction identity or arm: {molecule_id}, {arm}")
        key = (molecule_id, arm)
        if key in indexed:
            raise ValueError(f"Duplicate prediction record: {key}")
        if status not in {"ok", "error", "unavailable"}:
            raise ValueError(f"Invalid prediction status: {status}")
        if status == "ok":
            scores = record.get("scores")
            if not isinstance(scores, dict) or any(e not in scores for e in ENZYMES):
                raise ValueError(f"Successful prediction lacks five scores: {key}")
            for score in scores.values():
                if (isinstance(score, bool) or not isinstance(score, (int, float))
                        or not np.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError(f"Invalid successful prediction score: {key}")
        indexed[key] = record
    return cohort, indexed


def _coverage(ids, records, arms, population):
    rows = []
    for arm in arms:
        statuses = [records.get((m, arm), {}).get("status", "missing") for m in ids]
        rows.append({
            "population": population, "arm": arm, "planned": len(ids),
            "attempted": statuses.count("ok") + statuses.count("error"),
            "succeeded": statuses.count("ok"), "error": statuses.count("error"),
            "unavailable": statuses.count("unavailable"), "missing": statuses.count("missing"),
        })
    return rows


def _bootstrap(y, scores, scaffolds, replicates, rng):
    """Reuse each cluster draw across every arm and both metrics."""
    draws = {(arm, metric): [] for arm in scores for metric in METRICS}
    if len(np.unique(y)) < 2:
        return draws, replicates
    groups = [np.flatnonzero(scaffolds == scaffold) for scaffold in np.unique(scaffolds)]
    skipped = 0
    for _ in range(replicates):
        selected = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in selected])
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) < 2:
            skipped += 1
            continue
        for arm, values in scores.items():
            for metric, function in METRICS.items():
                draws[arm, metric].append(float(function(sampled_y, values[indices])))
    return draws, skipped


def _interval(values):
    if len(values) == 0:
        return None, None
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def evaluate(cohort, records, bootstrap=1000, seed=42, arms=ARMS):
    """Return metric rows and explicit cohort coverage for the classifier arm."""
    if not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    cohort, records = _inputs(cohort, records, arms)
    ids = cohort.molecule_id.tolist()
    coverage = _coverage(ids, records, arms, "full_selected")
    rng = np.random.default_rng(seed)
    metric_rows, contrast_rows, support = [], [], []
    paired = cohort[cohort.molecule_id.map(lambda m: all(
        records.get((m, arm), {}).get("status") == "ok" for arm in arms
    )).astype(bool)]
    for enzyme in ENZYMES:
        observed = paired[paired[enzyme].notna()]
        y = observed[enzyme].to_numpy(dtype=int)
        scores = {arm: np.asarray([
            records[m, arm]["scores"][enzyme] for m in observed.molecule_id
        ], dtype=float) for arm in arms}
        common = {
            "population": "+".join(arms), "enzyme": enzyme, "purpose": "primary",
            "n_population": len(cohort), "n_labeled": int(cohort[enzyme].notna().sum()),
            "n_evaluable": len(y), "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "n_scaffolds": int(observed.scaffold.nunique()),
        }
        support.append(common)
        defined = len(np.unique(y)) == 2
        draws, skipped = _bootstrap(y, scores, observed.scaffold.to_numpy(), bootstrap, rng)
        details = {
            **common, "bootstrap_valid": bootstrap - skipped, "bootstrap_skipped": skipped,
            "undefined_reason": None if defined else "requires_both_classes",
        }
        for arm in arms:
            for metric, function in METRICS.items():
                low, high = _interval(draws[arm, metric])
                metric_rows.append({
                    **details, "arm": arm, "metric": metric,
                    "value": float(function(y, scores[arm])) if defined else None,
                    "ci_low": low, "ci_high": high,
                })
        # Arms share each resample, so a difference of draws is the paired contrast.
        # Marginal intervals may overlap while the paired interval excludes zero.
        for first, second in combinations(arms, 2):
            for metric, function in METRICS.items():
                difference = [a - b for a, b in zip(draws[first, metric], draws[second, metric])]
                low, high = _interval(difference)
                contrast_rows.append({
                    **details, "contrast": f"{first}-{second}", "metric": metric,
                    "value": float(function(y, scores[first]) - function(y, scores[second]))
                    if defined else None,
                    "ci_low": low, "ci_high": high,
                    "excludes_zero": low is not None and high is not None and (low > 0 or high < 0),
                })
    summary = {
        "cohort_size": len(cohort),
        "bootstrap": {"replicates": bootstrap, "seed": seed, "unit": "scaffold", "confidence": 0.95},
        "coverage": coverage, "label_support": support,
        "notes": [
            "Metrics condition on successful predictions; failures are not assigned invented scores.",
            "Attempted counts ok plus error records; unavailable and missing are reported separately.",
            "Single-class samples have undefined AUC and AP here; bootstrap draws with one class are skipped.",
            "Intervals reflect scaffold resampling, not model-training variability.",
            "Scores are uncalibrated ranking scores on one exploratory split, not calibrated risk.",
        ],
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=LLM_ARMS)
    args = parser.parse_args()
    cohort = pd.read_csv(args.run_dir / "cohort.csv", dtype={"molecule_id": str, "scaffold": str})
    prediction_path = args.run_dir / "predictions.jsonl"
    if prediction_path.exists():
        source = prediction_path.open(encoding="utf-8")
    else:
        source = gzip.open(args.run_dir / "predictions.jsonl.gz", "rt", encoding="utf-8")
    with source:
        records = [json.loads(line) for line in source if line.strip()]
    metrics, contrasts, summary = evaluate(cohort, records, args.bootstrap, args.seed,
                                           tuple(args.arms))
    metrics.to_csv(args.run_dir / "metrics.csv", index=False)
    if not contrasts.empty:
        contrasts.to_csv(args.run_dir / "contrasts.csv", index=False)
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Evaluated {len(cohort)} molecules.")
    print(metrics[["arm", "enzyme", "metric", "n_evaluable", "n_positive", "value", "ci_low", "ci_high"]].to_string(index=False))
    if not contrasts.empty:
        print("\nPaired differences (same resamples; zero outside the interval is a real gap):")
        print(contrasts[["contrast", "enzyme", "metric", "value", "ci_low", "ci_high",
                         "excludes_zero"]].to_string(index=False))


if __name__ == "__main__":
    main()
