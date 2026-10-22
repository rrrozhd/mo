"""Reproduce retrospective CYP3A4 batch outcomes from frozen predictions, offline."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from rdkit import rdBase

from ..shortlist import select_shortlist

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("classifier", "smiles")
BUDGETS = (24, 48, 96)
POLICIES = (("score_only", 1.0), ("diversity_cap", 0.8))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_population(cohort_path, predictions_path, manifest_path):
    """Use the same observed-label, successful-prediction population for both arms."""
    manifest = json.loads(Path(manifest_path).read_text())
    if sha256(cohort_path) != manifest["cohort_sha256"]:
        raise ValueError("Frozen cohort hash differs from the manifest")
    cohort = pd.read_csv(cohort_path, dtype={"molecule_id": str}, keep_default_na=False)
    if cohort.molecule_id.eq("").any() or cohort.molecule_id.duplicated().any():
        raise ValueError("Cohort molecule IDs must be nonempty and unique")
    cohort["CYP3A4"] = pd.to_numeric(cohort.CYP3A4.replace("", np.nan))
    if not cohort.CYP3A4.dropna().isin([0, 1]).all():
        raise ValueError("CYP3A4 labels must be binary or missing")
    ids = set(cohort.molecule_id)
    scores = {arm: {} for arm in ARMS}
    seen = set()
    with gzip.open(predictions_path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            arm = record["arm"]
            if arm not in ARMS:
                continue
            identifier = str(record["molecule_id"])
            if identifier not in ids:
                raise ValueError("Prediction molecule is outside the cohort")
            key = (identifier, arm)
            if key in seen:
                raise ValueError("Duplicate prediction for molecule and arm")
            seen.add(key)
            if record["status"] != "ok":
                continue
            score = record["scores"]["CYP3A4"]
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not np.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Successful prediction has an invalid score")
            scores[arm][identifier] = float(score)
    for arm in ARMS:
        cohort[f"score_{arm}"] = cohort.molecule_id.map(scores[arm])
    labelled = cohort.CYP3A4.notna()
    paired = cohort[[f"score_{arm}" for arm in ARMS]].notna().all(axis=1)
    population = cohort.loc[labelled & paired].copy()
    if population.empty:
        raise ValueError("No labelled molecules with paired scores")
    coverage = {
        "cohort_molecules": len(cohort),
        "missing_endpoint_labels": int((~labelled).sum()),
        "labelled_without_paired_scores": int((labelled & ~paired).sum()),
        "evaluated_molecules": len(population),
        "positive_labels": int(population.CYP3A4.sum()),
        "successful_scores": {arm: len(scores[arm]) for arm in ARMS},
    }
    return population, coverage


def evaluate_population(population, budgets=BUDGETS):
    """Select with structures and scores only, then attach observed assay outcomes."""
    observations = population.set_index("molecule_id")
    summary, selected_rows = [], []
    for arm in ARMS:
        candidates = [{"id": str(row.molecule_id), "canonical_smiles": row.smiles,
                       "score": float(getattr(row, f"score_{arm}"))}
                      for row in population.itertuples()]
        for policy, cap in POLICIES:
            for budget in budgets:
                result = select_shortlist(candidates, budget, max_similarity=cap)
                if not result["complete"]:
                    raise ValueError(f"Incomplete batch: {arm}/{policy}/{budget}: "
                                     f"{result['shortfall_reason']}")
                selected = result["selected"]
                identifiers = [row["id"] for row in selected]
                measured = observations.loc[identifiers]
                cutoff = min(row["score"] for row in selected)
                summary.append({
                    "arm": arm, "policy": policy, "budget": budget,
                    "max_similarity": cap, "selected_count": len(selected),
                    "hits": int(measured.CYP3A4.sum()),
                    "unique_scaffolds": int(measured.scaffold.nunique()),
                    "cutoff_score": cutoff,
                    "tied_at_cutoff": int(population[f"score_{arm}"].eq(cutoff).sum()),
                    "selected_from_cutoff": sum(row["score"] == cutoff for row in selected),
                })
                for rank, row in enumerate(selected, 1):
                    observation = observations.loc[row["id"]]
                    selected_rows.append({
                        "arm": arm, "policy": policy, "budget": budget,
                        "rank": rank, "molecule_id": row["id"],
                        "smiles": row["canonical_smiles"], "score": row["score"],
                        "observed_hit": int(observation.CYP3A4),
                        "scaffold": observation.scaffold,
                    })
    return pd.DataFrame(summary), pd.DataFrame(selected_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / "results/full-llm/cohort.csv")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "results/full-llm/predictions.jsonl.gz")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "results/full-llm/cohort_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/screening-budget")
    args = parser.parse_args()
    population, coverage = load_population(args.cohort, args.predictions, args.manifest)
    summary, selected = evaluate_population(population)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    selected.to_csv(args.output / "selected-compounds.csv", index=False)
    from .screening_budget_figure import render
    render(summary, args.output, population=coverage["evaluated_molecules"])
    source_paths = [args.cohort, args.predictions, args.manifest, Path(__file__),
                    ROOT / "src/research/screening_budget_figure.py",
                    ROOT / "src/shortlist.py", ROOT / "src/chemistry.py",
                    ROOT / "requirements.lock.txt"]
    provenance = {
        "endpoint": "CYP3A4", "budgets": list(BUDGETS),
        "policies": dict(POLICIES), "tie_break": "original molecule_id ascending",
        "coverage": coverage, "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion, "pandas": pd.__version__,
        "sources": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                    sha256(path) for path in source_paths},
        "outputs": {name: sha256(args.output / name)
                    for name in ("summary.csv", "selected-compounds.csv")},
        "interpretation": "Post-hoc retrospective outcomes; no prospective validation. "
                          "Earlier classifier-family selection used test performance. "
                          "Tie-order sensitivity and statistical uncertainty are not estimated.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
