import gzip
import hashlib
import json

import pandas as pd
import pytest

from src.research.screening_budget import evaluate_population, load_population


@pytest.fixture
def inputs(tmp_path):
    cohort = pd.DataFrame({
        "molecule_id": ["b", "a", "c", "unlabelled"],
        "smiles": ["CCCCCC", "CCCCCCC", "c1ccccc1", "CCO"],
        "scaffold": ["", "", "c1ccccc1", ""],
        "CYP3A4": [0, 1, 1, None],
    })
    cohort_path = tmp_path / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    records = [{"molecule_id": identifier, "arm": arm, "status": "ok",
                "scores": {"CYP3A4": score}}
               for arm in ("classifier", "smiles")
               for identifier, score in zip(cohort.molecule_id, [.9, .9, .7, 1.0])]
    predictions = tmp_path / "predictions.jsonl.gz"
    with gzip.open(predictions, "wt") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cohort_sha256": hashlib.sha256(
        cohort_path.read_bytes()).hexdigest()}))
    return cohort_path, predictions, manifest, records


def test_unlabelled_candidates_are_excluded_not_counted_as_negative(inputs):
    population, coverage = load_population(*inputs[:3])
    assert set(population.molecule_id) == {"a", "b", "c"}
    assert coverage["cohort_molecules"] == 4
    assert coverage["missing_endpoint_labels"] == 1
    assert coverage["evaluated_molecules"] == 3
    assert coverage["positive_labels"] == 2


def test_outcomes_cannot_change_selection_and_ties_ignore_input_order(inputs):
    population, _ = load_population(*inputs[:3])
    before, selected = evaluate_population(population, budgets=(2,))
    changed = population.iloc[::-1].copy()
    changed["CYP3A4"] = 1 - changed.CYP3A4
    after, selected_changed = evaluate_population(changed, budgets=(2,))
    keys = ["arm", "policy", "budget", "molecule_id"]
    pd.testing.assert_frame_equal(selected[keys], selected_changed[keys])
    assert (before.hits + after.hits).eq(2).all()
    first = before[(before.arm == "smiles") & (before.policy == "score_only")].iloc[0]
    assert first.tied_at_cutoff == 2
    assert first.selected_from_cutoff == 2


def test_cohort_hash_is_verified(inputs):
    inputs[0].write_text(inputs[0].read_text() + "\n")
    with pytest.raises(ValueError, match="cohort hash"):
        load_population(*inputs[:3])


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "Duplicate prediction"),
    ("unknown", "outside the cohort"),
    ("invalid_score", "score"),
])
def test_predictions_cannot_silently_change_population(inputs, mutation, match):
    records = inputs[3]
    if mutation == "duplicate":
        records.append(records[0])
    elif mutation == "unknown":
        records[0]["molecule_id"] = "unknown"
    else:
        records[0]["scores"]["CYP3A4"] = True
    with gzip.open(inputs[1], "wt") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)
    with pytest.raises(ValueError, match=match):
        load_population(*inputs[:3])


def test_failed_predictions_are_excluded_from_both_arms_and_counted(inputs):
    records = inputs[3]
    records[0] = {"molecule_id": "b", "arm": "classifier", "status": "error"}
    with gzip.open(inputs[1], "wt") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)
    population, coverage = load_population(*inputs[:3])
    assert set(population.molecule_id) == {"a", "c"}
    assert coverage["labelled_without_paired_scores"] == 1


def test_incomplete_batches_cannot_be_presented_as_full_budget(inputs):
    population, _ = load_population(*inputs[:3])
    with pytest.raises(ValueError, match="Incomplete batch"):
        evaluate_population(population, budgets=(4,))
