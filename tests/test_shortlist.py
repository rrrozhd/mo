from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import DataStructs, rdBase

from src.chemistry import canonical_smiles, featurize, fingerprints


def shortlist_module():
    from src import shortlist
    return shortlist


def rows():
    return [{"id": name, "canonical_smiles": smiles, "score": score}
            for name, smiles, score in [("a", "CCCCCC", .9), ("b", "CCCCCCC", .8),
                                         ("c", "c1ccccc1", .7), ("d", "CCO", .6)]]


def ids(result):
    return [row["id"] for row in result["selected"]]


def test_diversity_changes_selection_and_reports_skipped_candidates():
    select = shortlist_module().select_shortlist
    assert ids(select(rows(), 2, max_similarity=1)) == ["a", "b"]
    diverse = select(rows(), 2, max_similarity=.5)
    assert ids(diverse) == ["a", "c"]
    assert diverse["complete"] and diverse["shortfall_reason"] is None
    assert diverse["requested_count"] == diverse["selected_count"] == 2
    skipped = {row["id"]: row["reason"] for row in diverse["skipped"]}
    assert "similarity" in skipped["b"]
    assert "budget" in skipped["d"]


def test_mandatory_inclusion_reserves_space_and_exclusion_changes_results():
    select = shortlist_module().select_shortlist
    assert ids(select(rows(), 2, include_ids=["b"], max_similarity=.5)) == ["b", "c"]
    result = select(rows(), 2, include_ids=["d"], exclude_ids=["a"], max_similarity=1)
    assert ids(result) == ["b", "d"]
    assert {row["id"]: row["reason"] for row in result["skipped"]}["a"] == "excluded"


def test_score_ties_are_id_ordered_independent_of_input_order():
    select = shortlist_module().select_shortlist
    candidates = [{**row, "score": .5} for row in rows()]
    result = select(list(reversed(candidates)), 2, max_similarity=1)
    assert ids(result) == ["a", "b"]
    assert [row["id"] for row in result["ranked"]] == ["a", "b", "c", "d"]


def test_similarity_threshold_is_inclusive():
    select = shortlist_module().select_shortlist
    candidates = rows()[:2]
    fps = fingerprints([row["canonical_smiles"] for row in candidates])
    threshold = DataStructs.TanimotoSimilarity(*fps)
    assert ids(select(candidates, 2, max_similarity=threshold)) == ["a", "b"]
    result = select(candidates, 2, max_similarity=threshold - .001)
    assert ids(result) == ["a"] and not result["complete"]
    assert "similarity" in result["shortfall_reason"]


def test_shortfalls_do_not_relax_constraints_or_invent_candidates():
    select = shortlist_module().select_shortlist
    result = select(rows(), 4, exclude_ids=["d"], max_similarity=1)
    assert ids(result) == ["a", "b", "c"]
    assert not result["complete"] and result["selected_count"] == 3
    assert "eligible" in result["shortfall_reason"]
    empty = select([], 1)
    assert empty["selected"] == [] and not empty["complete"]


@pytest.mark.parametrize("kwargs,match", [
    ({"budget": 0}, "budget"), ({"budget": -1}, "budget"),
    ({"budget": True}, "budget"), ({"budget": 1.5}, "budget"),
    ({"max_similarity": -1}, "max_similarity"),
    ({"max_similarity": 1.01}, "max_similarity"),
    ({"max_similarity": float("nan")}, "max_similarity"),
    ({"max_similarity": True}, "max_similarity"),
    ({"include_ids": ["missing"]}, "Unknown"),
    ({"exclude_ids": ["missing"]}, "Unknown"),
    ({"include_ids": ["a"], "exclude_ids": ["a"]}, "both"),
    ({"include_ids": ["a", "b"], "max_similarity": .5}, "Mandatory"),
    ({"include_ids": ["a", "b", "c"]}, "budget"),
    ({"include_ids": ["a", "a"]}, "Duplicate"),
    ({"exclude_ids": "a"}, "list"),
])
def test_invalid_selection_constraints_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        shortlist_module().select_shortlist(rows(), **{"budget": 2, **kwargs})


@pytest.mark.parametrize("candidate,match", [
    ({"id": "a", "canonical_smiles": "CCN", "score": .3}, "Duplicate ID"),
    ({"id": "e", "canonical_smiles": "CCCCCC", "score": .3}, "Duplicate structure"),
    ({"id": "e", "canonical_smiles": "invalid", "score": .3}, "Invalid SMILES"),
    ({"id": "e", "canonical_smiles": "CCN", "score": float("nan")}, "score"),
    ({"id": "e", "canonical_smiles": "CCN", "score": 1.1}, "score"),
    ({"id": 3, "canonical_smiles": "CCN", "score": .3}, "ID"),
])
def test_selector_rejects_invalid_candidate_rows(candidate, match):
    with pytest.raises(ValueError, match=match):
        shortlist_module().select_shortlist(rows() + [candidate], 2)


class Predictor:
    def __init__(self):
        self.batches = []

    def predict_proba(self, features):
        self.batches.append(features)
        return np.tile([.2, .8], (len(features), 1))


@pytest.fixture
def screen(monkeypatch):
    module = shortlist_module()
    model = Predictor()
    bundle = {"enzyme": "CYP3A4", "rdkit_version": rdBase.rdkitVersion,
              "descriptor_names": ["MolWt"], "model": model,
              "train_smiles": ["CCO", "c1ccccc1"], "domain_threshold": .5,
              "validation_scores": [.1, .5, .8, .9]}
    loads = []

    def load(path):
        loads.append(Path(path).name)
        assert Path(path).name == "CYP3A4.joblib"
        return bundle

    monkeypatch.setattr(module.joblib, "load", load)
    instance = module.CYP3A4Screen("unused")
    return instance, model, bundle, loads


def test_scoring_uses_one_batch_and_only_target_model_without_test_labels(screen):
    instance, model, _, loads = screen
    candidates = [{"id": "ethanol", "smiles": "OCC", "CYP3A4": 0},
                  {"id": "amine", "smiles": "CCN", "CYP3A4": 1}]
    results = instance.score(candidates)
    assert loads == ["CYP3A4.joblib"]
    assert len(model.batches) == 1 and model.batches[0].shape == (2, 2049)
    assert results[0]["canonical_smiles"] == "CCO"
    assert results[0]["nearest_training_similarity"] == 1
    assert results[0]["similarity_flag"] is False
    assert results[1]["similarity_flag"] is True
    assert all(row["score"] == .8 and row["validation_percentile"] == .75 for row in results)
    assert all(row["confidence"] == "uncalibrated" and "CYP3A4" not in row for row in results)
    reversed_labels = [{**candidate, "CYP3A4": 1 - candidate["CYP3A4"]} for candidate in candidates]
    assert instance.score(reversed_labels) == results


@pytest.mark.parametrize("candidates,match", [
    ([{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "broken"}], "Invalid SMILES"),
    ([{"id": "a", "smiles": "CCO"}, {"id": "a", "smiles": "CCN"}], "Duplicate ID"),
    ([{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "OCC"}], "Duplicate structure"),
    ([{"id": "", "smiles": "CCO"}], "ID"),
])
def test_all_candidate_validation_happens_before_prediction(screen, candidates, match):
    instance, model, _, _ = screen
    with pytest.raises(ValueError, match=match):
        instance.score(candidates)
    assert model.batches == []


def test_empty_scoring_does_not_call_model(screen):
    instance, model, _, _ = screen
    assert instance.score([]) == [] and model.batches == []


def test_actual_bundle_small_heldout_batch_matches_direct_classifier():
    module = shortlist_module()
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    if not (artifacts / "models" / "CYP3A4.joblib").exists():
        pytest.skip("Local trained CYP3A4 artifact is unavailable")
    heldout = pd.read_csv(artifacts / "test.csv", nrows=3)[["molecule_id", "smiles"]]
    candidates = [{"id": identifier, "smiles": smiles}
                  for identifier, smiles in heldout.itertuples(index=False, name=None)]
    screen = module.CYP3A4Screen(artifacts)
    result = screen.score(candidates)
    bundle = module.joblib.load(artifacts / "models" / "CYP3A4.joblib")
    expected = bundle["model"].predict_proba(featurize(heldout.smiles, bundle["descriptor_names"]))[:, 1]
    np.testing.assert_allclose([row["score"] for row in result], expected)
    assert all(0 <= row["validation_percentile"] <= 1 for row in result)
    assert all(0 <= row["nearest_training_similarity"] <= 1 for row in result)
    assert ids(module.select_shortlist(result, 2, max_similarity=1)) == [
        row["id"] for row in sorted(result, key=lambda row: (-row["score"], row["id"]))[:2]]
