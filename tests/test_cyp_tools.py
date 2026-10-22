import numpy as np
import pandas as pd
import pytest

from src import cyp_tools
from src.chemistry import ENZYMES, fingerprints


def corpus_fixture(monkeypatch):
    """A tiny labelled corpus; labels are synthetic and never a scientific result."""
    rows = [
        {"molecule_id": "a", "smiles": "CCOc1ccccc1", "scaffold": "", "split": "train",
         "CYP1A2": 1.0, "CYP2C9": 1.0, "CYP2C19": 1.0, "CYP2D6": 1.0, "CYP3A4": 0.0},
        {"molecule_id": "b", "smiles": "CCOc1ccccc1C", "scaffold": "", "split": "train",
         "CYP1A2": 1.0, "CYP2C9": 1.0, "CYP2C19": 1.0, "CYP2D6": 1.0, "CYP3A4": 1.0},
        {"molecule_id": "c", "smiles": "CCOc1ccccc1CC", "scaffold": "", "split": "train",
         "CYP1A2": 1.0, "CYP2C9": 1.0, "CYP2C19": 1.0, "CYP2D6": 1.0, "CYP3A4": np.nan},
        {"molecule_id": "d", "smiles": "CCCCCCCCCCCCN", "scaffold": "", "split": "train",
         "CYP1A2": 0.0, "CYP2C9": 0.0, "CYP2C19": 0.0, "CYP2D6": 0.0, "CYP3A4": 0.0},
        # A held-out molecule near-identical to the query; it must never be returned.
        {"molecule_id": "e", "smiles": "CCOc1ccccc1CCC", "scaffold": "", "split": "test",
         "CYP1A2": 0.0, "CYP2C9": 0.0, "CYP2C19": 0.0, "CYP2D6": 0.0, "CYP3A4": 0.0},
    ]
    frame = pd.DataFrame(rows)
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    monkeypatch.setattr(cyp_tools, "_corpus",
                        lambda artifacts=cyp_tools.ARTIFACTS: (train, fingerprints(train["smiles"].tolist())))
    return frame


def test_measured_label_filters_to_negative_dataset_labels(monkeypatch):
    corpus_fixture(monkeypatch)
    result = cyp_tools.find_analogs("CCOc1ccccc1C", "CYP3A4", measured_label=0,
                                    max_results=5, min_similarity=0.1)
    assert result["analogs"][0]["smiles"] == "CCOc1ccccc1"  # Closest negative label first.
    assert all(hit["measured_label"] == 0 for hit in result["analogs"])
    assert "CCOc1ccccc1C" not in [hit["smiles"] for hit in result["analogs"]]  # Positive label.


def test_unmeasured_labels_and_the_query_itself_are_skipped(monkeypatch):
    corpus_fixture(monkeypatch)
    result = cyp_tools.find_analogs("CCOc1ccccc1CC", "CYP3A4", max_results=5, min_similarity=0.1)
    returned = [hit["smiles"] for hit in result["analogs"]]
    assert "CCOc1ccccc1CC" not in returned  # Absent label and self-match both excluded.
    assert set(returned) == {"CCOc1ccccc1", "CCOc1ccccc1C"}


def test_min_similarity_stops_the_search(monkeypatch):
    corpus_fixture(monkeypatch)
    loose = cyp_tools.find_analogs("CCOc1ccccc1C", "CYP3A4", max_results=5, min_similarity=0.0)
    strict = cyp_tools.find_analogs("CCOc1ccccc1C", "CYP3A4", max_results=5, min_similarity=0.9)
    assert len(strict["analogs"]) < len(loose["analogs"])


def test_invalid_arguments_are_refused(monkeypatch):
    corpus_fixture(monkeypatch)
    with pytest.raises(ValueError, match="enzyme must be"):
        cyp_tools.find_analogs("CCO", "CYP9Z9")
    with pytest.raises(ValueError, match="measured_label"):
        cyp_tools.find_analogs("CCO", "CYP3A4", measured_label=7)
    with pytest.raises(ValueError, match="max_results"):
        cyp_tools.find_analogs("CCO", "CYP3A4", max_results=0)


def test_compare_structures_reports_identity_and_similarity():
    same = cyp_tools.compare_structures("CCOc1ccccc1", "CCOc1ccccc1")
    assert same["exact_identity_equal"] and same["fingerprint_tanimoto"] == 1.0
    other = cyp_tools.compare_structures("CCOc1ccccc1", "CCCCCCCCCCCCN")
    assert not other["exact_identity_equal"] and other["fingerprint_tanimoto"] < 0.5


def test_held_out_molecules_are_never_returned_as_measured_evidence(monkeypatch):
    """A test-split neighbour would be the evaluation set's answer key, not evidence."""
    corpus_fixture(monkeypatch)
    result = cyp_tools.find_analogs("CCOc1ccccc1C", "CYP3A4", measured_label=0,
                                    max_results=10, min_similarity=0.0)
    returned = [hit["smiles"] for hit in result["analogs"]]
    assert "CCOc1ccccc1CCC" not in returned
    assert all(hit["split"] == "train" for hit in result["analogs"])


def write_analysis_artifacts(path, score):
    (path / "models").mkdir(parents=True)
    for enzyme in ENZYMES:
        (path / "models" / f"{enzyme}.joblib").write_text(str(score))
    pd.DataFrame([{"molecule_id": "fixture", "smiles": "CCN", "split": "train",
                   **{enzyme: int(score > .5) for enzyme in ENZYMES}}]).to_csv(
        path / "manifest.csv", index=False)


def test_scores_and_corpus_use_selected_artifact_directory_and_refresh_changed_files(tmp_path, monkeypatch):
    from pathlib import Path
    from src import depictions

    first, second = tmp_path / "first", tmp_path / "second"
    write_analysis_artifacts(first, .2)
    write_analysis_artifacts(second, .8)
    loaded = []

    class Suite:
        descriptor_names = []

        def __init__(self, artifacts):
            loaded.append(Path(artifacts))
            self.score = float((Path(artifacts) / "models/CYP3A4.joblib").read_text())

        def predict(self, vector):
            return {"scores": {enzyme: self.score for enzyme in ENZYMES},
                    "validation_percentile": {enzyme: .5 for enzyme in ENZYMES}}

        def similarity(self, query):
            return {enzyme: {"outside_training_distribution": False} for enzyme in ENZYMES}

    monkeypatch.setattr(cyp_tools, "ModelSuite", Suite)
    for path, expected in ((first, .2), (second, .8), (first, .2)):
        assert cyp_tools.predict_cyp("CCO", artifacts=path)["enzymes"]["CYP3A4"]["score"] == expected
        assert depictions._score("CCO", "CYP3A4", artifacts=path) == expected
        result = cyp_tools.find_analogs("CCO", "CYP3A4", min_similarity=0, artifacts=path)
        assert result["analogs"][0]["measured_label"] == int(expected > .5)
    assert loaded == [first.resolve(), second.resolve()]
    (first / "models/CYP3A4.joblib").write_text("0.4")
    assert cyp_tools.predict_cyp("CCO", artifacts=first)["enzymes"]["CYP3A4"]["score"] == .4
    assert depictions._score("CCO", "CYP3A4", artifacts=first) == .4
    frame = pd.read_csv(first / "manifest.csv")
    frame.loc[0, "CYP3A4"] = 1
    frame.to_csv(first / "manifest.csv", index=False)
    assert cyp_tools.find_analogs("CCO", "CYP3A4", min_similarity=0, artifacts=first)["analogs"][0]["measured_label"] == 1


@pytest.mark.parametrize("name, target", [("show_attribution", "attribution"),
                                          ("show_neighbourhood", "neighbourhood")])
def test_depiction_wrappers_forward_selected_artifacts(tmp_path, monkeypatch, name, target):
    from src import depictions

    received = []
    monkeypatch.setattr(depictions, target, lambda *args, **kwargs: received.append((args, kwargs)) or {})
    assert getattr(cyp_tools, name)("CCO", "CYP3A4", artifacts=tmp_path) == {}
    assert received == [(("CCO", "CYP3A4"), {"artifacts": tmp_path})]
