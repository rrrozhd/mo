from pathlib import Path

import pytest

from src import depictions

ETHANOL = "CCO"
ANISOLE = "COc1ccccc1"


def requires_artifacts():
    if not (Path("artifacts") / "manifest.csv").exists():
        pytest.skip("Local trained CYP artifacts are unavailable")


def test_identical_molecules_have_nothing_highlighted():
    result = depictions.mcs_pair(ANISOLE, ANISOLE)
    assert result["panels"]["a"]["differing_atoms"] == []
    assert result["panels"]["b"]["differing_atoms"] == []
    assert result["panels"]["a"]["svg"].startswith("<?xml")


def test_a_substituent_shows_up_as_the_difference():
    result = depictions.mcs_pair(ANISOLE, "COc1ccccc1Cl")
    assert result["shared_atom_count"] >= 7
    assert len(result["panels"]["b"]["differing_atoms"]) == 1  # Only the chlorine differs.
    assert result["panels"]["a"]["differing_atoms"] == []


def test_depictions_refuse_unparseable_and_oversized_input():
    with pytest.raises(ValueError, match="Invalid SMILES"):
        depictions.mcs_pair("not a molecule", ETHANOL)
    with pytest.raises(ValueError, match="at most"):
        depictions.mcs_pair("C" * (depictions.MAX_SMILES + 1), ETHANOL)
    with pytest.raises(ValueError, match="enzyme must be"):
        depictions.neighbourhood(ETHANOL, "CYP9Z9")


def test_attribution_reports_unscorable_atoms_instead_of_calling_them_zero():
    """Deleting an aromatic atom leaves nothing that sanitizes; that is 'no answer', not 'no effect'."""
    requires_artifacts()
    result = depictions.attribution(ANISOLE, "CYP3A4")
    assert result["atoms_total"] == len(result["atoms"]) == 8
    assert result["atoms_scored"] + result["atoms_unscorable"] == result["atoms_total"]
    assert result["atoms_unscorable"] >= 6  # The six aromatic ring atoms cannot be deleted.
    unscorable = [atom for atom in result["atoms"] if atom["status"] == "deletion_invalid"]
    assert all(atom["weight"] is None for atom in unscorable)
    assert all(atom["weight"] is not None
               for atom in result["atoms"] if atom["status"] == "scored")
    assert "log-odds" in result["weight_units"]
    assert result["svg"].startswith("<?xml")


def test_attribution_is_capped_rather_than_running_forever(monkeypatch):
    requires_artifacts()
    monkeypatch.setattr(depictions, "MAX_ATTRIBUTION_ATOMS", 3)
    with pytest.raises(ValueError, match="limited to 3 atoms"):
        depictions.attribution(ANISOLE, "CYP3A4")


@pytest.mark.parametrize("smiles", ["C", "[Na+]", "[He]"])
def test_single_atom_attribution_marks_empty_deletion_unscorable(monkeypatch, smiles):
    def score(structure, _enzyme, **kwargs):
        assert structure, "An empty molecule cannot be scored"
        return 0.5

    monkeypatch.setattr(depictions, "_score", score)
    result = depictions.attribution(smiles, "CYP3A4")
    assert result["atoms_scored"] == 0
    assert result["atoms_unscorable"] == 1
    assert result["atoms"] == [{"index": 0, "status": "deletion_invalid", "weight": None}]
    assert "<svg" in result["svg"]


def test_neighbourhood_puts_the_query_first_with_measured_neighbours():
    requires_artifacts()
    result = depictions.neighbourhood(ANISOLE, "CYP3A4", neighbours=10)
    query, neighbours = result["points"][0], result["points"][1:]
    assert query["is_query"] and query["measured_label"] is None
    assert len(neighbours) == 10
    assert all(point["measured_label"] in (0, 1) for point in neighbours)
    # Neighbours are ordered by similarity, so the first is the closest.
    assert neighbours == sorted(neighbours, key=lambda point: -point["similarity"])


def test_attribution_passes_artifacts_for_original_and_deleted_atoms(tmp_path, monkeypatch):
    calls = []

    def score(smiles, enzyme, *, artifacts):
        calls.append((smiles, enzyme, artifacts))
        return .5

    monkeypatch.setattr(depictions, "_score", score)
    result = depictions.attribution("CCO", "CYP3A4", artifacts=tmp_path)
    assert result["atoms_scored"] == 3
    assert len(calls) == 4
    assert all(enzyme == "CYP3A4" and artifacts == tmp_path for _, enzyme, artifacts in calls)


def test_neighbourhood_uses_selected_training_corpus(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from src.chemistry import fingerprints

    paths = []
    frame = pd.DataFrame({"smiles": ["CCN", "CCCC"], "CYP3A4": [0, 1]})

    def corpus(artifacts):
        paths.append(artifacts)
        return frame, fingerprints(frame.smiles.tolist())

    class Projection:
        def fit_transform(self, distances):
            return np.zeros((len(distances), 2))

    monkeypatch.setattr(depictions, "_corpus", corpus)
    monkeypatch.setattr(depictions, "MDS", lambda **kwargs: Projection())
    result = depictions.neighbourhood("CCO", "CYP3A4", artifacts=tmp_path)
    assert paths == [tmp_path]
    assert [row["smiles"] for row in result["points"]] == ["CCO", "CCCC", "CCN"]
