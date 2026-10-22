import hashlib

import numpy as np
import pandas as pd
import pytest
from rdkit import DataStructs

from src.chemistry import (
    DESCRIPTOR_NAMES, ENZYMES, canonical_smiles, featurize, fingerprints, nearest_distances,
)
from src.train import classifier, merge_datasets, scaffold_manifest, train


def empty_datasets():
    return {enzyme: pd.DataFrame(columns=["Drug", "Y"]) for enzyme in ENZYMES}


def test_merge_preserves_unknowns_and_masks_conflicts_without_majority_vote():
    datasets = empty_datasets()
    datasets["CYP1A2"] = pd.DataFrame({
        "Drug": ["CCO", "OCC", "CCO", "CC", "broken", ""],
        "Y": [1, 0, 1, 0, 1, 0],
    })
    datasets["CYP2C9"] = pd.DataFrame({"Drug": ["OCC", "CCN"], "Y": [1, 0]})
    union, report = merge_datasets(datasets)
    assert list(union["smiles"]) == sorted(["CC", "CCN", "CCO"])
    ethanol = union.set_index("smiles").loc["CCO"]
    assert np.isnan(ethanol["CYP1A2"])
    assert ethanol["CYP2C9"] == 1
    assert np.isnan(ethanol["CYP2D6"])
    assert len(report["invalid_smiles"]) == 2
    assert report["conflicts"] == [{"smiles": "CCO", "enzyme": "CYP1A2"}]
    assert report["duplicate_observations"] == 2


def test_missing_and_invalid_labels_are_not_converted_to_negatives():
    datasets = empty_datasets()
    datasets["CYP1A2"] = pd.DataFrame({"Drug": ["CCO", "CCN", "CCC"], "Y": [np.nan, 2, 0]})
    union, report = merge_datasets(datasets)
    labels = union.set_index("smiles")["CYP1A2"]
    assert np.isnan(labels["CCO"])
    assert np.isnan(labels["CCN"])
    assert labels["CCC"] == 0
    assert len(report["invalid_labels"]) == 1


def test_global_scaffold_split_isolated_and_independent_of_input_order():
    smiles = ["CCO", "CCN", "CCC"]
    for ring_size in range(3, 16):
        smiles.extend(["C1" + "C" * (ring_size - 1) + "1", "N1" + "C" * (ring_size - 1) + "1"])
    union = pd.DataFrame({"smiles": sorted(map(canonical_smiles, smiles))})
    for enzyme in ENZYMES:
        union[enzyme] = np.nan
    manifest = scaffold_manifest(union)
    shuffled = scaffold_manifest(union.sample(frac=1, random_state=8))
    pd.testing.assert_frame_equal(manifest, shuffled)
    assert len(manifest) == len(union)
    assert set(manifest["split"]) == {"train", "valid", "test"}
    assert manifest.groupby("smiles")["split"].nunique().eq(1).all()
    assert manifest.groupby("scaffold")["split"].nunique().eq(1).all()
    assert manifest.loc[manifest["scaffold"] == "", "split"].nunique() == 1
    assert manifest.iloc[0]["molecule_id"] == hashlib.sha256(manifest.iloc[0]["smiles"].encode()).hexdigest()


def test_features_share_canonical_identity_and_preserve_descriptor_order():
    features = featurize(["CCO", "OCC"])
    assert features.shape == (2, 2048 + len(DESCRIPTOR_NAMES))
    np.testing.assert_allclose(features[0], features[1], equal_nan=True)
    reordered = featurize(["CCO"], descriptor_names=["MolLogP", "MolWt"])
    assert reordered.shape == (1, 2050)
    np.testing.assert_allclose(reordered[0, -2:], features[0, [2048 + DESCRIPTOR_NAMES.index("MolLogP"),
                                                            2048 + DESCRIPTOR_NAMES.index("MolWt")]])
    assert featurize([]).shape == (0, 2048 + len(DESCRIPTOR_NAMES))
    with pytest.raises(ValueError, match="Invalid SMILES"):
        featurize(["CCO", "invalid"])
    with pytest.raises(ValueError, match="Unknown RDKit"):
        featurize(["CCO"], descriptor_names=["missing_descriptor"])


def test_nearest_distance_excludes_self_but_keeps_duplicate_neighbors():
    fps = fingerprints(["CCO", "c1ccccc1", "CCN"])
    np.testing.assert_allclose(nearest_distances(fps, fps), 0)
    expected = [1 - max(DataStructs.TanimotoSimilarity(a, b) for j, b in enumerate(fps) if j != i)
                for i, a in enumerate(fps)]
    np.testing.assert_allclose(nearest_distances(fps, fps, exclude_self=True), expected)
    assert all(distance > 0 for distance in expected)
    duplicate_fps = fingerprints(["CCO", "OCC", "CCC"])
    assert nearest_distances(duplicate_fps, duplicate_fps, exclude_self=True)[0] == 0
    with pytest.raises(ValueError, match="at least two"):
        nearest_distances(fps[:1], fps[:1], exclude_self=True)
    with pytest.raises(ValueError, match="At least one"):
        nearest_distances(fps, [])


def test_frozen_models_match_feasibility_parameters():
    for enzyme in ("CYP2C19", "CYP2D6"):
        params = classifier(enzyme, seed=42, jobs=4).get_params()
        assert params["n_estimators"] == 300
        assert params["min_samples_leaf"] == 2
        assert params["class_weight"] is None
        assert params["random_state"] == 42
    for enzyme in ("CYP1A2", "CYP3A4"):
        params = classifier(enzyme, seed=42, jobs=4).get_params()
        assert params["n_estimators"] == 600
        assert params["learning_rate"] == 0.05
        assert params["num_leaves"] == 63
    params = classifier("CYP2C9", seed=42, jobs=4).get_params()
    assert params["n_estimators"] == 600
    assert params["learning_rate"] == 0.05
    assert params["max_depth"] == 7
    assert params["tree_method"] == "hist"
    for enzyme in ENZYMES:
        assert classifier(enzyme, seed=42, jobs=4, trees=3).get_params()["n_estimators"] == 3


def test_retraining_refuses_nonempty_output_without_changing_artifacts(tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    previous = output / "manifest.json"
    previous.write_text('{"previous": true}\n')
    with pytest.raises(FileExistsError, match="Use a new output directory"):
        train(tmp_path / "nonexistent_data", output)
    assert previous.read_text() == '{"previous": true}\n'
    assert list(output.iterdir()) == [previous]
