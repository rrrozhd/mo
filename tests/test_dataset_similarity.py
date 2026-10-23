"""The skeleton key must collapse stereoisomers that canonical SMILES separates."""

from src.research.dataset_similarity import _parsed
from src.chemistry.features import canonical_smiles

L_ALANINE = "C[C@H](N)C(=O)O"
D_ALANINE = "C[C@@H](N)C(=O)O"


def test_stereoisomers_share_one_skeleton_but_two_canonical_smiles():
    assert canonical_smiles(L_ALANINE) != canonical_smiles(D_ALANINE)
    assert _parsed(L_ALANINE)[1] == _parsed(D_ALANINE)[1]


def test_distinct_molecules_keep_distinct_skeletons():
    assert _parsed(L_ALANINE)[1] != _parsed("CCO")[1]


def test_invalid_structure_is_reported_not_raised():
    assert _parsed("not-a-molecule") == (None, None)
