"""Shared, ordered molecular features for training and inference."""

import os

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# The only endpoint-specific constant. Every label column, model bundle, score
# schema and metric row derives from it, so another panel of binary endpoints
# needs ENDPOINTS set and the models retrained against a cohort with those columns.
ENZYMES = tuple(name.strip() for name in os.environ["ENDPOINTS"].split(",") if name.strip()) \
    if os.environ.get("ENDPOINTS") else ("CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4")
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)
_DESCRIPTORS = dict(Descriptors._descList)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def _molecule(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def canonical_smiles(smiles):
    """Canonical isomeric SMILES; no salt stripping or tautomer normalization."""
    return Chem.MolToSmiles(_molecule(smiles), canonical=True, isomericSmiles=True)


def fingerprints(smiles_list):
    return [_MORGAN.GetFingerprint(_molecule(smiles)) for smiles in smiles_list]


def featurize(smiles_list, descriptor_names=None):
    """Morgan bits followed by descriptors in the saved order; failures are NaN.

    Non-finite descriptor values are imputed by the training-fitted pipeline.
    Invalid molecular input raises rather than silently changing row alignment.
    """
    names = DESCRIPTOR_NAMES if descriptor_names is None else tuple(descriptor_names)
    unknown = set(names) - _DESCRIPTORS.keys()
    if unknown:
        raise ValueError(f"Unknown RDKit descriptors: {sorted(unknown)}")
    smiles_list = list(smiles_list)
    features = np.empty((len(smiles_list), MORGAN_BITS + len(names)), dtype=np.float64)
    for row, smiles in enumerate(smiles_list):
        mol = _molecule(smiles)
        features[row, :MORGAN_BITS] = _MORGAN.GetFingerprintAsNumPy(mol)
        for column, name in enumerate(names, start=MORGAN_BITS):
            try:
                value = float(_DESCRIPTORS[name](mol))
            except (ValueError, TypeError, ArithmeticError, RuntimeError):
                value = np.nan
            features[row, column] = value if np.isfinite(value) else np.nan
    return features


def nearest_distances(query_fps, training_fps, exclude_self=False):
    """Nearest Tanimoto distance, optionally omitting the aligned diagonal.

    exclude_self is only for a training set compared with itself in the same
    order. A singleton has no other neighbor and is rejected explicitly.
    """
    if not training_fps:
        raise ValueError("At least one training fingerprint is required")
    if exclude_self and (len(query_fps) != len(training_fps) or len(training_fps) < 2):
        raise ValueError("Self-exclusion requires aligned sets with at least two rows")
    distances = np.empty(len(query_fps), dtype=float)
    for index, query in enumerate(query_fps):
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query, training_fps))
        if exclude_self:
            similarities[index] = -np.inf
        distances[index] = 1.0 - similarities.max()
    return distances
