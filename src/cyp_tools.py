"""Offline CYP profiles, measured training analogs, and structure analysis."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

from .chemistry import ENZYMES, canonical_smiles, featurize, fingerprints
from .models import ModelSuite

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MAX_ANALOGS = 25


def _model_hashes(artifacts):
    return tuple((f"models/{enzyme}.joblib",
                  sha256((artifacts / "models" / f"{enzyme}.joblib").read_bytes()).hexdigest())
                 for enzyme in ENZYMES)


@lru_cache(maxsize=2)
def _load_suite(artifacts, hashes):
    suite = ModelSuite(artifacts)
    if _model_hashes(artifacts) != hashes:
        raise ValueError("Model artifacts changed while loading")
    return suite


def _suite(artifacts=ARTIFACTS):
    path = Path(artifacts).resolve()
    return _load_suite(path, _model_hashes(path))


@lru_cache(maxsize=2)
def _load_corpus(artifacts, manifest_hash):
    """The labelled TRAINING corpus plus its fingerprints, cached by artifact content.

    Held-out partitions are excluded deliberately. Returning a validation or test
    molecule as 'measured' evidence would hand back the answer key of the set the
    models were scored on.
    """
    path = artifacts / "manifest.csv"
    frame = pd.read_csv(path, dtype={"molecule_id": str, "smiles": str})
    frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    result = frame, fingerprints(frame["smiles"].tolist())
    if sha256(path.read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("Training corpus artifact changed while loading")
    return result


def _corpus(artifacts=ARTIFACTS):
    path = Path(artifacts).resolve()
    return _load_corpus(path, sha256((path / "manifest.csv").read_bytes()).hexdigest())


def _enzyme(name):
    if name not in ENZYMES:
        raise ValueError(f"enzyme must be one of {', '.join(ENZYMES)}")
    return name


def predict_cyp(smiles: str, *, artifacts=ARTIFACTS) -> dict:
    """Score one molecule against all five CYP inhibition classifiers.

    Returns an uncalibrated ranking score per enzyme, where that score sits in the
    validation distribution, and whether the molecule is outside the training
    distribution for that enzyme. A high score is not a clinical risk statement.
    """
    query = canonical_smiles(smiles)
    suite = _suite(artifacts)
    prediction = suite.predict(featurize([query], suite.descriptor_names))
    domain = suite.similarity(query)
    return {
        "query_smiles": query,
        "enzymes": {
            enzyme: {
                "score": round(prediction["scores"][enzyme], 4),
                "validation_percentile": round(prediction["validation_percentile"][enzyme], 4),
                "outside_training_distribution": domain[enzyme]["outside_training_distribution"],
            }
            for enzyme in ENZYMES
        },
        "confidence": "uncalibrated",
    }


def find_analogs(smiles: str, enzyme: str, measured_label: int | None = None,
                 max_results: int = 5, min_similarity: float = 0.4, *, artifacts=ARTIFACTS) -> dict:
    """Find similar training molecules and report their recorded dataset label.

    Set measured_label to 0 for negative TDC Veith assay-dataset labels, 1 for
    positive labels, or leave it unset for the nearest neighbours regardless of
    label. The assay response can reflect inhibition or substrate competition;
    labels do not establish mechanism or general inactivity. Similarity is
    Morgan/Tanimoto. Missing endpoint labels are skipped, not treated as negatives.
    """
    query = canonical_smiles(smiles)
    enzyme = _enzyme(enzyme)
    if measured_label not in (None, 0, 1):
        raise ValueError("measured_label must be 0, 1 or unset")
    if not 1 <= max_results <= MAX_ANALOGS:
        raise ValueError(f"max_results must be between 1 and {MAX_ANALOGS}")
    frame, corpus_fps = _corpus(artifacts)
    similarities = DataStructs.BulkTanimotoSimilarity(fingerprints([query])[0], list(corpus_fps))
    labels = np.asarray(frame[enzyme], dtype="float64")
    corpus_smiles = frame["smiles"].tolist()
    corpus_split = frame["split"].tolist()
    hits = []
    for position in sorted(range(len(similarities)), key=lambda i: -similarities[i]):
        similarity = similarities[position]
        if similarity < min_similarity or len(hits) >= max_results:
            break
        label = labels[position]
        if np.isnan(label) or corpus_smiles[position] == query:
            continue
        if measured_label is not None and int(label) != measured_label:
            continue
        hits.append({
            "smiles": corpus_smiles[position],
            "similarity": round(float(similarity), 3),
            "measured_label": int(label),
            "measured_meaning": "positive dataset label" if int(label) else "negative dataset label",
            "split": corpus_split[position],
        })
    from .depictions import depict  # Imported here: depictions reads this module's corpus.

    for hit in hits:
        hit["svg"] = depict(hit["smiles"])
        hit["name"] = None
    return {
        "query_smiles": query, "enzyme": enzyme,
        "requested_label": measured_label, "min_similarity": min_similarity,
        "analogs": hits,
        "note": "Processed TDC Veith binary labels come from the training split only; "
                "no held-out molecule is returned as evidence. The original assay response "
                "can reflect inhibition or substrate competition. Exact source binarization "
                "and exclusion rules have not been independently reconstructed. "
                "Similar structure does not guarantee similar activity.",
    }


def compare_structures(first_smiles: str, second_smiles: str) -> dict:
    """Compare two exact structures: fingerprint similarity and whether they are identical."""
    left, right = canonical_smiles(first_smiles), canonical_smiles(second_smiles)
    pair = fingerprints([left, right])
    return {
        "a": left, "b": right,
        "fingerprint_tanimoto": round(float(DataStructs.TanimotoSimilarity(pair[0], pair[1])), 3),
        "exact_identity_equal": left == right,
    }


def show_structures(first_smiles: str, second_smiles: str) -> dict:
    """Draw two molecules side by side with the shared scaffold greyed and the rest highlighted.

    Call this whenever you propose an analog, so the chemist sees what actually changed
    instead of reading two SMILES strings.
    """
    from .depictions import mcs_pair

    return mcs_pair(first_smiles, second_smiles)


def show_neighbourhood(smiles: str, enzyme: str, *, artifacts=ARTIFACTS) -> dict:
    """Place a molecule among its nearest measured training neighbours, coloured by label."""
    from .depictions import neighbourhood

    return neighbourhood(smiles, enzyme, artifacts=artifacts)


def show_attribution(smiles: str, enzyme: str, *, artifacts=ARTIFACTS) -> dict:
    """Weigh each atom by how much deleting it moves the score. Slow; call it only when asked why."""
    from .depictions import attribution

    return attribution(smiles, enzyme, artifacts=artifacts)
