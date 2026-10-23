"""CYP3A4 assay scoring and deterministic, constrained greedy shortlisting.

Scores and validation percentiles are uncalibrated ranking context, not clinical
risk. Duplicate canonical structures are rejected; choose one ID per structure.
Canonicalization preserves salts and stereochemistry, without tautomer merging.
"""

from math import isfinite
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
from rdkit import DataStructs, rdBase

from ..chemistry.features import canonical_smiles, featurize, fingerprints, nearest_distances


def _validated_rows(candidates, structure_key):
    if not isinstance(candidates, list):
        raise ValueError("Candidates must be a list")
    rows, seen_ids, seen_structures = [], set(), {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be a dictionary")
        identifier = candidate.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Candidate ID must be a nonempty string")
        if identifier in seen_ids:
            raise ValueError(f"Duplicate ID: {identifier!r}")
        try:
            smiles = canonical_smiles(candidate.get(structure_key))
        except ValueError as exc:
            raise ValueError(f"Candidate {identifier!r}: {exc}") from exc
        if smiles in seen_structures:
            raise ValueError(f"Duplicate structure for {seen_structures[smiles]!r} and "
                             f"{identifier!r}; keep one ID per canonical structure")
        seen_ids.add(identifier)
        seen_structures[smiles] = identifier
        rows.append({**candidate, "canonical_smiles": smiles})
    return rows


class CYP3A4Screen:
    """Load only the trusted local CYP3A4 artifact; never access test labels."""

    def __init__(self, artifacts):
        self.bundle = joblib.load(Path(artifacts) / "models" / "CYP3A4.joblib")
        if self.bundle["enzyme"] != "CYP3A4":
            raise ValueError("Expected a CYP3A4 model bundle")
        if self.bundle["rdkit_version"] != rdBase.rdkitVersion:
            raise ValueError("RDKit version differs from training; use the locked environment")
        self.training_fps = fingerprints(self.bundle["train_smiles"])
        self.validation_scores = np.asarray(self.bundle["validation_scores"], dtype=float)
        if not self.validation_scores.size or not np.isfinite(self.validation_scores).all():
            raise ValueError("Model requires finite validation scores for percentiles")

    def score(self, candidates):
        """Validate the whole input before one prediction batch; ignore extra fields."""
        rows = _validated_rows(candidates, "smiles")
        if not rows:
            return []
        smiles = [row["canonical_smiles"] for row in rows]
        features = featurize(smiles, self.bundle["descriptor_names"])
        scores = self.bundle["model"].predict_proba(features)[:, 1]
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Model returned invalid scores")
        distances = nearest_distances(fingerprints(smiles), self.training_fps)
        return [{"id": row["id"], "canonical_smiles": row["canonical_smiles"],
                 "score": float(score),
                 "validation_percentile": float(np.mean(self.validation_scores <= score)),
                 "nearest_training_similarity": float(1 - distance),
                 "similarity_flag": bool(distance > self.bundle["domain_threshold"]),
                 "confidence": "uncalibrated"}
                for row, score, distance in zip(rows, scores, distances)]


def select_shortlist(scored, budget, include_ids=None, exclude_ids=None, max_similarity=.8):
    """Reserve mandatory picks, then greedily fill by score descending and ID ascending.

    All selected pairwise Morgan Tanimoto similarities must be <= max_similarity.
    A shortfall never relaxes constraints and is not proof no feasible batch exists:
    this greedy algorithm does not search for a globally optimal solution.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if (isinstance(max_similarity, bool) or not isinstance(max_similarity, Real)
            or not isfinite(float(max_similarity)) or not 0 <= float(max_similarity) <= 1):
        raise ValueError("max_similarity must be a finite number in [0, 1]")
    rows = _validated_rows(scored, "canonical_smiles")
    for row in rows:
        score = row.get("score")
        if (isinstance(score, bool) or not isinstance(score, Real)
                or not isfinite(float(score)) or not 0 <= float(score) <= 1):
            raise ValueError(f"Candidate {row['id']!r} score must be finite and in [0, 1]")
    ranked = sorted(rows, key=lambda row: (-row["score"], row["id"]))
    by_id = {row["id"]: row for row in ranked}
    constraints = []
    for name, values in (("include_ids", include_ids), ("exclude_ids", exclude_ids)):
        values = [] if values is None else values
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{name} must be a list of candidate IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs in {name}")
        unknown = set(values) - by_id.keys()
        if unknown:
            raise ValueError(f"Unknown IDs in {name}: {sorted(unknown)}")
        constraints.append(set(values))
    include, exclude = constraints
    if include & exclude:
        raise ValueError(f"IDs appear in both include_ids and exclude_ids: {sorted(include & exclude)}")
    if len(include) > budget:
        raise ValueError("Mandatory include_ids exceed budget")
    fps = dict(zip(by_id, fingerprints([row["canonical_smiles"] for row in ranked])))
    selected_ids = [row["id"] for row in ranked if row["id"] in include]

    def conflict(identifier, selected):
        for other in selected:
            similarity = DataStructs.TanimotoSimilarity(fps[identifier], fps[other])
            if similarity > max_similarity:
                return f"similarity to {other!r} ({similarity:.6g}) exceeds max_similarity={max_similarity}"
        return None

    for index, identifier in enumerate(selected_ids):
        reason = conflict(identifier, selected_ids[:index])
        if reason:
            raise ValueError(f"Mandatory candidate {identifier!r}: {reason}")
    skipped = []
    for row in ranked:
        identifier = row["id"]
        if identifier in include:
            continue
        if identifier in exclude:
            reason = "excluded"
        elif len(selected_ids) >= budget:
            reason = "budget filled"
        else:
            reason = conflict(identifier, selected_ids)
        if reason:
            skipped.append({"id": identifier, "reason": reason})
        else:
            selected_ids.append(identifier)
    selected = [row for row in ranked if row["id"] in selected_ids]
    complete = len(selected) == budget
    eligible_count = len(rows) - len(exclude)
    shortfall = None
    if not complete:
        shortfall = (f"Requested {budget}; {eligible_count} eligible candidates; greedy selection "
                     f"found {len(selected)} under the pairwise similarity constraint "
                     f"max_similarity={max_similarity}. Constraints were not relaxed.")
    return {"selected": selected, "ranked": ranked, "skipped": skipped,
            "requested_count": budget, "selected_count": len(selected), "complete": complete,
            "shortfall_reason": shortfall, "method": "score_ordered_greedy"}
