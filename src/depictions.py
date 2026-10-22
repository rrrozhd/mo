"""Renderings that show why a molecule scores as it does, without implying calibration.

Three views, all read-only over the frozen artifacts:
- mcs_pair: what actually differs between a query and a proposed analog.
- attribution: how the score moves when each atom's environment is removed. This is a
  perturbation of the model's own input, not a causal claim about chemistry.
- neighbourhood: where the query sits among TRAINING-split molecules with measured
  labels; held-out partitions are excluded so no answer key is shown.
"""

from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.manifold import MDS

from .chemistry import ENZYMES, canonical_smiles, featurize, fingerprints
from .cyp_tools import ARTIFACTS, _corpus, _suite

MAX_SMILES = 4000
MAX_ATTRIBUTION_ATOMS = 60
MCS_TIMEOUT_SECONDS = 5
SHARED_GREY = (0.85, 0.85, 0.85)
DIFFERENT_ORANGE = (1.0, 0.62, 0.20)
UNSCORABLE_GREY = (0.80, 0.80, 0.80)


def _molecule(smiles):
    if not isinstance(smiles, str) or not 0 < len(smiles) <= MAX_SMILES:
        raise ValueError(f"SMILES must be a nonempty string of at most {MAX_SMILES} characters")
    molecule = Chem.MolFromSmiles(canonical_smiles(smiles))
    if molecule is None:
        raise ValueError("SMILES could not be parsed")
    return molecule


def _enzyme(name):
    if name not in ENZYMES:
        raise ValueError(f"enzyme must be one of {', '.join(ENZYMES)}")
    return name


def _svg(molecule, highlights, colors, width=360, height=260, legend=""):
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, molecule, legend=legend,
        highlightAtoms=highlights, highlightAtomColors=colors)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def depict(smiles, width=260, height=200):
    """One molecule, drawn plainly. A chemist reads a structure, not a SMILES string."""
    return _svg(_molecule(smiles), [], {}, width, height)


def mcs_pair(first_smiles, second_smiles):
    """Grey the shared scaffold, highlight what differs, for both molecules."""
    left, right = _molecule(first_smiles), _molecule(second_smiles)
    common = rdFMCS.FindMCS([left, right], timeout=MCS_TIMEOUT_SECONDS,
                            ringMatchesRingOnly=True, completeRingsOnly=True)
    pattern = Chem.MolFromSmarts(common.smartsString) if common.smartsString else None
    panels = {}
    for key, molecule in (("a", left), ("b", right)):
        shared = set(molecule.GetSubstructMatch(pattern)) if pattern is not None else set()
        atoms = list(range(molecule.GetNumAtoms()))
        colors = {index: (SHARED_GREY if index in shared else DIFFERENT_ORANGE)
                  for index in atoms}
        panels[key] = {
            "svg": _svg(molecule, atoms, colors),
            "differing_atoms": [index for index in atoms if index not in shared],
        }
    return {
        "shared_atom_count": common.numAtoms,
        "shared_bond_count": common.numBonds,
        "mcs_smarts": common.smartsString,
        "timed_out": bool(common.canceled),
        "panels": panels,
        "note": "Grey is the maximum common substructure. Orange is what differs.",
    }


@lru_cache(maxsize=2048)
def _cached_score(smiles, enzyme, suite):
    return float(suite.predict(featurize([smiles], suite.descriptor_names))["scores"][enzyme])


def _score(smiles, enzyme, *, artifacts=ARTIFACTS):
    return _cached_score(smiles, enzyme, _suite(artifacts))


def _log_odds(probability):
    """Saturated probabilities hide the shift; log-odds keeps it visible near 0 and 1."""
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return float(np.log(clipped / (1 - clipped)))


def attribution(smiles, enzyme, *, artifacts=ARTIFACTS):
    """Score shift per atom when that atom's environment is removed from the input.

    The weight for an atom is the drop in the enzyme's score after deleting it. A large
    positive weight means the model leaned on that atom. It perturbs the model's own
    features, so it explains the model, not the chemistry.
    """
    molecule, enzyme = _molecule(smiles), _enzyme(enzyme)
    canonical = canonical_smiles(smiles)
    if molecule.GetNumAtoms() > MAX_ATTRIBUTION_ATOMS:
        raise ValueError(f"attribution is limited to {MAX_ATTRIBUTION_ATOMS} atoms")
    base = _score(canonical, enzyme, artifacts=artifacts)
    base_odds = _log_odds(base)
    atoms = []
    with rdBase.BlockLogs():
        for index in range(molecule.GetNumAtoms()):
            editable = Chem.RWMol(molecule)
            editable.RemoveAtom(index)
            fragment = editable.GetMol()
            try:
                if fragment.GetNumAtoms() == 0:
                    raise ValueError("Deletion leaves an empty molecule")
                Chem.SanitizeMol(fragment)
                smiles_out = Chem.MolToSmiles(fragment)
            except (ValueError, Chem.KekulizeException, Chem.AtomValenceException):
                # Deleting a ring or aromatic atom leaves nothing that sanitizes, so this
                # atom has no weight at all. Reporting it as zero would read as "ignored".
                atoms.append({"index": index, "status": "deletion_invalid", "weight": None})
                continue
            atoms.append({
                "index": index, "status": "scored",
                "weight": base_odds - _log_odds(_score(smiles_out, enzyme, artifacts=artifacts)),
                # Deleting a bridging atom compares against a two-part mixture, not an analog.
                "fragments_molecule": "." in smiles_out,
            })
    scored = [atom for atom in atoms if atom["status"] == "scored"]
    span = max((abs(atom["weight"]) for atom in scored), default=0.0) or 1.0
    colors, highlights = {}, []
    for atom in atoms:
        highlights.append(atom["index"])
        if atom["status"] != "scored":
            colors[atom["index"]] = UNSCORABLE_GREY
            continue
        intensity = min(1.0, abs(atom["weight"]) / span)
        colors[atom["index"]] = ((1.0, 1.0 - 0.75 * intensity, 1.0 - 0.75 * intensity)
                                 if atom["weight"] > 0
                                 else (1.0 - 0.75 * intensity, 1.0 - 0.45 * intensity, 1.0))
    return {
        "query_smiles": canonical, "enzyme": enzyme,
        "base_score": round(base, 4),
        "weight_units": "log-odds shift when the atom is deleted",
        "atoms_total": len(atoms),
        "atoms_scored": len(scored),
        "atoms_unscorable": len(atoms) - len(scored),
        "atoms": [{**atom, "weight": None if atom["weight"] is None else round(atom["weight"], 4)}
                  for atom in atoms],
        "svg": _svg(molecule, highlights, colors,
                    legend=f"{enzyme} {base:.2f} (uncalibrated)"),
        "note": "Red raises the score, blue lowers it, grey could not be scored: deleting a ring "
                "or aromatic atom leaves nothing that sanitizes, so single-atom deletion says "
                "nothing about those atoms. Weights marked fragments_molecule compare against a "
                "disconnected mixture. This perturbs the model's input; it is not a mechanistic claim.",
    }


def neighbourhood(smiles, enzyme, neighbours=40, *, artifacts=ARTIFACTS):
    """Place the query among its nearest measured training molecules, in two dimensions."""
    enzyme = _enzyme(enzyme)
    query = canonical_smiles(smiles)
    _molecule(query)
    if not 5 <= neighbours <= 200:
        raise ValueError("neighbours must be between 5 and 200")
    frame, corpus_fps = _corpus(artifacts)
    query_fp = fingerprints([query])[0]
    similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, list(corpus_fps)))
    labels = np.asarray(frame[enzyme], dtype="float64")
    measured = np.flatnonzero(~np.isnan(labels))
    chosen = measured[np.argsort(-similarities[measured])[:neighbours]]
    selected = [query_fp] + [corpus_fps[index] for index in chosen]
    size = len(selected)
    distances = np.zeros((size, size))
    for row in range(size):
        row_similarity = DataStructs.BulkTanimotoSimilarity(selected[row], list(selected))
        distances[row] = 1.0 - np.asarray(row_similarity)
    coordinates = np.asarray(
        MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(distances))
    corpus_smiles = frame["smiles"].tolist()
    points = [{"smiles": query, "is_query": True, "measured_label": None,
               "similarity": 1.0, "x": float(coordinates[0][0]), "y": float(coordinates[0][1])}]
    for offset, index in enumerate(chosen, start=1):
        points.append({
            "smiles": corpus_smiles[index], "is_query": False,
            "measured_label": int(labels[index]),
            "similarity": round(float(similarities[index]), 3),
            "x": float(coordinates[offset][0]), "y": float(coordinates[offset][1]),
        })
    return {
        "query_smiles": query, "enzyme": enzyme, "points": points,
        "note": "Training-split molecules only. Distances are Tanimoto on Morgan fingerprints, projected to two dimensions. "
                "Proximity here is fingerprint similarity, not a property landscape.",
    }
