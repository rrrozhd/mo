"""Load the frozen CYP bundles, score molecules, and flag out-of-domain structures."""

from pathlib import Path

import joblib
import numpy as np
from rdkit import rdBase

from ..chemistry.features import ENZYMES, fingerprints, nearest_distances


class ModelSuite:
    def __init__(self, artifacts):
        self.bundles = {
            enzyme: joblib.load(Path(artifacts) / "models" / f"{enzyme}.joblib")
            for enzyme in ENZYMES
        }
        self.descriptor_names = self.bundles[ENZYMES[0]]["descriptor_names"]
        for bundle in self.bundles.values():
            if bundle["rdkit_version"] != rdBase.rdkitVersion:
                raise ValueError("RDKit version differs from training; use the locked environment")
            if bundle["descriptor_names"] != self.descriptor_names:
                raise ValueError("Model feature schemas differ")
        self.training_fps = {
            enzyme: fingerprints(bundle["train_smiles"])
            for enzyme, bundle in self.bundles.items()
        }

    def predict(self, vector):
        scores, percentiles = {}, {}
        for enzyme, bundle in self.bundles.items():
            score = float(bundle["model"].predict_proba(vector)[0, 1])
            scores[enzyme] = score
            reference = np.asarray(bundle["validation_scores"])
            percentiles[enzyme] = float(np.mean(reference <= score))
        return {"scores": scores, "validation_percentile": percentiles,
                "confidence": "uncalibrated"}

    def similarity(self, smiles):
        query = fingerprints([smiles])
        result = {}
        for enzyme, bundle in self.bundles.items():
            distance = float(nearest_distances(query, self.training_fps[enzyme])[0])
            threshold = float(bundle["domain_threshold"])
            result[enzyme] = {"nearest_distance": distance, "threshold": threshold,
                              "outside_training_distribution": distance > threshold}
        return result
