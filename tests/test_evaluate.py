import gzip
import json
import sys
import unittest

import numpy as np
import pandas as pd

from src.benchmark.evaluate import ENZYMES, evaluate, main


def cohort(labels=(0, 0, 1, 1)):
    return pd.DataFrame({
        "molecule_id": [f"m{i}" for i in range(len(labels))],
        "smiles": ["C"] * len(labels), "scaffold": [f"s{i}" for i in range(len(labels))],
        **{enzyme: labels for enzyme in ENZYMES},
    })


def prediction(molecule_id, score, status="ok"):
    return {"molecule_id": molecule_id, "arm": "classifier", "status": status,
            "scores": {enzyme: score for enzyme in ENZYMES} if status == "ok" else None}


def records(scores=(0.1, 0.1, 0.9, 0.9)):
    return [prediction(f"m{i}", score) for i, score in enumerate(scores)]


def row(frame, **filters):
    for key, value in filters.items():
        frame = frame[frame[key] == value]
    return frame.iloc[0]


class EvaluationTests(unittest.TestCase):
    def test_perfect_ranking_scores_one_with_a_degenerate_interval(self):
        metrics, _contrasts, summary = evaluate(cohort(), records(), bootstrap=40)
        self.assertTrue((metrics.value == 1).all())
        self.assertTrue((metrics.bootstrap_valid + metrics.bootstrap_skipped == 40).all())
        self.assertTrue((metrics.purpose == "primary").all())
        self.assertEqual(summary["cohort_size"], 4)

    def test_sparse_labels_shrink_the_evaluable_population(self):
        data = cohort()
        data.loc[1, "CYP2D6"] = np.nan
        metrics, _contrasts, _ = evaluate(data, records(), bootstrap=0)
        sparse = row(metrics, enzyme="CYP2D6", metric="auc")
        self.assertEqual((sparse.n_population, sparse.n_labeled, sparse.n_evaluable), (4, 3, 3))
        self.assertEqual(row(metrics, enzyme="CYP1A2", metric="auc").n_evaluable, 4)

    def test_failures_and_missing_records_stay_separate_in_coverage(self):
        predictions = [r for r in records() if r["molecule_id"] != "m0"]
        predictions[0].update(status="error", scores=None)
        metrics, _contrasts, summary = evaluate(cohort(), predictions, bootstrap=10)
        coverage = row(pd.DataFrame(summary["coverage"]), arm="classifier")
        self.assertEqual((coverage.planned, coverage.attempted, coverage.succeeded), (4, 3, 2))
        self.assertEqual((coverage.error, coverage.missing), (1, 1))
        self.assertEqual(row(metrics, enzyme="CYP1A2", metric="auc").n_evaluable, 2)

    def test_cluster_bootstrap_preserves_clusters_and_is_reproducible(self):
        data = cohort()
        data["scaffold"] = ["negative", "negative", "positive", "positive"]
        first = evaluate(data, records(), bootstrap=40, seed=7)
        second = evaluate(data, records(), bootstrap=40, seed=7)
        pd.testing.assert_frame_equal(first[0], second[0])
        metric = row(first[0], enzyme="CYP1A2", metric="auc")
        # Resampling whole scaffolds often draws one class, so those draws are skipped.
        self.assertEqual((metric.value, metric.ci_low, metric.ci_high), (1, 1, 1))
        self.assertGreater(metric.bootstrap_skipped, 0)
        self.assertGreater(metric.bootstrap_valid, 0)

    def test_single_class_is_explicitly_undefined(self):
        metrics, _contrasts, _ = evaluate(cohort([1, 1, 1, 1]), records(), bootstrap=10)
        self.assertTrue(metrics.value.isna().all())
        self.assertTrue((metrics.undefined_reason == "requires_both_classes").all())
        self.assertTrue((metrics.bootstrap_skipped == 10).all())

    def test_duplicate_predictions_and_invalid_scores_rejected(self):
        predictions = records()
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            evaluate(cohort(), predictions + [predictions[0]], bootstrap=0)
        predictions[0]["scores"]["CYP1A2"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid successful prediction score"):
            evaluate(cohort(), predictions, bootstrap=0)

    def test_retired_llm_arms_are_rejected(self):
        for arm in ("tools", "smiles", "name"):
            stale = dict(records()[0], arm=arm)
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "Unexpected prediction"):
                evaluate(cohort(), [stale], bootstrap=0)


def test_cli_reads_released_gzip_predictions(tmp_path, monkeypatch):
    cohort().to_csv(tmp_path / "cohort.csv", index=False)
    with gzip.open(tmp_path / "predictions.jsonl.gz", "wt", encoding="utf-8") as output:
        output.write("\n".join(json.dumps(record) for record in records()) + "\n\n")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--run-dir", str(tmp_path), "--bootstrap", "0"])

    main()

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert metrics.value.eq(1).all()
    assert metrics.n_evaluable.eq(4).all()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["coverage"][0]["succeeded"] == 4


def test_cli_prefers_plain_predictions_when_both_files_exist(tmp_path, monkeypatch):
    cohort().to_csv(tmp_path / "cohort.csv", index=False)
    (tmp_path / "predictions.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records()) + "\n")
    with gzip.open(tmp_path / "predictions.jsonl.gz", "wt", encoding="utf-8") as output:
        output.write("\n".join(json.dumps(record) for record in records((0.9, 0.9, 0.1, 0.1))))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--run-dir", str(tmp_path), "--bootstrap", "0"])

    main()

    assert pd.read_csv(tmp_path / "metrics.csv").value.eq(1).all()


if __name__ == "__main__":
    unittest.main()
