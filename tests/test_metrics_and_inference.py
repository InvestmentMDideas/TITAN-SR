import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate  # noqa: E402
import inference  # noqa: E402
from analyses import leakage, lock_threshold  # noqa: E402


class FixedRecallMetricTests(unittest.TestCase):
    def test_cutoff_ties_are_all_screened_and_order_independent(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.9, 0.5, 0.5, 0.1])
        expected = evaluate.operating_point_at_recall(labels, scores, 0.95)
        self.assertEqual(expected["threshold"], 0.5)
        self.assertEqual(expected["screened"], 3)
        self.assertEqual(expected["achieved_recall"], 1.0)
        self.assertEqual(expected["specificity"], 0.5)
        self.assertEqual(expected["work_saved_at_recall"], 0.25)
        self.assertAlmostEqual(expected["wss_at_recall"], 0.20)

        for permutation in (
            [2, 1, 0, 3],
            [3, 2, 1, 0],
            [1, 3, 2, 0],
        ):
            actual = evaluate.operating_point_at_recall(
                labels[permutation], scores[permutation], 0.95
            )
            self.assertEqual(actual, expected)

    def test_standard_wss_is_not_raw_work_saved(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.5, 0.5, 0.1]
        raw = evaluate.work_saved_at_recall(labels, scores, 0.95)
        wss = evaluate.wss_at_recall(labels, scores, 0.95)
        self.assertAlmostEqual(raw - wss, 0.05)

    def test_locked_threshold_uses_an_observed_positive_score(self):
        positive_scores = np.asarray([0.9, 0.5, 0.5])
        threshold = lock_threshold.threshold_at_recall(positive_scores, 0.66)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(float((positive_scores >= threshold).mean()), 1.0)

    def test_output_names_distinguish_raw_work_saved_and_wss(self):
        result = evaluate.evaluate(
            {"r": {"labels": [1, 1, 0, 0], "score": [0.9, 0.5, 0.5, 0.1]}},
            bootstrap=20,
        )
        self.assertIn("work_saved_at_95_recall", result["summary"])
        self.assertIn("wss_at_95_recall", result["summary"])

    def test_evaluator_selects_an_explicit_historical_score_key(self):
        data = {
            "review": {
                "labels": [1, 1, 0, 0],
                "2model": [0.9, 0.8, 0.2, 0.1],
                "3model": [0.8, 0.7, 0.3, 0.2],
            }
        }
        result = evaluate.evaluate(data, bootstrap=10, score_key="3model")
        self.assertEqual(result["summary"]["n_reviews"], 1)
        self.assertNotIn("wss95", result["summary"])


class InferenceCompatibilityTests(unittest.TestCase):
    def test_two_model_variant_is_selectable(self):
        config = json.loads((ROOT / "config" / "ensemble.json").read_text())
        selected = inference.select_ensemble(config, "sensitivity_two_model_50_50")
        self.assertEqual(len(selected["members"]), 2)
        self.assertEqual(selected["weights"], [0.5, 0.5])

    def test_input_loader_preserves_titles_for_leakage_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["review_id", "criteria", "title", "abstract", "label"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "review_id": "r1",
                        "criteria": "RCT",
                        "title": "A useful title",
                        "abstract": "Text",
                        "label": "1",
                    }
                )
            reviews = inference.load_reviews(
                path, "review_id,criteria,title,abstract,label"
            )
        self.assertEqual(reviews["r1"]["titles"], ["A useful title"])

    def test_leakage_percentiles_average_ties(self):
        percentiles = leakage.average_percentiles([0.1, 0.5, 0.5, 0.9])
        np.testing.assert_allclose(percentiles, [0.0, 0.5, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
