from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from shortcut_audit.auditlib.metrics import (
    aggregate_patient_predictions,
    binary_classification_metrics,
    cosine_error,
    cosine_similarity,
    difference_from_native,
    featurewise_layer_norm,
    fold_mean_sample_std,
    layer_norm_mse,
    normalized_transition_gain,
    paired_patient_bootstrap_difference,
    percentage_improvement,
    raw_mse,
    summarize_binary_folds,
    summarize_transition_folds,
    transition_metrics,
)


class LatentMetricTests(unittest.TestCase):
    def test_featurewise_layer_norm_matches_population_variance_formula(self) -> None:
        values = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
        eps = 1e-5
        expected_mean = values.mean(axis=-1, keepdims=True)
        expected_variance = ((values - expected_mean) ** 2).mean(axis=-1, keepdims=True)
        expected = (values - expected_mean) / np.sqrt(expected_variance + eps)
        np.testing.assert_allclose(
            featurewise_layer_norm(values), expected, rtol=0, atol=1e-14
        )
        np.testing.assert_array_equal(featurewise_layer_norm(values)[1], np.zeros(3))

    def test_layer_norm_mse_and_raw_mse_are_distinct(self) -> None:
        prediction = np.array([[2.0, 4.0, 6.0], [10.0, 10.0, 10.0]])
        target = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        # Epsilon makes scaled affine copies differ very slightly; use the full
        # independent LayerNorm formula rather than assuming exact scale invariance.
        expected = (
            (featurewise_layer_norm(prediction) - featurewise_layer_norm(target)) ** 2
        ).mean(axis=-1)
        np.testing.assert_allclose(
            layer_norm_mse(prediction, target), expected, atol=0, rtol=0
        )
        self.assertLess(expected[0], 1e-9)
        self.assertEqual(expected[1], 0.0)
        np.testing.assert_allclose(raw_mse(prediction, target), [14.0 / 3.0, 64.0])

    def test_cosine_similarity_error_and_zero_vector_policy(self) -> None:
        prediction = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
        target = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        similarity = cosine_similarity(prediction, target)
        error = cosine_error(prediction, target)
        np.testing.assert_allclose(similarity[:2], [0.0, -1.0])
        np.testing.assert_allclose(error[:2], [1.0, 2.0])
        self.assertTrue(np.isnan(similarity[2]))
        self.assertTrue(np.isnan(error[2]))

    def test_transition_gain_direction_and_percentage(self) -> None:
        learned = np.array([1.0, 3.0, 2.0])
        copied = np.array([2.0, 2.0, 2.0])
        gain = normalized_transition_gain(learned, copied, epsilon=1e-12)
        self.assertGreater(gain[0], 0)
        self.assertLess(gain[1], 0)
        self.assertAlmostEqual(gain[2], 0.0)
        np.testing.assert_allclose(
            percentage_improvement(learned, copied), 100.0 * gain
        )
        self.assertAlmostEqual(gain[0], 0.5, places=11)

    def test_transition_metrics_preserves_metadata_and_primary_error(self) -> None:
        learned = np.array([[1.0, 2.0], [2.0, 1.0]])
        copied = np.array([[2.0, 1.0], [1.0, 2.0]])
        target = np.array([[1.0, 3.0], [1.0, 2.0]])
        metadata = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "fold": [0, 1],
                "transition": ["T0-T1", "T1-T2"],
            }
        )
        result = transition_metrics(learned, copied, target, metadata=metadata)
        self.assertEqual(result["patient_id"].tolist(), ["p1", "p2"])
        self.assertEqual(result["transition"].tolist(), ["T0-T1", "T1-T2"])
        self.assertGreater(result.loc[0, "normalized_transition_gain"], 0)
        self.assertLess(result.loc[1, "normalized_transition_gain"], 0)
        np.testing.assert_allclose(
            result["percentage_improvement"], 100 * result["normalized_transition_gain"]
        )

    def test_transition_input_boundaries_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            layer_norm_mse(np.ones((2, 3)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            featurewise_layer_norm(np.ones((2, 0)))
        with self.assertRaises(ValueError):
            normalized_transition_gain([-1.0], [1.0])


class BinaryMetricTests(unittest.TestCase):
    def test_binary_metrics_known_values(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.2, 0.9])
        metrics = binary_classification_metrics(labels, probabilities)
        self.assertAlmostEqual(metrics["auroc"], 0.75)
        self.assertAlmostEqual(metrics["auprc"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)

    def test_row_specific_thresholds(self) -> None:
        metrics = binary_classification_metrics(
            [0, 1], [0.6, 0.4], threshold=np.array([0.7, 0.3])
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)

    def test_single_class_fold_has_explicit_nan_discrimination(self) -> None:
        metrics = binary_classification_metrics([0, 0], [0.1, 0.8])
        self.assertTrue(np.isnan(metrics["auroc"]))
        self.assertTrue(np.isnan(metrics["auprc"]))
        self.assertTrue(np.isnan(metrics["sensitivity"]))
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)

        empty = binary_classification_metrics([], [])
        for metric in ("auroc", "auprc", "accuracy", "sensitivity", "specificity"):
            self.assertTrue(np.isnan(empty[metric]))

    def test_fold_mean_sample_std_and_pooled_oof_are_not_conflated(self) -> None:
        predictions = pd.DataFrame(
            {
                "fold": [0, 0, 1, 1],
                "y_true": [0, 1, 0, 1],
                "predicted_probability": [0.1, 0.9, 0.8, 0.2],
                "threshold": [0.5, 0.5, 0.5, 0.5],
            }
        )
        result = summarize_binary_folds(predictions, threshold="threshold")
        folds = result["fold_metrics"].set_index("fold")
        self.assertEqual(folds.loc[0, "auroc"], 1.0)
        self.assertEqual(folds.loc[1, "auroc"], 0.0)
        summary = result["fold_summary"].set_index("metric")
        self.assertAlmostEqual(summary.loc["auroc", "mean"], 0.5)
        self.assertAlmostEqual(summary.loc["auroc", "sample_std"], np.sqrt(0.5))
        self.assertAlmostEqual(result["pooled_oof"]["auroc"], 0.75)
        self.assertAlmostEqual(result["pooled_oof"]["auprc"], 5.0 / 6.0)

    def test_fold_summary_ignores_nan_and_uses_sample_std(self) -> None:
        folds = pd.DataFrame({"auroc": [0.6, 0.8, np.nan]})
        summary = fold_mean_sample_std(folds, metric_columns=["auroc"]).iloc[0]
        self.assertAlmostEqual(summary["mean"], 0.7)
        self.assertAlmostEqual(summary["sample_std"], np.sqrt(0.02))
        self.assertEqual(summary["n_valid_folds"], 2)

    def test_binary_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            binary_classification_metrics([0, 2], [0.1, 0.9])
        with self.assertRaises(ValueError):
            binary_classification_metrics([0, 1], [0.1, 1.1])
        with self.assertRaises(ValueError):
            binary_classification_metrics([0], [0.1], threshold=-0.1)


class PatientAggregationAndBootstrapTests(unittest.TestCase):
    @staticmethod
    def _paired_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        reference = pd.DataFrame(
            {
                "patient_id": ["a", "b", "c", "d"],
                "fold": [0, 0, 1, 1],
                "y_true": [0, 0, 1, 1],
                "predicted_probability": [0.1, 0.8, 0.2, 0.9],
            }
        )
        comparison = pd.DataFrame(
            {
                "patient_id": ["a", "b", "c", "d"],
                "fold": [0, 0, 1, 1],
                "y_true": [0, 0, 1, 1],
                "predicted_probability": [0.1, 0.2, 0.8, 0.9],
            }
        )
        return reference, comparison

    def test_patient_aggregation_means_repeated_transitions(self) -> None:
        repeated = pd.DataFrame(
            {
                "patient_id": ["a", "a", "b"],
                "fold": [0, 0, 1],
                "y_true": [0, 0, 1],
                "predicted_probability": [0.1, 0.3, 0.8],
            }
        )
        aggregated = aggregate_patient_predictions(
            repeated, "predicted_probability", constant_columns=["fold"]
        ).set_index("patient_id")
        self.assertAlmostEqual(aggregated.loc["a", "predicted_probability"], 0.2)
        self.assertEqual(aggregated.loc["a", "n_records"], 2)
        self.assertEqual(aggregated.loc["b", "n_records"], 1)

    def test_paired_bootstrap_direction_and_determinism(self) -> None:
        reference, comparison = self._paired_frames()
        first = paired_patient_bootstrap_difference(
            reference,
            comparison,
            pair_columns=["fold"],
            n_bootstrap=250,
            seed=37,
        )
        second = paired_patient_bootstrap_difference(
            reference,
            comparison,
            pair_columns=["fold"],
            n_bootstrap=250,
            seed=37,
        )
        assert_frame_equal(first["summary"], second["summary"])
        assert_frame_equal(first["bootstrap_samples"], second["bootstrap_samples"])
        summary = first["summary"].set_index("metric")
        self.assertAlmostEqual(summary.loc["auroc", "difference"], 0.25)
        self.assertGreater(summary.loc["auprc", "difference"], 0)
        self.assertEqual(
            summary.loc["auroc", "difference_direction"], "comparison - reference"
        )

        reversed_result = paired_patient_bootstrap_difference(
            comparison, reference, pair_columns=["fold"], n_bootstrap=20, seed=37
        )["summary"].set_index("metric")
        self.assertAlmostEqual(reversed_result.loc["auroc", "difference"], -0.25)

    def test_bootstrap_aggregates_duplicates_before_resampling(self) -> None:
        reference, comparison = self._paired_frames()
        repeated_reference = pd.concat(
            [
                reference.assign(
                    predicted_probability=reference["predicted_probability"] - 0.05
                ),
                reference.assign(
                    predicted_probability=reference["predicted_probability"] + 0.05
                ),
            ],
            ignore_index=True,
        )
        repeated_comparison = pd.concat(
            [
                comparison.assign(
                    predicted_probability=comparison["predicted_probability"] - 0.05
                ),
                comparison.assign(
                    predicted_probability=comparison["predicted_probability"] + 0.05
                ),
            ],
            ignore_index=True,
        )
        base = paired_patient_bootstrap_difference(
            reference, comparison, n_bootstrap=100, seed=11
        )
        repeated = paired_patient_bootstrap_difference(
            repeated_reference, repeated_comparison, n_bootstrap=100, seed=11
        )
        assert_frame_equal(base["summary"], repeated["summary"])
        assert_frame_equal(base["bootstrap_samples"], repeated["bootstrap_samples"])
        self.assertTrue(
            (repeated["patient_predictions"]["reference_n_records"] == 2).all()
        )

    def test_bootstrap_single_class_returns_nan_ci(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["a", "b"],
                "y_true": [0, 0],
                "predicted_probability": [0.1, 0.2],
            }
        )
        result = paired_patient_bootstrap_difference(
            frame, frame, n_bootstrap=20, seed=0
        )
        for row in result["summary"].itertuples():
            self.assertTrue(np.isnan(row.difference))
            self.assertTrue(np.isnan(row.ci_lower))
            self.assertTrue(np.isnan(row.ci_upper))
            self.assertEqual(row.n_valid_bootstrap, 0)

    def test_bootstrap_requires_valid_pairs(self) -> None:
        reference, comparison = self._paired_frames()
        with self.assertRaises(ValueError):
            paired_patient_bootstrap_difference(
                reference, comparison.iloc[:-1], n_bootstrap=10
            )
        mismatched = comparison.copy()
        mismatched.loc[0, "y_true"] = 1
        with self.assertRaises(ValueError):
            paired_patient_bootstrap_difference(reference, mismatched, n_bootstrap=10)

    def test_transition_fold_summary_uses_equal_patient_weight(self) -> None:
        transitions = pd.DataFrame(
            {
                "patient_id": ["a", "a", "b", "c"],
                "fold": [0, 0, 0, 1],
                "normalized_transition_gain": [0.2, 0.4, 0.5, 0.0],
            }
        )
        result = summarize_transition_folds(
            transitions, metric_columns=["normalized_transition_gain"]
        )
        patient = result["patient_metrics"].set_index("patient_id")
        self.assertAlmostEqual(patient.loc["a", "normalized_transition_gain"], 0.3)
        folds = result["fold_metrics"].set_index("fold")
        self.assertAlmostEqual(folds.loc[0, "normalized_transition_gain"], 0.4)
        self.assertAlmostEqual(folds.loc[1, "normalized_transition_gain"], 0.0)
        self.assertAlmostEqual(
            result["pooled_transition_metrics"]["normalized_transition_gain"], 0.275
        )
        self.assertAlmostEqual(
            result["pooled_patient_metrics"]["normalized_transition_gain"], 0.8 / 3.0
        )
        fold_summary = result["fold_summary"].set_index("metric")
        self.assertAlmostEqual(
            fold_summary.loc["normalized_transition_gain", "mean"], 0.2
        )
        self.assertAlmostEqual(
            fold_summary.loc["normalized_transition_gain", "sample_std"], np.sqrt(0.08)
        )


class NativeDifferenceTests(unittest.TestCase):
    def test_difference_sign_and_zero_division_guard(self) -> None:
        result = difference_from_native(
            {"auroc": 0.8, "zero_metric": 0.0},
            {"auroc": 0.7, "zero_metric": 0.1},
        ).set_index("metric")
        self.assertAlmostEqual(result.loc["auroc", "absolute_difference"], -0.1)
        self.assertAlmostEqual(result.loc["auroc", "relative_difference"], -0.125)
        self.assertAlmostEqual(
            result.loc["auroc", "relative_percentage_difference"], -12.5
        )
        self.assertTrue(result.loc["auroc", "relative_defined"])
        self.assertTrue(np.isnan(result.loc["zero_metric", "relative_difference"]))
        self.assertFalse(result.loc["zero_metric", "relative_defined"])
        self.assertEqual(
            result.loc["auroc", "difference_direction"], "comparison - native"
        )

    def test_scalar_difference_returns_auditable_row(self) -> None:
        result = difference_from_native(0.5, 0.75)
        self.assertEqual(result.loc[0, "metric"], "value")
        self.assertAlmostEqual(result.loc[0, "absolute_difference"], 0.25)
        self.assertAlmostEqual(result.loc[0, "relative_difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
