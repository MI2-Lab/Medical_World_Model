from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import torch

from ispy_jepa_tmi_clean.corejepa.config import ReadoutConfig
from ispy_jepa_tmi_clean.corejepa.models.response_state import FutureResponseState
from shortcut_audit.auditlib.contracts import DECISION_POINTS, PREDICTION_COLUMNS
from shortcut_audit.auditlib.readouts import (
    AuditReadoutConfig,
    fit_fold_readout,
    load_readout_bundle,
    predict_fold_readout,
    save_readout_bundle,
    select_validation_threshold,
)


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    n_patients = 30
    labels = np.arange(n_patients, dtype=np.int64) % 2
    states = rng.normal(0.0, 0.35, size=(n_patients, 3, 4)).astype(np.float32)
    for landmark in range(3):
        states[:, landmark, 0] += labels * (0.6 + 0.25 * landmark)
        states[:, landmark, 1] -= labels * (0.25 + 0.10 * landmark)
    patient_ids = np.asarray([f"P{index:03d}" for index in range(n_patients)])
    return states, labels, patient_ids


def _small_config() -> AuditReadoutConfig:
    return AuditReadoutConfig(
        penalties=("l1", "l2"),
        c_grid=(0.03, 0.3),
        landmark_weights=(2.0, 1.0, 0.5),
        max_iter=1000,
        random_state=17,
    )


class ThresholdSelectionTest(unittest.TestCase):
    def test_balanced_accuracy_tie_break_is_stable_and_auditable(self) -> None:
        # Thresholds 0.8, 0.6 and 0.4 have the same best balanced accuracy.
        # Of the two closest to 0.5 (0.4 and 0.6), the fixed rule chooses lower.
        labels = np.asarray([1, 1, 0, 1, 0, 0])
        probabilities = np.asarray([0.8, 0.6, 0.6, 0.4, 0.4, 0.2])

        selection = select_validation_threshold(labels, probabilities)

        self.assertAlmostEqual(selection["threshold"], 0.4)
        self.assertAlmostEqual(selection["balanced_accuracy"], 2.0 / 3.0)
        self.assertEqual(selection["objective"], "balanced_accuracy")
        self.assertEqual(selection["tie_break"], "closest_to_0.5_then_lower")
        self.assertEqual(selection["n_best_score_candidates"], 3)
        self.assertEqual(selection["n_final_tie_candidates"], 2)

    def test_threshold_rejects_one_class_or_invalid_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "两个类别"):
            select_validation_threshold([0, 0], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            select_validation_threshold([0, 1], [0.1, 1.1])


class FoldReadoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.states, self.labels, self.patient_ids = _synthetic_data()
        self.train = np.arange(0, 16)
        self.validation = np.arange(16, 24)
        self.test = np.arange(24, 30)

    def _fit(self, labels: np.ndarray | None = None):
        return fit_fold_readout(
            self.states,
            self.labels if labels is None else labels,
            self.patient_ids,
            self.train,
            self.validation,
            fold=2,
            test_indices=self.test,
            config=_small_config(),
        )

    def test_test_label_changes_cannot_affect_model_hyperparameter_or_threshold(
        self,
    ) -> None:
        first = self._fit()
        changed = self.labels.copy()
        changed[self.test] = 1 - changed[self.test]
        second = self._fit(changed)

        self.assertEqual(
            first.hyperparameter_selection, second.hyperparameter_selection
        )
        self.assertEqual(first.thresholds, second.thresholds)
        self.assertEqual(first.threshold_selection, second.threshold_selection)
        self.assertEqual(first.grid_search, second.grid_search)

        first_prediction = predict_fold_readout(
            first,
            self.states[self.test],
            self.patient_ids[self.test],
            self.labels[self.test],
            checkpoint="fold2.pt",
        )
        second_prediction = predict_fold_readout(
            second,
            self.states[self.test],
            self.patient_ids[self.test],
            changed[self.test],
            checkpoint="fold2.pt",
        )
        probability_columns = [
            "patient_id",
            "fold",
            "decision_point",
            "predicted_probability",
            "predicted_label",
            "threshold",
        ]
        assert_frame_equal(
            first_prediction[probability_columns],
            second_prediction[probability_columns],
        )
        self.assertTrue(
            np.array_equal(
                1 - first_prediction["y_true"].to_numpy(),
                second_prediction["y_true"].to_numpy(),
            )
        )

    def test_predictions_cover_each_patient_and_decision_point(self) -> None:
        bundle = self._fit()
        prediction = predict_fold_readout(
            bundle,
            self.states[self.test],
            self.patient_ids[self.test],
            np.zeros(len(self.test), dtype=np.int64),
            checkpoint="runs/retrained/fold_2/best.pt",
            audit_condition="repeated_t0_c1",
        )

        self.assertEqual(tuple(prediction.columns), PREDICTION_COLUMNS)
        self.assertEqual(len(prediction), len(self.test) * 3)
        self.assertEqual(set(prediction["decision_point"]), set(DECISION_POINTS))
        self.assertTrue(prediction.groupby("patient_id").size().eq(3).all())
        self.assertEqual(bundle.feature_dim, 20 * self.states.shape[2] + 3)
        for decision_point in DECISION_POINTS:
            rows = prediction[prediction["decision_point"] == decision_point]
            self.assertTrue(
                rows["threshold"].eq(bundle.thresholds[decision_point]).all()
            )
            expected = (rows["predicted_probability"] >= rows["threshold"]).astype(int)
            np.testing.assert_array_equal(rows["predicted_label"], expected)

        estimator = bundle.model.named_steps["logisticregression"]
        self.assertEqual(estimator.class_weight, "balanced")
        self.assertEqual(estimator.solver, "liblinear")
        json.dumps(bundle.audit_metadata())

    def test_hyperparameter_ties_prefer_smaller_c_then_l2(self) -> None:
        states = np.zeros((20, 3, 2), dtype=np.float32)
        labels = np.arange(20) % 2
        patient_ids = [f"Z{index:02d}" for index in range(20)]
        bundle = fit_fold_readout(
            states,
            labels,
            patient_ids,
            range(0, 10),
            range(10, 16),
            fold=0,
            test_indices=range(16, 20),
            config=AuditReadoutConfig(
                penalties=("l1", "l2"),
                c_grid=(1.0, 0.01),
                max_iter=200,
            ),
        )

        self.assertEqual(bundle.hyperparameter_selection["penalty"], "l2")
        self.assertEqual(bundle.hyperparameter_selection["C"], 0.01)
        self.assertEqual(
            bundle.hyperparameter_selection["tie_break"], "smaller_C_then_l2"
        )

    def test_bundle_round_trip_preserves_predictions(self) -> None:
        bundle = self._fit()
        expected = predict_fold_readout(
            bundle,
            self.states[self.test],
            self.patient_ids[self.test],
            self.labels[self.test],
            checkpoint="fold2.pt",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_readout_bundle(bundle, Path(directory) / "readout.pkl")
            restored = load_readout_bundle(path)
            actual = predict_fold_readout(
                restored,
                self.states[self.test],
                self.patient_ids[self.test],
                self.labels[self.test],
                checkpoint="fold2.pt",
            )
        assert_frame_equal(expected, actual)
        self.assertEqual(bundle.audit_metadata(), restored.audit_metadata())

    def test_patient_and_shape_alignment_fail_closed(self) -> None:
        duplicate_ids = self.patient_ids.copy()
        duplicate_ids[1] = duplicate_ids[0]
        with self.assertRaisesRegex(ValueError, "重复 ID"):
            fit_fold_readout(
                self.states,
                self.labels,
                duplicate_ids,
                self.train,
                self.validation,
                fold=2,
                test_indices=self.test,
                config=_small_config(),
            )
        with self.assertRaisesRegex(ValueError, "不得重叠"):
            fit_fold_readout(
                self.states,
                self.labels,
                self.patient_ids,
                self.train,
                np.asarray([15, 16, 17, 18]),
                fold=2,
                test_indices=self.test,
                config=_small_config(),
            )

        bundle = self._fit()
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            predict_fold_readout(
                bundle,
                self.states[self.test][::-1],
                self.patient_ids[self.test][::-1],
                self.labels[self.test][::-1],
                checkpoint="fold2.pt",
            )
        with self.assertRaisesRegex(ValueError, "response dim"):
            predict_fold_readout(
                bundle,
                np.zeros((len(self.test), 3, 5), dtype=np.float32),
                self.patient_ids[self.test],
                self.labels[self.test],
                checkpoint="fold2.pt",
            )

    def test_accepts_repo_readout_config_and_future_response_state(self) -> None:
        torch.manual_seed(7)
        module = FutureResponseState(
            geometry_dim=9,
            latent_dim=8,
            condition_dim=7,
            temporal_condition_dim=4,
            response_dim=4,
            hidden_dim=8,
            depth=1,
            heads=2,
            dropout=0.0,
            experts=2,
            expert_hidden_dim=8,
            gate_hidden_dim=4,
            gate_temperature=0.4,
            expert_scale=0.1,
            expert_init_std=0.005,
            latent_scale=0.05,
            film_scale=0.1,
        ).eval()
        with torch.no_grad():
            output = module(torch.rand(20, 3, 9), torch.rand(20, 3, 7))
        states = output.future_state.numpy()
        labels = np.arange(20) % 2
        patient_ids = [f"F{index:02d}" for index in range(20)]
        core_config = ReadoutConfig(
            penalties=("l2",),
            c_grid=(0.1,),
            landmark_weights=(2.0, 1.0, 0.5),
            max_iter=300,
        )

        bundle = fit_fold_readout(
            states,
            labels,
            patient_ids,
            range(0, 10),
            range(10, 16),
            fold=4,
            test_indices=range(16, 20),
            config=core_config,
        )
        prediction = predict_fold_readout(
            bundle,
            states[16:20],
            patient_ids[16:20],
            labels[16:20],
            checkpoint="future-response-smoke.pt",
        )

        self.assertEqual(bundle.response_dim, 4)
        self.assertEqual(bundle.feature_dim, 83)
        self.assertEqual(len(prediction), 12)


if __name__ == "__main__":
    unittest.main()
