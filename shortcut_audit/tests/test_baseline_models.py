from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from pandas.testing import assert_frame_equal

from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from shortcut_audit.auditlib.baseline_features import (
    ClinicalFeatureSpec,
    clinical_features,
    clinical_geometry_features,
    geometry_features,
    static_t0_features,
    timepoint_only_features,
)
from shortcut_audit.auditlib.baseline_models import (
    BASELINE_FEATURE_SETS,
    BaselineReadoutConfig,
    fit_fold_baseline,
    load_baseline_bundle,
    predict_fold_baseline,
    save_baseline_bundle,
)
from shortcut_audit.auditlib.contracts import DECISION_POINTS, PREDICTION_COLUMNS


def _config() -> BaselineReadoutConfig:
    return BaselineReadoutConfig(
        penalties=("l1", "l2"),
        c_grid=(0.03, 0.3),
        decision_weights=(2.0, 1.0, 0.5),
        max_iter=1000,
        random_state=29,
    )


class SyntheticBaselines:
    def __init__(self) -> None:
        rng = np.random.default_rng(20260806)
        self.n = 40
        self.labels = np.arange(self.n, dtype=np.int64) % 2
        self.patient_ids = np.asarray([f"P{index:03d}" for index in range(self.n)])
        self.train = np.arange(0, 20)
        self.validation = np.arange(20, 30)
        self.test = np.arange(30, 40)

        self.records = [
            PatientRecord(
                patient_id=str(self.patient_ids[index]),
                cohort="ispy2",
                arm="arm-a" if index % 2 == 0 else "arm-b",
                hr=int(self.labels[index]),
                her2=int(index % 3 == 0),
                mp=int(index % 4 < 2),
                age=42.0 + index,
                manifest_path=Path("unused.json"),
                pcr=int(self.labels[index]),
            )
            for index in range(self.n)
        ]
        # Clinical vocabulary and age normalization are train-fitted only.
        self.clinical_spec = ClinicalFeatureSpec.fit(
            [self.records[index] for index in self.train]
        )
        geometry = rng.uniform(0.5, 2.0, size=(self.n, 4, 9)).astype(np.float32)
        geometry[:, :3, 0] += self.labels[:, None] * np.asarray(
            [0.15, 0.35, 0.60], dtype=np.float32
        )
        self.geometry = geometry
        t0_state = rng.normal(0.0, 0.4, size=(self.n, 6)).astype(np.float32)
        t0_state[:, 0] += self.labels * 0.8
        self.t0_state = t0_state

    def feature_sets(self) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
        clinical = clinical_features(self.records, self.clinical_spec)
        geometry = geometry_features(self.geometry)
        combined = clinical_geometry_features(
            self.records, self.geometry, self.clinical_spec
        )
        timepoint = timepoint_only_features(self.n)
        static = static_t0_features(self.t0_state)

        geometry_names = tuple(
            f"geometry_feature_{index:03d}" for index in range(geometry.shape[-1])
        )
        return {
            "F1": (clinical, tuple(self.clinical_spec.feature_names)),
            "F2": (geometry, geometry_names),
            "F3": (
                combined,
                (
                    *self.clinical_spec.feature_names_without_decision,
                    *geometry_names,
                ),
            ),
            "F4": (
                timepoint,
                tuple(f"decision={name}" for name in DECISION_POINTS),
            ),
            "F5": (
                static,
                (
                    *(f"static_t0_state_{index:03d}" for index in range(6)),
                    *(f"decision={name}" for name in DECISION_POINTS),
                ),
            ),
        }


class FoldBaselineModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = SyntheticBaselines()

    def _fit(self, baseline_id: str, labels: np.ndarray | None = None):
        features, names = self.data.feature_sets()[baseline_id]
        return fit_fold_baseline(
            features,
            self.data.labels if labels is None else labels,
            self.data.patient_ids,
            self.data.train,
            self.data.validation,
            fold=3,
            baseline_id=baseline_id,
            feature_names=names,
            test_indices=self.data.test,
            config=_config(),
            feature_provenance={
                "constructor": BASELINE_FEATURE_SETS[baseline_id],
                "clinical_spec_fit": "fold_train_only",
            },
        )

    def test_all_five_feature_sets_fit_one_shared_class_balanced_model(self) -> None:
        feature_sets = self.data.feature_sets()
        self.assertEqual(set(feature_sets), set(BASELINE_FEATURE_SETS))
        for baseline_id, (features, names) in feature_sets.items():
            with self.subTest(baseline_id=baseline_id):
                bundle = self._fit(baseline_id)
                prediction = predict_fold_baseline(
                    bundle,
                    features[self.data.test],
                    self.data.patient_ids[self.data.test],
                    self.data.labels[self.data.test],
                    feature_names=names,
                    checkpoint=f"fold3/{baseline_id}/readout.pkl",
                )

                self.assertEqual(bundle.baseline_id, baseline_id)
                self.assertEqual(
                    bundle.baseline_name, BASELINE_FEATURE_SETS[baseline_id]
                )
                self.assertEqual(bundle.feature_dim, features.shape[-1])
                self.assertEqual(bundle.feature_names, names)
                self.assertEqual(tuple(prediction.columns), PREDICTION_COLUMNS)
                self.assertEqual(len(prediction), len(self.data.test) * 3)
                self.assertEqual(
                    prediction["audit_condition"].unique().tolist(),
                    [
                        f"simplified_baseline_{baseline_id.lower()}_"
                        f"{BASELINE_FEATURE_SETS[baseline_id]}"
                    ],
                )
                estimator = bundle.model.named_steps["logisticregression"]
                self.assertEqual(estimator.class_weight, "balanced")
                self.assertEqual(estimator.solver, "liblinear")
                self.assertEqual(bundle.model.n_features_in_, features.shape[-1])
                self.assertEqual(
                    bundle.audit_metadata()["supervision_contract"]["model_fit"],
                    "train_only_shared_three_decision_stack",
                )

                weights = np.asarray([2.0, 1.0, 0.5])
                for row in bundle.grid_search:
                    auc = row["validation_decision_point_auroc"]
                    expected = np.dot(
                        [auc[name] for name in DECISION_POINTS], weights
                    ) / weights.sum()
                    self.assertAlmostEqual(
                        row["validation_selection_auroc"], expected
                    )

    def test_changing_test_labels_cannot_change_model_threshold_or_probability(
        self,
    ) -> None:
        features, names = self.data.feature_sets()["F3"]
        first = self._fit("F3")
        changed = self.data.labels.copy()
        changed[self.data.test] = 1 - changed[self.data.test]
        second = self._fit("F3", changed)

        self.assertEqual(first.grid_search, second.grid_search)
        self.assertEqual(
            first.hyperparameter_selection, second.hyperparameter_selection
        )
        self.assertEqual(first.threshold_selection, second.threshold_selection)
        self.assertEqual(first.thresholds, second.thresholds)
        for attribute in ("coef_", "intercept_"):
            np.testing.assert_array_equal(
                getattr(first.model.named_steps["logisticregression"], attribute),
                getattr(second.model.named_steps["logisticregression"], attribute),
            )

        first_prediction = predict_fold_baseline(
            first,
            features[self.data.test],
            self.data.patient_ids[self.data.test],
            self.data.labels[self.data.test],
            feature_names=names,
            checkpoint="f3.pkl",
        )
        second_prediction = predict_fold_baseline(
            second,
            features[self.data.test],
            self.data.patient_ids[self.data.test],
            changed[self.data.test],
            feature_names=names,
            checkpoint="f3.pkl",
        )
        model_output_columns = [
            "patient_id",
            "fold",
            "decision_point",
            "audit_condition",
            "predicted_probability",
            "predicted_label",
            "threshold",
            "checkpoint",
        ]
        assert_frame_equal(
            first_prediction[model_output_columns],
            second_prediction[model_output_columns],
        )
        np.testing.assert_array_equal(
            1 - first_prediction["y_true"].to_numpy(),
            second_prediction["y_true"].to_numpy(),
        )

        # Fit does not even attempt to convert held-out outcome objects.
        poisoned = self.data.labels.astype(object)
        poisoned[self.data.test] = "not-visible-during-fit"
        poisoned_bundle = self._fit("F3", poisoned)
        self.assertEqual(first.grid_search, poisoned_bundle.grid_search)
        self.assertEqual(first.thresholds, poisoned_bundle.thresholds)

    def test_prediction_shape_patient_order_feature_order_and_decision_order(
        self,
    ) -> None:
        features, names = self.data.feature_sets()["F2"]
        bundle = self._fit("F2")
        prediction = predict_fold_baseline(
            bundle,
            features[self.data.test],
            self.data.patient_ids[self.data.test],
            self.data.labels[self.data.test],
            feature_names=names,
            checkpoint="f2.pkl",
        )
        expected_keys = [
            (str(patient_id), decision_point)
            for patient_id in self.data.patient_ids[self.data.test]
            for decision_point in DECISION_POINTS
        ]
        self.assertEqual(
            list(zip(prediction["patient_id"], prediction["decision_point"])),
            expected_keys,
        )
        for decision_point in DECISION_POINTS:
            rows = prediction[prediction["decision_point"] == decision_point]
            self.assertTrue(
                rows["threshold"].eq(bundle.thresholds[decision_point]).all()
            )
            np.testing.assert_array_equal(
                rows["predicted_label"],
                (
                    rows["predicted_probability"] >= rows["threshold"]
                ).astype(int),
            )

        with self.assertRaisesRegex(ValueError, "IDs/order"):
            predict_fold_baseline(
                bundle,
                features[self.data.test][::-1],
                self.data.patient_ids[self.data.test][::-1],
                self.data.labels[self.data.test][::-1],
                feature_names=names,
                checkpoint="f2.pkl",
            )
        swapped_names = (names[1], names[0], *names[2:])
        with self.assertRaisesRegex(ValueError, "feature_names/order"):
            predict_fold_baseline(
                bundle,
                features[self.data.test],
                self.data.patient_ids[self.data.test],
                self.data.labels[self.data.test],
                feature_names=swapped_names,
                checkpoint="f2.pkl",
            )
        with self.assertRaisesRegex(ValueError, "features dim"):
            predict_fold_baseline(
                bundle,
                features[self.data.test, :, :-1],
                self.data.patient_ids[self.data.test],
                self.data.labels[self.data.test],
                feature_names=names,
                checkpoint="f2.pkl",
            )

    def test_bundle_round_trip_preserves_model_and_provenance(self) -> None:
        features, names = self.data.feature_sets()["F5"]
        bundle = self._fit("F5")
        expected = predict_fold_baseline(
            bundle,
            features[self.data.test],
            self.data.patient_ids[self.data.test],
            self.data.labels[self.data.test],
            feature_names=names,
            checkpoint="f5.pkl",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_baseline_bundle(bundle, Path(directory) / "f5.pkl")
            restored = load_baseline_bundle(path)
            actual = predict_fold_baseline(
                restored,
                features[self.data.test],
                self.data.patient_ids[self.data.test],
                self.data.labels[self.data.test],
                feature_names=names,
                checkpoint="f5.pkl",
            )
        assert_frame_equal(expected, actual)
        self.assertEqual(bundle.audit_metadata(), restored.audit_metadata())
        metadata = restored.audit_metadata()
        self.assertEqual(metadata["tensor_contract"], "[N,3,F]")
        self.assertEqual(
            metadata["feature_provenance"]["clinical_spec_fit"],
            "fold_train_only",
        )
        self.assertIn("scikit_learn", metadata["software_provenance"])


if __name__ == "__main__":
    unittest.main()
