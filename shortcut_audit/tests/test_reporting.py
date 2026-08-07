from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from shortcut_audit.auditlib.reporting import (
    AuditReportingBundle,
    COPY_TRANSITIONS,
    build_audit_reporting_bundle,
    build_prediction_report,
    generate_required_figures,
    infer_figure_condition_roles,
    plotting_backend_status,
    REQUIRED_FIGURE_FILENAMES,
    REPORTING_INPUT_CONTRACTS,
    run_reporting_pipeline,
    save_reporting_tables,
    summarize_copy_latent_metrics,
    summarize_perturbation_latent_metrics,
    validate_copy_latent_metrics,
    validate_oof_predictions,
)


EXPECTED_FOLDS = (0, 1)
PATIENTS_BY_FOLD = {
    0: ("p00", "p01", "p02", "p03"),
    1: ("p10", "p11", "p12", "p13"),
}
LABELS = {
    "p00": 0,
    "p01": 0,
    "p02": 1,
    "p03": 1,
    "p10": 0,
    "p11": 0,
    "p12": 1,
    "p13": 1,
}
DECISIONS = ("T0", "T0-T1", "T0-T2")
BASELINES = (
    "simplified_baseline_f1_clinical_only",
    "simplified_baseline_f2_geometry_only",
    "simplified_baseline_f3_clinical_plus_geometry",
    "simplified_baseline_f4_timepoint_only",
    "simplified_baseline_f5_static_t0_imaging",
)


def _prediction_row(
    patient_id: str,
    fold: int,
    decision: str,
    condition: str,
    probability: float,
    *,
    donor_patient_id: object = pd.NA,
    repetition_id: object = pd.NA,
    matching_distance: float = np.nan,
) -> dict[str, object]:
    threshold = 0.5
    return {
        "patient_id": patient_id,
        "fold": fold,
        "decision_point": decision,
        "audit_condition": condition,
        "y_true": LABELS[patient_id],
        "predicted_probability": float(np.clip(probability, 0.001, 0.999)),
        "predicted_label": int(probability >= threshold),
        "threshold": threshold,
        "checkpoint": f"fold_{fold:02d}/best_corejepa.pt",
        "donor_patient_id": donor_patient_id,
        "repetition_id": repetition_id,
        "matching_distance": matching_distance,
    }


def synthetic_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    condition_shift = {
        "native": 0.0,
        "repeated_t0_c1": -0.02,
        "repeated_t0_c2": -0.07,
    }
    for fold, patients in PATIENTS_BY_FOLD.items():
        for patient_index, patient_id in enumerate(patients):
            native_base = (0.08, 0.42, 0.58, 0.92)[patient_index]
            for decision_index, decision in enumerate(DECISIONS):
                direction = 1 if LABELS[patient_id] else -1
                native_probability = native_base + direction * 0.015 * decision_index
                for condition, magnitude in condition_shift.items():
                    rows.append(
                        _prediction_row(
                            patient_id,
                            fold,
                            decision,
                            condition,
                            native_probability + direction * magnitude,
                        )
                    )
                for baseline_index, condition in enumerate(BASELINES):
                    shrink = 0.12 + 0.025 * baseline_index
                    baseline_probability = 0.5 + (native_probability - 0.5) * (1.0 - shrink)
                    rows.append(
                        _prediction_row(
                            patient_id,
                            fold,
                            decision,
                            condition,
                            baseline_probability,
                        )
                    )

            # Temporal order audit is defined only at T0-T2.
            native_t2 = native_base + (1 if LABELS[patient_id] else -1) * 0.03
            rows.append(
                _prediction_row(
                    patient_id,
                    fold,
                    "T0-T2",
                    "temporal_order_swap",
                    0.5 + (native_t2 - 0.5) * 0.72,
                )
            )

            # Two valid held-out-fold donors per recipient, shared across decisions.
            donor_one = patients[(patient_index + 1) % len(patients)]
            donor_two = patients[(patient_index + 2) % len(patients)]
            for repetition, donor in ((1, donor_one), (2, donor_two)):
                for decision_index, decision in enumerate(DECISIONS):
                    direction = 1 if LABELS[patient_id] else -1
                    native_probability = native_base + direction * 0.015 * decision_index
                    donor_probability = 0.5 + (native_probability - 0.5) * (
                        0.72 - 0.05 * repetition
                    )
                    rows.append(
                        _prediction_row(
                            patient_id,
                            fold,
                            decision,
                            "matched_followup_swap",
                            donor_probability,
                            donor_patient_id=donor,
                            repetition_id=repetition,
                            matching_distance=0.1 * repetition + 0.01 * patient_index,
                        )
                    )
    return pd.DataFrame(rows)


def synthetic_copy_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    epsilon = 1e-12
    for fold, patients in PATIENTS_BY_FOLD.items():
        for patient_index, patient_id in enumerate(patients):
            for transition_index, transition in enumerate(COPY_TRANSITIONS):
                copied = 0.45 + 0.03 * transition_index + 0.01 * patient_index
                learned = copied * (0.72 + 0.02 * fold)
                gain = (copied - learned) / (copied + epsilon)
                rows.append(
                    {
                        "patient_id": patient_id,
                        "fold": fold,
                        "transition": transition,
                        "learned_layer_norm_mse": learned,
                        "copy_layer_norm_mse": copied,
                        "learned_raw_mse": learned * 1.5,
                        "copy_raw_mse": copied * 1.5,
                        "learned_cosine_similarity": 0.7,
                        "copy_cosine_similarity": 0.6,
                        "learned_cosine_error": 0.3,
                        "copy_cosine_error": 0.4,
                        "normalized_transition_gain": gain,
                        "percentage_improvement": 100.0 * gain,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_paired_latent() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, patients in PATIENTS_BY_FOLD.items():
        for patient_index, patient_id in enumerate(patients):
            for condition in ("repeated_t0_c1", "repeated_t0_c2", "temporal_order_swap"):
                for transition_index, transition in enumerate(COPY_TRANSITIONS):
                    native = 0.25 + 0.01 * patient_index
                    change = 0.0 if condition == "repeated_t0_c1" else 0.03 + 0.01 * transition_index
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "fold": fold,
                            "transition": transition,
                            "audit_condition": condition,
                            "native_layer_norm_mse": native,
                            "perturbed_layer_norm_mse": native + change,
                            "latent_error_change": change,
                            "native_cosine_similarity": 0.8,
                            "perturbed_cosine_similarity": 0.8 - change,
                            "response_state_mean_abs_change": change / 2,
                            "response_state_l2_change": change,
                            "response_state_cosine_similarity": 1.0 - change,
                        }
                    )
    return pd.DataFrame(rows)


def synthetic_donor_latent() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, patients in PATIENTS_BY_FOLD.items():
        for patient_index, patient_id in enumerate(patients):
            for repetition in (1, 2):
                donor = patients[(patient_index + repetition) % len(patients)]
                for transition_index, transition in enumerate(COPY_TRANSITIONS):
                    native = 0.26 + 0.01 * patient_index
                    change = 0.02 * repetition + 0.005 * transition_index
                    rows.append(
                        {
                            "recipient_patient_id": patient_id,
                            "donor_patient_id": donor,
                            "fold": fold,
                            "audit_repetition": repetition,
                            "matching_distance": 0.1 * repetition,
                            "transition": transition,
                            "native_layer_norm_mse": native,
                            "donor_layer_norm_mse": native + change,
                            "latent_error_change": change,
                            "native_cosine_similarity": 0.8,
                            "donor_cosine_similarity": 0.8 - change,
                            "response_state_mean_abs_change": change / 2,
                            "response_state_l2_change": change,
                        }
                    )
    return pd.DataFrame(rows)


class PredictionAlignmentTests(unittest.TestCase):
    def test_validates_oof_native_alignment_and_donor_coverage(self) -> None:
        result = validate_oof_predictions(
            synthetic_predictions(), expected_folds=EXPECTED_FOLDS
        )
        self.assertEqual(result.expected_folds, EXPECTED_FOLDS)
        donor = result.coverage.loc[
            result.coverage["audit_condition"].eq("matched_followup_swap")
        ]
        self.assertTrue(donor["is_donor_condition"].all())
        self.assertTrue(donor["complete_native_coverage"].all())
        self.assertTrue(donor["n_repetitions"].eq(2).all())
        self.assertEqual(
            result.frame.groupby("patient_id")["fold"].nunique().max(), 1
        )

    def test_rejects_label_fold_and_non_donor_coverage_drift(self) -> None:
        source = synthetic_predictions()
        bad_label = source.copy()
        index = bad_label.index[
            bad_label["audit_condition"].eq("repeated_t0_c1")
        ][0]
        bad_label.loc[index, "y_true"] = 1 - int(bad_label.loc[index, "y_true"])
        with self.assertRaisesRegex(ValueError, "label"):
            validate_oof_predictions(bad_label, expected_folds=EXPECTED_FOLDS)

        bad_fold = source.copy()
        index = bad_fold.index[
            bad_fold["audit_condition"].eq("repeated_t0_c1")
        ][0]
        bad_fold.loc[index, "fold"] = 1 - int(bad_fold.loc[index, "fold"])
        with self.assertRaisesRegex(ValueError, "\u591a\u6298"):
            validate_oof_predictions(bad_fold, expected_folds=EXPECTED_FOLDS)

        missing = source.drop(
            source.index[
                source["audit_condition"].eq("repeated_t0_c2")
                & source["patient_id"].eq("p00")
            ]
        )
        with self.assertRaisesRegex(ValueError, "\u672a\u5b8c\u6574\u8986\u76d6"):
            validate_oof_predictions(missing, expected_folds=EXPECTED_FOLDS)
        allowed = validate_oof_predictions(
            missing,
            expected_folds=EXPECTED_FOLDS,
            allow_incomplete_conditions=["repeated_t0_c2"],
        )
        coverage = allowed.coverage.loc[
            allowed.coverage["audit_condition"].eq("repeated_t0_c2"),
            "coverage_fraction",
        ]
        self.assertTrue(coverage.lt(1.0).all())

    def test_rejects_cross_fold_donor_and_changed_readout_provenance(self) -> None:
        source = synthetic_predictions()
        cross_fold = source.copy()
        index = cross_fold.index[
            cross_fold["audit_condition"].eq("matched_followup_swap")
            & cross_fold["patient_id"].eq("p00")
        ][0]
        cross_fold.loc[index, "donor_patient_id"] = "p10"
        # Donor provenance must be constant across decisions; update the entire pair
        # so the more important held-out-fold isolation check is exercised.
        repetition = cross_fold.loc[index, "repetition_id"]
        mask = (
            cross_fold["audit_condition"].eq("matched_followup_swap")
            & cross_fold["patient_id"].eq("p00")
            & cross_fold["repetition_id"].eq(repetition)
        )
        cross_fold.loc[mask, "donor_patient_id"] = "p10"
        with self.assertRaisesRegex(ValueError, "\u540c\u4e00 held-out fold"):
            validate_oof_predictions(cross_fold, expected_folds=EXPECTED_FOLDS)

        changed_threshold = source.copy()
        index = changed_threshold.index[
            changed_threshold["audit_condition"].eq("repeated_t0_c1")
            & changed_threshold["patient_id"].eq("p00")
        ][0]
        changed_threshold.loc[index, "threshold"] = 0.4
        changed_threshold.loc[index, "predicted_label"] = int(
            changed_threshold.loc[index, "predicted_probability"] >= 0.4
        )
        with self.assertRaisesRegex(ValueError, "threshold"):
            validate_oof_predictions(changed_threshold, expected_folds=EXPECTED_FOLDS)


class PredictionSummaryTests(unittest.TestCase):
    def test_builds_fold_pooled_difference_and_paired_bootstrap(self) -> None:
        report = build_prediction_report(
            synthetic_predictions(),
            expected_folds=EXPECTED_FOLDS,
            n_bootstrap=60,
            seed=17,
        )
        self.assertEqual(set(report.fold_metrics["fold"]), set(EXPECTED_FOLDS))
        fold_summary = report.fold_summary.loc[
            report.fold_summary["audit_condition"].eq("native")
            & report.fold_summary["decision_point"].eq("T0")
            & report.fold_summary["metric"].eq("auroc")
        ].iloc[0]
        expected = report.fold_metrics.loc[
            report.fold_metrics["audit_condition"].eq("native")
            & report.fold_metrics["decision_point"].eq("T0"),
            "auroc",
        ].std(ddof=1)
        self.assertAlmostEqual(fold_summary["sample_std"], expected)

        difference = report.native_differences.loc[
            report.native_differences["audit_condition"].eq("repeated_t0_c2")
            & report.native_differences["decision_point"].eq("T0-T2")
            & report.native_differences["metric"].eq("auroc")
        ].iloc[0]
        self.assertEqual(difference["difference_direction"], "comparison - native")
        self.assertAlmostEqual(
            difference["absolute_difference"],
            difference["comparison"] - difference["native"],
        )
        self.assertEqual(set(report.paired_bootstrap["metric"]), {"auroc", "auprc"})
        self.assertTrue(report.paired_bootstrap["n_bootstrap"].eq(60).all())
        self.assertFalse(report.repetition_metrics.empty)
        self.assertEqual(
            set(report.repetition_metrics["aggregation_scope"]),
            {"fold", "pooled_oof"},
        )

    def test_bootstrap_is_deterministic_and_donor_is_patient_weighted(self) -> None:
        first = build_prediction_report(
            synthetic_predictions(),
            expected_folds=EXPECTED_FOLDS,
            n_bootstrap=30,
            seed=99,
        )
        second = build_prediction_report(
            synthetic_predictions(),
            expected_folds=EXPECTED_FOLDS,
            n_bootstrap=30,
            seed=99,
        )
        assert_frame_equal(first.paired_bootstrap, second.paired_bootstrap)
        assert_frame_equal(first.bootstrap_samples, second.bootstrap_samples)
        donor_patient = first.patient_predictions.loc[
            first.patient_predictions["audit_condition"].eq("matched_followup_swap")
        ]
        self.assertTrue(donor_patient["n_records"].eq(2).all())
        self.assertEqual(donor_patient["patient_id"].nunique(), len(LABELS))


class LatentReportingTests(unittest.TestCase):
    def test_copy_formula_validation_and_equal_patient_summary(self) -> None:
        source = synthetic_copy_metrics()
        validated = validate_copy_latent_metrics(
            source, expected_folds=EXPECTED_FOLDS
        )
        self.assertEqual(len(validated), len(LABELS) * len(COPY_TRANSITIONS))
        report = summarize_copy_latent_metrics(
            source, expected_folds=EXPECTED_FOLDS, n_bootstrap=40, seed=7
        )
        self.assertEqual(
            set(report.fold_metrics["transition_scope"]),
            {*COPY_TRANSITIONS, "ALL"},
        )
        self.assertEqual(
            report.patient_metrics.loc[
                report.patient_metrics["transition_scope"].eq("ALL")
            ]["patient_id"].nunique(),
            len(LABELS),
        )
        self.assertEqual(
            set(report.bootstrap_summary["transition_scope"]),
            {*COPY_TRANSITIONS, "ALL"},
        )
        self.assertIn(
            "copy_minus_learned_layer_norm_mse",
            set(report.bootstrap_summary["metric"]),
        )
        self.assertTrue(report.bootstrap_summary["n_bootstrap"].eq(40).all())
        self.assertTrue(report.bootstrap_summary["n_valid_bootstrap"].eq(40).all())
        self.assertEqual(
            len(report.bootstrap_samples),
            len(report.bootstrap_summary) * 40,
        )
        bad = source.copy()
        bad.loc[0, "normalized_transition_gain"] += 0.1
        with self.assertRaisesRegex(ValueError, "G \u516c\u5f0f"):
            validate_copy_latent_metrics(bad, expected_folds=EXPECTED_FOLDS)

    def test_unifies_paired_and_donor_latent_metrics(self) -> None:
        report = summarize_perturbation_latent_metrics(
            synthetic_paired_latent(),
            synthetic_donor_latent(),
            expected_folds=EXPECTED_FOLDS,
        )
        self.assertIsNotNone(report)
        assert report is not None
        self.assertIn("comparison_layer_norm_mse", report.records)
        self.assertEqual(
            set(report.records["audit_condition"]),
            {
                "repeated_t0_c1",
                "repeated_t0_c2",
                "temporal_order_swap",
                "matched_followup_swap",
            },
        )
        donor = report.patient_metrics.loc[
            report.patient_metrics["audit_condition"].eq("matched_followup_swap")
        ]
        self.assertTrue(donor["n_records"].eq(2).all())

    def test_bundle_validates_all_inputs_against_native(self) -> None:
        bundle = build_audit_reporting_bundle(
            synthetic_predictions(),
            copy_latent_metrics=synthetic_copy_metrics(),
            paired_perturbation_metrics=synthetic_paired_latent(),
            donor_metrics=synthetic_donor_latent(),
            expected_folds=EXPECTED_FOLDS,
            n_bootstrap=20,
        )
        self.assertIsInstance(bundle, AuditReportingBundle)
        self.assertIsNotNone(bundle.copy_latent)
        self.assertIsNotNone(bundle.perturbation_latent)


class FigureAndPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = build_audit_reporting_bundle(
            synthetic_predictions(),
            copy_latent_metrics=synthetic_copy_metrics(),
            paired_perturbation_metrics=synthetic_paired_latent(),
            donor_metrics=synthetic_donor_latent(),
            expected_folds=EXPECTED_FOLDS,
            n_bootstrap=10,
        )

    def test_plotting_preflight_and_role_inference(self) -> None:
        status = plotting_backend_status()
        self.assertTrue(status["ready"])
        roles = infer_figure_condition_roles(
            self.bundle.prediction.predictions["audit_condition"].unique()
        )
        self.assertEqual(len(roles.repeated_t0), 2)
        self.assertEqual(len(roles.temporal_order), 1)
        self.assertEqual(len(roles.followup_swap), 1)
        self.assertEqual(len(roles.simplified_baselines), 5)

    def test_generates_exactly_eight_atomic_non_overwriting_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = generate_required_figures(
                self.bundle, directory, dpi=72
            )
            self.assertEqual(len(result.artifacts), 8)
            self.assertTrue(result.manifest_path.is_file())
            with result.manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(len(manifest["artifacts"]), 8)
            self.assertEqual(
                manifest["difference_direction"], "comparison - native"
            )
            for artifact in manifest["artifacts"]:
                path = Path(artifact["path"])
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
                self.assertEqual(len(artifact["sha256"]), 64)
                self.assertTrue(artifact["error_bar"])
                self.assertTrue(artifact["native_reference"])
            with self.assertRaises(FileExistsError):
                generate_required_figures(self.bundle, directory, dpi=72)

    def test_saves_tables_with_hash_manifest_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = save_reporting_tables(self.bundle, directory)
            self.assertTrue(manifest_path.is_file())
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            names = {artifact["table"] for artifact in manifest["artifacts"]}
            self.assertIn("fold_metrics", names)
            self.assertIn("pooled_oof", names)
            self.assertIn("copy_fold_summary", names)
            self.assertIn("copy_bootstrap", names)
            self.assertIn("copy_bootstrap_samples", names)
            for artifact in manifest["artifacts"]:
                self.assertTrue(Path(artifact["path"]).is_file())
                self.assertEqual(len(artifact["sha256"]), 64)
            with self.assertRaises(FileExistsError):
                save_reporting_tables(self.bundle, directory)

    def test_one_click_pipeline_and_public_contracts(self) -> None:
        self.assertEqual(
            REPORTING_INPUT_CONTRACTS["predictions"][0], "patient_id"
        )
        self.assertEqual(len(REQUIRED_FIGURE_FILENAMES), 8)
        with tempfile.TemporaryDirectory() as directory:
            result = run_reporting_pipeline(
                synthetic_predictions(),
                directory,
                copy_latent_metrics=synthetic_copy_metrics(),
                paired_perturbation_metrics=synthetic_paired_latent(),
                donor_metrics=synthetic_donor_latent(),
                expected_folds=EXPECTED_FOLDS,
                n_bootstrap=10,
                dpi=72,
            )
            self.assertTrue(result.table_manifest_path.is_file())
            self.assertTrue(result.figures.manifest_path.is_file())
            self.assertEqual(
                {path.name for path in Path(directory, "figures").glob("*.png")},
                set(REQUIRED_FIGURE_FILENAMES),
            )


if __name__ == "__main__":
    unittest.main()
