from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FORMAL_STAGE_B_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
)
MODEL_READY_SRC = (
    REPO_ROOT / "additional_experiments" / "c1b_model_ready_ftv_sanity" / "src"
)
sys.path[:0] = [str(FORMAL_STAGE_B_SRC), str(MODEL_READY_SRC), str(ROOT / "src")]

from c1b_spatial_audit.analysis import (  # noqa: E402
    AggregationResult,
    FINAL_CORE_POOLINGS,
    FTV_TABLE_COLUMNS,
    NUISANCE_TARGETS,
    PRIVATE_JOINED_COLUMNS,
    ProbeStageData,
    TABLE1_COLUMNS,
    TABLE4_COLUMNS,
    TABLE5_COLUMNS,
    TABLE6_COLUMNS,
    _expected_cells,
    build_feature_contract_table,
    build_ftv_table,
    build_private_diagnostic_join,
    build_recovery_table,
    build_table6,
    deployable_local_gate,
    load_audit_config,
    padding_geometry_gate,
    pair_frozen_p0_errors,
    pooled_stage_natural_metrics,
    strong_oracle_gate,
    undertraining_gate,
    unique_classification,
    write_aggregation_outputs,
)
from c1b_spatial_audit.contracts import ARMS, FOLDS, POOLINGS, SEEDS, TIMEPOINTS  # noqa: E402
from c1b_spatial_audit.sidecars import (  # noqa: E402
    NUISANCE_COLUMNS,
    OCCUPANCY_COLUMNS,
    assign_occupancy_quartiles,
)


CONFIG = load_audit_config(ROOT / "configs/audit.json")


def synthetic_ftv_metric_inputs():
    natural_rows = []
    transformed_rows = []
    for seed in SEEDS:
        for arm in ARMS:
            for pooling in FINAL_CORE_POOLINGS[arm]:
                feature_dim = 384 if pooling == "PLOCAL+GLOBAL" else 192
                for endpoint_index, endpoint in enumerate((*TIMEPOINTS, "macro")):
                    common = {
                        "stage": "final",
                        "seed_base": seed,
                        "arm": arm,
                        "pooling": pooling,
                        "feature_dim": feature_dim,
                        "task": "static",
                        "target_name": "FTV",
                        "endpoint": endpoint,
                        "analysis_scope": "primary_measurement_valid",
                        "target_semantics": (
                            "static_ftv_log_winsor_median_iqr_inverse_natural"
                        ),
                    }
                    natural_rows.append(
                        {
                            **common,
                            "scale": "natural",
                            "aggregation": "pooled_outer_test_folds",
                            "n_test": 100,
                            "spearman": 0.3 + endpoint_index / 100,
                            "pearson": 0.25,
                            "r2": 0.1,
                            "rmse": 2.0,
                            "mae": 1.0,
                            "b0_rmse": 2.5,
                            "rmse_gain_over_b0": 0.2,
                            "prediction_target_variance_ratio": 0.8,
                            "calibration_slope": 0.7,
                            "calibration_intercept": 0.1,
                            "calibration_mean_bias": 0.05,
                        }
                    )
                    transformed_rows.append(
                        {
                            **common,
                            "scale": "transformed_outer_train",
                            "transformed_fold_count": 5,
                            "transformed_spearman_fold_mean": 0.2,
                            "transformed_spearman_fold_sd": 0.01,
                            "transformed_r2_fold_mean": 0.15,
                            "transformed_r2_fold_sd": 0.02,
                            "transformed_rmse_fold_mean": 0.9,
                            "transformed_mae_fold_mean": 0.7,
                        }
                    )
    data = ProbeStageData(
        stage="final",
        predictions=pd.DataFrame(),
        metrics=pd.DataFrame(),
        selections=pd.DataFrame(),
        identities=pd.DataFrame(),
        secondary_present=False,
    )
    return data, pd.DataFrame(natural_rows), pd.DataFrame(transformed_rows)


def synthetic_recovery_input(*, nonpositive_deficit: bool = False) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for legacy_arm, new_arm in (("L1", "N1"), ("L3", "N3")):
            legacy = 0.3 if nonpositive_deficit and new_arm == "N1" else 0.6
            new_p0 = 0.4
            values = {
                "P0": new_p0,
                "PVALID": new_p0 + 0.11,
                "PLOCAL": new_p0 + 0.11,
                "PLOCAL+GLOBAL": new_p0 + 0.12,
                "PORACLE": new_p0 + 0.16,
            }
            rows.append(
                {
                    "stage": "final",
                    "seed_base": seed,
                    "arm": legacy_arm,
                    "pooling": "P0",
                    "endpoint": "macro",
                    "availability": "AVAILABLE",
                    "spearman": legacy,
                }
            )
            rows.extend(
                {
                    "stage": "final",
                    "seed_base": seed,
                    "arm": new_arm,
                    "pooling": pooling,
                    "endpoint": "macro",
                    "availability": "AVAILABLE",
                    "spearman": value,
                }
                for pooling, value in values.items()
            )
    return pd.DataFrame(rows)


def synthetic_padding_table() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for target in ("padding_fraction", "valid_source_fraction"):
            p0 = 0.30 if target == "padding_fraction" else 0.10
            pvalid = 0.20 if target == "padding_fraction" else 0.05
            for pooling, value in (("P0", p0), ("PVALID", pvalid)):
                rows.append(
                    {
                        "stage": "final",
                        "seed_base": seed,
                        "arm": "N1",
                        "pooling": pooling,
                        "target_name": target,
                        "endpoint": "macro",
                        "availability": "AVAILABLE",
                        "natural_r2": value,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_nuisance() -> pd.DataFrame:
    rows = []
    formal = [f"P{index:03d}" for index in range(375)]
    extras = [f"X{index:03d}" for index in range(808 - len(formal))]
    for patient_index, patient in enumerate((*formal, *extras)):
        for visit_index, visit in enumerate(TIMEPOINTS):
            factor = (1.2, 1.7, 2.3)[(patient_index + visit_index) % 3]
            row = {
                "patient_id": patient,
                "visit": visit,
                "padding_fraction": 0.2,
                "valid_source_fraction": 0.8,
                "native_spacing_x_mm": 0.7,
                "native_spacing_y_mm": 0.8,
                "native_spacing_z_mm": 2.0,
                "acquisition_fov_x_mm": 100.0,
                "acquisition_fov_y_mm": 110.0,
                "acquisition_fov_z_mm": 160.0,
                "max_resample_factor": factor,
                "resize_anisotropy": factor,
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=NUISANCE_COLUMNS)


def synthetic_occupancy() -> pd.DataFrame:
    rows = []
    counter = 0
    for patient_index in range(375):
        patient = f"P{patient_index:03d}"
        for visit in TIMEPOINTS:
            counter += 1
            occupancy = counter / 2000.0
            rows.append(
                {
                    "patient_id": patient,
                    "visit": visit,
                    "support_source_positive_voxels": 10,
                    "support_retained_positive_voxels": 10,
                    "support_nn_target_positive_voxels": 8,
                    "support_source_volume_mm3": 16.2,
                    "support_retained_source_volume_mm3": 16.2,
                    "valid_source_voxels": 1000,
                    "valid_source_volume_mm3": 1620.0,
                    "lesion_occupancy": occupancy,
                }
            )
    frame = assign_occupancy_quartiles(pd.DataFrame(rows))
    return frame.loc[:, OCCUPANCY_COLUMNS]


def synthetic_paired() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for patient_index in range(375):
            patient = f"P{patient_index:03d}"
            fold = patient_index % 5
            for visit_index, endpoint in enumerate(TIMEPOINTS):
                truth = float(patient_index + visit_index + 1)
                l1_prediction = truth + ((patient_index % 7) - 3) * 0.1
                n1_prediction = truth + ((patient_index % 9) - 4) * 0.15
                l1_error = abs(l1_prediction - truth)
                n1_error = abs(n1_prediction - truth)
                rows.append(
                    {
                        "seed_base": seed,
                        "fold": fold,
                        "endpoint": endpoint,
                        "patient_id": patient,
                        "y_true": truth,
                        "l1_prediction": l1_prediction,
                        "n1_prediction": n1_prediction,
                        "l1_abs_error": l1_error,
                        "n1_abs_error": n1_error,
                        "paired_abs_error_difference": n1_error - l1_error,
                    }
                )
    return pd.DataFrame(rows)


class NaturalMetricAndMatrixTests(unittest.TestCase):
    def test_natural_metrics_pool_all_five_folds_before_macro(self) -> None:
        rows = []
        for fold in FOLDS:
            for endpoint_index, endpoint in enumerate(TIMEPOINTS):
                rows.append(
                    {
                        "patient_id": f"P{fold}_{endpoint}",
                        "split": "test",
                        "arm": "N1",
                        "seed_base": 2026,
                        "fold": fold,
                        "pooling": "P0",
                        "feature_dim": 192,
                        "task": "static",
                        "target_name": "FTV",
                        "endpoint": endpoint,
                        "analysis_scope": "primary_measurement_valid",
                        "target_semantics": "synthetic",
                        "y_true": float(fold + endpoint_index),
                        "y_pred": float(fold + endpoint_index) + 0.1,
                        "b0_prediction": 0.0,
                        "test_predict_call_count": 1,
                    }
                )
        data = ProbeStageData(
            stage="final",
            predictions=pd.DataFrame(rows),
            metrics=pd.DataFrame(),
            selections=pd.DataFrame(),
            identities=pd.DataFrame(),
            secondary_present=False,
        )
        result = pooled_stage_natural_metrics(data)
        self.assertEqual(set(result["endpoint"]), {*TIMEPOINTS, "macro"})
        self.assertTrue(result["aggregation"].str.contains("pooled").all())
        duplicate = data.predictions.copy()
        duplicate.loc[duplicate["fold"].eq(1), "patient_id"] = duplicate.loc[
            duplicate["fold"].eq(0), "patient_id"
        ].to_numpy()
        with self.assertRaisesRegex(ValueError, "multiple outer-test folds"):
            pooled_stage_natural_metrics(
                ProbeStageData(
                    stage="final",
                    predictions=duplicate,
                    metrics=pd.DataFrame(),
                    selections=pd.DataFrame(),
                    identities=pd.DataFrame(),
                    secondary_present=False,
                )
            )

    def test_frozen_probe_identity_counts_are_exact(self) -> None:
        self.assertEqual(len(_expected_cells("final")), 160)
        self.assertEqual(len(_expected_cells("s3")), 100)


class RectangularTableTests(unittest.TestCase):
    def test_static_table_preserves_exact_legacy_na_rows(self) -> None:
        data, natural, transformed = synthetic_ftv_metric_inputs()
        result = build_ftv_table(
            data, natural, transformed, task="static", config=CONFIG
        )
        self.assertEqual(tuple(result.columns), FTV_TABLE_COLUMNS)
        self.assertEqual(len(result), 200)
        unavailable = result.loc[result["availability"].eq("NA")]
        self.assertEqual(len(unavailable), 40)
        self.assertEqual(
            set(unavailable.loc[unavailable["pooling"].eq("PVALID"), "status_reason"]),
            {CONFIG["legacy_pvalid"]},
        )
        self.assertEqual(
            set(unavailable.loc[unavailable["pooling"].eq("PORACLE"), "status_reason"]),
            {CONFIG["legacy_poracle"]},
        )
        self.assertTrue(unavailable["spearman"].isna().all())


class GateTests(unittest.TestCase):
    def test_recovery_thresholds_and_nonpositive_deficit_are_fail_closed(self) -> None:
        recovery = build_recovery_table(synthetic_recovery_input())
        self.assertTrue(strong_oracle_gate(recovery, stage="final", config=CONFIG)["supported"])
        self.assertTrue(deployable_local_gate(recovery, stage="final", config=CONFIG)["supported"])
        self.assertTrue(
            padding_geometry_gate(
                recovery, synthetic_padding_table(), config=CONFIG
            )["supported"]
        )

        nonpositive = build_recovery_table(
            synthetic_recovery_input(nonpositive_deficit=True)
        )
        row = nonpositive.loc[
            nonpositive["new_arm"].eq("N1")
            & nonpositive["pooling"].eq("PORACLE")
        ].iloc[0]
        self.assertFalse(bool(row["recovery_defined"]))
        self.assertTrue(np.isnan(row["recovery_ratio"]))
        self.assertFalse(
            strong_oracle_gate(nonpositive, stage="final", config=CONFIG)["supported"]
        )

    def test_unique_classification_uses_frozen_hierarchy_and_unique_next(self) -> None:
        c = unique_classification(
            final_oracle_strong=False,
            s3_oracle_strong=False,
            padding_geometry_supported=True,
            deployable_local_supported=True,
        )
        self.assertEqual(c["code"], "C")
        b = unique_classification(
            final_oracle_strong=True,
            s3_oracle_strong=None,
            padding_geometry_supported=True,
            deployable_local_supported=True,
        )
        self.assertEqual(b["code"], "B")
        a = unique_classification(
            final_oracle_strong=True,
            s3_oracle_strong=None,
            padding_geometry_supported=False,
            deployable_local_supported=True,
        )
        self.assertEqual(a["code"], "A")
        d_final = unique_classification(
            final_oracle_strong=True,
            s3_oracle_strong=None,
            padding_geometry_supported=False,
            deployable_local_supported=False,
        )
        self.assertEqual(d_final["next"], "Learned Spatial Response Aggregation Pilot")
        d_s3 = unique_classification(
            final_oracle_strong=False,
            s3_oracle_strong=True,
            padding_geometry_supported=False,
            deployable_local_supported=False,
        )
        self.assertEqual(d_s3["next"], "Preserve Higher-Resolution Spatial Features Pilot")
        with self.assertRaisesRegex(ValueError, "requires completed"):
            unique_classification(
                final_oracle_strong=False,
                s3_oracle_strong=None,
                padding_geometry_supported=False,
                deployable_local_supported=False,
            )

    def test_undertraining_is_recomputed_from_exact_40_cells(self) -> None:
        rows = [
            {
                "seed": seed,
                "arm": arm,
                "fold": fold,
                "configured_max_epoch": 12,
                "hit_configured_max_epoch": False,
                "selected_in_last_two_observed_epochs": False,
                "last_three_normalized_validation_state_slope": 0.2,
            }
            for seed in SEEDS
            for arm in ARMS
            for fold in FOLDS
        ]
        summary = {
            "schema_version": 1,
            "status": "COMPLETE",
            "new_training_performed": False,
            "undertraining_thresholds": CONFIG["undertraining"],
            "undertraining_plausible": {"N1": False, "N3": False},
            "any_n_arm_undertraining_plausible": False,
            "arm_summary": {
                arm: {
                    "cells": 10,
                    "hit_configured_max_rate": 0.0,
                    "selected_last_two_rate": 0.0,
                    "median_last_three_normalized_slope": 0.2,
                }
                for arm in ("N1", "N3")
            },
        }
        result = undertraining_gate(pd.DataFrame(rows), summary, config=CONFIG)
        self.assertFalse(result["any_n_arm_undertraining_plausible"])
        with self.assertRaisesRegex(ValueError, "40-cell"):
            undertraining_gate(pd.DataFrame(rows[:-1]), summary, config=CONFIG)
        malformed = pd.DataFrame(rows).astype(
            {"hit_configured_max_epoch": "object"}
        )
        malformed.loc[0, "hit_configured_max_epoch"] = "not-a-boolean"
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            undertraining_gate(malformed, summary, config=CONFIG)


class FrozenDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nuisance = synthetic_nuisance()
        cls.occupancy = synthetic_occupancy()
        cls.paired = synthetic_paired()

    def test_table1_reports_native_legacy_and_fixed_c1b_jump(self) -> None:
        table = build_feature_contract_table(self.nuisance, CONFIG)
        self.assertEqual(len(table), 4)
        final_c1b = table.loc[
            table["stage"].eq("final") & table["input_contract"].eq("c1b")
        ].iloc[0]
        self.assertEqual(final_c1b["feature_shape_zyx"], "14x22x20")
        self.assertAlmostEqual(final_c1b["jump_x_mm_median"], 7.2)
        self.assertAlmostEqual(final_c1b["jump_z_mm_median"], 16.0)
        self.assertNotIn("patient_id", table.columns)

    def test_occupancy_and_downsampling_join_old_p0_without_refit(self) -> None:
        joined = build_private_diagnostic_join(
            self.paired,
            self.occupancy,
            self.nuisance,
            downsampling_bins=CONFIG["downsampling_bins"],
        )
        self.assertEqual(tuple(joined.columns), PRIVATE_JOINED_COLUMNS)
        self.assertEqual(len(joined), 2 * 375 * 4)
        table6 = build_table6(joined, downsampling_bins=CONFIG["downsampling_bins"])
        self.assertEqual(len(table6), 72)
        self.assertNotIn("patient_id", table6.columns)
        self.assertEqual(
            set(table6.loc[table6["analysis"].eq("downsampling_bin"), "stratum"]),
            set(CONFIG["downsampling_bins"]),
        )
        invalid = self.nuisance.copy()
        invalid.loc[invalid["patient_id"].eq("P000"), "max_resample_factor"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            build_private_diagnostic_join(
                self.paired,
                self.occupancy,
                invalid,
                downsampling_bins=CONFIG["downsampling_bins"],
            )

    def test_pairing_rejects_unmatched_arm_population(self) -> None:
        base = pd.DataFrame(
            [
                {
                    "seed_base": 2026,
                    "fold": 0,
                    "endpoint": "T0",
                    "patient_id": "P0",
                    "arm": arm,
                    "y_true": 1.0,
                    "y_pred": value,
                }
                for arm, value in (("L1", 0.9), ("N1", 1.1))
            ]
        )
        paired = pair_frozen_p0_errors(base)
        self.assertEqual(len(paired), 1)
        with self.assertRaisesRegex(ValueError, "do not pair"):
            pair_frozen_p0_errors(base.loc[base["arm"].eq("L1")])


class OutputContractTests(unittest.TestCase):
    def test_public_0644_private_0600_atomic_nonoverwrite(self) -> None:
        table1 = pd.DataFrame(
            [
                {
                    **{column: np.nan for column in TABLE1_COLUMNS},
                    "stage": stage,
                    "input_contract": contract,
                }
                for stage in ("final", "s3")
                for contract in ("legacy", "c1b")
            ],
            columns=TABLE1_COLUMNS,
        )
        private = pd.DataFrame(
            [
                {
                    column: (
                        "PRIVATE-ID"
                        if column == "patient_id"
                        else "T0"
                        if column == "endpoint"
                        else "Q1"
                        if column in {"occupancy_quartile", "downsampling_bin"}
                        else 1
                    )
                    for column in PRIVATE_JOINED_COLUMNS
                }
            ],
            columns=PRIVATE_JOINED_COLUMNS,
        )
        result = AggregationResult(
            table1=table1,
            table2=pd.DataFrame(columns=FTV_TABLE_COLUMNS),
            table3=pd.DataFrame(columns=FTV_TABLE_COLUMNS),
            table4=pd.DataFrame(columns=TABLE4_COLUMNS),
            table5=pd.DataFrame(columns=TABLE5_COLUMNS),
            table6=pd.DataFrame(columns=TABLE6_COLUMNS),
            gates={
                "schema_version": 1,
                "status": "COMPLETE",
                "natural_metrics": "pooled_five_outer_test_folds_before_metric",
                "transformed_metrics": "outer_fold_summaries_only",
                "new_training_performed": False,
                "probe_refit_during_aggregation": False,
                "final_stage": {},
                "conditional_s3": {
                    "trigger_status": "NOT_TRIGGERED_FINAL_ORACLE_STRONG"
                },
                "training_budget": {},
                "classification": {
                    "code": "A",
                    "classification": "A POOLING BOTTLENECK",
                    "next": "Local–Global Response State Pilot",
                },
            },
            private_joined=private,
        )
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_aggregation_outputs(result, output_dir=directory)
            for name, path in outputs.items():
                expected = 0o600 if name == "private_joined" else 0o644
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected)
            self.assertEqual(len(outputs), 8)
            with self.assertRaises(FileExistsError):
                write_aggregation_outputs(result, output_dir=directory)

    def test_aggregation_source_contains_no_fit_or_refit_path(self) -> None:
        source = (ROOT / "src/c1b_spatial_audit/analysis.py").read_text(encoding="utf-8")
        self.assertIn("pooled_oof_natural_metrics", source)
        self.assertNotIn("select_ridge", source)
        self.assertNotIn(".fit(", source)
        cli = (ROOT / "scripts/aggregate_results.py").read_text(encoding="utf-8")
        self.assertIn("--execute", cli)


if __name__ == "__main__":
    unittest.main()
