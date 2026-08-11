from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


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
    FTV_TABLE_COLUMNS,
    NUISANCE_TARGETS,
    TABLE1_COLUMNS,
    TABLE4_COLUMNS,
    TABLE5_COLUMNS,
    TABLE6_COLUMNS,
)
from c1b_spatial_audit.contracts import ARMS, FOLDS, SEEDS, TIMEPOINTS, TRANSITIONS  # noqa: E402
from c1b_spatial_audit.figures import (  # noqa: E402
    ACTIVATION_AGGREGATE_KEYS,
    ActivationAggregate,
    FIGURE_DPI,
    FIGURE_FILENAMES,
    FigureInputPaths,
    TABLE7_COLUMNS,
    aggregate_normalized_abs_activations,
    render_public_figures,
    select_patient_ids_by_sha256,
    validate_activation_aggregate,
    write_private_activation_aggregate,
)
from c1b_spatial_audit.sidecars import SIDECAR_KEYS  # noqa: E402
import c1b_spatial_audit.figures as figures_module  # noqa: E402


FINAL_POOLINGS = ("P0", "PVALID", "PLOCAL", "PLOCAL+GLOBAL", "PORACLE")
S3_POOLINGS = ("P0", "PLOCAL", "PORACLE")
NUISANCE_POOLINGS = ("P0", "PVALID", "PLOCAL")
LOCKED_ACTIVATION_SHA256 = "c" * 64
PRIVATE_CHECKPOINT_SENTINEL = Path("/private/formal/checkpoints/selected.pt")
REAL_FIXED_ACTIVATION_BINDING = figures_module._fixed_activation_checkpoint_binding


def synthetic_table1() -> pd.DataFrame:
    rows = []
    specifications = {
        ("final", "legacy"): ("32x96x96", 128, "4x12x12", 8, 47),
        ("final", "c1b"): ("112x176x160", 128, "14x22x20", 8, 47),
        ("s3", "legacy"): ("32x96x96", 64, "8x24x24", 4, 23),
        ("s3", "c1b"): ("112x176x160", 64, "28x44x40", 4, 23),
    }
    for (stage, contract), (input_shape, channels, feature_shape, jump, rf) in specifications.items():
        row = {
            "stage": stage,
            "analysis_role": "primary" if stage == "final" else "conditional_secondary",
            "input_contract": contract,
            "input_shape_zyx": input_shape,
            "feature_channels": channels,
            "feature_shape_zyx": feature_shape,
            "jump_input_voxels": jump,
            "center_offset_input_voxels": 0,
            "theoretical_receptive_field_input_voxels": rf,
            "local_window_mm_xyz": "64x64x64",
            "spacing_basis": "synthetic_public_aggregate",
        }
        for axis in "xyz":
            row[f"jump_{axis}_mm_median"] = float(jump)
            row[f"jump_{axis}_mm_q25"] = float(jump) - 0.5
            row[f"jump_{axis}_mm_q75"] = float(jump) + 0.5
        rows.append(row)
    return pd.DataFrame(rows, columns=TABLE1_COLUMNS)


def synthetic_ftv_table(
    *,
    static: bool,
    include_s3: bool = True,
    include_secondary: bool = True,
) -> pd.DataFrame:
    endpoints = (*TIMEPOINTS, "macro") if static else (*TRANSITIONS, "macro")
    rows = []
    stage_poolings = {"final": FINAL_POOLINGS}
    if include_s3:
        stage_poolings["s3"] = S3_POOLINGS
    for stage, poolings in stage_poolings.items():
        for seed_index, seed in enumerate(SEEDS):
            for arm_index, arm in enumerate(ARMS):
                arm_poolings = tuple(poolings)
                if stage == "final" and include_secondary and arm in {"N1", "N3"}:
                    arm_poolings = (*arm_poolings, "PLOCAL+PVALID_SECONDARY")
                for pooling_index, pooling in enumerate(arm_poolings):
                    for endpoint_index, endpoint in enumerate(endpoints):
                        unavailable = arm in {"L1", "L3"} and (
                            (stage == "final" and pooling in {"PVALID", "PORACLE"})
                            or (stage == "s3" and pooling == "PORACLE")
                        )
                        value = (
                            0.15
                            + 0.03 * arm_index
                            + 0.025 * pooling_index
                            + 0.005 * seed_index
                            + 0.002 * endpoint_index
                        )
                        feature_dim = (
                            64
                            if stage == "s3"
                            else 384
                            if pooling in {"PLOCAL+GLOBAL", "PLOCAL+PVALID_SECONDARY"}
                            else 192
                        )
                        row = {
                            "stage": stage,
                            "analysis_role": (
                                "secondary_sensitivity"
                                if pooling == "PLOCAL+PVALID_SECONDARY"
                                else "primary"
                                if stage == "final"
                                else "conditional_s3"
                            ),
                            "seed_base": seed,
                            "arm": arm,
                            "pooling": pooling,
                            "endpoint": endpoint,
                            "analysis_scope": "primary_measurement_valid",
                            "availability": "NA" if unavailable else "AVAILABLE",
                            "status_reason": "synthetic_NA" if unavailable else "",
                            "feature_dim": np.nan if unavailable else feature_dim,
                            "aggregation": (
                                np.nan
                                if unavailable
                                else "mean_of_pooled_endpoint_metrics"
                                if endpoint == "macro"
                                else "pooled_outer_test_folds"
                            ),
                            "n_test": np.nan if unavailable else 80,
                            "spearman": np.nan if unavailable else value,
                            "pearson": np.nan if unavailable else value - 0.01,
                            "natural_r2": np.nan if unavailable else value - 0.08,
                            "rmse": np.nan if unavailable else 2.0 - value,
                            "mae": np.nan if unavailable else 1.0 - value / 2,
                            "b0_rmse": np.nan if unavailable else 2.2,
                            "rmse_gain_over_b0": np.nan if unavailable else 0.1,
                            "prediction_target_variance_ratio": np.nan if unavailable else 0.8,
                            "calibration_slope": np.nan if unavailable else 0.7,
                            "calibration_intercept": np.nan if unavailable else 0.1,
                            "calibration_mean_bias": np.nan if unavailable else 0.02,
                            "transformed_scale": np.nan if unavailable else "outer_train",
                            "transformed_fold_count": np.nan if unavailable else 5,
                            "transformed_spearman_fold_mean": np.nan if unavailable else value,
                            "transformed_spearman_fold_sd": np.nan if unavailable else 0.02,
                            "transformed_r2_fold_mean": np.nan if unavailable else value - 0.1,
                            "transformed_r2_fold_sd": np.nan if unavailable else 0.03,
                            "transformed_rmse_fold_mean": np.nan if unavailable else 0.9,
                            "transformed_mae_fold_mean": np.nan if unavailable else 0.7,
                        }
                        rows.append(row)
    return pd.DataFrame(rows, columns=FTV_TABLE_COLUMNS)


def synthetic_table4(*, include_s3: bool = True) -> pd.DataFrame:
    rows = []
    stages = {"final": FINAL_POOLINGS}
    if include_s3:
        stages["s3"] = S3_POOLINGS
    final_gain = {
        "P0": 0.0,
        "PVALID": 0.06,
        "PLOCAL": 0.10,
        "PLOCAL+GLOBAL": 0.12,
        "PORACLE": 0.16,
    }
    s3_gain = {"P0": 0.0, "PLOCAL": 0.09, "PORACLE": 0.15}
    for stage, poolings in stages.items():
        gains = final_gain if stage == "final" else s3_gain
        for seed_index, seed in enumerate(SEEDS):
            for new_arm, legacy_arm, role in (
                ("N1", "L1", "primary"),
                ("N3", "L3", "secondary_replication"),
            ):
                legacy = 0.60 + 0.005 * seed_index
                new = 0.40 + 0.005 * seed_index
                deficit = legacy - new
                for pooling in poolings:
                    gain = gains[pooling]
                    rows.append(
                        {
                            "stage": stage,
                            "analysis_role": role,
                            "seed_base": seed,
                            "new_arm": new_arm,
                            "matched_legacy_arm": legacy_arm,
                            "pooling": pooling,
                            "legacy_p0_spearman": legacy,
                            "new_p0_spearman": new,
                            "legacy_deficit": deficit,
                            "pooling_spearman": new + gain,
                            "absolute_gain_vs_new_p0": gain,
                            "recovery_ratio": gain / deficit,
                            "recovery_defined": True,
                            "status_reason": "",
                        }
                    )
    return pd.DataFrame(rows, columns=TABLE4_COLUMNS)


def synthetic_table5() -> pd.DataFrame:
    rows = []
    endpoints = (*TIMEPOINTS, "macro")
    for seed_index, seed in enumerate(SEEDS):
        for arm_index, arm in enumerate(ARMS):
            for pooling_index, pooling in enumerate(NUISANCE_POOLINGS):
                for target_index, target in enumerate(NUISANCE_TARGETS):
                    for endpoint_index, endpoint in enumerate(endpoints):
                        unavailable = arm in {"L1", "L3"} and pooling == "PVALID"
                        value = (
                            0.08
                            + 0.01 * arm_index
                            + 0.02 * pooling_index
                            + 0.004 * target_index
                            + 0.002 * seed_index
                            + 0.001 * endpoint_index
                        )
                        rows.append(
                            {
                                "stage": "final",
                                "seed_base": seed,
                                "arm": arm,
                                "pooling": pooling,
                                "target_name": target,
                                "endpoint": endpoint,
                                "availability": "NA" if unavailable else "AVAILABLE",
                                "status_reason": "synthetic_NA" if unavailable else "",
                                "feature_dim": np.nan if unavailable else 192,
                                "aggregation": (
                                    np.nan
                                    if unavailable
                                    else "mean_of_pooled_endpoint_metrics"
                                    if endpoint == "macro"
                                    else "pooled_outer_test_folds"
                                ),
                                "n_test": np.nan if unavailable else 100,
                                "spearman": np.nan if unavailable else value,
                                "pearson": np.nan if unavailable else value - 0.01,
                                "natural_r2": np.nan if unavailable else value + 0.04,
                                "rmse": np.nan if unavailable else 0.8,
                                "mae": np.nan if unavailable else 0.6,
                                "standardized_scale": np.nan if unavailable else "outer_train",
                                "standardized_fold_count": np.nan if unavailable else 5,
                                "standardized_spearman_fold_mean": np.nan if unavailable else value,
                                "standardized_r2_fold_mean": np.nan if unavailable else value,
                                "standardized_r2_fold_sd": np.nan if unavailable else 0.02,
                            }
                        )
    return pd.DataFrame(rows, columns=TABLE5_COLUMNS)


def synthetic_table6() -> pd.DataFrame:
    rows = []
    for seed_index, seed in enumerate(SEEDS):
        for endpoint_index, endpoint in enumerate(TIMEPOINTS):
            identities = [
                *(('occupancy_quartile', quartile) for quartile in ("Q1", "Q2", "Q3", "Q4")),
                ("occupancy_correlation", "ALL"),
                *(('downsampling_bin', label) for label in ("<=1.5", "(1.5,2]", ">2")),
                ("downsampling_correlation", "ALL"),
            ]
            for row_index, (analysis, stratum) in enumerate(identities):
                base = -0.04 + 0.015 * row_index + 0.004 * endpoint_index
                rows.append(
                    {
                        "analysis": analysis,
                        "seed_base": seed,
                        "endpoint": endpoint,
                        "stratum": stratum,
                        "n": 90,
                        "l1_spearman": 0.5,
                        "n1_spearman": 0.5 + base,
                        "n1_minus_l1_spearman": base,
                        "l1_mae": 0.5,
                        "n1_mae": 0.5 - base,
                        "n1_minus_l1_mae": -base,
                        "mean_paired_abs_error_difference": -base,
                        "stratifier_error_difference_spearman": 0.1 + 0.01 * seed_index,
                        "status_reason": "",
                    }
                )
    return pd.DataFrame(rows, columns=TABLE6_COLUMNS)


def synthetic_table7() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for arm_index, arm in enumerate(ARMS):
            for fold in FOLDS:
                selected = 2 + (fold + arm_index) % 2
                observed = 6 + (fold % 2)
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "fold": fold,
                        "selected_epoch": selected,
                        "observed_max_epoch": observed,
                        "configured_max_epoch": 12,
                        "hit_configured_max_epoch": False,
                        "selected_in_last_two_observed_epochs": False,
                        "selected_validation_state_loss": 0.1,
                        "final_validation_state_loss": 0.25,
                        "final_minus_selected_state_loss": 0.15,
                        "last_three_normalized_validation_state_slope": 0.2 + arm_index / 100,
                        "early_stopping_reason": "patience_exhausted_after_selected_epoch",
                        "selection_mode": "primary",
                        "optimization_safety_pass": True,
                        "history_sha256": "a" * 64,
                        "selection_sha256": "b" * 64,
                    }
                )
    return pd.DataFrame(rows, columns=TABLE7_COLUMNS)


def write_synthetic_sidecar(path: Path) -> None:
    patient_ids = np.asarray([f"SYNTHETIC_{index:04d}" for index in range(808)])
    valid = np.ones((808, 4, 14, 22, 20), dtype=np.float32)
    oracle_valid = np.zeros((808, 4), dtype=bool)
    oracle_valid.reshape(-1)[:1500] = True
    oracle = np.zeros_like(valid)
    oracle[oracle_valid, 7, 11, 10] = 1.0
    local = np.zeros((14, 22, 20), dtype=np.float32)
    local[4:10, 7:15, 6:14] = 1.0
    legacy_local = np.zeros((808, 4, 4, 12, 12), dtype=np.float32)
    legacy_local[:, :, 1:3, 4:8, 4:8] = 1.0
    np.savez_compressed(
        path,
        patient_id=patient_ids,
        c1b_valid_weight_final=valid,
        c1b_oracle_weight_final=oracle,
        c1b_oracle_valid=oracle_valid,
        c1b_local_weight_final=local,
        legacy_local_weight_final=legacy_local,
    )
    path.chmod(0o600)


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        tables = {
            "table1": synthetic_table1(),
            "table2": synthetic_ftv_table(static=True),
            "table3": synthetic_ftv_table(static=False),
            "table4": synthetic_table4(),
            "table5": synthetic_table5(),
            "table6": synthetic_table6(),
            "table7": synthetic_table7(),
        }
        cls.table_paths = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name, frame in tables.items():
                path = cls.root / f"{name}.csv"
                frame.to_csv(path, index=False)
                cls.table_paths[name] = path
        cls.sidecar = cls.root / "audit_sidecars.private.npz"
        write_synthetic_sidecar(cls.sidecar)
        cls.activation = cls.root / "activation_aggregate.private.npz"
        volume = np.linspace(0.01, 1.0, 14 * 22 * 20, dtype=np.float32).reshape(14, 22, 20)
        write_private_activation_aggregate(
            ActivationAggregate(volume, LOCKED_ACTIVATION_SHA256), cls.activation
        )
        cls.paths = FigureInputPaths(
            table1=cls.table_paths["table1"],
            table2=cls.table_paths["table2"],
            table3=cls.table_paths["table3"],
            table4=cls.table_paths["table4"],
            table5=cls.table_paths["table5"],
            table6=cls.table_paths["table6"],
            table7=cls.table_paths["table7"],
            sidecar=cls.sidecar,
            activation_aggregate=cls.activation,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        patcher = mock.patch.object(
            figures_module,
            "_fixed_activation_checkpoint_binding",
            return_value=(PRIVATE_CHECKPOINT_SENTINEL, LOCKED_ACTIVATION_SHA256),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _variant_paths(
        self,
        name: str,
        *,
        include_s3: bool = True,
        include_secondary: bool = True,
        replacements: dict[str, pd.DataFrame] | None = None,
        sidecar: Path | None = None,
        activation: Path | None = None,
    ) -> FigureInputPaths:
        destination = self.root / name
        destination.mkdir()
        tables = {
            "table1": synthetic_table1(),
            "table2": synthetic_ftv_table(
                static=True,
                include_s3=include_s3,
                include_secondary=include_secondary,
            ),
            "table3": synthetic_ftv_table(
                static=False,
                include_s3=include_s3,
                include_secondary=include_secondary,
            ),
            "table4": synthetic_table4(include_s3=include_s3),
            "table5": synthetic_table5(),
            "table6": synthetic_table6(),
            "table7": synthetic_table7(),
        }
        tables.update(replacements or {})
        paths: dict[str, Path] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for table_name, frame in tables.items():
                path = destination / f"{table_name}.csv"
                frame.to_csv(path, index=False)
                paths[table_name] = path
        return FigureInputPaths(
            **paths,
            sidecar=sidecar or self.sidecar,
            activation_aggregate=activation or self.activation,
        )

    def test_hash_selection_and_normalized_activation_math(self) -> None:
        patient_ids = ("zeta", "alpha", "beta", "gamma")
        expected = tuple(
            sorted(
                patient_ids,
                key=lambda value: (hashlib.sha256(value.encode()).digest(), value),
            )[:2]
        )
        self.assertEqual(select_patient_ids_by_sha256(patient_ids, count=2), expected)
        activations = np.asarray(
            [
                [[[[[1.0, 2.0]]]]],
                [[[[[10.0, 20.0]]]]],
            ],
            dtype=np.float32,
        )
        aggregate = aggregate_normalized_abs_activations(activations)
        np.testing.assert_allclose(aggregate, np.asarray([[[0.5, 1.0]]], dtype=np.float32))

    def test_private_activation_schema_permissions_and_nonoverwrite(self) -> None:
        value = validate_activation_aggregate(self.activation)
        self.assertEqual(value.activation_mean_zyx.shape, (14, 22, 20))
        self.assertEqual(stat.S_IMODE(self.activation.stat().st_mode), 0o600)
        with np.load(self.activation, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), ACTIVATION_AGGREGATE_KEYS)
            self.assertNotIn("patient_id", archive.files)
            self.assertFalse(any("path" in name for name in archive.files))
        with self.assertRaises(FileExistsError):
            write_private_activation_aggregate(value, self.activation)

    def test_render_exact_twelve_public_pngs_with_s3(self) -> None:
        output = self.root / "public_figures"
        rendered = render_public_figures(self.paths, output_dir=output)
        self.assertEqual(tuple(rendered), FIGURE_FILENAMES)
        self.assertEqual(
            {path.name for path in output.glob("*.png")}, set(FIGURE_FILENAMES)
        )
        self.assertEqual(len(list(output.glob("*.png"))), 12)
        for path in rendered.values():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                dpi = image.info.get("dpi")
                self.assertIsNotNone(dpi)
                self.assertGreaterEqual(float(dpi[0]), 180.0)
                self.assertGreaterEqual(float(dpi[1]), 180.0)
            raw = path.read_bytes()
            self.assertNotIn(b"SYNTHETIC_0000", raw)
            self.assertNotIn(str(self.root).encode(), raw)
            self.assertNotIn(str(PRIVATE_CHECKPOINT_SENTINEL).encode(), raw)
        self.assertEqual(plt.get_fignums(), [])
        with self.assertRaises(FileExistsError):
            render_public_figures(self.paths, output_dir=output)

    def test_render_exact_twelve_public_pngs_without_s3(self) -> None:
        paths = self._variant_paths("without_s3", include_s3=False)
        output = self.root / "public_figures_without_s3"
        rendered = render_public_figures(paths, output_dir=output)
        self.assertEqual(tuple(rendered), FIGURE_FILENAMES)
        self.assertEqual(len(list(output.glob("*.png"))), 12)

    def test_ftv_feature_dimensions_fail_closed_before_png(self) -> None:
        cases = (
            (
                "final_concat",
                lambda frame: (
                    frame["stage"].eq("final")
                    & frame["arm"].eq("N1")
                    & frame["pooling"].eq("PLOCAL+GLOBAL")
                ),
                192,
            ),
            (
                "secondary_concat",
                lambda frame: frame["pooling"].eq("PLOCAL+PVALID_SECONDARY"),
                192,
            ),
            (
                "s3_raw",
                lambda frame: (
                    frame["stage"].eq("s3")
                    & frame["arm"].eq("N1")
                    & frame["pooling"].eq("P0")
                ),
                192,
            ),
        )
        for name, selector, wrong_dimension in cases:
            with self.subTest(name=name):
                table2 = synthetic_ftv_table(static=True)
                table2.loc[selector(table2), "feature_dim"] = wrong_dimension
                paths = self._variant_paths(
                    f"wrong_dimension_{name}", replacements={"table2": table2}
                )
                output = self.root / f"wrong_dimension_output_{name}"
                with self.assertRaisesRegex(ValueError, "feature_dim"):
                    render_public_figures(paths, output_dir=output)
                self.assertFalse(output.exists())

    def test_ftv_role_scope_and_aggregation_fail_closed(self) -> None:
        cases = (
            ("role", "analysis_role", "conditional_s3", "analysis-role"),
            ("scope", "analysis_scope", "wrong_scope", "analysis scope"),
            (
                "aggregation",
                "aggregation",
                "outer_fold_mean",
                "mean_of_pooled_endpoint_metrics",
            ),
        )
        for name, column, wrong_value, message in cases:
            with self.subTest(name=name):
                table2 = synthetic_ftv_table(static=True)
                selected = (
                    table2["stage"].eq("final")
                    & table2["arm"].eq("N1")
                    & table2["pooling"].eq("P0")
                    & table2["endpoint"].eq("macro")
                )
                table2.loc[selected, column] = wrong_value
                paths = self._variant_paths(
                    f"wrong_contract_{name}", replacements={"table2": table2}
                )
                output = self.root / f"wrong_contract_output_{name}"
                with self.assertRaisesRegex(ValueError, message):
                    render_public_figures(paths, output_dir=output)
                self.assertFalse(output.exists())

    def test_table1_and_table5_metadata_fail_closed(self) -> None:
        table1_cases = (
            ("input", "input_shape_zyx", "111x176x160", "frozen geometry"),
            ("role", "analysis_role", "primary", "frozen geometry"),
            ("window", "local_window_mm_xyz", "32x32x32", "local-window"),
        )
        for name, column, wrong_value, message in table1_cases:
            with self.subTest(table1=name):
                table1 = synthetic_table1()
                selected = table1["stage"].eq("s3") & table1["input_contract"].eq(
                    "c1b"
                )
                table1.loc[selected, column] = wrong_value
                paths = self._variant_paths(
                    f"wrong_table1_{name}", replacements={"table1": table1}
                )
                with self.assertRaisesRegex(ValueError, message):
                    render_public_figures(
                        paths, output_dir=self.root / f"wrong_table1_output_{name}"
                    )

        for name, column, wrong_value, message in (
            (
                "aggregation",
                "aggregation",
                "outer_fold_mean",
                "mean_of_pooled_endpoint_metrics",
            ),
            ("dimension", "feature_dim", 384, "feature_dim"),
        ):
            with self.subTest(table5=name):
                table5 = synthetic_table5()
                selected = (
                    table5["arm"].eq("N1")
                    & table5["pooling"].eq("PLOCAL")
                    & table5["endpoint"].eq("macro")
                )
                table5.loc[selected, column] = wrong_value
                paths = self._variant_paths(
                    f"wrong_table5_{name}", replacements={"table5": table5}
                )
                with self.assertRaisesRegex(ValueError, message):
                    render_public_figures(
                        paths, output_dir=self.root / f"wrong_table5_output_{name}"
                    )

    def test_table2_table3_secondary_presence_must_match(self) -> None:
        table3 = synthetic_ftv_table(static=False, include_secondary=False)
        paths = self._variant_paths(
            "secondary_mismatch", replacements={"table3": table3}
        )
        output = self.root / "secondary_mismatch_output"
        with self.assertRaisesRegex(ValueError, "secondary-pooling presence"):
            render_public_figures(paths, output_dir=output)
        self.assertFalse(output.exists())

    def test_wrong_activation_checkpoint_sha_fails_before_png(self) -> None:
        output = self.root / "wrong_activation_sha_output"
        with mock.patch.object(
            figures_module,
            "_fixed_activation_checkpoint_binding",
            return_value=(PRIVATE_CHECKPOINT_SENTINEL, "d" * 64),
        ):
            with self.assertRaisesRegex(ValueError, "2026/N1/fold-0"):
                render_public_figures(self.paths, output_dir=output)
        self.assertFalse(output.exists())

    def test_live_activation_checkpoint_path_and_sha_binding(self) -> None:
        checkpoint = self.root / "synthetic_selected.pt"
        checkpoint.write_bytes(b"frozen checkpoint placeholder")
        digest = "d" * 64
        key = "seed_2026/N1/fold_0"
        lock = {
            "selected_checkpoints": {
                key: {
                    "path": "locked/selected.pt",
                    "sha256": digest,
                    "size_bytes": checkpoint.stat().st_size,
                }
            }
        }
        with (
            mock.patch(
                "c1b_spatial_audit.runtime.verify_preregistration", return_value=lock
            ),
            mock.patch.object(figures_module, "checkpoint_path", return_value=checkpoint),
            mock.patch.object(figures_module, "relative", return_value="locked/selected.pt"),
            mock.patch.object(figures_module, "file_sha256", return_value=digest),
        ):
            observed_path, observed_digest = REAL_FIXED_ACTIVATION_BINDING()
        self.assertEqual(observed_path, checkpoint.resolve())
        self.assertEqual(observed_digest, digest)

        bad_lock = {
            "selected_checkpoints": {
                key: {
                    "path": "wrong/selected.pt",
                    "sha256": digest,
                    "size_bytes": checkpoint.stat().st_size,
                }
            }
        }
        with (
            mock.patch(
                "c1b_spatial_audit.runtime.verify_preregistration", return_value=bad_lock
            ),
            mock.patch.object(figures_module, "checkpoint_path", return_value=checkpoint),
            mock.patch.object(figures_module, "relative", return_value="locked/selected.pt"),
        ):
            with self.assertRaisesRegex(ValueError, "path differs"):
                REAL_FIXED_ACTIVATION_BINDING()

    def test_formal_activation_helpers_use_only_encoder_and_exact_16x4(self) -> None:
        import torch

        class Encoder:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, flat: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                self.input_shape = tuple(flat.shape)
                spatial = torch.ones((8, 2, 2, 3, 4), dtype=torch.float32)
                spatial[:, 1] = 2.0
                return spatial

        class EncoderOnlyModel:
            def __init__(self) -> None:
                self.encoder = Encoder()

            def __call__(self, *_: object, **__: object) -> object:
                raise AssertionError("full model forward is forbidden")

            @property
            def response_projection(self) -> object:
                raise AssertionError("response projection is forbidden")

        model = EncoderOnlyModel()
        image = torch.ones((2, 4, 1, 2, 3, 4), dtype=torch.float32)
        maps = figures_module._encode_normalized_abs_activation_maps(
            model,
            image,
            torch.device("cpu"),
            input_channels=1,
            input_shape_zyx=(2, 3, 4),
            channel_count=2,
            feature_shape_zyx=(2, 3, 4),
        )
        self.assertEqual(model.encoder.calls, 1)
        self.assertEqual(model.encoder.input_shape, (8, 1, 2, 3, 4))
        np.testing.assert_allclose(maps.numpy(), 0.75)

        selected_ids = tuple(f"PRIVATE_{index:02d}" for index in range(16))
        loader = [
            {
                "patient_id": selected_ids[index : index + 4],
                "image": torch.tensor(index // 4 + 1, dtype=torch.float32),
                "ftv_target": "must_not_be_read",
            }
            for index in range(0, 16, 4)
        ]

        def encode_batch(token: torch.Tensor) -> torch.Tensor:
            value = float(token.item()) / 4.0
            return torch.full(
                (16, 14, 22, 20), value, dtype=torch.float32
            )

        aggregate = figures_module._aggregate_selected_encoder_loader(
            loader, selected_ids, encode_batch
        )
        np.testing.assert_allclose(aggregate, 0.625)
        private_path = self.root / "formal_helper_activation.private.npz"
        write_private_activation_aggregate(
            ActivationAggregate(aggregate, LOCKED_ACTIVATION_SHA256), private_path
        )
        with np.load(private_path, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), ACTIVATION_AGGREGATE_KEYS)
            self.assertNotIn("patient_id", archive.files)
            self.assertNotIn("patient_ids", archive.files)
            self.assertFalse(any("path" in name for name in archive.files))
        raw = private_path.read_bytes()
        self.assertFalse(any(value.encode() in raw for value in selected_ids))

    def test_sidecar_must_be_owner_only(self) -> None:
        sidecar = self.root / "world_readable_sidecar.private.npz"
        write_synthetic_sidecar(sidecar)
        sidecar.chmod(0o644)
        paths = self._variant_paths("bad_sidecar_mode", sidecar=sidecar)
        output = self.root / "bad_sidecar_mode_output"
        with self.assertRaisesRegex(PermissionError, "sidecar must be owner-only"):
            render_public_figures(paths, output_dir=output)
        self.assertFalse(output.exists())

    def test_public_publication_failure_rolls_back_all_pngs(self) -> None:
        output = self.root / "atomic_rollback_output"
        real_link = os.link
        publication_calls = 0

        def fail_third_publication(source: object, destination: object) -> None:
            nonlocal publication_calls
            publication_calls += 1
            if publication_calls == 3:
                raise OSError("synthetic atomic publication failure")
            real_link(source, destination)

        with mock.patch.object(figures_module.os, "link", side_effect=fail_third_publication):
            with self.assertRaisesRegex(OSError, "synthetic atomic"):
                render_public_figures(self.paths, output_dir=output)
        self.assertEqual(list(output.glob("*.png")), [])
        self.assertEqual(list(output.glob(".*.png")), [])
        self.assertEqual(plt.get_fignums(), [])

    def test_missing_public_row_fails_before_any_png(self) -> None:
        broken = pd.read_csv(self.table_paths["table4"]).iloc[:-1]
        broken_path = self.root / "table4_broken.csv"
        broken.to_csv(broken_path, index=False)
        paths = FigureInputPaths(
            table1=self.paths.table1,
            table2=self.paths.table2,
            table3=self.paths.table3,
            table4=broken_path,
            table5=self.paths.table5,
            table6=self.paths.table6,
            table7=self.paths.table7,
            sidecar=self.paths.sidecar,
            activation_aggregate=self.paths.activation_aggregate,
        )
        output = self.root / "broken_output"
        with self.assertRaisesRegex(ValueError, "Table 4"):
            render_public_figures(paths, output_dir=output)
        self.assertFalse(output.exists())
        self.assertEqual(plt.get_fignums(), [])

    def test_singleton_occupancy_requires_undefined_spearman(self) -> None:
        table6 = synthetic_table6()
        singleton = (
            table6["analysis"].eq("occupancy_quartile")
            & table6["seed_base"].eq(SEEDS[0])
            & table6["endpoint"].eq(TIMEPOINTS[0])
            & table6["stratum"].eq("Q1")
        )
        table6.loc[singleton, "n"] = 1
        table6.loc[
            singleton,
            ["l1_spearman", "n1_spearman", "n1_minus_l1_spearman"],
        ] = np.nan
        paths = self._variant_paths(
            "singleton_occupancy", replacements={"table6": table6}
        )
        output = self.root / "singleton_occupancy_output"
        rendered = render_public_figures(paths, output_dir=output)
        self.assertEqual(len(rendered), 12)

        invalid = table6.copy()
        invalid.loc[singleton, "n"] = 2
        invalid_paths = self._variant_paths(
            "invalid_singleton_occupancy", replacements={"table6": invalid}
        )
        invalid_output = self.root / "invalid_singleton_occupancy_output"
        with self.assertRaisesRegex(ValueError, "missing/non-finite"):
            render_public_figures(invalid_paths, output_dir=invalid_output)
        self.assertFalse(invalid_output.exists())


if __name__ == "__main__":
    unittest.main()
