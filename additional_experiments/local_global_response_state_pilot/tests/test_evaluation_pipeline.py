from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SEALED_SRC = (
    EXPERIMENT_ROOT.parent
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
)
for source in (SRC_ROOT, SEALED_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from lg_response_pilot import analysis, features, figures, probes  # noqa: E402


CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "pilot.json"
LOCK_SHA256 = "a" * 64


def pilot_config() -> dict[str, object]:
    return analysis.load_pilot_config(CONFIG_PATH)


def load_postprocessing_module():
    path = EXPERIMENT_ROOT / "scripts" / "run_postprocessing.py"
    spec = importlib.util.spec_from_file_location("pilot_run_postprocessing_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complete_audit_frames(config: dict[str, object]):
    selections: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    task_contracts = (
        (
            "static",
            tuple(config["probes"]["static_endpoints"][:-1]),
            "transformed_outer_train",
            "static_ftv_log_winsor_median_iqr_inverse_natural",
        ),
        (
            "delta",
            tuple(config["probes"]["delta_endpoints"][:-1]),
            "standardized_outer_train",
            "literal_ftv_end_minus_ftv_start",
        ),
    )
    for seed in config["training"]["seed_bases"]:
        for arm_index, arm in enumerate(config["arms"]):
            for fold in config["training"]["folds"]:
                identity = {"seed_base": seed, "arm": arm, "fold": fold}
                selections.append(identity)
                histories.append({**identity, "epoch": 1})
                for scope in analysis.ANALYSIS_SCOPES:
                    for task, endpoints, transformed_scale, semantics in task_contracts:
                        for endpoint in endpoints:
                            common = {
                                **identity,
                                "analysis_scope": scope,
                                "task": task,
                                "endpoint": endpoint,
                                "target_semantics": semantics,
                            }
                            for scale in ("natural", transformed_scale):
                                metrics.append(
                                    {
                                        **common,
                                        "scale": scale,
                                        "n_test": 1,
                                    }
                                )
                            patient = f"synthetic_{seed}_{fold}_{scope}_{task}_{endpoint}"
                            predictions.append(
                                {
                                    **common,
                                    "analysis_scale": transformed_scale,
                                    "patient_id": patient,
                                    "split": "test",
                                    "y_true": float(fold + 1),
                                    "y_pred": float(fold + 1 + arm_index / 100),
                                    "b0_prediction": 1.0,
                                    "y_true_analysis": float(fold),
                                    "y_pred_analysis": float(fold + arm_index / 100),
                                    "b0_prediction_analysis": 0.0,
                                    "test_predict_call_count": 1,
                                }
                            )
    return tuple(pd.DataFrame(rows) for rows in (selections, histories, metrics, predictions))


def macro_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    static_spearman = {
        2026: {
            "GAP0": 0.20, "GAP3": 0.21, "LOCAL0": 0.31,
            "LOCAL3": 0.33, "LG0": 0.31, "LG3": 0.34,
        },
        3026: {
            "GAP0": 0.22, "GAP3": 0.23, "LOCAL0": 0.34,
            "LOCAL3": 0.36, "LG0": 0.37, "LG3": 0.39,
        },
    }
    delta_spearman = {
        2026: {
            "GAP0": 0.10, "GAP3": 0.11, "LOCAL0": 0.12,
            "LOCAL3": 0.14, "LG0": 0.13, "LG3": 0.16,
        },
        3026: {
            "GAP0": 0.08, "GAP3": 0.09, "LOCAL0": 0.10,
            "LOCAL3": 0.12, "LG0": 0.11, "LG3": 0.12,
        },
    }
    static_r2 = {
        seed: {arm: 0.05 + index / 100 for index, arm in enumerate(analysis.CANONICAL_ARMS)}
        for seed in (2026, 3026)
    }
    rows: dict[str, list[dict[str, object]]] = {"static": [], "delta": []}
    for seed in (2026, 3026):
        for arm in analysis.CANONICAL_ARMS:
            common = {
                "seed_base": seed,
                "arm": arm,
                "endpoint": "macro",
                "analysis_scope": analysis.PRIMARY_SCOPE,
            }
            rows["static"].append(
                {
                    **common,
                    "task": "static",
                    "natural_spearman": static_spearman[seed][arm],
                    "natural_r2": static_r2[seed][arm],
                    "natural_prediction_target_variance_ratio": 0.4,
                    "natural_calibration_slope": 0.5,
                    "natural_calibration_intercept": 0.0,
                    "natural_calibration_mean_bias": 0.0,
                    "natural_n_test": 100,
                }
            )
            rows["delta"].append(
                {
                    **common,
                    "task": "delta",
                    "natural_spearman": delta_spearman[seed][arm],
                    "natural_r2": 0.02,
                    "natural_prediction_target_variance_ratio": 0.3,
                    "natural_calibration_slope": 0.4,
                    "natural_calibration_intercept": 0.0,
                    "natural_calibration_mean_bias": 0.0,
                    "natural_n_test": 80,
                }
            )
    return pd.DataFrame(rows["static"]), pd.DataFrame(rows["delta"])


def safety_table(pass_count: int = 9) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = ("GAP3-GAP0", "LOCAL3-LOCAL0", "LG3-LG0")
    for comparison in comparisons:
        for index, (seed, fold) in enumerate(
            (seed, fold) for seed in (2026, 3026) for fold in range(5)
        ):
            passed = index < pass_count
            rows.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "comparison": comparison,
                    "state_loss_degradation_fraction": 0.04 if passed else 0.06,
                    "safety_pass": passed,
                }
            )
    return pd.DataFrame(rows)


class ConfigAndMetricTest(unittest.TestCase):
    def test_exact_nested_gate_schema_is_fail_closed(self) -> None:
        config = pilot_config()
        self.assertEqual(
            set(config["gates"]),
            {
                "A_LOCAL_STATE_WORKS",
                "B_LOCAL_GLOBAL_ADDS_VALUE",
                "C_GROUNDING_COMPATIBILITY",
                "D_OPTIMIZATION_SAFETY",
            },
        )
        changed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        changed["gates"]["A_LOCAL_STATE_WORKS"]["wrong_key"] = changed[
            "gates"
        ]["A_LOCAL_STATE_WORKS"].pop(
            "static_macro_spearman_gain_each_seed_min"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "threshold fields drifted"):
                analysis.load_pilot_config(path)

    def test_metric_formula_uses_prior_calibration_orientation(self) -> None:
        truth = np.asarray([0.0, 1.0, 2.0, 3.0])
        prediction = 0.25 + 0.5 * truth
        result = probes.metric_values(truth, prediction, np.mean(truth))
        self.assertAlmostEqual(result["calibration_slope"], 0.5)
        self.assertAlmostEqual(result["calibration_intercept"], 0.25)
        self.assertAlmostEqual(result["prediction_target_variance_ratio"], 0.25)

    def test_ridge_is_train_val_only_and_test_predict_is_single_use(self) -> None:
        train_x = np.arange(24, dtype=float).reshape(8, 3)
        train_y = np.linspace(-1.0, 1.0, 8)
        val_x = np.arange(12, dtype=float).reshape(4, 3) + 0.5
        val_y = np.linspace(-0.5, 0.5, 4)
        selected = probes.select_ridge(
            train_x,
            train_y,
            val_x,
            val_y,
            probes.ALPHAS,
            standardize_target=True,
        )
        self.assertIn(selected.alpha, probes.ALPHAS)
        self.assertEqual(int(selected.x_scaler.n_samples_seen_), len(train_x))
        guard = probes.TestPredictGuard()
        matrix = selected.x_scaler.transform(val_x)
        guard.predict(selected.model, matrix)
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            guard.predict(selected.model, matrix)

    def test_natural_metrics_pool_five_folds_before_endpoint_macro(self) -> None:
        config = pilot_config()
        rows: list[dict[str, object]] = []
        for endpoint_index, endpoint in enumerate(("T0", "T1", "T2", "T3")):
            for fold in range(5):
                truth = float(fold + endpoint_index)
                rows.append(
                    {
                        "patient_id": f"p_{endpoint}_{fold}",
                        "seed_base": 2026,
                        "arm": "GAP0",
                        "fold": fold,
                        "analysis_scope": analysis.PRIMARY_SCOPE,
                        "task": "static",
                        "endpoint": endpoint,
                        "target_semantics": "static",
                        "y_true": truth,
                        "y_pred": 0.8 * truth + 0.2,
                        "b0_prediction": float(endpoint_index),
                    }
                )
        result = analysis.pooled_natural_metrics(pd.DataFrame(rows), config)
        endpoint = result.loc[result["endpoint"].eq("T0")].iloc[0]
        raw = pd.DataFrame(rows).loc[lambda frame: frame["endpoint"].eq("T0")]
        expected = probes.metric_values(
            raw["y_true"].to_numpy(float),
            raw["y_pred"].to_numpy(float),
            raw["b0_prediction"].to_numpy(float),
        )
        self.assertEqual(endpoint["aggregation"], "pooled_5fold_oof")
        self.assertAlmostEqual(endpoint["r2"], expected["r2"])
        macro = result.loc[result["endpoint"].eq("macro")].iloc[0]
        self.assertEqual(
            macro["aggregation"], "unweighted_mean_of_pooled_endpoint_metrics"
        )


class MatrixAndLockTest(unittest.TestCase):
    def test_cell_chain_binds_live_feature_selection_and_checkpoint_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / filename
                for name, filename in {
                    "feature": "response_state.private.npz",
                    "feature_metadata": "response_state.private.metadata.json",
                    "selected": "selected.pt",
                    "selection": "selection.json",
                    "probe_metadata": "probe_metadata.json",
                }.items()
            }
            for name, content in (
                ("feature", b"feature"),
                ("selected", b"checkpoint"),
                ("selection", b"selection"),
            ):
                paths[name].write_bytes(content)
            identity = {"arm": "GAP0", "seed_base": 2026, "fold": 0}
            feature_metadata = {
                "schema_version": 1,
                **identity,
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": LOCK_SHA256,
                "feature_path": str(paths["feature"].resolve()),
                "feature_sha256": analysis.file_sha256(paths["feature"]),
                "checkpoint_path": str(paths["selected"].resolve()),
                "checkpoint_sha256": analysis.file_sha256(paths["selected"]),
                "selection_path": str(paths["selection"].resolve()),
                "selection_sha256": analysis.file_sha256(paths["selection"]),
            }
            paths["feature_metadata"].write_text(
                json.dumps(feature_metadata), encoding="utf-8"
            )
            binding = {
                "feature_path": str(paths["feature"].resolve()),
                "feature_sha256": analysis.file_sha256(paths["feature"]),
                "feature_metadata_path": str(paths["feature_metadata"].resolve()),
                "feature_metadata_sha256": analysis.file_sha256(
                    paths["feature_metadata"]
                ),
                "checkpoint_path": str(paths["selected"].resolve()),
                "checkpoint_sha256": analysis.file_sha256(paths["selected"]),
                "selection_path": str(paths["selection"].resolve()),
                "selection_sha256": analysis.file_sha256(paths["selection"]),
            }
            probe_metadata = {
                "feature_sha256": binding["feature_sha256"],
                "feature_metadata_sha256": binding["feature_metadata_sha256"],
                "feature_checkpoint_binding": binding,
                "feature_checkpoint_binding_sha256": features.canonical_sha256(
                    binding
                ),
            }
            paths["probe_metadata"].write_text(
                json.dumps(probe_metadata), encoding="utf-8"
            )
            observed = analysis._validate_cell_chain(
                paths,
                identity,
                feature_metadata,
                probe_metadata,
                LOCK_SHA256,
            )
            self.assertEqual(observed["checkpoint_sha256"], binding["checkpoint_sha256"])
            paths["selection"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "selection_sha256 binding drifted"):
                analysis._validate_cell_chain(
                    paths,
                    identity,
                    feature_metadata,
                    probe_metadata,
                    LOCK_SHA256,
                )

    def test_exact_sixty_cell_probe_grid_and_one_test_prediction(self) -> None:
        config = pilot_config()
        frames = complete_audit_frames(config)
        analysis._audit_matrix(*frames, config)
        self.assertEqual(len(frames[0]), 60)
        self.assertEqual(len(frames[2]), 1680)
        broken = frames[2].iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "metric grid"):
            analysis._audit_matrix(frames[0], frames[1], broken, frames[3], config)

    def test_postprocessing_accepts_current_selection_lock_payload(self) -> None:
        module = load_postprocessing_module()
        config = pilot_config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = (Path(directory) / "checkpoints").resolve()
            cells = module._cells(
                checkpoint_root,
                (Path(directory) / "features").resolve(),
                (Path(directory) / "predictions").resolve(),
                config,
                ("cuda:0", "cuda:1", "cuda:2"),
            )
            runs = []
            for cell in cells:
                cell.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "seed_base": cell.seed,
                    "arm": cell.arm,
                    "fold": cell.fold,
                    "test_data_used": False,
                    "preregistration_status": "PASS",
                    "preregistration_lock_sha256": LOCK_SHA256,
                    "preregistration": {
                        "status": "PASS",
                        "lock_sha256": LOCK_SHA256,
                    },
                }
                cell.selection.write_text(json.dumps(payload), encoding="utf-8")
                cell.history.write_text("epoch,val_state_loss\n1,1.0\n", encoding="utf-8")
                cell.checkpoint.write_bytes(b"selected")
                runs.append(
                    {
                        "seed_base": cell.seed,
                        "arm": cell.arm,
                        "fold": cell.fold,
                        "selection_path": str(cell.selection),
                    }
                )
            completion = {
                "schema_version": 1,
                "status": "COMPLETE",
                "run_count": 60,
                "preregistration": {
                    "status": "PASS",
                    "lock_sha256": LOCK_SHA256,
                },
                "config_sha256": "c" * 64,
                "stage_a_sentinel_sha256": "a" * 64,
                "data_contract_sha256": "d" * 64,
                "runs": runs,
            }
            (checkpoint_root / "matrix_complete.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            observed = module._validate_matrix(
                checkpoint_root,
                cells,
                LOCK_SHA256,
                "c" * 64,
                "a" * 64,
                "d" * 64,
            )
            self.assertEqual(observed["run_count"], 60)
            changed = json.loads(cells[0].selection.read_text(encoding="utf-8"))
            changed["preregistration_lock_sha256"] = "b" * 64
            cells[0].selection.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding drifted"):
                module._validate_matrix(
                    checkpoint_root,
                    cells,
                    LOCK_SHA256,
                    "c" * 64,
                    "a" * 64,
                    "d" * 64,
                )

    def test_aggregation_rejects_checkpoint_probe_cross_root_mix(self) -> None:
        config = pilot_config()
        config["arms"] = {"GAP0": config["arms"]["GAP0"]}
        config["training"]["seed_bases"] = [2026]
        config["training"]["folds"] = [0]
        config["training"]["formal_cells"] = 1

        def write(path: Path, content: bytes) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(0o600)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_a = (root / "checkpoint_a").resolve()
            checkpoint_b = (root / "checkpoint_b").resolve()
            feature_b = (root / "feature_b").resolve()
            probe_b = (root / "probe_b").resolve()
            cell_a = checkpoint_a / "seed_2026" / "GAP0" / "fold_0"
            cell_b = checkpoint_b / "seed_2026" / "GAP0" / "fold_0"
            feature_cell = feature_b / "seed_2026" / "GAP0" / "fold_0"
            probe_cell = probe_b / "seed_2026" / "GAP0" / "fold_0"
            for cell in (cell_a, cell_b):
                write(cell / "selected.pt", b"selected checkpoint")
                write(cell / "selection.json", b"{}")
            history = b"epoch,val_state_loss\n1,1.0\n"
            write(cell_a / "history.csv", history)
            selection_a = {
                "arm": "GAP0",
                "seed_base": 2026,
                "fold": 0,
                "test_data_used": False,
                "history_sha256": hashlib.sha256(history).hexdigest(),
                "preregistration_status": "PASS",
                "preregistration_lock_sha256": LOCK_SHA256,
                "preregistration": {
                    "status": "PASS",
                    "lock_sha256": LOCK_SHA256,
                },
            }
            write(
                cell_a / "selection.json",
                json.dumps(selection_a).encode("utf-8"),
            )
            matrix = {
                "status": "COMPLETE",
                "run_count": 1,
                "preregistration": {
                    "status": "PASS",
                    "lock_sha256": LOCK_SHA256,
                },
                "config_sha256": "c" * 64,
                "stage_a_sentinel_sha256": "a" * 64,
                "data_contract_sha256": "d" * 64,
                "runs": [
                    {
                        "seed_base": 2026,
                        "arm": "GAP0",
                        "fold": 0,
                        "selection_path": str(cell_a / "selection.json"),
                    }
                ],
            }
            matrix_path = checkpoint_a / "matrix_complete.json"
            write(matrix_path, json.dumps(matrix).encode("utf-8"))
            feature_path = feature_cell / "response_state.private.npz"
            write(feature_path, b"private feature")
            feature_metadata_path = (
                feature_cell / "response_state.private.metadata.json"
            )
            feature_metadata = {
                "schema_version": 1,
                "arm": "GAP0",
                "seed_base": 2026,
                "fold": 0,
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": LOCK_SHA256,
                "feature_path": str(feature_path),
                "feature_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                # Deliberately bind the probe/feature root B to checkpoint root B.
                "checkpoint_path": str(cell_b / "selected.pt"),
                "checkpoint_sha256": hashlib.sha256(
                    (cell_b / "selected.pt").read_bytes()
                ).hexdigest(),
                "selection_path": str(cell_b / "selection.json"),
                "selection_sha256": hashlib.sha256(
                    (cell_b / "selection.json").read_bytes()
                ).hexdigest(),
            }
            write(
                feature_metadata_path,
                json.dumps(feature_metadata).encode("utf-8"),
            )
            for name in (
                "probe_metrics.csv",
                "ridge_predictions.private.csv",
                "ridge_selection.csv",
            ):
                write(probe_cell / name, b"placeholder\n")
            probe_metadata = {
                "arm": "GAP0",
                "seed_base": 2026,
                "fold": 0,
                "test_used_for_scaler_or_selection": False,
                "refit_after_alpha_selection": False,
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": LOCK_SHA256,
            }
            write(
                probe_cell / "probe_metadata.json",
                json.dumps(probe_metadata).encode("utf-8"),
            )
            postprocessing = {
                "status": "COMPLETE",
                "cells": 1,
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": LOCK_SHA256,
                "patient_level_outputs_private": True,
                "config_sha256": "c" * 64,
                "stage_a_sentinel_sha256": "a" * 64,
                "data_contract_sha256": "d" * 64,
                "data_provenance_sha256": "e" * 64,
                "matrix_completion_sha256": hashlib.sha256(
                    matrix_path.read_bytes()
                ).hexdigest(),
            }
            write(
                probe_b / "postprocessing_complete.private.json",
                json.dumps(postprocessing).encode("utf-8"),
            )
            with self.assertRaisesRegex(
                ValueError, "checkpoint_path differs from the supplied formal roots"
            ):
                analysis.collect_complete_matrix(
                    checkpoint_root=checkpoint_a,
                    feature_root=feature_b,
                    probe_root=probe_b,
                    config=config,
                    preregistration_lock_sha256=LOCK_SHA256,
                    source_provenance={
                        "config_sha256": "c" * 64,
                        "stage_a_sentinel_sha256": "a" * 64,
                        "data_contract_sha256": "d" * 64,
                        "data_provenance_sha256": "e" * 64,
                    },
                )


class GatesTablesAndFiguresTest(unittest.TestCase):
    def test_gates_choose_local_global_and_classify_unsafe_override(self) -> None:
        config = pilot_config()
        table2, table3 = macro_tables()
        decision = analysis.evaluate_gates(table2, table3, safety_table(9), config)
        self.assertTrue(decision["gates"]["A_LOCAL_STATE_WORKS"]["pass"])
        self.assertTrue(decision["gates"]["B_LOCAL_GLOBAL_ADDS_VALUE"]["pass"])
        self.assertTrue(decision["gates"]["C_GROUNDING_COMPATIBILITY"]["pass"])
        self.assertTrue(decision["gates"]["D_OPTIMIZATION_SAFETY"]["pass"])
        self.assertEqual(decision["winner"], "LOCAL_GLOBAL")
        self.assertEqual(
            decision["classification"],
            "B. LOCAL-GLOBAL STATE VALIDATED IN PILOT",
        )
        unsafe = analysis.evaluate_gates(table2, table3, safety_table(8), config)
        self.assertEqual(
            unsafe["classification"],
            "D. REPRESENTATION IMPROVED BUT GROUNDING UNSAFE",
        )

    def test_gate_metrics_reject_nan_instead_of_vacuously_passing(self) -> None:
        config = pilot_config()
        table2, table3 = macro_tables()
        table2.loc[
            table2["seed_base"].eq(2026) & table2["arm"].eq("LOCAL0"),
            "natural_r2",
        ] = math.nan
        with self.assertRaisesRegex(ValueError, "must be finite in both seeds"):
            analysis.evaluate_gates(table2, table3, safety_table(9), config)

    def test_seven_tables_and_architecture_parameter_counts(self) -> None:
        config = pilot_config()
        self.assertEqual(set(analysis.TABLE_FILENAMES), {f"table{i}" for i in range(1, 8)})
        table1 = analysis.architecture_table(config)
        indexed = table1.set_index("arm")
        projection_counts = indexed["response_projection_parameter_count"].to_dict()
        self.assertEqual(projection_counts["GAP0"], 25_152)
        self.assertEqual(projection_counts["LOCAL0"], 25_152)
        self.assertEqual(projection_counts["LG0"], 49_728)
        self.assertTrue(indexed["parameter_count_scope"].eq("trainable_total").all())
        self.assertGreater(
            indexed.loc["GAP3", "parameter_count"],
            indexed.loc["GAP0", "parameter_count"],
        )
        self.assertGreater(indexed.loc["GAP3", "ftv_head_parameter_count"], 0)
        self.assertEqual(indexed.loc["GAP0", "ftv_head_parameter_count"], 0)
        table2, table3 = macro_tables()
        effects = analysis.effect_table(
            (table2, table3),
            (("LOCAL0-GAP0", "LOCAL0", "GAP0"),),
        )
        self.assertEqual(set(effects["comparison"]), {"LOCAL0-GAP0"})
        calibration = analysis.prediction_calibration_table(table2, table3)
        self.assertEqual(len(calibration), 24)

    def test_all_ten_figures_render_from_deidentified_tables(self) -> None:
        config = pilot_config()
        table1 = analysis.architecture_table(config)
        table2, table3 = macro_tables()
        fold_rows = []
        for seed in (2026, 3026):
            for fold in range(5):
                for task in ("static", "delta"):
                    for comparison in ("LOCAL0-GAP0", "LG0-LOCAL0"):
                        fold_rows.append(
                            {
                                "seed_base": seed,
                                "fold": fold,
                                "task": task,
                                "comparison": comparison,
                                "effect_spearman": 0.01 * (fold + 1),
                            }
                        )
        history_rows = [
            {
                "arm": arm,
                "epoch": epoch,
                "val_state_loss": 1.0 / epoch,
                "val_ftv_loss": 0.5 / epoch,
            }
            for arm in analysis.CANONICAL_ARMS
            for epoch in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = figures.render_required_figures(
                table1=table1,
                table2=table2,
                table3=table3,
                table4=pd.DataFrame(),
                table6=safety_table(9),
                fold_effects=pd.DataFrame(fold_rows),
                histories=pd.DataFrame(history_rows),
                output_dir=directory,
            )
            self.assertEqual(tuple(path.name for path in paths), figures.FIGURE_FILENAMES)
            for path in paths:
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)


class PrivacyAndEntrypointTest(unittest.TestCase):
    def test_private_feature_and_prediction_writers_are_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / "response_state.private.npz"
            metadata = root / "response_state.private.metadata.json"
            prediction = root / "ridge_predictions.private.csv"
            probe_metadata = root / "probe_metadata.json"
            features._atomic_npz(feature, value=np.asarray([1], dtype=np.int64))
            features._atomic_json(metadata, {"status": "ok"}, private=True)
            probes._atomic_csv(prediction, pd.DataFrame({"value": [1]}))
            probes._atomic_json(probe_metadata, {"status": "ok"})
            for path in (feature, metadata, prediction, probe_metadata):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_every_evaluation_entrypoint_verifies_lock_before_arguments(self) -> None:
        for name in (
            "export_features.py",
            "run_probes.py",
            "run_postprocessing.py",
            "aggregate_results.py",
            "generate_figures.py",
        ):
            source = (EXPERIMENT_ROOT / "scripts" / name).read_text(encoding="utf-8")
            main = source.index("def main() -> None:")
            verify = source.index("verify_preregistration()", main)
            parse = source.index("parse_args()", main)
            self.assertLess(verify, parse, name)

    def test_public_writers_reject_private_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "absolute private path"):
                analysis._atomic_public_json(
                    Path(directory) / "summary.json", {"source": "/data/private"}
                )


if __name__ == "__main__":
    unittest.main()
