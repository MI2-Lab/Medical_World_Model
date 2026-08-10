from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import c1b_stage_b.analysis as analysis  # noqa: E402
from c1b_stage_b.analysis import (  # noqa: E402
    FIGURE_NAMES,
    METRICS,
    _audit_four_arm_matrix_contract,
    _audit_oof_prediction_contract,
    _claim_formal_aggregation,
    aggregate_stage_b,
    collect_complete_matrix,
    optimization_table,
    paired_effects,
    pooled_oof_metrics,
    validate_formal_aggregation_inputs,
)
from c1b_stage_b.contracts import (  # noqa: E402
    ARMS,
    FOLDS,
    G3_SRC,
    SEED_BASES,
    canonical_sha256,
    file_sha256,
)
from c1b_stage_b.gate import StageAAuthorization  # noqa: E402
from c1b_stage_b.postprocess import (  # noqa: E402
    FORMAL_DEVICES,
    FORMAL_HYPERPARAMETERS,
    FORMAL_POSTPROCESS_TAG,
    build_postprocess_cells,
)


STATIC_ENDPOINTS = ("T0", "T1", "T2", "T3")
DELTA_ENDPOINTS = ("T0→T1", "T1→T2", "T2→T3")
SCOPES = ("primary_measurement_valid", "observable_only")


def _metric_values(
    truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(truth, baseline)))
    return {
        "spearman": float(spearmanr(truth, prediction).statistic),
        "pearson": float(pearsonr(truth, prediction).statistic),
        "r2": float(r2_score(truth, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, prediction)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / b0_rmse,
        "prediction_target_variance_ratio": float(np.var(prediction) / np.var(truth)),
    }


class SyntheticFormalStageB:
    """Small but structurally exact 40-cell formal matrix and postprocess tree."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.checkpoint_root = self.root / "checkpoints" / FORMAL_POSTPROCESS_TAG
        self.feature_root = self.root / "features" / FORMAL_POSTPROCESS_TAG
        self.probe_root = self.root / "predictions" / FORMAL_POSTPROCESS_TAG
        self.output_dir = self.root / "metrics"
        self.figure_dir = self.root / "figures"
        self.data_contract = (
            self.root / "manifests" / "stage_b_data_contract.private.json"
        )
        self.data_contract.parent.mkdir(parents=True, exist_ok=True)
        self.data_contract.write_text(
            json.dumps({"schema_version": 1, "synthetic": True}) + "\n",
            encoding="utf-8",
        )
        (self.root / "scripts").mkdir(parents=True, exist_ok=True)
        for name in ("aggregate_stage_b.py", "run_stage_b_postprocessing.py"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.data_contract_sha256 = file_sha256(self.data_contract)
        self.authorization = StageAAuthorization(
            self.root / "STAGE_A_GO.json",
            "a" * 64,
            {},
            10,
            40,
            "b" * 64,
        )
        self.cells = build_postprocess_cells(
            self.checkpoint_root, self.feature_root, self.probe_root
        )

    @staticmethod
    def _identity(seed: int, arm: str, fold: int) -> dict[str, object]:
        return {"arm": arm, "seed_base": seed, "fold": fold}

    @staticmethod
    def _arm_slope(arm: str, seed: int) -> float:
        return {
            "L1": 0.55,
            "L3": 0.70,
            "N1": 0.75,
            "N3": 0.95,
        }[arm] + (0.01 if seed == SEED_BASES[-1] else 0.0)

    def _endpoint_rows(
        self,
        *,
        seed: int,
        arm: str,
        fold: int,
        scope: str,
        task: str,
        endpoint: str,
        endpoint_index: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        patient_ids = (f"P{fold}_0", f"P{fold}_1")
        patient_offset = np.asarray([0.0, 1.0], dtype=np.float64)
        if task == "static":
            truth = 10.0 + 2.0 * fold + 3.0 * endpoint_index + patient_offset
            baseline = np.full(2, 12.0 + endpoint_index, dtype=np.float64)
            truth_analysis = np.log(truth + 0.5)
            analysis_baseline = np.log(baseline + 0.5)
            analysis_scale = "transformed_outer_train"
            semantics = "static_ftv_log_winsor_median_iqr_inverse_natural"
        else:
            truth = -3.0 + 0.7 * fold + 0.5 * endpoint_index + 0.4 * patient_offset
            baseline = np.zeros(2, dtype=np.float64)
            truth_analysis = (truth - 0.25) / 1.5
            analysis_baseline = (baseline - 0.25) / 1.5
            analysis_scale = "standardized_outer_train"
            semantics = "literal_ftv_end_minus_ftv_start"
        prediction = baseline + self._arm_slope(arm, seed) * (truth - baseline)
        prediction += np.asarray([-0.03, 0.03]) * (1 + ARMS.index(arm))
        if task == "static":
            prediction = np.maximum(prediction, 0.0)
            prediction_analysis = np.log(prediction + 0.5)
        else:
            prediction_analysis = (prediction - 0.25) / 1.5

        common = {
            **self._identity(seed, arm, fold),
            "task": task,
            "endpoint": endpoint,
            "analysis_scope": scope,
            "target_semantics": semantics,
            "selected_alpha": 1.0,
            "n_train": 12,
            "n_val": 4,
            "n_test": 2,
        }
        metric_rows = [
            {
                **common,
                "scale": "natural",
                **_metric_values(truth, prediction, baseline),
            },
            {
                **common,
                "scale": analysis_scale,
                **_metric_values(
                    truth_analysis, prediction_analysis, analysis_baseline
                ),
            },
        ]
        prediction_rows = [
            {
                "patient_id": patient_id,
                **common,
                "split": "test",
                "y_true": float(truth[index]),
                "y_pred": float(prediction[index]),
                "b0_prediction": float(baseline[index]),
                "y_true_analysis": float(truth_analysis[index]),
                "y_pred_analysis": float(prediction_analysis[index]),
                "b0_prediction_analysis": float(analysis_baseline[index]),
                "analysis_scale": analysis_scale,
                "test_predict_call_count": 1,
            }
            for index, patient_id in enumerate(patient_ids)
        ]
        return metric_rows, prediction_rows

    def _probe_frames(
        self, seed: int, arm: str, fold: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        metric_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []
        for scope in SCOPES:
            for task, endpoints in (
                ("static", STATIC_ENDPOINTS),
                ("delta", DELTA_ENDPOINTS),
            ):
                semantics = (
                    "static_ftv_log_winsor_median_iqr_inverse_natural"
                    if task == "static"
                    else "literal_ftv_end_minus_ftv_start"
                )
                for endpoint_index, endpoint in enumerate(endpoints):
                    metrics, predictions = self._endpoint_rows(
                        seed=seed,
                        arm=arm,
                        fold=fold,
                        scope=scope,
                        task=task,
                        endpoint=endpoint,
                        endpoint_index=endpoint_index,
                    )
                    metric_rows.extend(metrics)
                    prediction_rows.extend(predictions)
                    selection_rows.append(
                        {
                            **self._identity(seed, arm, fold),
                            "analysis_scope": scope,
                            "task": task,
                            "endpoint": endpoint,
                            "target_semantics": semantics,
                            "selected_alpha": 1.0,
                            "test_used_for_scaler": False,
                            "test_used_for_alpha_selection": False,
                            "test_predict_call_count": 1,
                        }
                    )
        metric_frame = pd.DataFrame(metric_rows)
        macros: list[dict[str, object]] = []
        for keys, group in metric_frame.groupby(
            ["analysis_scope", "task", "scale"], sort=False
        ):
            semantics = (
                "static_ftv_log_winsor_median_iqr_inverse_natural"
                if keys[1] == "static"
                else "literal_ftv_end_minus_ftv_start"
            )
            macros.append(
                {
                    **self._identity(seed, arm, fold),
                    "analysis_scope": keys[0],
                    "task": keys[1],
                    "scale": keys[2],
                    "endpoint": "macro",
                    "target_semantics": semantics,
                    "selected_alpha": math.nan,
                    "n_train": int(group["n_train"].sum()),
                    "n_val": int(group["n_val"].sum()),
                    "n_test": int(group["n_test"].sum()),
                    **{metric: float(group[metric].mean()) for metric in METRICS},
                }
            )
        return (
            pd.DataFrame(selection_rows),
            pd.DataFrame(prediction_rows),
            pd.concat([metric_frame, pd.DataFrame(macros)], ignore_index=True),
        )

    def _code_sha256(self) -> dict[str, str]:
        paths = {
            "postprocess_driver": ROOT / "scripts" / "run_stage_b_postprocessing.py",
            "feature_cli": ROOT / "scripts" / "export_stage_b_features.py",
            "probe_cli": ROOT / "scripts" / "run_stage_b_probes.py",
            "aggregate_cli": ROOT / "scripts" / "aggregate_stage_b.py",
            **{
                f"c1b_stage_b/{path.name}": path
                for path in sorted((ROOT / "src" / "c1b_stage_b").glob("*.py"))
            },
            "upstream_g3/model.py": G3_SRC / "dgrs" / "model.py",
            "upstream_g3/training.py": G3_SRC / "dgrs" / "training.py",
            "upstream_g3/targets.py": G3_SRC / "dgrs" / "targets.py",
        }
        return {name: file_sha256(path) for name, path in paths.items()}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def write(self) -> None:
        probe_metadata_hashes: dict[str, str] = {}
        feature_metadata_hashes: dict[str, str] = {}
        selection_history_sha256: dict[str, dict[str, str]] = {}
        matrix_runs: list[dict[str, object]] = []
        for cell in self.cells:
            identity = self._identity(cell.seed_base, cell.arm, cell.fold)
            cell.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            history = pd.DataFrame(
                [
                    {
                        **identity,
                        "epoch": 1,
                        "patient_order_sha256": f"order-{cell.seed_base}-{cell.fold}",
                        "dropped_logical_tail_patients": 0,
                        "train_optimizer_steps": 1,
                        "val_state_loss": 1.0,
                        "val_ftv_loss": 0.5 if cell.arm in {"L3", "N3"} else 0.0,
                    }
                ]
            )
            history.to_csv(cell.history_path, index=False)
            state = {"L1": 1.0, "L3": 1.04, "N1": 0.9, "N3": 0.918}[cell.arm]
            selection = {
                "schema_version": 1,
                "selection_mode": "primary",
                "experiment_pass": True,
                **identity,
                "effective_seed": cell.seed_base + cell.fold,
                "test_data_used": False,
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "global_fallback_restart": False,
                "finite_status": True,
                "selected_epoch": 1,
                "selected_validation_total_loss": state + 0.2,
                "selected_validation_base_loss": state + 0.1,
                "selected_validation_state_loss": state,
                "selected_validation_ftv_loss": (
                    0.5 if cell.arm in {"L3", "N3"} else 0.0
                ),
                "selected_representation_std": 0.2 + 0.01 * ARMS.index(cell.arm),
                "paired_initialization_sha256": f"init-{cell.seed_base}-{cell.fold}",
                "train_patient_sha256": f"train-{cell.fold}",
                "val_patient_sha256": f"val-{cell.fold}",
                "data_provenance_sha256": "c" * 64,
                "hyperparameters": dict(FORMAL_HYPERPARAMETERS),
                "history_sha256": file_sha256(cell.history_path),
            }
            self._write_json(cell.selection_path, selection)
            cell.checkpoint_path.write_bytes(
                f"selected-{cell.seed_base}-{cell.arm}-{cell.fold}".encode()
            )
            matrix_runs.append(
                {
                    **identity,
                    "selection_path": str(cell.selection_path),
                }
            )
            key = f"seed_{cell.seed_base}/{cell.arm}/fold_{cell.fold}"
            selection_history_sha256[key] = {
                "selection_sha256": file_sha256(cell.selection_path),
                "history_sha256": file_sha256(cell.history_path),
            }

            cell.feature_dir.mkdir(parents=True, exist_ok=True)
            cell.feature_path.write_bytes(
                f"feature-{cell.seed_base}-{cell.arm}-{cell.fold}".encode()
            )
            feature_metadata = {
                "schema_version": 1,
                "stage": "B",
                **identity,
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "feature_tensor": "online_preprojector_r",
                "ftv_head_called": False,
                "test_labels_used": False,
                "feature_path": str(cell.feature_path),
                "feature_sha256": file_sha256(cell.feature_path),
                "feature_shape": [10, 4, 192],
                "checkpoint_path": str(cell.checkpoint_path),
                "checkpoint_sha256": file_sha256(cell.checkpoint_path),
                "feature_implementation_sha256": file_sha256(
                    ROOT / "src" / "c1b_stage_b" / "features.py"
                ),
            }
            self._write_json(cell.feature_metadata_path, feature_metadata)
            feature_metadata_hashes[key] = file_sha256(cell.feature_metadata_path)

            cell.probe_dir.mkdir(parents=True, exist_ok=True)
            ridge, predictions, metrics = self._probe_frames(
                cell.seed_base, cell.arm, cell.fold
            )
            paths = {
                "ridge_selection.csv": cell.probe_dir / "ridge_selection.csv",
                "ridge_predictions.private.csv": (
                    cell.probe_dir / "ridge_predictions.private.csv"
                ),
                "probe_metrics.csv": cell.probe_dir / "probe_metrics.csv",
            }
            ridge.to_csv(paths["ridge_selection.csv"], index=False)
            predictions.to_csv(paths["ridge_predictions.private.csv"], index=False)
            metrics.to_csv(paths["probe_metrics.csv"], index=False)
            probe_metadata = {
                "schema_version": 1,
                **identity,
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "test_used_for_scaler_or_selection": False,
                "outer_test_predict_calls_per_cell": 1,
                "feature_asset_name": cell.feature_path.name,
                "feature_sha256": file_sha256(cell.feature_path),
                "feature_metadata_name": cell.feature_metadata_path.name,
                "feature_metadata_sha256": file_sha256(cell.feature_metadata_path),
                "probe_implementation_sha256": file_sha256(
                    ROOT / "src" / "c1b_stage_b" / "probes.py"
                ),
                "target_adapter_sha256": file_sha256(
                    ROOT / "src" / "c1b_stage_b" / "targets.py"
                ),
                "output_sha256": {
                    name: file_sha256(path) for name, path in paths.items()
                },
            }
            self._write_json(cell.probe_metadata_path, probe_metadata)
            probe_metadata_hashes[key] = file_sha256(cell.probe_metadata_path)

        matrix_completion_path = self.checkpoint_root / "matrix_complete.json"
        self._write_json(
            matrix_completion_path,
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "run_count": 40,
                "devices": list(FORMAL_DEVICES),
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "batch_contract": {
                    "effective": 32,
                    "physical": 4,
                    "accumulation": 8,
                    "global_for_all_arms": True,
                },
                "runs": matrix_runs,
            },
        )
        code_sha256 = self._code_sha256()
        selection_history_inventory_sha256 = canonical_sha256(
            selection_history_sha256
        )
        claim_path = self.feature_root / "postprocessing_claim.json"
        self._write_json(
            claim_path,
            {
                "schema_version": 1,
                "status": "CLAIMED",
                "formal_tag": FORMAL_POSTPROCESS_TAG,
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "data_contract_path": str(self.data_contract),
                "data_contract_sha256": self.data_contract_sha256,
                "matrix_complete_sha256": file_sha256(matrix_completion_path),
                "selection_history_inventory_sha256": (
                    selection_history_inventory_sha256
                ),
                "postprocess_driver_sha256": code_sha256["postprocess_driver"],
                "nonresumable": True,
            },
        )
        preflight_path = self.feature_root / "postprocessing_preflight.json"
        self._write_json(
            preflight_path,
            {
                "schema_version": 1,
                "status": "PREFLIGHT_PASS",
                "formal_tag": FORMAL_POSTPROCESS_TAG,
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "data_contract_path": str(self.data_contract),
                "data_contract_sha256": self.data_contract_sha256,
                "claim_sha256": file_sha256(claim_path),
                "paths": {
                    "checkpoint_root": str(self.checkpoint_root),
                    "feature_root": str(self.feature_root),
                    "probe_root": str(self.probe_root),
                },
                "matrix": {
                    "matrix_complete_sha256": file_sha256(matrix_completion_path),
                    "run_count": 40,
                    "batch_contract": {
                        "effective": 32,
                        "physical": 4,
                        "accumulation": 8,
                        "global_for_all_arms": True,
                    },
                },
                "selection_history_sha256": selection_history_sha256,
                "selection_history_inventory_sha256": (
                    selection_history_inventory_sha256
                ),
                "scheduler": {"run_count": 40},
                "cell_inventory": [
                    {
                        "index": cell.index,
                        "seed_base": cell.seed_base,
                        "fold": cell.fold,
                        "arm": cell.arm,
                        "feature_device": cell.device,
                    }
                    for cell in self.cells
                ],
                "code_sha256": code_sha256,
                "execution_requested": True,
            },
        )
        feature_completion_path = self.feature_root / "feature_export_complete.json"
        self._write_json(
            feature_completion_path,
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "formal_tag": FORMAL_POSTPROCESS_TAG,
                "run_count": 40,
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "data_contract_sha256": self.data_contract_sha256,
                "matrix_complete_sha256": file_sha256(matrix_completion_path),
                "claim_sha256": file_sha256(claim_path),
                "preflight_sha256": file_sha256(preflight_path),
                "selection_history_sha256": selection_history_sha256,
                "selection_history_inventory_sha256": (
                    selection_history_inventory_sha256
                ),
                "feature_metadata_sha256": feature_metadata_hashes,
            },
        )
        self._write_json(
            self.probe_root / "postprocessing_complete.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "formal_tag": FORMAL_POSTPROCESS_TAG,
                "run_count": 40,
                "stage_a_sentinel_sha256": self.authorization.sha256,
                "data_contract_sha256": self.data_contract_sha256,
                "matrix_complete_sha256": file_sha256(matrix_completion_path),
                "claim_sha256": file_sha256(claim_path),
                "preflight_sha256": file_sha256(preflight_path),
                "feature_export_complete_sha256": file_sha256(
                    feature_completion_path
                ),
                "selection_history_sha256": selection_history_sha256,
                "selection_history_inventory_sha256": (
                    selection_history_inventory_sha256
                ),
                "probe_metadata_sha256": probe_metadata_hashes,
                "code_sha256": code_sha256,
            },
        )


class StageBAggregationTests(unittest.TestCase):
    @staticmethod
    def _validate_formal(fixture: SyntheticFormalStageB) -> dict[str, object]:
        with mock.patch.object(analysis, "EXPERIMENT_ROOT", fixture.root):
            return validate_formal_aggregation_inputs(
                checkpoint_root=fixture.checkpoint_root,
                feature_root=fixture.feature_root,
                probe_root=fixture.probe_root,
                authorization=fixture.authorization,
                data_contract=fixture.data_contract,
                data_contract_sha256=fixture.data_contract_sha256,
            )

    def test_formal_completion_and_probe_metadata_hash_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFormalStageB(Path(directory))
            fixture.write()
            result = self._validate_formal(fixture)
            self.assertEqual(len(result["probe_metadata_sha256"]), 40)

            completion_path = fixture.probe_root / "postprocessing_complete.json"
            original_completion = completion_path.read_text(encoding="utf-8")
            completion = json.loads(original_completion)
            first_key = sorted(completion["probe_metadata_sha256"])[0]
            completion["probe_metadata_sha256"][first_key] = "0" * 64
            completion_path.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "completion|metadata|SHA|hash"):
                self._validate_formal(fixture)
            completion_path.write_text(original_completion, encoding="utf-8")

            metadata_path = fixture.cells[0].probe_metadata_path
            metadata_path.write_text(
                metadata_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "completion|metadata|SHA|hash"):
                self._validate_formal(fixture)

    def test_training_selection_tampering_fails_the_completion_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFormalStageB(Path(directory))
            fixture.write()
            self._validate_formal(fixture)

            selection_path = fixture.cells[0].selection_path
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["selected_validation_state_loss"] = 123.0
            fixture._write_json(selection_path, selection)

            with self.assertRaisesRegex(
                ValueError, "selection_sha256|selection/history|training"
            ):
                self._validate_formal(fixture)

    def test_history_tampering_with_synchronized_selection_hash_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFormalStageB(Path(directory))
            fixture.write()
            self._validate_formal(fixture)

            cell = fixture.cells[0]
            history = pd.read_csv(cell.history_path)
            history.loc[0, "val_state_loss"] = 999.0
            history.to_csv(cell.history_path, index=False)
            selection = json.loads(cell.selection_path.read_text(encoding="utf-8"))
            selection["history_sha256"] = file_sha256(cell.history_path)
            fixture._write_json(cell.selection_path, selection)
            self.assertEqual(
                json.loads(cell.selection_path.read_text(encoding="utf-8"))[
                    "history_sha256"
                ],
                file_sha256(cell.history_path),
            )

            with self.assertRaisesRegex(
                ValueError, "selection_sha256|history_sha256|selection/history|training"
            ):
                self._validate_formal(fixture)

    def test_failed_figure_staging_publishes_nothing_and_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFormalStageB(Path(directory))
            fixture.write()
            with (
                mock.patch.object(analysis, "EXPERIMENT_ROOT", fixture.root),
                mock.patch.object(
                    analysis,
                    "make_stage_b_figures",
                    side_effect=RuntimeError("synthetic figure failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic figure failure"):
                    aggregate_stage_b(
                        checkpoint_root=fixture.checkpoint_root,
                        feature_root=fixture.feature_root,
                        probe_root=fixture.probe_root,
                        output_dir=fixture.output_dir,
                        figure_dir=fixture.figure_dir,
                        authorization=fixture.authorization,
                        data_contract=fixture.data_contract,
                        data_contract_sha256=fixture.data_contract_sha256,
                    )

            claim_path = fixture.output_dir / "stage_b_aggregation_claim.json"
            final_tables = {
                "table2_static_ftv.csv",
                "table3_literal_observed_delta_ftv.csv",
                "table4_optimization_safety.csv",
                "table5_difference_in_differences.csv",
                "table5_fold_level_sensitivity.csv",
            }
            self.assertTrue(claim_path.is_file())
            self.assertFalse(
                any((fixture.output_dir / name).exists() for name in final_tables)
            )
            self.assertFalse(
                (fixture.output_dir / "stage_b_aggregation_summary.json").exists()
            )
            self.assertFalse(
                any((fixture.figure_dir / name).exists() for name in FIGURE_NAMES)
            )
            self.assertEqual(
                list(fixture.output_dir.glob(".stage_b_aggregation_*")), []
            )
            self.assertEqual(
                list(fixture.figure_dir.glob(".stage_b_aggregation_*")), []
            )
            original_claim = claim_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "already claimed"):
                _claim_formal_aggregation(claim_path, {"status": "SECOND"})
            self.assertEqual(claim_path.read_bytes(), original_claim)

    def test_exact_matrix_metrics_are_pooled_per_seed_and_did_is_seed_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFormalStageB(Path(directory))
            fixture.write()
            selections, metrics, histories, predictions = collect_complete_matrix(
                fixture.checkpoint_root,
                fixture.probe_root,
                fixture.authorization,
            )
            self.assertEqual(len(selections), 40)
            self.assertEqual(len(metrics), 1_440)
            self.assertEqual(len(predictions), 1_120)
            _audit_four_arm_matrix_contract(selections, metrics, histories)
            _audit_oof_prediction_contract(metrics, predictions)
            pooled = pooled_oof_metrics(predictions)
            self.assertEqual(len(pooled), 144)
            row = pooled.loc[
                pooled["seed_base"].eq(SEED_BASES[0])
                & pooled["arm"].eq("N3")
                & pooled["analysis_scope"].eq("primary_measurement_valid")
                & pooled["task"].eq("static")
                & pooled["endpoint"].eq("T0")
            ].iloc[0]
            raw = predictions.loc[
                predictions["seed_base"].eq(SEED_BASES[0])
                & predictions["arm"].eq("N3")
                & predictions["analysis_scope"].eq("primary_measurement_valid")
                & predictions["task"].eq("static")
                & predictions["endpoint"].eq("T0")
            ]
            truth = raw["y_true"].to_numpy(float)
            prediction = raw["y_pred"].to_numpy(float)
            baseline = raw["b0_prediction"].to_numpy(float)
            self.assertAlmostEqual(float(row["r2"]), r2_score(truth, prediction))
            self.assertAlmostEqual(
                float(row["rmse"]), math.sqrt(mean_squared_error(truth, prediction))
            )
            expected_gain = 1.0 - math.sqrt(
                mean_squared_error(truth, prediction)
            ) / math.sqrt(mean_squared_error(truth, baseline))
            self.assertAlmostEqual(float(row["rmse_gain_over_b0"]), expected_gain)
            self.assertAlmostEqual(
                float(row["prediction_target_variance_ratio"]),
                float(np.var(prediction) / np.var(truth)),
            )
            expected_slope = float(
                np.mean((truth - np.mean(truth)) * (prediction - np.mean(prediction)))
                / np.var(truth)
            )
            expected_intercept = float(
                np.mean(prediction) - expected_slope * np.mean(truth)
            )
            self.assertAlmostEqual(float(row["calibration_slope"]), expected_slope)
            self.assertAlmostEqual(
                float(row["calibration_intercept"]), expected_intercept
            )
            self.assertAlmostEqual(
                float(row["calibration_mean_bias"]), float(np.mean(prediction - truth))
            )
            effects = paired_effects(pooled)
            primary = effects.loc[
                effects["task"].eq("static")
                & effects["endpoint"].eq("macro")
                & effects["metric"].eq("r2")
            ]
            self.assertEqual(set(primary["seed_base"]), set(SEED_BASES))
            self.assertNotIn("fold", primary.columns)
            for effect in primary.itertuples(index=False):
                current = pooled.loc[
                    pooled["seed_base"].eq(effect.seed_base)
                    & pooled["analysis_scope"].eq("primary_measurement_valid")
                    & pooled["task"].eq("static")
                    & pooled["endpoint"].eq("macro")
                ].set_index("arm")
                expected = (
                    float(current.loc["N3", "r2"])
                    - float(current.loc["N1", "r2"])
                    - float(current.loc["L3", "r2"])
                    + float(current.loc["L1", "r2"])
                )
                self.assertAlmostEqual(effect.difference_in_differences, expected)

    def test_whole_task_fold_or_seed_omission_fails_the_exact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFormalStageB(Path(directory))
            fixture.write()
            selections, metrics, histories, predictions = collect_complete_matrix(
                fixture.checkpoint_root,
                fixture.probe_root,
                fixture.authorization,
            )

            cases = {
                "task": (
                    metrics.loc[metrics["task"].ne("delta")],
                    predictions.loc[predictions["task"].ne("delta")],
                ),
                "fold": (
                    metrics.loc[metrics["fold"].ne(FOLDS[-1])],
                    predictions.loc[predictions["fold"].ne(FOLDS[-1])],
                ),
                "seed": (
                    metrics.loc[metrics["seed_base"].ne(SEED_BASES[-1])],
                    predictions.loc[
                        predictions["seed_base"].ne(SEED_BASES[-1])
                    ],
                ),
            }
            for label, (current_metrics, current_predictions) in cases.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        ValueError, "formal Cartesian grid|coverage|inventory|complete"
                    ):
                        _audit_four_arm_matrix_contract(
                            selections, current_metrics, histories
                        )
                        _audit_oof_prediction_contract(
                            current_metrics, current_predictions
                        )

    def test_optimization_safety_includes_exact_five_percent_boundary(self) -> None:
        rows: list[dict[str, object]] = []
        for seed in SEED_BASES:
            for fold in FOLDS:
                for arm in ARMS:
                    state = {
                        "L1": 1.0,
                        "L3": 1.05,
                        "N1": 1.0,
                        "N3": 1.050001,
                    }[arm]
                    rows.append(
                        {
                            "arm": arm,
                            "seed_base": seed,
                            "fold": fold,
                            "selected_epoch": 1,
                            "selected_validation_total_loss": state + 0.2,
                            "selected_validation_base_loss": state + 0.1,
                            "selected_validation_state_loss": state,
                            "selected_validation_ftv_loss": 0.5,
                            "selected_representation_std": 0.2,
                            "selection_mode": "primary",
                            "finite_status": True,
                            "experiment_pass": arm != "N3",
                        }
                    )
        table = optimization_table(pd.DataFrame(rows))
        self.assertTrue(table.loc[table["arm"].eq("L3"), "optimization_safety_pass"].all())
        self.assertTrue(
            np.allclose(
                table.loc[table["arm"].eq("L3"), "base_degradation_fraction"],
                0.05,
            )
        )
        self.assertFalse(
            table.loc[table["arm"].eq("N3"), "optimization_safety_pass"].any()
        )
        self.assertTrue(
            table.loc[table["arm"].eq("N3"), "state_loss_degradation_gt_5pct"].all()
        )

    def test_full_forty_cell_aggregate_has_exact_tables_and_nine_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFormalStageB(Path(directory))
            fixture.write()
            _, _, _, expected_predictions = collect_complete_matrix(
                fixture.checkpoint_root,
                fixture.probe_root,
                fixture.authorization,
            )
            from matplotlib.axes import Axes

            original_scatter = Axes.scatter
            figure6_sources: list[tuple[str, np.ndarray, np.ndarray]] = []

            def recording_scatter(axis, x, y, *args, **kwargs):
                x_values = np.asarray(x)
                y_values = np.asarray(y)
                label = kwargs.get("label")
                if (
                    isinstance(label, str)
                    and label.startswith("figure6:")
                    and x_values.ndim == y_values.ndim == 1
                    and len(x_values) == len(y_values)
                ):
                    _, arm, _, _ = label.split(":", maxsplit=3)
                    figure6_sources.append(
                        (
                            arm,
                            x_values.astype(float, copy=True),
                            y_values.astype(float, copy=True),
                        )
                    )
                return original_scatter(axis, x, y, *args, **kwargs)

            with (
                mock.patch.object(analysis, "EXPERIMENT_ROOT", fixture.root),
                mock.patch.object(Axes, "scatter", recording_scatter),
            ):
                summary = aggregate_stage_b(
                    checkpoint_root=fixture.checkpoint_root,
                    feature_root=fixture.feature_root,
                    probe_root=fixture.probe_root,
                    output_dir=fixture.output_dir,
                    figure_dir=fixture.figure_dir,
                    authorization=fixture.authorization,
                    data_contract=fixture.data_contract,
                    data_contract_sha256=fixture.data_contract_sha256,
                )
            self.assertEqual(summary["run_count"], 40)
            table2 = pd.read_csv(fixture.output_dir / "table2_static_ftv.csv")
            table3 = pd.read_csv(
                fixture.output_dir / "table3_literal_observed_delta_ftv.csv"
            )
            table4 = pd.read_csv(fixture.output_dir / "table4_optimization_safety.csv")
            table5 = pd.read_csv(
                fixture.output_dir / "table5_difference_in_differences.csv"
            )
            fold_effects = pd.read_csv(
                fixture.output_dir / "table5_fold_level_sensitivity.csv"
            )
            self.assertEqual(len(table2), 440)
            self.assertEqual(len(table3), 32)
            self.assertEqual(len(table4), 40)
            self.assertEqual(len(table5), 10)
            self.assertEqual(len(fold_effects), 720)
            self.assertEqual(
                set(table2.columns),
                {
                    "arm", "seed_base", "fold", "task", "endpoint",
                    "analysis_scope", "target_semantics", "selected_alpha",
                    "n_train", "n_val", "n_test", "scale", "aggregation",
                    "calibration_slope", "calibration_intercept", "calibration_mean_bias",
                    *METRICS,
                },
            )
            self.assertEqual(
                set(table3.columns),
                {
                    "seed_base", "arm", "task", "endpoint", "analysis_scope",
                    "target_semantics", "scale", "aggregation", "n_test",
                    "calibration_slope", "calibration_intercept", "calibration_mean_bias",
                    *METRICS,
                },
            )
            self.assertEqual(
                set(table4.columns),
                {
                    "arm", "seed_base", "fold", "selected_epoch",
                    "selected_validation_total_loss", "selected_validation_base_loss",
                    "selected_validation_state_loss", "paired_baseline_arm",
                    "paired_baseline_state_loss", "state_loss_degradation_fraction",
                    "base_degradation_fraction", "state_loss_degradation_gt_5pct",
                    "selected_validation_ftv_loss", "selected_representation_std",
                    "selection_mode", "finite", "optimization_safety_pass",
                },
            )
            self.assertEqual(
                set(table5.columns),
                {
                    "seed_base", "task", "endpoint", "target_semantics", "metric",
                    "L3_minus_L1", "N3_minus_N1", "N1_minus_L1",
                    "difference_in_differences", "fold_aggregation",
                },
            )
            self.assertEqual(
                tuple(sorted(path.name for path in fixture.figure_dir.iterdir())),
                tuple(sorted(FIGURE_NAMES)),
            )
            self.assertTrue(
                all((fixture.figure_dir / name).stat().st_size > 0 for name in FIGURE_NAMES)
            )
            for arm in ARMS:
                expected = expected_predictions.loc[
                    expected_predictions["arm"].eq(arm)
                    & expected_predictions["task"].eq("static")
                    & expected_predictions["analysis_scope"].eq(
                        "primary_measurement_valid"
                    ),
                    ["y_true", "y_pred"],
                ].to_numpy(float)
                expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
                parts = [
                    np.column_stack((x_values, y_values))
                    for source_arm, x_values, y_values in figure6_sources
                    if source_arm == arm
                ]
                self.assertTrue(parts, f"Figure 6 emitted no scatter rows for {arm}")
                observed = np.concatenate(parts)
                observed = observed[np.lexsort((observed[:, 1], observed[:, 0]))]
                self.assertEqual(len(observed), len(expected))
                self.assertTrue(
                    np.allclose(observed, expected),
                    f"Figure 6 did not source natural predicted-vs-true rows for {arm}",
                )
            table_hashes = summary["table_sha256"]
            expected_tables = {
                "table2_static_ftv.csv",
                "table3_literal_observed_delta_ftv.csv",
                "table4_optimization_safety.csv",
                "table5_difference_in_differences.csv",
                "table5_fold_level_sensitivity.csv",
            }
            self.assertEqual(set(table_hashes), expected_tables)
            for name, digest in table_hashes.items():
                self.assertEqual(digest, file_sha256(fixture.output_dir / name))


if __name__ == "__main__":
    unittest.main()
