"""Complete-matrix aggregation, paired effects, DiD, and Stage B figures."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .contracts import (
    ARMS,
    EXPERIMENT_ROOT,
    FOLDS,
    SEED_BASES,
    canonical_sha256,
    file_sha256,
    require_sha256,
)
from .gate import StageAAuthorization
from .postprocess import FORMAL_DEVICES, FORMAL_POSTPROCESS_TAG


METRICS = (
    "spearman",
    "pearson",
    "r2",
    "rmse",
    "mae",
    "b0_rmse",
    "rmse_gain_over_b0",
    "prediction_target_variance_ratio",
)

CALIBRATION_METRICS = (
    "calibration_slope",
    "calibration_intercept",
    "calibration_mean_bias",
)

ANALYSIS_SCOPES = (
    "primary_measurement_valid",
    "observable_only",
)

TASK_CONTRACT = {
    "static": {
        "endpoints": ("T0", "T1", "T2", "T3"),
        "scales": ("natural", "transformed_outer_train"),
        "target_semantics": "static_ftv_log_winsor_median_iqr_inverse_natural",
    },
    "delta": {
        "endpoints": ("T0→T1", "T1→T2", "T2→T3"),
        "scales": ("natural", "standardized_outer_train"),
        "target_semantics": "literal_ftv_end_minus_ftv_start",
    },
}

FIGURE_NAMES = (
    "04_static_ftv_spearman.png",
    "05_static_ftv_natural_r2.png",
    "06_static_ftv_predicted_vs_true_natural.png",
    "07_literal_delta_ftv_spearman.png",
    "08_literal_delta_ftv_natural_r2.png",
    "09_state_loss_degradation_heatmap.png",
    "10_representation_std.png",
    "11_grounding_difference_in_differences.png",
    "12_representative_training_curves.png",
)

TABLE_FILENAMES = {
    "static": "table2_static_ftv.csv",
    "delta": "table3_literal_observed_delta_ftv.csv",
    "optimization": "table4_optimization_safety.csv",
    "effects": "table5_difference_in_differences.csv",
    "fold_effects": "table5_fold_level_sensitivity.csv",
}

TABLE_COLUMNS = {
    "static": (
        "seed_base", "fold", "arm", "analysis_scope", "task", "endpoint",
        "target_semantics", "scale", "aggregation", "selected_alpha",
        "n_train", "n_val", "n_test", *METRICS, *CALIBRATION_METRICS,
    ),
    "delta": (
        "seed_base", "arm", "analysis_scope", "task", "endpoint",
        "target_semantics", "scale", "aggregation", "n_test", *METRICS,
        *CALIBRATION_METRICS,
    ),
    "optimization": (
        "arm", "seed_base", "fold", "selected_epoch",
        "selected_validation_total_loss", "selected_validation_base_loss",
        "selected_validation_state_loss", "paired_baseline_arm",
        "paired_baseline_state_loss", "state_loss_degradation_fraction",
        "base_degradation_fraction", "state_loss_degradation_gt_5pct",
        "selected_validation_ftv_loss", "selected_representation_std",
        "selection_mode", "finite", "optimization_safety_pass",
    ),
    "effects": (
        "seed_base", "fold_aggregation", "task", "endpoint",
        "target_semantics", "metric", "L3_minus_L1", "N3_minus_N1",
        "N1_minus_L1", "difference_in_differences",
    ),
    "fold_effects": (
        "seed_base", "fold", "task", "endpoint", "target_semantics",
        "metric", "L3_minus_L1", "N3_minus_N1", "N1_minus_L1",
        "difference_in_differences",
    ),
}

TABLE_ROW_COUNTS = {
    "static": 440,
    "delta": 32,
    "optimization": 40,
    "effects": 10,
    "fold_effects": 720,
}

AGGREGATION_CLAIM_NAME = "stage_b_aggregation_claim.json"
AGGREGATION_SUMMARY_NAME = "stage_b_aggregation_summary.json"


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _require_contract_fields(
    payload: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    drifted = [field for field, value in expected.items() if payload.get(field) != value]
    if drifted:
        raise ValueError(f"{label} contract drifted at: {', '.join(sorted(drifted))}")


def _formal_roots() -> tuple[Path, Path, Path]:
    tag = FORMAL_POSTPROCESS_TAG
    return (
        (EXPERIMENT_ROOT / "checkpoints" / tag).resolve(),
        (EXPERIMENT_ROOT / "features" / tag).resolve(),
        (EXPERIMENT_ROOT / "predictions" / tag).resolve(),
    )


def _require_exact_formal_roots(
    checkpoint_root: str | Path,
    feature_root: str | Path,
    probe_root: str | Path,
) -> tuple[Path, Path, Path]:
    observed = tuple(
        Path(path).resolve() for path in (checkpoint_root, feature_root, probe_root)
    )
    expected = _formal_roots()
    if observed != expected:
        raise ValueError(
            "formal aggregation inputs must be the exact checkpoints/features/"
            f"predictions/{FORMAL_POSTPROCESS_TAG} roots"
        )
    return observed


def _require_exact_formal_output_roots(
    output_dir: str | Path, figure_dir: str | Path
) -> tuple[Path, Path]:
    observed = (Path(output_dir).resolve(), Path(figure_dir).resolve())
    expected = (
        (EXPERIMENT_ROOT / "metrics").resolve(),
        (EXPERIMENT_ROOT / "figures").resolve(),
    )
    if observed != expected:
        raise ValueError("formal aggregation outputs must be the experiment metrics/figures roots")
    return observed


def _cell_key(seed: int, arm: str, fold: int) -> str:
    return f"seed_{seed}/{arm}/fold_{fold}"


def _expected_cell_keys() -> set[str]:
    return {
        _cell_key(seed, arm, fold)
        for seed in SEED_BASES
        for arm in ARMS
        for fold in FOLDS
    }


def _require_hash_inventory(
    payload: Any,
    expected_paths: Mapping[str, Path],
    label: str,
) -> dict[str, str]:
    if not isinstance(payload, Mapping) or set(payload) != set(expected_paths):
        raise ValueError(
            f"{label} inventory does not match the exact required artifacts"
        )
    observed: dict[str, str] = {}
    for key, path in expected_paths.items():
        expected = require_sha256(str(payload[key]), f"{label} {key}")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{label} hash drifted for {key}")
        observed[key] = expected
    return observed


def _require_selection_history_inventory(
    payload: Any,
    checkpoint_root: Path,
) -> dict[str, dict[str, str]]:
    expected_keys = _expected_cell_keys()
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError(
            "selection/history SHA inventory is not the exact formal 40-cell inventory"
        )
    observed: dict[str, dict[str, str]] = {}
    for seed in SEED_BASES:
        for arm in ARMS:
            for fold in FOLDS:
                key = _cell_key(seed, arm, fold)
                row = payload[key]
                if not isinstance(row, Mapping) or set(row) != {
                    "selection_sha256",
                    "history_sha256",
                }:
                    raise ValueError(f"selection/history SHA row drifted for {key}")
                paths = {
                    "selection_sha256": (
                        _cell_path(checkpoint_root, seed, arm, fold) / "selection.json"
                    ),
                    "history_sha256": (
                        _cell_path(checkpoint_root, seed, arm, fold) / "history.csv"
                    ),
                }
                observed[key] = {}
                for field, path in paths.items():
                    expected = require_sha256(str(row[field]), f"{key} {field}")
                    if not path.is_file() or file_sha256(path) != expected:
                        raise ValueError(f"training {field} drifted for {key}")
                    observed[key][field] = expected
    return observed


def _expected_postprocess_cell_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (seed, fold, arm) in enumerate(
        (seed, fold, arm)
        for seed in SEED_BASES
        for fold in FOLDS
        for arm in ARMS
    ):
        rows.append(
            {
                "index": index,
                "seed_base": seed,
                "fold": fold,
                "arm": arm,
                "feature_device": FORMAL_DEVICES[index % len(FORMAL_DEVICES)],
            }
        )
    return rows


def validate_formal_aggregation_inputs(
    *,
    checkpoint_root: str | Path,
    feature_root: str | Path,
    probe_root: str | Path,
    authorization: StageAAuthorization,
    data_contract: str | Path,
    data_contract_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable matrix -> feature -> probe completion chain."""

    checkpoints, features, probes = _require_exact_formal_roots(
        checkpoint_root, feature_root, probe_root
    )
    contract_path = Path(data_contract).resolve()
    contract_sha256 = require_sha256(data_contract_sha256, "Stage B data contract")
    if not contract_path.is_file() or file_sha256(contract_path) != contract_sha256:
        raise ValueError("Stage B data-contract SHA-256 mismatch")

    matrix_path = checkpoints / "matrix_complete.json"
    matrix = _read_json_object(matrix_path, "matrix completion")
    _require_contract_fields(
        matrix,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "run_count": 40,
            "stage_a_sentinel_sha256": authorization.sha256,
            "devices": list(FORMAL_DEVICES),
        },
        "matrix completion",
    )
    batch = matrix.get("batch_contract")
    if not isinstance(batch, Mapping):
        raise ValueError("matrix completion batch contract is missing")
    _require_contract_fields(
        batch,
        {
            "effective": 32,
            "physical": 4,
            "accumulation": 8,
            "global_for_all_arms": True,
        },
        "matrix batch",
    )
    runs = matrix.get("runs")
    if not isinstance(runs, list) or len(runs) != 40:
        raise ValueError("matrix completion run inventory is not exactly 40 rows")
    observed_runs: dict[tuple[int, int, str], Path] = {}
    for row in runs:
        if not isinstance(row, Mapping):
            raise ValueError("matrix completion contains a non-object run row")
        key = (
            int(row.get("seed_base", -1)),
            int(row.get("fold", -1)),
            str(row.get("arm", "")),
        )
        if key in observed_runs:
            raise ValueError("matrix completion contains a duplicate run identity")
        observed_runs[key] = Path(str(row.get("selection_path", ""))).resolve()
    expected_runs = {
        (seed, fold, arm): _cell_path(checkpoints, seed, arm, fold) / "selection.json"
        for seed in SEED_BASES
        for fold in FOLDS
        for arm in ARMS
    }
    if observed_runs != expected_runs:
        raise ValueError("matrix completion is not the exact formal 40-cell inventory")
    matrix_sha256 = file_sha256(matrix_path)

    postprocess_path = probes / "postprocessing_complete.json"
    postprocess = _read_json_object(postprocess_path, "postprocessing completion")
    selection_history_sha256 = _require_selection_history_inventory(
        postprocess.get("selection_history_sha256"), checkpoints
    )
    selection_history_inventory_sha256 = canonical_sha256(
        selection_history_sha256
    )
    _require_contract_fields(
        postprocess,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "formal_tag": FORMAL_POSTPROCESS_TAG,
            "run_count": 40,
            "stage_a_sentinel_sha256": authorization.sha256,
            "data_contract_sha256": contract_sha256,
            "matrix_complete_sha256": matrix_sha256,
            "selection_history_inventory_sha256": (
                selection_history_inventory_sha256
            ),
        },
        "postprocessing completion",
    )

    claim_path = features / "postprocessing_claim.json"
    preflight_path = features / "postprocessing_preflight.json"
    feature_completion_path = features / "feature_export_complete.json"
    linked_paths = {
        "claim_sha256": claim_path,
        "preflight_sha256": preflight_path,
        "feature_export_complete_sha256": feature_completion_path,
    }
    linked_hashes: dict[str, str] = {}
    for field, path in linked_paths.items():
        expected = require_sha256(str(postprocess.get(field, "")), field)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"postprocessing completion chain drifted at {field}")
        linked_hashes[field] = expected

    claim = _read_json_object(claim_path, "postprocessing claim")
    _require_contract_fields(
        claim,
        {
            "schema_version": 1,
            "status": "CLAIMED",
            "formal_tag": FORMAL_POSTPROCESS_TAG,
            "stage_a_sentinel_sha256": authorization.sha256,
            "data_contract_sha256": contract_sha256,
            "matrix_complete_sha256": matrix_sha256,
            "selection_history_inventory_sha256": (
                selection_history_inventory_sha256
            ),
            "nonresumable": True,
        },
        "postprocessing claim",
    )

    preflight = _read_json_object(preflight_path, "postprocessing preflight")
    _require_contract_fields(
        preflight,
        {
            "schema_version": 1,
            "status": "PREFLIGHT_PASS",
            "formal_tag": FORMAL_POSTPROCESS_TAG,
            "stage_a_sentinel_sha256": authorization.sha256,
            "data_contract_path": str(contract_path),
            "data_contract_sha256": contract_sha256,
            "claim_sha256": linked_hashes["claim_sha256"],
            "execution_requested": True,
            "cell_inventory": _expected_postprocess_cell_inventory(),
            "selection_history_sha256": selection_history_sha256,
            "selection_history_inventory_sha256": (
                selection_history_inventory_sha256
            ),
        },
        "postprocessing preflight",
    )
    if preflight.get("paths") != {
        "checkpoint_root": str(checkpoints),
        "feature_root": str(features),
        "probe_root": str(probes),
    }:
        raise ValueError("postprocessing preflight formal roots drifted")
    expected_matrix_evidence = {
        "matrix_complete_sha256": matrix_sha256,
        "run_count": 40,
        "batch_contract": dict(batch),
    }
    if preflight.get("matrix") != expected_matrix_evidence:
        raise ValueError("postprocessing preflight matrix evidence drifted")

    feature_completion = _read_json_object(
        feature_completion_path, "feature-export completion"
    )
    _require_contract_fields(
        feature_completion,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "formal_tag": FORMAL_POSTPROCESS_TAG,
            "run_count": 40,
            "stage_a_sentinel_sha256": authorization.sha256,
            "data_contract_sha256": contract_sha256,
            "matrix_complete_sha256": matrix_sha256,
            "claim_sha256": linked_hashes["claim_sha256"],
            "preflight_sha256": linked_hashes["preflight_sha256"],
            "selection_history_sha256": selection_history_sha256,
            "selection_history_inventory_sha256": (
                selection_history_inventory_sha256
            ),
        },
        "feature-export completion",
    )

    feature_metadata_paths = {
        _cell_key(seed, arm, fold): (
            _cell_path(features, seed, arm, fold)
            / "response_state.private.metadata.json"
        )
        for seed in SEED_BASES
        for arm in ARMS
        for fold in FOLDS
    }
    probe_metadata_paths = {
        _cell_key(seed, arm, fold): (
            _cell_path(probes, seed, arm, fold) / "probe_metadata.json"
        )
        for seed in SEED_BASES
        for arm in ARMS
        for fold in FOLDS
    }
    feature_metadata_hashes = _require_hash_inventory(
        feature_completion.get("feature_metadata_sha256"),
        feature_metadata_paths,
        "feature metadata",
    )
    probe_metadata_hashes = _require_hash_inventory(
        postprocess.get("probe_metadata_sha256"),
        probe_metadata_paths,
        "probe metadata",
    )
    code_sha256 = postprocess.get("code_sha256")
    if not isinstance(code_sha256, Mapping) or preflight.get("code_sha256") != code_sha256:
        raise ValueError("postprocessing code inventory is missing or differs from preflight")
    for required_code in (
        "postprocess_driver",
        "aggregate_cli",
        "c1b_stage_b/analysis.py",
        "c1b_stage_b/features.py",
        "c1b_stage_b/probes.py",
        "c1b_stage_b/targets.py",
    ):
        require_sha256(str(code_sha256.get(required_code, "")), required_code)

    for seed in SEED_BASES:
        for arm in ARMS:
            for fold in FOLDS:
                key = _cell_key(seed, arm, fold)
                feature_dir = _cell_path(features, seed, arm, fold)
                probe_dir = _cell_path(probes, seed, arm, fold)
                feature_asset = feature_dir / "response_state.private.npz"
                feature_metadata_path = feature_metadata_paths[key]
                checkpoint_path = (
                    _cell_path(checkpoints, seed, arm, fold) / "selected.pt"
                )
                feature_metadata = _read_json_object(
                    feature_metadata_path, f"feature metadata {key}"
                )
                _require_contract_fields(
                    feature_metadata,
                    {
                        "schema_version": 1,
                        "stage": "B",
                        "arm": arm,
                        "seed_base": seed,
                        "fold": fold,
                        "stage_a_sentinel_sha256": authorization.sha256,
                        "feature_tensor": "online_preprojector_r",
                        "ftv_head_called": False,
                        "test_labels_used": False,
                        "feature_path": str(feature_asset),
                        "checkpoint_path": str(checkpoint_path),
                        "feature_implementation_sha256": code_sha256[
                            "c1b_stage_b/features.py"
                        ],
                    },
                    f"feature metadata {key}",
                )
                if (
                    not feature_asset.is_file()
                    or feature_metadata.get("feature_sha256")
                    != file_sha256(feature_asset)
                    or not checkpoint_path.is_file()
                    or feature_metadata.get("checkpoint_sha256")
                    != file_sha256(checkpoint_path)
                ):
                    raise ValueError(f"feature asset/checkpoint hash drifted for {key}")
                probe_metadata = _read_json_object(
                    probe_metadata_paths[key], f"probe metadata {key}"
                )
                _require_contract_fields(
                    probe_metadata,
                    {
                        "schema_version": 1,
                        "arm": arm,
                        "seed_base": seed,
                        "fold": fold,
                        "stage_a_sentinel_sha256": authorization.sha256,
                        "test_used_for_scaler_or_selection": False,
                        "outer_test_predict_calls_per_cell": 1,
                        "feature_asset_name": feature_asset.name,
                        "feature_metadata_name": feature_metadata_path.name,
                        "feature_metadata_sha256": feature_metadata_hashes[key],
                        "probe_implementation_sha256": code_sha256[
                            "c1b_stage_b/probes.py"
                        ],
                        "target_adapter_sha256": code_sha256[
                            "c1b_stage_b/targets.py"
                        ],
                    },
                    f"probe metadata {key}",
                )
                if (
                    not feature_asset.is_file()
                    or probe_metadata.get("feature_sha256")
                    != file_sha256(feature_asset)
                ):
                    raise ValueError(f"probe-to-feature hash binding drifted for {key}")
                output_paths = {
                    "ridge_selection.csv": probe_dir / "ridge_selection.csv",
                    "ridge_predictions.private.csv": (
                        probe_dir / "ridge_predictions.private.csv"
                    ),
                    "probe_metrics.csv": probe_dir / "probe_metrics.csv",
                }
                _require_hash_inventory(
                    probe_metadata.get("output_sha256"),
                    output_paths,
                    f"probe output {key}",
                )

    implementation_paths = {
        "c1b_stage_b/analysis.py": Path(__file__).resolve(),
        "aggregate_cli": (EXPERIMENT_ROOT / "scripts" / "aggregate_stage_b.py").resolve(),
        "postprocess_driver": (
            EXPERIMENT_ROOT / "scripts" / "run_stage_b_postprocessing.py"
        ).resolve(),
    }
    for name, path in implementation_paths.items():
        expected = require_sha256(str(code_sha256.get(name, "")), name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"formal aggregation implementation drifted after postprocessing: {name}")
    if claim.get("postprocess_driver_sha256") != code_sha256.get("postprocess_driver"):
        raise ValueError("postprocessing driver hash differs between claim and completion")

    return {
        "formal_tag": FORMAL_POSTPROCESS_TAG,
        "data_contract_sha256": contract_sha256,
        "matrix_complete_sha256": matrix_sha256,
        "postprocessing_claim_sha256": linked_hashes["claim_sha256"],
        "postprocessing_preflight_sha256": linked_hashes["preflight_sha256"],
        "feature_export_complete_sha256": linked_hashes[
            "feature_export_complete_sha256"
        ],
        "postprocessing_complete_sha256": file_sha256(postprocess_path),
        "selection_history_sha256": selection_history_sha256,
        "selection_history_inventory_sha256": (
            selection_history_inventory_sha256
        ),
        "feature_metadata_sha256": feature_metadata_hashes,
        "probe_metadata_sha256": probe_metadata_hashes,
        "postprocessing_code_sha256": dict(code_sha256),
    }


def _cell_path(root: Path, seed: int, arm: str, fold: int) -> Path:
    return root / f"seed_{seed}" / arm / f"fold_{fold}"


def collect_complete_matrix(
    checkpoint_root: str | Path,
    probe_root: str | Path,
    authorization: StageAAuthorization,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_root = Path(checkpoint_root).resolve()
    probe_root = Path(probe_root).resolve()
    selections: list[dict[str, Any]] = []
    metrics: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    missing: list[str] = []
    for seed in SEED_BASES:
        for arm in ARMS:
            for fold in FOLDS:
                run = _cell_path(checkpoint_root, seed, arm, fold)
                probe = _cell_path(probe_root, seed, arm, fold)
                required = {
                    "selection": run / "selection.json",
                    "history": run / "history.csv",
                    "ridge_selection": probe / "ridge_selection.csv",
                    "metrics": probe / "probe_metrics.csv",
                    "predictions": probe / "ridge_predictions.private.csv",
                    "probe_metadata": probe / "probe_metadata.json",
                }
                absent = [str(path) for path in required.values() if not path.is_file()]
                if absent:
                    missing.extend(absent)
                    continue
                selection = json.loads(required["selection"].read_text(encoding="utf-8"))
                expected = {"arm": arm, "seed_base": seed, "fold": fold}
                if any(selection.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"selection identity mismatch: {required['selection']}")
                if selection.get("test_data_used") is not False:
                    raise ValueError("checkpoint selection used test data")
                if selection.get("stage_a_sentinel_sha256") != authorization.sha256:
                    raise ValueError("Stage A authorization differs across the result matrix")
                if selection.get("history_sha256") != file_sha256(required["history"]):
                    raise ValueError("training history differs from its selection record")
                probe_metadata = json.loads(
                    required["probe_metadata"].read_text(encoding="utf-8")
                )
                if not isinstance(probe_metadata, dict) or int(
                    probe_metadata.get("schema_version", -1)
                ) != 1:
                    raise ValueError("probe metadata must be a schema-v1 JSON object")
                for key, value in expected.items():
                    if probe_metadata.get(key) != value:
                        raise ValueError(
                            f"probe metadata identity mismatch at {key}: "
                            f"{required['probe_metadata']}"
                        )
                if probe_metadata.get("stage_a_sentinel_sha256") != authorization.sha256:
                    raise ValueError("probe metadata uses a different Stage A authorization")
                if probe_metadata.get("test_used_for_scaler_or_selection") is not False:
                    raise ValueError("probe metadata reports test use for fitting/selection")
                output_hashes = probe_metadata.get("output_sha256")
                expected_output_paths = (
                    required["ridge_selection"],
                    required["metrics"],
                    required["predictions"],
                )
                if not isinstance(output_hashes, dict) or any(
                    output_hashes.get(path.name) != file_sha256(path)
                    for path in expected_output_paths
                ):
                    raise ValueError("probe CSV SHA-256 evidence is missing or stale")
                selections.append(selection)
                for name, container in (
                    ("metrics", metrics), ("history", histories), ("predictions", predictions)
                ):
                    frame = pd.read_csv(required[name])
                    for identity, expected_value in expected.items():
                        if identity not in frame or not frame[identity].eq(expected_value).all():
                            raise ValueError(
                                f"{name} identity mismatch for {required[name]}: {identity}"
                            )
                    frame["arm"] = arm
                    frame["seed_base"] = seed
                    frame["fold"] = fold
                    container.append(frame)
    if missing:
        preview = "\n".join(missing[:12])
        raise RuntimeError(
            f"Stage B matrix is incomplete ({len(missing)} missing artifacts); no aggregation or conclusion is allowed:\n{preview}"
        )
    if len(selections) != len(SEED_BASES) * len(ARMS) * len(FOLDS):
        raise AssertionError("complete matrix cardinality mismatch")
    selection_frame = pd.DataFrame(selections)
    metric_frame = pd.concat(metrics, ignore_index=True)
    history_frame = pd.concat(histories, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    _audit_four_arm_matrix_contract(selection_frame, metric_frame, history_frame)
    _audit_oof_prediction_contract(metric_frame, prediction_frame)
    return selection_frame, metric_frame, history_frame, prediction_frame


def _expected_metric_grid() -> set[tuple[Any, ...]]:
    rows: set[tuple[Any, ...]] = set()
    for seed in SEED_BASES:
        for arm in ARMS:
            for fold in FOLDS:
                for scope in ANALYSIS_SCOPES:
                    for task, contract in TASK_CONTRACT.items():
                        for scale in contract["scales"]:
                            for endpoint in (*contract["endpoints"], "macro"):
                                rows.add(
                                    (
                                        seed,
                                        arm,
                                        fold,
                                        scope,
                                        task,
                                        endpoint,
                                        scale,
                                        contract["target_semantics"],
                                    )
                                )
    return rows


def _expected_prediction_grid() -> set[tuple[Any, ...]]:
    return {
        (
            seed,
            arm,
            fold,
            scope,
            task,
            endpoint,
            contract["target_semantics"],
            contract["scales"][1],
        )
        for seed in SEED_BASES
        for arm in ARMS
        for fold in FOLDS
        for scope in ANALYSIS_SCOPES
        for task, contract in TASK_CONTRACT.items()
        for endpoint in contract["endpoints"]
    }


def _audit_oof_prediction_contract(
    metrics: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Require exact five-fold OOF uniqueness and paired four-arm targets."""

    required = {
        "patient_id", "arm", "seed_base", "fold", "task", "endpoint",
        "analysis_scope", "target_semantics", "split", "y_true", "y_pred",
        "b0_prediction", "y_true_analysis", "y_pred_analysis",
        "b0_prediction_analysis", "analysis_scale", "test_predict_call_count",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"Stage B predictions miss required columns: {missing}")
    if not predictions["split"].eq("test").all():
        raise ValueError("pooled OOF assets may contain only outer-test predictions")
    if not predictions["test_predict_call_count"].eq(1).all():
        raise ValueError("one or more probe cells evaluated outer test more than once")
    prediction_group_keys = [
        "seed_base", "arm", "fold", "analysis_scope", "task", "endpoint",
        "target_semantics", "analysis_scale",
    ]
    observed_prediction_grid = {
        tuple(row)
        for row in predictions[prediction_group_keys].drop_duplicates().itertuples(
            index=False, name=None
        )
    }
    expected_prediction_grid = _expected_prediction_grid()
    if observed_prediction_grid != expected_prediction_grid:
        missing = len(expected_prediction_grid.difference(observed_prediction_grid))
        extra = len(observed_prediction_grid.difference(expected_prediction_grid))
        raise ValueError(
            "OOF predictions are not the exact formal Cartesian grid "
            f"({missing} missing, {extra} unexpected groups)"
        )
    group_sizes = predictions.groupby(prediction_group_keys, sort=False).size()
    if len(group_sizes) != len(expected_prediction_grid) or (group_sizes <= 0).any():
        raise ValueError("one or more formal OOF endpoint/fold groups are empty")
    identity = [
        "seed_base", "arm", "task", "endpoint", "analysis_scope", "patient_id"
    ]
    if predictions.duplicated(identity).any():
        raise ValueError("a patient appears in more than one outer-test fold for one OOF endpoint")

    natural = metrics.loc[
        metrics["analysis_scope"].isin(predictions["analysis_scope"].unique())
        & metrics["scale"].eq("natural")
        & metrics["endpoint"].ne("macro")
    ]
    count_keys = ["seed_base", "arm", "fold", "task", "endpoint", "analysis_scope"]
    expected_counts = natural.set_index(count_keys)["n_test"]
    if expected_counts.index.duplicated().any():
        raise ValueError("natural probe metrics contain duplicate fold endpoint rows")
    observed_counts = predictions.groupby(count_keys, sort=False).size()
    expected_counts = expected_counts.astype(int).sort_index()
    observed_counts = observed_counts.astype(int).sort_index()
    if not expected_counts.index.equals(observed_counts.index) or not expected_counts.equals(
        observed_counts
    ):
        raise ValueError("OOF prediction counts disagree with the fold probe metrics")

    fold_keys = ["seed_base", "arm", "task", "endpoint", "analysis_scope"]
    for keys, rows in predictions.groupby(fold_keys, sort=False):
        if set(rows["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"pooled OOF group does not cover all five folds: {keys}")

    paired_keys = [
        "seed_base", "task", "endpoint", "analysis_scope", "target_semantics", "patient_id"
    ]
    for keys, rows in predictions.groupby(paired_keys, sort=False):
        if set(rows["arm"]) != set(ARMS) or len(rows) != len(ARMS):
            raise ValueError(f"OOF patient coverage differs across arms for {keys}")
        for column in (
            "fold",
            "y_true",
            "b0_prediction",
            "y_true_analysis",
            "b0_prediction_analysis",
        ):
            values = rows[column].to_numpy()
            if column == "fold":
                aligned = np.all(values == values[0])
            else:
                aligned = np.allclose(values.astype(float), float(values[0]), rtol=0.0, atol=1e-12)
            if not aligned:
                raise ValueError(f"paired OOF {column} differs across arms for {keys}")


def _pooled_natural_metrics(rows: pd.DataFrame) -> dict[str, float]:
    truth = rows["y_true"].to_numpy(dtype=float)
    prediction = rows["y_pred"].to_numpy(dtype=float)
    baseline = rows["b0_prediction"].to_numpy(dtype=float)
    if not len(truth) or not all(np.isfinite(value).all() for value in (truth, prediction, baseline)):
        raise FloatingPointError("pooled OOF prediction rows are empty or non-finite")
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(truth, baseline)))
    target_variance = float(np.var(truth, ddof=0))
    if target_variance > 0:
        calibration_slope = float(
            np.mean((truth - np.mean(truth)) * (prediction - np.mean(prediction)))
            / target_variance
        )
        calibration_intercept = float(
            np.mean(prediction) - calibration_slope * np.mean(truth)
        )
    else:
        calibration_slope = calibration_intercept = math.nan
    if np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        spearman = pearson = math.nan
    else:
        spearman = float(spearmanr(truth, prediction).statistic)
        pearson = float(pearsonr(truth, prediction).statistic)
    return {
        "spearman": spearman if math.isfinite(spearman) else math.nan,
        "pearson": pearson if math.isfinite(pearson) else math.nan,
        "r2": float(r2_score(truth, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, prediction)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / b0_rmse if b0_rmse > 0 else math.nan,
        "prediction_target_variance_ratio": (
            float(np.var(prediction, ddof=0)) / target_variance
            if target_variance > 0 else math.nan
        ),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "calibration_mean_bias": float(np.mean(prediction - truth)),
    }


def pooled_oof_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recompute nonlinear natural-scale endpoints after pooling five test folds."""

    required = {
        "patient_id", "seed_base", "arm", "fold", "task", "endpoint",
        "analysis_scope", "target_semantics", "analysis_scale", "y_true",
        "y_pred", "b0_prediction",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"pooled OOF predictions miss required columns: {missing}")
    prediction_grid_keys = [
        "seed_base", "arm", "fold", "analysis_scope", "task", "endpoint",
        "target_semantics", "analysis_scale",
    ]
    observed_grid = set(
        predictions[prediction_grid_keys].drop_duplicates().itertuples(
            index=False, name=None
        )
    )
    expected_grid = _expected_prediction_grid()
    if observed_grid != expected_grid:
        raise ValueError("pooled OOF input does not contain the exact five-fold Cartesian grid")

    group_keys = [
        "seed_base", "arm", "task", "endpoint", "analysis_scope", "target_semantics"
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_keys, sort=False):
        if group["patient_id"].astype(str).duplicated().any():
            raise ValueError(f"pooled OOF group contains a repeated patient: {keys}")
        rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "scale": "natural",
                "aggregation": "pooled_5fold_oof",
                "n_test": len(group),
                **_pooled_natural_metrics(group),
            }
        )
    endpoint_frame = pd.DataFrame(rows)
    expected = {
        task: set(contract["endpoints"])
        for task, contract in TASK_CONTRACT.items()
    }
    macros: list[dict[str, Any]] = []
    macro_keys = ["seed_base", "arm", "task", "analysis_scope", "target_semantics"]
    for keys, group in endpoint_frame.groupby(macro_keys, sort=False):
        task = str(keys[2])
        if set(group["endpoint"]) != expected[task]:
            raise ValueError(f"pooled OOF endpoint coverage drifted for {keys}")
        macros.append(
            {
                **dict(zip(macro_keys, keys, strict=True)),
                "endpoint": "macro",
                "scale": "natural",
                "aggregation": "mean_of_pooled_endpoint_metrics",
                "n_test": int(group["n_test"].sum()),
                **{
                    metric: float(group[metric].mean())
                    for metric in (*METRICS, *CALIBRATION_METRICS)
                },
            }
        )
    pooled = pd.concat([endpoint_frame, pd.DataFrame(macros)], ignore_index=True)
    expected_rows = (
        len(SEED_BASES)
        * len(ARMS)
        * len(ANALYSIS_SCOPES)
        * sum(len(contract["endpoints"]) + 1 for contract in TASK_CONTRACT.values())
    )
    if len(pooled) != expected_rows:
        raise ValueError(
            f"pooled OOF metrics must contain exactly {expected_rows} rows"
        )
    pooled_keys = [
        "seed_base", "arm", "task", "endpoint", "analysis_scope",
        "target_semantics", "scale",
    ]
    if pooled.duplicated(pooled_keys).any():
        raise ValueError("pooled OOF metrics contain duplicate rows")
    return pooled


def _audit_four_arm_matrix_contract(
    selections: pd.DataFrame, metrics: pd.DataFrame, histories: pd.DataFrame
) -> None:
    """Refuse aggregation if any paired run contract or epoch order drifted."""

    selection_keys = ["seed_base", "arm", "fold"]
    expected_cells = {
        (seed, arm, fold)
        for seed in SEED_BASES
        for arm in ARMS
        for fold in FOLDS
    }
    if any(column not in selections for column in selection_keys):
        raise ValueError("selection matrix identity columns are missing")
    observed_cells = set(
        selections[selection_keys].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells or len(selections) != 40:
        raise ValueError("selection matrix is not the exact 2-seed x 4-arm x 5-fold grid")

    metric_keys = [
        "seed_base", "arm", "fold", "analysis_scope", "task", "endpoint",
        "scale", "target_semantics",
    ]
    required_metric_columns = set(metric_keys) | {
        "selected_alpha", "n_train", "n_val", "n_test", *METRICS,
    }
    if missing := sorted(required_metric_columns.difference(metrics.columns)):
        raise ValueError(f"probe metrics miss required columns: {missing}")
    if metrics.duplicated(metric_keys).any():
        raise ValueError("probe metrics contain duplicate formal Cartesian rows")
    observed_metric_grid = set(
        metrics[metric_keys].itertuples(index=False, name=None)
    )
    expected_metric_grid = _expected_metric_grid()
    if observed_metric_grid != expected_metric_grid or len(metrics) != len(
        expected_metric_grid
    ):
        missing = len(expected_metric_grid.difference(observed_metric_grid))
        extra = len(observed_metric_grid.difference(expected_metric_grid))
        raise ValueError(
            "probe metrics are not the exact formal Cartesian grid "
            f"({missing} missing, {extra} unexpected rows)"
        )

    required_history = {
        "seed_base", "arm", "fold", "epoch", "patient_order_sha256",
        "dropped_logical_tail_patients", "train_optimizer_steps",
    }
    if missing := sorted(required_history.difference(histories.columns)):
        raise ValueError(f"training histories miss required columns: {missing}")
    history_cells = set(
        histories[selection_keys].drop_duplicates().itertuples(index=False, name=None)
    )
    if history_cells != expected_cells:
        raise ValueError("training histories do not cover the exact formal 40 cells")
    for key, rows in histories.groupby(selection_keys, sort=False):
        epochs = rows["epoch"].astype(int)
        if epochs.duplicated().any() or set(epochs) != set(range(1, int(epochs.max()) + 1)):
            raise ValueError(f"training history epochs are incomplete or duplicated for {key}")

    matrix_batch_contracts = {
        (
            int(value["physical_batch_size"]),
            int(value["accumulation_steps"]),
        )
        for value in selections["hyperparameters"]
    }
    if len(matrix_batch_contracts) != 1:
        raise ValueError(
            "the formal matrix mixes batch contracts; a 2/16 OOM fallback must "
            "restart all forty runs"
        )
    matrix_pair = next(iter(matrix_batch_contracts))
    if matrix_pair not in {(4, 8), (2, 16)}:
        raise ValueError("matrix contains an unregistered physical/accumulation contract")
    matrix_fallback_flags = set(bool(value) for value in selections["global_fallback_restart"])
    if matrix_fallback_flags != {matrix_pair == (2, 16)}:
        raise ValueError("the matrix does not carry one global OOM-restart contract")

    for seed in SEED_BASES:
        for fold in FOLDS:
            current = selections.loc[
                selections["seed_base"].eq(seed) & selections["fold"].eq(fold)
            ]
            if set(current["arm"]) != set(ARMS) or len(current) != len(ARMS):
                raise ValueError(f"seed {seed}/fold {fold} is not an exact four-arm cell")
            for column in (
                "paired_initialization_sha256",
                "train_patient_sha256",
                "val_patient_sha256",
                "data_provenance_sha256",
            ):
                if current[column].nunique(dropna=False) != 1:
                    raise ValueError(f"seed {seed}/fold {fold} paired {column} drifted")
            hyperparameters = current["hyperparameters"].map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
            )
            if hyperparameters.nunique(dropna=False) != 1:
                raise ValueError(f"seed {seed}/fold {fold} hyperparameters differ across arms")
            resolved = current.iloc[0]["hyperparameters"]
            pair = (int(resolved["physical_batch_size"]), int(resolved["accumulation_steps"]))
            if pair not in {(4, 8), (2, 16)}:
                raise ValueError("matrix contains an unregistered physical/accumulation contract")
            fallback_flags = set(bool(value) for value in current["global_fallback_restart"])
            if fallback_flags != {pair == (2, 16)}:
                raise ValueError("global 2/16 restart provenance is missing or arm-specific")
            history = histories.loc[
                histories["seed_base"].eq(seed) & histories["fold"].eq(fold)
            ]
            epoch_one = history.loc[history["epoch"].eq(1)]
            if set(epoch_one["arm"]) != set(ARMS):
                raise ValueError(f"seed {seed}/fold {fold} lacks epoch-one evidence for all arms")
            for epoch, epoch_rows in history.groupby("epoch", sort=False):
                for column in (
                    "patient_order_sha256",
                    "dropped_logical_tail_patients",
                    "train_optimizer_steps",
                ):
                    if epoch_rows[column].nunique(dropna=False) != 1:
                        raise ValueError(
                            f"seed {seed}/fold {fold}/epoch {epoch} {column} differs across arms"
                        )
    primary = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["scale"].eq("natural")
    ]
    expected = {
        "static": {"T0", "T1", "T2", "T3", "macro"},
        "delta": {"T0→T1", "T1→T2", "T2→T3", "macro"},
    }
    for (seed, fold, arm, task), rows in primary.groupby(
        ["seed_base", "fold", "arm", "task"], sort=False
    ):
        if set(rows["endpoint"]) != expected[str(task)]:
            raise ValueError(f"probe endpoint coverage drifted for {seed}/{fold}/{arm}/{task}")
        if task == "delta" and set(rows["target_semantics"]) != {
            "literal_ftv_end_minus_ftv_start"
        }:
            raise ValueError("delta probe is not literal natural FTV subtraction")
    transformed_static = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["scale"].eq("transformed_outer_train")
        & metrics["task"].eq("static")
    ]
    for (seed, fold, arm), rows in transformed_static.groupby(
        ["seed_base", "fold", "arm"], sort=False
    ):
        if set(rows["endpoint"]) != {"T0", "T1", "T2", "T3", "macro"}:
            raise ValueError(
                f"transformed static endpoint coverage drifted for {seed}/{fold}/{arm}"
            )


def paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["scale"].eq("natural")
    ].copy()
    keys = ["seed_base"]
    if "fold" in primary.columns:
        keys.append("fold")
    keys.extend(["task", "endpoint", "target_semantics"])
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        wide = primary.pivot(index=keys, columns="arm", values=metric).reset_index()
        if set(ARMS).difference(wide.columns):
            raise ValueError(f"probe metric {metric} lacks one or more Stage B arms")
        for row in wide.itertuples(index=False):
            values = row._asdict()
            l_effect = float(values["L3"] - values["L1"])
            n_effect = float(values["N3"] - values["N1"])
            rows.append(
                {
                    **{key: values[key] for key in keys},
                    "metric": metric,
                    "L3_minus_L1": l_effect,
                    "N3_minus_N1": n_effect,
                    "N1_minus_L1": float(values["N1"] - values["L1"]),
                    "difference_in_differences": n_effect - l_effect,
                }
            )
    return pd.DataFrame(rows)


def optimization_table(selections: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (str(row.arm), int(row.seed_base), int(row.fold)): row
        for row in selections.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for seed in SEED_BASES:
        for fold in FOLDS:
            for arm in ARMS:
                row = lookup[(arm, seed, fold)]
                if arm in {"L3", "N3"}:
                    baseline_arm = "L1" if arm == "L3" else "N1"
                    baseline = float(lookup[(baseline_arm, seed, fold)].selected_validation_state_loss)
                    selected_state = float(row.selected_validation_state_loss)
                    if not math.isfinite(baseline) or baseline <= 0:
                        raise ValueError("paired baseline state loss must be finite and positive")
                    degradation = selected_state / baseline - 1.0
                    threshold_exceeded = selected_state > 1.05 * baseline
                    safety_pass = not threshold_exceeded and bool(row.experiment_pass)
                else:
                    baseline_arm = arm
                    baseline = float(row.selected_validation_state_loss)
                    degradation = 0.0
                    threshold_exceeded = False
                    safety_pass = bool(row.experiment_pass)
                rows.append(
                    {
                        "arm": arm,
                        "seed_base": seed,
                        "fold": fold,
                        "selected_epoch": int(row.selected_epoch),
                        "selected_validation_total_loss": float(
                            row.selected_validation_total_loss
                        ),
                        "selected_validation_base_loss": float(
                            row.selected_validation_base_loss
                        ),
                        "selected_validation_state_loss": float(row.selected_validation_state_loss),
                        "paired_baseline_arm": baseline_arm,
                        "paired_baseline_state_loss": baseline,
                        "state_loss_degradation_fraction": degradation,
                        "base_degradation_fraction": degradation,
                        "state_loss_degradation_gt_5pct": threshold_exceeded,
                        "selected_validation_ftv_loss": float(row.selected_validation_ftv_loss),
                        "selected_representation_std": float(row.selected_representation_std),
                        "selection_mode": str(row.selection_mode),
                        "finite": bool(row.finite_status) and all(
                            math.isfinite(float(value))
                            for value in (
                                row.selected_validation_state_loss,
                                row.selected_representation_std,
                            )
                        ),
                        "optimization_safety_pass": safety_pass,
                    }
                )
    return pd.DataFrame(rows)


def optimization_difference_in_differences(
    optimization: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the safety interaction at the preregistered independent seed level."""

    rows: list[dict[str, Any]] = []
    for seed, group in optimization.groupby("seed_base", sort=False):
        expected_cells = {(arm, fold) for arm in ARMS for fold in FOLDS}
        observed_cells = set(zip(group["arm"], group["fold"], strict=False))
        if observed_cells != expected_cells or len(group) != len(expected_cells):
            raise ValueError("optimization DiD requires all four arms x five folds")
        arm_means = group.groupby("arm", sort=False).agg(
            state_loss_degradation_fraction=(
                "state_loss_degradation_fraction",
                "mean",
            ),
            selected_validation_state_loss=("selected_validation_state_loss", "mean"),
        )
        if set(arm_means.index) != set(ARMS):
            raise ValueError("optimization DiD requires all four arms")
        legacy = float(
            arm_means.loc["L3", "state_loss_degradation_fraction"]
        )
        new = float(arm_means.loc["N3", "state_loss_degradation_fraction"])
        rows.append(
            {
                "seed_base": int(seed),
                "fold_aggregation": "mean_of_five_paired_folds",
                "task": "optimization",
                "endpoint": "selected_validation_state_loss_degradation",
                "target_semantics": "paired_grounded_vs_no_ground_fraction",
                "metric": "state_loss_degradation_fraction",
                "L3_minus_L1": legacy,
                "N3_minus_N1": new,
                "N1_minus_L1": (
                    float(arm_means.loc["N1", "selected_validation_state_loss"])
                    / float(arm_means.loc["L1", "selected_validation_state_loss"])
                    - 1.0
                ),
                "difference_in_differences": new - legacy,
            }
        )
    result = pd.DataFrame(rows)
    if set(result.get("seed_base", pd.Series(dtype=int)).astype(int)) != set(SEED_BASES):
        raise ValueError("optimization DiD requires both preregistered training seeds")
    return result


def _comparison_figure(metrics: pd.DataFrame, task: str, metric: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["scale"].eq("natural")
        & metrics["task"].eq(task)
        & metrics["endpoint"].eq("macro")
    ]
    values = [subset.loc[subset["arm"].eq(arm), metric].to_numpy(float) for arm in ARMS]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(values, tick_labels=ARMS, showmeans=True)
    for index, array in enumerate(values, start=1):
        axis.scatter(np.full(len(array), index), array, alpha=0.65, s=20)
    axis.set_ylabel(metric)
    axis.set_title(f"Stage B {task} macro {metric}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_stage_b_figures(
    metrics: pd.DataFrame,
    effects: pd.DataFrame,
    optimization: pd.DataFrame,
    histories: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: str | Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(figure_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in FIGURE_NAMES]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite Stage B figures")
    _comparison_figure(metrics, "static", "spearman", paths[0])
    _comparison_figure(metrics, "static", "r2", paths[1])

    static = predictions.loc[
        predictions["task"].eq("static")
        & predictions["analysis_scope"].eq("primary_measurement_valid")
    ].copy()
    endpoint_colors = dict(
        zip(TASK_CONTRACT["static"]["endpoints"], plt.get_cmap("tab10").colors, strict=False)
    )
    seed_markers = {SEED_BASES[0]: "o", SEED_BASES[1]: "x"}
    seed_linestyles = {SEED_BASES[0]: "-", SEED_BASES[1]: ":"}
    low = float(min(static["y_true"].min(), static["y_pred"].min()))
    high = float(max(static["y_true"].max(), static["y_pred"].max()))
    padding = max((high - low) * 0.03, 1e-12)
    low, high = low - padding, high + padding
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for axis, arm in zip(axes.flat, ARMS, strict=True):
        arm_rows = static.loc[static["arm"].eq(arm)]
        for endpoint in TASK_CONTRACT["static"]["endpoints"]:
            color = endpoint_colors[endpoint]
            for seed in SEED_BASES:
                group = arm_rows.loc[
                    arm_rows["endpoint"].eq(endpoint)
                    & arm_rows["seed_base"].eq(seed)
                ]
                axis.scatter(
                    group["y_true"],
                    group["y_pred"],
                    color=color,
                    marker=seed_markers[seed],
                    alpha=0.16,
                    s=10,
                    linewidths=0.5,
                    label=f"figure6:{arm}:{endpoint}:seed_{seed}",
                )
                calibration = _pooled_natural_metrics(group)
                slope = calibration["calibration_slope"]
                intercept = calibration["calibration_intercept"]
                if math.isfinite(slope) and math.isfinite(intercept):
                    x_values = np.asarray([low, high])
                    axis.plot(
                        x_values,
                        slope * x_values + intercept,
                        color=color,
                        linestyle=seed_linestyles[seed],
                        linewidth=0.9,
                        alpha=0.75,
                    )
        axis.plot(
            [low, high], [low, high], linestyle="--", color="black", linewidth=1
        )
        axis.set_title(f"{arm}: five-fold OOF by training seed")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("observed natural FTV")
    for axis in axes[:, 0]:
        axis.set_ylabel("predicted natural FTV")
    endpoint_handles = [
        Line2D([0], [0], color=endpoint_colors[endpoint], linewidth=2, label=endpoint)
        for endpoint in TASK_CONTRACT["static"]["endpoints"]
    ]
    seed_handles = [
        Line2D(
            [0], [0], color="black", marker=seed_markers[seed],
            linestyle=seed_linestyles[seed], label=f"seed {seed} pooled OOF",
        )
        for seed in SEED_BASES
    ]
    figure.legend(
        handles=[*endpoint_handles, *seed_handles],
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle("Static natural-FTV calibration; shared axes and identity line")
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    figure.savefig(paths[2], dpi=180)
    plt.close(figure)

    _comparison_figure(metrics, "delta", "spearman", paths[3])
    _comparison_figure(metrics, "delta", "r2", paths[4])

    grounded = optimization.loc[optimization["arm"].isin(["L3", "N3"])].copy()
    grounded["cell"] = grounded["seed_base"].astype(str) + "/f" + grounded["fold"].astype(str)
    heat = grounded.pivot(
        index="arm", columns="cell", values="state_loss_degradation_fraction"
    )
    figure, axis = plt.subplots(figsize=(10, 2.8))
    image = axis.imshow(heat.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-0.05, vmax=0.10)
    axis.set_yticks(range(len(heat.index)), labels=heat.index)
    axis.set_xticks(range(len(heat.columns)), labels=heat.columns, rotation=45, ha="right")
    figure.colorbar(image, ax=axis, label="validation state-loss degradation")
    figure.tight_layout()
    figure.savefig(paths[5], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    std_values = [
        optimization.loc[optimization["arm"].eq(arm), "selected_representation_std"]
        .to_numpy(float)
        for arm in ARMS
    ]
    axis.boxplot(std_values, tick_labels=ARMS, showmeans=True)
    for index, values in enumerate(std_values, start=1):
        axis.scatter(np.full(len(values), index), values, alpha=0.65, s=20)
    axis.set_ylabel("selected validation representation std")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[6], dpi=180)
    plt.close(figure)

    interaction = effects.loc[
        effects["metric"].eq("r2") & effects["endpoint"].eq("macro")
    ]
    figure, axis = plt.subplots(figsize=(5, 5))
    for task, marker in (("static", "o"), ("delta", "s")):
        part = interaction.loc[interaction["task"].eq(task)]
        axis.scatter(part["L3_minus_L1"], part["N3_minus_N1"], label=task, marker=marker)
    limits = axis.get_xlim()
    lower, upper = min(limits[0], axis.get_ylim()[0]), max(limits[1], axis.get_ylim()[1])
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("L3 - L1 macro natural R²")
    axis.set_ylabel("N3 - N1 macro natural R²")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[7], dpi=180)
    plt.close(figure)

    representative = histories.loc[
        histories["seed_base"].eq(SEED_BASES[0]) & histories["fold"].eq(0)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arm in ARMS:
        part = representative.loc[representative["arm"].eq(arm)].sort_values("epoch")
        axes[0].plot(part["epoch"], part["val_state_loss"], marker="o", label=arm)
        if arm in {"L3", "N3"}:
            axes[1].plot(part["epoch"], part["val_ftv_loss"], marker="o", label=arm)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation state loss")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation Direct-FTV loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[8], dpi=180)
    plt.close(figure)
    return paths


def _claim_formal_aggregation(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a persistent one-owner claim without an overwrite race."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"formal aggregation root is already claimed: {path}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _table_with_exact_contract(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    columns = list(TABLE_COLUMNS[name])
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    result = result.loc[:, columns]
    expected_rows = TABLE_ROW_COUNTS[name]
    if len(result) != expected_rows:
        raise ValueError(
            f"{TABLE_FILENAMES[name]} must contain exactly {expected_rows} rows; "
            f"observed {len(result)}"
        )
    if tuple(result.columns) != TABLE_COLUMNS[name]:
        raise AssertionError(f"{TABLE_FILENAMES[name]} schema construction failed")
    return result


def _publish_staged_files(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Atomically publish each file; roll back this transaction on any exception."""

    published: list[Path] = []
    try:
        for source, destination in pairs:
            if not source.is_file() or source.stat().st_size <= 0:
                raise ValueError(f"staged aggregate artifact is missing or empty: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            published.append(destination)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def aggregate_stage_b(
    *,
    checkpoint_root: str | Path,
    feature_root: str | Path,
    probe_root: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    authorization: StageAAuthorization,
    data_contract: str | Path,
    data_contract_sha256: str,
) -> dict[str, Any]:
    input_evidence = validate_formal_aggregation_inputs(
        checkpoint_root=checkpoint_root,
        feature_root=feature_root,
        probe_root=probe_root,
        authorization=authorization,
        data_contract=data_contract,
        data_contract_sha256=data_contract_sha256,
    )
    selections, metrics, histories, predictions = collect_complete_matrix(
        checkpoint_root, probe_root, authorization
    )
    pooled_metrics = pooled_oof_metrics(predictions)
    effects = paired_effects(pooled_metrics)
    fold_effects = paired_effects(metrics)
    optimization = optimization_table(selections)
    optimization_did = optimization_difference_in_differences(optimization)
    primary_representation_did = effects.loc[
        effects["endpoint"].eq("macro")
        & effects["metric"].isin(["spearman", "r2"])
        & effects["task"].isin(["static", "delta"])
    ].copy()
    primary_representation_did["fold_aggregation"] = "pooled_five_fold_oof"
    did_table = pd.concat(
        [primary_representation_did, optimization_did], ignore_index=True
    )
    output, figures = _require_exact_formal_output_roots(output_dir, figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: output / filename for name, filename in TABLE_FILENAMES.items()
    }
    summary_path = output / AGGREGATION_SUMMARY_NAME
    claim_path = output / AGGREGATION_CLAIM_NAME
    figure_paths_expected = [figures / name for name in FIGURE_NAMES]
    if any(
        path.exists()
        for path in (*outputs.values(), summary_path, *figure_paths_expected)
    ):
        raise FileExistsError("refusing to overwrite Stage B aggregate outputs")
    primary_natural = pooled_metrics.loc[
        pooled_metrics["analysis_scope"].eq("primary_measurement_valid")
        & pooled_metrics["scale"].eq("natural")
    ]
    static_fold = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["task"].eq("static")
    ].copy()
    static_fold["aggregation"] = "outer_fold"
    static_table = pd.concat(
        [
            static_fold,
            primary_natural.loc[primary_natural["task"].eq("static")],
        ],
        ignore_index=True,
        sort=False,
    )
    tables = {
        "static": _table_with_exact_contract(static_table, "static"),
        "delta": _table_with_exact_contract(
            primary_natural.loc[primary_natural["task"].eq("delta")], "delta"
        ),
        "optimization": _table_with_exact_contract(optimization, "optimization"),
        "effects": _table_with_exact_contract(did_table, "effects"),
        "fold_effects": _table_with_exact_contract(fold_effects, "fold_effects"),
    }

    implementation_paths = {
        "analysis": Path(__file__).resolve(),
        "aggregate_cli": (EXPERIMENT_ROOT / "scripts" / "aggregate_stage_b.py").resolve(),
        "postprocess_driver": (
            EXPERIMENT_ROOT / "scripts" / "run_stage_b_postprocessing.py"
        ).resolve(),
    }
    implementation_sha256 = {
        name: file_sha256(path) for name, path in implementation_paths.items()
    }
    _claim_formal_aggregation(
        claim_path,
        {
            "schema_version": 1,
            "status": "CLAIMED",
            "formal_tag": FORMAL_POSTPROCESS_TAG,
            "stage_a_sentinel_sha256": authorization.sha256,
            "data_contract_sha256": input_evidence["data_contract_sha256"],
            "matrix_complete_sha256": input_evidence["matrix_complete_sha256"],
            "postprocessing_complete_sha256": input_evidence[
                "postprocessing_complete_sha256"
            ],
            "selection_history_inventory_sha256": input_evidence[
                "selection_history_inventory_sha256"
            ],
            "implementation_sha256": implementation_sha256,
            "table_targets": [path.name for path in outputs.values()],
            "figure_targets": list(FIGURE_NAMES),
            "summary_target": summary_path.name,
            "summary_is_commit_marker": True,
            "nonresumable": True,
        },
    )

    table_stage: Path | None = None
    figure_stage: Path | None = None
    try:
        table_stage = Path(
            tempfile.mkdtemp(prefix=".stage_b_aggregation_tables.", dir=output)
        )
        figure_stage = Path(
            tempfile.mkdtemp(prefix=".stage_b_aggregation_figures.", dir=figures)
        )
        staged_tables = {
            name: table_stage / TABLE_FILENAMES[name] for name in TABLE_FILENAMES
        }
        for name, frame in tables.items():
            _atomic_csv(staged_tables[name], frame)
        staged_figures = make_stage_b_figures(
            pooled_metrics,
            effects,
            optimization,
            histories,
            predictions,
            figure_stage,
        )
        if [path.name for path in staged_figures] != list(FIGURE_NAMES):
            raise ValueError("Stage B figure inventory drifted")

        # Staged results never become formal if any completion, metadata, output,
        # or implementation hash changes while tables and figures are generated.
        final_input_evidence = validate_formal_aggregation_inputs(
            checkpoint_root=checkpoint_root,
            feature_root=feature_root,
            probe_root=probe_root,
            authorization=authorization,
            data_contract=data_contract,
            data_contract_sha256=data_contract_sha256,
        )
        if final_input_evidence != input_evidence:
            raise RuntimeError("formal aggregation inputs changed during staging")
        if any(
            file_sha256(path) != implementation_sha256[name]
            for name, path in implementation_paths.items()
        ):
            raise RuntimeError("formal aggregation implementation changed during staging")

        table_sha256 = {
            path.name: file_sha256(path) for path in staged_tables.values()
        }
        figure_sha256 = {path.name: file_sha256(path) for path in staged_figures}
        summary = {
            "schema_version": 2,
            "status": "COMPLETE",
            "formal_tag": FORMAL_POSTPROCESS_TAG,
            "matrix_complete": True,
            "postprocessing_complete": True,
            "run_count": len(selections),
            "seeds": list(SEED_BASES),
            "folds": list(FOLDS),
            "arms": list(ARMS),
            "stage_a_sentinel_sha256": authorization.sha256,
            "aggregation_claim_sha256": file_sha256(claim_path),
            "input_sha256": input_evidence,
            "implementation_sha256": implementation_sha256,
            "table_sha256": table_sha256,
            "table_rows": {
                TABLE_FILENAMES[name]: len(frame) for name, frame in tables.items()
            },
            "table_columns": {
                TABLE_FILENAMES[name]: list(frame.columns)
                for name, frame in tables.items()
            },
            "figure_sha256": figure_sha256,
            "comparisons": ["L3-L1", "N3-N1", "N1-L1", "(N3-N1)-(L3-L1)"],
            "independent_unit": "training seed; folds and visits are not replicates",
            "primary_representation_aggregation": (
                "five-fold pooled OOF recomputed separately per seed"
            ),
            "macro_definition": "unweighted mean of endpoint metrics",
            "macro_n_test_definition": "sum of endpoint observations, not unique patients",
            "frozen_metric_name_mapping": {
                "mean_baseline_rmse_gain": "rmse_gain_over_b0",
                "prediction_variance_ratio": "prediction_target_variance_ratio",
            },
            "calibration_diagnostics": [
                "natural_r2",
                "prediction_target_variance_ratio",
                "calibration_slope",
                "calibration_intercept",
                "calibration_mean_bias",
                "four-arm shared-axis seed-pooled OOF identity plot",
            ],
            "calibration_coefficients_are_descriptive": True,
            "fold_level_effects_retained_as_sensitivity": True,
            "literal_delta_enforced": True,
            "static_transformed_and_inverse_natural_reported": True,
            "pearson_reported": True,
            "optimization_safety_metric": (
                "selected validation state loss <= 1.05 x paired baseline"
            ),
            "test_used_for_checkpoint_selection": False,
            "pilot_seed_scope": "two preregistered training seeds; no broader multiseed claim",
            "publication": (
                "all artifacts generated and hashed in hidden staging; summary published last "
                "as the commit marker; exception rollback removes any newly linked result files"
            ),
            "interpretation": (
                "not auto-classified; apply only preregistered Outcomes A-D to the "
                "complete paired tables and select one next-step priority"
            ),
        }
        staged_summary = table_stage / AGGREGATION_SUMMARY_NAME
        _atomic_json(staged_summary, summary)
        publish_pairs = [
            *((staged_tables[name], outputs[name]) for name in TABLE_FILENAMES),
            *((path, figures / path.name) for path in staged_figures),
            (staged_summary, summary_path),
        ]
        _publish_staged_files(publish_pairs)
        return summary
    finally:
        if table_stage is not None:
            shutil.rmtree(table_stage, ignore_errors=True)
        if figure_stage is not None:
            shutil.rmtree(figure_stage, ignore_errors=True)


__all__ = [
    "CALIBRATION_METRICS",
    "TABLE_COLUMNS",
    "TABLE_FILENAMES",
    "TABLE_ROW_COUNTS",
    "aggregate_stage_b",
    "collect_complete_matrix",
    "make_stage_b_figures",
    "optimization_difference_in_differences",
    "optimization_table",
    "paired_effects",
    "pooled_oof_metrics",
    "validate_formal_aggregation_inputs",
]
