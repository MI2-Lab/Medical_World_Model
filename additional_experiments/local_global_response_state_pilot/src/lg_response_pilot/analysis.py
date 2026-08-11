"""Complete-matrix aggregation and preregistered pilot decisions.

Natural-scale endpoint metrics are recomputed only after pooling all five
outer-test folds.  Endpoint macros are unweighted means of those pooled
metrics.  Fold-specific transformed spaces are never concatenated; their
metrics are summarized across folds instead.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import canonical_sha256, file_sha256, require_sha256
from .probes import metric_values


CANONICAL_ARMS = ("GAP0", "GAP3", "LOCAL0", "LOCAL3", "LG0", "LG3")
PRIMARY_SCOPE = "primary_measurement_valid"
ANALYSIS_SCOPES = (PRIMARY_SCOPE, "observable_only")
METRICS = (
    "spearman",
    "pearson",
    "r2",
    "rmse",
    "mae",
    "b0_rmse",
    "rmse_gain_over_b0",
    "prediction_target_variance_ratio",
    "calibration_slope",
    "calibration_intercept",
    "calibration_mean_bias",
)
TABLE_FILENAMES = {
    "table1": "table1_architecture_contract.csv",
    "table2": "table2_static_ftv.csv",
    "table3": "table3_observed_delta_ftv.csv",
    "table4": "table4_paired_architecture_effects.csv",
    "table5": "table5_grounding_effects.csv",
    "table6": "table6_optimization_safety.csv",
    "table7": "table7_prediction_variance_calibration.csv",
}
EXACT_ARM_SPECS = {
    "GAP0": {"pooling": "GAP", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": False},
    "GAP3": {"pooling": "GAP", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": True},
    "LOCAL0": {"pooling": "fixed_64mm_LOCAL", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": False},
    "LOCAL3": {"pooling": "fixed_64mm_LOCAL", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": True},
    "LG0": {"pooling": "concat(fixed_64mm_LOCAL,GAP)", "projection": "Linear(256,192)+LayerNorm(192)", "grounded": False},
    "LG3": {"pooling": "concat(fixed_64mm_LOCAL,GAP)", "projection": "Linear(256,192)+LayerNorm(192)", "grounded": True},
}
EXACT_OBJECTIVE = {
    "formula": "L_JEPA + 0.25 * L_FTV for grounded arms; L_JEPA otherwise",
    "lambda_ftv": 0.25,
    "sigreg_weight": 0.09,
    "step_weights": [2.0, 1.0, 0.5],
    "grounding_mask": "measurement_valid AND frozen grounding_observable_mask; loss-side only",
    "ftv_transform": "outer-train observable log(FTV+epsilon), 1/99 winsorization, median/IQR",
    "delta_supervision": False,
}
EXACT_PROBES = {
    "feature": "selected frozen online pre-projector r_t [N,4,192]",
    "ridge_alphas": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "selection": "outer train fit and validation analysis-space MSE; smallest-alpha tie break",
    "static_endpoints": ["T0", "T1", "T2", "T3", "macro"],
    "delta_endpoints": ["T0_to_T1", "T1_to_T2", "T2_to_T3", "macro"],
    "delta_definition": "literal natural FTV_(t+1)-FTV_t from delta r",
    "natural_aggregation": "pool five outer-test folds before each endpoint metric; macro is unweighted endpoint mean",
    "transformed_aggregation": "fold summaries only; never pool incompatible fold transforms",
    "primary_scope": "measurement_valid",
    "sensitivity_scope": "measurement_valid AND grounding_observable",
}
EXACT_OPERATIONALIZATION = {
    "natural_r2_systematic_worsening": "effect is strictly negative in both seeds",
    "meaningful_delta_spearman_gain": 0.02,
    "thresholds_are_descriptive_not_statistical_significance": True,
    "folds_are_paired_sensitivity_not_independent_replicates": True,
}
EXACT_GATES = {
    "A_LOCAL_STATE_WORKS": {
        "static_macro_spearman_gain_each_seed_min": 0.1,
        "delta_macro_spearman_gain_each_seed_strictly_gt": 0.0,
        "static_natural_r2_systematic_worsening_forbidden": True,
    },
    "B_LOCAL_GLOBAL_ADDS_VALUE": {
        "static_macro_spearman_gain_each_seed_min": 0.0,
        "static_macro_spearman_gain_at_least_one_seed_min": 0.02,
        "static_natural_r2_systematic_worsening_forbidden": True,
    },
    "C_GROUNDING_COMPATIBILITY": {
        "candidate_static_macro_spearman_gain_each_seed_strictly_gt": 0.0,
        "candidate_delta_macro_spearman_gain_at_least_one_seed_min": 0.02,
        "static_natural_r2_systematic_worsening_forbidden": True,
    },
    "D_OPTIMIZATION_SAFETY": {
        "candidate_paired_folds_required": 9,
        "candidate_paired_folds_total": 10,
        "maximum_state_loss_degradation_fraction": 0.05,
    },
}
EXACT_SELECTION_RULE = {
    "if_A_pass_and_B_fail": "LOCAL",
    "if_A_pass_and_B_pass": "LOCAL_GLOBAL",
    "if_A_fail": "GAP and classification C",
    "classification_D_override": "A passes but grounded final candidate fails Gate D",
}
EXACT_NEXT_STAGE_POLICY = {
    "direct_FTV_plus_LD_from_this_pilot": False,
    "after_A_or_B": "first perform larger multi-seed confirmation of the selected architecture",
    "FTV_plus_LD_condition": "only after confirmation, and only if pilot grounding compatibility and safety are supported",
}


def _exact_json(value: Any, expected: Any, label: str) -> None:
    observed = json.dumps(value, sort_keys=True, separators=(",", ":"))
    frozen = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    if observed != frozen:
        raise ValueError(f"pilot {label} drifted from the exact preregistered contract")


def _validate_scientific_contract(payload: Mapping[str, Any]) -> None:
    _exact_json(payload.get("arms"), EXACT_ARM_SPECS, "arm specifications")
    _exact_json(payload.get("objective"), EXACT_OBJECTIVE, "FTV-only objective")
    _exact_json(payload.get("probes"), EXACT_PROBES, "probe protocol")
    _exact_json(
        payload.get("gate_operationalization"),
        EXACT_OPERATIONALIZATION,
        "gate operationalization",
    )
    _exact_json(payload.get("gates"), EXACT_GATES, "Gate A-D thresholds")
    _exact_json(
        payload.get("selection_rule"), EXACT_SELECTION_RULE, "selection rule"
    )
    _exact_json(
        payload.get("next_stage_policy"),
        EXACT_NEXT_STAGE_POLICY,
        "next-stage policy",
    )
    training = _require_mapping(payload.get("training"), "training")
    expected_training = {
        "physical_batch_size": 4,
        "accumulation_steps": 8,
        "logical_batch_size": 32,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("pilot physical4/accum8/logical32 contract drifted")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"{label} must be finite")
    return observed


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate the exact preregistered Gate A-D schema."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"pilot config is missing or invalid: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("pilot config must be a schema-v1 JSON object")
    if payload.get("experiment") != "local_global_response_state_pilot":
        raise ValueError("pilot config names a different experiment")
    arms = _require_mapping(payload.get("arms"), "arms")
    if tuple(arms) != CANONICAL_ARMS:
        raise ValueError("pilot arm order/set differs from the six preregistered arms")
    training = _require_mapping(payload.get("training"), "training")
    if tuple(training.get("seed_bases", ())) != (2026, 3026):
        raise ValueError("pilot seeds differ from 2026/3026")
    if tuple(training.get("folds", ())) != tuple(range(5)):
        raise ValueError("pilot folds differ from 0..4")
    if int(training.get("formal_cells", -1)) != 60:
        raise ValueError("pilot formal matrix must contain exactly 60 cells")

    operational = _require_mapping(
        payload.get("gate_operationalization"), "gate_operationalization"
    )
    if operational.get("natural_r2_systematic_worsening") != (
        "effect is strictly negative in both seeds"
    ):
        raise ValueError("natural-R2 worsening operationalization drifted")
    if _finite_number(
        operational.get("meaningful_delta_spearman_gain"),
        "meaningful delta gain",
    ) != 0.02:
        raise ValueError("meaningful delta Spearman gain must remain 0.02")

    gates = _require_mapping(payload.get("gates"), "gates")
    exact_gate_keys = {
        "A_LOCAL_STATE_WORKS",
        "B_LOCAL_GLOBAL_ADDS_VALUE",
        "C_GROUNDING_COMPATIBILITY",
        "D_OPTIMIZATION_SAFETY",
    }
    if set(gates) != exact_gate_keys:
        raise ValueError("pilot gate set differs from exact Gate A-D")
    expected_fields = {
        "A_LOCAL_STATE_WORKS": {
            "static_macro_spearman_gain_each_seed_min",
            "delta_macro_spearman_gain_each_seed_strictly_gt",
            "static_natural_r2_systematic_worsening_forbidden",
        },
        "B_LOCAL_GLOBAL_ADDS_VALUE": {
            "static_macro_spearman_gain_each_seed_min",
            "static_macro_spearman_gain_at_least_one_seed_min",
            "static_natural_r2_systematic_worsening_forbidden",
        },
        "C_GROUNDING_COMPATIBILITY": {
            "candidate_static_macro_spearman_gain_each_seed_strictly_gt",
            "candidate_delta_macro_spearman_gain_at_least_one_seed_min",
            "static_natural_r2_systematic_worsening_forbidden",
        },
        "D_OPTIMIZATION_SAFETY": {
            "candidate_paired_folds_required",
            "candidate_paired_folds_total",
            "maximum_state_loss_degradation_fraction",
        },
    }
    for gate_name, fields in expected_fields.items():
        row = _require_mapping(gates[gate_name], gate_name)
        if set(row) != fields:
            raise ValueError(f"{gate_name} threshold fields drifted")
        for field, value in row.items():
            if field.endswith("forbidden"):
                if value is not True:
                    raise ValueError(f"{gate_name}.{field} must be true")
            else:
                _finite_number(value, f"{gate_name}.{field}")
    if int(gates["D_OPTIMIZATION_SAFETY"]["candidate_paired_folds_total"]) != 10:
        raise ValueError("Gate D total must be ten paired folds")
    _validate_scientific_contract(payload)
    return payload


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _cell_path(root: Path, seed: int, arm: str, fold: int) -> Path:
    return root / f"seed_{seed}" / arm / f"fold_{fold}"


def _assert_owner_private_tree(root: Path, label: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"{label} root is missing: {root}")
    for path in root.rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            if (path.stat().st_mode & 0o777) != 0o600:
                raise PermissionError(f"{label} artifact is not owner-only 0600: {path}")


def _validate_cell_chain(
    paths: Mapping[str, Path],
    identity: Mapping[str, Any],
    feature_metadata: Mapping[str, Any],
    probe_metadata: Mapping[str, Any],
    lock_sha256: str,
) -> dict[str, Any]:
    """Bind one probe cell to live feature, selection, and checkpoint assets."""

    if feature_metadata.get("schema_version") != 1:
        raise ValueError("feature metadata schema drifted")
    if any(feature_metadata.get(key) != value for key, value in identity.items()):
        raise ValueError(f"feature metadata identity differs for {identity}")
    if (
        feature_metadata.get("preregistration_lock")
        != "PREREGISTRATION_LOCK.json"
        or feature_metadata.get("preregistration_lock_sha256") != lock_sha256
    ):
        raise ValueError("feature metadata preregistration binding drifted")

    expected_paths = {
        "feature_path": paths["feature"].resolve(),
        "checkpoint_path": paths["selected"].resolve(),
        "selection_path": paths["selection"].resolve(),
    }
    for field, expected in expected_paths.items():
        observed = Path(str(feature_metadata.get(field, ""))).resolve()
        if observed != expected:
            raise ValueError(
                f"feature metadata {field} differs from the supplied formal roots"
            )
    expected_hashes = {
        "feature_sha256": file_sha256(paths["feature"]),
        "checkpoint_sha256": file_sha256(paths["selected"]),
        "selection_sha256": file_sha256(paths["selection"]),
    }
    for field, expected in expected_hashes.items():
        if feature_metadata.get(field) != expected:
            raise ValueError(f"feature metadata live {field} binding drifted")

    feature_metadata_sha256 = file_sha256(paths["feature_metadata"])
    binding = {
        "feature_path": str(expected_paths["feature_path"]),
        "feature_sha256": expected_hashes["feature_sha256"],
        "feature_metadata_path": str(paths["feature_metadata"].resolve()),
        "feature_metadata_sha256": feature_metadata_sha256,
        "checkpoint_path": str(expected_paths["checkpoint_path"]),
        "checkpoint_sha256": expected_hashes["checkpoint_sha256"],
        "selection_path": str(expected_paths["selection_path"]),
        "selection_sha256": expected_hashes["selection_sha256"],
    }
    if probe_metadata.get("feature_checkpoint_binding") != binding:
        raise ValueError("probe metadata feature/checkpoint chain differs")
    if probe_metadata.get("feature_checkpoint_binding_sha256") != canonical_sha256(
        binding
    ):
        raise ValueError("probe metadata feature/checkpoint chain hash drifted")
    if (
        probe_metadata.get("feature_sha256") != expected_hashes["feature_sha256"]
        or probe_metadata.get("feature_metadata_sha256")
        != feature_metadata_sha256
    ):
        raise ValueError("probe metadata feature asset binding drifted")
    return {
        "seed_base": int(identity["seed_base"]),
        "arm": str(identity["arm"]),
        "fold": int(identity["fold"]),
        "selection_sha256": expected_hashes["selection_sha256"],
        "checkpoint_sha256": expected_hashes["checkpoint_sha256"],
        "feature_sha256": expected_hashes["feature_sha256"],
        "feature_metadata_sha256": feature_metadata_sha256,
        "probe_metadata_sha256": file_sha256(paths["probe_metadata"]),
    }


def collect_complete_matrix(
    *,
    checkpoint_root: str | Path,
    feature_root: str | Path,
    probe_root: str | Path,
    config: Mapping[str, Any],
    preregistration_lock_sha256: str,
    source_provenance: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect exact selections, histories, fold metrics, and private OOF rows."""

    lock_sha256 = require_sha256(
        preregistration_lock_sha256, "preregistration lock"
    )
    checkpoints = Path(checkpoint_root).resolve()
    features = Path(feature_root).resolve()
    probes = Path(probe_root).resolve()
    arms = tuple(config["arms"])
    seeds = tuple(int(value) for value in config["training"]["seed_bases"])
    folds = tuple(int(value) for value in config["training"]["folds"])
    expected_cells = len(arms) * len(seeds) * len(folds)
    expected_source_keys = {
        "config_sha256",
        "stage_a_sentinel_sha256",
        "data_contract_sha256",
        "data_provenance_sha256",
    }
    if set(source_provenance) != expected_source_keys or any(
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in source_provenance.values()
    ):
        raise ValueError("canonical config/data source provenance is incomplete")
    _assert_owner_private_tree(checkpoints, "checkpoint")
    _assert_owner_private_tree(features, "feature")
    _assert_owner_private_tree(probes, "prediction")
    matrix_completion_path = checkpoints / "matrix_complete.json"
    matrix_completion = _read_json(matrix_completion_path, "matrix completion")
    matrix_lock = matrix_completion.get("preregistration")
    if (
        matrix_completion.get("status") != "COMPLETE"
        or int(matrix_completion.get("run_count", -1)) != expected_cells
        or not isinstance(matrix_lock, Mapping)
        or matrix_lock.get("status") != "PASS"
        or matrix_lock.get("lock_sha256") != lock_sha256
    ):
        raise ValueError("matrix completion is incomplete or uses another lock")
    if any(
        matrix_completion.get(key) != source_provenance[key]
        for key in (
            "config_sha256",
            "stage_a_sentinel_sha256",
            "data_contract_sha256",
        )
    ):
        raise ValueError("matrix completion canonical config/data provenance drifted")
    postprocessing_completion = _read_json(
        probes / "postprocessing_complete.private.json",
        "postprocessing completion",
    )
    if (
        postprocessing_completion.get("status") != "COMPLETE"
        or int(postprocessing_completion.get("cells", -1)) != expected_cells
        or postprocessing_completion.get("preregistration_lock")
        != "PREREGISTRATION_LOCK.json"
        or postprocessing_completion.get("preregistration_lock_sha256")
        != lock_sha256
        or postprocessing_completion.get("patient_level_outputs_private") is not True
    ):
        raise ValueError("postprocessing completion is incomplete or uses another lock")
    if any(
        postprocessing_completion.get(key) != value
        for key, value in source_provenance.items()
    ):
        raise ValueError(
            "postprocessing completion canonical config/data provenance drifted"
        )
    if postprocessing_completion.get("matrix_completion_sha256") != file_sha256(
        matrix_completion_path
    ):
        raise ValueError("postprocessing completion matrix hash drifted")
    matrix_runs = matrix_completion.get("runs")
    if not isinstance(matrix_runs, list) or len(matrix_runs) != expected_cells:
        raise ValueError("matrix completion run inventory is incomplete")
    observed_matrix_runs = {
        (int(row["seed_base"]), str(row["arm"]), int(row["fold"])): Path(
            str(row["selection_path"])
        ).resolve()
        for row in matrix_runs
        if isinstance(row, Mapping)
    }
    expected_matrix_runs = {
        (seed, arm, fold): _cell_path(checkpoints, seed, arm, fold)
        / "selection.json"
        for seed in seeds
        for arm in arms
        for fold in folds
    }
    if observed_matrix_runs != expected_matrix_runs:
        raise ValueError("matrix completion does not belong to checkpoint_root")
    selections: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    cell_chain: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in seeds:
        for arm in arms:
            for fold in folds:
                checkpoint = _cell_path(checkpoints, seed, arm, fold)
                feature = _cell_path(features, seed, arm, fold)
                probe = _cell_path(probes, seed, arm, fold)
                paths = {
                    "selection": checkpoint / "selection.json",
                    "history": checkpoint / "history.csv",
                    "selected": checkpoint / "selected.pt",
                    "feature": feature / "response_state.private.npz",
                    "feature_metadata": feature
                    / "response_state.private.metadata.json",
                    "probe_metadata": probe / "probe_metadata.json",
                    "metrics": probe / "probe_metrics.csv",
                    "predictions": probe / "ridge_predictions.private.csv",
                    "ridge_selection": probe / "ridge_selection.csv",
                }
                absent = [str(path) for path in paths.values() if not path.is_file()]
                if absent:
                    missing.extend(absent)
                    continue
                selection = _read_json(paths["selection"], "training selection")
                identity = {"arm": arm, "seed_base": seed, "fold": fold}
                if any(selection.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"training selection identity differs for {identity}")
                if selection.get("test_data_used") is not False:
                    raise ValueError("training selection reports test-data use")
                if (
                    selection.get("preregistration_status") != "PASS"
                    or selection.get("preregistration_lock_sha256") != lock_sha256
                    or selection.get("preregistration")
                    != {"status": "PASS", "lock_sha256": lock_sha256}
                ):
                    raise ValueError("training selection preregistration binding drifted")
                if selection.get("history_sha256") != file_sha256(paths["history"]):
                    raise ValueError(f"training history hash drifted for {identity}")
                if paths["selected"].stat().st_size <= 0:
                    raise ValueError(f"selected checkpoint is empty for {identity}")
                metadata = _read_json(paths["probe_metadata"], "probe metadata")
                if any(metadata.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"probe metadata identity differs for {identity}")
                if metadata.get("test_used_for_scaler_or_selection") is not False:
                    raise ValueError("probe metadata reports forbidden test fitting")
                if metadata.get("refit_after_alpha_selection") is not False:
                    raise ValueError("probe metadata reports post-selection refitting")
                if (
                    metadata.get("preregistration_lock")
                    != "PREREGISTRATION_LOCK.json"
                    or metadata.get("preregistration_lock_sha256") != lock_sha256
                ):
                    raise ValueError("probe metadata preregistration binding drifted")
                feature_metadata = _read_json(
                    paths["feature_metadata"], "feature metadata"
                )
                cell_chain.append(
                    _validate_cell_chain(
                        paths,
                        identity,
                        feature_metadata,
                        metadata,
                        lock_sha256,
                    )
                )
                hashes = metadata.get("output_sha256")
                if not isinstance(hashes, Mapping) or any(
                    hashes.get(path.name) != file_sha256(path)
                    for path in (
                        paths["metrics"],
                        paths["predictions"],
                        paths["ridge_selection"],
                    )
                ):
                    raise ValueError(f"probe output hashes drifted for {identity}")
                selections.append(selection)
                for key, container in (
                    ("history", histories),
                    ("metrics", metric_frames),
                    ("predictions", prediction_frames),
                ):
                    frame = pd.read_csv(paths[key])
                    for field, value in identity.items():
                        if field not in frame or not frame[field].eq(value).all():
                            raise ValueError(f"{key} identity differs at {field}")
                    container.append(frame)
                history = histories[-1]
                if history["epoch"].duplicated().any():
                    raise ValueError(f"training history repeats an epoch for {identity}")
                chosen = history.loc[
                    history["epoch"].eq(int(selection["selected_epoch"]))
                ]
                if len(chosen) != 1:
                    raise ValueError(f"selected epoch is absent from history for {identity}")
                selected_history = chosen.iloc[0]
                for selection_field, history_field in (
                    ("selected_validation_total_loss", "val_loss"),
                    ("selected_validation_base_loss", "val_base_objective"),
                    ("selected_validation_state_loss", "val_state_loss"),
                    ("selected_validation_ftv_loss", "val_ftv_loss"),
                    ("selected_representation_std", "val_representation_std"),
                ):
                    selected_value = float(selection[selection_field])
                    history_value = float(selected_history[history_field])
                    if not (
                        math.isnan(selected_value)
                        and math.isnan(history_value)
                    ) and not math.isclose(
                        selected_value,
                        history_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"selection/history metric differs at {selection_field}"
                        )
                if (
                    selection.get("delta_ftv_used") is not False
                    or selection.get("pcr_used") is not False
                ):
                    raise ValueError("training selection reports a forbidden downstream target")
                ridge = pd.read_csv(paths["ridge_selection"])
                for field, value in identity.items():
                    if field not in ridge or not ridge[field].eq(value).all():
                        raise ValueError(f"ridge selection identity differs at {field}")
                if len(ridge) != 2 * (4 + 3):
                    raise ValueError("ridge selection does not contain fourteen endpoints")
                for field in (
                    "test_used_for_scaler",
                    "test_used_for_alpha_selection",
                    "refit_after_alpha_selection",
                ):
                    if field not in ridge or not ridge[field].eq(False).all():
                        raise ValueError(f"ridge selection violates {field}")
                if not ridge["test_predict_call_count"].eq(1).all():
                    raise ValueError("ridge selection predicted outer test more than once")
    if missing:
        raise RuntimeError(
            f"pilot matrix is incomplete ({len(missing)} missing artifacts); "
            f"first missing: {missing[0]}"
        )
    if len(selections) != expected_cells:
        raise AssertionError("complete pilot matrix cardinality differs from 60")
    arm_order = {arm: index for index, arm in enumerate(arms)}
    cell_chain.sort(
        key=lambda row: (
            int(row["seed_base"]),
            int(row["fold"]),
            arm_order[str(row["arm"])],
        )
    )
    if postprocessing_completion.get("cell_chain") != cell_chain:
        raise ValueError("postprocessing completion cell chain differs from live assets")
    if postprocessing_completion.get("cell_chain_sha256") != canonical_sha256(
        cell_chain
    ):
        raise ValueError("postprocessing completion cell-chain hash drifted")
    selection_frame = pd.DataFrame(selections)
    history_frame = pd.concat(histories, ignore_index=True)
    metric_frame = pd.concat(metric_frames, ignore_index=True)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    _audit_matrix(selection_frame, history_frame, metric_frame, prediction_frame, config)
    return selection_frame, history_frame, metric_frame, prediction_frame


def _audit_matrix(
    selections: pd.DataFrame,
    histories: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    arms = set(config["arms"])
    seeds = set(int(value) for value in config["training"]["seed_bases"])
    folds = set(int(value) for value in config["training"]["folds"])
    expected = {(seed, arm, fold) for seed in seeds for arm in arms for fold in folds}
    for label, frame in (
        ("selection", selections),
        ("history", histories),
        ("metric", metrics),
        ("prediction", predictions),
    ):
        required = {"seed_base", "arm", "fold"}
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{label} table misses identity columns: {missing}")
        observed = set(
            frame[["seed_base", "arm", "fold"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if observed != expected:
            raise ValueError(f"{label} table does not cover the exact 60 cells")
    if selections.duplicated(["seed_base", "arm", "fold"]).any():
        raise ValueError("training selections contain duplicate cells")
    expected_metric_grid: set[tuple[Any, ...]] = set()
    expected_prediction_grid: set[tuple[Any, ...]] = set()
    for seed, arm, fold in expected:
        for scope in ANALYSIS_SCOPES:
            for task, endpoints, scale, semantics in (
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
            ):
                for endpoint in endpoints:
                    for current_scale in ("natural", scale):
                        expected_metric_grid.add(
                            (
                                seed,
                                arm,
                                fold,
                                scope,
                                task,
                                endpoint,
                                semantics,
                                current_scale,
                            )
                        )
                    expected_prediction_grid.add(
                        (
                            seed,
                            arm,
                            fold,
                            scope,
                            task,
                            endpoint,
                            semantics,
                            scale,
                        )
                    )
    metric_keys = [
        "seed_base",
        "arm",
        "fold",
        "analysis_scope",
        "task",
        "endpoint",
        "target_semantics",
        "scale",
    ]
    if missing := sorted(set(metric_keys + ["n_test"]).difference(metrics.columns)):
        raise ValueError(f"fold probe metrics miss contract columns: {missing}")
    if metrics.duplicated(metric_keys).any():
        raise ValueError("fold probe metrics contain duplicate endpoint rows")
    observed_metric_grid = set(
        metrics[metric_keys].itertuples(index=False, name=None)
    )
    if observed_metric_grid != expected_metric_grid:
        raise ValueError("fold probe metric grid differs from the exact pilot contract")
    prediction_grid_keys = [
        "seed_base",
        "arm",
        "fold",
        "analysis_scope",
        "task",
        "endpoint",
        "target_semantics",
        "analysis_scale",
    ]
    prediction_required = {
        *prediction_grid_keys,
        "patient_id",
        "split",
        "y_true",
        "y_pred",
        "b0_prediction",
        "y_true_analysis",
        "y_pred_analysis",
        "b0_prediction_analysis",
        "test_predict_call_count",
    }
    if missing := sorted(prediction_required.difference(predictions.columns)):
        raise ValueError(f"OOF predictions miss contract columns: {missing}")
    observed_prediction_grid = set(
        predictions[prediction_grid_keys]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_prediction_grid != expected_prediction_grid:
        raise ValueError("OOF prediction grid differs from the exact pilot contract")
    if not predictions["split"].eq("test").all():
        raise ValueError("OOF prediction files contain non-test rows")
    if not predictions["test_predict_call_count"].eq(1).all():
        raise ValueError("one or more endpoint probes predicted outer test more than once")
    identity = [
        "seed_base",
        "arm",
        "analysis_scope",
        "task",
        "endpoint",
        "patient_id",
    ]
    if predictions.duplicated(identity).any():
        raise ValueError("a patient appears in multiple outer-test folds")
    count_keys = [
        "seed_base",
        "arm",
        "fold",
        "analysis_scope",
        "task",
        "endpoint",
        "target_semantics",
    ]
    expected_counts = (
        metrics.loc[metrics["scale"].eq("natural")]
        .set_index(count_keys)["n_test"]
        .astype(int)
        .sort_index()
    )
    observed_counts = predictions.groupby(count_keys, sort=False).size().sort_index()
    if (
        expected_counts.index.duplicated().any()
        or not expected_counts.index.equals(observed_counts.index)
        or not expected_counts.equals(observed_counts)
    ):
        raise ValueError("OOF row counts differ from natural fold metrics")
    paired = [
        "seed_base",
        "analysis_scope",
        "task",
        "endpoint",
        "target_semantics",
        "patient_id",
    ]
    for keys, group in predictions.groupby(paired, sort=False):
        if set(group["arm"]) != arms or len(group) != len(arms):
            raise ValueError(f"paired OOF arm coverage differs for {keys}")
        for column in (
            "fold",
            "analysis_scale",
            "y_true",
            "b0_prediction",
            "y_true_analysis",
            "b0_prediction_analysis",
        ):
            values = group[column].to_numpy()
            if column in {"fold", "analysis_scale"}:
                aligned = np.all(values == values[0])
            else:
                aligned = np.allclose(
                    values.astype(float), float(values[0]), rtol=0.0, atol=1e-12
                )
            if not aligned:
                raise ValueError(f"paired OOF {column} differs across arms for {keys}")


def pooled_natural_metrics(
    predictions: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Pool five OOF folds, compute natural endpoints, then endpoint macros."""

    required = {
        "patient_id",
        "seed_base",
        "arm",
        "fold",
        "analysis_scope",
        "task",
        "endpoint",
        "target_semantics",
        "y_true",
        "y_pred",
        "b0_prediction",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"OOF predictions miss required columns: {missing}")
    folds = set(int(value) for value in config["training"]["folds"])
    keys = [
        "seed_base",
        "arm",
        "analysis_scope",
        "task",
        "endpoint",
        "target_semantics",
    ]
    rows: list[dict[str, Any]] = []
    for identity, group in predictions.groupby(keys, sort=False):
        if set(group["fold"].astype(int)) != folds:
            raise ValueError(f"pooled OOF group does not contain all five folds: {identity}")
        if group["patient_id"].astype(str).duplicated().any():
            raise ValueError(f"pooled OOF group repeats a patient: {identity}")
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "scale": "natural",
                "aggregation": "pooled_5fold_oof",
                "n_test": len(group),
                **metric_values(
                    group["y_true"].to_numpy(float),
                    group["y_pred"].to_numpy(float),
                    group["b0_prediction"].to_numpy(float),
                ),
            }
        )
    endpoint_frame = pd.DataFrame(rows)
    expected_endpoints = {
        "static": set(config["probes"]["static_endpoints"][:-1]),
        "delta": set(config["probes"]["delta_endpoints"][:-1]),
    }
    macros: list[dict[str, Any]] = []
    macro_keys = ["seed_base", "arm", "analysis_scope", "task", "target_semantics"]
    for identity, group in endpoint_frame.groupby(macro_keys, sort=False):
        task = str(identity[3])
        if set(group["endpoint"]) != expected_endpoints[task]:
            raise ValueError(f"natural endpoint coverage differs for {identity}")
        macros.append(
            {
                **dict(zip(macro_keys, identity, strict=True)),
                "endpoint": "macro",
                "scale": "natural",
                "aggregation": "unweighted_mean_of_pooled_endpoint_metrics",
                "n_test": int(group["n_test"].sum()),
                **{metric: float(group[metric].mean()) for metric in METRICS},
            }
        )
    return pd.concat([endpoint_frame, pd.DataFrame(macros)], ignore_index=True)


def transformed_fold_summaries(
    fold_metrics: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Average metrics across fold-specific transforms without pooling rows."""

    transformed = fold_metrics.loc[~fold_metrics["scale"].eq("natural")].copy()
    keys = [
        "seed_base",
        "arm",
        "analysis_scope",
        "task",
        "endpoint",
        "target_semantics",
        "scale",
    ]
    rows: list[dict[str, Any]] = []
    for identity, group in transformed.groupby(keys, sort=False):
        if set(group["fold"].astype(int)) != set(config["training"]["folds"]):
            raise ValueError(f"transformed summary lacks five folds: {identity}")
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "aggregation": "mean_of_5_fold_specific_transform_metrics",
                "n_folds": int(group["fold"].nunique()),
                "n_test": int(group["n_test"].sum()),
                **{metric: float(group[metric].mean()) for metric in METRICS},
            }
        )
    endpoints = pd.DataFrame(rows)
    expected_endpoints = {
        "static": set(config["probes"]["static_endpoints"][:-1]),
        "delta": set(config["probes"]["delta_endpoints"][:-1]),
    }
    macro_keys = [
        "seed_base",
        "arm",
        "analysis_scope",
        "task",
        "target_semantics",
        "scale",
    ]
    macros: list[dict[str, Any]] = []
    for identity, group in endpoints.groupby(macro_keys, sort=False):
        task = str(identity[3])
        if set(group["endpoint"]) != expected_endpoints[task]:
            raise ValueError(f"transformed endpoint coverage differs for {identity}")
        macros.append(
            {
                **dict(zip(macro_keys, identity, strict=True)),
                "endpoint": "macro",
                "aggregation": "unweighted_endpoint_mean_of_fold_summaries",
                "n_folds": 5,
                "n_test": int(group["n_test"].sum()),
                **{metric: float(group[metric].mean()) for metric in METRICS},
            }
        )
    return pd.concat([endpoints, pd.DataFrame(macros)], ignore_index=True)


def architecture_table(config: Mapping[str, Any]) -> pd.DataFrame:
    from .model import build_model

    rows: list[dict[str, Any]] = []
    for arm, spec in config["arms"].items():
        input_dim = 256 if arm.startswith("LG") else 128
        projection_parameters = input_dim * 192 + 192
        layer_norm_parameters = 2 * 192
        model = build_model(arm, 2026)
        counts = model.parameter_counts()
        expected_projection = projection_parameters + layer_norm_parameters
        if counts["response_projection"] != expected_projection:
            raise AssertionError(f"model/config projection count differs for {arm}")
        rows.append(
            {
                "arm": arm,
                "pooling": spec["pooling"],
                "response_dimension": 192,
                "trainable_projection": spec["projection"],
                "projection_input_dimension": input_dim,
                "parameter_count": counts["trainable_total"],
                "parameter_count_scope": "trainable_total",
                "response_projection_parameter_count": expected_projection,
                "response_projection_parameter_count_scope": (
                    "online_response_projection_including_layernorm"
                ),
                "ftv_head_parameter_count": counts["ftv_head"],
                "frozen_parameter_count": counts["frozen_total"],
                "all_model_parameter_count": counts["all_model_parameters"],
                "grounded": bool(spec["grounded"]),
            }
        )
        del model
    return pd.DataFrame(rows)


def task_table(
    natural: pd.DataFrame,
    transformed: pd.DataFrame,
    *,
    task: str,
    scope: str = PRIMARY_SCOPE,
) -> pd.DataFrame:
    natural_rows = natural.loc[
        natural["task"].eq(task) & natural["analysis_scope"].eq(scope)
    ].copy()
    transformed_rows = transformed.loc[
        transformed["task"].eq(task) & transformed["analysis_scope"].eq(scope)
    ].copy()
    keys = ["seed_base", "arm", "task", "endpoint", "analysis_scope"]
    natural_keep = keys + ["n_test", "aggregation", *METRICS]
    transformed_keep = keys + ["n_folds", "aggregation", *METRICS]
    natural_rows = natural_rows[natural_keep].rename(
        columns={
            "n_test": "natural_n_test",
            "aggregation": "natural_aggregation",
            **{name: f"natural_{name}" for name in METRICS},
        }
    )
    transformed_rows = transformed_rows[transformed_keep].rename(
        columns={
            "n_folds": "transformed_n_folds",
            "aggregation": "transformed_aggregation",
            **{name: f"transformed_{name}" for name in METRICS},
        }
    )
    merged = natural_rows.merge(
        transformed_rows, on=keys, how="inner", validate="one_to_one"
    )
    if len(merged) != len(natural_rows) or len(merged) != len(transformed_rows):
        raise ValueError(f"natural/transformed {task} table keys do not align")
    arm_order = {arm: index for index, arm in enumerate(CANONICAL_ARMS)}
    endpoint_order = {
        endpoint: index
        for index, endpoint in enumerate(
            ("T0", "T1", "T2", "T3", "T0_to_T1", "T1_to_T2", "T2_to_T3", "macro")
        )
    }
    merged["_arm_order"] = merged["arm"].map(arm_order)
    merged["_endpoint_order"] = merged["endpoint"].map(endpoint_order)
    return merged.sort_values(
        ["seed_base", "_arm_order", "_endpoint_order"]
    ).drop(columns=["_arm_order", "_endpoint_order"]).reset_index(drop=True)


def effect_table(
    task_tables: Sequence[pd.DataFrame],
    comparisons: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    source = pd.concat(task_tables, ignore_index=True)
    identity = ["seed_base", "task", "endpoint", "analysis_scope"]
    metric_columns = [
        column
        for column in source.columns
        if column.startswith("natural_") or column.startswith("transformed_")
    ]
    metric_columns = [
        column
        for column in metric_columns
        if column
        not in {
            "natural_n_test",
            "natural_aggregation",
            "transformed_n_folds",
            "transformed_aggregation",
        }
    ]
    rows: list[dict[str, Any]] = []
    indexed = source.set_index(identity + ["arm"])
    for label, left, right in comparisons:
        for key in source[identity].drop_duplicates().itertuples(index=False, name=None):
            try:
                left_row = indexed.loc[(*key, left)]
                right_row = indexed.loc[(*key, right)]
            except KeyError as error:
                raise ValueError(f"effect comparison {label} lacks a paired row") from error
            rows.append(
                {
                    **dict(zip(identity, key, strict=True)),
                    "comparison": label,
                    "left_arm": left,
                    "right_arm": right,
                    **{
                        f"effect_{metric}": float(left_row[metric] - right_row[metric])
                        for metric in metric_columns
                    },
                }
            )
    return pd.DataFrame(rows)


def optimization_safety_table(
    selections: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    threshold = float(
        config["gates"]["D_OPTIMIZATION_SAFETY"][
            "maximum_state_loss_degradation_fraction"
        ]
    )
    pairs = (
        ("GAP3-GAP0", "GAP3", "GAP0"),
        ("LOCAL3-LOCAL0", "LOCAL3", "LOCAL0"),
        ("LG3-LG0", "LG3", "LG0"),
    )
    required = {
        "seed_base",
        "arm",
        "fold",
        "selected_validation_state_loss",
        "selection_mode",
    }
    if missing := sorted(required.difference(selections.columns)):
        raise ValueError(f"selection table misses safety fields: {missing}")
    indexed = selections.set_index(["seed_base", "fold", "arm"])
    rows: list[dict[str, Any]] = []
    for seed in config["training"]["seed_bases"]:
        for fold in config["training"]["folds"]:
            for label, grounded, baseline in pairs:
                base_row = indexed.loc[(seed, fold, baseline)]
                grounded_row = indexed.loc[(seed, fold, grounded)]
                base_loss = _finite_number(
                    base_row["selected_validation_state_loss"], "baseline state loss"
                )
                grounded_loss = _finite_number(
                    grounded_row["selected_validation_state_loss"],
                    "grounded state loss",
                )
                if base_loss <= 0:
                    raise ValueError("paired baseline state loss must be positive")
                degradation = grounded_loss / base_loss - 1.0
                rows.append(
                    {
                        "seed_base": int(seed),
                        "fold": int(fold),
                        "comparison": label,
                        "grounded_arm": grounded,
                        "baseline_arm": baseline,
                        "baseline_selected_validation_state_loss": base_loss,
                        "grounded_selected_validation_state_loss": grounded_loss,
                        "state_loss_degradation_fraction": degradation,
                        "maximum_allowed_degradation_fraction": threshold,
                        "safety_pass": bool(degradation <= threshold + 1e-12),
                        "grounded_selection_mode": grounded_row["selection_mode"],
                    }
                )
    return pd.DataFrame(rows)


def prediction_calibration_table(
    static: pd.DataFrame, delta: pd.DataFrame
) -> pd.DataFrame:
    source = pd.concat([static, delta], ignore_index=True)
    columns = [
        "seed_base",
        "arm",
        "task",
        "endpoint",
        "analysis_scope",
        "natural_n_test",
        "natural_prediction_target_variance_ratio",
        "natural_calibration_slope",
        "natural_calibration_intercept",
        "natural_calibration_mean_bias",
        "natural_r2",
        "natural_spearman",
    ]
    return source[columns].copy()


def paired_fold_effects(
    fold_metrics: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Descriptive fold sensitivity; folds are not treated as replicates."""

    natural = fold_metrics.loc[
        fold_metrics["scale"].eq("natural")
        & fold_metrics["analysis_scope"].eq(PRIMARY_SCOPE)
    ].copy()
    macro_keys = ["seed_base", "arm", "fold", "task"]
    macros = natural.groupby(macro_keys, as_index=False)[list(METRICS)].mean()
    indexed = macros.set_index(macro_keys)
    comparisons = (
        ("LOCAL0-GAP0", "LOCAL0", "GAP0"),
        ("LG0-LOCAL0", "LG0", "LOCAL0"),
        ("LOCAL3-GAP3", "LOCAL3", "GAP3"),
        ("LG3-LOCAL3", "LG3", "LOCAL3"),
        ("GAP3-GAP0", "GAP3", "GAP0"),
        ("LOCAL3-LOCAL0", "LOCAL3", "LOCAL0"),
        ("LG3-LG0", "LG3", "LG0"),
    )
    rows: list[dict[str, Any]] = []
    for seed in config["training"]["seed_bases"]:
        for fold in config["training"]["folds"]:
            for task in ("static", "delta"):
                for label, left, right in comparisons:
                    left_row = indexed.loc[(seed, left, fold, task)]
                    right_row = indexed.loc[(seed, right, fold, task)]
                    rows.append(
                        {
                            "seed_base": int(seed),
                            "fold": int(fold),
                            "task": task,
                            "comparison": label,
                            "left_arm": left,
                            "right_arm": right,
                            **{
                                f"effect_{metric}": float(
                                    left_row[metric] - right_row[metric]
                                )
                                for metric in METRICS
                            },
                        }
                    )
    return pd.DataFrame(rows)


def _macro_metric(table: pd.DataFrame, arm: str, metric: str) -> dict[int, float]:
    rows = table.loc[table["arm"].eq(arm) & table["endpoint"].eq("macro")]
    observed = {
        int(row.seed_base): float(getattr(row, metric))
        for row in rows.itertuples(index=False)
    }
    if set(observed) != {2026, 3026}:
        raise ValueError(f"macro metric {arm}/{metric} does not cover both seeds")
    if not all(math.isfinite(value) for value in observed.values()):
        raise ValueError(f"macro metric {arm}/{metric} must be finite in both seeds")
    return observed


def _effect(left: Mapping[int, float], right: Mapping[int, float]) -> dict[int, float]:
    return {seed: float(left[seed] - right[seed]) for seed in sorted(left)}


def evaluate_gates(
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table6: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_scientific_contract(config)
    gates = config["gates"]
    static_spearman = {
        arm: _macro_metric(table2, arm, "natural_spearman") for arm in CANONICAL_ARMS
    }
    static_r2 = {
        arm: _macro_metric(table2, arm, "natural_r2") for arm in CANONICAL_ARMS
    }
    delta_spearman = {
        arm: _macro_metric(table3, arm, "natural_spearman") for arm in CANONICAL_ARMS
    }

    a_static = _effect(static_spearman["LOCAL0"], static_spearman["GAP0"])
    a_delta = _effect(delta_spearman["LOCAL0"], delta_spearman["GAP0"])
    a_r2 = _effect(static_r2["LOCAL0"], static_r2["GAP0"])
    a_cfg = gates["A_LOCAL_STATE_WORKS"]
    a_checks = {
        "static_each_seed": all(
            value >= float(a_cfg["static_macro_spearman_gain_each_seed_min"])
            for value in a_static.values()
        ),
        "delta_each_seed": all(
            value
            > float(a_cfg["delta_macro_spearman_gain_each_seed_strictly_gt"])
            for value in a_delta.values()
        ),
        "natural_r2_not_systematically_worse": not all(
            value < 0 for value in a_r2.values()
        ),
    }
    gate_a = all(a_checks.values())

    b_static = _effect(static_spearman["LG0"], static_spearman["LOCAL0"])
    b_r2 = _effect(static_r2["LG0"], static_r2["LOCAL0"])
    b_cfg = gates["B_LOCAL_GLOBAL_ADDS_VALUE"]
    b_checks = {
        "static_each_seed_nonnegative": all(
            value >= float(b_cfg["static_macro_spearman_gain_each_seed_min"])
            for value in b_static.values()
        ),
        "static_at_least_one_seed_meaningful": any(
            value
            >= float(b_cfg["static_macro_spearman_gain_at_least_one_seed_min"])
            for value in b_static.values()
        ),
        "natural_r2_not_systematically_worse": not all(
            value < 0 for value in b_r2.values()
        ),
    }
    gate_b = all(b_checks.values())

    if gate_a and gate_b:
        winner = "LOCAL_GLOBAL"
        candidate_base, candidate_grounded = "LG0", "LG3"
    elif gate_a:
        winner = "LOCAL"
        candidate_base, candidate_grounded = "LOCAL0", "LOCAL3"
    else:
        winner = "GAP"
        candidate_base, candidate_grounded = "GAP0", "GAP3"

    c_static = _effect(
        static_spearman[candidate_grounded], static_spearman[candidate_base]
    )
    c_delta = _effect(
        delta_spearman[candidate_grounded], delta_spearman[candidate_base]
    )
    c_r2 = _effect(static_r2[candidate_grounded], static_r2[candidate_base])
    c_cfg = gates["C_GROUNDING_COMPATIBILITY"]
    c_checks = {
        "static_each_seed_positive": all(
            value
            > float(c_cfg["candidate_static_macro_spearman_gain_each_seed_strictly_gt"])
            for value in c_static.values()
        ),
        "delta_at_least_one_seed_meaningful": any(
            value
            >= float(
                c_cfg["candidate_delta_macro_spearman_gain_at_least_one_seed_min"]
            )
            for value in c_delta.values()
        ),
        "natural_r2_not_systematically_worse": not all(
            value < 0 for value in c_r2.values()
        ),
    }
    gate_c = all(c_checks.values())

    d_cfg = gates["D_OPTIMIZATION_SAFETY"]
    comparison = f"{candidate_grounded}-{candidate_base}"
    candidate_safety = table6.loc[table6["comparison"].eq(comparison)]
    expected_total = int(d_cfg["candidate_paired_folds_total"])
    if len(candidate_safety) != expected_total:
        raise ValueError("candidate safety table does not contain exactly ten folds")
    safety_passes = int(candidate_safety["safety_pass"].sum())
    required_passes = int(d_cfg["candidate_paired_folds_required"])
    d_checks = {
        "paired_fold_count_exact": len(candidate_safety) == expected_total,
        "minimum_safe_folds": safety_passes >= required_passes,
    }
    gate_d = all(d_checks.values())

    if not gate_a:
        classification = "C. FROZEN ADVANTAGE DID NOT TRANSFER"
    elif not gate_d:
        classification = "D. REPRESENTATION IMPROVED BUT GROUNDING UNSAFE"
    elif gate_b:
        classification = "B. LOCAL-GLOBAL STATE VALIDATED IN PILOT"
    else:
        classification = "A. LOCAL STATE VALIDATED IN PILOT"

    return {
        "schema_version": 1,
        "pilot_only_not_statistical_significance": True,
        "folds_are_paired_sensitivity_not_independent_replicates": True,
        "winner": winner,
        "candidate_base_arm": candidate_base,
        "candidate_grounded_arm": candidate_grounded,
        "classification": classification,
        "gates": {
            "A_LOCAL_STATE_WORKS": {
                "pass": gate_a,
                "thresholds": dict(a_cfg),
                "checks": a_checks,
                "static_macro_spearman_effect_by_seed": a_static,
                "delta_macro_spearman_effect_by_seed": a_delta,
                "static_natural_r2_effect_by_seed": a_r2,
            },
            "B_LOCAL_GLOBAL_ADDS_VALUE": {
                "pass": gate_b,
                "thresholds": dict(b_cfg),
                "checks": b_checks,
                "static_macro_spearman_effect_by_seed": b_static,
                "static_natural_r2_effect_by_seed": b_r2,
            },
            "C_GROUNDING_COMPATIBILITY": {
                "pass": gate_c,
                "thresholds": dict(c_cfg),
                "checks": c_checks,
                "candidate_comparison": comparison,
                "static_macro_spearman_effect_by_seed": c_static,
                "delta_macro_spearman_effect_by_seed": c_delta,
                "static_natural_r2_effect_by_seed": c_r2,
            },
            "D_OPTIMIZATION_SAFETY": {
                "pass": gate_d,
                "thresholds": dict(d_cfg),
                "checks": d_checks,
                "candidate_comparison": comparison,
                "safe_folds": safety_passes,
                "total_folds": expected_total,
                "required_safe_folds": required_passes,
            },
        },
        "next_stage": {
            "direct_ftv_plus_ld_authorized_by_this_pilot": False,
            "selected_architecture_requires_multiseed_confirmation_first": bool(gate_a),
            "pilot_supports_ftv_plus_ld_after_confirmation": bool(
                gate_a and gate_c and gate_d
            ),
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        observed = float(value)
        return observed if math.isfinite(observed) else None
    return value


def _assert_public_frame(frame: pd.DataFrame, label: str) -> None:
    forbidden_columns = {"patient_id", "cache_path", "checkpoint_path", "feature_path"}
    if forbidden := sorted(forbidden_columns.intersection(frame.columns)):
        raise ValueError(f"public {label} contains identifier/path columns: {forbidden}")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype(str)
        if values.str.contains(r"/(?:data|home)/", regex=True).any():
            raise ValueError(f"public {label} contains an absolute private path")


def _atomic_public_csv(path: Path, frame: pd.DataFrame) -> None:
    _assert_public_frame(frame, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, 0o644)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_public_json(path: Path, payload: Mapping[str, Any]) -> None:
    safe = _json_safe(payload)
    encoded = json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if "/data/" in encoded or "/home/" in encoded:
        raise ValueError(f"public JSON contains an absolute private path: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_report_context(decision: Mapping[str, Any]) -> dict[str, Any]:
    gates = decision["gates"]
    candidate_supported_for_later_ftv_ld = bool(
        decision["next_stage"]["pilot_supports_ftv_plus_ld_after_confirmation"]
    )
    return {
        "schema_version": 1,
        "language_for_final_report": "zh-CN",
        "final_report_prose_written": False,
        "winner": decision["winner"],
        "classification": decision["classification"],
        "answers": {
            "1_local0_stably_better_than_gap0": gates["A_LOCAL_STATE_WORKS"]["pass"],
            "2_frozen_local_advantage_transferred": gates["A_LOCAL_STATE_WORKS"]["pass"],
            "3_lg0_stably_better_than_local0": gates[
                "B_LOCAL_GLOBAL_ADDS_VALUE"
            ]["pass"],
            "4_global_branch_necessary": decision["winner"] == "LOCAL_GLOBAL",
            "5_ftv_grounding_effective_for_candidate": gates[
                "C_GROUNDING_COMPATIBILITY"
            ]["pass"],
            "5_grounding_evidence_table": TABLE_FILENAMES["table5"],
            "6_observed_delta_ftv_improvement": {
                "local0_minus_gap0_macro_spearman_by_seed": gates[
                    "A_LOCAL_STATE_WORKS"
                ]["delta_macro_spearman_effect_by_seed"],
                "grounded_candidate_minus_base_macro_spearman_by_seed": gates[
                    "C_GROUNDING_COMPATIBILITY"
                ]["delta_macro_spearman_effect_by_seed"],
            },
            "7_natural_scale_r2_improvement": {
                "local0_minus_gap0_static_macro_by_seed": gates[
                    "A_LOCAL_STATE_WORKS"
                ]["static_natural_r2_effect_by_seed"],
                "lg0_minus_local0_static_macro_by_seed": gates[
                    "B_LOCAL_GLOBAL_ADDS_VALUE"
                ]["static_natural_r2_effect_by_seed"],
                "grounded_candidate_minus_base_static_macro_by_seed": gates[
                    "C_GROUNDING_COMPATIBILITY"
                ]["static_natural_r2_effect_by_seed"],
            },
            "8_prediction_compression_relief_evidence_table": TABLE_FILENAMES[
                "table7"
            ],
            "9_grounded_optimization_safety": gates[
                "D_OPTIMIZATION_SAFETY"
            ],
            "10_selected_architecture": decision["winner"],
            "11_spatial_aggregation_interpretation": (
                "SUPPORTED_AS_END_TO_END_BOTTLENECK_IN_PILOT"
                if gates["A_LOCAL_STATE_WORKS"]["pass"]
                else "FROZEN_ADVANTAGE_DID_NOT_TRANSFER;_CAUSE_REQUIRES_REANALYSIS"
            ),
            "12_direct_entry_to_ftv_plus_ld": False,
            "13_ftv_plus_ld_architecture_after_confirmation": (
                decision["winner"] if candidate_supported_for_later_ftv_ld else None
            ),
        },
        "next_stage": decision["next_stage"],
    }


def aggregate_pilot(
    *,
    checkpoint_root: str | Path,
    feature_root: str | Path,
    probe_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    preregistration_lock_sha256: str,
    source_provenance: Mapping[str, str],
) -> dict[str, Any]:
    """Create Tables 1-7, ten figures, and machine-readable decisions."""

    lock_sha256 = require_sha256(
        preregistration_lock_sha256, "preregistration lock"
    )
    config = load_pilot_config(config_path)
    selections, histories, fold_metrics, predictions = collect_complete_matrix(
        checkpoint_root=checkpoint_root,
        feature_root=feature_root,
        probe_root=probe_root,
        config=config,
        preregistration_lock_sha256=lock_sha256,
        source_provenance=source_provenance,
    )
    natural = pooled_natural_metrics(predictions, config)
    transformed = transformed_fold_summaries(fold_metrics, config)
    table1 = architecture_table(config)
    table2 = task_table(natural, transformed, task="static")
    table3 = task_table(natural, transformed, task="delta")
    table4 = effect_table(
        (table2, table3),
        (
            ("LOCAL0-GAP0", "LOCAL0", "GAP0"),
            ("LG0-LOCAL0", "LG0", "LOCAL0"),
            ("LOCAL3-GAP3", "LOCAL3", "GAP3"),
            ("LG3-LOCAL3", "LG3", "LOCAL3"),
        ),
    )
    table5 = effect_table(
        (table2, table3),
        (
            ("GAP3-GAP0", "GAP3", "GAP0"),
            ("LOCAL3-LOCAL0", "LOCAL3", "LOCAL0"),
            ("LG3-LG0", "LG3", "LG0"),
        ),
    )
    table6 = optimization_safety_table(selections, config)
    table7 = prediction_calibration_table(table2, table3)
    fold_effects = paired_fold_effects(fold_metrics, config)
    decision = evaluate_gates(table2, table3, table6, config)
    report_context = build_report_context(decision)

    outputs = Path(output_dir).resolve()
    figures = Path(figure_dir).resolve()
    tables = {
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "table5": table5,
        "table6": table6,
        "table7": table7,
    }
    destinations = [outputs / TABLE_FILENAMES[key] for key in tables]
    destinations += [
        outputs / "natural_pooled_metrics.csv",
        outputs / "transformed_fold_summaries.csv",
        outputs / "paired_fold_effects.csv",
        outputs / "training_trajectories.csv",
        outputs / "decision_summary.json",
        outputs / "report_context.json",
        outputs / "aggregation_summary.json",
    ]
    from .figures import FIGURE_FILENAMES, render_required_figures

    destinations += [figures / name for name in FIGURE_FILENAMES]
    if existing := [str(path) for path in destinations if path.exists()]:
        raise FileExistsError(f"refusing to overwrite formal aggregation output: {existing[0]}")
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for key, frame in tables.items():
        _atomic_public_csv(outputs / TABLE_FILENAMES[key], frame)
    _atomic_public_csv(outputs / "natural_pooled_metrics.csv", natural)
    _atomic_public_csv(outputs / "transformed_fold_summaries.csv", transformed)
    _atomic_public_csv(outputs / "paired_fold_effects.csv", fold_effects)
    public_histories = histories.copy()
    path_columns = [column for column in public_histories if "path" in column.lower()]
    public_histories = public_histories.drop(columns=path_columns, errors="ignore")
    _atomic_public_csv(outputs / "training_trajectories.csv", public_histories)
    _atomic_public_json(outputs / "decision_summary.json", decision)
    _atomic_public_json(outputs / "report_context.json", report_context)

    render_required_figures(
        table1=table1,
        table2=table2,
        table3=table3,
        table4=table4,
        table6=table6,
        fold_effects=fold_effects,
        histories=public_histories,
        output_dir=figures,
    )
    artifact_paths = {
        path.name: file_sha256(path)
        for path in [
            *(outputs / TABLE_FILENAMES[key] for key in tables),
            outputs / "natural_pooled_metrics.csv",
            outputs / "transformed_fold_summaries.csv",
            outputs / "paired_fold_effects.csv",
            outputs / "training_trajectories.csv",
            outputs / "decision_summary.json",
            outputs / "report_context.json",
            *(figures / name for name in FIGURE_FILENAMES),
        ]
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_cells": 60,
        "natural_aggregation": "pool_5fold_oof_then_unweighted_endpoint_macro",
        "transformed_aggregation": "fold_metric_summaries_only",
        "patient_level_outputs_private": True,
        "public_outputs_deidentified": True,
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": lock_sha256,
        **dict(source_provenance),
        "decision": decision,
        "artifact_sha256": artifact_paths,
    }
    _atomic_public_json(outputs / "aggregation_summary.json", summary)
    return summary


__all__ = [
    "CANONICAL_ARMS",
    "METRICS",
    "PRIMARY_SCOPE",
    "TABLE_FILENAMES",
    "aggregate_pilot",
    "architecture_table",
    "build_report_context",
    "collect_complete_matrix",
    "effect_table",
    "evaluate_gates",
    "load_pilot_config",
    "optimization_safety_table",
    "paired_fold_effects",
    "pooled_natural_metrics",
    "prediction_calibration_table",
    "task_table",
    "transformed_fold_summaries",
]
