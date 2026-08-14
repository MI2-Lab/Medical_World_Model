"""Exact, fold-matched FTV probes for frozen response-state assets.

This module mirrors the confirmed LOCAL3 probe contract while accepting both
Goal-F factorized assets and the unseparated F0 control.  It is deliberately
outcome-free: its only target input is a mapping of sealed Stage-B
``FTVRecord`` objects.

The public result is a table of natural-scale, five-fold pooled OOF metrics.
The second returned table contains patient-level OOF values and is private by
contract; callers must keep it below the experiment's ignored
``predictions/`` tree if they persist it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .stageb import fit_grounding_transform


TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0_to_T1", "T1_to_T2", "T2_to_T3")
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class SelectedRidge:
    """Train-fitted preprocessing and validation-selected Ridge model."""

    x_scaler: StandardScaler
    y_scaler: StandardScaler | None
    model: Ridge
    alpha: float
    validation_mse_analysis: float
    alpha_grid: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _NormalizedAsset:
    patient_id: np.ndarray
    split: np.ndarray
    states: Mapping[str, np.ndarray]
    arm: str
    source_arm: str
    seed_base: int
    fold: int


@dataclass(frozen=True)
class _PreparedSplit:
    patient_id: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray


def _field(asset: Any, name: str, default: Any = None) -> Any:
    if isinstance(asset, Mapping):
        return asset.get(name, default)
    return getattr(asset, name, default)


def _scalar(value: Any, name: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"asset {name} must be scalar")
    return array.item()


def _normalize_asset(asset: Any) -> _NormalizedAsset:
    patient_id = np.asarray(_field(asset, "patient_id"))
    split = np.asarray(_field(asset, "split"))
    if patient_id.ndim != 1 or not len(patient_id):
        raise ValueError("asset patient_id must be a nonempty vector")
    patient_id = patient_id.astype(str)
    split = split.astype(str)
    if split.shape != patient_id.shape or set(split) != {"train", "val", "test"}:
        raise ValueError("asset split must align and contain train/val/test")
    if len(set(patient_id)) != len(patient_id):
        raise ValueError("asset contains duplicate patient IDs")

    state = _field(asset, "state")
    response_state = _field(asset, "response_state")
    z_r = _field(asset, "z_R")
    if state is not None or response_state is not None:
        raw_states = {"F0": state if state is not None else response_state}
        source_arm_value = _field(asset, "arm", "F0")
        source_arm = str(_scalar(source_arm_value, "arm"))
        arm = "F0"
    elif z_r is not None:
        z_p = _field(asset, "z_P")
        full = _field(asset, "full")
        if z_p is None:
            raise ValueError("factorized asset is missing z_P")
        raw_states = {"z_R": z_r, "z_P": z_p}
        if full is None:
            full = np.concatenate((np.asarray(z_r), np.asarray(z_p)), axis=-1)
        raw_states["full"] = full
        source_arm_value = _field(asset, "arm")
        if source_arm_value is None:
            raise ValueError("factorized asset is missing arm")
        arm = source_arm = str(_scalar(source_arm_value, "arm"))
    else:
        raise ValueError("asset has neither F0 state nor factorized z_R/z_P states")

    states: dict[str, np.ndarray] = {}
    for name, value in raw_states.items():
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[:2] != (len(patient_id), 4) or not array.shape[2]:
            raise ValueError(f"asset state {name} must have shape [N,4,D]")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"asset state {name} must be finite floating point")
        states[name] = array
    if "full" in states:
        expected = np.concatenate((states["z_R"], states["z_P"]), axis=-1)
        if states["full"].shape != expected.shape or not np.array_equal(states["full"], expected):
            raise ValueError("factorized full state is not exact [z_R,z_P] concatenation")

    seed = int(_scalar(_field(asset, "seed_base"), "seed_base"))
    fold = int(_scalar(_field(asset, "fold"), "fold"))
    if fold not in range(5):
        raise ValueError("asset fold must be 0..4")
    return _NormalizedAsset(patient_id, split, states, arm, source_arm, seed, fold)


def _alpha_grid(values: Iterable[float]) -> tuple[float, ...]:
    grid = tuple(sorted(set(float(value) for value in values)))
    if not grid or any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("Ridge alpha grid must contain finite positive values")
    return grid


def select_ridge(
    train_matrix: np.ndarray,
    train_target: np.ndarray,
    validation_matrix: np.ndarray,
    validation_target: np.ndarray,
    alphas: Iterable[float] = ALPHAS,
    *,
    standardize_target: bool,
) -> SelectedRidge:
    """Select alpha without accepting an outer-test argument."""

    x_train = np.asarray(train_matrix, dtype=np.float64)
    x_validation = np.asarray(validation_matrix, dtype=np.float64)
    y_train_raw = np.asarray(train_target, dtype=np.float64).reshape(-1)
    y_validation_raw = np.asarray(validation_target, dtype=np.float64).reshape(-1)
    if x_train.ndim != 2 or x_validation.ndim != 2 or x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Ridge train/validation matrices must share a feature dimension")
    if len(x_train) != len(y_train_raw) or len(x_validation) != len(y_validation_raw):
        raise ValueError("Ridge feature/target row counts differ")
    if min(len(y_train_raw), len(y_validation_raw)) < 2:
        raise ValueError("Ridge train/validation targets need at least two rows")
    if not all(
        np.isfinite(value).all()
        for value in (x_train, x_validation, y_train_raw, y_validation_raw)
    ):
        raise FloatingPointError("Ridge train/validation inputs are non-finite")

    x_scaler = StandardScaler().fit(x_train)
    x0 = x_scaler.transform(x_train)
    x1 = x_scaler.transform(x_validation)
    if standardize_target:
        y_scaler: StandardScaler | None = StandardScaler().fit(y_train_raw[:, None])
        y0 = y_scaler.transform(y_train_raw[:, None]).reshape(-1)
        y1 = y_scaler.transform(y_validation_raw[:, None]).reshape(-1)
    else:
        y_scaler = None
        y0, y1 = y_train_raw, y_validation_raw

    candidates: list[tuple[float, float, Ridge]] = []
    for alpha in _alpha_grid(alphas):
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-8,
            max_iter=10_000,
        ).fit(x0, y0)
        mse = float(mean_squared_error(y1, model.predict(x1)))
        if not math.isfinite(mse):
            raise FloatingPointError("validation analysis-space MSE is non-finite")
        candidates.append((alpha, mse, model))
    best_mse = min(item[1] for item in candidates)
    alpha, mse, model = min(
        (item for item in candidates if item[1] <= best_mse + 1e-12),
        key=lambda item: item[0],
    )
    return SelectedRidge(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model=model,
        alpha=float(alpha),
        validation_mse_analysis=float(mse),
        alpha_grid=tuple((float(a), float(score)) for a, score, _ in candidates),
    )


def _target(record: Any, task: str, index: int) -> tuple[float, bool]:
    values = np.asarray(record.values, dtype=np.float64)
    valid = np.asarray(record.measurement_valid, dtype=bool) & np.isfinite(values)
    if values.shape != (4,) or valid.shape != (4,):
        raise ValueError("Stage-B FTVRecord values/measurement_valid must have shape [4]")
    if task == "static":
        return float(values[index]), bool(valid[index])
    if task == "delta":
        value = float(values[index + 1] - values[index])
        return value, bool(valid[index] and valid[index + 1] and math.isfinite(value))
    raise ValueError("task must be static or delta")


def _prepare_split(
    asset: _NormalizedAsset,
    state: np.ndarray,
    records: Mapping[str, Any],
    *,
    split: str,
    task: str,
    index: int,
) -> _PreparedSplit:
    patient_ids: list[str] = []
    matrices: list[np.ndarray] = []
    targets: list[float] = []
    for row in np.flatnonzero(asset.split == split):
        patient_id = str(asset.patient_id[row])
        record = records.get(patient_id)
        if record is None:
            continue
        target, valid = _target(record, task, index)
        if not valid:
            continue
        feature = state[row, index] if task == "static" else state[row, index + 1] - state[row, index]
        feature = np.asarray(feature, dtype=np.float64)
        if feature.ndim != 1 or not np.isfinite(feature).all():
            raise FloatingPointError("response probe feature is invalid")
        patient_ids.append(patient_id)
        matrices.append(feature)
        targets.append(target)
    if not matrices:
        raise ValueError(f"no measurement-valid {split} rows for {task}/{index}")
    return _PreparedSplit(
        tuple(patient_ids),
        np.stack(matrices).astype(np.float64, copy=False),
        np.asarray(targets, dtype=np.float64),
    )


def _run_endpoint(
    asset: _NormalizedAsset,
    state_name: str,
    state: np.ndarray,
    records: Mapping[str, Any],
    *,
    task: str,
    index: int,
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    train = _prepare_split(asset, state, records, split="train", task=task, index=index)
    validation = _prepare_split(asset, state, records, split="val", task=task, index=index)
    target_transform: Any | None = None
    if task == "static":
        outer_train_ids = tuple(asset.patient_id[asset.split == "train"].astype(str))
        target_transform, _ = fit_grounding_transform(
            records,
            outer_train_ids,
            asset.fold,
            apply_ids=(),
        )
        train_analysis, train_valid = target_transform.transform_values(
            train.target, np.ones(train.target.shape, dtype=bool)
        )
        validation_analysis, validation_valid = target_transform.transform_values(
            validation.target, np.ones(validation.target.shape, dtype=bool)
        )
        if not train_valid.all() or not validation_valid.all():
            raise AssertionError("measurement-valid static values became invalid in frozen transform")
        selected = select_ridge(
            train.matrix,
            train_analysis,
            validation.matrix,
            validation_analysis,
            alphas,
            standardize_target=False,
        )
        train_analysis_for_baseline = train_analysis
        analysis_scale = "transformed_outer_train"
    else:
        selected = select_ridge(
            train.matrix,
            train.target,
            validation.matrix,
            validation.target,
            alphas,
            standardize_target=True,
        )
        if selected.y_scaler is None:
            raise AssertionError("literal-delta probe lost its outer-train target scaler")
        train_analysis_for_baseline = selected.y_scaler.transform(train.target[:, None]).reshape(-1)
        analysis_scale = "standardized_outer_train"

    # Test is materialized only after the complete train/validation decision.
    test = _prepare_split(asset, state, records, split="test", task=task, index=index)
    predicted_analysis = np.asarray(
        selected.model.predict(selected.x_scaler.transform(test.matrix)), dtype=np.float64
    ).reshape(-1)
    test_predict_call_count = 1
    if task == "static":
        truth_analysis, truth_valid = target_transform.transform_values(
            test.target, np.ones(test.target.shape, dtype=bool)
        )
        if not truth_valid.all():
            raise AssertionError("measurement-valid test values became invalid in frozen transform")
        predicted_natural = np.asarray(target_transform.inverse(predicted_analysis), dtype=np.float64)
        target_semantics = "static_ftv_log_winsor_median_iqr_inverse_natural"
    else:
        truth_analysis = selected.y_scaler.transform(test.target[:, None]).reshape(-1)
        predicted_natural = selected.y_scaler.inverse_transform(
            predicted_analysis[:, None]
        ).reshape(-1)
        target_semantics = "literal_ftv_end_minus_ftv_start"
    if not np.isfinite(predicted_natural).all():
        raise FloatingPointError("inverse natural FTV predictions are non-finite")

    endpoint = TIMEPOINTS[index] if task == "static" else TRANSITIONS[index]
    natural_baseline = float(np.mean(train.target))
    analysis_baseline = float(np.mean(train_analysis_for_baseline))
    return [
        {
            "patient_id": patient_id,
            "fold": asset.fold,
            "seed_base": asset.seed_base,
            "arm": asset.arm,
            "source_arm": asset.source_arm,
            "state": state_name,
            "task": task,
            "endpoint": endpoint,
            "analysis_scope": "primary_measurement_valid",
            "target_semantics": target_semantics,
            "y_true": float(test.target[row]),
            "y_pred": float(predicted_natural[row]),
            "b0_prediction": natural_baseline,
            "y_true_analysis": float(truth_analysis[row]),
            "y_pred_analysis": float(predicted_analysis[row]),
            "b0_prediction_analysis": analysis_baseline,
            "analysis_scale": analysis_scale,
            "selected_alpha": selected.alpha,
            "validation_mse_analysis_space": selected.validation_mse_analysis,
            "n_train": len(train.patient_id),
            "n_val": len(validation.patient_id),
            "test_used_for_scaler": False,
            "test_used_for_alpha_selection": False,
            "refit_after_alpha_selection": False,
            "test_predict_call_count": test_predict_call_count,
        }
        for row, patient_id in enumerate(test.patient_id)
    ]


def _safe_correlation(function: Any, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return math.nan
    value = float(function(truth, prediction).statistic)
    return value if math.isfinite(value) else math.nan


def _natural_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    truth = group["y_true"].to_numpy(dtype=np.float64)
    prediction = group["y_pred"].to_numpy(dtype=np.float64)
    baseline = group["b0_prediction"].to_numpy(dtype=np.float64)
    if len(truth) < 2 or not all(np.isfinite(value).all() for value in (truth, prediction, baseline)):
        raise FloatingPointError("pooled OOF metric inputs are too small or non-finite")
    target_variance = float(np.var(truth, ddof=0))
    prediction_variance = float(np.var(prediction, ddof=0))
    if target_variance > 0:
        covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
        calibration_slope = covariance / target_variance
        calibration_intercept = float(prediction.mean() - calibration_slope * truth.mean())
    else:
        calibration_slope = calibration_intercept = math.nan
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(truth, baseline)))
    return {
        "n": int(len(group)),
        "spearman": _safe_correlation(spearmanr, truth, prediction),
        "pearson": _safe_correlation(pearsonr, truth, prediction),
        "r2": float(r2_score(truth, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, prediction)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / b0_rmse if b0_rmse > 0 else math.nan,
        "prediction_target_variance_ratio": (
            prediction_variance / target_variance if target_variance > 0 else math.nan
        ),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "calibration_mean_bias": float(np.mean(prediction - truth)),
    }


def _expected_ids(records: Mapping[str, Any], task: str, index: int) -> set[str]:
    return {
        str(patient_id)
        for patient_id, record in records.items()
        if _target(record, task, index)[1]
    }


def run_matched_response_probes(
    assets: Sequence[Any],
    records: Mapping[str, Any],
    *,
    alphas: Sequence[float] = ALPHAS,
    states: Sequence[str] | None = None,
    expected_folds: Sequence[int] | None = (0, 1, 2, 3, 4),
    expected_measurement_valid_patient_count: int | None = 375,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched frozen-state probes and return ``(metrics, private_oof)``.

    Metrics are pooled across outer-test folds on the natural FTV scale.  Each
    endpoint Spearman correlation is computed from its complete OOF vector;
    ``macro`` is the unweighted mean over four static or three delta endpoints.
    """

    if not assets:
        raise ValueError("at least one frozen response-state asset is required")
    if not records:
        raise ValueError("Stage-B FTVRecord mapping is empty")
    normalized = [_normalize_asset(asset) for asset in assets]
    requested = None if states is None else set(states)
    unknown = set() if requested is None else requested.difference(
        {name for asset in normalized for name in asset.states}
    )
    if unknown:
        raise ValueError(f"requested states are unavailable: {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    for asset in normalized:
        for state_name, state in asset.states.items():
            if requested is not None and state_name not in requested:
                continue
            for task, count in (("static", 4), ("delta", 3)):
                for index in range(count):
                    rows.extend(
                        _run_endpoint(
                            asset,
                            state_name,
                            state,
                            records,
                            task=task,
                            index=index,
                            alphas=alphas,
                        )
                    )
    private_oof = pd.DataFrame(rows)
    if private_oof.empty:
        raise ValueError("no requested response states were evaluated")

    identity = ["seed_base", "arm", "state"]
    metric_rows: list[dict[str, Any]] = []
    endpoint_keys = identity + ["task", "endpoint"]
    for key, group in private_oof.groupby(endpoint_keys, sort=True):
        common = dict(zip(endpoint_keys, key, strict=True))
        if group["patient_id"].duplicated().any():
            raise AssertionError(f"pooled OOF repeats a patient for {key}")
        task, endpoint = str(key[-2]), str(key[-1])
        index = TIMEPOINTS.index(endpoint) if task == "static" else TRANSITIONS.index(endpoint)
        expected_ids = _expected_ids(records, task, index)
        if set(group["patient_id"].astype(str)) != expected_ids:
            raise AssertionError(f"pooled OOF patient coverage differs for {key}")
        if (
            expected_measurement_valid_patient_count is not None
            and len(expected_ids) != int(expected_measurement_valid_patient_count)
        ):
            raise AssertionError(
                f"measurement-valid endpoint expected {expected_measurement_valid_patient_count}, "
                f"got {len(expected_ids)}"
            )
        if expected_folds is not None and set(group["fold"].astype(int)) != set(expected_folds):
            raise AssertionError(f"pooled OOF fold coverage differs for {key}")
        if not group["test_predict_call_count"].eq(1).all():
            raise AssertionError("outer-test prediction was not single-use")
        metric_rows.append(
            common
            | {
                "analysis_scope": "primary_measurement_valid",
                "scale": "natural",
                "aggregation": "pooled_outer_test_oof",
                **_natural_metrics(group),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    macro_rows: list[dict[str, Any]] = []
    metric_names = (
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
    for key, group in metrics.groupby(identity + ["task"], sort=True):
        task = str(key[-1])
        expected_endpoints = set(TIMEPOINTS if task == "static" else TRANSITIONS)
        if set(group["endpoint"]) != expected_endpoints:
            raise AssertionError(f"macro endpoint coverage differs for {key}")
        macro_rows.append(
            dict(zip(identity + ["task"], key, strict=True))
            | {
                "endpoint": "macro",
                "analysis_scope": "primary_measurement_valid",
                "scale": "natural",
                "aggregation": "unweighted_mean_of_pooled_endpoint_metrics",
                "n": int(group["n"].sum()),
                **{name: float(group[name].mean()) for name in metric_names},
            }
        )
    metrics = pd.concat((metrics, pd.DataFrame(macro_rows)), ignore_index=True)
    private_oof = private_oof.sort_values(
        ["seed_base", "arm", "state", "task", "endpoint", "patient_id"], kind="stable"
    ).reset_index(drop=True)
    return metrics.sort_values(
        ["seed_base", "arm", "state", "task", "endpoint"], kind="stable"
    ).reset_index(drop=True), private_oof


__all__ = [
    "ALPHAS",
    "TIMEPOINTS",
    "TRANSITIONS",
    "SelectedRidge",
    "run_matched_response_probes",
    "select_ridge",
]
