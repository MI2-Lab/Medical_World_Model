"""Frozen static-FTV and literal-natural-delta Ridge probes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .contracts import ARMS, FOLDS, SEED_BASES, TIMEPOINTS, TRANSITIONS, file_sha256
from .data import FTVRecord
from .gate import StageAAuthorization
from .targets import literal_delta_targets, static_targets


ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class PreparedSplit:
    patient_ids: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class SelectedRidge:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    model: Ridge
    alpha: float
    validation_mse_standardized: float
    alpha_grid: tuple[tuple[float, float], ...]


@dataclass
class TestPredictGuard:
    calls: int = 0

    def predict(self, model: Ridge, matrix: np.ndarray) -> np.ndarray:
        if self.calls:
            raise RuntimeError("outer-test prediction is single-use")
        self.calls += 1
        return np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)


def select_ridge(
    train_matrix: np.ndarray,
    train_target: np.ndarray,
    validation_matrix: np.ndarray,
    validation_target: np.ndarray,
    alphas: Iterable[float] = ALPHAS,
) -> SelectedRidge:
    """Select alpha from train/validation only; test is absent by signature."""

    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    validation_matrix = np.asarray(validation_matrix, dtype=np.float64)
    train_target = np.asarray(train_target, dtype=np.float64).reshape(-1)
    validation_target = np.asarray(validation_target, dtype=np.float64).reshape(-1)
    if train_matrix.ndim != 2 or validation_matrix.ndim != 2:
        raise ValueError("Ridge matrices must be two-dimensional")
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise ValueError("train/validation feature dimensions differ")
    if len(train_matrix) != len(train_target) or len(validation_matrix) != len(validation_target):
        raise ValueError("Ridge X/y row count mismatch")
    if min(len(train_target), len(validation_target)) < 2:
        raise ValueError("Ridge train/validation split has fewer than two targets")
    if not all(
        np.isfinite(values).all()
        for values in (train_matrix, validation_matrix, train_target, validation_target)
    ):
        raise FloatingPointError("Ridge train/validation inputs are non-finite")
    alpha_grid = tuple(sorted(set(float(value) for value in alphas)))
    if not alpha_grid or any(value <= 0 or not math.isfinite(value) for value in alpha_grid):
        raise ValueError("Ridge alpha grid must contain finite positive values")
    x_scaler = StandardScaler().fit(train_matrix)
    y_scaler = StandardScaler().fit(train_target[:, None])
    x_train = x_scaler.transform(train_matrix)
    x_validation = x_scaler.transform(validation_matrix)
    y_train = y_scaler.transform(train_target[:, None]).reshape(-1)
    y_validation = y_scaler.transform(validation_target[:, None]).reshape(-1)
    results: list[tuple[float, float, Ridge]] = []
    for alpha in alpha_grid:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-8,
            max_iter=10_000,
        ).fit(x_train, y_train)
        mse = float(mean_squared_error(y_validation, model.predict(x_validation)))
        if not math.isfinite(mse):
            raise FloatingPointError("validation standardized MSE is non-finite")
        results.append((alpha, mse, model))
    best = min(score for _, score, _ in results)
    alpha, score, model = min(
        (item for item in results if item[1] <= best + 1e-12), key=lambda item: item[0]
    )
    return SelectedRidge(
        x_scaler,
        y_scaler,
        model,
        float(alpha),
        float(score),
        tuple((float(a), float(mse)) for a, mse, _ in results),
    )


def _load_feature_asset(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).resolve()
    if not source.name.endswith(".private.npz"):
        raise ValueError("Stage B feature asset must be a private NPZ")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
    if set(arrays) != required:
        raise ValueError(f"feature NPZ schema drifted: {sorted(arrays)}")
    patient_ids = arrays["patient_id"].astype(str)
    splits = arrays["split"].astype(str)
    response = arrays["response_state"]
    if response.shape != (len(patient_ids), 4, 192) or response.dtype != np.float32:
        raise ValueError("feature response_state must be float32 [N,4,192]")
    if len(set(patient_ids)) != len(patient_ids) or set(splits) != {"train", "val", "test"}:
        raise ValueError("feature patient/split coverage is invalid")
    if not np.isfinite(response).all():
        raise FloatingPointError("feature response_state is non-finite")
    arm = str(np.asarray(arrays["arm"]).item())
    seed_base = int(np.asarray(arrays["seed_base"]).item())
    fold = int(np.asarray(arrays["fold"]).item())
    if arm not in ARMS or seed_base not in SEED_BASES or fold not in FOLDS:
        raise ValueError("feature arm/seed/fold identity is outside the frozen matrix")
    return arrays


def _validate_feature_split_contract(
    arrays: Mapping[str, np.ndarray], folds: pd.DataFrame
) -> None:
    """Bind a feature asset to the exact locked outer-fold patient split."""

    fold = int(np.asarray(arrays["fold"]).item())
    current = folds.loc[folds["fold"].eq(fold), ["patient_id", "split"]].copy()
    current["patient_id"] = current["patient_id"].astype(str)
    current["split"] = current["split"].astype(str)
    if current["patient_id"].duplicated().any():
        raise ValueError(f"locked fold {fold} contains duplicate patient identities")
    expected = dict(zip(current["patient_id"], current["split"], strict=True))
    patient_ids = arrays["patient_id"].astype(str)
    split_labels = arrays["split"].astype(str)
    if len(patient_ids) != len(expected) or set(patient_ids) != set(expected):
        raise ValueError("feature patients do not exactly cover the locked primary cohort")
    observed = dict(zip(patient_ids, split_labels, strict=True))
    if observed != expected:
        raise ValueError("feature train/val/test labels disagree with the locked fold manifest")


def _prepare_split(
    arrays: Mapping[str, np.ndarray],
    records: Mapping[str, FTVRecord],
    *,
    split: str,
    task: str,
    index: int,
    observable_only: bool,
) -> PreparedSplit:
    patient_ids: list[str] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    ids = arrays["patient_id"].astype(str)
    splits = arrays["split"].astype(str)
    response = np.asarray(arrays["response_state"], dtype=np.float32)
    for row_index in np.flatnonzero(splits == split):
        patient_id = str(ids[row_index])
        record = records.get(patient_id)
        if record is None:
            continue
        if task == "static":
            values, valid = static_targets(record, observable_only=observable_only)
            feature = response[row_index, index]
        elif task == "delta":
            values, valid = literal_delta_targets(
                record.values,
                record.measurement_valid,
                record.observable if observable_only else None,
            )
            feature = response[row_index, index + 1] - response[row_index, index]
        else:
            raise ValueError("task must be static or delta")
        if not bool(valid[index]):
            continue
        if feature.shape != (192,) or not np.isfinite(feature).all():
            raise FloatingPointError("probe feature is invalid")
        patient_ids.append(patient_id)
        features.append(feature)
        targets.append(float(values[index]))
    if not features:
        raise ValueError(f"no valid {split} rows for {task}/{index}")
    return PreparedSplit(
        tuple(patient_ids),
        np.stack(features).astype(np.float64, copy=False),
        np.asarray(targets, dtype=np.float64),
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    b0_rmse = float(math.sqrt(mean_squared_error(y_true, np.full(y_true.shape, baseline))))
    target_variance = float(np.var(y_true, ddof=0))
    correlation = spearmanr(y_true, y_pred).statistic
    return {
        "spearman": float(correlation) if math.isfinite(float(correlation)) else math.nan,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / b0_rmse if b0_rmse > 0 else math.nan,
        "prediction_target_variance_ratio": (
            float(np.var(y_pred, ddof=0)) / target_variance if target_variance > 0 else math.nan
        ),
    }


def _run_cell(
    arrays: Mapping[str, np.ndarray],
    records: Mapping[str, FTVRecord],
    *,
    task: str,
    index: int,
    observable_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train = _prepare_split(
        arrays, records, split="train", task=task, index=index, observable_only=observable_only
    )
    validation = _prepare_split(
        arrays, records, split="val", task=task, index=index, observable_only=observable_only
    )
    selected = select_ridge(train.matrix, train.target, validation.matrix, validation.target)
    # The test matrix/target is constructed only after alpha is locked.
    test = _prepare_split(
        arrays, records, split="test", task=task, index=index, observable_only=observable_only
    )
    test_matrix = selected.x_scaler.transform(test.matrix)
    guard = TestPredictGuard()
    predicted_standardized = guard.predict(selected.model, test_matrix)
    if guard.calls != 1:
        raise AssertionError("outer-test Ridge predict call count is not one")
    predicted_natural = selected.y_scaler.inverse_transform(
        predicted_standardized[:, None]
    ).reshape(-1)
    truth_standardized = selected.y_scaler.transform(test.target[:, None]).reshape(-1)
    natural_baseline = float(np.mean(train.target))
    standardized_baseline = float(
        selected.y_scaler.transform(np.asarray([[natural_baseline]])).item()
    )
    natural_metrics = _metrics(test.target, predicted_natural, natural_baseline)
    standardized_metrics = _metrics(
        truth_standardized, predicted_standardized, standardized_baseline
    )
    endpoint = TIMEPOINTS[index] if task == "static" else TRANSITIONS[index]
    common = {
        "task": task,
        "endpoint": endpoint,
        "analysis_scope": "observable_only" if observable_only else "primary_measurement_valid",
        "target_semantics": "natural_ftv" if task == "static" else "literal_ftv_end_minus_ftv_start",
        "selected_alpha": selected.alpha,
        "n_train": len(train.patient_ids),
        "n_val": len(validation.patient_ids),
        "n_test": len(test.patient_ids),
    }
    metric_rows = [
        {**common, "scale": "natural", **natural_metrics},
        {**common, "scale": "standardized_train_only", **standardized_metrics},
    ]
    prediction_rows = [
        {
            "patient_id": patient_id,
            **common,
            "split": "test",
            "y_true": float(test.target[row]),
            "y_pred": float(predicted_natural[row]),
            "y_true_standardized": float(truth_standardized[row]),
            "y_pred_standardized": float(predicted_standardized[row]),
            "b0_prediction": natural_baseline,
            "test_predict_call_count": guard.calls,
        }
        for row, patient_id in enumerate(test.patient_ids)
    ]
    selection = {
        **common,
        "validation_mse_standardized": selected.validation_mse_standardized,
        "alpha_validation_mse_json": json.dumps(dict(selected.alpha_grid), sort_keys=True),
        "x_scaler_train_rows": int(selected.x_scaler.n_samples_seen_),
        "y_scaler_train_rows": int(selected.y_scaler.n_samples_seen_),
        "x_scaler_mean_json": json.dumps(selected.x_scaler.mean_.tolist()),
        "x_scaler_scale_json": json.dumps(selected.x_scaler.scale_.tolist()),
        "y_scaler_mean": float(selected.y_scaler.mean_[0]),
        "y_scaler_scale": float(selected.y_scaler.scale_[0]),
        "test_used_for_scaler": False,
        "test_used_for_alpha_selection": False,
        "test_predict_call_count": guard.calls,
    }
    return selection, prediction_rows, metric_rows


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_ftv_probes(
    *,
    feature_path: str | Path,
    records: Mapping[str, FTVRecord],
    folds: pd.DataFrame,
    authorization: StageAAuthorization,
    output_dir: str | Path,
) -> dict[str, Any]:
    arrays = _load_feature_asset(feature_path)
    _validate_feature_split_contract(arrays, folds)
    output = Path(output_dir).resolve()
    paths = {
        "selection": output / "ridge_selection.csv",
        "prediction": output / "ridge_predictions.private.csv",
        "metrics": output / "probe_metrics.csv",
        "metadata": output / "probe_metadata.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite a Stage B probe asset")
    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for observable_only in (False, True):
        for task, count in (("static", 4), ("delta", 3)):
            for index in range(count):
                selection, predictions, metrics = _run_cell(
                    arrays,
                    records,
                    task=task,
                    index=index,
                    observable_only=observable_only,
                )
                selection_rows.append(selection)
                prediction_rows.extend(predictions)
                metric_rows.extend(metrics)
    arm = str(np.asarray(arrays["arm"]).item())
    seed_base = int(np.asarray(arrays["seed_base"]).item())
    fold = int(np.asarray(arrays["fold"]).item())
    identifying = {"arm": arm, "seed_base": seed_base, "fold": fold}
    for rows in (selection_rows, prediction_rows, metric_rows):
        for row in rows:
            row.update(identifying)
    metric_frame = pd.DataFrame(metric_rows)
    grouping = ["analysis_scope", "task", "scale"]
    macros: list[dict[str, Any]] = []
    metric_names = (
        "spearman", "r2", "rmse", "mae", "b0_rmse", "rmse_gain_over_b0",
        "prediction_target_variance_ratio",
    )
    for keys, group in metric_frame.groupby(grouping, sort=False):
        macros.append(
            {
                "analysis_scope": keys[0],
                "task": keys[1],
                "scale": keys[2],
                "endpoint": "macro",
                "target_semantics": (
                    "natural_ftv" if keys[1] == "static" else "literal_ftv_end_minus_ftv_start"
                ),
                "selected_alpha": math.nan,
                "n_train": int(group["n_train"].sum()),
                "n_val": int(group["n_val"].sum()),
                "n_test": int(group["n_test"].sum()),
                **{name: float(group[name].mean()) for name in metric_names},
                **identifying,
            }
        )
    metric_frame = pd.concat([metric_frame, pd.DataFrame(macros)], ignore_index=True)
    _atomic_csv(paths["selection"], pd.DataFrame(selection_rows))
    _atomic_csv(paths["prediction"], pd.DataFrame(prediction_rows))
    _atomic_csv(paths["metrics"], metric_frame)
    metadata = {
        "schema_version": 1,
        **identifying,
        "feature_path": str(Path(feature_path).resolve()),
        "feature_sha256": file_sha256(feature_path),
        "stage_a_sentinel_sha256": authorization.sha256,
        "alpha_grid": list(ALPHAS),
        "feature_scaler": "StandardScaler fit on outer train only",
        "target_scaler": "StandardScaler fit on outer train natural values only",
        "alpha_selection": "minimum validation standardized MSE; smallest-alpha tie break",
        "outer_test_predict_calls_per_cell": 1,
        "static_target": "natural FTV",
        "delta_target": "literal FTV[t+1] - FTV[t] (no log)",
        "primary_scope": "all measurement-valid visits",
        "sensitivity_scope": "measurement-valid and grounding-observable visits",
        "test_used_for_scaler_or_selection": False,
    }
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{paths['metadata'].name}.", suffix=".tmp", dir=paths["metadata"].parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temporary).replace(paths["metadata"])
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata


__all__ = ["ALPHAS", "run_ftv_probes", "select_ridge"]
