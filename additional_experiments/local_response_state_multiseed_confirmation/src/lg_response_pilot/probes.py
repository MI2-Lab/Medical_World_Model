"""Sealed outer-fold Ridge probes for static and literal-delta FTV.

The implementation intentionally mirrors the completed Stage-B probe:

* response features are standardized on outer train only;
* alpha is selected on validation analysis-space MSE with a smallest-alpha
  tie break;
* no model is refit after alpha selection;
* each endpoint makes exactly one call to ``predict`` on outer test;
* static FTV uses the frozen outer-train log/winsor/median-IQR transform and
  is inverted before natural-scale evaluation;
* observed change is literal natural ``FTV[t+1] - FTV[t]`` from ``delta-r``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .features import (
    ARMS,
    FOLDS,
    SEED_BASES,
    canonical_sha256,
    file_sha256,
    ordered_patient_sha256,
    require_sha256,
)


TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0_to_T1", "T1_to_T2", "T2_to_T3")
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SCOPES = ("primary_measurement_valid", "observable_only")


@dataclass(frozen=True)
class PreparedSplit:
    patient_ids: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class SelectedRidge:
    x_scaler: StandardScaler
    y_scaler: StandardScaler | None
    model: Ridge
    alpha: float
    validation_mse_analysis: float
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
    *,
    standardize_target: bool,
) -> SelectedRidge:
    """Fit/select using train and validation only; test is absent by signature."""

    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    validation_matrix = np.asarray(validation_matrix, dtype=np.float64)
    train_target = np.asarray(train_target, dtype=np.float64).reshape(-1)
    validation_target = np.asarray(validation_target, dtype=np.float64).reshape(-1)
    if train_matrix.ndim != 2 or validation_matrix.ndim != 2:
        raise ValueError("Ridge feature matrices must be two-dimensional")
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    if len(train_matrix) != len(train_target) or len(validation_matrix) != len(
        validation_target
    ):
        raise ValueError("Ridge feature/target row counts differ")
    if min(len(train_target), len(validation_target)) < 2:
        raise ValueError("Ridge train/validation target sets need at least two rows")
    values = (train_matrix, validation_matrix, train_target, validation_target)
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("Ridge train/validation inputs are non-finite")

    alpha_grid = tuple(sorted(set(float(value) for value in alphas)))
    if not alpha_grid or any(
        not math.isfinite(value) or value <= 0 for value in alpha_grid
    ):
        raise ValueError("Ridge alpha grid must contain finite positive values")
    x_scaler = StandardScaler().fit(train_matrix)
    x_train = x_scaler.transform(train_matrix)
    x_validation = x_scaler.transform(validation_matrix)
    if standardize_target:
        y_scaler: StandardScaler | None = StandardScaler().fit(train_target[:, None])
        y_train = y_scaler.transform(train_target[:, None]).reshape(-1)
        y_validation = y_scaler.transform(validation_target[:, None]).reshape(-1)
    else:
        y_scaler = None
        y_train = train_target
        y_validation = validation_target

    candidates: list[tuple[float, float, Ridge]] = []
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
            raise FloatingPointError("validation analysis-space MSE is non-finite")
        candidates.append((alpha, mse, model))
    best = min(score for _, score, _ in candidates)
    alpha, score, model = min(
        (item for item in candidates if item[1] <= best + 1e-12),
        key=lambda item: item[0],
    )
    return SelectedRidge(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model=model,
        alpha=float(alpha),
        validation_mse_analysis=float(score),
        alpha_grid=tuple((float(a), float(mse)) for a, mse, _ in candidates),
    )


def load_feature_asset(
    path: str | Path,
    *,
    allowed_arms: Sequence[str] = ARMS,
    seed_bases: Sequence[int] = SEED_BASES,
    folds: Sequence[int] = FOLDS,
) -> dict[str, np.ndarray]:
    source = Path(path).resolve()
    if not source.name.endswith(".private.npz"):
        raise ValueError("response feature asset must be a private NPZ")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
    if set(arrays) != required:
        raise ValueError(f"response feature schema drifted: {sorted(arrays)}")
    patient_ids = arrays["patient_id"].astype(str)
    split_labels = arrays["split"].astype(str)
    response = arrays["response_state"]
    if response.dtype != np.float32 or response.shape != (len(patient_ids), 4, 192):
        raise ValueError("response_state must be float32 [N,4,192]")
    if not np.isfinite(response).all():
        raise FloatingPointError("response_state contains non-finite values")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("response feature asset contains duplicate patients")
    if set(split_labels) != {"train", "val", "test"}:
        raise ValueError("response feature asset does not contain all three splits")
    arm = str(np.asarray(arrays["arm"]).item()).upper()
    seed = int(np.asarray(arrays["seed_base"]).item())
    fold = int(np.asarray(arrays["fold"]).item())
    if arm not in set(allowed_arms) or seed not in set(seed_bases) or fold not in set(folds):
        raise ValueError("feature identity is outside the configured confirmation matrix")
    arrays["arm"] = np.asarray(arm)
    return arrays


def validate_feature_split_contract(
    arrays: Mapping[str, np.ndarray], folds: pd.DataFrame
) -> None:
    required = {"patient_id", "fold", "split"}
    if missing := sorted(required.difference(folds.columns)):
        raise ValueError(f"locked fold table misses columns: {missing}")
    fold = int(np.asarray(arrays["fold"]).item())
    current = folds.loc[folds["fold"].eq(fold), ["patient_id", "split"]].copy()
    current["patient_id"] = current["patient_id"].astype(str)
    current["split"] = current["split"].astype(str)
    if current["patient_id"].duplicated().any():
        raise ValueError(f"locked fold {fold} contains duplicate patients")
    expected = dict(zip(current["patient_id"], current["split"], strict=True))
    observed = dict(
        zip(
            arrays["patient_id"].astype(str),
            arrays["split"].astype(str),
            strict=True,
        )
    )
    if observed != expected:
        raise ValueError(
            "feature patients/splits do not exactly match the locked primary cohort"
        )


def validate_feature_metadata(
    feature_path: str | Path,
    arrays: Mapping[str, np.ndarray],
    authorization: Any,
    data_provenance: Mapping[str, Any],
    preregistration_lock_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    source = Path(feature_path).resolve()
    metadata_path = source.with_suffix(".metadata.json")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("feature metadata is missing or invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("feature metadata must be a schema-v1 object")
    expected = {
        "experiment": "local_response_state_multiseed_confirmation",
        "arm": str(np.asarray(arrays["arm"]).item()),
        "seed_base": int(np.asarray(arrays["seed_base"]).item()),
        "fold": int(np.asarray(arrays["fold"]).item()),
        "feature_tensor": "online_preprojector_response_state",
        "feature_dtype": "float32",
        "cohort": "exact_locked_primary_train_validation_test",
        "stage_a_sentinel_sha256": str(authorization.sha256),
        "ftv_head_called": False,
        "test_labels_used": False,
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": require_sha256(
            preregistration_lock_sha256, "preregistration lock"
        ),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"feature metadata differs at {key}")
    if Path(str(payload.get("feature_path", ""))).resolve() != source:
        raise ValueError("feature metadata path differs from the probe input")
    if payload.get("feature_sha256") != file_sha256(source):
        raise ValueError("feature metadata SHA-256 differs from the probe input")
    if payload.get("feature_shape") != list(np.asarray(arrays["response_state"]).shape):
        raise ValueError("feature metadata shape differs from the tensor")
    patient_ids = arrays["patient_id"].astype(str)
    if payload.get("patient_order_sha256") != ordered_patient_sha256(patient_ids):
        raise ValueError("feature metadata patient-order hash differs")
    if payload.get("current_data_contract_provenance_sha256") != canonical_sha256(
        data_provenance
    ):
        raise ValueError("feature metadata differs from the current data contract")
    checkpoint = Path(str(payload.get("checkpoint_path", ""))).resolve()
    if not checkpoint.is_file() or payload.get("checkpoint_sha256") != file_sha256(
        checkpoint
    ):
        raise ValueError("feature metadata checkpoint binding drifted")
    selection = Path(str(payload.get("selection_path", ""))).resolve()
    if not selection.is_file() or payload.get("selection_sha256") != file_sha256(
        selection
    ):
        raise ValueError("feature metadata selection binding drifted")
    return metadata_path, payload


def _prepare_split(
    arrays: Mapping[str, np.ndarray],
    records: Mapping[str, Any],
    *,
    split: str,
    task: str,
    index: int,
    observable_only: bool,
) -> PreparedSplit:
    from c1b_stage_b.targets import literal_delta_targets, static_targets

    patient_ids: list[str] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    ids = arrays["patient_id"].astype(str)
    labels = arrays["split"].astype(str)
    response = np.asarray(arrays["response_state"], dtype=np.float32)
    for row in np.flatnonzero(labels == split):
        patient_id = str(ids[row])
        record = records.get(patient_id)
        if record is None:
            continue
        if task == "static":
            values, valid = static_targets(record, observable_only=observable_only)
            feature = response[row, index]
        elif task == "delta":
            values, valid = literal_delta_targets(
                record.values,
                record.measurement_valid,
                record.observable if observable_only else None,
            )
            feature = response[row, index + 1] - response[row, index]
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
        patient_ids=tuple(patient_ids),
        matrix=np.stack(features).astype(np.float64, copy=False),
        target=np.asarray(targets, dtype=np.float64),
    )


def _safe_correlation(function: Any, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return math.nan
    value = float(function(truth, prediction).statistic)
    return value if math.isfinite(value) else math.nan


def metric_values(
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline: float | np.ndarray,
) -> dict[str, float]:
    """Return the preregistered metrics and descriptive calibration slope.

    The slope intentionally uses the prior study's descriptive formula
    ``Cov(true, pred) / Var(true)`` rather than a regression of true on pred.
    """

    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    baseline_array = np.broadcast_to(
        np.asarray(baseline, dtype=np.float64), truth.shape
    ).copy()
    if len(truth) < 2 or not all(
        np.isfinite(value).all() for value in (truth, prediction, baseline_array)
    ):
        raise FloatingPointError("metric inputs are too small or non-finite")
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(truth, baseline_array)))
    target_variance = float(np.var(truth, ddof=0))
    prediction_variance = float(np.var(prediction, ddof=0))
    if target_variance > 0:
        covariance = float(
            np.mean((truth - np.mean(truth)) * (prediction - np.mean(prediction)))
        )
        calibration_slope = covariance / target_variance
        calibration_intercept = float(
            np.mean(prediction) - calibration_slope * np.mean(truth)
        )
    else:
        calibration_slope = calibration_intercept = math.nan
    return {
        "spearman": _safe_correlation(spearmanr, truth, prediction),
        "pearson": _safe_correlation(pearsonr, truth, prediction),
        "r2": float(r2_score(truth, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, prediction)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (
            (b0_rmse - rmse) / b0_rmse if b0_rmse > 0 else math.nan
        ),
        "prediction_target_variance_ratio": (
            prediction_variance / target_variance
            if target_variance > 0
            else math.nan
        ),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "calibration_mean_bias": float(np.mean(prediction - truth)),
    }


def _run_endpoint(
    arrays: Mapping[str, np.ndarray],
    records: Mapping[str, Any],
    *,
    task: str,
    index: int,
    observable_only: bool,
    alphas: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from c1b_stage_b.targets import fit_static_probe_transform

    prepared = {
        split: _prepare_split(
            arrays,
            records,
            split=split,
            task=task,
            index=index,
            observable_only=observable_only,
        )
        for split in ("train", "val")
    }
    train = prepared["train"]
    validation = prepared["val"]
    fold = int(np.asarray(arrays["fold"]).item())
    target_transform: Any | None = None
    if task == "static":
        outer_train_ids = tuple(
            arrays["patient_id"].astype(str)[
                arrays["split"].astype(str) == "train"
            ]
        )
        target_transform = fit_static_probe_transform(records, outer_train_ids, fold)
        train_analysis, train_valid = target_transform.transform_values(
            train.target, np.ones(train.target.shape, dtype=bool)
        )
        validation_analysis, validation_valid = target_transform.transform_values(
            validation.target, np.ones(validation.target.shape, dtype=bool)
        )
        if not train_valid.all() or not validation_valid.all():
            raise AssertionError("valid static targets became invalid during transform")
        selected = select_ridge(
            train.matrix,
            train_analysis,
            validation.matrix,
            validation_analysis,
            alphas,
            standardize_target=False,
        )
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
            raise AssertionError("literal-delta probe lost its outer-train scaler")
        train_analysis = selected.y_scaler.transform(train.target[:, None]).reshape(-1)
        analysis_scale = "standardized_outer_train"

    # Construct and predict outer test only after every fit/selection decision.
    test = _prepare_split(
        arrays,
        records,
        split="test",
        task=task,
        index=index,
        observable_only=observable_only,
    )
    guard = TestPredictGuard()
    predicted_analysis = guard.predict(
        selected.model, selected.x_scaler.transform(test.matrix)
    )
    if guard.calls != 1:
        raise AssertionError("outer test was not predicted exactly once")
    if task == "static":
        truth_analysis, truth_valid = target_transform.transform_values(
            test.target, np.ones(test.target.shape, dtype=bool)
        )
        if not truth_valid.all():
            raise AssertionError("valid test static targets became invalid")
        predicted_natural = np.asarray(
            target_transform.inverse(predicted_analysis), dtype=np.float64
        )
        transform_payload = target_transform.to_dict()
    else:
        predicted_natural = selected.y_scaler.inverse_transform(
            predicted_analysis[:, None]
        ).reshape(-1)
        truth_analysis = selected.y_scaler.transform(test.target[:, None]).reshape(-1)
        transform_payload = {
            "value_transform": "literal_natural_delta",
            "standardization": "outer_train_standard_scaler",
            "train_rows": int(selected.y_scaler.n_samples_seen_),
        }
    if not np.isfinite(predicted_natural).all():
        raise FloatingPointError("inverse natural predictions are non-finite")

    natural_baseline = float(np.mean(train.target))
    analysis_baseline = float(np.mean(train_analysis))
    natural_metrics = metric_values(test.target, predicted_natural, natural_baseline)
    analysis_metrics = metric_values(
        truth_analysis, predicted_analysis, analysis_baseline
    )
    endpoint = TIMEPOINTS[index] if task == "static" else TRANSITIONS[index]
    scope = "observable_only" if observable_only else "primary_measurement_valid"
    common = {
        "task": task,
        "endpoint": endpoint,
        "analysis_scope": scope,
        "target_semantics": (
            "static_ftv_log_winsor_median_iqr_inverse_natural"
            if task == "static"
            else "literal_ftv_end_minus_ftv_start"
        ),
        "selected_alpha": selected.alpha,
        "n_train": len(train.patient_ids),
        "n_val": len(validation.patient_ids),
        "n_test": len(test.patient_ids),
    }
    metric_rows = [
        {**common, "scale": "natural", **natural_metrics},
        {**common, "scale": analysis_scale, **analysis_metrics},
    ]
    prediction_rows = [
        {
            "patient_id": patient_id,
            **common,
            "split": "test",
            "y_true": float(test.target[row]),
            "y_pred": float(predicted_natural[row]),
            "b0_prediction": natural_baseline,
            "y_true_analysis": float(truth_analysis[row]),
            "y_pred_analysis": float(predicted_analysis[row]),
            "b0_prediction_analysis": analysis_baseline,
            "analysis_scale": analysis_scale,
            "test_predict_call_count": 1,
        }
        for row, patient_id in enumerate(test.patient_ids)
    ]
    selection = {
        **common,
        "validation_mse_analysis_space": selected.validation_mse_analysis,
        "alpha_validation_mse_json": json.dumps(
            dict(selected.alpha_grid), sort_keys=True
        ),
        "x_scaler_train_rows": int(selected.x_scaler.n_samples_seen_),
        "x_scaler_mean_json": json.dumps(selected.x_scaler.mean_.tolist()),
        "x_scaler_scale_json": json.dumps(selected.x_scaler.scale_.tolist()),
        "target_transform_json": json.dumps(transform_payload, sort_keys=True),
        "y_scaler_mean": (
            math.nan
            if selected.y_scaler is None
            else float(selected.y_scaler.mean_[0])
        ),
        "y_scaler_scale": (
            math.nan
            if selected.y_scaler is None
            else float(selected.y_scaler.scale_[0])
        ),
        "test_used_for_scaler": False,
        "test_used_for_alpha_selection": False,
        "refit_after_alpha_selection": False,
        "test_predict_call_count": 1,
    }
    return selection, prediction_rows, metric_rows


def _atomic_csv(path: Path, frame: pd.DataFrame, *, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        # Every artifact in predictions/ is private, including deidentified
        # helper tables and provenance metadata.
        os.chmod(temporary, 0o600)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_ftv_probes(
    *,
    feature_path: str | Path,
    records: Mapping[str, Any],
    folds: pd.DataFrame,
    authorization: Any,
    data_provenance: Mapping[str, Any],
    output_dir: str | Path,
    preregistration_lock_sha256: str,
    alphas: Sequence[float] = ALPHAS,
    allowed_arms: Sequence[str] = ARMS,
) -> dict[str, Any]:
    arrays = load_feature_asset(feature_path, allowed_arms=allowed_arms)
    validate_feature_split_contract(arrays, folds)
    feature_metadata_path, feature_metadata = validate_feature_metadata(
        feature_path,
        arrays,
        authorization,
        data_provenance,
        preregistration_lock_sha256,
    )
    output = Path(output_dir).resolve()
    paths = {
        "selection": output / "ridge_selection.csv",
        "prediction": output / "ridge_predictions.private.csv",
        "metrics": output / "probe_metrics.csv",
        "metadata": output / "probe_metadata.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite a confirmation probe asset")

    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for observable_only in (False, True):
        for task, count in (("static", 4), ("delta", 3)):
            for index in range(count):
                selection, predictions, metrics = _run_endpoint(
                    arrays,
                    records,
                    task=task,
                    index=index,
                    observable_only=observable_only,
                    alphas=alphas,
                )
                selection_rows.append(selection)
                prediction_rows.extend(predictions)
                metric_rows.extend(metrics)

    identity = {
        "arm": str(np.asarray(arrays["arm"]).item()),
        "seed_base": int(np.asarray(arrays["seed_base"]).item()),
        "fold": int(np.asarray(arrays["fold"]).item()),
    }
    for rows in (selection_rows, prediction_rows, metric_rows):
        for row in rows:
            row.update(identity)
    _atomic_csv(paths["selection"], pd.DataFrame(selection_rows))
    _atomic_csv(paths["prediction"], pd.DataFrame(prediction_rows), private=True)
    _atomic_csv(paths["metrics"], pd.DataFrame(metric_rows))

    target_adapter = Path(__file__).resolve().parents[3] / (
        "c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/targets.py"
    )
    feature_checkpoint_binding = {
        "feature_path": str(Path(feature_path).resolve()),
        "feature_sha256": feature_metadata["feature_sha256"],
        "feature_metadata_path": str(feature_metadata_path.resolve()),
        "feature_metadata_sha256": file_sha256(feature_metadata_path),
        "checkpoint_path": feature_metadata["checkpoint_path"],
        "checkpoint_sha256": feature_metadata["checkpoint_sha256"],
        "selection_path": feature_metadata["selection_path"],
        "selection_sha256": feature_metadata["selection_sha256"],
    }
    metadata = {
        "schema_version": 1,
        "experiment": "local_response_state_multiseed_confirmation",
        **identity,
        "feature_asset_name": Path(feature_path).name,
        "feature_sha256": file_sha256(feature_path),
        "feature_metadata_name": feature_metadata_path.name,
        "feature_metadata_sha256": file_sha256(feature_metadata_path),
        "feature_checkpoint_binding": feature_checkpoint_binding,
        "feature_checkpoint_binding_sha256": canonical_sha256(
            feature_checkpoint_binding
        ),
        "probe_implementation_sha256": file_sha256(Path(__file__)),
        "target_adapter_sha256": file_sha256(target_adapter),
        "stage_a_sentinel_sha256": str(authorization.sha256),
        "alpha_grid": [float(value) for value in alphas],
        "feature_scaler": "StandardScaler fit on outer train only",
        "static_target_transform": (
            "outer-train observable log(FTV+epsilon), winsor 1/99, median/IQR; "
            "inverse to natural FTV"
        ),
        "delta_target": "literal natural FTV_end_minus_FTV_start from delta-r",
        "alpha_selection": (
            "minimum validation analysis-space MSE; smallest-alpha tie break"
        ),
        "refit_after_alpha_selection": False,
        "test_used_for_scaler_or_selection": False,
        "outer_test_predict_calls_per_endpoint": 1,
        "analysis_scopes": list(SCOPES),
        "patient_level_outputs_private": True,
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": require_sha256(
            preregistration_lock_sha256, "preregistration lock"
        ),
        "output_sha256": {
            path.name: file_sha256(path)
            for path in (
                paths["selection"],
                paths["prediction"],
                paths["metrics"],
            )
        },
    }
    _atomic_json(paths["metadata"], metadata)
    return metadata


__all__ = [
    "ALPHAS",
    "ARMS",
    "FOLDS",
    "SCOPES",
    "SEED_BASES",
    "TIMEPOINTS",
    "TRANSITIONS",
    "PreparedSplit",
    "SelectedRidge",
    "TestPredictGuard",
    "load_feature_asset",
    "metric_values",
    "run_ftv_probes",
    "select_ridge",
    "validate_feature_metadata",
    "validate_feature_split_contract",
]
