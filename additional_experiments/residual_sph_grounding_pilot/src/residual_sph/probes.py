"""Fold-safe frozen-state Ridge probes for FTV, SPH, and residual SPH."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from .contracts import arm_spec
from .targets import FoldTargetBundle, VISITS


ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
TRANSITIONS = ("T0_to_T1", "T1_to_T2", "T2_to_T3")


@dataclass(frozen=True)
class FeatureAsset:
    patient_ids: tuple[str, ...]
    splits: np.ndarray
    response_state: np.ndarray
    runtime_arm: str
    analysis_arm: str
    seed_base: int
    fold: int


@dataclass(frozen=True)
class SelectedRidge:
    x_scaler: StandardScaler
    y_scaler: StandardScaler | None
    model: Ridge
    alpha: float
    validation_mse: float
    alpha_grid: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SelectedStateReconstructionRidge:
    """Fold-fitted scalar-target to multivariate-state diagnostic."""

    target_scaler: StandardScaler
    state_scaler: StandardScaler
    model: Ridge
    alpha: float
    validation_mse: float
    alpha_grid: tuple[tuple[float, float], ...]


def load_feature_asset(
    path: str | Path,
    *,
    analysis_arm: str,
    seed_base: int,
    fold: int,
) -> FeatureAsset:
    source = Path(path).resolve()
    if not source.name.endswith(".private.npz"):
        raise ValueError("feature asset must be a private NPZ")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
    if set(arrays) != required:
        raise ValueError(f"feature schema drifted: {sorted(arrays)}")
    ids = tuple(arrays["patient_id"].astype(str).tolist())
    splits = arrays["split"].astype(str)
    response = arrays["response_state"]
    runtime_arm = str(np.asarray(arrays["arm"]).item()).upper()
    arm = str(analysis_arm).upper()
    if arm == "S0":
        if runtime_arm != "LOCAL3":
            raise ValueError("S0 feature must come from confirmed LOCAL3")
    elif runtime_arm != arm or arm_spec(arm).name != arm:
        raise ValueError("feature arm differs from requested analysis arm")
    if int(np.asarray(arrays["seed_base"]).item()) != int(seed_base):
        raise ValueError("feature seed mismatch")
    if int(np.asarray(arrays["fold"]).item()) != int(fold):
        raise ValueError("feature fold mismatch")
    if len(ids) != len(set(ids)) or set(splits) != {"train", "val", "test"}:
        raise ValueError("feature patient/split contract drifted")
    if response.dtype != np.float32 or response.shape != (len(ids), 4, 192):
        raise ValueError("response_state must be float32 [N,4,192]")
    if not np.isfinite(response).all():
        raise FloatingPointError("response_state contains non-finite values")
    splits.setflags(write=False)
    response.setflags(write=False)
    return FeatureAsset(ids, splits, response, runtime_arm, arm, int(seed_base), int(fold))


def validate_target_alignment(asset: FeatureAsset, bundle: FoldTargetBundle) -> None:
    lookup = {patient_id: split for patient_id, split in zip(asset.patient_ids, asset.splits, strict=True)}
    if set(bundle.patient_ids).difference(lookup):
        raise ValueError("feature asset is missing target-cohort patients")
    for patient_id, split in zip(bundle.patient_ids, bundle.splits, strict=True):
        if lookup[patient_id] != split:
            raise ValueError("feature and target split labels differ")


def select_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    standardize_target: bool,
    alphas: Sequence[float] = ALPHAS,
) -> SelectedRidge:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
    if x_train.ndim != 2 or x_val.ndim != 2 or x_train.shape[1] != x_val.shape[1]:
        raise ValueError("Ridge feature shapes differ")
    if len(x_train) != len(y_train) or len(x_val) != len(y_val):
        raise ValueError("Ridge row counts differ")
    if min(len(y_train), len(y_val)) < 2:
        raise ValueError("Ridge train/validation need at least two rows")
    if not all(np.isfinite(value).all() for value in (x_train, x_val, y_train, y_val)):
        raise FloatingPointError("Ridge input contains non-finite values")
    alpha_grid = tuple(sorted(set(float(value) for value in alphas)))
    if alpha_grid != ALPHAS:
        raise ValueError("formal Ridge alpha grid drifted")
    x_scaler = StandardScaler().fit(x_train)
    train_x = x_scaler.transform(x_train)
    val_x = x_scaler.transform(x_val)
    if standardize_target:
        y_scaler: StandardScaler | None = StandardScaler().fit(y_train[:, None])
        train_y = y_scaler.transform(y_train[:, None]).reshape(-1)
        val_y = y_scaler.transform(y_val[:, None]).reshape(-1)
    else:
        y_scaler = None
        train_y, val_y = y_train, y_val
    candidates: list[tuple[float, float, Ridge]] = []
    for alpha in alpha_grid:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-8,
            max_iter=10_000,
        ).fit(train_x, train_y)
        mse = float(mean_squared_error(val_y, model.predict(val_x)))
        if not math.isfinite(mse):
            raise FloatingPointError("validation MSE is non-finite")
        candidates.append((alpha, mse, model))
    best = min(value for _, value, _ in candidates)
    alpha, score, model = min(
        (item for item in candidates if item[1] <= best + 1e-12),
        key=lambda item: item[0],
    )
    return SelectedRidge(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model=model,
        alpha=float(alpha),
        validation_mse=float(score),
        alpha_grid=tuple((float(a), float(mse)) for a, mse, _ in candidates),
    )


def select_state_reconstruction_ridge(
    target_train: np.ndarray,
    state_train: np.ndarray,
    target_val: np.ndarray,
    state_val: np.ndarray,
    *,
    alphas: Sequence[float] = ALPHAS,
) -> SelectedStateReconstructionRidge:
    """Select a fold-safe Ridge from one phenotype coordinate to state.

    Both the scalar input and each state dimension are standardized from the
    outer-training split only.  Validation chooses alpha using mean squared
    error over all standardized state coordinates.  Test data is not accepted
    by this function, making selection structurally test-blind.
    """

    train_target = np.asarray(target_train, dtype=np.float64).reshape(-1, 1)
    val_target = np.asarray(target_val, dtype=np.float64).reshape(-1, 1)
    train_state = np.asarray(state_train, dtype=np.float64)
    val_state = np.asarray(state_val, dtype=np.float64)
    if train_state.ndim != 2 or val_state.ndim != 2:
        raise ValueError("state reconstruction targets must be two-dimensional")
    if train_state.shape[1] != val_state.shape[1] or train_state.shape[1] < 1:
        raise ValueError("state reconstruction dimensions differ")
    if len(train_target) != len(train_state) or len(val_target) != len(val_state):
        raise ValueError("state reconstruction row counts differ")
    if min(len(train_target), len(val_target)) < 2:
        raise ValueError("state reconstruction needs at least two train/validation rows")
    if not all(
        np.isfinite(value).all()
        for value in (train_target, val_target, train_state, val_state)
    ):
        raise FloatingPointError("state reconstruction input contains non-finite values")
    if float(np.var(train_target, ddof=0)) <= 0.0:
        raise ValueError("state reconstruction phenotype target is constant")

    alpha_grid = tuple(sorted(set(float(value) for value in alphas)))
    if alpha_grid != ALPHAS:
        raise ValueError("formal Ridge alpha grid drifted")
    target_scaler = StandardScaler().fit(train_target)
    state_scaler = StandardScaler().fit(train_state)
    scaled_train_target = target_scaler.transform(train_target)
    scaled_val_target = target_scaler.transform(val_target)
    scaled_train_state = state_scaler.transform(train_state)
    scaled_val_state = state_scaler.transform(val_state)
    candidates: list[tuple[float, float, Ridge]] = []
    for alpha in alpha_grid:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-8,
            max_iter=10_000,
        ).fit(scaled_train_target, scaled_train_state)
        prediction = model.predict(scaled_val_target)
        mse = float(np.mean((scaled_val_state - prediction) ** 2))
        if not math.isfinite(mse):
            raise FloatingPointError("state reconstruction validation MSE is non-finite")
        candidates.append((alpha, mse, model))
    best = min(value for _, value, _ in candidates)
    alpha, score, model = min(
        (item for item in candidates if item[1] <= best + 1e-12),
        key=lambda item: item[0],
    )
    return SelectedStateReconstructionRidge(
        target_scaler=target_scaler,
        state_scaler=state_scaler,
        model=model,
        alpha=float(alpha),
        validation_mse=float(score),
        alpha_grid=tuple((float(a), float(mse)) for a, mse, _ in candidates),
    )


def state_reconstruction_metrics(
    true_standardized_state: np.ndarray,
    predicted_standardized_state: np.ndarray,
) -> dict[str, float | int]:
    """Return interpretable aggregate metrics for multivariate state recovery.

    ``state_variance_weighted_r2`` is ``1 - total SSE / total SST`` after each
    state coordinate is centered on its held-out fold mean.  It is therefore a
    variance-weighted average of coordinate-wise R2 values.  State coordinates
    are already in an outer-train-fitted standardized space; fold-specific
    values must be aggregated by held-out row count, never pooled across folds.
    """

    truth = np.asarray(true_standardized_state, dtype=np.float64)
    prediction = np.asarray(predicted_standardized_state, dtype=np.float64)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("state reconstruction truth/prediction shapes differ")
    if min(truth.shape) < 2:
        raise ValueError("state reconstruction metric needs at least 2 rows and dimensions")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("state reconstruction metric contains non-finite values")
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    coordinate_sst = np.sum(centered**2, axis=0)
    squared_error = (truth - prediction) ** 2
    total_sst = float(np.sum(coordinate_sst))
    if total_sst <= 0.0:
        raise ValueError("held-out state has zero total variance")
    variance_weighted_r2 = 1.0 - float(np.sum(squared_error)) / total_sst
    valid_coordinates = coordinate_sst > 0.0
    coordinate_r2 = 1.0 - np.sum(squared_error[:, valid_coordinates], axis=0) / (
        coordinate_sst[valid_coordinates]
    )
    metrics: dict[str, float | int] = {
        "n_test": int(len(truth)),
        "state_dimension": int(truth.shape[1]),
        "nonconstant_test_state_dimensions": int(valid_coordinates.sum()),
        "state_variance_weighted_r2": float(variance_weighted_r2),
        "state_uniform_average_r2": float(np.mean(coordinate_r2)),
        "state_standardized_rmse": float(np.sqrt(np.mean(squared_error))),
        "state_standardized_mae": float(np.mean(np.abs(truth - prediction))),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError("state reconstruction metric is non-finite")
    return metrics


def _indices(asset: FeatureAsset, bundle: FoldTargetBundle, split: str) -> tuple[np.ndarray, np.ndarray]:
    asset_index = {patient_id: index for index, patient_id in enumerate(asset.patient_ids)}
    target_rows = np.flatnonzero(bundle.splits == split)
    feature_rows = np.asarray([asset_index[bundle.patient_ids[row]] for row in target_rows], dtype=np.int64)
    return feature_rows, target_rows


def _fit_predict(
    x: Mapping[str, np.ndarray],
    y: Mapping[str, np.ndarray],
    *,
    standardize_target: bool,
) -> tuple[SelectedRidge, np.ndarray, np.ndarray]:
    selected = select_ridge(
        x["train"], y["train"], x["val"], y["val"], standardize_target=standardize_target
    )
    prediction = np.asarray(
        selected.model.predict(selected.x_scaler.transform(x["test"])),
        dtype=np.float64,
    ).reshape(-1)
    if selected.y_scaler is not None:
        prediction_analysis = prediction.copy()
        prediction = selected.y_scaler.inverse_transform(prediction[:, None]).reshape(-1)
        truth_analysis = selected.y_scaler.transform(y["test"][:, None]).reshape(-1)
    else:
        prediction_analysis = prediction
        truth_analysis = y["test"]
    if not np.isfinite(prediction).all():
        raise FloatingPointError("test prediction is non-finite")
    return selected, prediction_analysis, truth_analysis


def run_fold_probes(
    asset: FeatureAsset,
    bundle: FoldTargetBundle,
    *,
    ftv_records: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return aggregate-safe selections and private test prediction rows."""

    validate_target_alignment(asset, bundle)
    split_rows = {
        split: _indices(asset, bundle, split) for split in ("train", "val", "test")
    }
    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    def execute(
        *,
        task: str,
        endpoint: str,
        feature_for: Callable[[np.ndarray], np.ndarray],
        analysis_target: np.ndarray,
        natural_target: np.ndarray,
        inverse_prediction: Callable[[np.ndarray, np.ndarray], np.ndarray],
        standardize_target: bool,
    ) -> None:
        x: dict[str, np.ndarray] = {}
        y: dict[str, np.ndarray] = {}
        target_rows_by_split: dict[str, np.ndarray] = {}
        for split, (feature_rows, target_rows) in split_rows.items():
            x[split] = feature_for(feature_rows).astype(np.float64, copy=False)
            y[split] = np.asarray(analysis_target[target_rows], dtype=np.float64)
            target_rows_by_split[split] = target_rows
        selected, predicted_analysis, truth_analysis = _fit_predict(
            x, y, standardize_target=standardize_target
        )
        if selected.y_scaler is not None:
            predicted_for_inverse = selected.y_scaler.inverse_transform(
                predicted_analysis[:, None]
            ).reshape(-1)
        else:
            predicted_for_inverse = predicted_analysis
        test_rows = target_rows_by_split["test"]
        predicted_natural = inverse_prediction(predicted_for_inverse, test_rows)
        truth_natural = np.asarray(natural_target[test_rows], dtype=np.float64)
        common = {
            "record_type": "state_to_target_probe",
            "arm": asset.analysis_arm,
            "seed_base": asset.seed_base,
            "fold": asset.fold,
            "task": task,
            "endpoint": endpoint,
            "selected_alpha": selected.alpha,
            "validation_mse_analysis": selected.validation_mse,
            "n_train": len(y["train"]),
            "n_val": len(y["val"]),
            "n_test": len(y["test"]),
            "test_predict_call_count": 1,
            "refit_after_selection": False,
        }
        selection_rows.append(
            {
                **common,
                "alpha_validation_mse": dict(selected.alpha_grid),
                "target_standardized_inside_probe": standardize_target,
                "test_used_for_selection": False,
            }
        )
        for offset, target_row in enumerate(test_rows):
            prediction_rows.append(
                {
                    **common,
                    "patient_id": bundle.patient_ids[int(target_row)],
                    "split": "test",
                    "y_true_analysis": float(truth_analysis[offset]),
                    "y_pred_analysis": float(predicted_analysis[offset]),
                    "y_true_natural": float(truth_natural[offset]),
                    "y_pred_natural": float(predicted_natural[offset]),
                    "ftv_control": float(bundle.natural_ftv[target_row, VISITS.index(endpoint)])
                    if endpoint in VISITS
                    else math.nan,
                }
            )

    def execute_state_reconstruction(
        *,
        task: str,
        endpoint: str,
        analysis_target: np.ndarray,
        target_coordinate: str,
    ) -> None:
        target_by_split: dict[str, np.ndarray] = {}
        state_by_split: dict[str, np.ndarray] = {}
        for split, (feature_rows, target_rows) in split_rows.items():
            target_by_split[split] = np.asarray(
                analysis_target[target_rows], dtype=np.float64
            )
            state_by_split[split] = np.asarray(
                asset.response_state[feature_rows, VISITS.index(endpoint)],
                dtype=np.float64,
            )
        selected = select_state_reconstruction_ridge(
            target_by_split["train"],
            state_by_split["train"],
            target_by_split["val"],
            state_by_split["val"],
        )
        test_target = selected.target_scaler.transform(
            target_by_split["test"][:, None]
        )
        true_standardized_state = selected.state_scaler.transform(
            state_by_split["test"]
        )
        predicted_standardized_state = np.asarray(
            selected.model.predict(test_target), dtype=np.float64
        )
        metrics = state_reconstruction_metrics(
            true_standardized_state, predicted_standardized_state
        )
        selection_rows.append(
            {
                "record_type": "target_to_state_diagnostic",
                "arm": asset.analysis_arm,
                "seed_base": asset.seed_base,
                "fold": asset.fold,
                "task": task,
                "endpoint": endpoint,
                "target_coordinate": target_coordinate,
                "state_coordinate": (
                    "response_state_192D_dimensionwise_StandardScaler_outer_train"
                ),
                "selected_alpha": selected.alpha,
                "validation_mse_analysis": selected.validation_mse,
                "alpha_validation_mse": dict(selected.alpha_grid),
                "n_train": len(target_by_split["train"]),
                "n_val": len(target_by_split["val"]),
                "n_test": len(target_by_split["test"]),
                "test_predict_call_count": 1,
                "refit_after_selection": False,
                "test_used_for_selection": False,
                "target_and_state_scalers_fit_scope": "outer_train_only",
                "fold_aggregation_requirement": (
                    "outer_test_n_weighted_fold_metrics_no_cross_fold_state_pooling"
                ),
                **metrics,
            }
        )

    # Static FTV retains the distinct pooled-visit robust LOCAL3 transform.
    from c1b_stage_b.targets import fit_static_probe_transform

    # Preserve the confirmed LOCAL3 transform fit population exactly.  The
    # broader 808-person fold contributes the full outer-train identity list;
    # only measurement-valid/observable FTV visits enter the pooled values.
    outer_train_ids = tuple(
        patient_id
        for patient_id, split in zip(asset.patient_ids, asset.splits, strict=True)
        if split == "train"
    )
    ftv_transform = fit_static_probe_transform(ftv_records, outer_train_ids, bundle.fold)
    transformed_ftv, valid_ftv = ftv_transform.transform_values(
        bundle.natural_ftv, np.ones(bundle.natural_ftv.shape, dtype=bool)
    )
    if not valid_ftv.all():
        raise AssertionError("complete-case FTV became invalid under static transform")
    for visit_index, visit in enumerate(VISITS):
        execute(
            task="static_ftv",
            endpoint=visit,
            feature_for=lambda rows, index=visit_index: asset.response_state[rows, index],
            analysis_target=transformed_ftv[:, visit_index],
            natural_target=bundle.natural_ftv[:, visit_index],
            inverse_prediction=lambda values, rows: np.asarray(ftv_transform.inverse(values), dtype=np.float64),
            standardize_target=False,
        )
        execute_state_reconstruction(
            task="ftv_to_state",
            endpoint=visit,
            analysis_target=transformed_ftv[:, visit_index],
            target_coordinate="fold_train_fitted_STATIC_FTV_analysis_coordinate",
        )
        execute_state_reconstruction(
            task="sph_to_state",
            endpoint=visit,
            analysis_target=bundle.s1_targets[:, visit_index],
            target_coordinate="fold_train_fitted_SPH_z",
        )
        execute_state_reconstruction(
            task="sph_res_to_state",
            endpoint=visit,
            analysis_target=bundle.s2_targets[:, visit_index],
            target_coordinate="fold_train_fitted_residual_SPH_z",
        )
    for interval_index, interval in enumerate(TRANSITIONS):
        natural_delta = bundle.natural_ftv[:, interval_index + 1] - bundle.natural_ftv[:, interval_index]
        execute(
            task="observed_delta_ftv",
            endpoint=interval,
            feature_for=lambda rows, index=interval_index: (
                asset.response_state[rows, index + 1] - asset.response_state[rows, index]
            ),
            analysis_target=natural_delta,
            natural_target=natural_delta,
            inverse_prediction=lambda values, rows: values,
            standardize_target=True,
        )
    for visit_index, visit in enumerate(VISITS):
        fitted = bundle.residualizers[visit_index]
        execute(
            task="raw_sph",
            endpoint=visit,
            feature_for=lambda rows, index=visit_index: asset.response_state[rows, index],
            analysis_target=bundle.s1_targets[:, visit_index],
            natural_target=bundle.natural_sphericity[:, visit_index],
            inverse_prediction=lambda values, rows, transform=fitted: transform.inverse_s1_target(values),
            standardize_target=False,
        )
        execute(
            task="sph_res",
            endpoint=visit,
            feature_for=lambda rows, index=visit_index: asset.response_state[rows, index],
            analysis_target=bundle.s2_targets[:, visit_index],
            natural_target=bundle.natural_sphericity[:, visit_index],
            inverse_prediction=lambda values, rows, transform=fitted, index=visit_index: transform.reconstruct_sphericity(
                values, bundle.natural_ftv[rows, index]
            ),
            standardize_target=False,
        )
    return selection_rows, prediction_rows


__all__ = [
    "ALPHAS",
    "FeatureAsset",
    "SelectedRidge",
    "SelectedStateReconstructionRidge",
    "load_feature_asset",
    "run_fold_probes",
    "select_ridge",
    "select_state_reconstruction_ridge",
    "state_reconstruction_metrics",
    "validate_target_alignment",
]
