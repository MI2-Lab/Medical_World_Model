"""Dimension-agnostic, outer-fold-isolated probes for frozen pooled states.

This module deliberately owns no Ridge-selection or FTV-target implementation.
Those primitives are imported from, and SHA-bound to, the completed Stage-B
experiment.  The adapter here only generalizes the feature dimension, enforces
per-visit representation validity, writes private OOF predictions, and pools
natural-scale OOF metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .contracts import (
    FOLDS,
    TIMEPOINTS,
    TRANSITIONS,
    UPSTREAM_ROOT,
    UPSTREAM_SOURCE_SHA256,
    canonical_sha256,
    file_sha256,
)


_FORMAL_PROBE_RELATIVE = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
    "src/c1b_stage_b/probes.py"
)
_FORMAL_TARGET_RELATIVE = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
    "src/c1b_stage_b/targets.py"
)


def _load_formal_probe_api() -> tuple[Any, Any]:
    """Import the exact immutable Stage-B probe/target implementations."""

    probe_path = (UPSTREAM_ROOT / "src" / "c1b_stage_b" / "probes.py").resolve()
    target_path = probe_path.with_name("targets.py")
    expected = {
        probe_path: UPSTREAM_SOURCE_SHA256[_FORMAL_PROBE_RELATIVE],
        target_path: UPSTREAM_SOURCE_SHA256[_FORMAL_TARGET_RELATIVE],
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ImportError(f"immutable Stage-B source hash drifted: {path}")
    source_root = str((UPSTREAM_ROOT / "src").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    probe_module = importlib.import_module("c1b_stage_b.probes")
    target_module = importlib.import_module("c1b_stage_b.targets")
    if Path(inspect.getfile(probe_module.select_ridge)).resolve() != probe_path:
        raise ImportError("select_ridge did not resolve to the immutable Stage-B source")
    if Path(inspect.getfile(target_module.fit_static_probe_transform)).resolve() != target_path:
        raise ImportError(
            "fit_static_probe_transform did not resolve to the immutable Stage-B source"
        )
    return probe_module, target_module


_FORMAL_PROBES, _FORMAL_TARGETS = _load_formal_probe_api()

# Public aliases intentionally point at the old function objects.  Tests and
# downstream metadata can therefore prove that no locally copied Ridge/FTV
# implementation was substituted.
ALPHAS = _FORMAL_PROBES.ALPHAS
select_ridge = _FORMAL_PROBES.select_ridge
TestPredictGuard = _FORMAL_PROBES.TestPredictGuard
fit_static_probe_transform = _FORMAL_TARGETS.fit_static_probe_transform
static_targets = _FORMAL_TARGETS.static_targets
literal_delta_targets = _FORMAL_TARGETS.literal_delta_targets


METRIC_NAMES = (
    "spearman",
    "pearson",
    "r2",
    "rmse",
    "mae",
    "b0_rmse",
    "rmse_gain_over_b0",
    "prediction_target_variance_ratio",
)
CALIBRATION_NAMES = (
    "calibration_slope",
    "calibration_intercept",
    "calibration_mean_bias",
)
ANALYSIS_SCOPES = ("primary_measurement_valid", "observable_only")


def ordered_patient_sha256(patient_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in patient_ids).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@dataclass(frozen=True)
class FrozenStateAsset:
    """One frozen checkpoint/pooling cell with arbitrary feature dimension."""

    patient_id: np.ndarray
    split: np.ndarray
    state: np.ndarray
    state_valid: np.ndarray
    arm: str
    seed_base: int
    fold: int
    pooling: str
    source_path: Path | None = None
    metadata_path: Path | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patient_id = np.asarray(self.patient_id).astype(str)
        split = np.asarray(self.split).astype(str)
        state = np.asarray(self.state)
        state_valid_raw = np.asarray(self.state_valid)
        if patient_id.ndim != 1 or not len(patient_id):
            raise ValueError("patient_id must be a nonempty one-dimensional array")
        if split.shape != patient_id.shape:
            raise ValueError("split must align one-to-one with patient_id")
        if len(set(patient_id)) != len(patient_id) or np.any(patient_id == ""):
            raise ValueError("patient_id values must be nonempty and unique")
        if set(split) != {"train", "val", "test"}:
            raise ValueError("split must contain nonempty train, val, and test partitions")
        if state.ndim != 3 or state.shape[:2] != (len(patient_id), 4) or state.shape[2] <= 0:
            raise ValueError("state must have shape [N,4,D] with arbitrary positive D")
        if state.dtype != np.dtype(np.float32):
            raise ValueError("frozen pooled state must be float32")
        if state_valid_raw.dtype != np.dtype(bool) or state_valid_raw.shape != state.shape[:2]:
            raise ValueError("state_valid must be boolean [N,4]")
        if not np.isfinite(state[state_valid_raw]).all():
            raise FloatingPointError("state is non-finite at a state_valid location")
        arm = str(self.arm).strip().upper()
        pooling = str(self.pooling).strip().upper()
        if not arm or not pooling:
            raise ValueError("arm and pooling identities must be nonempty")
        fold = int(self.fold)
        if fold not in FOLDS:
            raise ValueError(f"fold must be one of {FOLDS}")
        source_path = None if self.source_path is None else Path(self.source_path).resolve()
        metadata_path = None if self.metadata_path is None else Path(self.metadata_path).resolve()
        object.__setattr__(self, "patient_id", patient_id.copy())
        object.__setattr__(self, "split", split.copy())
        object.__setattr__(self, "state", np.ascontiguousarray(state))
        object.__setattr__(self, "state_valid", np.ascontiguousarray(state_valid_raw))
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "seed_base", int(self.seed_base))
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "pooling", pooling)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "metadata_path", metadata_path)
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))

    @property
    def feature_dim(self) -> int:
        return int(self.state.shape[2])

    @property
    def patient_order_sha256(self) -> str:
        return ordered_patient_sha256(self.patient_id)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed_base": self.seed_base,
            "fold": self.fold,
            "pooling": self.pooling,
            "feature_dim": self.feature_dim,
        }


def _resolve_metadata_feature_path(value: Any, metadata_path: Path) -> set[Path]:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return {candidate.resolve()}
    return {
        (metadata_path.parent / candidate).resolve(),
        (Path.cwd() / candidate).resolve(),
    }


def load_frozen_state_asset(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    require_metadata: bool = True,
) -> FrozenStateAsset:
    """Load the frozen per-pooling NPZ and validate its same-stem metadata."""

    source = Path(path).resolve()
    if not source.name.endswith(".private.npz"):
        raise ValueError("identifier-bearing pooled state must end in .private.npz")
    expected_keys = {
        "patient_id",
        "split",
        "state",
        "state_valid",
        "arm",
        "seed_base",
        "fold",
        "pooling",
    }
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(
                "pooled-state NPZ schema drifted: "
                f"expected={sorted(expected_keys)}, observed={sorted(archive.files)}"
            )
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata_source = (
        source.with_suffix(".metadata.json")
        if metadata_path is None
        else Path(metadata_path).resolve()
    )
    metadata: dict[str, Any] = {}
    if metadata_source.is_file():
        payload = json.loads(metadata_source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("pooled-state metadata must be a JSON object")
        metadata = payload
    elif require_metadata:
        raise FileNotFoundError(f"pooled-state metadata is missing: {metadata_source}")
    asset = FrozenStateAsset(
        patient_id=arrays["patient_id"],
        split=arrays["split"],
        state=arrays["state"],
        state_valid=arrays["state_valid"],
        arm=str(np.asarray(arrays["arm"]).item()),
        seed_base=int(np.asarray(arrays["seed_base"]).item()),
        fold=int(np.asarray(arrays["fold"]).item()),
        pooling=str(np.asarray(arrays["pooling"]).item()),
        source_path=source,
        metadata_path=metadata_source if metadata else None,
        source_metadata=metadata,
    )
    if metadata:
        for key, expected in asset.identity.items():
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"pooled-state metadata identity drifted at {key}")
        feature_digest = _require_sha256(metadata.get("feature_sha256"), "feature_sha256")
        if feature_digest != file_sha256(source):
            raise ValueError("pooled-state metadata feature SHA-256 drifted")
        if "feature_path" in metadata and source not in _resolve_metadata_feature_path(
            metadata["feature_path"], metadata_source
        ):
            raise ValueError("pooled-state metadata feature path drifted")
        if metadata.get("state_shape", list(asset.state.shape)) != list(asset.state.shape):
            raise ValueError("pooled-state metadata state_shape drifted")
        if metadata.get("state_valid_shape", list(asset.state_valid.shape)) != list(
            asset.state_valid.shape
        ):
            raise ValueError("pooled-state metadata state_valid_shape drifted")
        if metadata.get("patient_order_sha256", asset.patient_order_sha256) != (
            asset.patient_order_sha256
        ):
            raise ValueError("pooled-state metadata patient-order hash drifted")
        if "checkpoint_sha256" in metadata:
            _require_sha256(metadata["checkpoint_sha256"], "checkpoint_sha256")
    return asset


@dataclass(frozen=True)
class PreparedSplit:
    patient_ids: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class ProbeResult:
    identity: Mapping[str, Any]
    selection: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    source_path: Path | None = None
    source_metadata_path: Path | None = None


def _prepared(
    patient_ids: Sequence[str], features: Sequence[np.ndarray], targets: Sequence[float]
) -> PreparedSplit:
    if not features:
        raise ValueError("probe split has no valid rows")
    matrix = np.stack(features).astype(np.float64, copy=False)
    target = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(patient_ids):
        raise ValueError("prepared feature matrix is invalid")
    if target.shape != (len(patient_ids),):
        raise ValueError("prepared target vector is invalid")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise FloatingPointError("prepared probe rows are non-finite")
    return PreparedSplit(tuple(patient_ids), matrix, target)


def _prepare_ftv_split(
    asset: FrozenStateAsset,
    records: Mapping[str, Any],
    *,
    split: str,
    task: str,
    index: int,
    observable_only: bool,
) -> PreparedSplit:
    patient_ids: list[str] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    for row_index in np.flatnonzero(asset.split == split):
        patient_id = str(asset.patient_id[row_index])
        record = records.get(patient_id)
        if record is None:
            continue
        if task == "static":
            values, target_valid = static_targets(record, observable_only=observable_only)
            feature_valid = bool(asset.state_valid[row_index, index])
            feature = asset.state[row_index, index]
        elif task == "delta":
            values, target_valid = literal_delta_targets(
                record.values,
                record.measurement_valid,
                record.observable if observable_only else None,
            )
            feature_valid = bool(
                asset.state_valid[row_index, index]
                and asset.state_valid[row_index, index + 1]
            )
            feature = asset.state[row_index, index + 1] - asset.state[row_index, index]
        else:
            raise ValueError("FTV task must be static or delta")
        if not feature_valid or not bool(np.asarray(target_valid, dtype=bool)[index]):
            continue
        if feature.shape != (asset.feature_dim,) or not np.isfinite(feature).all():
            raise FloatingPointError("selected frozen state is invalid")
        patient_ids.append(patient_id)
        features.append(feature)
        targets.append(float(np.asarray(values, dtype=np.float64)[index]))
    return _prepared(patient_ids, features, targets)


def _normalize_continuous_targets(
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    normalized: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for raw_name, patient_values in targets.items():
        name = str(raw_name).strip()
        if not name or not isinstance(patient_values, Mapping):
            raise ValueError("continuous targets require nonempty names and patient mappings")
        rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for raw_patient, item in patient_values.items():
            patient_id = str(raw_patient)
            if isinstance(item, tuple) and len(item) == 2:
                values = np.asarray(item[0], dtype=np.float64)
                valid_raw = np.asarray(item[1])
                if valid_raw.dtype != np.dtype(bool):
                    raise ValueError("continuous target validity must be boolean")
                valid = valid_raw.copy()
            else:
                values = np.asarray(item, dtype=np.float64)
                valid = np.isfinite(values)
            if values.shape != (4,) or valid.shape != (4,):
                raise ValueError("continuous target values/validity must have shape [4]")
            if not np.isfinite(values[valid]).all():
                raise FloatingPointError("continuous target is non-finite at a valid visit")
            rows[patient_id] = (values, valid)
        if not rows:
            raise ValueError(f"continuous target {name!r} has no patient rows")
        normalized[name] = rows
    if not normalized:
        raise ValueError("at least one continuous target is required")
    return normalized


def _prepare_continuous_split(
    asset: FrozenStateAsset,
    target_rows: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    split: str,
    index: int,
) -> PreparedSplit:
    patient_ids: list[str] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    for row_index in np.flatnonzero(asset.split == split):
        patient_id = str(asset.patient_id[row_index])
        row = target_rows.get(patient_id)
        if row is None:
            continue
        values, target_valid = row
        if not bool(asset.state_valid[row_index, index]) or not bool(target_valid[index]):
            continue
        feature = asset.state[row_index, index]
        if not np.isfinite(feature).all():
            raise FloatingPointError("selected frozen state is invalid")
        patient_ids.append(patient_id)
        features.append(feature)
        targets.append(float(values[index]))
    return _prepared(patient_ids, features, targets)


def _run_prepared_cell(
    *,
    prepare: Callable[[str], PreparedSplit],
    common: Mapping[str, Any],
    target_transform: Any | None,
    standardized_value_transform: str = "identity_natural",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit/select without a test argument, then construct and predict test once."""

    train = prepare("train")
    validation = prepare("val")
    if target_transform is not None:
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
            ALPHAS,
            standardize_target=False,
        )
        analysis_scale = "transformed_outer_train"
        target_transform_payload = target_transform.to_dict()
    else:
        selected = select_ridge(
            train.matrix,
            train.target,
            validation.matrix,
            validation.target,
            ALPHAS,
            standardize_target=True,
        )
        if selected.y_scaler is None:
            raise AssertionError("standardized continuous target lost its train scaler")
        train_analysis = selected.y_scaler.transform(train.target[:, None]).reshape(-1)
        analysis_scale = "standardized_outer_train"
        target_transform_payload = {
            "value_transform": str(standardized_value_transform),
            "standardization": "outer_train_standard_scaler",
            "train_rows": int(selected.y_scaler.n_samples_seen_),
        }

    # Test construction is deliberately below all fit/selection calls.
    test = prepare("test")
    test_matrix = selected.x_scaler.transform(test.matrix)
    guard = TestPredictGuard()
    predicted_analysis = guard.predict(selected.model, test_matrix)
    if guard.calls != 1:
        raise AssertionError("outer-test Ridge prediction count is not exactly one")
    if target_transform is not None:
        truth_analysis, truth_valid = target_transform.transform_values(
            test.target, np.ones(test.target.shape, dtype=bool)
        )
        if not truth_valid.all():
            raise AssertionError("valid static test targets became invalid during transform")
        predicted_natural = target_transform.inverse(predicted_analysis)
    else:
        assert selected.y_scaler is not None
        truth_analysis = selected.y_scaler.transform(test.target[:, None]).reshape(-1)
        predicted_natural = selected.y_scaler.inverse_transform(
            predicted_analysis[:, None]
        ).reshape(-1)

    natural_baseline = float(np.mean(train.target))
    analysis_baseline = float(np.mean(train_analysis))
    cell_common = {
        **dict(common),
        "selected_alpha": selected.alpha,
        "n_train": len(train.patient_ids),
        "n_val": len(validation.patient_ids),
        "n_test": len(test.patient_ids),
    }
    metric_rows = [
        {
            **cell_common,
            "scale": "natural",
            "aggregation": "outer_fold",
            **_FORMAL_PROBES._metrics(test.target, predicted_natural, natural_baseline),
        },
        {
            **cell_common,
            "scale": analysis_scale,
            "aggregation": "outer_fold",
            **_FORMAL_PROBES._metrics(
                truth_analysis, predicted_analysis, analysis_baseline
            ),
        },
    ]
    prediction_rows = [
        {
            "patient_id": patient_id,
            **cell_common,
            "split": "test",
            "y_true": float(test.target[row]),
            "y_pred": float(predicted_natural[row]),
            "y_true_analysis": float(truth_analysis[row]),
            "y_pred_analysis": float(predicted_analysis[row]),
            "b0_prediction": natural_baseline,
            "b0_prediction_analysis": analysis_baseline,
            "analysis_scale": analysis_scale,
            "test_predict_call_count": guard.calls,
        }
        for row, patient_id in enumerate(test.patient_ids)
    ]
    selection = {
        **cell_common,
        "validation_mse_analysis_space": selected.validation_mse_standardized,
        "alpha_validation_mse_json": json.dumps(
            dict(selected.alpha_grid), sort_keys=True
        ),
        "x_scaler_train_rows": int(selected.x_scaler.n_samples_seen_),
        "target_transform_json": json.dumps(
            target_transform_payload, sort_keys=True
        ),
        "x_scaler_mean_json": json.dumps(selected.x_scaler.mean_.tolist()),
        "x_scaler_scale_json": json.dumps(selected.x_scaler.scale_.tolist()),
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
        "test_predict_call_count": guard.calls,
    }
    return selection, prediction_rows, metric_rows


def _with_fold_macros(metrics: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        "arm",
        "seed_base",
        "fold",
        "pooling",
        "feature_dim",
        "task",
        "target_name",
        "analysis_scope",
        "target_semantics",
        "scale",
    ]
    expected_by_task = {
        "static": set(TIMEPOINTS),
        "delta": set(TRANSITIONS),
        "nuisance": set(TIMEPOINTS),
        "continuous": set(TIMEPOINTS),
    }
    macros: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(group_keys, sort=False, dropna=False):
        task = str(keys[5])
        expected = expected_by_task.get(task)
        if expected is not None and set(group["endpoint"]) != expected:
            raise ValueError(f"endpoint coverage drifted for fold macro: {keys}")
        macros.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "endpoint": "macro",
                "selected_alpha": math.nan,
                "n_train": int(group["n_train"].sum()),
                "n_val": int(group["n_val"].sum()),
                "n_test": int(group["n_test"].sum()),
                "aggregation": "mean_of_outer_fold_endpoint_metrics",
                **{name: float(group[name].mean()) for name in METRIC_NAMES},
            }
        )
    return pd.concat([metrics, pd.DataFrame(macros)], ignore_index=True, sort=False)


def _result(
    asset: FrozenStateAsset,
    selections: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> ProbeResult:
    selection_frame = pd.DataFrame(selections)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = _with_fold_macros(pd.DataFrame(metrics))
    if selection_frame.empty or prediction_frame.empty or metric_frame.empty:
        raise ValueError("probe result may not be empty")
    return ProbeResult(
        identity=asset.identity,
        selection=selection_frame,
        predictions=prediction_frame,
        metrics=metric_frame,
        source_path=asset.source_path,
        source_metadata_path=asset.metadata_path,
    )


def run_ftv_probe_cell(
    asset: FrozenStateAsset,
    records: Mapping[str, Any],
    *,
    analysis_scopes: Sequence[str] = ANALYSIS_SCOPES,
) -> ProbeResult:
    """Run the exact Stage-B static/literal-delta contract for one pooled cell."""

    scopes = tuple(str(value) for value in analysis_scopes)
    if not scopes or not set(scopes).issubset(set(ANALYSIS_SCOPES)):
        raise ValueError(f"analysis_scopes must be a nonempty subset of {ANALYSIS_SCOPES}")
    outer_train_ids = tuple(asset.patient_id[asset.split == "train"].astype(str))
    target_transform = fit_static_probe_transform(records, outer_train_ids, asset.fold)
    selections: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for scope in scopes:
        observable_only = scope == "observable_only"
        for task, endpoints in (("static", TIMEPOINTS), ("delta", TRANSITIONS)):
            for index, endpoint in enumerate(endpoints):
                common = {
                    **asset.identity,
                    "task": task,
                    "target_name": "FTV",
                    "endpoint": endpoint,
                    "analysis_scope": scope,
                    "target_semantics": (
                        "static_ftv_log_winsor_median_iqr_inverse_natural"
                        if task == "static"
                        else "literal_ftv_end_minus_ftv_start"
                    ),
                }
                prepare = lambda split, task=task, index=index, observable_only=observable_only: (  # noqa: E731
                    _prepare_ftv_split(
                        asset,
                        records,
                        split=split,
                        task=task,
                        index=index,
                        observable_only=observable_only,
                    )
                )
                selection, cell_predictions, cell_metrics = _run_prepared_cell(
                    prepare=prepare,
                    common=common,
                    target_transform=target_transform if task == "static" else None,
                    standardized_value_transform=(
                        "literal_natural_delta" if task == "delta" else "identity_natural"
                    ),
                )
                selections.append(selection)
                predictions.extend(cell_predictions)
                metrics.extend(cell_metrics)
    return _result(asset, selections, predictions, metrics)


def run_continuous_probe_cell(
    asset: FrozenStateAsset,
    targets: Mapping[str, Mapping[str, Any]],
    *,
    task: str = "nuisance",
    analysis_scope: str = "target_valid",
    target_semantics: Mapping[str, str] | None = None,
) -> ProbeResult:
    """Run the same isolated Ridge protocol for visit-level continuous targets."""

    task = str(task).strip()
    analysis_scope = str(analysis_scope).strip()
    if not task or not analysis_scope:
        raise ValueError("continuous probe task and analysis_scope must be nonempty")
    normalized = _normalize_continuous_targets(targets)
    semantic_map = {} if target_semantics is None else dict(target_semantics)
    if not set(semantic_map).issubset(normalized):
        raise ValueError("target_semantics contains an unknown continuous target")
    selections: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for target_name, target_rows in normalized.items():
        semantics = semantic_map.get(target_name, f"continuous_natural::{target_name}")
        for index, endpoint in enumerate(TIMEPOINTS):
            common = {
                **asset.identity,
                "task": task,
                "target_name": target_name,
                "endpoint": endpoint,
                "analysis_scope": analysis_scope,
                "target_semantics": semantics,
            }
            prepare = lambda split, index=index, target_rows=target_rows: (  # noqa: E731
                _prepare_continuous_split(
                    asset, target_rows, split=split, index=index
                )
            )
            selection, cell_predictions, cell_metrics = _run_prepared_cell(
                prepare=prepare,
                common=common,
                target_transform=None,
            )
            selections.append(selection)
            predictions.extend(cell_predictions)
            metrics.extend(cell_metrics)
    return _result(asset, selections, predictions, metrics)


def combine_probe_results(*results: ProbeResult) -> ProbeResult:
    if not results:
        raise ValueError("at least one ProbeResult is required")
    identity = dict(results[0].identity)
    if any(dict(result.identity) != identity for result in results[1:]):
        raise ValueError("cannot combine probe results from different feature cells")
    source_paths = {result.source_path for result in results}
    metadata_paths = {result.source_metadata_path for result in results}
    if len(source_paths) != 1 or len(metadata_paths) != 1:
        raise ValueError("combined probe results disagree on source provenance")
    selection = pd.concat([result.selection for result in results], ignore_index=True)
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    metrics = pd.concat([result.metrics for result in results], ignore_index=True)
    selection_keys = [
        "arm", "seed_base", "fold", "pooling", "task", "target_name",
        "endpoint", "analysis_scope",
    ]
    prediction_keys = [*selection_keys, "patient_id"]
    if selection.duplicated(selection_keys).any():
        raise ValueError("combined probe selections contain duplicate cells")
    if predictions.duplicated(prediction_keys).any():
        raise ValueError("combined probe predictions contain duplicate OOF rows")
    return ProbeResult(
        identity=identity,
        selection=selection,
        predictions=predictions,
        metrics=metrics,
        source_path=results[0].source_path,
        source_metadata_path=results[0].source_metadata_path,
    )


def _safe_correlation(function: Any, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return math.nan
    value = float(function(truth, prediction).statistic)
    return value if math.isfinite(value) else math.nan


def _pooled_natural_values(rows: pd.DataFrame) -> dict[str, float]:
    truth = rows["y_true"].to_numpy(dtype=np.float64)
    prediction = rows["y_pred"].to_numpy(dtype=np.float64)
    baseline = rows["b0_prediction"].to_numpy(dtype=np.float64)
    if not len(truth) or not all(
        np.isfinite(values).all() for values in (truth, prediction, baseline)
    ):
        raise FloatingPointError("pooled natural OOF rows are empty or non-finite")
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(truth, baseline)))
    target_variance = float(np.var(truth, ddof=0))
    if target_variance > 0:
        calibration_slope = float(
            np.mean(
                (truth - np.mean(truth))
                * (prediction - np.mean(prediction))
            )
            / target_variance
        )
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
            float(np.var(prediction, ddof=0)) / target_variance
            if target_variance > 0
            else math.nan
        ),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "calibration_mean_bias": float(np.mean(prediction - truth)),
    }


def pooled_oof_natural_metrics(
    predictions: pd.DataFrame,
    *,
    expected_folds: Iterable[int] = FOLDS,
    expected_endpoints: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Pool fixed outer-test rows, then recompute nonlinear natural metrics."""

    required = {
        "patient_id",
        "split",
        "arm",
        "seed_base",
        "fold",
        "pooling",
        "feature_dim",
        "task",
        "target_name",
        "endpoint",
        "analysis_scope",
        "target_semantics",
        "y_true",
        "y_pred",
        "b0_prediction",
        "test_predict_call_count",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"OOF predictions miss required columns: {missing}")
    if predictions.empty or not predictions["split"].eq("test").all():
        raise ValueError("pooled OOF input must contain only nonempty outer-test rows")
    if not predictions["test_predict_call_count"].eq(1).all():
        raise ValueError("one or more OOF rows did not come from a single test prediction")
    folds = {int(value) for value in expected_folds}
    if not folds:
        raise ValueError("expected_folds may not be empty")
    endpoint_map = {
        "static": TIMEPOINTS,
        "delta": TRANSITIONS,
        "nuisance": TIMEPOINTS,
        "continuous": TIMEPOINTS,
    }
    if expected_endpoints is not None:
        endpoint_map.update(
            {str(key): tuple(str(value) for value in values) for key, values in expected_endpoints.items()}
        )
    group_keys = [
        "seed_base",
        "arm",
        "pooling",
        "feature_dim",
        "task",
        "target_name",
        "endpoint",
        "analysis_scope",
        "target_semantics",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_keys, sort=False, dropna=False):
        if set(group["fold"].astype(int)) != folds:
            raise ValueError(f"OOF group does not cover the exact expected folds: {keys}")
        if group["patient_id"].astype(str).duplicated().any():
            raise ValueError(f"patient appears in multiple outer-test folds: {keys}")
        rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "scale": "natural",
                "aggregation": "pooled_outer_test_folds",
                "n_test": len(group),
                **_pooled_natural_values(group),
            }
        )
    endpoint_frame = pd.DataFrame(rows)
    macro_keys = [key for key in group_keys if key != "endpoint"]
    macros: list[dict[str, Any]] = []
    for keys, group in endpoint_frame.groupby(macro_keys, sort=False, dropna=False):
        task = str(group["task"].iloc[0])
        if task in endpoint_map and set(group["endpoint"]) != set(endpoint_map[task]):
            raise ValueError(f"OOF endpoint coverage drifted for macro: {keys}")
        macros.append(
            {
                **dict(zip(macro_keys, keys, strict=True)),
                "endpoint": "macro",
                "scale": "natural",
                "aggregation": "mean_of_pooled_endpoint_metrics",
                "n_test": int(group["n_test"].sum()),
                **{
                    name: float(group[name].mean())
                    for name in (*METRIC_NAMES, *CALIBRATION_NAMES)
                },
            }
        )
    result = pd.concat(
        [endpoint_frame, pd.DataFrame(macros)], ignore_index=True, sort=False
    )
    unique_keys = [*macro_keys, "endpoint", "scale"]
    if result.duplicated(unique_keys).any():
        raise ValueError("pooled OOF metrics contain duplicate result rows")
    return result


def _atomic_csv(path: Path, frame: pd.DataFrame, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False)
        os.chmod(temporary_path, mode)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_probe_outputs(
    result: ProbeResult,
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Any],
    feature_path: str | Path | None = None,
    feature_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one cell's selections, private OOF rows, metrics, and provenance."""

    output = Path(output_dir).resolve()
    paths = {
        "selection": output / "ridge_selection.csv",
        "prediction": output / "ridge_predictions.private.csv",
        "metrics": output / "probe_metrics.csv",
        "metadata": output / "probe_metadata.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite a frozen probe output")
    source = result.source_path if feature_path is None else Path(feature_path).resolve()
    source_metadata = (
        result.source_metadata_path
        if feature_metadata_path is None
        else Path(feature_metadata_path).resolve()
    )
    if source is not None:
        if not source.is_file() or not source.name.endswith(".private.npz"):
            raise ValueError("probe source must be an existing private pooled-state NPZ")
    if source_metadata is not None and not source_metadata.is_file():
        raise FileNotFoundError("probe source metadata is missing")
    provenance_payload = _jsonable(dict(provenance))
    provenance_sha256 = canonical_sha256(provenance_payload)
    _atomic_csv(paths["selection"], result.selection)
    _atomic_csv(paths["prediction"], result.predictions)
    _atomic_csv(paths["metrics"], result.metrics)
    metadata = {
        "schema_version": 1,
        **dict(result.identity),
        "feature_path": None if source is None else str(source),
        "feature_sha256": None if source is None else file_sha256(source),
        "feature_metadata_path": (
            None if source_metadata is None else str(source_metadata)
        ),
        "feature_metadata_sha256": (
            None if source_metadata is None else file_sha256(source_metadata)
        ),
        "patient_identifiers_private": True,
        "prediction_asset_private": True,
        "state_valid_enforced": True,
        "test_used_for_scaler_or_selection": False,
        "outer_test_predict_calls_per_cell": 1,
        "alpha_grid": list(ALPHAS),
        "ridge_selection_implementation": "immutable_stage_b_select_ridge",
        "static_target_implementation": "immutable_stage_b_target_adapter",
        "formal_probe_source_sha256": UPSTREAM_SOURCE_SHA256[_FORMAL_PROBE_RELATIVE],
        "formal_target_source_sha256": UPSTREAM_SOURCE_SHA256[_FORMAL_TARGET_RELATIVE],
        "probe_adapter_sha256": file_sha256(Path(__file__)),
        "provenance": provenance_payload,
        "provenance_sha256": provenance_sha256,
        "tasks": sorted(result.selection["task"].astype(str).unique()),
        "target_names": sorted(result.selection["target_name"].astype(str).unique()),
        "selection_rows": len(result.selection),
        "prediction_rows": len(result.predictions),
        "metric_rows": len(result.metrics),
        "output_sha256": {
            paths["selection"].name: file_sha256(paths["selection"]),
            paths["prediction"].name: file_sha256(paths["prediction"]),
            paths["metrics"].name: file_sha256(paths["metrics"]),
        },
    }
    _atomic_json(paths["metadata"], metadata)
    return metadata


__all__ = [
    "ALPHAS",
    "ANALYSIS_SCOPES",
    "CALIBRATION_NAMES",
    "FrozenStateAsset",
    "METRIC_NAMES",
    "PreparedSplit",
    "ProbeResult",
    "TestPredictGuard",
    "combine_probe_results",
    "fit_static_probe_transform",
    "literal_delta_targets",
    "load_frozen_state_asset",
    "ordered_patient_sha256",
    "pooled_oof_natural_metrics",
    "run_continuous_probe_cell",
    "run_ftv_probe_cell",
    "select_ridge",
    "static_targets",
    "write_probe_outputs",
]
