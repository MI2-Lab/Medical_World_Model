"""Fold-specific logistic readouts for the five simplified-input baselines.

The caller constructs one feature tensor with shape ``[N, 3, F]`` using
``baseline_features``.  A *single* class-balanced logistic model is fitted to
the three decision points stacked along the sample axis.  Model selection and
threshold selection are deliberately separated from held-out reporting:

* every candidate is fitted on fold-train patients only;
* penalty/C are selected by weighted fold-validation AUROC;
* one threshold per decision point is selected by validation balanced
  accuracy; and
* held-out outcomes are read only after probabilities and thresholded labels
  have been computed.

The bundle records feature names as an order contract.  Prediction therefore
requires the same names in the same order, in addition to the exact held-out
patient order recorded at fit time.  Pickled bundles must only be loaded from
trusted sources because pickle is executable data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DECISION_POINTS, validate_prediction_frame
from .readouts import select_validation_threshold


BUNDLE_SCHEMA_VERSION = "shortcut_audit.simplified_baseline_readout.v1"
BASELINE_FEATURE_SETS = {
    "F1": "clinical_only",
    "F2": "geometry_only",
    "F3": "clinical_plus_geometry",
    "F4": "timepoint_only",
    "F5": "static_t0_imaging",
}
HYPERPARAMETER_TIE_BREAK = "smaller_C_then_l2"


@dataclass(frozen=True)
class BaselineReadoutConfig:
    """Locked grid and validation weights for a simplified baseline."""

    penalties: tuple[str, ...] = ("l1", "l2")
    c_grid: tuple[float, ...] = (
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
    )
    decision_weights: tuple[float, float, float] = (2.0, 1.0, 0.5)
    max_iter: int = 5000
    random_state: int = 0


@dataclass(frozen=True)
class FoldBaselineBundle:
    """Serializable fold-locked model and complete selection provenance."""

    schema_version: str
    baseline_id: str
    baseline_name: str
    fold: int
    feature_dim: int
    feature_names: tuple[str, ...]
    model: Any
    thresholds: dict[str, float]
    hyperparameter_selection: dict[str, Any]
    threshold_selection: dict[str, dict[str, Any]]
    grid_search: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    train_patient_ids: tuple[str, ...]
    validation_patient_ids: tuple[str, ...]
    test_patient_ids: tuple[str, ...]
    feature_provenance: dict[str, Any]
    software_provenance: dict[str, str]

    def audit_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable provenance without estimator weights."""

        return {
            "schema_version": self.schema_version,
            "baseline_id": self.baseline_id,
            "baseline_name": self.baseline_name,
            "fold": self.fold,
            "tensor_contract": "[N,3,F]",
            "decision_points": list(DECISION_POINTS),
            "feature_dim": self.feature_dim,
            "feature_names": list(self.feature_names),
            "thresholds": dict(self.thresholds),
            "hyperparameter_selection": dict(self.hyperparameter_selection),
            "threshold_selection": {
                name: dict(values) for name, values in self.threshold_selection.items()
            },
            "grid_search": [dict(values) for values in self.grid_search],
            "config": dict(self.config),
            "train_patient_ids": list(self.train_patient_ids),
            "validation_patient_ids": list(self.validation_patient_ids),
            "test_patient_ids": list(self.test_patient_ids),
            "feature_provenance": dict(self.feature_provenance),
            "software_provenance": dict(self.software_provenance),
            "supervision_contract": {
                "model_fit": "train_only_shared_three_decision_stack",
                "hyperparameter_selection": "validation_weighted_auroc_only",
                "threshold_selection": "validation_balanced_accuracy_only",
                "test": "prediction_and_reporting_only",
            },
        }


def _baseline_id(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in BASELINE_FEATURE_SETS:
        raise ValueError(
            f"baseline_id 必须为 {tuple(BASELINE_FEATURE_SETS)}，收到 {value!r}"
        )
    return normalized


def _coerce_config(
    config: BaselineReadoutConfig | Mapping[str, Any] | None,
) -> BaselineReadoutConfig:
    if config is None:
        output = BaselineReadoutConfig()
    elif isinstance(config, BaselineReadoutConfig):
        output = config
    elif isinstance(config, Mapping):
        output = BaselineReadoutConfig(**dict(config))
    else:
        raise TypeError("config 必须为 BaselineReadoutConfig、mapping 或 None")

    penalties = tuple(str(value).lower() for value in output.penalties)
    if not penalties or any(value not in {"l1", "l2"} for value in penalties):
        raise ValueError("penalties 必须是非空的 l1/l2 序列")
    if len(set(penalties)) != len(penalties):
        raise ValueError("penalties 不得重复")
    c_grid = tuple(float(value) for value in output.c_grid)
    if not c_grid or any(not np.isfinite(value) or value <= 0 for value in c_grid):
        raise ValueError("c_grid 必须包含有限正数")
    if len(set(c_grid)) != len(c_grid):
        raise ValueError("c_grid 不得重复")
    weights = tuple(float(value) for value in output.decision_weights)
    if len(weights) != len(DECISION_POINTS):
        raise ValueError("decision_weights 必须恰有三个元素")
    if any(not np.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("decision_weights 必须为有限正数")
    if isinstance(output.max_iter, bool) or int(output.max_iter) != output.max_iter:
        raise ValueError("max_iter 必须为正整数")
    if int(output.max_iter) <= 0:
        raise ValueError("max_iter 必须为正整数")
    if (
        isinstance(output.random_state, bool)
        or int(output.random_state) != output.random_state
    ):
        raise ValueError("random_state 必须为整数")
    return BaselineReadoutConfig(
        penalties=penalties,
        c_grid=c_grid,
        decision_weights=weights,
        max_iter=int(output.max_iter),
        random_state=int(output.random_state),
    )


def _features(values: Any, *, name: str = "features") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1] != len(DECISION_POINTS) or array.shape[2] <= 0:
        raise ValueError(f"{name} 必须为 [N,3,F] 且 F>0，收到 {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} 不得为空")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} 必须为数值数组")
    output = array.astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError(f"{name} 含非有限值")
    return output


def _feature_names(values: Sequence[str], feature_dim: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("feature_names 必须为字符串序列，不接受单个字符串")
    names = tuple(str(value).strip() for value in values)
    if len(names) != feature_dim:
        raise ValueError(
            f"feature_names 长度 {len(names)} != feature dimension {feature_dim}"
        )
    if any(not value for value in names):
        raise ValueError("feature_names 不得含空名称")
    if len(set(names)) != len(names):
        raise ValueError("feature_names 不得重复")
    return names


def _patient_ids(
    values: Sequence[str], n_rows: int, *, name: str = "patient_ids"
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError(f"{name} 必须一维且长度等于 features ({n_rows})")
    normalized = np.asarray([str(value).strip() for value in array], dtype=object)
    if any(value == "" for value in normalized):
        raise ValueError(f"{name} 不得含空 ID")
    if len(set(normalized.tolist())) != len(normalized):
        raise ValueError(f"{name} 含重复 ID")
    return normalized


def _indices(values: Sequence[int], n_rows: int, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} 必须为一维非空索引")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} 必须为整数索引，不接受布尔 mask")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} 必须为整数索引") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} 必须为有限整数索引")
    output = numeric.astype(np.int64)
    if (output < 0).any() or (output >= n_rows).any():
        raise IndexError(f"{name} 越界")
    if len(np.unique(output)) != len(output):
        raise ValueError(f"{name} 含重复索引")
    return output


def _optional_indices(
    values: Sequence[int] | None, n_rows: int, *, name: str
) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.int64)
    return _indices(values, n_rows, name=name)


def _selected_binary_labels(
    values: Any, indices: np.ndarray, n_rows: int, *, name: str
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n_rows:
        raise ValueError(f"labels 必须一维且长度等于 features ({n_rows})")
    # Index before conversion so held-out labels are never inspected by fit or
    # validation selection, even when callers use one aligned object array.
    selected_raw = raw[indices]
    try:
        selected = selected_raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} labels 必须为 0/1") from error
    if not np.isfinite(selected).all() or not np.isin(selected, (0.0, 1.0)).all():
        raise ValueError(f"{name} labels 必须为 0/1")
    output = selected.astype(np.int64)
    if len(np.unique(output)) != 2:
        raise ValueError(f"{name} labels 必须同时包含两个类别")
    return output


def _reporting_binary_labels(values: Any, n_rows: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n_rows:
        raise ValueError(f"reporting labels 必须一维且长度等于 features ({n_rows})")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("reporting labels 必须为 0/1") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("reporting labels 必须为 0/1")
    return numeric.astype(np.int64)


def _json_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("feature_provenance 必须为 mapping 或 None")
    try:
        serialized = json.dumps(dict(values), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError("feature_provenance 必须可 JSON 序列化且不得含 NaN") from error
    restored = json.loads(serialized)
    if not isinstance(restored, dict):
        raise TypeError("feature_provenance 必须序列化为 JSON object")
    return restored


def _model(
    penalty: str, c_value: float, config: BaselineReadoutConfig
) -> Any:
    arguments: dict[str, Any] = {
        "C": c_value,
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": config.max_iter,
        "random_state": config.random_state,
    }
    penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
    if penalty_default == "deprecated":
        arguments["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        arguments["penalty"] = penalty
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(**arguments),
    )


def _probabilities_by_decision(
    model: Any, features: np.ndarray
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for decision_index, decision_point in enumerate(DECISION_POINTS):
        probability = np.asarray(
            model.predict_proba(features[:, decision_index, :])[:, 1],
            dtype=np.float64,
        )
        if probability.shape != (len(features),) or not np.isfinite(probability).all():
            raise RuntimeError(f"{decision_point} baseline 返回无效 probability")
        if ((probability < 0.0) | (probability > 1.0)).any():
            raise RuntimeError(f"{decision_point} baseline probability 超出 [0,1]")
        output[decision_point] = probability
    return output


def _validate_bundle(bundle: FoldBaselineBundle) -> None:
    if not isinstance(bundle, FoldBaselineBundle):
        raise TypeError("baseline bundle 类型错误")
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"不支持的 baseline bundle schema：{bundle.schema_version}")
    baseline_id = _baseline_id(bundle.baseline_id)
    if bundle.baseline_name != BASELINE_FEATURE_SETS[baseline_id]:
        raise ValueError("bundle baseline_id/name 不一致")
    if isinstance(bundle.fold, bool) or bundle.fold not in range(5):
        raise ValueError("bundle fold 必须为 0..4")
    if bundle.feature_dim <= 0:
        raise ValueError("bundle feature_dim 必须为正数")
    _feature_names(bundle.feature_names, bundle.feature_dim)
    if set(bundle.thresholds) != set(DECISION_POINTS):
        raise ValueError("bundle thresholds 必须覆盖三个 decision point")
    if any(
        not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in bundle.thresholds.values()
    ):
        raise ValueError("bundle threshold 无效")
    if set(bundle.threshold_selection) != set(DECISION_POINTS):
        raise ValueError("bundle threshold selection provenance 不完整")
    if not hasattr(bundle.model, "predict_proba"):
        raise TypeError("bundle model 不支持 predict_proba")
    if getattr(bundle.model, "n_features_in_", bundle.feature_dim) != bundle.feature_dim:
        raise ValueError("bundle estimator feature dimension 不一致")
    train = set(bundle.train_patient_ids)
    validation = set(bundle.validation_patient_ids)
    test = set(bundle.test_patient_ids)
    if not train or not validation:
        raise ValueError("bundle train/validation patient IDs 不得为空")
    if len(train) != len(bundle.train_patient_ids):
        raise ValueError("bundle train patient IDs 含重复")
    if len(validation) != len(bundle.validation_patient_ids):
        raise ValueError("bundle validation patient IDs 含重复")
    if len(test) != len(bundle.test_patient_ids):
        raise ValueError("bundle test patient IDs 含重复")
    if train & validation or train & test or validation & test:
        raise ValueError("bundle train/validation/test patient IDs 必须互斥")
    _json_mapping(bundle.feature_provenance)
    _json_mapping(bundle.software_provenance)
    # Ensure metadata remains portable rather than discovering an opaque value
    # only after a costly formal fold has finished.
    json.dumps(bundle.audit_metadata(), sort_keys=True, allow_nan=False)


def fit_fold_baseline(
    features: Any,
    labels: Any,
    patient_ids: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    fold: int,
    baseline_id: str,
    feature_names: Sequence[str],
    test_indices: Sequence[int] | None = None,
    config: BaselineReadoutConfig | Mapping[str, Any] | None = None,
    feature_provenance: Mapping[str, Any] | None = None,
) -> FoldBaselineBundle:
    """Fit one fold-specific shared logistic model for F1--F5.

    ``labels`` may be aligned to all rows, but only the train and validation
    selections are converted or inspected.  ``test_indices`` records identity
    provenance and does not expose held-out outcomes to fitting.
    """

    if isinstance(fold, bool) or int(fold) != fold or int(fold) not in range(5):
        raise ValueError("fold 必须为 0..4")
    fold = int(fold)
    normalized_id = _baseline_id(baseline_id)
    normalized_config = _coerce_config(config)
    feature_array = _features(features)
    n_rows, _, feature_dim = feature_array.shape
    names = _feature_names(feature_names, feature_dim)
    id_array = _patient_ids(patient_ids, n_rows)
    train = _indices(train_indices, n_rows, name="train_indices")
    validation = _indices(validation_indices, n_rows, name="validation_indices")
    test = _optional_indices(test_indices, n_rows, name="test_indices")
    if set(train.tolist()) & set(validation.tolist()):
        raise ValueError("train_indices 与 validation_indices 不得重叠")
    if set(train.tolist()) & set(test.tolist()) or set(validation.tolist()) & set(
        test.tolist()
    ):
        raise ValueError("test_indices 必须与 train/validation 互斥")

    train_labels = _selected_binary_labels(labels, train, n_rows, name="train")
    validation_labels = _selected_binary_labels(
        labels, validation, n_rows, name="validation"
    )
    train_x = feature_array[train].reshape(-1, feature_dim)
    train_y = np.repeat(train_labels, len(DECISION_POINTS))
    validation_features = feature_array[validation]

    weights = np.asarray(normalized_config.decision_weights, dtype=np.float64)
    grid_rows: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, float, int] | None = None
    for penalty in normalized_config.penalties:
        for c_value in normalized_config.c_grid:
            candidate = _model(penalty, c_value, normalized_config)
            candidate.fit(train_x, train_y)
            validation_probabilities = _probabilities_by_decision(
                candidate, validation_features
            )
            decision_auc = {
                decision_point: float(
                    roc_auc_score(
                        validation_labels,
                        validation_probabilities[decision_point],
                    )
                )
                for decision_point in DECISION_POINTS
            }
            weighted_auc = float(
                np.dot([decision_auc[name] for name in DECISION_POINTS], weights)
                / weights.sum()
            )
            row = {
                "penalty": penalty,
                "C": float(c_value),
                "validation_selection_auroc": weighted_auc,
                "validation_decision_point_auroc": decision_auc,
            }
            grid_rows.append(row)
            key = (
                weighted_auc,
                -math.log10(float(c_value)),
                int(penalty == "l2"),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
                best_model = candidate

    if best_model is None or best_row is None:
        raise RuntimeError("baseline hyperparameter grid 为空")

    validation_probabilities = _probabilities_by_decision(
        best_model, validation_features
    )
    threshold_selection = {
        decision_point: select_validation_threshold(
            validation_labels, validation_probabilities[decision_point]
        )
        for decision_point in DECISION_POINTS
    }
    thresholds = {
        decision_point: float(threshold_selection[decision_point]["threshold"])
        for decision_point in DECISION_POINTS
    }
    selected = {
        **best_row,
        "selection_rule": "weighted_validation_auroc",
        "decision_weights": {
            name: float(weight)
            for name, weight in zip(DECISION_POINTS, weights, strict=True)
        },
        "tie_break": HYPERPARAMETER_TIE_BREAK,
    }
    bundle = FoldBaselineBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        baseline_id=normalized_id,
        baseline_name=BASELINE_FEATURE_SETS[normalized_id],
        fold=fold,
        feature_dim=int(feature_dim),
        feature_names=names,
        model=best_model,
        thresholds=thresholds,
        hyperparameter_selection=selected,
        threshold_selection=threshold_selection,
        grid_search=tuple(grid_rows),
        config=asdict(normalized_config),
        train_patient_ids=tuple(str(value) for value in id_array[train]),
        validation_patient_ids=tuple(str(value) for value in id_array[validation]),
        test_patient_ids=tuple(str(value) for value in id_array[test]),
        feature_provenance=_json_mapping(feature_provenance),
        software_provenance={
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    )
    _validate_bundle(bundle)
    return bundle


def predict_fold_baseline(
    bundle: FoldBaselineBundle,
    features: Any,
    patient_ids: Sequence[str],
    labels: Any,
    *,
    feature_names: Sequence[str],
    checkpoint: str,
    audit_condition: str | None = None,
    enforce_test_patient_order: bool = True,
) -> pd.DataFrame:
    """Predict held-out rows and return the common prediction CSV contract.

    Probabilities and labels thresholded from them are computed before the
    reporting outcomes are validated.  Therefore ``labels`` has no route into
    model fitting, hyperparameter choice, threshold choice, or probabilities.
    """

    _validate_bundle(bundle)
    feature_array = _features(features)
    if feature_array.shape[2] != bundle.feature_dim:
        raise ValueError(
            f"features dim {feature_array.shape[2]} != bundle {bundle.feature_dim}"
        )
    received_names = _feature_names(feature_names, bundle.feature_dim)
    if received_names != bundle.feature_names:
        raise ValueError("prediction feature_names/order 与 bundle 不一致")
    id_array = _patient_ids(patient_ids, len(feature_array))
    if enforce_test_patient_order and bundle.test_patient_ids:
        received_ids = tuple(str(value) for value in id_array)
        if received_ids != bundle.test_patient_ids:
            raise ValueError("prediction patient IDs/order 与 bundle test split 不一致")

    # Keep this before all access to labels: held-out outcomes are reporting
    # metadata only and cannot alter probability or threshold computation.
    probabilities = _probabilities_by_decision(bundle.model, feature_array)
    predictions = {
        decision_point: (
            probabilities[decision_point] >= bundle.thresholds[decision_point]
        ).astype(np.int64)
        for decision_point in DECISION_POINTS
    }

    label_array = _reporting_binary_labels(labels, len(feature_array))
    condition = (
        f"simplified_baseline_{bundle.baseline_id.lower()}_{bundle.baseline_name}"
        if audit_condition is None
        else str(audit_condition).strip()
    )
    checkpoint = str(checkpoint).strip()
    rows: list[dict[str, Any]] = []
    for row, patient_id in enumerate(id_array):
        for decision_point in DECISION_POINTS:
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "fold": bundle.fold,
                    "decision_point": decision_point,
                    "audit_condition": condition,
                    "y_true": int(label_array[row]),
                    "predicted_probability": float(
                        probabilities[decision_point][row]
                    ),
                    "predicted_label": int(predictions[decision_point][row]),
                    "threshold": float(bundle.thresholds[decision_point]),
                    "checkpoint": checkpoint,
                    "donor_patient_id": pd.NA,
                    "repetition_id": pd.NA,
                    "matching_distance": np.nan,
                }
            )
    return validate_prediction_frame(pd.DataFrame(rows))


def save_baseline_bundle(bundle: FoldBaselineBundle, path: str | Path) -> Path:
    """Atomically pickle a validated baseline bundle."""

    _validate_bundle(bundle)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_baseline_bundle(path: str | Path) -> FoldBaselineBundle:
    """Load and validate a trusted bundle produced by this module."""

    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    _validate_bundle(bundle)
    return bundle


__all__ = [
    "BASELINE_FEATURE_SETS",
    "BUNDLE_SCHEMA_VERSION",
    "BaselineReadoutConfig",
    "FoldBaselineBundle",
    "fit_fold_baseline",
    "load_baseline_bundle",
    "predict_fold_baseline",
    "save_baseline_bundle",
]
