"""Fold-specific frozen landmark readouts for audit retraining.

This module intentionally lives outside ``corejepa``.  It reuses the clean
repository's exact :func:`landmark_features` transform, but keeps outcome
supervision confined to a fold's training and validation patients:

* candidate logistic regressions are fit on training patients only;
* the hyperparameter is selected on validation AUROC only;
* one deterministic threshold per decision point is selected on validation
  balanced accuracy only; and
* held-out labels are accepted only by the reporting function, after all
  probabilities and labels have already been computed.

The resulting bundle contains the fitted sklearn pipeline and the full
selection audit trail and can be pickled with :func:`save_readout_bundle`.
Only load bundles from trusted sources because pickle is executable data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ispy_jepa_tmi_clean.corejepa.readout.flr import landmark_features

from .contracts import DECISION_POINTS, validate_prediction_frame


BUNDLE_SCHEMA_VERSION = "shortcut_audit.fold_readout.v1"
THRESHOLD_OBJECTIVE = "balanced_accuracy"
THRESHOLD_CANDIDATES = "unique_validation_probabilities_plus_0_and_1"
THRESHOLD_TIE_BREAK = "closest_to_0.5_then_lower"
HYPERPARAMETER_TIE_BREAK = "smaller_C_then_l2"


@dataclass(frozen=True)
class AuditReadoutConfig:
    """Locked selection grid for a class-balanced frozen readout."""

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
    landmark_weights: tuple[float, float, float] = (2.0, 1.0, 0.5)
    max_iter: int = 5000
    random_state: int = 0


@dataclass(frozen=True)
class FoldReadoutBundle:
    """Serializable, fold-locked readout plus its selection provenance."""

    schema_version: str
    fold: int
    response_dim: int
    feature_dim: int
    model: Any
    thresholds: dict[str, float]
    hyperparameter_selection: dict[str, Any]
    threshold_selection: dict[str, dict[str, Any]]
    grid_search: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    train_patient_ids: tuple[str, ...]
    validation_patient_ids: tuple[str, ...]
    test_patient_ids: tuple[str, ...]

    def audit_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable audit trail, excluding model weights."""

        return {
            "schema_version": self.schema_version,
            "fold": self.fold,
            "response_dim": self.response_dim,
            "feature_dim": self.feature_dim,
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
            "supervision_contract": {
                "model_fit": "train_only",
                "hyperparameter_selection": "validation_only",
                "threshold_selection": "validation_only",
                "test": "prediction_and_reporting_only",
            },
        }


def _coerce_config(
    config: AuditReadoutConfig | Mapping[str, Any] | Any | None
) -> AuditReadoutConfig:
    if config is None:
        output = AuditReadoutConfig()
    elif isinstance(config, AuditReadoutConfig):
        output = config
    elif isinstance(config, Mapping):
        output = AuditReadoutConfig(**dict(config))
    else:
        # This deliberately supports the repository's corejepa.config.ReadoutConfig
        # without importing it or changing the core package.
        required = ("penalties", "c_grid", "landmark_weights", "max_iter")
        missing = [name for name in required if not hasattr(config, name)]
        if missing:
            raise TypeError(f"readout config 缺少字段：{missing}")
        output = AuditReadoutConfig(
            penalties=tuple(config.penalties),
            c_grid=tuple(config.c_grid),
            landmark_weights=tuple(config.landmark_weights),
            max_iter=int(config.max_iter),
        )

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
    weights = tuple(float(value) for value in output.landmark_weights)
    if len(weights) != len(DECISION_POINTS):
        raise ValueError("landmark_weights 必须恰有三个元素")
    if any(not np.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("landmark_weights 必须为有限正数")
    if int(output.max_iter) <= 0:
        raise ValueError("max_iter 必须为正整数")
    if (
        isinstance(output.random_state, bool)
        or int(output.random_state) != output.random_state
    ):
        raise ValueError("random_state 必须为整数")
    return AuditReadoutConfig(
        penalties=penalties,
        c_grid=c_grid,
        landmark_weights=weights,
        max_iter=int(output.max_iter),
        random_state=int(output.random_state),
    )


def _states(values: Any, *, name: str = "states") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1] != 3 or array.shape[2] <= 0:
        raise ValueError(f"{name} 必须为 [N,3,Ds]，收到 {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} 不得为空")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} 必须为数值数组")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 含非有限值")
    return array


def _patient_ids(
    values: Sequence[str], n_rows: int, *, name: str = "patient_ids"
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError(f"{name} 必须是一维且长度等于 states ({n_rows})")
    normalized = np.asarray([str(value).strip() for value in array], dtype=object)
    if any(value == "" for value in normalized):
        raise ValueError(f"{name} 不得含空 ID")
    if len(set(normalized.tolist())) != len(normalized):
        raise ValueError(f"{name} 含重复 ID")
    return normalized


def _indices(values: Sequence[int], n_rows: int, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} 必须是一维非空索引")
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
    values: Sequence[int] | None,
    n_rows: int,
    *,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.int64)
    return _indices(values, n_rows, name=name)


def _selected_binary_labels(
    values: Any, indices: np.ndarray, n_rows: int, *, name: str
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n_rows:
        raise ValueError(f"labels 必须是一维且长度等于 states ({n_rows})")
    # Index first: values outside train/validation are intentionally never
    # converted, inspected, or used by selection.
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
    """Validate held-out labels for reporting without requiring both classes."""

    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n_rows:
        raise ValueError(f"reporting labels 必须是一维且长度等于 states ({n_rows})")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("reporting labels 必须为 0/1") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("reporting labels 必须为 0/1")
    return numeric.astype(np.int64)


def _stack_landmarks(
    states: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features = [landmark_features(states, landmark) for landmark in range(3)]
    return np.concatenate(features, axis=0), np.concatenate([labels] * 3, axis=0)


def _model(penalty: str, c_value: float, config: AuditReadoutConfig) -> Any:
    arguments: dict[str, Any] = {
        "C": c_value,
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": config.max_iter,
        "random_state": config.random_state,
    }
    # sklearn 1.8 deprecated ``penalty=`` in favor of l1_ratio=0/1, while the
    # repository permits sklearn>=1.3.  Detect the API by its default so both
    # versions express the same l1/l2 grid without deprecation warnings.
    penalty_default = (
        inspect.signature(LogisticRegression).parameters["penalty"].default
    )
    if penalty_default == "deprecated":
        arguments["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        arguments["penalty"] = penalty
    estimator = LogisticRegression(**arguments)
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        estimator,
    )


def _probabilities_by_landmark(model: Any, states: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for landmark, decision_point in enumerate(DECISION_POINTS):
        probability = np.asarray(
            model.predict_proba(landmark_features(states, landmark))[:, 1],
            dtype=np.float64,
        )
        if probability.shape != (len(states),) or not np.isfinite(probability).all():
            raise RuntimeError(f"{decision_point} readout 返回无效 probability")
        if ((probability < 0.0) | (probability > 1.0)).any():
            raise RuntimeError(f"{decision_point} readout probability 超出 [0,1]")
        output[decision_point] = probability
    return output


def select_validation_threshold(labels: Any, probabilities: Any) -> dict[str, Any]:
    """Select a threshold using validation labels only.

    Candidates are the unique validation probabilities together with ``0`` and
    ``1``.  The objective is balanced accuracy.  Exact/near ties first choose
    the candidate closest to ``0.5`` and then the lower threshold.  Returning
    the full rule and operating point makes the choice independently auditable.
    """

    label_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("labels 与 probabilities 必须为一维")
    if len(label_array) == 0 or len(label_array) != len(probability_array):
        raise ValueError("labels 与 probabilities 必须等长且非空")
    try:
        label_float = label_array.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("labels 必须为 0/1") from error
    if not np.isfinite(label_float).all() or not np.isin(label_float, (0.0, 1.0)).all():
        raise ValueError("labels 必须为 0/1")
    labels_int = label_float.astype(np.int64)
    if len(np.unique(labels_int)) != 2:
        raise ValueError("threshold validation labels 必须同时包含两个类别")
    if (
        not np.isfinite(probability_array).all()
        or ((probability_array < 0.0) | (probability_array > 1.0)).any()
    ):
        raise ValueError("probabilities 必须为 [0,1] 内有限值")

    candidates = np.unique(np.concatenate((probability_array, [0.0, 1.0])))
    positive = labels_int == 1
    negative = ~positive
    records: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        prediction = probability_array >= threshold
        sensitivity = float(prediction[positive].mean())
        specificity = float((~prediction[negative]).mean())
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        records.append((float(threshold), balanced_accuracy, sensitivity, specificity))

    scores = np.asarray([record[1] for record in records], dtype=np.float64)
    best_score = float(scores.max())
    score_ties = [
        record
        for record in records
        if math.isclose(record[1], best_score, rel_tol=0.0, abs_tol=1e-12)
    ]
    minimum_distance = min(abs(record[0] - 0.5) for record in score_ties)
    distance_ties = [
        record
        for record in score_ties
        if math.isclose(
            abs(record[0] - 0.5), minimum_distance, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    chosen = min(distance_ties, key=lambda record: record[0])
    return {
        "threshold": chosen[0],
        "balanced_accuracy": chosen[1],
        "sensitivity": chosen[2],
        "specificity": chosen[3],
        "objective": THRESHOLD_OBJECTIVE,
        "candidate_rule": THRESHOLD_CANDIDATES,
        "tie_break": THRESHOLD_TIE_BREAK,
        "n_candidates": int(len(candidates)),
        "n_best_score_candidates": int(len(score_ties)),
        "n_final_tie_candidates": int(len(distance_ties)),
    }


def _validate_bundle(bundle: FoldReadoutBundle) -> None:
    if not isinstance(bundle, FoldReadoutBundle):
        raise TypeError("readout bundle 类型错误")
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"不支持的 readout bundle schema：{bundle.schema_version}")
    if isinstance(bundle.fold, bool) or bundle.fold not in range(5):
        raise ValueError("bundle fold 必须为 0..4")
    if bundle.response_dim <= 0 or bundle.feature_dim != 20 * bundle.response_dim + 3:
        raise ValueError("bundle response_dim/feature_dim contract 不一致")
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
    train = set(bundle.train_patient_ids)
    validation = set(bundle.validation_patient_ids)
    test = set(bundle.test_patient_ids)
    if not train or not validation:
        raise ValueError("bundle train/validation patient IDs 不得为空")
    if len(train) != len(bundle.train_patient_ids) or len(validation) != len(
        bundle.validation_patient_ids
    ):
        raise ValueError("bundle patient IDs 含重复")
    if len(test) != len(bundle.test_patient_ids):
        raise ValueError("bundle test patient IDs 含重复")
    if train & validation or train & test or validation & test:
        raise ValueError("bundle train/validation/test patient IDs 必须互斥")


def fit_fold_readout(
    states: Any,
    labels: Any,
    patient_ids: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    fold: int,
    test_indices: Sequence[int] | None = None,
    config: AuditReadoutConfig | Mapping[str, Any] | Any | None = None,
) -> FoldReadoutBundle:
    """Fit and select one fold-specific shared-landmark frozen readout.

    ``states`` must be the repository's ``FutureResponseState.future_state``
    representation with shape ``[N,3,Ds]``.  Although ``labels`` is aligned to
    all ``N`` rows for convenient use with ``frozen_states.npz``, only entries
    selected by ``train_indices`` and ``validation_indices`` are ever inspected.
    ``test_indices`` supplies identity provenance only; no held-out label enters
    fitting, model selection, or threshold selection.
    """

    if isinstance(fold, bool) or int(fold) != fold or int(fold) not in range(5):
        raise ValueError("fold 必须为 0..4")
    fold = int(fold)
    normalized_config = _coerce_config(config)
    state_array = _states(states)
    n_rows = len(state_array)
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
    train_states = state_array[train]
    validation_states = state_array[validation]
    train_x, train_y = _stack_landmarks(train_states, train_labels)

    selection_weights = np.asarray(normalized_config.landmark_weights, dtype=np.float64)
    grid_rows: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, float, int] | None = None
    for penalty in normalized_config.penalties:
        for c_value in normalized_config.c_grid:
            candidate_model = _model(penalty, c_value, normalized_config)
            candidate_model.fit(train_x, train_y)
            validation_probabilities = _probabilities_by_landmark(
                candidate_model, validation_states
            )
            landmark_auc = {
                decision_point: float(
                    roc_auc_score(
                        validation_labels, validation_probabilities[decision_point]
                    )
                )
                for decision_point in DECISION_POINTS
            }
            weighted_auc = float(
                np.dot(
                    [landmark_auc[name] for name in DECISION_POINTS],
                    selection_weights,
                )
                / selection_weights.sum()
            )
            row = {
                "penalty": penalty,
                "C": float(c_value),
                "validation_selection_auroc": weighted_auc,
                "validation_decision_point_auroc": landmark_auc,
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
                best_model = candidate_model

    if best_model is None or best_row is None:
        raise RuntimeError("readout hyperparameter grid 为空")

    validation_probabilities = _probabilities_by_landmark(best_model, validation_states)
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
        "landmark_weights": {
            name: float(weight)
            for name, weight in zip(DECISION_POINTS, selection_weights, strict=True)
        },
        "tie_break": HYPERPARAMETER_TIE_BREAK,
    }
    bundle = FoldReadoutBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        fold=fold,
        response_dim=int(state_array.shape[2]),
        feature_dim=int(train_x.shape[1]),
        model=best_model,
        thresholds=thresholds,
        hyperparameter_selection=selected,
        threshold_selection=threshold_selection,
        grid_search=tuple(grid_rows),
        config=asdict(normalized_config),
        train_patient_ids=tuple(str(value) for value in id_array[train]),
        validation_patient_ids=tuple(str(value) for value in id_array[validation]),
        test_patient_ids=tuple(str(value) for value in id_array[test]),
    )
    _validate_bundle(bundle)
    return bundle


def predict_fold_readout(
    bundle: FoldReadoutBundle,
    states: Any,
    patient_ids: Sequence[str],
    labels: Any,
    *,
    checkpoint: str,
    audit_condition: str = "native",
    enforce_test_patient_order: bool = True,
) -> pd.DataFrame:
    """Predict all three decision points and return the audit CSV contract.

    Probabilities and thresholded labels are computed before ``labels`` is
    validated or added to the output.  Consequently held-out outcomes can only
    affect the reporting column ``y_true``.  If the bundle records test patient
    IDs, their exact order is enforced by default to catch fold/order drift.
    """

    _validate_bundle(bundle)
    state_array = _states(states)
    if state_array.shape[2] != bundle.response_dim:
        raise ValueError(
            f"states response dim {state_array.shape[2]} != bundle {bundle.response_dim}"
        )
    id_array = _patient_ids(patient_ids, len(state_array))
    if enforce_test_patient_order and bundle.test_patient_ids:
        received = tuple(str(value) for value in id_array)
        if received != bundle.test_patient_ids:
            raise ValueError("prediction patient IDs/order 与 bundle test split 不一致")

    # Do not move this below any access to labels: held-out outcomes must have
    # no path into probability or threshold computation.
    probabilities = _probabilities_by_landmark(bundle.model, state_array)
    predictions = {
        decision_point: (
            probabilities[decision_point] >= bundle.thresholds[decision_point]
        ).astype(np.int64)
        for decision_point in DECISION_POINTS
    }

    label_array = _reporting_binary_labels(labels, len(state_array))
    rows: list[dict[str, Any]] = []
    checkpoint = str(checkpoint).strip()
    audit_condition = str(audit_condition).strip()
    for row, patient_id in enumerate(id_array):
        for decision_point in DECISION_POINTS:
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "fold": bundle.fold,
                    "decision_point": decision_point,
                    "audit_condition": audit_condition,
                    "y_true": int(label_array[row]),
                    "predicted_probability": float(probabilities[decision_point][row]),
                    "predicted_label": int(predictions[decision_point][row]),
                    "threshold": float(bundle.thresholds[decision_point]),
                    "checkpoint": checkpoint,
                    "donor_patient_id": pd.NA,
                    "repetition_id": pd.NA,
                    "matching_distance": np.nan,
                }
            )
    return validate_prediction_frame(pd.DataFrame(rows))


def predict_readout_probability_matrix(
    bundle: FoldReadoutBundle,
    states: Any,
) -> np.ndarray:
    """对任意对齐 state 行返回 ``[N,3]`` 概率，不读取 ID 或 label。

    该入口用于 donor repetitions 等允许同一 recipient 出现多次的审计。调用方仍须
    通过 prediction contract 保存 patient/donor provenance。
    """

    _validate_bundle(bundle)
    state_array = _states(states)
    if state_array.shape[2] != bundle.response_dim:
        raise ValueError(
            f"states response dim {state_array.shape[2]} != bundle {bundle.response_dim}"
        )
    probabilities = _probabilities_by_landmark(bundle.model, state_array)
    return np.stack(
        [probabilities[decision_point] for decision_point in DECISION_POINTS], axis=1
    )


def save_readout_bundle(bundle: FoldReadoutBundle, path: str | Path) -> Path:
    """Atomically pickle a validated fold readout bundle."""

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


def load_readout_bundle(path: str | Path) -> FoldReadoutBundle:
    """Load and validate a trusted bundle written by this module."""

    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    _validate_bundle(bundle)
    return bundle


__all__ = [
    "AuditReadoutConfig",
    "BUNDLE_SCHEMA_VERSION",
    "FoldReadoutBundle",
    "fit_fold_readout",
    "load_readout_bundle",
    "predict_fold_readout",
    "predict_readout_probability_matrix",
    "save_readout_bundle",
    "select_validation_threshold",
]
