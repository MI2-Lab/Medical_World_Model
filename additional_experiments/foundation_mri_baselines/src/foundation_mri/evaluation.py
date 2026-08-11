"""Model-independent, leakage-guarded evaluation for frozen MRI features.

All hyperparameter selectors in this module deliberately omit test arguments.
The outer-test partition is transformed and predicted exactly once, after a
validation-only selection object exists.  Patient-level outputs are returned
to callers, but the only writer that accepts them requires ``.private.csv``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

from .data import ClinicalTable, FOLDS, FoldManifest, SEED


DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
TIMING_SCHEMAS = {
    "T0": "z0",
    "T0-T1": "concat(z0,z1,z1-z0)",
    "T0-T2": "concat(z0,z1,z2,z1-z0,z2-z1,z2-z0)",
}
TIMING_MULTIPLIERS = {"T0": 1, "T0-T1": 3, "T0-T2": 6}
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
PENALTIES = ("l1", "l2")
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
LOGISTIC_MAX_ITER = 20_000
BINARY_LOGISTIC_TOL = 1e-7
MULTICLASS_LOGISTIC_TOL = 1e-4
RIDGE_MAX_ITER = 10_000
RIDGE_TOL = 1e-8
ECE_BINS = 10
SELECTION_WORKER_ENV = "FOUNDATION_MRI_SELECTION_WORKERS"

_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]*$")
_ABSOLUTE_WINDOWS_RE = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_PROGRESS_KEY_FRAGMENTS = (
    "metric",
    "auc",
    "score",
    "probability",
    "patient",
    "target",
    "y_true",
    "y_pred",
)
_METRIC_FREE_PROGRESS_LOCK = threading.Lock()
_METRIC_FREE_PROGRESS_FD: int | None = None
_METRIC_FREE_PROGRESS_PATH: Path | None = None


def _selection_worker_count(task_count: int) -> int:
    raw = os.environ.get(SELECTION_WORKER_ENV, "1").strip()
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(f"{SELECTION_WORKER_ENV} must be an integer") from error
    if workers < 1 or workers > 32:
        raise ValueError(f"{SELECTION_WORKER_ENV} must be between 1 and 32")
    return min(workers, max(1, int(task_count)))


def _ordered_selection_map(function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
    workers = _selection_worker_count(len(items))
    if workers == 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mri-select") as executor:
        return list(executor.map(function, items))


def _safe_label(value: str, label: str) -> str:
    text = str(value).strip()
    if not _SAFE_LABEL_RE.fullmatch(text):
        raise ValueError(f"{label} must be a path-free safe identifier")
    return text


def configure_metric_free_progress(path: str | Path | None) -> Path | None:
    """Configure an exclusive, private JSONL progress sink, or disable it.

    The descriptor remains open so a later path replacement cannot redirect
    progress records.  Reusing an existing path fails closed via ``O_EXCL``.
    """

    global _METRIC_FREE_PROGRESS_FD, _METRIC_FREE_PROGRESS_PATH
    if path is None:
        with _METRIC_FREE_PROGRESS_LOCK:
            if _METRIC_FREE_PROGRESS_FD is not None:
                os.close(_METRIC_FREE_PROGRESS_FD)
            _METRIC_FREE_PROGRESS_FD = None
            _METRIC_FREE_PROGRESS_PATH = None
        return None

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with _METRIC_FREE_PROGRESS_LOCK:
            if _METRIC_FREE_PROGRESS_FD is not None:
                os.close(_METRIC_FREE_PROGRESS_FD)
            _METRIC_FREE_PROGRESS_FD = descriptor
            _METRIC_FREE_PROGRESS_PATH = destination
            descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return destination


def _metric_free_progress_scalar(key: str, value: Any) -> str | int | float | bool | None:
    name = str(key)
    lowered = name.lower()
    if not _SAFE_LABEL_RE.fullmatch(name):
        raise ValueError(f"progress field key is unsafe: {name!r}")
    if any(fragment in lowered for fragment in _FORBIDDEN_PROGRESS_KEY_FRAGMENTS):
        raise ValueError(f"progress field key is forbidden: {name!r}")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"progress field {name!r} must be finite")
        return value
    raise TypeError(f"progress field {name!r} must be a JSON scalar")


def metric_free_progress(event: str, **safe_fields: Any) -> None:
    """Append one outcome-blind progress record and mirror it to stderr."""

    event_name = _safe_label(event, "progress event")
    payload: dict[str, Any] = {"event": event_name}
    payload.update(
        {
            str(key): _metric_free_progress_scalar(str(key), value)
            for key, value in safe_fields.items()
        }
    )
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    encoded = (line + "\n").encode("utf-8")
    with _METRIC_FREE_PROGRESS_LOCK:
        if _METRIC_FREE_PROGRESS_FD is None:
            return
        view = memoryview(encoded)
        while view:
            written = os.write(_METRIC_FREE_PROGRESS_FD, view)
            if written <= 0:
                raise OSError("progress JSONL write made no progress")
            view = view[written:]
        # The private JSONL descriptor is the authoritative progress record and
        # remains fail-closed.  A detached PTY must not invalidate an otherwise
        # durable record or terminate a multi-hour formal run.
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except (BrokenPipeError, OSError):
            pass


def _logistic_candidate_context(
    *, solver: str, penalty: str, c_value: float, estimator_class: str | None
) -> str:
    context = f"{solver} logistic candidate penalty={penalty} C={float(c_value):g}"
    if estimator_class is not None:
        context += f" estimator_class={estimator_class}"
    return context


def _validated_logistic_iterations(
    model: LogisticRegression,
    *,
    solver: str,
    penalty: str,
    c_value: float,
    estimator_class: str | None = None,
) -> tuple[int, ...]:
    """Require integer counts in liblinear's valid ``0 <= n_iter < max_iter`` range."""

    context = _logistic_candidate_context(
        solver=solver,
        penalty=penalty,
        c_value=c_value,
        estimator_class=estimator_class,
    )
    raw_max_iter = getattr(model, "max_iter", None)
    if (
        isinstance(raw_max_iter, (bool, np.bool_))
        or not isinstance(raw_max_iter, (int, np.integer))
        or int(raw_max_iter) <= 0
    ):
        raise RuntimeError(f"{context} has invalid max_iter={raw_max_iter!r}")
    raw_iterations = getattr(model, "n_iter_", None)
    try:
        iteration_array = np.asarray(raw_iterations)
    except Exception as error:
        raise RuntimeError(f"{context} has invalid n_iter_") from error
    if (
        iteration_array.size == 0
        or np.issubdtype(iteration_array.dtype, np.bool_)
        or not np.issubdtype(iteration_array.dtype, np.integer)
    ):
        raise RuntimeError(f"{context} has empty or invalid n_iter_={raw_iterations!r}")
    iterations = np.asarray(iteration_array, dtype=np.int64).reshape(-1)
    if np.any(iterations < 0):
        raise RuntimeError(f"{context} has negative n_iter_={iterations.tolist()}")
    max_iter = int(raw_max_iter)
    if np.any(iterations >= max_iter):
        raise RuntimeError(
            f"{context} did not converge strictly before max_iter={max_iter}: "
            f"n_iter_={iterations.tolist()}"
        )
    return tuple(int(value) for value in iterations)


def _fit_validated_logistic(
    model: LogisticRegression,
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    solver: str,
    penalty: str,
    c_value: float,
    estimator_class: str | None = None,
) -> tuple[LogisticRegression, tuple[int, ...]]:
    """Fit once, promote convergence warnings, and validate solver state."""

    context = _logistic_candidate_context(
        solver=solver,
        penalty=penalty,
        c_value=c_value,
        estimator_class=estimator_class,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = model.fit(matrix, labels)
    except ConvergenceWarning as error:
        raise RuntimeError(f"{context} emitted ConvergenceWarning") from error
    iterations = _validated_logistic_iterations(
        fitted,
        solver=solver,
        penalty=penalty,
        c_value=c_value,
        estimator_class=estimator_class,
    )
    for parameter_name in ("coef_", "intercept_"):
        raw_parameter = getattr(fitted, parameter_name, None)
        try:
            parameter = np.asarray(raw_parameter, dtype=np.float64)
        except Exception as error:
            raise RuntimeError(
                f"{context} has invalid fitted {parameter_name}"
            ) from error
        if parameter.size == 0 or not np.isfinite(parameter).all():
            raise RuntimeError(f"{context} has empty or non-finite fitted {parameter_name}")
    return fitted, iterations


def _logistic_penalty_kwargs(penalty: str) -> dict[str, Any]:
    """Bridge sklearn's 1.8 penalty-to-l1_ratio API transition."""

    default = inspect.signature(LogisticRegression).parameters["penalty"].default
    if default == "deprecated":
        return {"l1_ratio": 1.0 if penalty == "l1" else 0.0}
    return {"penalty": penalty}


def _array_audit(prefix: str, values: np.ndarray) -> dict[str, Any]:
    """Compact deterministic provenance for potentially very wide parameters."""

    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    if array.size == 0 or not np.isfinite(array).all():
        raise FloatingPointError(f"{prefix} audit requires nonempty finite values")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    flat = array.reshape(-1)
    return {
        f"{prefix}_shape_json": json.dumps(list(array.shape), separators=(",", ":")),
        f"{prefix}_dtype": "float64_le",
        f"{prefix}_sha256": digest.hexdigest(),
        f"{prefix}_min": float(np.min(flat)),
        f"{prefix}_max": float(np.max(flat)),
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat, ddof=0)),
        f"{prefix}_nonzero_count": int(np.count_nonzero(flat)),
        f"{prefix}_l1_norm": float(np.linalg.norm(flat, ord=1)),
        f"{prefix}_l2_norm": float(np.linalg.norm(flat, ord=2)),
        f"{prefix}_max_abs": float(np.max(np.abs(flat))),
    }


def timing_matrix(representation: np.ndarray, decision_point: str) -> np.ndarray:
    """Construct the frozen 1/3/6-block longitudinal feature contract."""

    values = np.asarray(representation)
    if values.ndim != 3 or values.shape[1] != 4 or values.shape[2] <= 0:
        raise ValueError("representation must have shape [N,4,D] with D > 0")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise FloatingPointError("representation must be finite numeric values")
    decision_point = str(decision_point)
    if decision_point == "T0":
        matrix = values[:, 0]
    elif decision_point == "T0-T1":
        z0, z1 = values[:, 0], values[:, 1]
        matrix = np.concatenate((z0, z1, z1 - z0), axis=1)
    elif decision_point == "T0-T2":
        z0, z1, z2 = values[:, 0], values[:, 1], values[:, 2]
        matrix = np.concatenate((z0, z1, z2, z1 - z0, z2 - z1, z2 - z0), axis=1)
    else:
        raise ValueError(f"decision_point must be one of {DECISION_POINTS}")
    expected = values.shape[2] * TIMING_MULTIPLIERS[decision_point]
    if matrix.shape != (len(values), expected):
        raise AssertionError("longitudinal timing feature dimension drifted")
    return np.ascontiguousarray(matrix, dtype=np.float64)


@dataclass(frozen=True)
class ClinicalEncoder:
    """Train-fold-only age imputation and exact-arm one-hot encoding."""

    arms: tuple[str, ...]
    age_mean: float
    feature_names: tuple[str, ...]
    train_rows: int

    @classmethod
    def fit(cls, clinical: ClinicalTable, train_indices: Sequence[int]) -> "ClinicalEncoder":
        indices = np.asarray(train_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("clinical encoder requires nonempty one-dimensional train indices")
        if np.any(indices < 0) or np.any(indices >= len(clinical.patient_ids)):
            raise IndexError("clinical train index is out of bounds")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("clinical train indices are duplicated")
        arms = tuple(sorted(set(clinical.arm[indices].astype(str).tolist())))
        if not arms:
            raise ValueError("clinical outer-train has no exact treatment arms")
        ages = clinical.age[indices]
        finite = ages[np.isfinite(ages)]
        if finite.size == 0:
            raise ValueError("clinical outer-train has no finite ages")
        age_mean = float(np.mean(finite))
        names = ("HR", "HER2", "MammaPrint", "age", *(f"arm={arm}" for arm in arms))
        return cls(arms, age_mean, tuple(names), int(len(indices)))

    def transform(self, clinical: ClinicalTable) -> np.ndarray:
        unknown = sorted(set(clinical.arm.astype(str).tolist()).difference(self.arms))
        if unknown:
            raise ValueError(
                "validation/test contains an exact arm absent from outer-train: "
                f"{unknown}"
            )
        matrix = np.zeros((len(clinical.patient_ids), len(self.feature_names)), dtype=np.float64)
        matrix[:, 0] = clinical.hr
        matrix[:, 1] = clinical.her2
        matrix[:, 2] = clinical.mp
        matrix[:, 3] = np.where(np.isfinite(clinical.age), clinical.age, self.age_mean)
        arm_index = {arm: index for index, arm in enumerate(self.arms)}
        for row, arm in enumerate(clinical.arm.astype(str)):
            matrix[row, 4 + arm_index[arm]] = 1.0
        if not np.isfinite(matrix).all():
            raise FloatingPointError("encoded clinical features contain NaN/Inf")
        return matrix


@dataclass(frozen=True)
class SelectedLogistic:
    scaler: StandardScaler
    model: LogisticRegression
    penalty: str
    c_value: float
    validation_auroc: float
    validation_auprc: float
    threshold: float
    youden: float
    validation_sensitivity: float
    validation_specificity: float
    grid: tuple[dict[str, Any], ...]


def _validate_binary_selection_inputs(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(train_matrix, dtype=np.float64)
    validation_x = np.asarray(validation_matrix, dtype=np.float64)
    train_y = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    validation_y = np.asarray(validation_labels, dtype=np.int64).reshape(-1)
    if train_x.ndim != 2 or validation_x.ndim != 2 or train_x.shape[1] <= 0:
        raise ValueError("logistic matrices must be nonempty and two-dimensional")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("train/validation logistic feature dimensions differ")
    if len(train_x) != len(train_y) or len(validation_x) != len(validation_y):
        raise ValueError("logistic X/y row counts differ")
    if set(train_y.tolist()) != {0, 1} or set(validation_y.tolist()) != {0, 1}:
        raise ValueError("logistic train and validation must each contain both classes")
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise FloatingPointError("logistic train/validation features contain NaN/Inf")
    return train_x, train_y, validation_x, validation_y


def _youden_threshold(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, float, float, float]:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    if truth.shape != score.shape or set(truth.tolist()) != {0, 1}:
        raise ValueError("Youden threshold requires aligned two-class validation rows")
    candidates = np.unique(np.concatenate(([0.0, 1.0], score)))
    rows: list[tuple[float, float, float, float]] = []
    positives = int(np.count_nonzero(truth == 1))
    negatives = int(np.count_nonzero(truth == 0))
    for threshold in candidates:
        predicted = score >= threshold
        sensitivity = float(np.count_nonzero(predicted & (truth == 1)) / positives)
        specificity = float(np.count_nonzero(~predicted & (truth == 0)) / negatives)
        rows.append(
            (float(threshold), sensitivity + specificity - 1.0, sensitivity, specificity)
        )
    best = max(row[1] for row in rows)
    tied = [row for row in rows if row[1] >= best - 1e-12]
    return min(tied, key=lambda row: (abs(row[0] - 0.5), row[0]))


def select_logistic(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    *,
    penalties: Sequence[str] = PENALTIES,
    c_grid: Sequence[float] = C_GRID,
    random_state: int = SEED,
) -> SelectedLogistic:
    """Select penalty/C using validation only; there is no test argument."""

    train_x, train_y, validation_x, validation_y = _validate_binary_selection_inputs(
        train_matrix, train_labels, validation_matrix, validation_labels
    )
    penalty_values = tuple(dict.fromkeys(str(value).lower() for value in penalties))
    if not penalty_values or not set(penalty_values).issubset(PENALTIES):
        raise ValueError(f"penalties must be a nonempty subset of {PENALTIES}")
    c_values = tuple(sorted(set(float(value) for value in c_grid)))
    if not c_values or any(value <= 0 or not math.isfinite(value) for value in c_values):
        raise ValueError("C grid must contain finite positive values")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    candidates: list[
        tuple[float, float, float, int, str, LogisticRegression, np.ndarray]
    ] = []
    rows: list[dict[str, Any]] = []
    penalty_order = {value: index for index, value in enumerate(PENALTIES)}
    specifications = tuple(
        (penalty, c_value) for penalty in penalty_values for c_value in c_values
    )

    def fit_candidate(specification: tuple[str, float]):
        penalty, c_value = specification
        metric_free_progress(
            "candidate_started",
            family="binary",
            solver="liblinear",
            penalty=penalty,
            C=float(c_value),
            max_iter=LOGISTIC_MAX_ITER,
            tol=BINARY_LOGISTIC_TOL,
        )
        fit_started = time.perf_counter()
        model = LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=LOGISTIC_MAX_ITER,
            tol=BINARY_LOGISTIC_TOL,
            random_state=int(random_state),
            **_logistic_penalty_kwargs(penalty),
        )
        model, iterations = _fit_validated_logistic(
            model,
            train_scaled,
            train_y,
            solver="liblinear",
            penalty=penalty,
            c_value=c_value,
        )
        metric_free_progress(
            "candidate_completed",
            family="binary",
            solver="liblinear",
            penalty=penalty,
            C=float(c_value),
            n_iter=max(iterations),
            elapsed_seconds=float(time.perf_counter() - fit_started),
        )
        probability = np.asarray(
            model.predict_proba(validation_scaled)[:, 1], dtype=np.float64
        )
        auroc = float(roc_auc_score(validation_y, probability))
        auprc = float(average_precision_score(validation_y, probability))
        if not math.isfinite(auroc) or not math.isfinite(auprc):
            raise FloatingPointError("validation logistic metric is non-finite")
        row = {
            "penalty": penalty,
            "C": float(c_value),
            "val_auroc": auroc,
            "val_auprc": auprc,
            "solver": "liblinear",
            "tol": BINARY_LOGISTIC_TOL,
            "max_iter": LOGISTIC_MAX_ITER,
            "n_iter": max(iterations),
            "n_iter_json": json.dumps(iterations, separators=(",", ":")),
            "n_iter_contract": "integer_0_le_n_iter_lt_max_iter",
            "convergence_warning_observed": False,
            "converged_before_max_iter": True,
        }
        candidate = (
            auroc,
            auprc,
            c_value,
            penalty_order[penalty],
            penalty,
            model,
            probability,
        )
        return row, candidate

    for row, candidate in _ordered_selection_map(fit_candidate, specifications):
        rows.append(row)
        candidates.append(candidate)
    best_auroc = max(row[0] for row in candidates)
    tied_auc = [row for row in candidates if row[0] >= best_auroc - 1e-12]
    best_auprc = max(row[1] for row in tied_auc)
    tied_metrics = [row for row in tied_auc if row[1] >= best_auprc - 1e-12]
    chosen = min(tied_metrics, key=lambda row: (row[2], row[3]))
    auroc, auprc, c_value, _, penalty, model, probability = chosen
    threshold, youden, sensitivity, specificity = _youden_threshold(
        validation_y, probability
    )
    return SelectedLogistic(
        scaler,
        model,
        penalty,
        float(c_value),
        float(auroc),
        float(auprc),
        threshold,
        youden,
        sensitivity,
        specificity,
        tuple(rows),
    )


@dataclass
class _SingleUseProbabilityGuard:
    calls: int = 0

    def predict(self, model: LogisticRegression, matrix: np.ndarray) -> np.ndarray:
        if self.calls != 0:
            raise RuntimeError("outer-test predict_proba is single-use")
        self.calls += 1
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)


@dataclass
class _SingleUseMulticlassProbabilityGuard:
    calls: int = 0

    def predict(
        self, model: ExplicitOneVsRestLogistic, matrix: np.ndarray
    ) -> np.ndarray:
        if self.calls != 0:
            raise RuntimeError("outer-test multiclass predict_proba is single-use")
        self.calls += 1
        return np.asarray(model.predict_proba(matrix), dtype=np.float64)


@dataclass
class _SingleUseRegressionGuard:
    calls: int = 0

    def predict(self, model: Ridge, matrix: np.ndarray) -> np.ndarray:
        if self.calls != 0:
            raise RuntimeError("outer-test Ridge predict is single-use")
        self.calls += 1
        return np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class EvaluationResult:
    predictions: pd.DataFrame
    selections: pd.DataFrame


def _normalise_patient_inputs(
    patient_ids: Sequence[str], targets: Sequence[float], fold_manifest: FoldManifest
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(patient_ids, dtype=str)
    target = np.asarray(targets)
    if ids.ndim != 1 or target.shape != (len(ids),) or len(ids) == 0:
        raise ValueError("patient IDs and targets must align one-to-one")
    if len(set(ids.tolist())) != len(ids) or any(not value for value in ids):
        raise ValueError("patient IDs must be nonempty and unique")
    unknown = sorted(set(ids.tolist()).difference(fold_manifest.patient_ids.tolist()))
    if unknown:
        raise ValueError(f"evaluation contains {len(unknown)} patients outside the fold manifest")
    return ids, target


MatrixInput = np.ndarray | Mapping[int, np.ndarray] | Callable[[int], np.ndarray]


def _matrix_provider(
    matrices: MatrixInput, n_rows: int
) -> Callable[[int], np.ndarray]:
    def validate(value: np.ndarray, fold: int) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != n_rows or matrix.shape[1] <= 0:
            raise ValueError(f"fold {fold} feature matrix must have shape [N,P], P > 0")
        if not np.isfinite(matrix).all():
            raise FloatingPointError(f"fold {fold} feature matrix contains NaN/Inf")
        return np.ascontiguousarray(matrix)

    if callable(matrices):
        return lambda fold: validate(matrices(fold), fold)
    if isinstance(matrices, Mapping):
        if set(int(key) for key in matrices) != set(FOLDS):
            raise ValueError(f"fold-specific matrices must contain exactly folds {FOLDS}")
        raw = {int(key): value for key, value in matrices.items()}
        return lambda fold: validate(raw[fold], fold)
    common = validate(matrices, FOLDS[0])
    return lambda fold: common


def evaluate_binary_cv(
    *,
    patient_ids: Sequence[str],
    targets: Sequence[int],
    fold_manifest: FoldManifest,
    matrices: MatrixInput,
    target_name: str,
    model_name: str,
    spatial: str,
    timing: str,
    analysis_population: str,
    penalties: Sequence[str] = PENALTIES,
    c_grid: Sequence[float] = C_GRID,
    require_manifest_pcr_match: bool = False,
) -> EvaluationResult:
    """Run five outer folds and emit exactly one OOF test score per patient."""

    ids, raw_target = _normalise_patient_inputs(patient_ids, targets, fold_manifest)
    target = np.asarray(raw_target, dtype=np.int64)
    if not np.isin(target, (0, 1)).all():
        raise ValueError("binary target must contain only 0/1")
    if require_manifest_pcr_match and not np.array_equal(
        target, fold_manifest.labels_for(ids)
    ):
        raise ValueError("pCR target disagrees with the locked fold manifest")
    model = _safe_label(model_name, "model_name")
    target_label = _safe_label(target_name, "target_name")
    spatial_label = _safe_label(spatial, "spatial")
    population = _safe_label(analysis_population, "analysis_population")
    if timing not in DECISION_POINTS:
        raise ValueError(f"timing must be one of {DECISION_POINTS}")
    matrix_for_fold = _matrix_provider(matrices, len(ids))
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fold_matrix = matrix_for_fold(fold)
        roles = fold_manifest.roles(fold, ids)
        train_indices = np.flatnonzero(roles == "train")
        validation_indices = np.flatnonzero(roles == "val")
        test_indices = np.flatnonzero(roles == "test")
        fold_progress = {
            "task_family": "binary",
            "model": model,
            "spatial": spatial_label,
            "timing_or_endpoint": timing,
            "fold": int(fold),
            "feature_dim": int(fold_matrix.shape[1]),
        }
        metric_free_progress("fold_started", **fold_progress)
        selected = select_logistic(
            fold_matrix[train_indices],
            target[train_indices],
            fold_matrix[validation_indices],
            target[validation_indices],
            penalties=penalties,
            c_grid=c_grid,
            random_state=SEED + fold,
        )
        metric_free_progress("fold_selection_completed", **fold_progress)
        # Test slicing, transformation and prediction occur only after selection.
        test_scaled = selected.scaler.transform(fold_matrix[test_indices])
        guard = _SingleUseProbabilityGuard()
        probability = guard.predict(selected.model, test_scaled)
        if guard.calls != 1 or probability.shape != (len(test_indices),):
            raise AssertionError("outer-test probability call contract failed")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise FloatingPointError("outer-test probabilities are invalid")
        common = {
            "target": target_label,
            "model": model,
            "spatial": spatial_label,
            "timing": timing,
            "analysis_population": population,
            "split_seed": SEED,
            "fold_manifest_sha256": fold_manifest.sha256,
            "fold": fold,
            "feature_dim": int(fold_matrix.shape[1]),
        }
        prediction_rows.extend(
            {
                "patient_id": str(ids[index]),
                **common,
                "split": "test",
                "y_true": int(target[index]),
                "y_score": float(probability[row]),
                "predicted_label": int(probability[row] >= selected.threshold),
                "threshold": selected.threshold,
                "selected_penalty": selected.penalty,
                "selected_C": selected.c_value,
                "test_used_for_scaler": False,
                "test_used_for_hyperparameter_selection": False,
                "test_used_for_threshold_selection": False,
                "test_predict_proba_call_count": guard.calls,
            }
            for row, index in enumerate(test_indices)
        )
        selection_rows.append(
            {
                **common,
                "n_train": int(len(train_indices)),
                "n_val": int(len(validation_indices)),
                "n_test": int(len(test_indices)),
                "train_positive": int(target[train_indices].sum()),
                "val_positive": int(target[validation_indices].sum()),
                "test_positive": int(target[test_indices].sum()),
                "selected_penalty": selected.penalty,
                "selected_C": selected.c_value,
                "val_auroc": selected.validation_auroc,
                "val_auprc": selected.validation_auprc,
                "selected_threshold": selected.threshold,
                "val_youden": selected.youden,
                "val_sensitivity": selected.validation_sensitivity,
                "val_specificity": selected.validation_specificity,
                "grid_validation_metrics_json": json.dumps(selected.grid, separators=(",", ":")),
                "selection_rule": (
                    "max_validation_AUROC_then_AUPRC_then_smaller_C_then_l1_l2"
                ),
                "solver": "liblinear",
                "selection_parallel_workers": _selection_worker_count(
                    len(set(penalties)) * len(set(c_grid))
                ),
                "class_weight": "balanced",
                "random_state": SEED + fold,
                "max_iter": LOGISTIC_MAX_ITER,
                "tol": BINARY_LOGISTIC_TOL,
                "grid_max_n_iter": max(row["n_iter"] for row in selected.grid),
                "all_grid_candidates_converged_before_max_iter": all(
                    row["converged_before_max_iter"] for row in selected.grid
                ),
                "n_iter_contract": "integer_0_le_n_iter_lt_max_iter",
                "convergence_warning_observed": False,
                "convergence_gate": (
                    "integer_0_le_n_iter_lt_max_iter_and_no_convergence_warning_"
                    "before_validation"
                ),
                "scaler_train_rows": int(selected.scaler.n_samples_seen_),
                **_array_audit("scaler_mean", selected.scaler.mean_),
                **_array_audit("scaler_scale", selected.scaler.scale_),
                **_array_audit("coef", selected.model.coef_),
                **_array_audit("intercept", selected.model.intercept_),
                "test_used_for_scaler": False,
                "test_used_for_hyperparameter_selection": False,
                "test_used_for_threshold_selection": False,
                "test_predict_proba_call_count": guard.calls,
            }
        )
        metric_free_progress("fold_completed", **fold_progress)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    if len(predictions) != len(ids) or predictions.duplicated(["patient_id"]).any():
        raise ValueError("binary OOF predictions must cover every analysis patient exactly once")
    if set(predictions["patient_id"]) != set(ids.tolist()):
        raise ValueError("binary OOF patient coverage drifted")
    if len(selections) != len(FOLDS) or set(selections["fold"]) != set(FOLDS):
        raise ValueError("binary selection rows do not cover all five folds")
    return EvaluationResult(predictions, selections)


def _binary_calibration(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Fit logistic calibration ``y ~ intercept + slope*logit(p)`` by IRLS."""

    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    if len(y) < 3 or set(y.tolist()) != {0.0, 1.0} or np.ptp(logit) <= 1e-12:
        return math.nan, math.nan
    design = np.column_stack((np.ones(len(y), dtype=np.float64), logit))
    prevalence = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    beta = np.asarray([math.log(prevalence / (1.0 - prevalence)), 1.0], dtype=np.float64)
    for _ in range(100):
        linear = np.clip(design @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-8)
        information = design.T @ (weights[:, None] * design)
        information.flat[:: information.shape[0] + 1] += 1e-9
        score = design.T @ (y - fitted)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        step = np.clip(step, -5.0, 5.0)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    if not np.isfinite(beta).all():
        return math.nan, math.nan
    return float(beta[1]), float(beta[0])


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y.shape != score.shape or len(y) == 0 or set(y.tolist()) != {0, 1}:
        raise ValueError("binary metrics require aligned nonempty two-class inputs")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("binary scores must be finite probabilities in [0,1]")
    slope, intercept = _binary_calibration(y, score)
    ece = fixed_bin_ece(y, score, n_bins=ECE_BINS)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece_10bin": ece,
    }


def fixed_bin_ece(
    y_true: Sequence[int] | Sequence[bool],
    probability: Sequence[float],
    *,
    n_bins: int = ECE_BINS,
) -> float:
    """Equal-width fixed-bin expected calibration error.

    Bins are ``[0, .1), ..., [.9, 1]`` for the locked ten-bin setting. Empty
    bins contribute zero; nonempty bins are weighted by their patient count.
    """

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    if truth.shape != score.shape or len(truth) == 0 or not np.isin(truth, (0.0, 1.0)).all():
        raise ValueError("ECE requires aligned binary truth/probability inputs")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("ECE probabilities must be finite and in [0,1]")
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or int(n_bins) < 2:
        raise ValueError("ECE n_bins must be an integer >= 2")
    n_bins = int(n_bins)
    bin_index = np.minimum((score * n_bins).astype(np.int64), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        selected = bin_index == index
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(truth[selected])) - float(np.mean(score[selected]))
            )
    return float(error)


_BINARY_GROUP_COLUMNS = (
    "target",
    "model",
    "spatial",
    "timing",
    "analysis_population",
    "split_seed",
    "fold_manifest_sha256",
)
_BINARY_METRIC_NAMES = (
    "auroc",
    "auprc",
    "brier",
    "calibration_slope",
    "calibration_intercept",
    "ece_10bin",
)


def aggregate_binary_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return public pooled-OOF and mean-outer-fold binary metrics."""

    required = {"patient_id", "fold", "split", "y_true", "y_score", *_BINARY_GROUP_COLUMNS}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"binary predictions are missing columns: {missing}")
    if predictions.empty or set(predictions["split"].astype(str)) != {"test"}:
        raise ValueError("binary aggregation accepts test-only OOF predictions")
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(list(_BINARY_GROUP_COLUMNS), sort=True, dropna=False):
        if group.duplicated("patient_id").any() or set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"binary OOF fold/patient coverage drifted for {keys}")
        common = dict(zip(_BINARY_GROUP_COLUMNS, keys, strict=True))
        pooled = binary_metrics(group["y_true"], group["y_score"])
        rows.append(
            {
                **common,
                "aggregation": "pooled_oof",
                "n": int(len(group)),
                "positive": int(group["y_true"].sum()),
                "n_folds": len(FOLDS),
                "ece_bin_contract": "10_equal_width_bins_[0,1]",
                **pooled,
            }
        )
        fold_values = [
            binary_metrics(current["y_true"], current["y_score"])
            for _, current in group.groupby("fold", sort=True)
        ]
        rows.append(
            {
                **common,
                "aggregation": "outer_fold_macro",
                "n": int(len(group)),
                "positive": int(group["y_true"].sum()),
                "n_folds": len(fold_values),
                "ece_bin_contract": "10_equal_width_bins_[0,1]",
                **{
                    name: float(np.nanmean([value[name] for value in fold_values]))
                    for name in _BINARY_METRIC_NAMES
                },
            }
        )
    output = pd.DataFrame(rows)
    ensure_public_safe(output)
    return output


@dataclass(frozen=True)
class ExplicitOneVsRestLogistic:
    """Four explicit balanced binary estimators with normalized OVR probabilities."""

    classes_: np.ndarray
    estimators_: tuple[LogisticRegression, ...]

    def __post_init__(self) -> None:
        classes = np.asarray(self.classes_, dtype=str).reshape(-1).copy()
        estimators = tuple(self.estimators_)
        if len(classes) != 4 or len(set(classes.tolist())) != 4:
            raise ValueError("explicit one-vs-rest model requires exactly four classes")
        if len(estimators) != len(classes):
            raise ValueError("explicit one-vs-rest estimator/class count differs")
        feature_dims: set[int] = set()
        for estimator in estimators:
            if tuple(np.asarray(estimator.classes_, dtype=np.int64).tolist()) != (0, 1):
                raise ValueError("one-vs-rest binary estimator classes must be (0,1)")
            coefficient = np.asarray(estimator.coef_, dtype=np.float64)
            intercept = np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)
            if coefficient.ndim != 2 or coefficient.shape[0] != 1 or intercept.shape != (1,):
                raise ValueError("one-vs-rest binary estimator parameter shape drifted")
            if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
                raise FloatingPointError("one-vs-rest estimator parameters contain NaN/Inf")
            feature_dims.add(int(coefficient.shape[1]))
        if len(feature_dims) != 1:
            raise ValueError("one-vs-rest estimators disagree on feature dimension")
        classes.flags.writeable = False
        object.__setattr__(self, "classes_", classes)
        object.__setattr__(self, "estimators_", estimators)

    @property
    def coef_(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.vstack([estimator.coef_[0] for estimator in self.estimators_]),
            dtype=np.float64,
        )

    @property
    def intercept_(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(
                [estimator.intercept_[0] for estimator in self.estimators_],
                dtype=np.float64,
            )
        )

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != self.coef_.shape[1]:
            raise ValueError("one-vs-rest prediction matrix shape is invalid")
        if not np.isfinite(values).all():
            raise FloatingPointError("one-vs-rest prediction matrix contains NaN/Inf")
        raw_columns: list[np.ndarray] = []
        for estimator in self.estimators_:
            binary_probability = np.asarray(
                estimator.predict_proba(values), dtype=np.float64
            )
            if binary_probability.shape != (len(values), 2):
                raise ValueError("one-vs-rest binary probability shape drifted")
            raw_columns.append(binary_probability[:, 1])
        raw_probability = np.column_stack(raw_columns)
        if (
            not np.isfinite(raw_probability).all()
            or np.any(raw_probability < 0.0)
            or np.any(raw_probability > 1.0)
        ):
            raise FloatingPointError(
                "one-vs-rest positive-class sigmoid probabilities must be finite and in [0,1]"
            )
        normalizer = raw_probability.sum(axis=1, keepdims=True)
        if not np.isfinite(normalizer).all() or np.any(normalizer <= 0.0):
            raise FloatingPointError("one-vs-rest probability normalizer is invalid")
        probability = raw_probability / normalizer
        if (
            not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12)
        ):
            raise FloatingPointError("normalized one-vs-rest probabilities are invalid")
        return np.ascontiguousarray(probability, dtype=np.float64)


@dataclass(frozen=True)
class SelectedMulticlassLogistic:
    scaler: StandardScaler
    model: ExplicitOneVsRestLogistic
    penalty: str
    c_value: float
    validation_macro_ovr_auroc: float
    validation_macro_ovr_auprc: float
    grid: tuple[dict[str, Any], ...]


def multiclass_metrics(
    y_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=str).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64)
    class_order = tuple(str(value) for value in classes)
    if (
        not class_order
        or len(set(class_order)) != len(class_order)
        or probability.shape != (len(truth), len(class_order))
    ):
        raise ValueError("multiclass truth/probability/class dimensions are invalid")
    if set(truth.tolist()) != set(class_order):
        raise ValueError("multiclass metrics require every locked class")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("multiclass probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("multiclass probability rows must sum to one")
    one_hot = label_binarize(truth, classes=list(class_order)).astype(np.float64)
    if one_hot.shape != probability.shape:
        raise AssertionError("multiclass one-hot shape drifted")
    predicted_index = np.argmax(probability, axis=1)
    truth_index = np.asarray([class_order.index(value) for value in truth], dtype=np.int64)
    confidence = probability[np.arange(len(truth)), predicted_index]
    correct = predicted_index == truth_index
    return {
        "macro_ovr_auroc": float(
            roc_auc_score(
                truth,
                probability,
                labels=list(class_order),
                multi_class="ovr",
                average="macro",
            )
        ),
        "macro_ovr_auprc": float(
            average_precision_score(one_hot, probability, average="macro")
        ),
        "multiclass_brier": float(np.mean(np.sum((one_hot - probability) ** 2, axis=1))),
        "toplabel_ece_10bin": fixed_bin_ece(correct, confidence, n_bins=ECE_BINS),
        "accuracy": float(np.mean(correct)),
    }


def select_multiclass_logistic(
    train_matrix: np.ndarray,
    train_targets: Sequence[str],
    validation_matrix: np.ndarray,
    validation_targets: Sequence[str],
    *,
    classes: Sequence[str],
    penalties: Sequence[str] = PENALTIES,
    c_grid: Sequence[float] = C_GRID,
    random_state: int = SEED,
) -> SelectedMulticlassLogistic:
    """Validation-only explicit four-class OVR L1/L2 selection; test is absent."""

    train_x = np.asarray(train_matrix, dtype=np.float64)
    validation_x = np.asarray(validation_matrix, dtype=np.float64)
    train_y = np.asarray(train_targets, dtype=str).reshape(-1)
    validation_y = np.asarray(validation_targets, dtype=str).reshape(-1)
    class_order = tuple(str(value) for value in classes)
    if train_x.ndim != 2 or validation_x.ndim != 2 or train_x.shape[1] <= 0:
        raise ValueError("multiclass matrices must be nonempty and two-dimensional")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("multiclass train/validation feature dimensions differ")
    if len(train_x) != len(train_y) or len(validation_x) != len(validation_y):
        raise ValueError("multiclass X/y row counts differ")
    if len(class_order) != 4 or len(set(class_order)) != len(class_order):
        raise ValueError("multiclass classes must contain exactly four unique labels")
    if set(train_y.tolist()) != set(class_order) or set(validation_y.tolist()) != set(class_order):
        raise ValueError("multiclass train and validation must contain every locked class")
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise FloatingPointError("multiclass train/validation features contain NaN/Inf")
    penalty_values = tuple(dict.fromkeys(str(value).lower() for value in penalties))
    if not penalty_values or not set(penalty_values).issubset(PENALTIES):
        raise ValueError(f"penalties must be a nonempty subset of {PENALTIES}")
    c_values = tuple(sorted(set(float(value) for value in c_grid)))
    if not c_values or any(value <= 0 or not math.isfinite(value) for value in c_values):
        raise ValueError("C grid must contain finite positive values")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    candidates: list[
        tuple[float, float, float, int, str, ExplicitOneVsRestLogistic]
    ] = []
    rows: list[dict[str, Any]] = []
    penalty_order = {value: index for index, value in enumerate(PENALTIES)}
    model_class_order = tuple(sorted(class_order))
    specifications = tuple(
        (penalty, c_value) for penalty in penalty_values for c_value in c_values
    )

    def fit_candidate(specification: tuple[str, float]):
        penalty, c_value = specification
        metric_free_progress(
            "candidate_started",
            family="multiclass",
            solver="explicit_one_vs_rest_liblinear",
            penalty=penalty,
            C=float(c_value),
            max_iter=LOGISTIC_MAX_ITER,
            tol=MULTICLASS_LOGISTIC_TOL,
        )
        candidate_fit_started = time.perf_counter()
        estimators: list[LogisticRegression] = []
        n_iter_by_class: dict[str, tuple[int, ...]] = {}
        for estimator_index, estimator_class in enumerate(model_class_order):
            metric_free_progress(
                "estimator_started",
                family="multiclass",
                solver="liblinear",
                penalty=penalty,
                C=float(c_value),
                estimator_class=estimator_class,
                estimator_index=estimator_index,
                max_iter=LOGISTIC_MAX_ITER,
                tol=MULTICLASS_LOGISTIC_TOL,
            )
            estimator_fit_started = time.perf_counter()
            binary_target = (train_y == estimator_class).astype(np.int64)
            estimator = LogisticRegression(
                C=c_value,
                solver="liblinear",
                class_weight="balanced",
                max_iter=LOGISTIC_MAX_ITER,
                tol=MULTICLASS_LOGISTIC_TOL,
                random_state=int(random_state),
                **_logistic_penalty_kwargs(penalty),
            )
            estimator, iterations = _fit_validated_logistic(
                estimator,
                train_scaled,
                binary_target,
                solver="liblinear",
                penalty=penalty,
                c_value=c_value,
                estimator_class=estimator_class,
            )
            metric_free_progress(
                "estimator_completed",
                family="multiclass",
                solver="liblinear",
                penalty=penalty,
                C=float(c_value),
                estimator_class=estimator_class,
                estimator_index=estimator_index,
                n_iter=max(iterations),
                elapsed_seconds=float(time.perf_counter() - estimator_fit_started),
            )
            estimators.append(estimator)
            n_iter_by_class[estimator_class] = iterations
        model = ExplicitOneVsRestLogistic(
            np.asarray(model_class_order, dtype=str), tuple(estimators)
        )
        flat_iterations = tuple(
            iteration
            for estimator_class in model_class_order
            for iteration in n_iter_by_class[estimator_class]
        )
        metric_free_progress(
            "candidate_completed",
            family="multiclass",
            solver="explicit_one_vs_rest_liblinear",
            penalty=penalty,
            C=float(c_value),
            estimator_count=len(estimators),
            n_iter=max(flat_iterations),
            elapsed_seconds=float(time.perf_counter() - candidate_fit_started),
        )
        probability = model.predict_proba(validation_scaled)
        metrics = multiclass_metrics(validation_y, probability, model.classes_)
        auroc = metrics["macro_ovr_auroc"]
        auprc = metrics["macro_ovr_auprc"]
        row = {
            "penalty": penalty,
            "C": float(c_value),
            "val_macro_ovr_auroc": auroc,
            "val_macro_ovr_auprc": auprc,
            "solver": "explicit_one_vs_rest_liblinear",
            "underlying_solver": "liblinear",
            "multiclass_strategy": "four_balanced_binary_ovr_sigmoid_then_row_normalize",
            "tol": MULTICLASS_LOGISTIC_TOL,
            "max_iter": LOGISTIC_MAX_ITER,
            "estimator_count": len(estimators),
            "n_iter": max(flat_iterations),
            "n_iter_json": json.dumps(flat_iterations, separators=(",", ":")),
            "n_iter_by_class_json": json.dumps(
                n_iter_by_class, sort_keys=True, separators=(",", ":")
            ),
            "n_iter_contract": "integer_0_le_n_iter_lt_max_iter",
            "convergence_warning_observed": False,
            "converged_before_max_iter": True,
        }
        candidate = (auroc, auprc, c_value, penalty_order[penalty], penalty, model)
        return row, candidate

    for row, candidate in _ordered_selection_map(fit_candidate, specifications):
        rows.append(row)
        candidates.append(candidate)
    best_auroc = max(row[0] for row in candidates)
    tied_auc = [row for row in candidates if row[0] >= best_auroc - 1e-12]
    best_auprc = max(row[1] for row in tied_auc)
    tied_metrics = [row for row in tied_auc if row[1] >= best_auprc - 1e-12]
    auroc, auprc, c_value, _, penalty, model = min(
        tied_metrics, key=lambda row: (row[2], row[3])
    )
    return SelectedMulticlassLogistic(
        scaler,
        model,
        penalty,
        float(c_value),
        float(auroc),
        float(auprc),
        tuple(rows),
    )


def evaluate_multiclass_cv(
    *,
    patient_ids: Sequence[str],
    targets: Sequence[str],
    classes: Sequence[str],
    fold_manifest: FoldManifest,
    matrices: MatrixInput,
    target_name: str,
    model_name: str,
    spatial: str,
    timing: str,
    analysis_population: str,
    penalties: Sequence[str] = PENALTIES,
    c_grid: Sequence[float] = C_GRID,
) -> EvaluationResult:
    ids, raw_target = _normalise_patient_inputs(patient_ids, targets, fold_manifest)
    target = np.asarray(raw_target, dtype=str)
    class_order = tuple(str(value) for value in classes)
    if set(target.tolist()) != set(class_order):
        raise ValueError("multiclass target does not match the locked class vocabulary")
    labels = {
        "target": _safe_label(target_name, "target_name"),
        "model": _safe_label(model_name, "model_name"),
        "spatial": _safe_label(spatial, "spatial"),
        "timing": timing,
        "analysis_population": _safe_label(analysis_population, "analysis_population"),
        "split_seed": SEED,
        "fold_manifest_sha256": fold_manifest.sha256,
    }
    if timing not in DECISION_POINTS:
        raise ValueError(f"timing must be one of {DECISION_POINTS}")
    matrix_for_fold = _matrix_provider(matrices, len(ids))
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fold_matrix = matrix_for_fold(fold)
        roles = fold_manifest.roles(fold, ids)
        train_indices = np.flatnonzero(roles == "train")
        validation_indices = np.flatnonzero(roles == "val")
        test_indices = np.flatnonzero(roles == "test")
        fold_progress = {
            "task_family": "multiclass",
            "model": labels["model"],
            "spatial": labels["spatial"],
            "timing_or_endpoint": timing,
            "fold": int(fold),
            "feature_dim": int(fold_matrix.shape[1]),
        }
        metric_free_progress("fold_started", **fold_progress)
        selected = select_multiclass_logistic(
            fold_matrix[train_indices],
            target[train_indices],
            fold_matrix[validation_indices],
            target[validation_indices],
            classes=class_order,
            penalties=penalties,
            c_grid=c_grid,
            random_state=SEED + fold,
        )
        metric_free_progress("fold_selection_completed", **fold_progress)
        test_scaled = selected.scaler.transform(fold_matrix[test_indices])
        guard = _SingleUseMulticlassProbabilityGuard()
        probability = guard.predict(selected.model, test_scaled)
        model_classes = tuple(selected.model.classes_.astype(str))
        if guard.calls != 1 or probability.shape != (len(test_indices), len(class_order)):
            raise AssertionError("outer-test multiclass probability call contract failed")
        predicted = selected.model.classes_[np.argmax(probability, axis=1)].astype(str)
        common = {
            **labels,
            "fold": fold,
            "feature_dim": int(fold_matrix.shape[1]),
        }
        classes_json = json.dumps(model_classes, separators=(",", ":"))
        prediction_rows.extend(
            {
                "patient_id": str(ids[index]),
                **common,
                "split": "test",
                "y_true": str(target[index]),
                "y_pred": str(predicted[row]),
                "classes_json": classes_json,
                "probabilities_json": json.dumps(probability[row].tolist(), separators=(",", ":")),
                "selected_penalty": selected.penalty,
                "selected_C": selected.c_value,
                "test_predict_proba_call_count": guard.calls,
            }
            for row, index in enumerate(test_indices)
        )
        selection_rows.append(
            {
                **common,
                "classes_json": classes_json,
                "n_train": int(len(train_indices)),
                "n_val": int(len(validation_indices)),
                "n_test": int(len(test_indices)),
                "selected_penalty": selected.penalty,
                "selected_C": selected.c_value,
                "val_macro_ovr_auroc": selected.validation_macro_ovr_auroc,
                "val_macro_ovr_auprc": selected.validation_macro_ovr_auprc,
                "grid_validation_metrics_json": json.dumps(selected.grid, separators=(",", ":")),
                "selection_rule": (
                    "max_validation_macro_OVR_AUROC_then_AUPRC_then_smaller_C_then_l1_l2"
                ),
                "solver": "explicit_one_vs_rest_liblinear",
                "underlying_solver": "liblinear",
                "multiclass_strategy": "four_balanced_binary_ovr_sigmoid_then_row_normalize",
                "estimator_count_per_candidate": len(class_order),
                "selection_parallel_workers": _selection_worker_count(
                    len(set(penalties)) * len(set(c_grid))
                ),
                "class_weight": "balanced",
                "random_state": SEED + fold,
                "max_iter": LOGISTIC_MAX_ITER,
                "tol": MULTICLASS_LOGISTIC_TOL,
                "grid_max_n_iter": max(row["n_iter"] for row in selected.grid),
                "all_grid_candidates_converged_before_max_iter": all(
                    row["converged_before_max_iter"] for row in selected.grid
                ),
                "n_iter_contract": "integer_0_le_n_iter_lt_max_iter",
                "convergence_warning_observed": False,
                "convergence_gate": (
                    "every_ovr_estimator_integer_0_le_n_iter_lt_max_iter_and_no_"
                    "convergence_warning_before_any_validation_probability"
                ),
                "scaler_train_rows": int(selected.scaler.n_samples_seen_),
                **_array_audit("scaler_mean", selected.scaler.mean_),
                **_array_audit("scaler_scale", selected.scaler.scale_),
                **_array_audit("coef", selected.model.coef_),
                **_array_audit("intercept", selected.model.intercept_),
                "test_used_for_scaler": False,
                "test_used_for_hyperparameter_selection": False,
                "test_predict_proba_call_count": guard.calls,
            }
        )
        metric_free_progress("fold_completed", **fold_progress)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    if len(predictions) != len(ids) or predictions.duplicated("patient_id").any():
        raise ValueError("multiclass OOF predictions must cover each patient exactly once")
    return EvaluationResult(predictions, selections)


_MULTICLASS_METRIC_NAMES = (
    "macro_ovr_auroc",
    "macro_ovr_auprc",
    "multiclass_brier",
    "toplabel_ece_10bin",
    "accuracy",
)


def _multiclass_frame_metrics(frame: pd.DataFrame) -> dict[str, float]:
    class_payloads = {str(value) for value in frame["classes_json"]}
    if len(class_payloads) != 1:
        raise ValueError("multiclass predictions disagree on class order")
    classes = tuple(json.loads(next(iter(class_payloads))))
    probability = np.asarray(
        [json.loads(value) for value in frame["probabilities_json"]], dtype=np.float64
    )
    return multiclass_metrics(frame["y_true"].astype(str), probability, classes)


def aggregate_multiclass_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id", "fold", "split", "y_true", "classes_json", "probabilities_json",
        *_BINARY_GROUP_COLUMNS,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"multiclass predictions are missing columns: {missing}")
    if predictions.empty or set(predictions["split"].astype(str)) != {"test"}:
        raise ValueError("multiclass aggregation accepts test-only OOF predictions")
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(list(_BINARY_GROUP_COLUMNS), sort=True, dropna=False):
        if group.duplicated("patient_id").any() or set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"multiclass OOF fold/patient coverage drifted for {keys}")
        common = dict(zip(_BINARY_GROUP_COLUMNS, keys, strict=True))
        rows.append(
            {
                **common,
                "aggregation": "pooled_oof",
                "n": int(len(group)),
                "n_folds": len(FOLDS),
                "ece_bin_contract": "10_equal_width_bins_[0,1]_top_label",
                **_multiclass_frame_metrics(group),
            }
        )
        fold_values = [
            _multiclass_frame_metrics(current)
            for _, current in group.groupby("fold", sort=True)
        ]
        rows.append(
            {
                **common,
                "aggregation": "outer_fold_macro",
                "n": int(len(group)),
                "n_folds": len(fold_values),
                "ece_bin_contract": "10_equal_width_bins_[0,1]_top_label",
                **{
                    name: float(np.nanmean([value[name] for value in fold_values]))
                    for name in _MULTICLASS_METRIC_NAMES
                },
            }
        )
    output = pd.DataFrame(rows)
    ensure_public_safe(output)
    return output


@dataclass(frozen=True)
class SelectedRidge:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    model: Ridge
    alpha: float
    validation_mse: float
    grid: tuple[dict[str, Any], ...]
    target_transform: str
    train_transformed_min: float
    train_transformed_max: float
    validation_clipped_count: int


def _target_forward(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if transform == "identity":
        output = values
    elif transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p Ridge target contains a negative value")
        output = np.log1p(values)
    else:
        raise ValueError("target_transform must be identity or log1p")
    if not np.isfinite(output).all():
        raise FloatingPointError("transformed Ridge target contains NaN/Inf")
    return output


def _target_inverse(values: np.ndarray, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if transform == "identity":
        output = array
    elif transform == "log1p":
        with np.errstate(over="raise", invalid="raise"):
            output = np.expm1(array)
    else:
        raise ValueError("target_transform must be identity or log1p")
    if not np.isfinite(output).all():
        raise FloatingPointError("inverse-transformed Ridge prediction contains NaN/Inf")
    return output


def _bound_transformed_ridge_predictions(
    values: np.ndarray,
    *,
    target_transform: str,
    train_transformed_min: float,
    train_transformed_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen outer-train prediction-bound rule before inversion.

    Static FTV uses ``log1p`` and is bounded to the outer-train target range in
    that transformed domain.  Delta-FTV uses the identity transform and is not
    clipped.  Test targets are intentionally absent from this interface.
    """

    prediction = np.asarray(values, dtype=np.float64).reshape(-1)
    lower = float(train_transformed_min)
    upper = float(train_transformed_max)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("pre-clipping Ridge prediction contains NaN/Inf")
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise FloatingPointError("outer-train transformed Ridge bounds are invalid")
    if target_transform == "identity":
        return prediction.copy(), np.zeros(len(prediction), dtype=bool)
    if target_transform != "log1p":
        raise ValueError("target_transform must be identity or log1p")
    bounded = np.clip(prediction, lower, upper)
    if not np.isfinite(bounded).all():
        raise FloatingPointError("bounded transformed Ridge prediction contains NaN/Inf")
    return bounded, bounded != prediction


def _validated_ridge_iterations(model: Ridge, *, alpha: float) -> tuple[int, ...]:
    """Require a valid LSQR convergence record for every fitted alpha."""

    context = f"lsqr Ridge candidate alpha={float(alpha):g}"
    raw_max_iter = getattr(model, "max_iter", None)
    if (
        isinstance(raw_max_iter, (bool, np.bool_))
        or not isinstance(raw_max_iter, (int, np.integer))
        or int(raw_max_iter) <= 0
    ):
        raise RuntimeError(f"{context} has invalid max_iter={raw_max_iter!r}")
    raw_iterations = getattr(model, "n_iter_", None)
    try:
        iteration_array = np.asarray(raw_iterations)
    except Exception as error:
        raise RuntimeError(f"{context} has invalid n_iter_") from error
    if (
        iteration_array.size == 0
        or np.issubdtype(iteration_array.dtype, np.bool_)
        or not np.issubdtype(iteration_array.dtype, np.integer)
    ):
        raise RuntimeError(f"{context} has empty or invalid n_iter_={raw_iterations!r}")
    iterations = np.asarray(iteration_array, dtype=np.int64).reshape(-1)
    if np.any(iterations < 0):
        raise RuntimeError(f"{context} has negative n_iter_={iterations.tolist()}")
    max_iter = int(raw_max_iter)
    if np.any(iterations >= max_iter):
        raise RuntimeError(
            f"{context} did not converge strictly before max_iter={max_iter}: "
            f"n_iter_={iterations.tolist()}"
        )
    return tuple(int(value) for value in iterations)


def _fit_validated_ridge(
    model: Ridge,
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> tuple[Ridge, tuple[int, ...]]:
    """Fit one LSQR candidate and gate warning/iterations/parameters."""

    context = f"lsqr Ridge candidate alpha={float(alpha):g}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = model.fit(matrix, targets)
    except ConvergenceWarning as error:
        raise RuntimeError(f"{context} emitted ConvergenceWarning") from error
    iterations = _validated_ridge_iterations(fitted, alpha=alpha)
    for parameter_name in ("coef_", "intercept_"):
        try:
            parameter = np.asarray(
                getattr(fitted, parameter_name, None), dtype=np.float64
            )
        except Exception as error:
            raise RuntimeError(
                f"{context} has invalid fitted {parameter_name}"
            ) from error
        if parameter.size == 0 or not np.isfinite(parameter).all():
            raise RuntimeError(
                f"{context} has empty or non-finite fitted {parameter_name}"
            )
    return fitted, iterations


def select_ridge(
    train_matrix: np.ndarray,
    train_targets: np.ndarray,
    validation_matrix: np.ndarray,
    validation_targets: np.ndarray,
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    target_transform: str = "identity",
) -> SelectedRidge:
    """Select Ridge alpha by validation MSE; there is no test argument."""

    train_x = np.asarray(train_matrix, dtype=np.float64)
    validation_x = np.asarray(validation_matrix, dtype=np.float64)
    train_y_natural = np.asarray(train_targets, dtype=np.float64).reshape(-1)
    validation_y_natural = np.asarray(validation_targets, dtype=np.float64).reshape(-1)
    if train_x.ndim != 2 or validation_x.ndim != 2 or train_x.shape[1] <= 0:
        raise ValueError("Ridge matrices must be nonempty and two-dimensional")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("Ridge train/validation feature dimensions differ")
    if len(train_x) != len(train_y_natural) or len(validation_x) != len(validation_y_natural):
        raise ValueError("Ridge X/y row counts differ")
    if min(len(train_x), len(validation_x)) < 2:
        raise ValueError("Ridge train/validation partitions require at least two rows")
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise FloatingPointError("Ridge train/validation features contain NaN/Inf")
    train_y = _target_forward(train_y_natural, target_transform)
    validation_y = _target_forward(validation_y_natural, target_transform)
    alpha_values = tuple(sorted(set(float(value) for value in alphas)))
    if not alpha_values or any(value <= 0 or not math.isfinite(value) for value in alpha_values):
        raise ValueError("Ridge alpha grid must contain finite positive values")
    x_scaler = StandardScaler().fit(train_x)
    y_scaler = StandardScaler().fit(train_y[:, None])
    x_train = x_scaler.transform(train_x)
    x_validation = x_scaler.transform(validation_x)
    y_train = y_scaler.transform(train_y[:, None]).reshape(-1)
    y_validation = y_scaler.transform(validation_y[:, None]).reshape(-1)
    train_transformed_min = float(np.min(train_y))
    train_transformed_max = float(np.max(train_y))
    candidates: list[tuple[float, float, Ridge, int]] = []
    grid: list[dict[str, Any]] = []

    def fit_candidate(alpha: float):
        metric_free_progress(
            "candidate_started",
            family="ridge",
            solver="lsqr",
            alpha=float(alpha),
            max_iter=RIDGE_MAX_ITER,
            tol=RIDGE_TOL,
        )
        fit_started = time.perf_counter()
        model, iterations = _fit_validated_ridge(
            Ridge(
                alpha=alpha,
                fit_intercept=True,
                solver="lsqr",
                tol=RIDGE_TOL,
                max_iter=RIDGE_MAX_ITER,
            ),
            x_train,
            y_train,
            alpha=alpha,
        )
        validation_standardized_raw = np.asarray(
            model.predict(x_validation), dtype=np.float64
        ).reshape(-1)
        if not np.isfinite(validation_standardized_raw).all():
            raise FloatingPointError(
                "pre-clipping validation Ridge prediction contains NaN/Inf"
            )
        validation_transformed_raw = y_scaler.inverse_transform(
            validation_standardized_raw[:, None]
        ).reshape(-1)
        validation_transformed, validation_clipped = (
            _bound_transformed_ridge_predictions(
                validation_transformed_raw,
                target_transform=target_transform,
                train_transformed_min=train_transformed_min,
                train_transformed_max=train_transformed_max,
            )
        )
        validation_standardized = y_scaler.transform(
            validation_transformed[:, None]
        ).reshape(-1)
        mse = float(mean_squared_error(y_validation, validation_standardized))
        if not math.isfinite(mse):
            raise FloatingPointError("validation Ridge MSE is non-finite")
        clipped_count = int(np.count_nonzero(validation_clipped))
        elapsed = time.perf_counter() - fit_started
        metric_free_progress(
            "candidate_completed",
            family="ridge",
            solver="lsqr",
            alpha=float(alpha),
            elapsed_seconds=float(elapsed),
            n_iter=max(iterations),
        )
        return (
            {
                "alpha": float(alpha),
                "val_mse_standardized": mse,
                "solver": "lsqr",
                "tol": RIDGE_TOL,
                "max_iter": RIDGE_MAX_ITER,
                "n_iter": max(iterations),
                "n_iter_json": json.dumps(iterations, separators=(",", ":")),
                "n_iter_contract": "integer_0_le_n_iter_lt_max_iter",
                "convergence_warning_observed": False,
                "converged_before_max_iter": True,
                "validation_prediction_bound_policy": (
                    "outer_train_transformed_min_max"
                    if target_transform == "log1p"
                    else "none"
                ),
                "validation_predictions_clipped": clipped_count,
                "validation_clip_rate": clipped_count / len(validation_y),
            },
            (mse, alpha, model, clipped_count),
        )

    for row, candidate in _ordered_selection_map(fit_candidate, alpha_values):
        grid.append(row)
        candidates.append(candidate)
    best = min(row[0] for row in candidates)
    mse, alpha, model, validation_clipped_count = min(
        (row for row in candidates if row[0] <= best + 1e-12), key=lambda row: row[1]
    )
    return SelectedRidge(
        x_scaler,
        y_scaler,
        model,
        float(alpha),
        float(mse),
        tuple(grid),
        target_transform,
        train_transformed_min,
        train_transformed_max,
        validation_clipped_count,
    )


def evaluate_ridge_cv(
    *,
    patient_ids: Sequence[str],
    targets: Sequence[float],
    fold_manifest: FoldManifest,
    matrices: MatrixInput,
    target_name: str,
    model_name: str,
    spatial: str,
    task: str,
    endpoint: str,
    analysis_population: str,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    target_transform: str = "identity",
) -> EvaluationResult:
    """Run five validation-selected Ridge probes with single-use outer tests."""

    ids, raw_target = _normalise_patient_inputs(patient_ids, targets, fold_manifest)
    target = np.asarray(raw_target, dtype=np.float64)
    if not np.isfinite(target).all():
        raise FloatingPointError("Ridge target contains NaN/Inf")
    labels = {
        "target": _safe_label(target_name, "target_name"),
        "model": _safe_label(model_name, "model_name"),
        "spatial": _safe_label(spatial, "spatial"),
        "task": _safe_label(task, "task"),
        "endpoint": _safe_label(endpoint, "endpoint"),
        "analysis_population": _safe_label(analysis_population, "analysis_population"),
        "split_seed": SEED,
        "fold_manifest_sha256": fold_manifest.sha256,
    }
    matrix_for_fold = _matrix_provider(matrices, len(ids))
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fold_matrix = matrix_for_fold(fold)
        roles = fold_manifest.roles(fold, ids)
        train_indices = np.flatnonzero(roles == "train")
        validation_indices = np.flatnonzero(roles == "val")
        test_indices = np.flatnonzero(roles == "test")
        fold_progress = {
            "task_family": "ridge",
            "model": labels["model"],
            "spatial": labels["spatial"],
            "timing_or_endpoint": labels["endpoint"],
            "fold": int(fold),
            "feature_dim": int(fold_matrix.shape[1]),
        }
        metric_free_progress("fold_started", **fold_progress)
        selected = select_ridge(
            fold_matrix[train_indices],
            target[train_indices],
            fold_matrix[validation_indices],
            target[validation_indices],
            alphas=alphas,
            target_transform=target_transform,
        )
        metric_free_progress("fold_selection_completed", **fold_progress)
        test_scaled = selected.x_scaler.transform(fold_matrix[test_indices])
        guard = _SingleUseRegressionGuard()
        predicted_standardized = guard.predict(selected.model, test_scaled)
        if not np.isfinite(predicted_standardized).all():
            raise FloatingPointError(
                "pre-clipping outer-test Ridge prediction contains NaN/Inf"
            )
        predicted_transformed = selected.y_scaler.inverse_transform(
            predicted_standardized[:, None]
        ).reshape(-1)
        predicted_transformed, prediction_clipped = (
            _bound_transformed_ridge_predictions(
                predicted_transformed,
                target_transform=target_transform,
                train_transformed_min=selected.train_transformed_min,
                train_transformed_max=selected.train_transformed_max,
            )
        )
        predicted = _target_inverse(predicted_transformed, target_transform)
        if guard.calls != 1 or predicted.shape != (len(test_indices),):
            raise AssertionError("outer-test Ridge call contract failed")
        if not np.isfinite(predicted).all():
            raise FloatingPointError("outer-test Ridge prediction contains NaN/Inf")
        common = {
            **labels,
            "fold": fold,
            "feature_dim": int(fold_matrix.shape[1]),
        }
        train_mean = float(np.mean(target[train_indices]))
        prediction_rows.extend(
            {
                "patient_id": str(ids[index]),
                **common,
                "split": "test",
                "y_true": float(target[index]),
                "y_pred": float(predicted[row]),
                "train_mean_baseline": train_mean,
                "selected_alpha": selected.alpha,
                "target_transform": target_transform,
                "prediction_clipped_to_outer_train_bounds": bool(
                    prediction_clipped[row]
                ),
                "test_predict_call_count": guard.calls,
            }
            for row, index in enumerate(test_indices)
        )
        selection_rows.append(
            {
                **common,
                "n_train": int(len(train_indices)),
                "n_val": int(len(validation_indices)),
                "n_test": int(len(test_indices)),
                "selected_alpha": selected.alpha,
                "validation_mse_standardized": selected.validation_mse,
                "alpha_validation_metrics_json": json.dumps(selected.grid, separators=(",", ":")),
                "selection_parallel_workers": _selection_worker_count(len(set(alphas))),
                "target_transform": target_transform,
                "prediction_bound_policy": (
                    "outer_train_transformed_min_max"
                    if target_transform == "log1p"
                    else "none"
                ),
                "train_transformed_min": selected.train_transformed_min,
                "train_transformed_max": selected.train_transformed_max,
                "validation_predictions_clipped": selected.validation_clipped_count,
                "validation_clip_rate": selected.validation_clipped_count
                / len(validation_indices),
                "test_predictions_clipped": int(np.count_nonzero(prediction_clipped)),
                "test_clip_rate": float(np.mean(prediction_clipped)),
                "test_used_for_prediction_bounds": False,
                "solver": "lsqr",
                "tol": RIDGE_TOL,
                "max_iter": RIDGE_MAX_ITER,
                "selected_n_iter": max(
                    _validated_ridge_iterations(selected.model, alpha=selected.alpha)
                ),
                "grid_max_n_iter": max(row["n_iter"] for row in selected.grid),
                "n_iter_contract": "integer_0_le_n_iter_lt_max_iter",
                "convergence_warning_observed": False,
                "all_grid_candidates_converged_before_max_iter": all(
                    row["converged_before_max_iter"] for row in selected.grid
                ),
                "x_scaler_train_rows": int(selected.x_scaler.n_samples_seen_),
                "y_scaler_train_rows": int(selected.y_scaler.n_samples_seen_),
                **_array_audit("x_scaler_mean", selected.x_scaler.mean_),
                **_array_audit("x_scaler_scale", selected.x_scaler.scale_),
                "y_scaler_mean": float(selected.y_scaler.mean_[0]),
                "y_scaler_scale": float(selected.y_scaler.scale_[0]),
                **_array_audit("coef", selected.model.coef_),
                **_array_audit("intercept", selected.model.intercept_),
                "test_used_for_scaler": False,
                "test_used_for_alpha_selection": False,
                "test_predict_call_count": guard.calls,
            }
        )
        metric_free_progress("fold_completed", **fold_progress)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    if len(predictions) != len(ids) or predictions.duplicated("patient_id").any():
        raise ValueError("Ridge OOF predictions must cover every analysis patient exactly once")
    if set(predictions["patient_id"]) != set(ids.tolist()):
        raise ValueError("Ridge OOF patient coverage drifted")
    return EvaluationResult(predictions, selections)


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return math.nan
    first_rank = pd.Series(first).rank(method="average").to_numpy(dtype=np.float64)
    second_rank = pd.Series(second).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def continuous_metrics(
    y_true: Sequence[float], y_pred: Sequence[float], baseline: Sequence[float]
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    b0 = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or truth.shape != b0.shape or len(truth) == 0:
        raise ValueError("continuous metric inputs must be nonempty and aligned")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not np.isfinite(b0).all():
        raise FloatingPointError("continuous metric input contains NaN/Inf")
    variance = float(np.var(truth, ddof=0))
    if variance > 0:
        slope = float(np.mean((truth - np.mean(truth)) * (prediction - np.mean(prediction))) / variance)
        intercept = float(np.mean(prediction) - slope * np.mean(truth))
        pearson = float(np.corrcoef(truth, prediction)[0, 1]) if np.ptp(prediction) > 0 else math.nan
    else:
        slope = intercept = pearson = math.nan
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(truth, b0)))
    return {
        "spearman": _rank_correlation(truth, prediction),
        "pearson": pearson,
        "r2": float(r2_score(truth, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, prediction)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / b0_rmse if b0_rmse > 0 else math.nan,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "calibration_mean_bias": float(np.mean(prediction - truth)),
    }


_CONTINUOUS_GROUP_COLUMNS = (
    "target",
    "model",
    "spatial",
    "task",
    "endpoint",
    "analysis_population",
    "split_seed",
    "fold_manifest_sha256",
)
_CONTINUOUS_METRIC_NAMES = (
    "spearman",
    "pearson",
    "r2",
    "rmse",
    "mae",
    "b0_rmse",
    "rmse_gain_over_b0",
    "calibration_slope",
    "calibration_intercept",
    "calibration_mean_bias",
)


def aggregate_continuous_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id", "fold", "split", "y_true", "y_pred", "train_mean_baseline",
        *_CONTINUOUS_GROUP_COLUMNS,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"continuous predictions are missing columns: {missing}")
    if predictions.empty or set(predictions["split"].astype(str)) != {"test"}:
        raise ValueError("continuous aggregation accepts test-only OOF predictions")
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(list(_CONTINUOUS_GROUP_COLUMNS), sort=True, dropna=False):
        if group.duplicated("patient_id").any() or set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"continuous OOF fold/patient coverage drifted for {keys}")
        common = dict(zip(_CONTINUOUS_GROUP_COLUMNS, keys, strict=True))
        pooled = continuous_metrics(
            group["y_true"], group["y_pred"], group["train_mean_baseline"]
        )
        rows.append(
            {
                **common,
                "aggregation": "pooled_oof",
                "n": int(len(group)),
                "n_folds": len(FOLDS),
                **pooled,
            }
        )
        fold_values = [
            continuous_metrics(
                current["y_true"], current["y_pred"], current["train_mean_baseline"]
            )
            for _, current in group.groupby("fold", sort=True)
        ]
        rows.append(
            {
                **common,
                "aggregation": "outer_fold_macro",
                "n": int(len(group)),
                "n_folds": len(fold_values),
                **{
                    name: float(np.nanmean([value[name] for value in fold_values]))
                    for name in _CONTINUOUS_METRIC_NAMES
                },
            }
        )
    output = pd.DataFrame(rows)
    ensure_public_safe(output)
    return output


def ensure_public_safe(frame: pd.DataFrame) -> None:
    """Reject identifier/path-bearing columns and absolute path values."""

    forbidden_exact = {
        "patient_id",
        "clinical_patient_id",
        "raw_patient_id",
        "trial_id",
        "source_file",
        "source_path",
        "checkpoint_path",
    }
    offending = [
        column
        for column in frame.columns
        if str(column).lower() in forbidden_exact
        or str(column).lower().endswith("_path")
        or str(column).lower().endswith("_file")
    ]
    if offending:
        raise ValueError(f"public output contains identifier/path columns: {offending}")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        for raw in frame[column].dropna().astype(str):
            if raw.startswith("/") or _ABSOLUTE_WINDOWS_RE.match(raw):
                raise ValueError(f"public output contains an absolute path in {column}")


def _atomic_csv(frame: pd.DataFrame, path: Path, *, overwrite: bool, mode: int) -> None:
    if frame.empty:
        raise ValueError("refusing to write an empty CSV")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, mode)
        temporary.replace(path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def write_private_csv(
    frame: pd.DataFrame, path: str | Path, *, overwrite: bool = False
) -> Path:
    destination = Path(path)
    if not destination.name.endswith(".private.csv"):
        raise ValueError("patient-level/selection output must end in .private.csv")
    _atomic_csv(frame, destination, overwrite=overwrite, mode=0o600)
    return destination


def write_public_csv(
    frame: pd.DataFrame, path: str | Path, *, overwrite: bool = False
) -> Path:
    destination = Path(path)
    if destination.name.endswith(".private.csv"):
        raise ValueError("public CSV must not use the private filename suffix")
    ensure_public_safe(frame)
    _atomic_csv(frame, destination, overwrite=overwrite, mode=0o644)
    return destination


__all__ = [
    "BINARY_LOGISTIC_TOL",
    "C_GRID",
    "ClinicalEncoder",
    "DECISION_POINTS",
    "ECE_BINS",
    "EvaluationResult",
    "ExplicitOneVsRestLogistic",
    "LOGISTIC_MAX_ITER",
    "MULTICLASS_LOGISTIC_TOL",
    "PENALTIES",
    "RIDGE_ALPHAS",
    "SelectedLogistic",
    "SelectedMulticlassLogistic",
    "SelectedRidge",
    "TIMING_MULTIPLIERS",
    "TIMING_SCHEMAS",
    "aggregate_binary_predictions",
    "aggregate_continuous_predictions",
    "aggregate_multiclass_predictions",
    "binary_metrics",
    "configure_metric_free_progress",
    "continuous_metrics",
    "ensure_public_safe",
    "evaluate_binary_cv",
    "evaluate_multiclass_cv",
    "evaluate_ridge_cv",
    "select_logistic",
    "select_multiclass_logistic",
    "select_ridge",
    "timing_matrix",
    "fixed_bin_ece",
    "multiclass_metrics",
    "metric_free_progress",
    "write_private_csv",
    "write_public_csv",
]
