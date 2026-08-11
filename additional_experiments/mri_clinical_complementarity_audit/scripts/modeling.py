"""Fold-isolated linear models and paired uncertainty for the complementarity audit.

Every fitting entry point accepts only outer-train and outer-validation arrays.
Outer-test arrays are intentionally absent from those signatures.  The returned
objects retain the train-fitted scaler/model and can subsequently be applied to
held-out data exactly once by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize


SELECTION_ATOL = 1e-12


def _matrix(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
    expected_features: int | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric matrix") from error
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape [N,F] with N,F > 0; got {array.shape}"
        )
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError(f"{name} has {array.shape[0]} rows; expected {expected_rows}")
    if expected_features is not None and array.shape[1] != expected_features:
        raise ValueError(
            f"{name} has {array.shape[1]} features; expected {expected_features}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _vector(values: Any, *, name: str, expected_rows: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_rows is not None and array.size != expected_rows:
        raise ValueError(f"{name} has {array.size} rows; expected {expected_rows}")
    if pd.isna(array).any():
        raise ValueError(f"{name} contains missing values")
    return array


def _binary_labels(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
    require_both_classes: bool = True,
) -> np.ndarray:
    raw = _vector(values, name=name, expected_rows=expected_rows)
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain binary 0/1 labels") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only binary 0/1 labels")
    labels = numeric.astype(np.int64)
    if require_both_classes and set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def _multiclass_labels(
    values: Any, *, name: str, expected_rows: int | None = None
) -> np.ndarray:
    labels = _vector(values, name=name, expected_rows=expected_rows)
    for value in labels.tolist():
        try:
            hash(value)
        except TypeError as error:
            raise TypeError(f"{name} contains an unhashable class label") from error
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{name} contains an empty class label")
    return labels


def _probabilities(
    values: Any, *, name: str, expected_rows: int | None = None
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_rows is not None and array.size != expected_rows:
        raise ValueError(f"{name} has {array.size} rows; expected {expected_rows}")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain finite probabilities in [0,1]")
    return array


def _positive_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of positive numbers")
    try:
        raw = tuple(values)
        grid = tuple(sorted({float(value) for value in raw}))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an iterable of positive numbers") from error
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise TypeError(f"{name} may not contain booleans")
    if not grid or any(not np.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError(f"{name} must contain finite positive numbers")
    return grid


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    candidate_count: int
    tie_break: str = "closest_to_0.5_then_smaller_threshold"


def select_validation_balanced_threshold(
    labels: Any, probabilities: Any
) -> ThresholdSelection:
    """Select a binary threshold using validation labels only.

    Candidate thresholds are the observed validation probabilities plus 0, 0.5,
    and 1.  Ties within ``1e-12`` prefer the threshold closest to 0.5 and then
    the smaller threshold.
    """

    y = _binary_labels(labels, name="validation labels")
    probability = _probabilities(
        probabilities, name="validation probabilities", expected_rows=len(y)
    )
    candidates = np.unique(np.concatenate((probability, np.asarray([0.0, 0.5, 1.0]))))
    rows: list[tuple[float, float, float, float]] = []
    positives = y == 1
    negatives = ~positives
    for threshold in candidates:
        predicted = probability >= threshold
        sensitivity = float(np.mean(predicted[positives]))
        specificity = float(np.mean(~predicted[negatives]))
        balanced = 0.5 * (sensitivity + specificity)
        rows.append((float(threshold), balanced, sensitivity, specificity))
    best_value = max(row[1] for row in rows)
    eligible = [row for row in rows if row[1] >= best_value - SELECTION_ATOL]
    chosen = min(eligible, key=lambda row: (abs(row[0] - 0.5), row[0]))
    return ThresholdSelection(
        threshold=chosen[0],
        balanced_accuracy=chosen[1],
        sensitivity=chosen[2],
        specificity=chosen[3],
        candidate_count=len(rows),
    )


def binary_metrics(
    labels: Any, probabilities: Any, *, threshold: float = 0.5
) -> dict[str, int | float]:
    """Return strict binary probability and threshold metrics."""

    y = _binary_labels(labels, name="binary metric labels")
    probability = _probabilities(
        probabilities, name="binary metric probabilities", expected_rows=len(y)
    )
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and in [0,1]")
    predicted = (probability >= float(threshold)).astype(np.int64)
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier": float(np.mean(np.square(probability - y))),
        "threshold": float(threshold),
    }


@dataclass(frozen=True)
class BinaryCandidateScore:
    c_value: float
    validation_auroc: float


@dataclass(frozen=True)
class BinaryLogisticFit:
    scaler: StandardScaler
    model: LogisticRegression
    selected_c: float
    validation_auroc: float
    threshold_selection: ThresholdSelection
    grid_scores: tuple[BinaryCandidateScore, ...]
    feature_dim: int
    train_rows: int
    validation_rows: int

    def predict_proba(self, features: Any) -> np.ndarray:
        matrix = _matrix(
            features,
            name="binary prediction features",
            expected_features=self.feature_dim,
        )
        probability = np.asarray(
            self.model.predict_proba(self.scaler.transform(matrix))[:, 1],
            dtype=np.float64,
        )
        return _probabilities(
            probability, name="binary model probabilities", expected_rows=len(matrix)
        )

    def predict(self, features: Any) -> np.ndarray:
        return (
            self.predict_proba(features) >= self.threshold_selection.threshold
        ).astype(np.int64)


def fit_binary_logistic(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    c_grid: Iterable[float],
    *,
    class_weight: str | Mapping[int, float] | None = None,
    solver: str = "liblinear",
    max_iter: int = 10_000,
    random_state: int = 0,
) -> BinaryLogisticFit:
    """Fit train-only L2 logistic candidates and select C on validation AUROC."""

    train_x = _matrix(train_features, name="train features")
    validation_x = _matrix(
        validation_features,
        name="validation features",
        expected_features=train_x.shape[1],
    )
    train_y = _binary_labels(
        train_labels, name="train labels", expected_rows=len(train_x)
    )
    validation_y = _binary_labels(
        validation_labels,
        name="validation labels",
        expected_rows=len(validation_x),
    )
    grid = _positive_grid(c_grid, name="c_grid")
    max_iter = _positive_integer(max_iter, name="max_iter")
    if class_weight not in (None, "balanced") and not isinstance(class_weight, Mapping):
        raise ValueError("class_weight must be None, 'balanced', or a mapping")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    scored: list[tuple[BinaryCandidateScore, LogisticRegression]] = []
    for c_value in grid:
        candidate = LogisticRegression(
            penalty="l2",
            C=c_value,
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=int(random_state),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            candidate.fit(train_scaled, train_y)
        probability = candidate.predict_proba(validation_scaled)[:, 1]
        score = float(roc_auc_score(validation_y, probability))
        if not np.isfinite(score):
            raise RuntimeError(f"non-finite validation AUROC for C={c_value}")
        scored.append((BinaryCandidateScore(c_value, score), candidate))

    best_score = max(item[0].validation_auroc for item in scored)
    eligible = [
        item
        for item in scored
        if item[0].validation_auroc >= best_score - SELECTION_ATOL
    ]
    selected_score, selected_model = min(eligible, key=lambda item: item[0].c_value)
    validation_probability = selected_model.predict_proba(validation_scaled)[:, 1]
    threshold = select_validation_balanced_threshold(
        validation_y, validation_probability
    )
    return BinaryLogisticFit(
        scaler=scaler,
        model=selected_model,
        selected_c=selected_score.c_value,
        validation_auroc=selected_score.validation_auroc,
        threshold_selection=threshold,
        grid_scores=tuple(item[0] for item in scored),
        feature_dim=int(train_x.shape[1]),
        train_rows=int(len(train_x)),
        validation_rows=int(len(validation_x)),
    )


def _multiclass_probability_matrix(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
    expected_classes: int | None = None,
) -> np.ndarray:
    matrix = _matrix(
        values,
        name=name,
        expected_rows=expected_rows,
        expected_features=expected_classes,
    )
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(f"{name} contains values outside [0,1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return matrix


def multiclass_metrics(
    labels: Any, probabilities: Any, *, classes: Sequence[Any]
) -> dict[str, int | float]:
    """Return macro OVR AUROC/AUPRC and argmax balanced accuracy."""

    y = _multiclass_labels(labels, name="multiclass metric labels")
    class_array = _multiclass_labels(classes, name="classes")
    if len(set(class_array.tolist())) != len(class_array) or len(class_array) < 3:
        raise ValueError("classes must contain at least three unique labels")
    if set(y.tolist()) != set(class_array.tolist()):
        raise ValueError("multiclass metric labels must contain every declared class")
    probability = _multiclass_probability_matrix(
        probabilities,
        name="multiclass metric probabilities",
        expected_rows=len(y),
        expected_classes=len(class_array),
    )
    indicator = label_binarize(y, classes=class_array)
    prediction = class_array[np.argmax(probability, axis=1)]
    return {
        "n": int(len(y)),
        "n_classes": int(len(class_array)),
        "macro_ovr_auroc": float(
            roc_auc_score(
                y,
                probability,
                labels=class_array,
                multi_class="ovr",
                average="macro",
            )
        ),
        "macro_ovr_auprc": float(
            average_precision_score(indicator, probability, average="macro")
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }


@dataclass(frozen=True)
class MulticlassCandidateScore:
    c_value: float
    validation_macro_ovr_auroc: float
    validation_macro_ovr_auprc: float


@dataclass(frozen=True)
class MulticlassLogisticFit:
    scaler: StandardScaler
    model: LogisticRegression
    selected_c: float
    validation_macro_ovr_auroc: float
    validation_macro_ovr_auprc: float
    grid_scores: tuple[MulticlassCandidateScore, ...]
    classes: tuple[Any, ...]
    feature_dim: int
    train_rows: int
    validation_rows: int

    def predict_proba(self, features: Any) -> np.ndarray:
        matrix = _matrix(
            features,
            name="multiclass prediction features",
            expected_features=self.feature_dim,
        )
        probability = self.model.predict_proba(self.scaler.transform(matrix))
        if tuple(self.model.classes_.tolist()) != self.classes:
            raise RuntimeError("multiclass estimator class order changed")
        return _multiclass_probability_matrix(
            probability,
            name="multiclass model probabilities",
            expected_rows=len(matrix),
            expected_classes=len(self.classes),
        )

    def predict(self, features: Any) -> np.ndarray:
        probability = self.predict_proba(features)
        classes = np.asarray(self.classes)
        return classes[np.argmax(probability, axis=1)]


def fit_multiclass_logistic(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    c_grid: Iterable[float],
    *,
    solver: str = "liblinear",
    max_iter: int = 10_000,
    random_state: int = 0,
) -> MulticlassLogisticFit:
    """Fit balanced L2 OVR logistic candidates; select C by validation AUROC."""

    train_x = _matrix(train_features, name="train features")
    validation_x = _matrix(
        validation_features,
        name="validation features",
        expected_features=train_x.shape[1],
    )
    train_y = _multiclass_labels(
        train_labels, name="train labels", expected_rows=len(train_x)
    )
    validation_y = _multiclass_labels(
        validation_labels,
        name="validation labels",
        expected_rows=len(validation_x),
    )
    train_classes = set(train_y.tolist())
    validation_classes = set(validation_y.tolist())
    if len(train_classes) < 3:
        raise ValueError("multiclass train labels must contain at least three classes")
    if validation_classes != train_classes:
        raise ValueError("validation labels must contain exactly the train classes")
    grid = _positive_grid(c_grid, name="c_grid")
    max_iter = _positive_integer(max_iter, name="max_iter")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    scored: list[tuple[MulticlassCandidateScore, LogisticRegression]] = []
    for c_value in grid:
        candidate = LogisticRegression(
            penalty="l2",
            C=c_value,
            solver=solver,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=int(random_state),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            candidate.fit(train_scaled, train_y)
        probability = candidate.predict_proba(validation_scaled)
        metrics = multiclass_metrics(
            validation_y, probability, classes=candidate.classes_
        )
        score = MulticlassCandidateScore(
            c_value=c_value,
            validation_macro_ovr_auroc=float(metrics["macro_ovr_auroc"]),
            validation_macro_ovr_auprc=float(metrics["macro_ovr_auprc"]),
        )
        if not np.isfinite(score.validation_macro_ovr_auroc):
            raise RuntimeError(f"non-finite validation macro AUROC for C={c_value}")
        scored.append((score, candidate))

    best_score = max(item[0].validation_macro_ovr_auroc for item in scored)
    eligible = [
        item
        for item in scored
        if item[0].validation_macro_ovr_auroc >= best_score - SELECTION_ATOL
    ]
    selected_score, selected_model = min(eligible, key=lambda item: item[0].c_value)
    return MulticlassLogisticFit(
        scaler=scaler,
        model=selected_model,
        selected_c=selected_score.c_value,
        validation_macro_ovr_auroc=selected_score.validation_macro_ovr_auroc,
        validation_macro_ovr_auprc=selected_score.validation_macro_ovr_auprc,
        grid_scores=tuple(item[0] for item in scored),
        classes=tuple(selected_model.classes_.tolist()),
        feature_dim=int(train_x.shape[1]),
        train_rows=int(len(train_x)),
        validation_rows=int(len(validation_x)),
    )


@dataclass(frozen=True)
class FTVMRIResidualizer:
    """Train-fitted linear map from the available FTV prefix to MRI dimensions."""

    scaler: StandardScaler
    model: LinearRegression
    ftv_dim: int
    mri_dim: int
    train_rows: int

    def predict_ftv_associated_mri(self, ftv_features: Any) -> np.ndarray:
        ftv = _matrix(
            ftv_features,
            name="FTV residualization features",
            expected_features=self.ftv_dim,
        )
        prediction = np.asarray(
            self.model.predict(self.scaler.transform(ftv)), dtype=np.float64
        )
        return _matrix(
            prediction,
            name="FTV-associated MRI prediction",
            expected_rows=len(ftv),
            expected_features=self.mri_dim,
        )

    def transform(self, ftv_features: Any, mri_features: Any) -> np.ndarray:
        ftv = _matrix(
            ftv_features,
            name="FTV residualization features",
            expected_features=self.ftv_dim,
        )
        mri = _matrix(
            mri_features,
            name="MRI residualization features",
            expected_rows=len(ftv),
            expected_features=self.mri_dim,
        )
        return mri - self.predict_ftv_associated_mri(ftv)


def fit_ftv_mri_residualizer(
    train_ftv_features: Any, train_mri_features: Any
) -> FTVMRIResidualizer:
    """Fit ``FTV prefix -> each MRI dimension`` using outer-train only."""

    ftv = _matrix(train_ftv_features, name="train FTV features")
    mri = _matrix(train_mri_features, name="train MRI features", expected_rows=len(ftv))
    if len(ftv) < 2:
        raise ValueError("FTV residualizer requires at least two train patients")
    scaler = StandardScaler()
    ftv_scaled = scaler.fit_transform(ftv)
    model = LinearRegression(fit_intercept=True)
    model.fit(ftv_scaled, mri)
    return FTVMRIResidualizer(
        scaler=scaler,
        model=model,
        ftv_dim=int(ftv.shape[1]),
        mri_dim=int(mri.shape[1]),
        train_rows=int(len(ftv)),
    )


def clinical_probability_error(labels: Any, probabilities: Any) -> np.ndarray:
    """Return the auditable continuous clinical error ``y - p_clinical``."""

    y = _binary_labels(labels, name="clinical-error labels", require_both_classes=False)
    probability = _probabilities(
        probabilities, name="clinical probabilities", expected_rows=len(y)
    )
    return y.astype(np.float64) - probability


@dataclass(frozen=True)
class RidgeCandidateScore:
    alpha: float
    validation_mse: float


@dataclass(frozen=True)
class ClinicalErrorRidgeFit:
    scaler: StandardScaler
    model: Ridge
    selected_alpha: float
    validation_mse: float
    grid_scores: tuple[RidgeCandidateScore, ...]
    feature_dim: int
    train_rows: int
    validation_rows: int

    def predict(self, mri_features: Any) -> np.ndarray:
        matrix = _matrix(
            mri_features,
            name="clinical-error prediction features",
            expected_features=self.feature_dim,
        )
        prediction = np.asarray(
            self.model.predict(self.scaler.transform(matrix)), dtype=np.float64
        ).reshape(-1)
        if prediction.shape != (len(matrix),) or not np.isfinite(prediction).all():
            raise RuntimeError("clinical-error Ridge returned invalid predictions")
        return prediction


def _continuous_target(
    values: Any, *, name: str, expected_rows: int | None = None
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_rows is not None and array.size != expected_rows:
        raise ValueError(f"{name} has {array.size} rows; expected {expected_rows}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def fit_clinical_error_ridge(
    train_mri_features: Any,
    train_errors: Any,
    validation_mri_features: Any,
    validation_errors: Any,
    alphas: Iterable[float],
    *,
    solver: str = "lsqr",
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> ClinicalErrorRidgeFit:
    """Fit MRI -> clinical probability error; select alpha on validation MSE."""

    train_x = _matrix(train_mri_features, name="train MRI features")
    validation_x = _matrix(
        validation_mri_features,
        name="validation MRI features",
        expected_features=train_x.shape[1],
    )
    train_y = _continuous_target(
        train_errors, name="train clinical errors", expected_rows=len(train_x)
    )
    validation_y = _continuous_target(
        validation_errors,
        name="validation clinical errors",
        expected_rows=len(validation_x),
    )
    alpha_grid = _positive_grid(alphas, name="alphas")
    max_iter = _positive_integer(max_iter, name="max_iter")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    scored: list[tuple[RidgeCandidateScore, Ridge]] = []
    for alpha in alpha_grid:
        candidate = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver=solver,
            tol=float(tol),
            max_iter=max_iter,
        )
        candidate.fit(train_scaled, train_y)
        prediction = candidate.predict(validation_scaled)
        mse = float(mean_squared_error(validation_y, prediction))
        if not np.isfinite(mse):
            raise RuntimeError(f"non-finite validation MSE for alpha={alpha}")
        scored.append((RidgeCandidateScore(alpha, mse), candidate))
    best_mse = min(item[0].validation_mse for item in scored)
    eligible = [
        item for item in scored if item[0].validation_mse <= best_mse + SELECTION_ATOL
    ]
    selected_score, selected_model = min(eligible, key=lambda item: item[0].alpha)
    return ClinicalErrorRidgeFit(
        scaler=scaler,
        model=selected_model,
        selected_alpha=selected_score.alpha,
        validation_mse=selected_score.validation_mse,
        grid_scores=tuple(item[0] for item in scored),
        feature_dim=int(train_x.shape[1]),
        train_rows=int(len(train_x)),
        validation_rows=int(len(validation_x)),
    )


@dataclass(frozen=True)
class PairedBootstrapResult:
    summary: pd.DataFrame
    draws: pd.DataFrame
    n_patients: int
    fold_sizes: dict[int, int]
    bootstrap_unit: str = "patient_within_outer_fold"


def _prediction_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    patient_col: str,
    fold_col: str,
    label_col: str,
    probability_col: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    required = {patient_col, fold_col, label_col, probability_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    output = frame.loc[:, [patient_col, fold_col, label_col, probability_col]].copy()
    if output.empty:
        raise ValueError(f"{name} must not be empty")
    if output[patient_col].isna().any():
        raise ValueError(f"{name} contains missing patient IDs")
    output[patient_col] = output[patient_col].astype(str).str.strip()
    if output[patient_col].eq("").any():
        raise ValueError(f"{name} contains empty patient IDs")
    if output[patient_col].duplicated().any():
        raise ValueError(f"{name} must contain exactly one row per patient")
    try:
        fold_numeric = pd.to_numeric(output[fold_col], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} fold IDs must be integers") from error
    if (
        not np.isfinite(fold_numeric).all()
        or not np.equal(fold_numeric, np.floor(fold_numeric)).all()
        or np.any(fold_numeric < 0)
    ):
        raise ValueError(f"{name} fold IDs must be finite non-negative integers")
    output[fold_col] = fold_numeric.astype(np.int64)
    output[label_col] = _binary_labels(
        output[label_col].to_numpy(), name=f"{name} labels", expected_rows=len(output)
    )
    output[probability_col] = _probabilities(
        output[probability_col].to_numpy(),
        name=f"{name} probabilities",
        expected_rows=len(output),
    )
    return output


def paired_fold_stratified_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    fold_col: str = "fold",
    label_col: str = "y_true",
    probability_col: str = "predicted_probability",
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 260_811,
) -> PairedBootstrapResult:
    """Paired patient bootstrap with sampling performed separately within folds.

    AUROC/AUPRC improvements are ``comparison - reference``.  Brier improvement
    is ``reference - comparison``, so a positive number favors the comparison for
    every reported metric.  Percentile intervals omit single-class AUROC/AUPRC
    draws while retaining their valid Brier draws.
    """

    n_bootstrap = _positive_integer(n_bootstrap, name="n_bootstrap")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    reference_frame = _prediction_frame(
        reference,
        name="reference",
        patient_col=patient_col,
        fold_col=fold_col,
        label_col=label_col,
        probability_col=probability_col,
    ).rename(
        columns={
            label_col: "reference_label",
            probability_col: "reference_probability",
        }
    )
    comparison_frame = _prediction_frame(
        comparison,
        name="comparison",
        patient_col=patient_col,
        fold_col=fold_col,
        label_col=label_col,
        probability_col=probability_col,
    ).rename(
        columns={
            label_col: "comparison_label",
            probability_col: "comparison_probability",
        }
    )
    reference_ids = set(reference_frame[patient_col])
    comparison_ids = set(comparison_frame[patient_col])
    if reference_ids != comparison_ids:
        raise ValueError("reference and comparison patient sets must match exactly")
    paired = reference_frame.merge(
        comparison_frame,
        on=[patient_col, fold_col],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("reference and comparison fold assignments must match exactly")
    paired = paired.drop(columns="_merge").sort_values(
        [fold_col, patient_col], kind="mergesort"
    )
    if not np.array_equal(
        paired["reference_label"].to_numpy(),
        paired["comparison_label"].to_numpy(),
    ):
        raise ValueError("reference and comparison labels disagree")

    labels = paired["reference_label"].to_numpy(dtype=np.int64)
    reference_probability = paired["reference_probability"].to_numpy(dtype=np.float64)
    comparison_probability = paired["comparison_probability"].to_numpy(dtype=np.float64)
    reference_point = binary_metrics(labels, reference_probability)
    comparison_point = binary_metrics(labels, comparison_probability)

    rng = np.random.default_rng(int(seed))
    fold_sizes: dict[int, int] = {}
    sampled_blocks: list[np.ndarray] = []
    fold_values = paired[fold_col].to_numpy(dtype=np.int64)
    for fold in sorted(np.unique(fold_values)):
        positions = np.flatnonzero(fold_values == fold)
        if positions.size == 0:
            raise AssertionError("empty fold block")
        fold_sizes[int(fold)] = int(positions.size)
        sampled_blocks.append(
            rng.choice(positions, size=(n_bootstrap, positions.size), replace=True)
        )
    sampled_indices = np.concatenate(sampled_blocks, axis=1)
    draw_rows: list[dict[str, float | int]] = []
    for bootstrap_index, indices in enumerate(sampled_indices):
        sampled_y = labels[indices]
        sampled_reference = reference_probability[indices]
        sampled_comparison = comparison_probability[indices]
        brier_improvement = float(
            np.mean(np.square(sampled_reference - sampled_y))
            - np.mean(np.square(sampled_comparison - sampled_y))
        )
        auroc_improvement = math.nan
        auprc_improvement = math.nan
        if set(np.unique(sampled_y)) == {0, 1}:
            auroc_improvement = float(
                roc_auc_score(sampled_y, sampled_comparison)
                - roc_auc_score(sampled_y, sampled_reference)
            )
            auprc_improvement = float(
                average_precision_score(sampled_y, sampled_comparison)
                - average_precision_score(sampled_y, sampled_reference)
            )
        draw_rows.append(
            {
                "bootstrap_index": bootstrap_index,
                "auroc_improvement": auroc_improvement,
                "auprc_improvement": auprc_improvement,
                "brier_improvement": brier_improvement,
            }
        )
    draws = pd.DataFrame(draw_rows)
    alpha = 1.0 - float(confidence_level)
    specifications = (
        (
            "auroc",
            float(reference_point["auroc"]),
            float(comparison_point["auroc"]),
            "auroc_improvement",
            "comparison - reference",
        ),
        (
            "auprc",
            float(reference_point["auprc"]),
            float(comparison_point["auprc"]),
            "auprc_improvement",
            "comparison - reference",
        ),
        (
            "brier",
            float(reference_point["brier"]),
            float(comparison_point["brier"]),
            "brier_improvement",
            "reference - comparison (lower Brier is better)",
        ),
    )
    summary_rows: list[dict[str, Any]] = []
    for (
        metric,
        reference_value,
        comparison_value,
        draw_column,
        orientation,
    ) in specifications:
        improvement = (
            reference_value - comparison_value
            if metric == "brier"
            else comparison_value - reference_value
        )
        distribution = draws[draw_column].to_numpy(dtype=np.float64)
        finite = distribution[np.isfinite(distribution)]
        lower = float(np.quantile(finite, alpha / 2.0)) if finite.size else math.nan
        upper = (
            float(np.quantile(finite, 1.0 - alpha / 2.0)) if finite.size else math.nan
        )
        summary_rows.append(
            {
                "metric": metric,
                "reference": reference_value,
                "comparison": comparison_value,
                "improvement": improvement,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": float(confidence_level),
                "n_patients": int(len(paired)),
                "n_folds": int(len(fold_sizes)),
                "n_bootstrap": n_bootstrap,
                "n_valid_bootstrap": int(finite.size),
                "bootstrap_unit": "patient_within_outer_fold",
                "ci_method": "percentile",
                "orientation": orientation,
                "seed": int(seed),
            }
        )
    return PairedBootstrapResult(
        summary=pd.DataFrame(summary_rows),
        draws=draws,
        n_patients=int(len(paired)),
        fold_sizes=fold_sizes,
    )


__all__ = [
    "BinaryCandidateScore",
    "BinaryLogisticFit",
    "ClinicalErrorRidgeFit",
    "FTVMRIResidualizer",
    "MulticlassCandidateScore",
    "MulticlassLogisticFit",
    "PairedBootstrapResult",
    "RidgeCandidateScore",
    "ThresholdSelection",
    "binary_metrics",
    "clinical_probability_error",
    "fit_binary_logistic",
    "fit_clinical_error_ridge",
    "fit_ftv_mri_residualizer",
    "fit_multiclass_logistic",
    "multiclass_metrics",
    "paired_fold_stratified_bootstrap",
    "select_validation_balanced_threshold",
]
