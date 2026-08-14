"""Fold-isolated probes and patient-paired uncertainty for Goal F."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize


def _matrix(value: Any, name: str, rows: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must be nonempty [N,F]")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{name} row count drifted")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _binary(value: Any, name: str, *, both: bool = True) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or not result.size or pd.isna(result).any():
        raise ValueError(f"{name} must be a nonempty label vector")
    numeric = result.astype(np.float64)
    if not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain 0/1")
    output = numeric.astype(np.int64)
    if both and set(output) != {0, 1}:
        raise ValueError(f"{name} must contain both classes")
    return output


def _grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    output = tuple(sorted({float(value) for value in values}))
    if not output or any(not math.isfinite(value) or value <= 0 for value in output):
        raise ValueError(f"{name} must be finite and positive")
    return output


def binary_metrics(labels: Any, probability: Any) -> dict[str, float | int]:
    y = _binary(labels, "labels")
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probability contract drifted")
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean(np.square(y - p))),
    }


def regression_metrics(labels: Any, prediction: Any) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or len(y) < 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("regression metric contract drifted")
    rho = float(spearmanr(y, p).statistic) if np.ptp(p) > 0 and np.ptp(y) > 0 else 0.0
    return {
        "n": int(len(y)),
        "spearman": rho,
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(np.mean(np.abs(y - p))),
        "r2": float(1.0 - np.sum(np.square(y - p)) / np.sum(np.square(y - y.mean())))
        if np.ptp(y) > 0
        else 0.0,
    }


def multiclass_metrics(labels: Any, probability: Any, classes: Sequence[str]) -> dict[str, float | int]:
    y = np.asarray(labels).astype(str)
    p = np.asarray(probability, dtype=np.float64)
    classes_array = np.asarray(classes).astype(str)
    if y.ndim != 1 or p.shape != (len(y), len(classes_array)) or not np.isfinite(p).all():
        raise ValueError("multiclass metric contract drifted")
    indicator = label_binarize(y, classes=classes_array)
    return {
        "n": int(len(y)),
        "auroc_macro_ovr": float(roc_auc_score(indicator, p, average="macro", multi_class="ovr")),
        "accuracy": float(np.mean(classes_array[np.argmax(p, axis=1)] == y)),
    }


@dataclass(frozen=True)
class BinaryFit:
    scaler: StandardScaler
    model: LogisticRegression
    selected_c: float
    validation_auroc: float
    feature_dim: int

    def predict(self, matrix: Any) -> np.ndarray:
        x = _matrix(matrix, "binary prediction", None)
        if x.shape[1] != self.feature_dim:
            raise ValueError("binary prediction dimension drifted")
        return self.model.predict_proba(self.scaler.transform(x))[:, 1]


def fit_binary_logistic(
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    c_grid: Iterable[float],
    *,
    class_weight: str | Mapping[int, float] | None = None,
    random_state: int = 0,
    solver: str = "liblinear",
) -> BinaryFit:
    x0 = _matrix(x_train, "train features")
    x1 = _matrix(x_val, "validation features")
    if x0.shape[1] != x1.shape[1]:
        raise ValueError("train/validation feature dimension differs")
    y0, y1 = _binary(y_train, "train labels"), _binary(y_val, "validation labels")
    if len(y0) != len(x0) or len(y1) != len(x1):
        raise ValueError("binary label row count differs")
    if solver != "liblinear":
        raise ValueError("binary logistic solver must be the frozen liblinear contract")
    scaler = StandardScaler().fit(x0)
    tx0, tx1 = scaler.transform(x0), scaler.transform(x1)
    candidates: list[tuple[float, float, LogisticRegression]] = []
    for c_value in _grid(c_grid, "C grid"):
        model = LogisticRegression(
            C=c_value,
            solver=solver,
            max_iter=10000,
            class_weight=class_weight,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(tx0, y0)
        score = float(roc_auc_score(y1, model.predict_proba(tx1)[:, 1]))
        candidates.append((score, c_value, model))
    best_score = max(row[0] for row in candidates)
    score, c_value, model = min(
        (row for row in candidates if row[0] >= best_score - 1e-12), key=lambda row: row[1]
    )
    return BinaryFit(scaler, model, c_value, score, x0.shape[1])


@dataclass(frozen=True)
class MulticlassFit:
    scaler: StandardScaler
    model: LogisticRegression
    selected_c: float
    validation_auroc: float
    feature_dim: int
    classes: tuple[str, ...]

    def predict(self, matrix: Any) -> np.ndarray:
        x = _matrix(matrix, "multiclass prediction")
        if x.shape[1] != self.feature_dim:
            raise ValueError("multiclass prediction dimension drifted")
        raw = self.model.predict_proba(self.scaler.transform(x))
        lookup = {str(value): index for index, value in enumerate(self.model.classes_)}
        return np.column_stack([raw[:, lookup[value]] for value in self.classes])


def fit_multiclass_logistic(
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    c_grid: Iterable[float],
    *,
    random_state: int = 0,
    solver: str = "lbfgs",
    expected_classes: Sequence[str] | None = None,
) -> MulticlassFit:
    x0, x1 = _matrix(x_train, "train features"), _matrix(x_val, "validation features")
    y0, y1 = np.asarray(y_train).astype(str), np.asarray(y_val).astype(str)
    if x0.shape[1] != x1.shape[1] or len(y0) != len(x0) or len(y1) != len(x1):
        raise ValueError("multiclass train/validation contract drifted")
    if solver != "lbfgs":
        raise ValueError("multiclass logistic solver must be the frozen lbfgs contract")
    classes = tuple(
        sorted(set(y0))
        if expected_classes is None
        else tuple(str(value) for value in expected_classes)
    )
    if len(classes) < 3 or set(y0) != set(classes) or set(y1) != set(classes):
        raise ValueError("multiclass train/validation split lacks exact required classes")
    scaler = StandardScaler().fit(x0)
    tx0, tx1 = scaler.transform(x0), scaler.transform(x1)
    candidates: list[tuple[float, float, LogisticRegression]] = []
    for c_value in _grid(c_grid, "C grid"):
        model = LogisticRegression(
            C=c_value,
            solver=solver,
            max_iter=10000,
            class_weight="balanced",
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(tx0, y0)
        raw = model.predict_proba(tx1)
        lookup = {str(value): index for index, value in enumerate(model.classes_)}
        probability = np.column_stack([raw[:, lookup[value]] for value in classes])
        score = float(
            roc_auc_score(
                label_binarize(y1, classes=np.asarray(classes)),
                probability,
                average="macro",
                multi_class="ovr",
            )
        )
        candidates.append((score, c_value, model))
    best_score = max(row[0] for row in candidates)
    score, c_value, model = min(
        (row for row in candidates if row[0] >= best_score - 1e-12), key=lambda row: row[1]
    )
    return MulticlassFit(scaler, model, c_value, score, x0.shape[1], classes)


@dataclass(frozen=True)
class RidgeFit:
    x_scaler: StandardScaler
    y_mean: float
    y_scale: float
    model: Ridge
    selected_alpha: float
    validation_mse: float
    feature_dim: int

    def predict(self, matrix: Any) -> np.ndarray:
        x = _matrix(matrix, "ridge prediction")
        if x.shape[1] != self.feature_dim:
            raise ValueError("ridge prediction dimension drifted")
        standardized = self.model.predict(self.x_scaler.transform(x))
        return np.asarray(standardized, dtype=np.float64) * self.y_scale + self.y_mean


def fit_ridge(
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    alphas: Iterable[float],
) -> RidgeFit:
    x0, x1 = _matrix(x_train, "train features"), _matrix(x_val, "validation features")
    y0, y1 = np.asarray(y_train, dtype=np.float64).reshape(-1), np.asarray(y_val, dtype=np.float64).reshape(-1)
    if x0.shape[1] != x1.shape[1] or len(y0) != len(x0) or len(y1) != len(x1):
        raise ValueError("ridge train/validation contract drifted")
    if not np.isfinite(y0).all() or not np.isfinite(y1).all():
        raise ValueError("ridge targets contain non-finite values")
    scaler = StandardScaler().fit(x0)
    mean, scale = float(y0.mean()), float(y0.std(ddof=0))
    scale = scale if scale > 0 else 1.0
    ty0, ty1 = (y0 - mean) / scale, (y1 - mean) / scale
    tx0, tx1 = scaler.transform(x0), scaler.transform(x1)
    candidates: list[tuple[float, float, Ridge]] = []
    for alpha in _grid(alphas, "ridge alphas"):
        model = Ridge(alpha=alpha).fit(tx0, ty0)
        mse = float(mean_squared_error(ty1, model.predict(tx1)))
        candidates.append((mse, alpha, model))
    best_mse = min(row[0] for row in candidates)
    mse, alpha, model = min(
        (row for row in candidates if row[0] <= best_mse + 1e-12), key=lambda row: row[1]
    )
    return RidgeFit(scaler, mean, scale, model, alpha, mse, x0.shape[1])


@dataclass
class ClinicalEncoder:
    numeric_columns: tuple[str, ...] = (
        "label_hr",
        "label_her2",
        "label_mp",
        "age_at_screening",
    )
    categorical_columns: tuple[str, ...] = ("arm",)
    medians: dict[str, float] | None = None
    categories: dict[str, tuple[str, ...]] | None = None

    def fit(self, frame: pd.DataFrame) -> "ClinicalEncoder":
        self.medians = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").astype(float)
            median = float(values.median())
            if not math.isfinite(median):
                raise ValueError(f"clinical training column {column} has no finite median")
            self.medians[column] = median
        self.categories = {
            column: tuple(sorted(set(frame[column].astype(str))))
            for column in self.categorical_columns
        }
        if any(not values for values in self.categories.values()):
            raise ValueError("clinical training categorical vocabulary is empty")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.categories is None:
            raise RuntimeError("clinical encoder is not fitted")
        blocks: list[np.ndarray] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").astype(float)
            blocks.append(values.fillna(self.medians[column]).to_numpy()[:, None])
        for column in self.categorical_columns:
            values = frame[column].astype(str).to_numpy()
            blocks.append(
                np.column_stack([values == category for category in self.categories[column]]).astype(float)
            )
        return np.concatenate(blocks, axis=1).astype(np.float64)


@dataclass(frozen=True)
class LinearResidualizer:
    clinical_scaler: StandardScaler
    model: Ridge
    output_dim: int

    def transform(self, clinical: Any, phenotype: Any) -> np.ndarray:
        c, z = _matrix(clinical, "residualizer clinical"), _matrix(phenotype, "residualizer phenotype")
        if z.shape[1] != self.output_dim or len(c) != len(z):
            raise ValueError("linear residualizer dimension drifted")
        return z - self.model.predict(self.clinical_scaler.transform(c))


def fit_linear_residualizer(clinical_train: Any, phenotype_train: Any) -> LinearResidualizer:
    c, z = _matrix(clinical_train, "clinical train"), _matrix(phenotype_train, "phenotype train")
    if len(c) != len(z):
        raise ValueError("linear residualizer row count differs")
    scaler = StandardScaler().fit(c)
    model = Ridge(alpha=1.0).fit(scaler.transform(c), z)
    return LinearResidualizer(scaler, model, z.shape[1])


def paired_stratified_bootstrap(
    patient_id: Any,
    fold: Any,
    labels: Any,
    baseline_probability: Any,
    augmented_probability: Any,
    *,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Pair-resample patients within exact outer-fold x outcome strata."""

    ids = np.asarray(patient_id).astype(str)
    outer_fold = np.asarray(fold).astype(int)
    y = _binary(labels, "bootstrap labels")
    base = np.asarray(baseline_probability, dtype=np.float64)
    aug = np.asarray(augmented_probability, dtype=np.float64)
    if len(set(ids)) != len(ids):
        raise ValueError("patient IDs must be unique for patient bootstrap")
    if not (ids.shape == outer_fold.shape == y.shape == base.shape == aug.shape):
        raise ValueError("bootstrap vectors differ in length")
    if int(n_bootstrap) < 2000:
        raise ValueError("n_bootstrap must be at least 2000")
    strata = [np.flatnonzero((outer_fold == f) & (y == outcome)) for f in sorted(set(outer_fold)) for outcome in (0, 1)]
    if any(not len(index) for index in strata):
        raise ValueError("every outer-fold x outcome bootstrap stratum must be nonempty")
    rng = np.random.default_rng(random_state)
    draws = np.empty((int(n_bootstrap), 3), dtype=np.float64)
    for draw in range(int(n_bootstrap)):
        selected = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        base_metrics = binary_metrics(y[selected], base[selected])
        aug_metrics = binary_metrics(y[selected], aug[selected])
        draws[draw] = (
            float(aug_metrics["auroc"] - base_metrics["auroc"]),
            float(aug_metrics["auprc"] - base_metrics["auprc"]),
            float(base_metrics["brier"] - aug_metrics["brier"]),
        )
    alpha = (1.0 - float(confidence_level)) / 2.0
    names = ("delta_auroc", "delta_auprc", "brier_improvement")
    point_base, point_aug = binary_metrics(y, base), binary_metrics(y, aug)
    point = (
        float(point_aug["auroc"] - point_base["auroc"]),
        float(point_aug["auprc"] - point_base["auprc"]),
        float(point_base["brier"] - point_aug["brier"]),
    )
    summary: dict[str, Any] = {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "bootstrap_unit": "patient",
        "stratification": "outer_fold_x_outcome",
        "random_seed": int(random_state),
    }
    for column, name in enumerate(names):
        summary[name] = point[column]
        summary[f"{name}_ci_lower"] = float(np.quantile(draws[:, column], alpha))
        summary[f"{name}_ci_upper"] = float(np.quantile(draws[:, column], 1.0 - alpha))
    return summary, pd.DataFrame(draws, columns=names)


__all__ = [
    "BinaryFit",
    "ClinicalEncoder",
    "LinearResidualizer",
    "MulticlassFit",
    "RidgeFit",
    "binary_metrics",
    "fit_binary_logistic",
    "fit_linear_residualizer",
    "fit_multiclass_logistic",
    "fit_ridge",
    "multiclass_metrics",
    "paired_stratified_bootstrap",
    "regression_metrics",
]
