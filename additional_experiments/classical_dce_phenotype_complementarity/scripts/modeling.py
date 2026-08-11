"""Leakage-safe modeling utilities for the classical DCE experiment.

The functions in this module deliberately separate fitting data from validation
and test data.  In particular, winsor limits, log transforms, imputation
values, scaling statistics, model hyperparameters, and binary decision
thresholds are never learned from an outer-test fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    r2_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


ArrayLike = Any


def _as_float_matrix(X: ArrayLike, *, name: str = "X") -> tuple[np.ndarray, np.ndarray | None]:
    """Return a copied 2-D float matrix and optional dataframe column names."""

    columns = None
    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        raw_columns = list(X.columns)
        columns = np.asarray([str(column) for column in raw_columns], dtype=object)
        if len(set(columns.tolist())) != len(columns):
            raise ValueError(f"{name} has duplicate column names")
        try:
            array = X.to_numpy(dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only numeric values") from exc
    else:
        try:
            array = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only numeric values") from exc
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        array = np.array(array, dtype=float, copy=True)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D feature matrix; got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    # Treat infinities like other missing/non-observed numeric values.  This is
    # useful for relative-change features with a zero denominator.
    array[~np.isfinite(array)] = np.nan
    return array, columns


def _as_target_matrix(y: ArrayLike, *, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(y, dtype=float)
    was_one_dimensional = array.ndim == 1
    if was_one_dimensional:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D or 2-D numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains missing or infinite targets")
    return np.array(array, dtype=float, copy=True), was_one_dimensional


def complete_case_mask(X: ArrayLike) -> np.ndarray:
    """Return rows whose numeric feature values are all finite."""

    array, _ = _as_float_matrix(X)
    return np.all(np.isfinite(array), axis=1)


class NumericRadiomicsTransformer(BaseEstimator, TransformerMixin):
    """Train-fitted winsor/log/scale/missingness transform for radiomics.

    Parameters
    ----------
    winsor_quantiles:
        Lower and upper empirical quantiles, learned per column on training
        rows.  Pass ``None`` to disable clipping.
    log_columns:
        Column names, integer positions, or a boolean mask identifying columns
        that receive the selected log transform.
    log_transform:
        Either conventional ``"log1p"`` (requires observed values > -1) or
        ``"signed_log1p"``.
    missing_strategy:
        ``"strict"`` enforces the primary complete-case contract and raises if
        any input is non-finite. ``"median_indicator"`` implements the
        secondary analysis: train-only median imputation followed by one
        missingness indicator for every source column.
    with_scaling:
        Standardize transformed numeric columns with training means and
        population standard deviations. Missingness indicators are not scaled.

    Notes
    -----
    A column that is entirely missing in training receives a deterministic zero
    placeholder and is listed in ``all_missing_features_``.  Its indicator
    remains one for training rows, so this fallback does not invent observed
    information or consult validation/test values.
    """

    def __init__(
        self,
        *,
        winsor_quantiles: tuple[float, float] | None = (0.01, 0.99),
        log_columns: Sequence[str | int | bool] | None = None,
        log_transform: str = "log1p",
        missing_strategy: str = "strict",
        with_scaling: bool = True,
    ) -> None:
        self.winsor_quantiles = winsor_quantiles
        self.log_columns = log_columns
        self.log_transform = log_transform
        self.missing_strategy = missing_strategy
        self.with_scaling = with_scaling

    def _validated_missing_strategy(self) -> str:
        aliases = {
            "strict": "strict",
            "complete_case": "strict",
            "median_indicator": "median_indicator",
            "median+indicator": "median_indicator",
            "impute": "median_indicator",
            "secondary": "median_indicator",
        }
        try:
            return aliases[self.missing_strategy]
        except KeyError as exc:
            raise ValueError(
                "missing_strategy must be 'strict' or 'median_indicator'"
            ) from exc

    def _resolve_log_mask(self, columns: np.ndarray | None, n_features: int) -> np.ndarray:
        mask = np.zeros(n_features, dtype=bool)
        if self.log_columns is None:
            return mask

        requested = list(self.log_columns)
        if len(requested) == n_features and all(
            isinstance(value, (bool, np.bool_)) for value in requested
        ):
            return np.asarray(requested, dtype=bool)

        for value in requested:
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                index = int(value)
                if not 0 <= index < n_features:
                    raise ValueError(f"log column index {index} is out of bounds")
                mask[index] = True
            elif isinstance(value, str):
                if columns is None:
                    raise ValueError("named log_columns require a dataframe input during fit")
                matches = np.flatnonzero(columns == value)
                if len(matches) != 1:
                    raise ValueError(f"unknown log column {value!r}")
                mask[matches[0]] = True
            else:
                raise TypeError("log_columns entries must be column names or integer positions")
        return mask

    def _validate_quantiles(self) -> tuple[float, float] | None:
        if self.winsor_quantiles is None:
            return None
        if len(self.winsor_quantiles) != 2:
            raise ValueError("winsor_quantiles must contain exactly two values")
        lower, upper = (float(value) for value in self.winsor_quantiles)
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
        return lower, upper

    def _apply_train_fitted_numeric_steps(self, array: np.ndarray) -> np.ndarray:
        transformed = np.clip(array, self.clip_lower_, self.clip_upper_)
        if np.any(self.log_mask_):
            values = transformed[:, self.log_mask_]
            if self.log_transform == "log1p":
                observed = np.isfinite(values)
                if np.any(values[observed] <= -1.0):
                    raise ValueError("log1p columns contain an observed value <= -1 after clipping")
                transformed[:, self.log_mask_] = np.log1p(values)
            elif self.log_transform == "signed_log1p":
                transformed[:, self.log_mask_] = np.sign(values) * np.log1p(np.abs(values))
            else:
                raise ValueError("log_transform must be 'log1p' or 'signed_log1p'")
        return transformed

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "NumericRadiomicsTransformer":
        del y
        array, columns = _as_float_matrix(X)
        strategy = self._validated_missing_strategy()
        quantiles = self._validate_quantiles()
        if strategy == "strict" and np.any(~np.isfinite(array)):
            raise ValueError(
                "strict missingness strategy requires complete-case training rows"
            )

        self.n_features_in_ = array.shape[1]
        self.feature_names_in_ = (
            columns.copy()
            if columns is not None
            else np.asarray([f"x{index}" for index in range(self.n_features_in_)], dtype=object)
        )
        self._fit_had_named_columns_ = columns is not None
        self.log_mask_ = self._resolve_log_mask(columns, self.n_features_in_)
        self.missing_strategy_ = strategy

        lower_bounds = np.empty(self.n_features_in_, dtype=float)
        upper_bounds = np.empty(self.n_features_in_, dtype=float)
        all_missing = np.zeros(self.n_features_in_, dtype=bool)
        for column_index in range(self.n_features_in_):
            observed = array[np.isfinite(array[:, column_index]), column_index]
            if observed.size == 0:
                all_missing[column_index] = True
                lower_bounds[column_index] = 0.0
                upper_bounds[column_index] = 0.0
            elif quantiles is None:
                lower_bounds[column_index] = -np.inf
                upper_bounds[column_index] = np.inf
            else:
                lower_bounds[column_index], upper_bounds[column_index] = np.quantile(
                    observed, quantiles
                )
        self.clip_lower_ = lower_bounds
        self.clip_upper_ = upper_bounds
        self.all_missing_mask_ = all_missing
        self.all_missing_features_ = self.feature_names_in_[all_missing].copy()

        transformed = self._apply_train_fitted_numeric_steps(array)
        medians = np.empty(self.n_features_in_, dtype=float)
        for column_index in range(self.n_features_in_):
            observed = transformed[np.isfinite(transformed[:, column_index]), column_index]
            medians[column_index] = float(np.median(observed)) if observed.size else 0.0
        self.medians_ = medians

        filled = np.where(np.isfinite(transformed), transformed, medians)
        if self.with_scaling:
            self.mean_ = np.mean(filled, axis=0)
            scale = np.std(filled, axis=0, ddof=0)
            self.scale_ = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        else:
            self.mean_ = np.zeros(self.n_features_in_, dtype=float)
            self.scale_ = np.ones(self.n_features_in_, dtype=float)
        return self

    def _validate_transform_columns(
        self, array: np.ndarray, columns: np.ndarray | None
    ) -> None:
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {array.shape[1]} features, but fit saw {self.n_features_in_}"
            )
        if self._fit_had_named_columns_:
            if columns is None:
                raise ValueError("transform input must retain the dataframe columns used during fit")
            if not np.array_equal(columns, self.feature_names_in_):
                raise ValueError("transform dataframe columns differ from the fitted column order")

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, ["clip_lower_", "medians_", "mean_", "scale_"])
        array, columns = _as_float_matrix(X)
        self._validate_transform_columns(array, columns)
        missing = ~np.isfinite(array)
        if self.missing_strategy_ == "strict" and np.any(missing):
            raise ValueError("strict missingness strategy requires complete-case transform rows")

        transformed = self._apply_train_fitted_numeric_steps(array)
        transformed = np.where(np.isfinite(transformed), transformed, self.medians_)
        numeric = (transformed - self.mean_) / self.scale_
        if self.missing_strategy_ == "median_indicator":
            return np.concatenate((numeric, missing.astype(float)), axis=1)
        return numeric

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        check_is_fitted(self, ["feature_names_in_"])
        if input_features is not None:
            supplied = np.asarray([str(value) for value in input_features], dtype=object)
            if not np.array_equal(supplied, self.feature_names_in_):
                raise ValueError("input_features do not match the fitted features")
        names = self.feature_names_in_.astype(object, copy=True)
        if self.missing_strategy_ == "median_indicator":
            indicator_names = np.asarray(
                [f"{name}__missing" for name in names], dtype=object
            )
            names = np.concatenate((names, indicator_names))
        return names


class _IdentityNumericTransformer(BaseEstimator, TransformerMixin):
    """Finite-only identity used when callers explicitly omit preprocessing."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "_IdentityNumericTransformer":
        del y
        array, _ = _as_float_matrix(X)
        if not np.all(np.isfinite(array)):
            raise ValueError("X contains missing values but no preprocessor was supplied")
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, ["n_features_in_"])
        array, _ = _as_float_matrix(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("feature count differs from training")
        if not np.all(np.isfinite(array)):
            raise ValueError("X contains missing values but no preprocessor was supplied")
        return array


def _fresh_preprocessor(preprocessor: TransformerMixin | None) -> TransformerMixin:
    if preprocessor is None:
        return _IdentityNumericTransformer()
    return clone(preprocessor)


def _binary_y(y: ArrayLike, *, name: str, require_both: bool) -> np.ndarray:
    array = np.asarray(y)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must use binary labels 0 and 1") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(np.isin(numeric, (0.0, 1.0))):
        raise ValueError(f"{name} must use binary labels 0 and 1")
    result = numeric.astype(int)
    if require_both and np.unique(result).size != 2:
        raise ValueError(f"{name} must contain both binary classes")
    return result


def _probability_vector(probability: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(probability, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D probability array")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return array


def select_balanced_accuracy_threshold(
    y_validation: ArrayLike, validation_probability: ArrayLike
) -> tuple[float, float]:
    """Select a binary cutoff solely by validation balanced accuracy.

    Ties are resolved deterministically by proximity to 0.5 and then by the
    smaller threshold.
    """

    y = _binary_y(y_validation, name="y_validation", require_both=True)
    probability = _probability_vector(
        validation_probability, name="validation_probability"
    )
    if len(y) != len(probability):
        raise ValueError("validation labels and probabilities have different lengths")

    candidates = np.unique(
        np.concatenate(
            (
                probability,
                np.asarray([0.5, np.nextafter(np.max(probability), np.inf)]),
            )
        )
    )
    scores = np.asarray(
        [balanced_accuracy_score(y, probability >= threshold) for threshold in candidates]
    )
    best_score = float(np.max(scores))
    tied = candidates[np.isclose(scores, best_score, rtol=0.0, atol=1e-12)]
    threshold = min(tied.tolist(), key=lambda value: (abs(value - 0.5), value))
    return float(threshold), best_score


def _positive_class_probability(estimator: Any, X: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or len(matches) != 1:
        raise ValueError("binary estimator does not expose a unique positive class")
    return probabilities[:, matches[0]]


@dataclass(frozen=True)
class TunedBinaryClassifier:
    """A fitted train-only processor/model plus a validation-only threshold."""

    preprocessor: TransformerMixin
    estimator: Any
    model_type: str
    best_params: Mapping[str, float]
    threshold: float
    validation_auroc: float
    validation_balanced_accuracy: float
    search_results: tuple[Mapping[str, float], ...]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        return _positive_class_probability(self.estimator, transformed)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


def _normalized_model_type(model_type: str) -> str:
    normalized = model_type.lower().replace("-", "_")
    if normalized in {"lr", "logistic", "logistic_regression"}:
        return "logistic"
    if normalized in {"svm", "svc", "rbf", "rbf_svm"}:
        return "rbf_svm"
    raise ValueError("model_type must be logistic/LR or rbf_svm/SVM")


def _validated_positive_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _candidate_estimators(
    model_type: str,
    *,
    logistic_c_grid: Iterable[float],
    svm_c_grid: Iterable[float],
    svm_gamma_grid: Iterable[float],
    class_weight: str | Mapping[int, float] | None,
    random_state: int,
) -> Iterable[tuple[dict[str, float], Any]]:
    if model_type == "logistic":
        for c_value in _validated_positive_grid(logistic_c_grid, name="logistic_c_grid"):
            params = {"C": c_value}
            yield params, LogisticRegression(
                C=c_value,
                solver="lbfgs",
                class_weight=class_weight,
                max_iter=5000,
                random_state=random_state,
            )
    else:
        c_grid = _validated_positive_grid(svm_c_grid, name="svm_c_grid")
        gamma_grid = _validated_positive_grid(svm_gamma_grid, name="svm_gamma_grid")
        for c_value in c_grid:
            for gamma_value in gamma_grid:
                params = {"C": c_value, "gamma": gamma_value}
                yield params, SVC(
                    C=c_value,
                    gamma=gamma_value,
                    kernel="rbf",
                    probability=True,
                    class_weight=class_weight,
                    random_state=random_state,
                )


def tune_binary_classifier(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_validation: ArrayLike,
    y_validation: ArrayLike,
    *,
    model_type: str = "logistic",
    preprocessor: TransformerMixin | None = None,
    logistic_c_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    svm_c_grid: Iterable[float] = (0.1, 1.0, 10.0, 100.0),
    svm_gamma_grid: Iterable[float] = (0.001, 0.01, 0.1, 1.0),
    class_weight: str | Mapping[int, float] | None = None,
    random_state: int = 0,
) -> TunedBinaryClassifier:
    """Tune LR or RBF-SVM on validation AUROC without fitting validation data."""

    train_labels = _binary_y(y_train, name="y_train", require_both=True)
    validation_labels = _binary_y(
        y_validation, name="y_validation", require_both=True
    )
    processor = _fresh_preprocessor(preprocessor)
    train_features = processor.fit_transform(X_train)
    validation_features = processor.transform(X_validation)
    if len(train_features) != len(train_labels) or len(validation_features) != len(
        validation_labels
    ):
        raise ValueError("feature matrices and labels have different row counts")

    normalized_type = _normalized_model_type(model_type)
    best: tuple[float, dict[str, float], Any, np.ndarray] | None = None
    search_results: list[Mapping[str, float]] = []
    for params, estimator in _candidate_estimators(
        normalized_type,
        logistic_c_grid=logistic_c_grid,
        svm_c_grid=svm_c_grid,
        svm_gamma_grid=svm_gamma_grid,
        class_weight=class_weight,
        random_state=int(random_state),
    ):
        estimator.fit(train_features, train_labels)
        probability = _positive_class_probability(estimator, validation_features)
        validation_auroc = float(roc_auc_score(validation_labels, probability))
        search_results.append({**params, "validation_auroc": validation_auroc})
        if best is None or validation_auroc > best[0] + 1e-12:
            best = (validation_auroc, params, estimator, probability)

    if best is None:  # defensive; grids are checked above
        raise RuntimeError("no classifier candidates were evaluated")
    validation_auroc, best_params, estimator, probability = best
    threshold, validation_bac = select_balanced_accuracy_threshold(
        validation_labels, probability
    )
    return TunedBinaryClassifier(
        preprocessor=processor,
        estimator=estimator,
        model_type=normalized_type,
        best_params=dict(best_params),
        threshold=threshold,
        validation_auroc=validation_auroc,
        validation_balanced_accuracy=validation_bac,
        search_results=tuple(search_results),
    )


@dataclass(frozen=True)
class TunedMulticlassClassifier:
    preprocessor: TransformerMixin
    estimator: Any
    model_type: str
    best_params: Mapping[str, float]
    validation_auroc: float
    search_results: tuple[Mapping[str, float], ...]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(
            self.estimator.predict_proba(self.preprocessor.transform(X)), dtype=float
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.preprocessor.transform(X)))

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.estimator.classes_)


def tune_multiclass_classifier(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_validation: ArrayLike,
    y_validation: ArrayLike,
    *,
    model_type: str = "logistic",
    preprocessor: TransformerMixin | None = None,
    logistic_c_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    svm_c_grid: Iterable[float] = (0.1, 1.0, 10.0, 100.0),
    svm_gamma_grid: Iterable[float] = (0.001, 0.01, 0.1, 1.0),
    class_weight: str | Mapping[Any, float] | None = None,
    random_state: int = 0,
    classes: Sequence[Any] | None = None,
) -> TunedMulticlassClassifier:
    """Tune multiclass LR or RBF-SVM by validation macro OVR AUROC."""

    train_labels = np.asarray(y_train)
    validation_labels = np.asarray(y_validation)
    if train_labels.ndim != 1 or validation_labels.ndim != 1:
        raise ValueError("multiclass labels must be 1-D")
    expected_classes = np.asarray(np.unique(train_labels) if classes is None else list(classes))
    if expected_classes.ndim != 1 or len(np.unique(expected_classes)) != len(expected_classes):
        raise ValueError("classes must be a 1-D sequence without duplicates")
    if len(expected_classes) < 3:
        raise ValueError("tune_multiclass_classifier requires at least three classes")
    expected_set = set(expected_classes.tolist())
    if set(np.unique(train_labels).tolist()) != expected_set:
        raise ValueError("training labels do not contain exactly the requested classes")
    if set(np.unique(validation_labels).tolist()) != expected_set:
        raise ValueError("validation must contain every requested class for AUROC selection")

    processor = _fresh_preprocessor(preprocessor)
    train_features = processor.fit_transform(X_train)
    validation_features = processor.transform(X_validation)
    if len(train_features) != len(train_labels) or len(validation_features) != len(
        validation_labels
    ):
        raise ValueError("feature matrices and labels have different row counts")

    normalized_type = _normalized_model_type(model_type)
    best: tuple[float, dict[str, float], Any] | None = None
    search_results: list[Mapping[str, float]] = []
    for params, estimator in _candidate_estimators(
        normalized_type,
        logistic_c_grid=logistic_c_grid,
        svm_c_grid=svm_c_grid,
        svm_gamma_grid=svm_gamma_grid,
        class_weight=class_weight,
        random_state=int(random_state),
    ):
        estimator.fit(train_features, train_labels)
        probability = np.asarray(estimator.predict_proba(validation_features), dtype=float)
        score = float(
            roc_auc_score(
                validation_labels,
                probability,
                labels=np.asarray(estimator.classes_),
                multi_class="ovr",
                average="macro",
            )
        )
        search_results.append({**params, "validation_auroc": score})
        if best is None or score > best[0] + 1e-12:
            best = (score, params, estimator)
    if best is None:
        raise RuntimeError("no classifier candidates were evaluated")
    score, params, estimator = best
    return TunedMulticlassClassifier(
        preprocessor=processor,
        estimator=estimator,
        model_type=normalized_type,
        best_params=dict(params),
        validation_auroc=score,
        search_results=tuple(search_results),
    )


def binary_metrics(
    y_true: ArrayLike, probability: ArrayLike, *, threshold: float = 0.5
) -> dict[str, float | int]:
    """Compute binary AUROC, AUPRC, balanced accuracy, and Brier score."""

    y = _binary_y(y_true, name="y_true", require_both=False)
    prob = _probability_vector(probability, name="probability")
    if len(y) != len(prob):
        raise ValueError("y_true and probability have different lengths")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    has_both = np.unique(y).size == 2
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, prob)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y, prob)) if has_both else float("nan"),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, prob >= threshold))
            if has_both
            else float("nan")
        ),
        "brier": float(brier_score_loss(y, prob)),
        "threshold": float(threshold),
    }


def predict_binary(
    model: TunedBinaryClassifier, X: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-class probabilities and validation-thresholded labels."""

    probability = model.predict_proba(X)
    prediction = (probability >= model.threshold).astype(int)
    return probability, prediction


def multiclass_metrics(
    y_true: ArrayLike,
    probability: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, float | int]:
    """Compute macro OVR AUROC/AUPRC, balanced accuracy, and Brier score."""

    y = np.asarray(y_true)
    prob = np.asarray(probability, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_true must be a non-empty 1-D array")
    if prob.ndim != 2 or prob.shape[0] != len(y):
        raise ValueError("probability must have one row per y_true value")
    class_labels = np.asarray(np.unique(y) if labels is None else list(labels))
    if len(class_labels) < 2 or prob.shape[1] != len(class_labels):
        raise ValueError("probability columns must match the supplied labels")
    if not np.all(np.isfinite(prob)) or np.any(prob < 0.0):
        raise ValueError("multiclass probabilities must be finite and non-negative")
    row_sums = np.sum(prob, axis=1)
    if np.any(row_sums <= 0.0) or not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("multiclass probability rows must sum to one")
    label_to_index = {label: index for index, label in enumerate(class_labels.tolist())}
    try:
        indices = np.asarray([label_to_index[label] for label in y.tolist()], dtype=int)
    except KeyError as exc:
        raise ValueError("y_true contains a class absent from labels") from exc
    one_hot = np.zeros_like(prob)
    one_hot[np.arange(len(y)), indices] = 1.0

    present = np.asarray([np.any(indices == index) for index in range(len(class_labels))])
    absent = np.asarray([np.all(indices == index) for index in range(len(class_labels))])
    valid_binary_columns = present & ~absent
    per_class_auprc = np.full(len(class_labels), np.nan, dtype=float)
    for index in np.flatnonzero(valid_binary_columns):
        per_class_auprc[index] = average_precision_score(one_hot[:, index], prob[:, index])

    all_classes_present = np.all(present)
    if all_classes_present and len(class_labels) == 2:
        auroc = float(roc_auc_score(one_hot[:, 1], prob[:, 1]))
    elif all_classes_present:
        auroc = float(
            roc_auc_score(
                y,
                prob,
                labels=class_labels,
                multi_class="ovr",
                average="macro",
            )
        )
    else:
        auroc = float("nan")
    predictions = class_labels[np.argmax(prob, axis=1)]
    return {
        "n": int(len(y)),
        "n_classes": int(len(class_labels)),
        "auroc": auroc,
        "auprc": float(np.nanmean(per_class_auprc)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "brier": float(np.mean(np.sum((one_hot - prob) ** 2, axis=1))),
        "accuracy": float(np.mean(predictions == y)),
    }


def redundancy_metrics(ftv_true: ArrayLike, ftv_predicted: ArrayLike) -> dict[str, Any]:
    """R-squared and Spearman diagnostics, including each FTV output."""

    true, _ = _as_target_matrix(ftv_true, name="ftv_true")
    predicted, _ = _as_target_matrix(ftv_predicted, name="ftv_predicted")
    if true.shape != predicted.shape:
        raise ValueError("ftv_true and ftv_predicted must have identical shapes")

    r2_values: list[float] = []
    spearman_values: list[float] = []
    for index in range(true.shape[1]):
        if len(true) < 2 or np.ptp(true[:, index]) == 0.0:
            r2_values.append(float("nan"))
        else:
            r2_values.append(float(r2_score(true[:, index], predicted[:, index])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            correlation = spearmanr(true[:, index], predicted[:, index]).statistic
        spearman_values.append(float(correlation) if np.isfinite(correlation) else float("nan"))

    return {
        "r2": float(np.nanmean(r2_values)) if np.any(np.isfinite(r2_values)) else float("nan"),
        "spearman": (
            float(np.nanmean(spearman_values))
            if np.any(np.isfinite(spearman_values))
            else float("nan")
        ),
        "r2_per_output": r2_values,
        "spearman_per_output": spearman_values,
    }


@dataclass(frozen=True)
class RidgeRedundancyModel:
    preprocessor: TransformerMixin
    estimator: Ridge
    alpha: float
    n_outputs: int
    single_output: bool
    validation_metrics: Mapping[str, Any] | None
    search_results: tuple[Mapping[str, float], ...]

    def predict(self, X_nonftv: ArrayLike) -> np.ndarray:
        prediction = np.asarray(
            self.estimator.predict(self.preprocessor.transform(X_nonftv)), dtype=float
        )
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        return prediction[:, 0] if self.single_output else prediction


def fit_ridge_redundancy(
    X_nonftv_train: ArrayLike,
    ftv_train: ArrayLike,
    X_nonftv_validation: ArrayLike | None = None,
    ftv_validation: ArrayLike | None = None,
    *,
    alphas: Iterable[float] = (1.0,),
    preprocessor: TransformerMixin | None = None,
) -> RidgeRedundancyModel:
    """Fit NONFTV -> FTV ridge, optionally choosing alpha by validation R2."""

    targets, single_output = _as_target_matrix(ftv_train, name="ftv_train")
    if (X_nonftv_validation is None) != (ftv_validation is None):
        raise ValueError("validation features and targets must be supplied together")
    alpha_grid = tuple(float(value) for value in alphas)
    if not alpha_grid or any(not np.isfinite(value) or value < 0.0 for value in alpha_grid):
        raise ValueError("alphas must contain finite non-negative values")
    if len(alpha_grid) > 1 and X_nonftv_validation is None:
        raise ValueError("multiple ridge alphas require a held-out validation set")

    processor = _fresh_preprocessor(preprocessor)
    train_features = processor.fit_transform(X_nonftv_train)
    if len(train_features) != len(targets):
        raise ValueError("training features and FTV targets have different row counts")
    validation_features = None
    validation_targets = None
    if X_nonftv_validation is not None:
        validation_features = processor.transform(X_nonftv_validation)
        validation_targets, _ = _as_target_matrix(ftv_validation, name="ftv_validation")
        if len(validation_features) != len(validation_targets):
            raise ValueError("validation features and FTV targets have different row counts")
        if validation_targets.shape[1] != targets.shape[1]:
            raise ValueError("training and validation FTV targets have different widths")

    best: tuple[float, float, Ridge, Mapping[str, Any] | None] | None = None
    search_results: list[Mapping[str, float]] = []
    for alpha in alpha_grid:
        estimator = Ridge(alpha=alpha)
        estimator.fit(train_features, targets)
        metrics = None
        score = 0.0
        if validation_features is not None and validation_targets is not None:
            prediction = estimator.predict(validation_features)
            metrics = redundancy_metrics(validation_targets, prediction)
            score = float(metrics["r2"])
            if not np.isfinite(score):
                score = -float(np.mean((validation_targets - prediction) ** 2))
        search_results.append({"alpha": alpha, "validation_score": score})
        if best is None or score > best[0] + 1e-12:
            best = (score, alpha, estimator, metrics)
    if best is None:
        raise RuntimeError("no ridge candidates were evaluated")
    _, alpha, estimator, metrics = best
    return RidgeRedundancyModel(
        preprocessor=processor,
        estimator=estimator,
        alpha=alpha,
        n_outputs=targets.shape[1],
        single_output=single_output,
        validation_metrics=metrics,
        search_results=tuple(search_results),
    )


def evaluate_ridge_redundancy(
    model: RidgeRedundancyModel, X_nonftv: ArrayLike, ftv_true: ArrayLike
) -> dict[str, Any]:
    return redundancy_metrics(ftv_true, model.predict(X_nonftv))


class FTVResidualizer(BaseEstimator):
    """Train-fitted multioutput ridge residualization: FTV -> NONFTV.

    Fit on an outer-training partition, then call :meth:`transform` separately
    on train, validation, and test partitions.  The subtraction always occurs
    in the original NONFTV units.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        preprocessor: TransformerMixin | None = None,
    ) -> None:
        self.alpha = alpha
        self.preprocessor = preprocessor

    def fit(self, ftv: ArrayLike, nonftv: ArrayLike) -> "FTVResidualizer":
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        targets, single_output = _as_target_matrix(nonftv, name="nonftv")
        processor = _fresh_preprocessor(self.preprocessor)
        ftv_features = processor.fit_transform(ftv)
        if len(ftv_features) != len(targets):
            raise ValueError("FTV and NONFTV training rows differ")
        estimator = Ridge(alpha=float(self.alpha))
        estimator.fit(ftv_features, targets)
        self.preprocessor_ = processor
        self.estimator_ = estimator
        self.n_nonftv_outputs_ = targets.shape[1]
        self.single_nonftv_output_ = single_output
        return self

    def predict(self, ftv: ArrayLike) -> np.ndarray:
        check_is_fitted(self, ["preprocessor_", "estimator_", "n_nonftv_outputs_"])
        prediction = np.asarray(
            self.estimator_.predict(self.preprocessor_.transform(ftv)), dtype=float
        )
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        return prediction[:, 0] if self.single_nonftv_output_ else prediction

    def transform(self, ftv: ArrayLike, nonftv: ArrayLike) -> np.ndarray:
        check_is_fitted(self, ["preprocessor_", "estimator_", "n_nonftv_outputs_"])
        targets, single_output = _as_target_matrix(nonftv, name="nonftv")
        prediction = np.asarray(self.predict(ftv), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        if targets.shape != prediction.shape:
            raise ValueError("FTV rows or NONFTV output width differ from residualizer fit")
        residuals = targets - prediction
        return residuals[:, 0] if single_output else residuals

    def fit_transform(self, ftv: ArrayLike, nonftv: ArrayLike) -> np.ndarray:
        return self.fit(ftv, nonftv).transform(ftv, nonftv)


def _validate_fold_ids(fold_ids: ArrayLike, expected_length: int) -> np.ndarray:
    folds = np.asarray(fold_ids, dtype=object)
    if folds.ndim != 1 or len(folds) != expected_length:
        raise ValueError("fold_ids must be 1-D with one value per patient")
    for value in folds.tolist():
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            raise ValueError("fold_ids cannot contain missing values")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("fold_ids values must be hashable") from exc
    return folds


def _validate_unique_patients(patient_ids: ArrayLike | None, expected_length: int) -> None:
    if patient_ids is None:
        return
    patients = np.asarray(patient_ids, dtype=object)
    if patients.ndim != 1 or len(patients) != expected_length:
        raise ValueError("patient_ids must be 1-D with one value per prediction")
    seen: set[Any] = set()
    for patient in patients.tolist():
        try:
            duplicate = patient in seen
            seen.add(patient)
        except TypeError as exc:
            raise ValueError("patient_ids values must be hashable") from exc
        if duplicate:
            raise ValueError("patient_ids must be unique for patient-level bootstrap")


def _paired_effects(y: np.ndarray, baseline: np.ndarray, augmented: np.ndarray) -> dict[str, float]:
    return {
        "delta_auroc": float(roc_auc_score(y, augmented) - roc_auc_score(y, baseline)),
        "delta_auprc": float(
            average_precision_score(y, augmented) - average_precision_score(y, baseline)
        ),
        # Positive means the augmented model improves the proper scoring rule.
        "brier_improvement": float(
            brier_score_loss(y, baseline) - brier_score_loss(y, augmented)
        ),
    }


def paired_fold_stratified_bootstrap(
    y_true: ArrayLike,
    baseline_probability: ArrayLike,
    augmented_probability: ArrayLike,
    fold_ids: ArrayLike,
    *,
    patient_ids: ArrayLike | None = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 0,
    stratify_outcome: bool = True,
    return_distributions: bool = False,
) -> dict[str, Any]:
    """Paired patient bootstrap that preserves outer-fold strata.

    By default, samples are stratified jointly by outer fold and observed
    outcome.  This preserves fold composition while ensuring every replicate
    supports AUROC/AUPRC. Set ``stratify_outcome=False`` for fold-only strata;
    single-class replicates are then rejected with a clear error.

    ``n_bootstrap`` is intentionally required to be at least 2,000, matching
    the preregistered experiment contract.
    """

    if isinstance(n_bootstrap, (bool, np.bool_)) or int(n_bootstrap) != n_bootstrap:
        raise ValueError("n_bootstrap must be an integer")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 2000:
        raise ValueError("n_bootstrap must be at least 2000")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    y = _binary_y(y_true, name="y_true", require_both=True)
    baseline = _probability_vector(baseline_probability, name="baseline_probability")
    augmented = _probability_vector(augmented_probability, name="augmented_probability")
    if len(baseline) != len(y) or len(augmented) != len(y):
        raise ValueError("labels and paired prediction arrays have different lengths")
    folds = _validate_fold_ids(fold_ids, len(y))
    _validate_unique_patients(patient_ids, len(y))

    strata: dict[tuple[Any, ...], list[int]] = {}
    for index, (fold, outcome) in enumerate(zip(folds.tolist(), y.tolist())):
        key = (fold, outcome) if stratify_outcome else (fold,)
        strata.setdefault(key, []).append(index)
    stratum_indices = [np.asarray(indices, dtype=int) for indices in strata.values()]

    rng = np.random.default_rng(int(random_state))
    metric_names = ("delta_auroc", "delta_auprc", "brier_improvement")
    distributions = {
        metric: np.empty(n_bootstrap, dtype=float) for metric in metric_names
    }
    for draw in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in stratum_indices]
        )
        if np.unique(y[sampled]).size != 2:
            raise ValueError(
                "a fold-only bootstrap replicate has one class; use outcome stratification"
            )
        effects = _paired_effects(y[sampled], baseline[sampled], augmented[sampled])
        for metric in metric_names:
            distributions[metric][draw] = effects[metric]

    observed = _paired_effects(y, baseline, augmented)
    alpha = (1.0 - confidence_level) / 2.0
    result: dict[str, Any] = {
        "n_patients": int(len(y)),
        "n_bootstrap": n_bootstrap,
        "confidence_level": float(confidence_level),
        "random_state": int(random_state),
        "stratification": "outer_fold+outcome" if stratify_outcome else "outer_fold",
    }
    for metric in metric_names:
        lower, upper = np.quantile(distributions[metric], [alpha, 1.0 - alpha])
        result[metric] = {
            "estimate": observed[metric],
            "ci_low": float(lower),
            "ci_high": float(upper),
        }
    if return_distributions:
        result["distributions"] = {
            metric: values.copy() for metric, values in distributions.items()
        }
    return result


# Readable aliases used by runners and notebooks.
RadiomicsPreprocessor = NumericRadiomicsTransformer
MultiOutputFTVResidualizer = FTVResidualizer
compute_binary_metrics = binary_metrics
compute_multiclass_metrics = multiclass_metrics
fit_tuned_binary_classifier = tune_binary_classifier
paired_patient_bootstrap = paired_fold_stratified_bootstrap


__all__ = [
    "FTVResidualizer",
    "MultiOutputFTVResidualizer",
    "NumericRadiomicsTransformer",
    "RadiomicsPreprocessor",
    "RidgeRedundancyModel",
    "TunedBinaryClassifier",
    "TunedMulticlassClassifier",
    "binary_metrics",
    "complete_case_mask",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "evaluate_ridge_redundancy",
    "fit_ridge_redundancy",
    "fit_tuned_binary_classifier",
    "multiclass_metrics",
    "paired_fold_stratified_bootstrap",
    "paired_patient_bootstrap",
    "predict_binary",
    "redundancy_metrics",
    "select_balanced_accuracy_threshold",
    "tune_binary_classifier",
    "tune_multiclass_classifier",
]
