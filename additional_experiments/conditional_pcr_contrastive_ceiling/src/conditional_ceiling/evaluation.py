"""Fold-isolated downstream evaluation for the conditional pCR ceiling.

MRI prefixes are compacted by one PCA fitted on the outer-training rows only.
PCA dimension and L2-logistic ``C`` are selected exclusively by outer-validation
AUROC.  The held-out matrix is accepted only for prediction after selection; no
test label appears in a fitting signature.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from .metrics import binary_metrics, calibration_slope, ece10


PCA_DIMENSIONS = (8, 16, 32, 64)
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SELECTION_ATOL = 1e-12
FULL_POPULATION = "full_808"
FTV_POPULATION = "ftv_complete_375"
FULL_MODEL_FAMILIES = ("C", "M", "C+M")
FTV_MODEL_FAMILIES = ("F", "C+F", "C+F+M")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not boolean")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return int(parsed)


def _matrix(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
    expected_features: int | None = None,
) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty shape [N,F]; got {matrix.shape}")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(f"{name} has {matrix.shape[0]} rows; expected {expected_rows}")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            f"{name} has {matrix.shape[1]} features; expected {expected_features}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return matrix


def _binary_labels(values: Any, *, name: str, expected_rows: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != expected_rows:
        raise ValueError(f"{name} must have shape ({expected_rows},)")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must use integer 0/1 labels, not booleans")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain binary 0/1 labels") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only finite binary 0/1 labels")
    labels = numeric.astype(np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


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
    if not grid or any(not math.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError(f"{name} must contain finite positive numbers")
    return grid


def _dimensions(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("dimensions must be an iterable")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError("dimensions must be an iterable") from error
    parsed: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("dimensions must contain integers, not booleans")
        try:
            dimension = operator.index(value)
        except TypeError as error:
            raise TypeError("dimensions must contain integers") from error
        parsed.append(int(dimension))
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("dimensions must be non-empty and contain no duplicates")
    if not set(parsed).issubset(PCA_DIMENSIONS):
        raise ValueError(f"dimensions must be selected from {PCA_DIMENSIONS}")
    return tuple(sorted(parsed))


def _validated_class_weight(
    value: str | Mapping[int, float] | None,
) -> str | Mapping[int, float] | None:
    if value is None or value == "balanced":
        return value
    if not isinstance(value, Mapping) or not value:
        raise ValueError("class_weight must be None, 'balanced', or a mapping")
    return value


@dataclass(frozen=True)
class CandidateScore:
    dimension: int | None
    c_value: float
    validation_auroc: float


@dataclass(frozen=True)
class SelectedLogisticFit:
    """Selected fold-isolated classifier and its split predictions."""

    pca: PCA | None
    scaler: StandardScaler
    model: LogisticRegression
    selected_dimension: int | None
    selected_c: float
    validation_auroc: float
    candidate_scores: tuple[CandidateScore, ...]
    input_dim: int
    covariate_dim: int
    train_rows: int
    validation_rows: int
    train_probabilities: np.ndarray
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray | None
    selection_metric: str = "validation_auroc"
    tie_break: str = "smaller_dimension_then_smaller_C"

    @property
    def selected_C(self) -> float:  # noqa: N802 - ledger-compatible spelling
        return self.selected_c

    @property
    def feature_dim(self) -> int:
        compact = self.selected_dimension if self.pca is not None else self.input_dim
        assert compact is not None
        return int(compact + self.covariate_dim)

    def predict_proba(
        self, features: Any, *, covariates: Any | None = None
    ) -> np.ndarray:
        matrix = _matrix(
            features,
            name="prediction features",
            expected_features=self.input_dim,
        )
        if self.pca is not None:
            assert self.selected_dimension is not None
            design = self.pca.transform(matrix)[:, : self.selected_dimension]
        else:
            design = matrix
        if self.covariate_dim:
            if covariates is None:
                raise ValueError("prediction covariates are required for this fit")
            extra = _matrix(
                covariates,
                name="prediction covariates",
                expected_rows=len(matrix),
                expected_features=self.covariate_dim,
            )
            design = np.column_stack((extra, design))
        elif covariates is not None:
            raise ValueError("prediction covariates were supplied to a fit without covariates")
        probability = np.asarray(
            self.model.predict_proba(self.scaler.transform(design))[:, 1],
            dtype=np.float64,
        )
        if probability.shape != (len(matrix),) or not np.isfinite(probability).all():
            raise RuntimeError("logistic model returned invalid probabilities")
        return probability


CompactLogisticFit = SelectedLogisticFit


def _fit_candidate(
    train_design: np.ndarray,
    train_labels: np.ndarray,
    validation_design: np.ndarray,
    validation_labels: np.ndarray,
    *,
    c_value: float,
    class_weight: str | Mapping[int, float] | None,
    solver: str,
    max_iter: int,
    random_state: int,
) -> tuple[StandardScaler, LogisticRegression, float]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_design)
    validation_scaled = scaler.transform(validation_design)
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train_scaled, train_labels)
    probability = model.predict_proba(validation_scaled)[:, 1]
    score = float(roc_auc_score(validation_labels, probability))
    if not math.isfinite(score):
        raise RuntimeError("candidate validation AUROC is non-finite")
    return scaler, model, score


def fit_compact_logistic(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    test_features: Any | None = None,
    *,
    train_covariates: Any | None = None,
    validation_covariates: Any | None = None,
    test_covariates: Any | None = None,
    dimensions: Iterable[int] = PCA_DIMENSIONS,
    c_grid: Iterable[float] = C_GRID,
    class_weight: str | Mapping[int, float] | None = None,
    solver: str = "liblinear",
    max_iter: int = 10_000,
    random_state: int = 0,
) -> SelectedLogisticFit:
    """Fit outer-train PCA and select compact dimension/C on validation AUROC."""

    train_x = _matrix(train_features, name="train MRI features")
    validation_x = _matrix(
        validation_features,
        name="validation MRI features",
        expected_features=train_x.shape[1],
    )
    test_x = (
        None
        if test_features is None
        else _matrix(
            test_features,
            name="test MRI features",
            expected_features=train_x.shape[1],
        )
    )
    train_y = _binary_labels(train_labels, name="train labels", expected_rows=len(train_x))
    validation_y = _binary_labels(
        validation_labels,
        name="validation labels",
        expected_rows=len(validation_x),
    )
    dims = _dimensions(dimensions)
    grid = _positive_grid(c_grid, name="c_grid")
    iterations = _positive_integer(max_iter, name="max_iter")
    weights = _validated_class_weight(class_weight)
    if solver != "liblinear":
        raise ValueError("registered evaluation requires solver='liblinear'")
    rank_bound = min(train_x.shape[1], train_x.shape[0] - 1)
    if max(dims) > rank_bound:
        raise ValueError(
            "largest PCA dimension exceeds centered outer-train rank bound "
            f"min(F,N-1)={rank_bound}"
        )

    have_covariates = train_covariates is not None
    if have_covariates != (validation_covariates is not None):
        raise ValueError("train and validation covariates must be supplied together")
    if test_x is None and test_covariates is not None:
        raise ValueError("test_covariates require test_features")
    if test_x is not None and have_covariates != (test_covariates is not None):
        raise ValueError("test covariates must match the training covariate contract")
    if have_covariates:
        train_c = _matrix(
            train_covariates, name="train covariates", expected_rows=len(train_x)
        )
        validation_c = _matrix(
            validation_covariates,
            name="validation covariates",
            expected_rows=len(validation_x),
            expected_features=train_c.shape[1],
        )
        test_c = (
            None
            if test_x is None
            else _matrix(
                test_covariates,
                name="test covariates",
                expected_rows=len(test_x),
                expected_features=train_c.shape[1],
            )
        )
    else:
        train_c = validation_c = test_c = None

    pca = PCA(n_components=max(dims), svd_solver="full", whiten=False)
    pca.fit(train_x)
    train_compact = pca.transform(train_x)
    validation_compact = pca.transform(validation_x)
    test_compact = None if test_x is None else pca.transform(test_x)
    candidates: list[
        tuple[CandidateScore, StandardScaler, LogisticRegression, np.ndarray, np.ndarray]
    ] = []
    for dimension in dims:
        train_design = train_compact[:, :dimension]
        validation_design = validation_compact[:, :dimension]
        if train_c is not None:
            train_design = np.column_stack((train_c, train_design))
            validation_design = np.column_stack((validation_c, validation_design))
        for c_value in grid:
            scaler, model, score = _fit_candidate(
                train_design,
                train_y,
                validation_design,
                validation_y,
                c_value=c_value,
                class_weight=weights,
                solver=solver,
                max_iter=iterations,
                random_state=int(random_state),
            )
            candidates.append(
                (
                    CandidateScore(dimension, c_value, score),
                    scaler,
                    model,
                    train_design,
                    validation_design,
                )
            )
    best_score = max(item[0].validation_auroc for item in candidates)
    eligible = [
        item
        for item in candidates
        if item[0].validation_auroc >= best_score - SELECTION_ATOL
    ]
    selected = min(eligible, key=lambda item: (int(item[0].dimension or 0), item[0].c_value))
    score, scaler, model, selected_train, selected_validation = selected
    assert score.dimension is not None
    train_probability = model.predict_proba(scaler.transform(selected_train))[:, 1]
    validation_probability = model.predict_proba(scaler.transform(selected_validation))[:, 1]
    test_probability: np.ndarray | None = None
    if test_compact is not None:
        selected_test = test_compact[:, : score.dimension]
        if test_c is not None:
            selected_test = np.column_stack((test_c, selected_test))
        test_probability = model.predict_proba(scaler.transform(selected_test))[:, 1]
    return SelectedLogisticFit(
        pca=pca,
        scaler=scaler,
        model=model,
        selected_dimension=score.dimension,
        selected_c=score.c_value,
        validation_auroc=score.validation_auroc,
        candidate_scores=tuple(item[0] for item in candidates),
        input_dim=int(train_x.shape[1]),
        covariate_dim=0 if train_c is None else int(train_c.shape[1]),
        train_rows=int(len(train_x)),
        validation_rows=int(len(validation_x)),
        train_probabilities=np.asarray(train_probability, dtype=np.float64),
        validation_probabilities=np.asarray(validation_probability, dtype=np.float64),
        test_probabilities=(
            None if test_probability is None else np.asarray(test_probability, dtype=np.float64)
        ),
    )


def fit_l2_logistic(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    test_features: Any | None = None,
    *,
    c_grid: Iterable[float] = C_GRID,
    class_weight: str | Mapping[int, float] | None = None,
    solver: str = "liblinear",
    max_iter: int = 10_000,
    random_state: int = 0,
) -> SelectedLogisticFit:
    """Select a standardized non-PCA L2 logistic classifier on validation AUROC."""

    train_x = _matrix(train_features, name="train features")
    validation_x = _matrix(
        validation_features,
        name="validation features",
        expected_features=train_x.shape[1],
    )
    test_x = (
        None
        if test_features is None
        else _matrix(test_features, name="test features", expected_features=train_x.shape[1])
    )
    train_y = _binary_labels(train_labels, name="train labels", expected_rows=len(train_x))
    validation_y = _binary_labels(
        validation_labels, name="validation labels", expected_rows=len(validation_x)
    )
    grid = _positive_grid(c_grid, name="c_grid")
    iterations = _positive_integer(max_iter, name="max_iter")
    weights = _validated_class_weight(class_weight)
    if solver != "liblinear":
        raise ValueError("registered evaluation requires solver='liblinear'")
    candidates: list[tuple[CandidateScore, StandardScaler, LogisticRegression]] = []
    for c_value in grid:
        scaler, model, score = _fit_candidate(
            train_x,
            train_y,
            validation_x,
            validation_y,
            c_value=c_value,
            class_weight=weights,
            solver=solver,
            max_iter=iterations,
            random_state=int(random_state),
        )
        candidates.append((CandidateScore(None, c_value, score), scaler, model))
    best_score = max(item[0].validation_auroc for item in candidates)
    eligible = [item for item in candidates if item[0].validation_auroc >= best_score - SELECTION_ATOL]
    score, scaler, model = min(eligible, key=lambda item: item[0].c_value)
    train_probability = model.predict_proba(scaler.transform(train_x))[:, 1]
    validation_probability = model.predict_proba(scaler.transform(validation_x))[:, 1]
    test_probability = (
        None if test_x is None else model.predict_proba(scaler.transform(test_x))[:, 1]
    )
    return SelectedLogisticFit(
        pca=None,
        scaler=scaler,
        model=model,
        selected_dimension=None,
        selected_c=score.c_value,
        validation_auroc=score.validation_auroc,
        candidate_scores=tuple(item[0] for item in candidates),
        input_dim=int(train_x.shape[1]),
        covariate_dim=0,
        train_rows=int(len(train_x)),
        validation_rows=int(len(validation_x)),
        train_probabilities=np.asarray(train_probability, dtype=np.float64),
        validation_probabilities=np.asarray(validation_probability, dtype=np.float64),
        test_probabilities=(
            None if test_probability is None else np.asarray(test_probability, dtype=np.float64)
        ),
        tie_break="smaller_C",
    )


@dataclass(frozen=True)
class FoldEvaluation:
    fits: Mapping[str, SelectedLogisticFit]
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame


def _indices(values: Any, *, name: str, n_rows: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(raw.dtype, np.bool_):
        if len(raw) != n_rows:
            raise ValueError(f"{name} boolean mask has {len(raw)} rows; expected {n_rows}")
        indices = np.flatnonzero(raw)
    else:
        try:
            numeric = raw.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain integer row indices") from error
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name} must contain finite integer row indices")
        indices = numeric.astype(np.int64)
    if not len(indices) or np.any((indices < 0) | (indices >= n_rows)):
        raise ValueError(f"{name} must contain valid, non-empty row indices")
    if len(indices) != len(np.unique(indices)):
        raise ValueError(f"{name} contains duplicate row indices")
    return indices


def _split_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    if set(np.unique(labels)) != {0, 1}:
        return {
            "n": int(len(labels)),
            "n_positive": int(labels.sum()),
            "n_negative": int((labels == 0).sum()),
            "auroc": math.nan,
            "auprc": math.nan,
            "brier": float(np.mean(np.square(probabilities - labels))),
            "calibration_slope": math.nan,
            "ece10": ece10(labels, probabilities),
        }
    return binary_metrics(labels, probabilities)


def evaluate_feature_families(
    *,
    labels: Any,
    mri_features: Any,
    clinical_features: Any,
    train_indices: Any,
    validation_indices: Any,
    test_indices: Any,
    population: str,
    ftv_features: Any | None = None,
    patient_ids: Sequence[Any] | None = None,
    outer_fold: int = 0,
    dimensions: Iterable[int] = PCA_DIMENSIONS,
    c_grid: Iterable[float] = C_GRID,
    class_weight: str | Mapping[int, float] | None = None,
    random_state: int = 0,
) -> FoldEvaluation:
    """Evaluate the registered feature families for one outer fold.

    ``full_808`` produces exactly ``C/M/C+M``.  ``ftv_complete_375`` produces
    exactly ``F/C+F/C+F+M``.  Supplying FTV rows to the full estimand or omitting
    them from the FTV estimand is an error, preventing silent population mixing.
    """

    mri = _matrix(mri_features, name="aligned MRI features")
    clinical = _matrix(
        clinical_features, name="aligned clinical features", expected_rows=len(mri)
    )
    y = _binary_labels(labels, name="aligned labels", expected_rows=len(mri))
    if population not in {FULL_POPULATION, FTV_POPULATION}:
        raise ValueError(f"population must be {FULL_POPULATION!r} or {FTV_POPULATION!r}")
    if population == FULL_POPULATION:
        if ftv_features is not None:
            raise ValueError("FTV features may not enter the full_808 estimand")
        ftv = None
    else:
        if ftv_features is None:
            raise ValueError("ftv_complete_375 requires aligned FTV features")
        ftv = _matrix(ftv_features, name="aligned FTV features", expected_rows=len(mri))

    train = _indices(train_indices, name="train_indices", n_rows=len(mri))
    validation = _indices(validation_indices, name="validation_indices", n_rows=len(mri))
    test = _indices(test_indices, name="test_indices", n_rows=len(mri))
    combined = np.concatenate((train, validation, test))
    if len(combined) != len(mri) or len(np.unique(combined)) != len(mri):
        raise ValueError("train/validation/test indices must be disjoint and cover the population")
    if patient_ids is None:
        identifiers = np.asarray([f"row_{index}" for index in range(len(mri))], dtype=object)
    else:
        if isinstance(patient_ids, (str, bytes)) or len(patient_ids) != len(mri):
            raise ValueError(f"patient_ids must contain {len(mri)} aligned identifiers")
        identifiers = np.asarray([str(value) for value in patient_ids], dtype=object)
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("patient_ids must be non-empty and unique")

    def compact(covariates: np.ndarray | None = None) -> SelectedLogisticFit:
        return fit_compact_logistic(
            mri[train],
            y[train],
            mri[validation],
            y[validation],
            mri[test],
            train_covariates=None if covariates is None else covariates[train],
            validation_covariates=None if covariates is None else covariates[validation],
            test_covariates=None if covariates is None else covariates[test],
            dimensions=dimensions,
            c_grid=c_grid,
            class_weight=class_weight,
            random_state=random_state,
        )

    def direct(features: np.ndarray) -> SelectedLogisticFit:
        return fit_l2_logistic(
            features[train],
            y[train],
            features[validation],
            y[validation],
            features[test],
            c_grid=c_grid,
            class_weight=class_weight,
            random_state=random_state,
        )

    if population == FULL_POPULATION:
        fit_map: dict[str, SelectedLogisticFit] = {
            "C": direct(clinical),
            "M": compact(),
            "C+M": compact(clinical),
        }
    else:
        assert ftv is not None
        clinical_ftv = np.column_stack((clinical, ftv))
        fit_map = {
            "F": direct(ftv),
            "C+F": direct(clinical_ftv),
            "C+F+M": compact(clinical_ftv),
        }

    split_rows = (("train", train), ("validation", validation), ("test", test))
    prediction_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for family, fit in fit_map.items():
        probabilities_by_split = (
            fit.train_probabilities,
            fit.validation_probabilities,
            fit.test_probabilities,
        )
        for (split, rows), probabilities in zip(split_rows, probabilities_by_split, strict=True):
            assert probabilities is not None
            for local_index, row_index in enumerate(rows):
                prediction_rows.append(
                    {
                        "patient_id": str(identifiers[row_index]),
                        "fold": int(outer_fold),
                        "population": population,
                        "model_family": family,
                        "split": split,
                        "y_true": int(y[row_index]),
                        "predicted_probability": float(probabilities[local_index]),
                        "selected_dimension": fit.selected_dimension,
                        "selected_C": fit.selected_c,
                    }
                )
            diagnostic_rows.append(
                {
                    "population": population,
                    "fold": int(outer_fold),
                    "model_family": family,
                    "split": split,
                    "selected_dimension": fit.selected_dimension,
                    "selected_C": fit.selected_c,
                    "validation_selection_auroc": fit.validation_auroc,
                    **_split_metrics(y[rows], np.asarray(probabilities)),
                }
            )
    return FoldEvaluation(
        fits=MappingProxyType(fit_map),
        predictions=pd.DataFrame(prediction_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
    )


def aggregate_oof_metrics(
    predictions: pd.DataFrame,
    *,
    group_cols: Sequence[str] | None = None,
    patient_col: str = "patient_id",
    label_col: str = "y_true",
    probability_col: str = "predicted_probability",
    split_col: str = "split",
) -> pd.DataFrame:
    """Aggregate held-out patient predictions without returning patient rows."""

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    required = {patient_col, label_col, probability_col}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions misses required columns: {missing}")
    frame = predictions.copy()
    if split_col in frame.columns:
        frame = frame.loc[frame[split_col].eq("test")].copy()
    if frame.empty:
        raise ValueError("predictions contains no held-out/test rows")
    if group_cols is None:
        group_cols = tuple(
            column
            for column in ("population", "seed", "arm", "timing", "model_family")
            if column in frame.columns
        )
    groups = tuple(group_cols)
    missing_groups = sorted(set(groups) - set(frame.columns))
    if missing_groups:
        raise ValueError(f"predictions misses grouping columns: {missing_groups}")
    iterator = [((), frame)] if not groups else frame.groupby(list(groups), sort=True, dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in iterator:
        key_tuple = key if isinstance(key, tuple) else (key,)
        if group[patient_col].astype(str).duplicated().any():
            raise ValueError("OOF predictions repeat a patient within a model cell")
        metric = binary_metrics(group[label_col].to_numpy(), group[probability_col].to_numpy())
        rows.append({**dict(zip(groups, key_tuple, strict=True)), **metric})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ProfileProbeFit:
    scaler: StandardScaler
    model: LogisticRegression | OneVsRestClassifier
    classes: tuple[Any, ...]
    selected_c: float
    validation_score: float
    score_name: str
    train_probabilities: np.ndarray
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray | None
    feature_dim: int

    def predict_proba(self, features: Any) -> np.ndarray:
        matrix = _matrix(features, name="profile prediction features", expected_features=self.feature_dim)
        return np.asarray(self.model.predict_proba(self.scaler.transform(matrix)), dtype=np.float64)


def _categorical_labels(values: Any, *, name: str, expected_rows: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != expected_rows:
        raise ValueError(f"{name} must have shape ({expected_rows},)")
    if pd.isna(labels).any():
        raise ValueError(f"{name} contains missing values")
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{name} must contain at least two classes")
    return labels


def fit_profile_probe(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    test_features: Any | None = None,
    *,
    c_grid: Iterable[float] = C_GRID,
    class_weight: str | Mapping[Any, float] | None = "balanced",
    max_iter: int = 10_000,
    random_state: int = 0,
) -> ProfileProbeFit:
    """Fit a fold-isolated linear HR/HER2/subtype/treatment decodability probe."""

    train_x = _matrix(train_features, name="profile train features")
    validation_x = _matrix(
        validation_features,
        name="profile validation features",
        expected_features=train_x.shape[1],
    )
    test_x = None if test_features is None else _matrix(
        test_features, name="profile test features", expected_features=train_x.shape[1]
    )
    train_y = _categorical_labels(train_labels, name="profile train labels", expected_rows=len(train_x))
    validation_y = _categorical_labels(
        validation_labels, name="profile validation labels", expected_rows=len(validation_x)
    )
    classes = tuple(np.unique(train_y).tolist())
    if set(np.unique(validation_y).tolist()) != set(classes):
        raise ValueError("profile validation labels must contain every training class")
    grid = _positive_grid(c_grid, name="c_grid")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    candidate_rows: list[
        tuple[float, float, LogisticRegression | OneVsRestClassifier]
    ] = []
    for c_value in grid:
        base_model = LogisticRegression(
            penalty="l2",
            C=c_value,
            solver="liblinear",
            class_weight=class_weight,
            max_iter=_positive_integer(max_iter, name="max_iter"),
            random_state=int(random_state),
        )
        model: LogisticRegression | OneVsRestClassifier = (
            base_model if len(classes) == 2 else OneVsRestClassifier(base_model)
        )
        model.fit(train_scaled, train_y)
        probability = model.predict_proba(validation_scaled)
        if len(classes) == 2:
            score = float(roc_auc_score(validation_y, probability[:, 1]))
        else:
            score = float(
                roc_auc_score(
                    validation_y,
                    probability,
                    labels=np.asarray(classes),
                    multi_class="ovr",
                    average="macro",
                )
            )
        candidate_rows.append((score, c_value, model))
    best = max(row[0] for row in candidate_rows)
    score, c_value, model = min(
        (row for row in candidate_rows if row[0] >= best - SELECTION_ATOL),
        key=lambda row: row[1],
    )
    return ProfileProbeFit(
        scaler=scaler,
        model=model,
        classes=classes,
        selected_c=c_value,
        validation_score=score,
        score_name="auroc" if len(classes) == 2 else "macro_ovr_auroc",
        train_probabilities=np.asarray(model.predict_proba(train_scaled), dtype=np.float64),
        validation_probabilities=np.asarray(model.predict_proba(validation_scaled), dtype=np.float64),
        test_probabilities=(
            None
            if test_x is None
            else np.asarray(model.predict_proba(scaler.transform(test_x)), dtype=np.float64)
        ),
        feature_dim=int(train_x.shape[1]),
    )


def clinical_response_subgroups(hr: Any, her2: Any) -> np.ndarray:
    """Map HR/HER2 labels to the three registered within-profile pCR groups."""

    hr_values = np.asarray(hr)
    her2_values = np.asarray(her2)
    if hr_values.ndim != 1 or her2_values.shape != hr_values.shape or not len(hr_values):
        raise ValueError("hr and her2 must be aligned non-empty one-dimensional vectors")
    for name, values in (("hr", hr_values), ("her2", her2_values)):
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain binary 0/1 labels") from error
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain binary 0/1 labels")
    output = np.full(len(hr_values), "HR-/HER2-", dtype=object)
    output[(hr_values.astype(int) == 1) & (her2_values.astype(int) == 0)] = "HR+/HER2-"
    output[her2_values.astype(int) == 1] = "HER2+"
    return output


def subgroup_metrics(
    labels: Any,
    probabilities: Any,
    subgroups: Any,
    *,
    min_samples: int = 20,
    min_per_class: int = 5,
) -> pd.DataFrame:
    """Return aggregate pCR metrics within groups when sample size permits."""

    probability = np.asarray(probabilities, dtype=np.float64)
    y_raw = np.asarray(labels)
    group_values = np.asarray(subgroups)
    if probability.ndim != 1 or y_raw.shape != probability.shape or group_values.shape != probability.shape:
        raise ValueError("labels, probabilities, and subgroups must be aligned vectors")
    # Validate probabilities and binary values via a permissive local check because
    # an individual subgroup may contain only one class.
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0,1]")
    try:
        y = y_raw.astype(float)
    except (TypeError, ValueError) as error:
        raise TypeError("labels must contain binary 0/1 values") from error
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all() or pd.isna(group_values).any():
        raise ValueError("labels must be binary and subgroups must be non-missing")
    y = y.astype(np.int64)
    required_n = _positive_integer(min_samples, name="min_samples")
    required_class = _positive_integer(min_per_class, name="min_per_class")
    rows: list[dict[str, Any]] = []
    for subgroup in sorted(np.unique(group_values).tolist(), key=str):
        mask = group_values == subgroup
        group_y = y[mask]
        counts = np.bincount(group_y, minlength=2)
        eligible = int(mask.sum()) >= required_n and int(counts.min()) >= required_class
        row: dict[str, Any] = {
            "subgroup": subgroup,
            "n": int(mask.sum()),
            "n_negative": int(counts[0]),
            "n_positive": int(counts[1]),
            "eligible": bool(eligible),
            "status": "ok" if eligible else "insufficient_sample_or_class_count",
        }
        if eligible:
            row.update(binary_metrics(group_y, probability[mask]))
        else:
            row.update(
                {
                    "auroc": math.nan,
                    "auprc": math.nan,
                    "brier": math.nan,
                    "calibration_slope": math.nan,
                    "ece10": math.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_generalization_gaps(
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
) -> dict[str, float]:
    """Compute signed train/validation minus test gaps for registered metrics."""

    output: dict[str, float] = {}
    for metric in ("auroc", "auprc", "brier", "calibration_slope", "ece10"):
        if metric not in train_metrics or metric not in validation_metrics or metric not in test_metrics:
            raise ValueError(f"all split metrics must contain {metric!r}")
        output[f"train_test_{metric}_gap"] = float(train_metrics[metric]) - float(test_metrics[metric])
        output[f"validation_test_{metric}_gap"] = float(validation_metrics[metric]) - float(test_metrics[metric])
    return output


def generalization_gap_table(
    predictions: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("population", "model_family"),
) -> pd.DataFrame:
    """Summarize train/validation/test performance and generalization gaps."""

    required = {"split", "y_true", "predicted_probability", *group_cols}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions misses required columns: {missing}")
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(list(group_cols), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        by_split: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation", "test"):
            split_frame = group.loc[group["split"].eq(split)]
            if split_frame.empty:
                raise ValueError(f"model cell misses required split {split!r}")
            by_split[split] = _split_metrics(
                split_frame["y_true"].to_numpy(),
                split_frame["predicted_probability"].to_numpy(),
            )
        row = dict(zip(group_cols, key_tuple, strict=True))
        for split, values in by_split.items():
            for metric in ("auroc", "auprc", "brier", "calibration_slope", "ece10"):
                row[f"{split}_{metric}"] = values[metric]
        row.update(
            compute_generalization_gaps(
                by_split["train"], by_split["validation"], by_split["test"]
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


generalization_gaps = compute_generalization_gaps


__all__ = [
    "C_GRID",
    "FTV_MODEL_FAMILIES",
    "FTV_POPULATION",
    "FULL_MODEL_FAMILIES",
    "FULL_POPULATION",
    "PCA_DIMENSIONS",
    "CandidateScore",
    "CompactLogisticFit",
    "FoldEvaluation",
    "ProfileProbeFit",
    "SelectedLogisticFit",
    "aggregate_oof_metrics",
    "clinical_response_subgroups",
    "compute_generalization_gaps",
    "evaluate_feature_families",
    "fit_compact_logistic",
    "fit_l2_logistic",
    "fit_profile_probe",
    "generalization_gap_table",
    "generalization_gaps",
    "subgroup_metrics",
]
