"""Leak-resistant compact-representation and strict stacking primitives.

This module deliberately keeps every fitted object behind an explicit
``fit_*`` entry point whose signature contains training arrays only.  It does
not know about outer validation/test arrays, pCR-specific dimension selection,
or experiment output paths.  Those boundaries make accidental preprocessing
leakage harder when the primitives are composed by the audit runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


MAX_PCA_COMPONENTS = 64
DEFAULT_PROBABILITY_CLIP = 1e-6
_UINT32_MAX = 2**32 - 1


def _exact_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not boolean")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < minimum:
        qualifier = "non-negative" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return int(parsed)


def _sklearn_seed(value: Any, *, name: str) -> int:
    seed = _exact_integer(value, name=name)
    if seed > _UINT32_MAX:
        raise ValueError(f"{name} must be <= {_UINT32_MAX} for sklearn")
    return seed


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
    if expected_rows is not None and matrix.shape[0] != int(expected_rows):
        raise ValueError(
            f"{name} has {matrix.shape[0]} rows; expected {int(expected_rows)}"
        )
    if expected_features is not None and matrix.shape[1] != int(expected_features):
        raise ValueError(
            f"{name} has {matrix.shape[1]} features; expected {int(expected_features)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return matrix


def _binary_labels(
    values: Any,
    *,
    name: str,
    expected_rows: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_rows is not None and raw.size != int(expected_rows):
        raise ValueError(f"{name} has {raw.size} rows; expected {int(expected_rows)}")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must use integer 0/1 labels, not boolean labels")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain binary 0/1 labels") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only binary 0/1 labels")
    labels = numeric.astype(np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def _patient_ids(values: Sequence[Any], *, expected_rows: int | None = None) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("patient_ids must be a sequence, not a string")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError("patient_ids must be a finite sequence") from error
    if not raw:
        raise ValueError("patient_ids must not be empty")
    if expected_rows is not None and len(raw) != int(expected_rows):
        raise ValueError(
            f"patient_ids has {len(raw)} rows; expected {int(expected_rows)}"
        )
    parsed: list[str] = []
    for value in raw:
        if value is None or (
            isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
        ):
            raise ValueError("patient_ids contains a missing/non-finite identifier")
        text = str(value)
        if not text or text != text.strip():
            raise ValueError("patient_ids contains a blank or whitespace-padded identifier")
        parsed.append(text)
    if len(parsed) != len(set(parsed)):
        raise ValueError("patient_ids must be unique")
    return tuple(parsed)


def _positive_dimensions(
    values: Iterable[int], *, maximum: int, name: str = "dimensions"
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    parsed = tuple(_exact_integer(value, name=name, minimum=1) for value in raw)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} must not contain duplicates")
    if max(parsed) > int(maximum):
        raise ValueError(f"{name} may not exceed {int(maximum)}")
    return tuple(sorted(parsed))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numeric_parameter_sha256(
    *, kind: str, metadata: Mapping[str, Any], arrays: Mapping[str, Any]
) -> str:
    """Hash named numeric parameters with canonical float/int byte order."""

    digest = hashlib.sha256()
    header = {
        "schema_version": 1,
        "kind": str(kind),
        "metadata": dict(metadata),
        "array_names": sorted(str(name) for name in arrays),
    }
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if np.issubdtype(value.dtype, np.floating):
            canonical = np.ascontiguousarray(value, dtype="<f8")
            dtype_name = "float64_le"
        elif np.issubdtype(value.dtype, np.integer):
            canonical = np.ascontiguousarray(value, dtype="<i8")
            dtype_name = "int64_le"
        else:
            raise TypeError(f"hash parameter {name!r} must be numeric")
        digest.update(str(name).encode("utf-8"))
        digest.update(dtype_name.encode("ascii"))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _freeze_arrays(*arrays: np.ndarray) -> None:
    for array in arrays:
        array.setflags(write=False)


@dataclass(frozen=True)
class TrainOnlyPCA:
    """One centered PCA fitted only on its supplied training matrix."""

    model: PCA
    train_rows: int
    input_dim: int
    max_components: int
    parameter_sha256: str

    @property
    def fitted_transform_sha256(self) -> str:
        return self.parameter_sha256

    def transform(self, features: Any, n_components: int | None = None) -> np.ndarray:
        matrix = _matrix(
            features,
            name="PCA transform features",
            expected_features=self.input_dim,
        )
        dimension = (
            self.max_components
            if n_components is None
            else _exact_integer(n_components, name="n_components", minimum=1)
        )
        if dimension > self.max_components:
            raise ValueError(
                f"n_components may not exceed fitted maximum {self.max_components}"
            )
        transformed = np.asarray(self.model.transform(matrix), dtype=np.float64)
        if transformed.shape != (len(matrix), self.max_components):
            raise RuntimeError("PCA transform returned an unexpected shape")
        output = transformed[:, :dimension].copy()
        if not np.isfinite(output).all():
            raise RuntimeError("PCA transform returned NaN or infinity")
        return output

    def transform_slice(self, features: Any, n_components: int) -> np.ndarray:
        """Explicit alias documenting that candidate Mk uses a leading slice."""

        return self.transform(features, n_components=n_components)

    def component_variance_ledger(self) -> tuple[dict[str, int | float | str], ...]:
        ratio = np.asarray(self.model.explained_variance_ratio_, dtype=np.float64)
        variance = np.asarray(self.model.explained_variance_, dtype=np.float64)
        cumulative = np.cumsum(ratio)
        return tuple(
            {
                "component": int(index + 1),
                "explained_variance": float(variance[index]),
                "explained_variance_ratio": float(ratio[index]),
                "cumulative_explained_variance_ratio": float(cumulative[index]),
                "fitted_transform_sha256": self.parameter_sha256,
            }
            for index in range(self.max_components)
        )

    def variance_ledger(
        self, dimensions: Iterable[int]
    ) -> tuple[dict[str, int | float | str], ...]:
        """Return aggregate explained-variance evidence for candidate dimensions."""

        candidates = _positive_dimensions(
            dimensions, maximum=self.max_components, name="PCA dimensions"
        )
        ratio = np.asarray(self.model.explained_variance_ratio_, dtype=np.float64)
        cumulative = np.cumsum(ratio)
        rows: list[dict[str, int | float | str]] = []
        previous = 0
        for dimension in candidates:
            rows.append(
                {
                    "dimension": dimension,
                    "input_dim": self.input_dim,
                    "max_components": self.max_components,
                    "train_rows": self.train_rows,
                    "component_explained_variance_ratio": float(ratio[dimension - 1]),
                    "incremental_explained_variance_ratio": float(
                        ratio[previous:dimension].sum()
                    ),
                    "cumulative_explained_variance_ratio": float(
                        cumulative[dimension - 1]
                    ),
                    "fitted_transform_sha256": self.parameter_sha256,
                }
            )
            previous = dimension
        return tuple(rows)


def fit_train_pca(
    train_features: Any,
    *,
    max_components: int = MAX_PCA_COMPONENTS,
    svd_solver: str = "full",
    whiten: bool = False,
) -> TrainOnlyPCA:
    """Fit one centered PCA using training rows only.

    The centered rank bound ``N-1`` is enforced so the fitted maximum does not
    include a structurally null component.  The formal audit supplies more
    than 64 outer-train patients at every population/timing.
    """

    train = _matrix(train_features, name="PCA train features")
    components = _exact_integer(
        max_components, name="max_components", minimum=1
    )
    if components > MAX_PCA_COMPONENTS:
        raise ValueError(f"max_components may not exceed {MAX_PCA_COMPONENTS}")
    rank_bound = min(train.shape[1], train.shape[0] - 1)
    if components > rank_bound:
        raise ValueError(
            "max_components exceeds centered training rank bound "
            f"min(F,N-1)={rank_bound}"
        )
    if svd_solver != "full":
        raise ValueError("formal PCA requires svd_solver='full'")
    if not isinstance(whiten, (bool, np.bool_)):
        raise TypeError("whiten must be boolean")
    if bool(whiten):
        raise ValueError("formal PCA requires whiten=False")
    centered = train - train.mean(axis=0, keepdims=True)
    if not np.any(centered != 0.0):
        raise ValueError("PCA train features have zero total variance")

    model = PCA(n_components=components, svd_solver="full", whiten=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(train)
    fitted_arrays = (
        model.mean_,
        model.components_,
        model.explained_variance_,
        model.explained_variance_ratio_,
        model.singular_values_,
    )
    if any(not np.isfinite(np.asarray(value)).all() for value in fitted_arrays):
        raise RuntimeError("PCA fitted parameters contain NaN or infinity")
    parameter_hash = _numeric_parameter_sha256(
        kind="centered_train_only_pca",
        metadata={
            "train_rows": int(train.shape[0]),
            "input_dim": int(train.shape[1]),
            "max_components": components,
            "svd_solver": "full",
            "whiten": False,
            "noise_variance": float(model.noise_variance_),
        },
        arrays={
            "mean": model.mean_,
            "components": model.components_,
            "explained_variance": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "singular_values": model.singular_values_,
        },
    )
    _freeze_arrays(*fitted_arrays)
    return TrainOnlyPCA(
        model=model,
        train_rows=int(train.shape[0]),
        input_dim=int(train.shape[1]),
        max_components=components,
        parameter_sha256=parameter_hash,
    )


# Readable aliases for orchestration code and ledgers.
fit_train_only_pca = fit_train_pca


@dataclass(frozen=True)
class GaussianRandomProjection:
    """A label- and patient-independent Gaussian projection matrix."""

    matrix: np.ndarray
    input_dim: int
    output_dim: int
    seed: int
    matrix_sha256: str
    distribution: str = "gaussian_N_0_1_over_sqrt_k"

    @property
    def parameter_sha256(self) -> str:
        return self.matrix_sha256

    def transform(self, features: Any) -> np.ndarray:
        values = _matrix(
            features,
            name="random-projection features",
            expected_features=self.input_dim,
        )
        projected = np.asarray(values @ self.matrix, dtype=np.float64)
        if projected.shape != (len(values), self.output_dim):
            raise RuntimeError("random projection returned an unexpected shape")
        if not np.isfinite(projected).all():
            raise RuntimeError("random projection returned NaN or infinity")
        return projected

    def ledger_record(self) -> dict[str, int | str]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "seed": self.seed,
            "distribution": self.distribution,
            "matrix_sha256": self.matrix_sha256,
        }


def make_gaussian_random_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
) -> GaussianRandomProjection:
    """Create ``N(0, 1/k)`` weights from dimensions and a seed only."""

    source_dim = _exact_integer(input_dim, name="input_dim", minimum=1)
    target_dim = _exact_integer(output_dim, name="output_dim", minimum=1)
    random_seed = _exact_integer(seed, name="seed")
    rng = np.random.default_rng(random_seed)
    matrix = rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(target_dim),
        size=(source_dim, target_dim),
    ).astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("generated random-projection matrix is non-finite")
    matrix_hash = _numeric_parameter_sha256(
        kind="gaussian_random_projection",
        metadata={
            "input_dim": source_dim,
            "output_dim": target_dim,
            "seed": random_seed,
            "distribution": "gaussian_N_0_1_over_sqrt_k",
            "bit_generator": type(rng.bit_generator).__name__,
        },
        arrays={"projection_matrix": matrix},
    )
    _freeze_arrays(matrix)
    return GaussianRandomProjection(
        matrix=matrix,
        input_dim=source_dim,
        output_dim=target_dim,
        seed=random_seed,
        matrix_sha256=matrix_hash,
    )


gaussian_random_projection = make_gaussian_random_projection


def _validated_class_weight(
    value: str | Mapping[int, float] | None,
) -> str | dict[int, float] | None:
    if value is None:
        return value
    if isinstance(value, str):
        if value == "balanced":
            return value
        raise ValueError(
            "class_weight must be None, 'balanced', or a non-empty mapping"
        )
    if not isinstance(value, Mapping) or not value:
        raise ValueError("class_weight must be None, 'balanced', or a non-empty mapping")
    output: dict[int, float] = {}
    for key, weight in value.items():
        try:
            parsed_key = _exact_integer(key, name="class_weight key")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "class_weight keys must be binary integers 0 and/or 1"
            ) from error
        if parsed_key not in (0, 1):
            raise ValueError("class_weight keys must be binary integers 0 and/or 1")
        if isinstance(weight, (bool, np.bool_)):
            raise ValueError("class_weight values must be finite and positive")
        try:
            numeric = float(weight)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "class_weight values must be finite and positive"
            ) from error
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("class_weight values must be finite and positive")
        output[parsed_key] = numeric
    return output


@dataclass(frozen=True)
class FixedCLogisticFit:
    """Train-standardized binary L2 logistic fit with a preselected C."""

    scaler: StandardScaler
    model: LogisticRegression
    c_value: float
    feature_dim: int
    train_rows: int
    parameter_sha256: str

    def predict_proba(self, features: Any) -> np.ndarray:
        values = _matrix(
            features,
            name="fixed-C logistic prediction features",
            expected_features=self.feature_dim,
        )
        probability = np.asarray(
            self.model.predict_proba(self.scaler.transform(values))[:, 1],
            dtype=np.float64,
        )
        if probability.shape != (len(values),) or not np.isfinite(probability).all():
            raise RuntimeError("fixed-C logistic returned invalid probabilities")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise RuntimeError("fixed-C logistic returned probabilities outside [0,1]")
        return probability

    def decision_function(self, features: Any) -> np.ndarray:
        values = _matrix(
            features,
            name="fixed-C logistic decision features",
            expected_features=self.feature_dim,
        )
        decision = np.asarray(
            self.model.decision_function(self.scaler.transform(values)),
            dtype=np.float64,
        ).reshape(-1)
        if decision.shape != (len(values),) or not np.isfinite(decision).all():
            raise RuntimeError("fixed-C logistic returned invalid decision values")
        return decision

    def predict_logit(
        self, features: Any, *, clip: float = DEFAULT_PROBABILITY_CLIP
    ) -> np.ndarray:
        return probability_to_logit(self.predict_proba(features), clip=clip)


def fit_fixed_c_logistic(
    train_features: Any,
    train_labels: Any,
    c_value: float,
    *,
    class_weight: str | Mapping[int, float] | None = None,
    solver: str = "liblinear",
    max_iter: int = 10_000,
    random_state: int = 0,
) -> FixedCLogisticFit:
    """Fit scaler and L2 logistic coefficients on the supplied train rows only."""

    train = _matrix(train_features, name="fixed-C logistic train features")
    labels = _binary_labels(
        train_labels,
        name="fixed-C logistic train labels",
        expected_rows=len(train),
    )
    if isinstance(c_value, (bool, np.bool_)):
        raise TypeError("c_value must be numeric, not boolean")
    c_numeric = float(c_value)
    if not math.isfinite(c_numeric) or c_numeric <= 0.0:
        raise ValueError("c_value must be finite and positive")
    iterations = _exact_integer(max_iter, name="max_iter", minimum=1)
    seed = _sklearn_seed(random_state, name="random_state")
    if solver != "liblinear":
        raise ValueError("formal fixed-C logistic requires solver='liblinear'")
    weights = _validated_class_weight(class_weight)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(train)
    model = LogisticRegression(
        penalty="l2",
        C=c_numeric,
        solver="liblinear",
        class_weight=weights,
        max_iter=iterations,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaled, labels)
    fitted_arrays = (
        scaler.mean_,
        scaler.var_,
        scaler.scale_,
        model.coef_,
        model.intercept_,
        model.classes_,
        model.n_iter_,
    )
    if any(not np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in fitted_arrays):
        raise RuntimeError("fixed-C logistic fitted parameters are non-finite")
    weight_metadata: str | dict[str, float] | None
    if isinstance(weights, Mapping):
        weight_metadata = {str(key): float(weights[key]) for key in sorted(weights)}
    else:
        weight_metadata = weights
    parameter_hash = _numeric_parameter_sha256(
        kind="train_only_standardized_l2_logistic",
        metadata={
            "train_rows": int(train.shape[0]),
            "feature_dim": int(train.shape[1]),
            "c_value": c_numeric,
            "class_weight": weight_metadata,
            "solver": "liblinear",
            "max_iter": iterations,
            "random_state": seed,
        },
        arrays={
            "scaler_mean": scaler.mean_,
            "scaler_var": scaler.var_,
            "scaler_scale": scaler.scale_,
            "model_coef": model.coef_,
            "model_intercept": model.intercept_,
            "model_classes": model.classes_,
            "model_n_iter": model.n_iter_,
        },
    )
    _freeze_arrays(*fitted_arrays)
    return FixedCLogisticFit(
        scaler=scaler,
        model=model,
        c_value=c_numeric,
        feature_dim=int(train.shape[1]),
        train_rows=int(train.shape[0]),
        parameter_sha256=parameter_hash,
    )


fit_fixed_c_binary_logistic = fit_fixed_c_logistic


def probability_to_logit(
    probabilities: Any, *, clip: float = DEFAULT_PROBABILITY_CLIP
) -> np.ndarray:
    """Convert probabilities to finite logits using a symmetric fixed clip."""

    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("probabilities must be numeric") from error
    if values.size == 0 or values.ndim == 0:
        raise ValueError("probabilities must be a non-empty array")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0,1]")
    if isinstance(clip, (bool, np.bool_)):
        raise TypeError("clip must be numeric, not boolean")
    clip_value = float(clip)
    if not math.isfinite(clip_value) or not 0.0 < clip_value < 0.5:
        raise ValueError("clip must be finite and strictly between 0 and 0.5")
    bounded = np.clip(values, clip_value, 1.0 - clip_value)
    logits = np.log(bounded) - np.log1p(-bounded)
    if not np.isfinite(logits).all():
        raise RuntimeError("probability-to-logit conversion returned non-finite values")
    return logits


probabilities_to_logits = probability_to_logit


@dataclass(frozen=True)
class InnerFoldAssignments:
    """Stable patient-keyed inner folds, represented in caller row order."""

    patient_ids: tuple[str, ...]
    fold_by_row: np.ndarray
    n_splits: int
    seed: int
    assignment_sha256: str

    def indices(self, inner_fold: int) -> tuple[np.ndarray, np.ndarray]:
        fold = _exact_integer(inner_fold, name="inner_fold")
        if fold >= self.n_splits:
            raise ValueError(f"inner_fold must be in 0..{self.n_splits - 1}")
        validation = np.flatnonzero(self.fold_by_row == fold)
        train = np.flatnonzero(self.fold_by_row != fold)
        if not len(train) or not len(validation):
            raise RuntimeError("inner-fold assignment produced an empty partition")
        # A stable patient-ID order makes downstream fits invariant to caller
        # row order as well as to the assignment generation itself.
        train = train[
            np.argsort(np.asarray(self.patient_ids, dtype=str)[train], kind="stable")
        ]
        validation = validation[
            np.argsort(
                np.asarray(self.patient_ids, dtype=str)[validation], kind="stable"
            )
        ]
        return train, validation

    def fold_for(self, patient_id: Any) -> int:
        text = str(patient_id)
        try:
            index = self.patient_ids.index(text)
        except ValueError as error:
            raise KeyError(f"unknown patient_id: {text}") from error
        return int(self.fold_by_row[index])

    def records(self) -> tuple[dict[str, int | str], ...]:
        return tuple(
            {"patient_id": patient_id, "inner_fold": int(self.fold_by_row[index])}
            for index, patient_id in enumerate(self.patient_ids)
        )


def stratified_inner_assignments(
    patient_ids: Sequence[Any],
    labels: Any,
    *,
    n_splits: int = 5,
    seed: int = 260_813,
) -> InnerFoldAssignments:
    """Assign deterministic StratifiedKFold IDs after sorting by patient ID.

    Sorting before invoking sklearn makes the patient->fold mapping invariant to
    input row order.  The returned array is mapped back to the caller's row
    order for direct indexing of aligned feature matrices.
    """

    ids = _patient_ids(patient_ids)
    y = _binary_labels(labels, name="inner-fold labels", expected_rows=len(ids))
    splits = _exact_integer(n_splits, name="n_splits", minimum=2)
    random_seed = _sklearn_seed(seed, name="seed")
    counts = np.bincount(y, minlength=2)
    if int(counts.min()) < splits:
        raise ValueError(
            "each class must contain at least n_splits patients for stratification"
        )

    order = np.argsort(np.asarray(ids, dtype=str), kind="stable")
    sorted_ids = np.asarray(ids, dtype=str)[order]
    sorted_labels = y[order]
    sorted_fold = np.full(len(ids), -1, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=random_seed,
    )
    for inner_fold, (_, validation) in enumerate(
        splitter.split(np.zeros((len(ids), 1), dtype=np.uint8), sorted_labels)
    ):
        if np.any(sorted_fold[validation] != -1):
            raise AssertionError("inner validation rows were assigned more than once")
        sorted_fold[validation] = inner_fold
    if np.any(sorted_fold < 0) or set(np.unique(sorted_fold)) != set(range(splits)):
        raise AssertionError("inner-fold assignment does not cover every patient exactly once")
    fold_by_row = np.empty(len(ids), dtype=np.int64)
    fold_by_row[order] = sorted_fold

    records = [
        {
            "patient_id": str(sorted_ids[index]),
            "label": int(sorted_labels[index]),
            "inner_fold": int(sorted_fold[index]),
        }
        for index in range(len(ids))
    ]
    assignment_hash = _canonical_sha256(
        {
            "schema_version": 1,
            "kind": "patient_id_sorted_stratified_kfold",
            "n_splits": splits,
            "seed": random_seed,
            "records": records,
        }
    )
    _freeze_arrays(fold_by_row)
    return InnerFoldAssignments(
        patient_ids=ids,
        fold_by_row=fold_by_row,
        n_splits=splits,
        seed=random_seed,
        assignment_sha256=assignment_hash,
    )


stable_patient_stratified_folds = stratified_inner_assignments


@dataclass(frozen=True)
class StrictInnerOOFResult:
    """Exactly one held-out probability for each outer-train patient."""

    patient_ids: tuple[str, ...]
    probabilities: np.ndarray
    inner_fold: np.ndarray
    assignment_sha256: str
    prediction_sha256: str
    fold_fit_sha256: tuple[str, ...]

    @property
    def logits(self) -> np.ndarray:
        return probability_to_logit(self.probabilities)

    def records(self) -> tuple[dict[str, int | float | str], ...]:
        return tuple(
            {
                "patient_id": patient_id,
                "inner_fold": int(self.inner_fold[index]),
                "predicted_probability": float(self.probabilities[index]),
            }
            for index, patient_id in enumerate(self.patient_ids)
        )


InnerFitPredict = Callable[[np.ndarray, np.ndarray], Any]


def strict_inner_oof_probabilities(
    assignments: InnerFoldAssignments,
    fit_predict: InnerFitPredict,
    *,
    fold_fit_sha256: Sequence[str] | None = None,
) -> StrictInnerOOFResult:
    """Run a caller-supplied train/holdout callback once per inner fold.

    ``fit_predict`` receives disjoint, patient-ID-sorted integer indices and
    must return one probability for every holdout index.  This generic boundary
    lets the runner refit clinical encoders or PCA inside each inner fold while
    this helper enforces disjointness and exact OOF coverage.
    """

    if not isinstance(assignments, InnerFoldAssignments):
        raise TypeError("assignments must be an InnerFoldAssignments object")
    if not callable(fit_predict):
        raise TypeError("fit_predict must be callable")
    probabilities = np.full(len(assignments.patient_ids), np.nan, dtype=np.float64)
    coverage = np.zeros(len(assignments.patient_ids), dtype=np.int64)
    for inner_fold in range(assignments.n_splits):
        train, validation = assignments.indices(inner_fold)
        if np.intersect1d(train, validation).size:
            raise AssertionError("inner train and validation indices overlap")
        raw = fit_predict(train.copy(), validation.copy())
        try:
            predicted = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError("fit_predict must return numeric probabilities") from error
        if predicted.shape != (len(validation),):
            raise ValueError(
                "fit_predict returned shape "
                f"{predicted.shape}; expected {(len(validation),)}"
            )
        if not np.isfinite(predicted).all() or np.any(
            (predicted < 0.0) | (predicted > 1.0)
        ):
            raise ValueError("fit_predict returned invalid probabilities")
        if np.any(coverage[validation] != 0):
            raise AssertionError("an inner-OOF patient was predicted more than once")
        probabilities[validation] = predicted
        coverage[validation] += 1
    if not np.all(coverage == 1) or not np.isfinite(probabilities).all():
        raise AssertionError("strict inner-OOF predictions do not cover each patient once")

    if fold_fit_sha256 is None:
        fit_hashes: tuple[str, ...] = ()
    else:
        fit_hashes = tuple(str(value) for value in fold_fit_sha256)
        if len(fit_hashes) != assignments.n_splits:
            raise ValueError("fold_fit_sha256 must have one digest per inner fold")
        if any(
            len(value) != 64
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value)
            for value in fit_hashes
        ):
            raise ValueError("fold_fit_sha256 contains a malformed SHA-256 digest")

    canonical_records = sorted(
        (
            patient_id,
            int(assignments.fold_by_row[index]),
            float(probabilities[index]).hex(),
        )
        for index, patient_id in enumerate(assignments.patient_ids)
    )
    prediction_hash = _canonical_sha256(
        {
            "schema_version": 1,
            "kind": "strict_inner_oof_probabilities",
            "assignment_sha256": assignments.assignment_sha256,
            "records": canonical_records,
            "fold_fit_sha256": list(fit_hashes),
        }
    )
    inner_fold = assignments.fold_by_row.copy()
    _freeze_arrays(probabilities, inner_fold)
    return StrictInnerOOFResult(
        patient_ids=assignments.patient_ids,
        probabilities=probabilities,
        inner_fold=inner_fold,
        assignment_sha256=assignments.assignment_sha256,
        prediction_sha256=prediction_hash,
        fold_fit_sha256=fit_hashes,
    )


def fixed_c_inner_oof_probabilities(
    features: Any,
    labels: Any,
    patient_ids: Sequence[Any],
    c_value: float,
    *,
    n_splits: int = 5,
    seed: int = 260_813,
    class_weight: str | Mapping[int, float] | None = None,
    max_iter: int = 10_000,
    random_state: int = 0,
) -> StrictInnerOOFResult:
    """Convenience strict-OOF path for already materialized base features."""

    matrix = _matrix(features, name="inner-OOF features")
    y = _binary_labels(labels, name="inner-OOF labels", expected_rows=len(matrix))
    ids = _patient_ids(patient_ids, expected_rows=len(matrix))
    assignments = stratified_inner_assignments(
        ids, y, n_splits=n_splits, seed=seed
    )
    fit_hashes: list[str] = []

    def fit_predict(train: np.ndarray, validation: np.ndarray) -> np.ndarray:
        fit = fit_fixed_c_logistic(
            matrix[train],
            y[train],
            c_value,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        fit_hashes.append(fit.parameter_sha256)
        return fit.predict_proba(matrix[validation])

    result = strict_inner_oof_probabilities(assignments, fit_predict)
    # Rebuild once with hashes included in the provenance digest; predictions
    # are already frozen and no model is refit.
    return _strict_result_with_fit_hashes(result, tuple(fit_hashes))


def _strict_result_with_fit_hashes(
    result: StrictInnerOOFResult, fit_hashes: tuple[str, ...]
) -> StrictInnerOOFResult:
    if len(fit_hashes) != len(set(result.inner_fold.tolist())):
        raise AssertionError("fixed-C inner-OOF fit hash coverage drifted")
    records = sorted(
        (
            patient_id,
            int(result.inner_fold[index]),
            float(result.probabilities[index]).hex(),
        )
        for index, patient_id in enumerate(result.patient_ids)
    )
    prediction_hash = _canonical_sha256(
        {
            "schema_version": 1,
            "kind": "strict_inner_oof_probabilities",
            "assignment_sha256": result.assignment_sha256,
            "records": records,
            "fold_fit_sha256": list(fit_hashes),
        }
    )
    return StrictInnerOOFResult(
        patient_ids=result.patient_ids,
        probabilities=result.probabilities,
        inner_fold=result.inner_fold,
        assignment_sha256=result.assignment_sha256,
        prediction_sha256=prediction_hash,
        fold_fit_sha256=fit_hashes,
    )


__all__ = [
    "DEFAULT_PROBABILITY_CLIP",
    "FixedCLogisticFit",
    "GaussianRandomProjection",
    "InnerFoldAssignments",
    "MAX_PCA_COMPONENTS",
    "StrictInnerOOFResult",
    "TrainOnlyPCA",
    "fit_fixed_c_binary_logistic",
    "fit_fixed_c_logistic",
    "fit_train_only_pca",
    "fit_train_pca",
    "fixed_c_inner_oof_probabilities",
    "gaussian_random_projection",
    "make_gaussian_random_projection",
    "probabilities_to_logits",
    "probability_to_logit",
    "stable_patient_stratified_folds",
    "stratified_inner_assignments",
    "strict_inner_oof_probabilities",
]
