"""Outcome-free diagnostics for response and phenotype representations.

The functions in this module accept NumPy/scikit-learn style array-likes and
never inspect labels.  Statistics are fitted on *all rows passed to a
function*.  In particular, canonical correlations are descriptive in-sample
statistics: callers must pass training-fold rows only when the result informs
model selection, and must not tune regularization or ``top_k`` against held-out
pCR outcomes.  The function does not turn an in-sample canonical correlation
into an out-of-sample performance estimate.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_float_matrix(value: ArrayLike, name: str) -> FloatArray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from error
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one dimension")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def _validate_ddof(ddof: int, n_samples: int) -> int:
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(
        ddof, (int, np.integer)
    ):
        raise TypeError("ddof must be an integer")
    value = int(ddof)
    if value < 0 or value >= n_samples:
        raise ValueError(f"ddof must satisfy 0 <= ddof < n_samples ({n_samples})")
    return value


def _validate_nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _scaled_center(
    matrix: FloatArray, *, per_dimension: bool
) -> tuple[FloatArray, int | NDArray[np.int64]]:
    """Center after exact power-of-two scaling to avoid finite-input overflow."""

    if per_dimension:
        maxima = np.max(np.abs(matrix), axis=0)
        exponents = np.asarray(np.frexp(maxima)[1], dtype=np.int64)
        scaled = np.ldexp(matrix, -exponents[None, :])
        shifted = scaled - scaled[0:1]
        return shifted - shifted.mean(axis=0, keepdims=True), exponents
    maximum = float(np.max(np.abs(matrix)))
    exponent = int(math.frexp(maximum)[1]) if maximum else 0
    scaled = np.ldexp(matrix, -exponent)
    shifted = scaled - scaled[0:1]
    return shifted - shifted.mean(axis=0, keepdims=True), exponent


def _rescale_power_of_two(
    value: FloatArray, exponent: int | NDArray[np.int64], name: str
) -> FloatArray:
    """Exactly undo binary scaling, rejecting unrepresentable float64 output."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        output = np.ldexp(value, exponent)
    if not np.isfinite(output).all():
        raise OverflowError(
            f"{name} exceeds float64 range; rescale the representation before "
            "requesting this raw-scale statistic"
        )
    return np.asarray(output, dtype=np.float64)


def _stable_nonnegative_mean(value: FloatArray) -> float:
    maximum = float(np.max(value))
    if maximum == 0.0:
        return 0.0
    return float(maximum * np.mean(value / maximum))


def _scaled_covariance_eigenspectrum(
    matrix: FloatArray, ddof: int
) -> tuple[FloatArray, int]:
    centered, exponent = _scaled_center(matrix, per_dimension=False)
    covariance = centered.T @ centered / float(matrix.shape[0] - ddof)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(1, matrix.shape[1]) * scale
    if np.any(eigenvalues < -tolerance):  # pragma: no cover - LAPACK failure
        raise np.linalg.LinAlgError("computed covariance is not positive semidefinite")
    # Clip only negative eigensolver drift; effective/numerical rank apply their
    # own explicit relative tolerance without altering the reported spectrum.
    return np.maximum(eigenvalues, 0.0), int(exponent)


def per_dimension_variance_summary(
    representation: ArrayLike,
    *,
    ddof: int = 1,
    collapse_threshold: float = 1e-12,
) -> dict[str, Any]:
    """Return per-dimension variance/std and compact collapse summaries.

    ``collapse_threshold`` is applied to variance (not standard deviation).
    Constant inputs return exact finite zeros.  Sample variance (``ddof=1``)
    is the default, so at least two observations are required by default.
    """

    matrix = _as_float_matrix(representation, "representation")
    degrees = _validate_ddof(ddof, matrix.shape[0])
    threshold = _validate_nonnegative_finite(
        collapse_threshold, "collapse_threshold"
    )
    centered, exponents = _scaled_center(matrix, per_dimension=True)
    scaled_variance = np.sum(centered * centered, axis=0) / float(
        matrix.shape[0] - degrees
    )
    scaled_variance = np.maximum(scaled_variance, 0.0)
    variance = _rescale_power_of_two(
        scaled_variance, 2 * exponents, "per-dimension variance"
    )
    standard_deviation = _rescale_power_of_two(
        np.sqrt(scaled_variance), exponents, "per-dimension standard deviation"
    )
    collapsed = variance <= threshold
    return {
        "n_samples": int(matrix.shape[0]),
        "n_dimensions": int(matrix.shape[1]),
        "ddof": degrees,
        "collapse_threshold": threshold,
        "variance": variance,
        "std": standard_deviation,
        "mean_variance": _stable_nonnegative_mean(variance),
        "median_variance": float(np.median(variance)),
        "min_variance": float(np.min(variance)),
        "max_variance": float(np.max(variance)),
        "mean_std": _stable_nonnegative_mean(standard_deviation),
        "collapsed_mask": collapsed,
        "collapsed_dimensions": int(np.count_nonzero(collapsed)),
        "noncollapsed_dimensions": int(np.count_nonzero(~collapsed)),
        "collapsed_fraction": float(np.mean(collapsed)),
    }


# A concise alias for report code that already names this a variance/std summary.
variance_std_summary = per_dimension_variance_summary


def covariance_eigenspectrum(
    representation: ArrayLike, *, ddof: int = 1
) -> FloatArray:
    """Return covariance eigenvalues in descending order.

    An exact power-of-two rescaling precedes the centered cross-product, which
    avoids ``np.cov`` scalar special cases and intermediate overflow.  Negative
    round-off eigenvalues are clipped to zero, making constant/rank-deficient
    inputs safe for downstream entropy calculations.
    """

    matrix = _as_float_matrix(representation, "representation")
    degrees = _validate_ddof(ddof, matrix.shape[0])
    scaled_eigenvalues, exponent = _scaled_covariance_eigenspectrum(matrix, degrees)
    return _rescale_power_of_two(
        scaled_eigenvalues, 2 * exponent, "covariance eigenspectrum"
    )


def effective_rank(
    eigenvalues_or_representation: ArrayLike,
    *,
    ddof: int = 1,
    relative_tolerance: float | None = None,
) -> float:
    """Compute entropy effective rank, ``exp(-sum(p * log(p)))``.

    A one-dimensional input is interpreted as covariance eigenvalues.  A
    two-dimensional input is interpreted as a sample-by-dimension
    representation and its covariance spectrum is computed first.  Eigenvalues
    below a scale-relative numerical tolerance are treated as zero.  A zero
    spectrum has effective rank 0.0 (there is no represented direction), while
    a nonzero rank-one spectrum has effective rank 1.0.
    """

    values = np.asarray(eigenvalues_or_representation, dtype=np.float64)
    if values.ndim == 2:
        matrix = _as_float_matrix(values, "representation")
        degrees = _validate_ddof(ddof, matrix.shape[0])
        spectrum, _ = _scaled_covariance_eigenspectrum(matrix, degrees)
    elif values.ndim == 1 and values.size:
        spectrum = values
    else:
        raise ValueError(
            "effective_rank requires a nonempty eigenvalue vector or "
            "sample-by-dimension matrix"
        )
    if not np.isfinite(spectrum).all():
        raise ValueError("eigenvalues contain NaN or infinite values")
    scale = float(np.max(np.abs(spectrum)))
    if relative_tolerance is None:
        relative = np.finfo(np.float64).eps * max(1, spectrum.size)
    else:
        relative = _validate_nonnegative_finite(
            relative_tolerance, "relative_tolerance"
        )
    if scale == 0.0:
        return 0.0
    normalized = spectrum / scale
    if np.any(normalized < -relative):
        raise ValueError("eigenvalues must be nonnegative apart from round-off")
    cleaned = np.where(normalized > relative, normalized, 0.0)
    total = float(np.sum(cleaned))
    if total == 0.0:
        return 0.0
    probabilities = cleaned[cleaned > 0.0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def covariance_spectrum_summary(
    representation: ArrayLike,
    *,
    ddof: int = 1,
    relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Return covariance spectrum, explained fractions, and effective rank."""

    matrix = _as_float_matrix(representation, "representation")
    degrees = _validate_ddof(ddof, matrix.shape[0])
    scaled_eigenvalues, exponent = _scaled_covariance_eigenspectrum(matrix, degrees)
    eigenvalues = _rescale_power_of_two(
        scaled_eigenvalues, 2 * exponent, "covariance eigenspectrum"
    )
    scaled_total = float(np.sum(scaled_eigenvalues))
    explained = (
        scaled_eigenvalues / scaled_total
        if scaled_total > 0.0
        else np.zeros_like(scaled_eigenvalues)
    )
    trace = float(
        _rescale_power_of_two(
            np.asarray(scaled_total), 2 * exponent, "covariance trace"
        )
    )
    scale = float(scaled_eigenvalues[0]) if scaled_eigenvalues.size else 0.0
    relative = (
        np.finfo(np.float64).eps * max(1, eigenvalues.size)
        if relative_tolerance is None
        else _validate_nonnegative_finite(relative_tolerance, "relative_tolerance")
    )
    threshold = relative * scale
    return {
        "n_samples": int(matrix.shape[0]),
        "n_dimensions": int(matrix.shape[1]),
        "ddof": degrees,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained,
        "trace": trace,
        "effective_rank": effective_rank(
            scaled_eigenvalues, relative_tolerance=relative_tolerance
        ),
        "numerical_rank": int(np.count_nonzero(scaled_eigenvalues > threshold)),
        "internal_power_of_two_exponent": exponent,
    }


# Both spellings occur in representation reports; keep one implementation.
covariance_eigenspectrum_summary = covariance_spectrum_summary


def _standardize(matrix: FloatArray, ddof: int) -> tuple[FloatArray, FloatArray]:
    centered, _ = _scaled_center(matrix, per_dimension=True)
    variance = np.sum(centered * centered, axis=0) / float(
        matrix.shape[0] - ddof
    )
    variance = np.maximum(variance, 0.0)
    scale = np.sqrt(variance)
    standardized = np.zeros_like(centered)
    nonconstant = scale > 0.0
    standardized[:, nonconstant] = centered[:, nonconstant] / scale[nonconstant]
    return standardized, nonconstant


def cross_covariance(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    ddof: int = 1,
) -> FloatArray:
    """Return the raw, centered response/phenotype sample cross-covariance.

    Rows must already be paired.  Internal power-of-two scaling prevents
    intermediate overflow and is undone exactly.  If the mathematically raw
    covariance itself cannot fit in float64, an ``OverflowError`` asks the
    caller to rescale the representations rather than silently returning Inf.
    """

    response = _as_float_matrix(z_response, "z_response")
    phenotype = _as_float_matrix(z_phenotype, "z_phenotype")
    if response.shape[0] != phenotype.shape[0]:
        raise ValueError("z_response and z_phenotype must have the same row count")
    degrees = _validate_ddof(ddof, response.shape[0])
    centered_response, response_exponent = _scaled_center(
        response, per_dimension=True
    )
    centered_phenotype, phenotype_exponent = _scaled_center(
        phenotype, per_dimension=True
    )
    response_exponents = np.asarray(response_exponent, dtype=np.int64)
    phenotype_exponents = np.asarray(phenotype_exponent, dtype=np.int64)
    scaled = centered_response.T @ centered_phenotype / float(
        response.shape[0] - degrees
    )
    return _rescale_power_of_two(
        scaled,
        response_exponents[:, None] + phenotype_exponents[None, :],
        "raw cross-covariance",
    )


def _stable_frobenius_norm(matrix: FloatArray, name: str) -> float:
    maximum = float(np.max(np.abs(matrix)))
    if maximum == 0.0:
        return 0.0
    scaled_norm = float(np.sqrt(np.sum((matrix / maximum) ** 2)))
    if scaled_norm > np.finfo(np.float64).max / maximum:
        raise OverflowError(
            f"{name} exceeds float64 range; rescale the representation first"
        )
    return maximum * scaled_norm


def cross_covariance_norm(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    ddof: int = 1,
) -> float:
    """Return ``||Cov(z_response, z_phenotype)||_F`` on the raw feature scale."""

    covariance = cross_covariance(z_response, z_phenotype, ddof=ddof)
    return _stable_frobenius_norm(covariance, "raw cross-covariance norm")


def cross_covariance_summary(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    ddof: int = 1,
) -> dict[str, Any]:
    """Summarize raw cross-covariance and its Frobenius/mean-square norms."""

    response = _as_float_matrix(z_response, "z_response")
    phenotype = _as_float_matrix(z_phenotype, "z_phenotype")
    if response.shape[0] != phenotype.shape[0]:
        raise ValueError("z_response and z_phenotype must have the same row count")
    degrees = _validate_ddof(ddof, response.shape[0])
    covariance = cross_covariance(response, phenotype, ddof=degrees)
    frobenius = _stable_frobenius_norm(
        covariance, "raw cross-covariance norm"
    )
    if frobenius > math.sqrt(np.finfo(np.float64).max):
        raise OverflowError(
            "raw squared cross-covariance norm exceeds float64 range; "
            "rescale the representation first"
        )
    squared_frobenius = frobenius * frobenius
    mean_square = squared_frobenius / float(covariance.size)
    return {
        "n_samples": int(response.shape[0]),
        "response_dimensions": int(response.shape[1]),
        "phenotype_dimensions": int(phenotype.shape[1]),
        "ddof": degrees,
        "standardized": False,
        "cross_covariance": covariance,
        "frobenius_norm": frobenius,
        "squared_frobenius_norm": squared_frobenius,
        "mean_squared_norm": mean_square,
        "root_mean_squared_norm": float(math.sqrt(mean_square)),
    }


def standardized_cross_covariance(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    ddof: int = 1,
) -> FloatArray:
    """Return the feature-standardized response/phenotype cross-covariance.

    Each block is centered and divided by its own per-dimension standard
    deviation on the supplied cohort.  A constant dimension is mapped to zero
    instead of producing NaN/Inf.  Rows are assumed to be paired and already
    in the same patient/visit order.
    """

    response = _as_float_matrix(z_response, "z_response")
    phenotype = _as_float_matrix(z_phenotype, "z_phenotype")
    if response.shape[0] != phenotype.shape[0]:
        raise ValueError("z_response and z_phenotype must have the same row count")
    degrees = _validate_ddof(ddof, response.shape[0])
    standardized_response, _ = _standardize(response, degrees)
    standardized_phenotype, _ = _standardize(phenotype, degrees)
    return standardized_response.T @ standardized_phenotype / float(
        response.shape[0] - degrees
    )


def standardized_cross_covariance_summary(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    ddof: int = 1,
) -> dict[str, Any]:
    """Summarize standardized cross-covariance with comparable norm scales."""

    response = _as_float_matrix(z_response, "z_response")
    phenotype = _as_float_matrix(z_phenotype, "z_phenotype")
    if response.shape[0] != phenotype.shape[0]:
        raise ValueError("z_response and z_phenotype must have the same row count")
    degrees = _validate_ddof(ddof, response.shape[0])
    cross_covariance = standardized_cross_covariance(
        response, phenotype, ddof=degrees
    )
    squared_frobenius = float(np.sum(cross_covariance * cross_covariance))
    mean_square = squared_frobenius / float(cross_covariance.size)
    _, response_nonconstant = _standardize(response, degrees)
    _, phenotype_nonconstant = _standardize(phenotype, degrees)
    return {
        "n_samples": int(response.shape[0]),
        "response_dimensions": int(response.shape[1]),
        "phenotype_dimensions": int(phenotype.shape[1]),
        "ddof": degrees,
        "standardized": True,
        "cross_covariance": cross_covariance,
        "frobenius_norm": float(math.sqrt(squared_frobenius)),
        "squared_frobenius_norm": squared_frobenius,
        "mean_squared_norm": mean_square,
        "root_mean_squared_norm": float(math.sqrt(mean_square)),
        "response_constant_dimensions": int(np.count_nonzero(~response_nonconstant)),
        "phenotype_constant_dimensions": int(
            np.count_nonzero(~phenotype_nonconstant)
        ),
    }


def standardized_cross_covariance_norm(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    ddof: int = 1,
) -> float:
    """Return the Frobenius norm of standardized cross-covariance."""

    covariance = standardized_cross_covariance(
        z_response, z_phenotype, ddof=ddof
    )
    return _stable_frobenius_norm(
        covariance, "standardized cross-covariance norm"
    )


def _inverse_square_root_psd(matrix: FloatArray) -> FloatArray:
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(1, matrix.shape[0]) * scale
    inverse_roots = np.zeros_like(eigenvalues)
    keep = eigenvalues > tolerance
    inverse_roots[keep] = 1.0 / np.sqrt(eigenvalues[keep])
    return (eigenvectors * inverse_roots) @ eigenvectors.T


def _ridge_scaled_center(
    matrix: FloatArray, ridge: float
) -> tuple[FloatArray, FloatArray]:
    """Center in safe units and express a raw-unit ridge in those units."""

    maxima = np.maximum(np.max(np.abs(matrix), axis=0), math.sqrt(ridge))
    exponents = np.asarray(np.frexp(maxima)[1], dtype=np.int64)
    scaled = np.ldexp(matrix, -exponents[None, :])
    shifted = scaled - scaled[0:1]
    centered = shifted - shifted.mean(axis=0, keepdims=True)
    with np.errstate(over="raise", under="ignore"):
        scaled_ridge = np.ldexp(
            np.full(matrix.shape[1], ridge, dtype=np.float64), -2 * exponents
        )
    return centered, scaled_ridge


def _validate_top_k(top_k: int | None, maximum: int) -> int:
    if top_k is None:
        return maximum
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(
        top_k, (int, np.integer)
    ):
        raise TypeError("top_k must be an integer or None")
    value = int(top_k)
    if value < 1 or value > maximum:
        raise ValueError(f"top_k must satisfy 1 <= top_k <= {maximum}")
    return value


def regularized_canonical_correlations(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    regularization: float = 1e-3,
    top_k: int | None = None,
    ddof: int = 1,
    standardize: bool = True,
) -> FloatArray:
    """Return descending ridge-regularized canonical correlations.

    With ``standardize=True`` (the default), ridge is added to within-block
    correlation matrices and is therefore dimensionless.  With
    ``standardize=False``, ridge is in covariance units.  ``regularization=0``
    uses a pseudoinverse square root and remains defined for rank-deficient or
    constant inputs.

    This is an in-sample representation diagnostic.  Centering, scaling, and
    covariance estimates all use the supplied rows.  It uses no labels, but it
    must be computed on training-fold rows only if its value will select a
    checkpoint/hyperparameter; held-out outcomes must not be used to tune
    ``regularization`` or ``top_k``.
    """

    response = _as_float_matrix(z_response, "z_response")
    phenotype = _as_float_matrix(z_phenotype, "z_phenotype")
    if response.shape[0] != phenotype.shape[0]:
        raise ValueError("z_response and z_phenotype must have the same row count")
    degrees = _validate_ddof(ddof, response.shape[0])
    ridge = _validate_nonnegative_finite(regularization, "regularization")
    maximum = min(response.shape[1], phenotype.shape[1])
    components = _validate_top_k(top_k, maximum)

    if not isinstance(standardize, (bool, np.bool_)):
        raise TypeError("standardize must be boolean")
    if standardize:
        x, _ = _standardize(response, degrees)
        y, _ = _standardize(phenotype, degrees)
        ridge_x = np.full(response.shape[1], ridge, dtype=np.float64)
        ridge_y = np.full(phenotype.shape[1], ridge, dtype=np.float64)
    else:
        x, ridge_x = _ridge_scaled_center(response, ridge)
        y, ridge_y = _ridge_scaled_center(phenotype, ridge)
    divisor = float(response.shape[0] - degrees)
    covariance_x = (x.T @ x) / divisor
    covariance_y = (y.T @ y) / divisor
    cross_covariance = (x.T @ y) / divisor
    if ridge:
        covariance_x = covariance_x + np.diag(ridge_x)
        covariance_y = covariance_y + np.diag(ridge_y)
    whitened = (
        _inverse_square_root_psd(covariance_x)
        @ cross_covariance
        @ _inverse_square_root_psd(covariance_y)
    )
    correlations = np.linalg.svd(whitened, compute_uv=False)
    # Exact CCA is bounded by one.  Clipping removes only floating-point drift.
    correlations = np.clip(correlations, 0.0, 1.0)
    return correlations[:components]


def canonical_correlation_summary(
    z_response: ArrayLike,
    z_phenotype: ArrayLike,
    *,
    regularization: float = 1e-3,
    top_k: int | None = None,
    ddof: int = 1,
    standardize: bool = True,
) -> dict[str, Any]:
    """Return canonical correlations plus an explicit leakage-scope record."""

    response = _as_float_matrix(z_response, "z_response")
    phenotype = _as_float_matrix(z_phenotype, "z_phenotype")
    if response.shape[0] != phenotype.shape[0]:
        raise ValueError("z_response and z_phenotype must have the same row count")
    degrees = _validate_ddof(ddof, response.shape[0])
    correlations = regularized_canonical_correlations(
        response,
        phenotype,
        regularization=regularization,
        top_k=top_k,
        ddof=degrees,
        standardize=standardize,
    )
    return {
        "n_samples": int(response.shape[0]),
        "response_dimensions": int(response.shape[1]),
        "phenotype_dimensions": int(phenotype.shape[1]),
        "ddof": degrees,
        "canonical_correlations": correlations,
        "top_k": int(correlations.size),
        "regularization": float(regularization),
        "regularization_units": "correlation" if standardize else "covariance",
        "standardized": bool(standardize),
        "mean_canonical_correlation": float(np.mean(correlations)),
        "max_canonical_correlation": float(np.max(correlations)),
        "mean_squared_canonical_correlation": float(
            np.mean(correlations * correlations)
        ),
        "fit_scope": "supplied_rows_in_sample",
        "outcome_labels_used": False,
        "leakage_note": (
            "Use training-fold rows only if this diagnostic informs selection; "
            "do not tune regularization/top_k against held-out pCR."
        ),
    }


# Familiar short name while retaining the regularization-explicit primary API.
canonical_correlations = regularized_canonical_correlations


def paired_cosine_similarities(
    first_view: ArrayLike, second_view: ArrayLike
) -> FloatArray:
    """Return row-paired cosine similarities, using zero for a zero-norm row.

    Mapping undefined zero-vector cosines to zero prevents a collapsed pair of
    representations from being reported as perfectly augmentation-consistent.
    Row-wise max scaling avoids overflow for otherwise finite large inputs.
    """

    first = _as_float_matrix(first_view, "first_view")
    second = _as_float_matrix(second_view, "second_view")
    if first.shape != second.shape:
        raise ValueError("first_view and second_view must have identical shapes")

    first_max = np.max(np.abs(first), axis=1)
    second_max = np.max(np.abs(second), axis=1)
    valid = (first_max > 0.0) & (second_max > 0.0)
    similarities = np.zeros(first.shape[0], dtype=np.float64)
    if np.any(valid):
        scaled_first = first[valid] / first_max[valid, None]
        scaled_second = second[valid] / second_max[valid, None]
        scaled_first /= np.linalg.norm(scaled_first, axis=1, keepdims=True)
        scaled_second /= np.linalg.norm(scaled_second, axis=1, keepdims=True)
        similarities[valid] = np.einsum(
            "ij,ij->i", scaled_first, scaled_second
        )
    return np.clip(similarities, -1.0, 1.0)


def augmentation_cosine_consistency(
    first_view: ArrayLike, second_view: ArrayLike
) -> dict[str, Any]:
    """Summarize paired same-visit augmentation cosine consistency."""

    first = _as_float_matrix(first_view, "first_view")
    second = _as_float_matrix(second_view, "second_view")
    if first.shape != second.shape:
        raise ValueError("first_view and second_view must have identical shapes")
    similarities = paired_cosine_similarities(first, second)
    first_zero = np.all(first == 0.0, axis=1)
    second_zero = np.all(second == 0.0, axis=1)
    undefined = first_zero | second_zero
    return {
        "n_pairs": int(first.shape[0]),
        "per_pair_cosine": similarities,
        "mean_cosine": float(np.mean(similarities)),
        "median_cosine": float(np.median(similarities)),
        "std_cosine": float(np.std(similarities, ddof=0)),
        "min_cosine": float(np.min(similarities)),
        "max_cosine": float(np.max(similarities)),
        "zero_norm_first": int(np.count_nonzero(first_zero)),
        "zero_norm_second": int(np.count_nonzero(second_zero)),
        "undefined_pairs_mapped_to_zero": int(np.count_nonzero(undefined)),
        "zero_norm_policy": "cosine=0",
    }


def _validate_patient_ids(
    values: Sequence[Hashable], expected_rows: int, seed: Hashable
) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"patient IDs for seed {seed!r} must be a sequence")
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"patient IDs for seed {seed!r} must be one-dimensional")
    identifiers = tuple(array.tolist())
    if len(identifiers) != expected_rows:
        raise ValueError(
            f"patient ID count for seed {seed!r} does not match representation rows"
        )
    seen: dict[Hashable, int] = {}
    for index, identifier in enumerate(identifiers):
        if identifier is None or (
            isinstance(identifier, str) and not identifier.strip()
        ):
            raise ValueError(f"patient IDs for seed {seed!r} contain an empty value")
        if isinstance(identifier, (float, np.floating)) and not math.isfinite(
            float(identifier)
        ):
            raise ValueError(
                f"patient IDs for seed {seed!r} contain a non-finite value"
            )
        try:
            hash(identifier)
        except TypeError as error:
            raise ValueError(
                f"patient IDs for seed {seed!r} must be hashable"
            ) from error
        if identifier in seen:
            raise ValueError(
                f"duplicate patient ID {identifier!r} for seed {seed!r}"
            )
        seen[identifier] = index
    return identifiers


def _pairwise_distances(
    representation: FloatArray, metric: str | Callable[[Any, Any], float]
) -> FloatArray:
    # Lazy import keeps the basic collapse diagnostics usable in NumPy-only
    # environments while accepting every sklearn pairwise metric/callable.
    try:
        from sklearn.metrics import pairwise_distances
    except ImportError as error:  # pragma: no cover - project declares sklearn
        raise ImportError(
            "nearest-neighbor stability requires scikit-learn"
        ) from error
    distance_input = representation
    if isinstance(metric, str) and metric.casefold() == "cosine":
        # Cosine is row-scale invariant; bound each row before sklearn takes
        # dot products so very large finite representations remain valid.
        row_scale = np.max(np.abs(representation), axis=1)
        distance_input = np.zeros_like(representation)
        nonzero = row_scale > 0.0
        distance_input[nonzero] = (
            representation[nonzero] / row_scale[nonzero, None]
        )
    distances = np.asarray(
        pairwise_distances(distance_input, metric=metric), dtype=np.float64
    )
    if distances.shape != (representation.shape[0], representation.shape[0]):
        raise ValueError("metric did not produce a square pairwise distance matrix")
    if not np.isfinite(distances).all():
        raise ValueError("metric produced NaN or infinite pairwise distances")
    return distances


def nearest_neighbor_jaccard_stability(
    representations_by_seed: Mapping[Hashable, ArrayLike],
    patient_ids_by_seed: Mapping[Hashable, Sequence[Hashable]],
    *,
    top_k: int = 10,
    metric: str | Callable[[Any, Any], float] = "cosine",
    tie_rtol: float = 1e-12,
    tie_atol: float = 1e-15,
) -> dict[str, Any]:
    """Measure cross-seed top-k nearest-neighbor Jaccard stability.

    Every seed must contain exactly the same unique patient-ID set, but row
    order may differ.  Rows are aligned to the first seed before neighbors are
    computed.  The patient itself is excluded.  Equal-distance ties are broken
    deterministically by that aligned order for auditability, but a tie at the
    top-k boundary makes that patient/seed neighborhood ambiguous.  Primary
    Jaccard summaries exclude comparisons involving an ambiguous neighborhood
    or a fully collapsed seed; deterministic scores are retained separately.
    Thus two collapsed representations report an undefined (NaN), rather than
    spuriously perfect, primary stability.  Results otherwise aggregate every
    unordered seed pair and patient with equal weight.
    """

    if not isinstance(representations_by_seed, Mapping) or not isinstance(
        patient_ids_by_seed, Mapping
    ):
        raise TypeError("representations_by_seed and patient_ids_by_seed must be mappings")
    seeds = tuple(representations_by_seed)
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    if set(patient_ids_by_seed) != set(seeds):
        raise ValueError("representation and patient-ID mappings must have the same seeds")
    relative_tie_tolerance = _validate_nonnegative_finite(tie_rtol, "tie_rtol")
    absolute_tie_tolerance = _validate_nonnegative_finite(tie_atol, "tie_atol")

    matrices: dict[Hashable, FloatArray] = {}
    identifiers_by_seed: dict[Hashable, tuple[Hashable, ...]] = {}
    for seed in seeds:
        matrices[seed] = _as_float_matrix(
            representations_by_seed[seed], f"representations_by_seed[{seed!r}]"
        )
        identifiers_by_seed[seed] = _validate_patient_ids(
            patient_ids_by_seed[seed], matrices[seed].shape[0], seed
        )

    reference_ids = identifiers_by_seed[seeds[0]]
    reference_set = set(reference_ids)
    n_patients = len(reference_ids)
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(
        top_k, (int, np.integer)
    ):
        raise TypeError("top_k must be an integer")
    neighbors = int(top_k)
    if neighbors < 1 or neighbors >= n_patients:
        raise ValueError(f"top_k must satisfy 1 <= top_k < {n_patients}")

    aligned: dict[Hashable, FloatArray] = {}
    for seed in seeds:
        current_ids = identifiers_by_seed[seed]
        current_set = set(current_ids)
        if current_set != reference_set:
            missing = tuple(
                identifier for identifier in reference_ids if identifier not in current_set
            )
            extra = tuple(
                identifier for identifier in current_ids if identifier not in reference_set
            )
            raise ValueError(
                f"patient ID set mismatch for seed {seed!r}; "
                f"missing={missing[:5]!r}, extra={extra[:5]!r}"
            )
        row_for_id = {identifier: row for row, identifier in enumerate(current_ids)}
        aligned[seed] = matrices[seed][
            [row_for_id[identifier] for identifier in reference_ids]
        ]

    neighbor_indices: dict[Hashable, NDArray[np.int64]] = {}
    boundary_ties: dict[Hashable, NDArray[np.bool_]] = {}
    all_distances_tied: dict[Hashable, NDArray[np.bool_]] = {}
    collapsed_by_seed: dict[Hashable, bool] = {}
    index_tiebreaker = np.arange(n_patients)
    for seed in seeds:
        distances = _pairwise_distances(aligned[seed], metric)
        distances[np.arange(n_patients), np.arange(n_patients)] = np.inf
        seed_neighbors = np.empty((n_patients, neighbors), dtype=np.int64)
        seed_boundary_ties = np.zeros(n_patients, dtype=np.bool_)
        seed_all_tied = np.zeros(n_patients, dtype=np.bool_)
        for patient_index in range(n_patients):
            ordering = np.lexsort((index_tiebreaker, distances[patient_index]))
            seed_neighbors[patient_index] = ordering[:neighbors]
            candidate_distances = distances[patient_index, ordering[: n_patients - 1]]
            seed_all_tied[patient_index] = bool(
                np.allclose(
                    candidate_distances,
                    candidate_distances[0],
                    rtol=relative_tie_tolerance,
                    atol=absolute_tie_tolerance,
                )
            )
            if neighbors < n_patients - 1:
                seed_boundary_ties[patient_index] = bool(
                    np.isclose(
                        candidate_distances[neighbors - 1],
                        candidate_distances[neighbors],
                        rtol=relative_tie_tolerance,
                        atol=absolute_tie_tolerance,
                    )
                )
        neighbor_indices[seed] = seed_neighbors
        boundary_ties[seed] = seed_boundary_ties
        all_distances_tied[seed] = seed_all_tied
        collapsed_by_seed[seed] = bool(
            np.all(aligned[seed] == aligned[seed][0:1])
        )

    pair_summaries: list[dict[str, Any]] = []
    all_jaccards: list[FloatArray] = []
    all_deterministic_jaccards: list[FloatArray] = []
    for first_seed, second_seed in combinations(seeds, 2):
        deterministic = np.empty(n_patients, dtype=np.float64)
        for patient_index in range(n_patients):
            first_set = set(neighbor_indices[first_seed][patient_index].tolist())
            second_set = set(neighbor_indices[second_seed][patient_index].tolist())
            deterministic[patient_index] = len(first_set & second_set) / len(
                first_set | second_set
            )
        ambiguous = boundary_ties[first_seed] | boundary_ties[second_seed]
        if collapsed_by_seed[first_seed] or collapsed_by_seed[second_seed]:
            ambiguous[:] = True
        per_patient = np.where(ambiguous, np.nan, deterministic)
        valid = ~ambiguous
        all_jaccards.append(per_patient)
        all_deterministic_jaccards.append(deterministic)
        pair_summaries.append(
            {
                "first_seed": first_seed,
                "second_seed": second_seed,
                "per_patient_jaccard": per_patient,
                "deterministic_per_patient_jaccard": deterministic,
                "ambiguous_patient_mask": ambiguous,
                "valid_patient_comparisons": int(np.count_nonzero(valid)),
                "ambiguous_patient_comparisons": int(np.count_nonzero(ambiguous)),
                "mean_jaccard": (
                    float(np.mean(deterministic[valid])) if np.any(valid) else math.nan
                ),
                "median_jaccard": (
                    float(np.median(deterministic[valid])) if np.any(valid) else math.nan
                ),
                "min_jaccard": (
                    float(np.min(deterministic[valid])) if np.any(valid) else math.nan
                ),
                "max_jaccard": (
                    float(np.max(deterministic[valid])) if np.any(valid) else math.nan
                ),
                "deterministic_mean_jaccard": float(np.mean(deterministic)),
            }
        )

    stacked = np.stack(all_jaccards, axis=0)
    deterministic_stacked = np.stack(all_deterministic_jaccards, axis=0)
    valid_stacked = np.isfinite(stacked)
    valid_values = stacked[valid_stacked]
    per_patient_valid_counts = np.sum(valid_stacked, axis=0)
    per_patient_mean = np.full(n_patients, np.nan, dtype=np.float64)
    np.divide(
        np.nansum(stacked, axis=0),
        per_patient_valid_counts,
        out=per_patient_mean,
        where=per_patient_valid_counts > 0,
    )
    neighbors_by_seed = {
        seed: {
            patient_id: tuple(
                reference_ids[index]
                for index in neighbor_indices[seed][patient_index]
            )
            for patient_index, patient_id in enumerate(reference_ids)
        }
        for seed in seeds
    }
    metric_name = (
        metric
        if isinstance(metric, str)
        else getattr(metric, "__name__", repr(metric))
    )
    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "patient_ids": reference_ids,
        "n_patients": n_patients,
        "top_k": neighbors,
        "metric": metric_name,
        "tie_breaking": "aligned_patient_order",
        "tie_rtol": relative_tie_tolerance,
        "tie_atol": absolute_tie_tolerance,
        "primary_tie_policy": "exclude_boundary_ties_and_collapsed_seeds",
        "collapsed_by_seed": collapsed_by_seed,
        "collapsed_seeds": tuple(
            seed for seed in seeds if collapsed_by_seed[seed]
        ),
        "boundary_tie_mask_by_seed": boundary_ties,
        "boundary_tie_counts_by_seed": {
            seed: int(np.count_nonzero(boundary_ties[seed])) for seed in seeds
        },
        "all_distances_tied_counts_by_seed": {
            seed: int(np.count_nonzero(all_distances_tied[seed])) for seed in seeds
        },
        "neighbors_by_seed": neighbors_by_seed,
        "seed_pair_summaries": tuple(pair_summaries),
        "valid_patient_comparisons": int(valid_values.size),
        "ambiguous_patient_comparisons": int(stacked.size - valid_values.size),
        "per_patient_mean_jaccard": per_patient_mean,
        "mean_jaccard": float(np.mean(valid_values)) if valid_values.size else math.nan,
        "median_jaccard": (
            float(np.median(valid_values)) if valid_values.size else math.nan
        ),
        "min_jaccard": float(np.min(valid_values)) if valid_values.size else math.nan,
        "max_jaccard": float(np.max(valid_values)) if valid_values.size else math.nan,
        "deterministic_mean_jaccard": float(np.mean(deterministic_stacked)),
        "status": "ok" if valid_values.size else "undefined_no_unambiguous_neighbors",
    }


# Explicit cross-seed alias used by some report code.
cross_seed_nearest_neighbor_stability = nearest_neighbor_jaccard_stability


__all__ = [
    "augmentation_cosine_consistency",
    "canonical_correlation_summary",
    "canonical_correlations",
    "covariance_eigenspectrum",
    "covariance_eigenspectrum_summary",
    "covariance_spectrum_summary",
    "cross_covariance",
    "cross_covariance_norm",
    "cross_covariance_summary",
    "cross_seed_nearest_neighbor_stability",
    "effective_rank",
    "nearest_neighbor_jaccard_stability",
    "paired_cosine_similarities",
    "per_dimension_variance_summary",
    "regularized_canonical_correlations",
    "standardized_cross_covariance",
    "standardized_cross_covariance_norm",
    "standardized_cross_covariance_summary",
    "variance_std_summary",
]
