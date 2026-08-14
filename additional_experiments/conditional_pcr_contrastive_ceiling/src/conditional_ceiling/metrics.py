"""Classification metrics and paired uncertainty for the supervised ceiling.

The functions in this module are deliberately pure: they neither read nor write
experiment artifacts.  In particular, :func:`paired_fold_stratified_bootstrap`
returns aggregate intervals only; bootstrap draws and patient-level predictions
are not part of the public result.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_ECE_BINS = 10
DEFAULT_PROBABILITY_CLIP = 1e-6
DEFAULT_BOOTSTRAP_DRAWS = 5_000
METRIC_NAMES = ("auroc", "auprc", "brier", "calibration_slope", "ece10")


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


def _labels(
    values: Any,
    *,
    name: str = "labels",
    expected_rows: int | None = None,
    require_both_classes: bool = True,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_rows is not None and raw.size != expected_rows:
        raise ValueError(f"{name} has {raw.size} rows; expected {expected_rows}")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must use integer 0/1 values, not booleans")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain binary 0/1 values") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only finite binary 0/1 values")
    labels = numeric.astype(np.int64)
    if require_both_classes and set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def _probabilities(
    values: Any, *, name: str = "probabilities", expected_rows: int | None = None
) -> np.ndarray:
    try:
        probability = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if probability.ndim != 1 or probability.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_rows is not None and probability.size != expected_rows:
        raise ValueError(
            f"{name} has {probability.size} rows; expected {expected_rows}"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{name} must lie in [0,1]")
    return probability


def _clip_value(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("clip must be numeric, not boolean")
    try:
        clip = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("clip must be numeric") from error
    if not math.isfinite(clip) or not 0.0 < clip < 0.5:
        raise ValueError("clip must be finite and strictly between 0 and 0.5")
    return clip


def expected_calibration_error(
    labels: Any,
    probabilities: Any,
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Return equal-width expected calibration error.

    Bins are ``[0,.1), ... [.9,1]`` for the registered ten-bin metric.  Empty
    bins contribute zero and probability one belongs to the final bin.
    """

    bins = _positive_integer(n_bins, name="n_bins")
    y = _labels(labels, require_both_classes=False)
    probability = _probabilities(probabilities, expected_rows=len(y))
    bin_index = np.minimum((probability * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bin_index == index
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(y[mask].mean())
            )
    return float(error)


def ece10(labels: Any, probabilities: Any) -> float:
    """Registered ten-bin equal-width ECE."""

    return expected_calibration_error(labels, probabilities, n_bins=10)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def calibration_slope(
    labels: Any,
    probabilities: Any,
    *,
    clip: float = DEFAULT_PROBABILITY_CLIP,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> float:
    """Fit the unpenalized logistic calibration slope on clipped prediction logits.

    The intercept is free.  A constant prediction has no identifiable slope and
    returns ``NaN``.  A one-class resample likewise returns ``NaN``; this matters
    for percentile bootstrap draws but not for valid headline point estimates.
    """

    y = _labels(labels, require_both_classes=False)
    probability = _probabilities(probabilities, expected_rows=len(y))
    bound = _clip_value(clip)
    iterations = _positive_integer(max_iter, name="max_iter")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if set(np.unique(y)) != {0, 1}:
        return math.nan
    bounded = np.clip(probability, bound, 1.0 - bound)
    logit = np.log(bounded) - np.log1p(-bounded)
    center = float(logit.mean())
    x = logit - center
    if float(np.ptp(x)) <= np.finfo(np.float64).eps:
        return math.nan

    prevalence = float(y.mean())
    coefficients = np.asarray(
        [math.log(prevalence / (1.0 - prevalence)), 1.0], dtype=np.float64
    )
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    for _ in range(iterations):
        fitted = _sigmoid(design @ coefficients)
        weights = np.maximum(fitted * (1.0 - fitted), 1e-12)
        score = design.T @ (y - fitted)
        information = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(
                information + np.eye(2, dtype=np.float64) * 1e-12, score
            )
        except np.linalg.LinAlgError:
            return math.nan
        coefficients += step
        if not np.isfinite(coefficients).all():
            return math.nan
        if float(np.max(np.abs(step))) <= float(tolerance):
            break
    return float(coefficients[1])


def binary_metrics(
    labels: Any,
    probabilities: Any,
    *,
    ece_bins: int = DEFAULT_ECE_BINS,
    probability_clip: float = DEFAULT_PROBABILITY_CLIP,
) -> dict[str, int | float]:
    """Return the five registered binary probability metrics."""

    y = _labels(labels)
    probability = _probabilities(probabilities, expected_rows=len(y))
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(np.mean(np.square(probability - y))),
        "calibration_slope": calibration_slope(
            y, probability, clip=probability_clip
        ),
        f"ece{int(ece_bins)}": expected_calibration_error(
            y, probability, n_bins=ece_bins
        ),
    }


compute_binary_metrics = binary_metrics


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
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} misses required columns: {missing}")
    output = frame.loc[:, [patient_col, fold_col, label_col, probability_col]].copy()
    if output[patient_col].isna().any():
        raise ValueError(f"{name} contains missing patient IDs")
    output[patient_col] = output[patient_col].astype(str)
    if output[patient_col].eq("").any() or output[patient_col].duplicated().any():
        raise ValueError(f"{name} must contain exactly one row per non-empty patient ID")
    try:
        folds = pd.to_numeric(output[fold_col], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} fold IDs must be non-negative integers") from error
    if (
        not np.isfinite(folds).all()
        or not np.equal(folds, np.floor(folds)).all()
        or np.any(folds < 0.0)
    ):
        raise ValueError(f"{name} fold IDs must be non-negative integers")
    output[fold_col] = folds.astype(np.int64)
    output[label_col] = _labels(
        output[label_col].to_numpy(),
        name=f"{name} labels",
        expected_rows=len(output),
        require_both_classes=True,
    )
    output[probability_col] = _probabilities(
        output[probability_col].to_numpy(),
        name=f"{name} probabilities",
        expected_rows=len(output),
    )
    return output


def _metric_vector(
    labels: np.ndarray,
    probability: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, float]:
    """Compute only requested metrics; undefined resample quantities become NaN."""

    requested = set(metrics)
    result: dict[str, float] = {}
    has_both_classes = set(np.unique(labels)) == {0, 1}
    if "auroc" in requested:
        result["auroc"] = (
            float(roc_auc_score(labels, probability)) if has_both_classes else math.nan
        )
    if "auprc" in requested:
        result["auprc"] = (
            float(average_precision_score(labels, probability))
            if has_both_classes
            else math.nan
        )
    if "brier" in requested:
        result["brier"] = float(np.mean(np.square(probability - labels)))
    if "calibration_slope" in requested:
        result["calibration_slope"] = calibration_slope(labels, probability)
    if "ece10" in requested:
        result["ece10"] = ece10(labels, probability)
    return result


def paired_fold_stratified_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    fold_col: str = "fold",
    label_col: str = "y_true",
    probability_col: str = "predicted_probability",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    confidence_level: float = 0.95,
    seed: int = 260_812,
    metrics: Sequence[str] = METRIC_NAMES,
) -> pd.DataFrame:
    """Return paired percentile CIs after resampling patients within outer folds.

    ``delta`` is always the literal ``comparison - reference`` difference.  The
    additional ``improvement`` column orients lower-is-better Brier/ECE so that
    positive values uniformly favor the comparison.  Calibration slope has no
    monotone orientation and therefore keeps its literal difference.
    """

    draws = _positive_integer(n_bootstrap, name="n_bootstrap")
    if not math.isfinite(float(confidence_level)) or not 0.0 < float(
        confidence_level
    ) < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    requested = tuple(str(metric) for metric in metrics)
    unknown = sorted(set(requested) - set(METRIC_NAMES))
    if not requested or unknown:
        raise ValueError(f"metrics must be a non-empty subset of {METRIC_NAMES}; got {unknown}")

    ref = _prediction_frame(
        reference,
        name="reference",
        patient_col=patient_col,
        fold_col=fold_col,
        label_col=label_col,
        probability_col=probability_col,
    ).rename(columns={label_col: "reference_label", probability_col: "reference_probability"})
    cmp = _prediction_frame(
        comparison,
        name="comparison",
        patient_col=patient_col,
        fold_col=fold_col,
        label_col=label_col,
        probability_col=probability_col,
    ).rename(columns={label_col: "comparison_label", probability_col: "comparison_probability"})
    if set(ref[patient_col]) != set(cmp[patient_col]):
        raise ValueError("reference and comparison patient sets must match exactly")
    paired = ref.merge(
        cmp,
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
    ref_probability = paired["reference_probability"].to_numpy(dtype=np.float64)
    cmp_probability = paired["comparison_probability"].to_numpy(dtype=np.float64)
    ref_point = _metric_vector(labels, ref_probability, requested)
    cmp_point = _metric_vector(labels, cmp_probability, requested)

    rng = np.random.default_rng(int(seed))
    folds = paired[fold_col].to_numpy(dtype=np.int64)
    fold_positions = [np.flatnonzero(folds == fold) for fold in sorted(np.unique(folds))]
    sampled = np.concatenate(
        [rng.choice(block, size=(draws, len(block)), replace=True) for block in fold_positions],
        axis=1,
    )
    distributions = {metric: np.full(draws, np.nan, dtype=np.float64) for metric in requested}
    for draw_index, indices in enumerate(sampled):
        sampled_labels = labels[indices]
        sampled_ref = _metric_vector(
            sampled_labels, ref_probability[indices], requested
        )
        sampled_cmp = _metric_vector(
            sampled_labels, cmp_probability[indices], requested
        )
        for metric in requested:
            distributions[metric][draw_index] = sampled_cmp[metric] - sampled_ref[metric]

    alpha = 1.0 - float(confidence_level)
    rows: list[dict[str, Any]] = []
    for metric in requested:
        delta = float(cmp_point[metric] - ref_point[metric])
        finite = distributions[metric][np.isfinite(distributions[metric])]
        lower = float(np.quantile(finite, alpha / 2.0)) if finite.size else math.nan
        upper = float(np.quantile(finite, 1.0 - alpha / 2.0)) if finite.size else math.nan
        lower_is_better = metric in {"brier", "ece10"}
        rows.append(
            {
                "metric": metric,
                "point": delta,
                "reference": float(ref_point[metric]),
                "comparison": float(cmp_point[metric]),
                "reference_value": float(ref_point[metric]),
                "comparison_value": float(cmp_point[metric]),
                "delta": delta,
                "improvement": -delta if lower_is_better else delta,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": float(confidence_level),
                "n_patients": int(len(paired)),
                "n_folds": int(len(fold_positions)),
                "n_bootstrap": draws,
                "n_valid_bootstrap": int(finite.size),
                "bootstrap_unit": "patient_within_outer_fold",
                "ci_method": "percentile",
                "orientation": "comparison - reference",
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


paired_patient_bootstrap = paired_fold_stratified_bootstrap


__all__ = [
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_ECE_BINS",
    "DEFAULT_PROBABILITY_CLIP",
    "METRIC_NAMES",
    "binary_metrics",
    "calibration_slope",
    "compute_binary_metrics",
    "ece10",
    "expected_calibration_error",
    "paired_fold_stratified_bootstrap",
    "paired_patient_bootstrap",
]
