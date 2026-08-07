"""Deterministic metrics used by the CoRe-WM shortcut audit.

The functions in this module deliberately keep metric computation separate from
model execution.  In particular, transition metrics return one row per input
transition, while the corresponding summarizer first gives every patient equal
weight before computing fold summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


BINARY_METRICS = ("auroc", "auprc", "accuracy", "sensitivity", "specificity")
TRANSITION_METRICS = (
    "learned_layer_norm_mse",
    "copy_layer_norm_mse",
    "learned_raw_mse",
    "copy_raw_mse",
    "learned_cosine_similarity",
    "copy_cosine_similarity",
    "learned_cosine_error",
    "copy_cosine_error",
    "normalized_transition_gain",
    "percentage_improvement",
)


def _feature_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError(f"{name} must have a non-empty feature dimension")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _matching_feature_arrays(left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
    left_array = _feature_array(left, "left")
    right_array = _feature_array(right, "right")
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"feature arrays must have identical shapes, got "
            f"{left_array.shape} and {right_array.shape}"
        )
    return left_array, right_array


def featurewise_layer_norm(values: Any, eps: float = 1e-5) -> np.ndarray:
    """Apply the exact non-affine feature-wise LayerNorm used by the JEPA loss.

    The last dimension is normalized with the population variance (``ddof=0``),
    matching ``torch.nn.functional.layer_norm(values, (values.shape[-1],),
    eps=eps)``.  There is no learned scale or bias.
    """

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number")
    array = _feature_array(values, "values")
    mean = array.mean(axis=-1, keepdims=True)
    variance = ((array - mean) ** 2).mean(axis=-1, keepdims=True)
    return (array - mean) / np.sqrt(variance + eps)


def layer_norm_mse(prediction: Any, target: Any, eps: float = 1e-5) -> np.ndarray:
    """Return the original JEPA distance for every leading-dimension item.

    Both inputs are independently feature-wise LayerNorm-normalized before the
    squared difference is averaged over the final feature dimension.  No batch
    or transition reduction is performed.
    """

    prediction_array, target_array = _matching_feature_arrays(prediction, target)
    normalized_prediction = featurewise_layer_norm(prediction_array, eps=eps)
    normalized_target = featurewise_layer_norm(target_array, eps=eps)
    return ((normalized_prediction - normalized_target) ** 2).mean(axis=-1)


def raw_mse(prediction: Any, target: Any) -> np.ndarray:
    """Return unnormalized MSE over the final feature dimension."""

    prediction_array, target_array = _matching_feature_arrays(prediction, target)
    return ((prediction_array - target_array) ** 2).mean(axis=-1)


def cosine_similarity(prediction: Any, target: Any) -> np.ndarray:
    """Return cosine similarity over features; zero-norm pairs are ``NaN``."""

    prediction_array, target_array = _matching_feature_arrays(prediction, target)
    numerator = (prediction_array * target_array).sum(axis=-1)
    denominator = np.linalg.norm(prediction_array, axis=-1) * np.linalg.norm(
        target_array, axis=-1
    )
    similarity = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=similarity, where=denominator > 0)
    finite = np.isfinite(similarity)
    similarity[finite] = np.clip(similarity[finite], -1.0, 1.0)
    return similarity


def cosine_error(prediction: Any, target: Any) -> np.ndarray:
    """Return ``1 - cosine_similarity``; undefined zero-norm pairs stay ``NaN``."""

    return 1.0 - cosine_similarity(prediction, target)


def normalized_transition_gain(
    learned_error: Any,
    copy_error: Any,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute ``(copy_error - learned_error) / (copy_error + epsilon)``.

    Positive values mean the learned transition has lower error than copying the
    current state.  Negative values mean copying is better.
    """

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")
    learned = np.asarray(learned_error, dtype=np.float64)
    copied = np.asarray(copy_error, dtype=np.float64)
    try:
        learned, copied = np.broadcast_arrays(learned, copied)
    except ValueError as error:
        raise ValueError(
            "learned_error and copy_error are not broadcast-compatible"
        ) from error
    finite_values = np.isfinite(learned) & np.isfinite(copied)
    if np.any(learned[finite_values] < 0) or np.any(copied[finite_values] < 0):
        raise ValueError("errors must be non-negative")
    return (copied - learned) / (copied + epsilon)


def percentage_improvement(
    learned_error: Any,
    copy_error: Any,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return the stabilized learned-vs-copy error improvement in percent."""

    return 100.0 * normalized_transition_gain(learned_error, copy_error, epsilon)


def _metadata_frame(
    metadata: pd.DataFrame | Mapping[str, Any] | None, n_rows: int
) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame(index=np.arange(n_rows))
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.reset_index(drop=True).copy()
    elif isinstance(metadata, Mapping):
        columns: dict[str, Any] = {}
        for name, values in metadata.items():
            value_array = np.asarray(values)
            if value_array.ndim == 0:
                columns[name] = np.repeat(value_array.item(), n_rows)
            else:
                columns[name] = values
        frame = pd.DataFrame(columns)
    else:
        raise TypeError("metadata must be a pandas DataFrame, mapping, or None")
    if len(frame) != n_rows:
        raise ValueError(f"metadata has {len(frame)} rows but metrics have {n_rows}")
    return frame


def transition_metrics(
    learned: Any,
    copy_current: Any,
    target: Any,
    *,
    metadata: pd.DataFrame | Mapping[str, Any] | None = None,
    layer_norm_eps: float = 1e-5,
    gain_epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Compute and preserve one auditable record per transition.

    Inputs have shape ``[..., feature]`` and must be identical.  Leading
    dimensions are flattened in C order.  For multi-dimensional inputs callers
    should supply equally flattened metadata such as ``patient_id``, ``fold``,
    and ``transition``.
    """

    learned_array = _feature_array(learned, "learned")
    copy_array = _feature_array(copy_current, "copy_current")
    target_array = _feature_array(target, "target")
    if not (learned_array.shape == copy_array.shape == target_array.shape):
        raise ValueError(
            "learned, copy_current, and target must have identical shapes; got "
            f"{learned_array.shape}, {copy_array.shape}, and {target_array.shape}"
        )

    learned_normalized_error = np.asarray(
        layer_norm_mse(learned_array, target_array, eps=layer_norm_eps)
    ).reshape(-1)
    copy_normalized_error = np.asarray(
        layer_norm_mse(copy_array, target_array, eps=layer_norm_eps)
    ).reshape(-1)
    learned_similarity = np.asarray(
        cosine_similarity(learned_array, target_array)
    ).reshape(-1)
    copy_similarity = np.asarray(cosine_similarity(copy_array, target_array)).reshape(
        -1
    )
    n_rows = learned_normalized_error.size
    frame = _metadata_frame(metadata, n_rows)
    collisions = set(frame.columns) & ({"sample_index"} | set(TRANSITION_METRICS))
    if collisions:
        raise ValueError(
            f"metadata columns collide with metric columns: {sorted(collisions)}"
        )

    frame.insert(0, "sample_index", np.arange(n_rows, dtype=np.int64))
    frame["learned_layer_norm_mse"] = learned_normalized_error
    frame["copy_layer_norm_mse"] = copy_normalized_error
    frame["learned_raw_mse"] = np.asarray(raw_mse(learned_array, target_array)).reshape(
        -1
    )
    frame["copy_raw_mse"] = np.asarray(raw_mse(copy_array, target_array)).reshape(-1)
    frame["learned_cosine_similarity"] = learned_similarity
    frame["copy_cosine_similarity"] = copy_similarity
    frame["learned_cosine_error"] = 1.0 - learned_similarity
    frame["copy_cosine_error"] = 1.0 - copy_similarity
    frame["normalized_transition_gain"] = normalized_transition_gain(
        learned_normalized_error, copy_normalized_error, epsilon=gain_epsilon
    )
    frame["percentage_improvement"] = percentage_improvement(
        learned_normalized_error, copy_normalized_error, epsilon=gain_epsilon
    )
    return frame


def _validate_metric_columns(
    frame: pd.DataFrame, metric_columns: Sequence[str]
) -> list[str]:
    columns = list(metric_columns)
    if not columns:
        raise ValueError("metric_columns must not be empty")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"missing metric columns: {missing}")
    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"metric columns must be numeric: {non_numeric}")
    return columns


def aggregate_transition_metrics_by_patient(
    transitions: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    fold_col: str | None = "fold",
    metric_columns: Sequence[str] = TRANSITION_METRICS,
) -> pd.DataFrame:
    """Average repeated transitions within patient before higher-level summaries."""

    if patient_col not in transitions:
        raise KeyError(f"missing patient column: {patient_col}")
    if transitions[patient_col].isna().any():
        raise ValueError("patient identifiers must not be missing")
    metrics = _validate_metric_columns(transitions, metric_columns)
    group_columns: list[str] = []
    if fold_col is not None:
        if fold_col not in transitions:
            raise KeyError(f"missing fold column: {fold_col}")
        if transitions[fold_col].isna().any():
            raise ValueError("fold identifiers must not be missing")
        folds_per_patient = transitions.groupby(patient_col, sort=False)[
            fold_col
        ].nunique()
        if (folds_per_patient > 1).any():
            offending = folds_per_patient[folds_per_patient > 1].index.tolist()
            raise ValueError(f"patients occur in multiple folds: {offending[:5]}")
        group_columns.append(fold_col)
    group_columns.append(patient_col)

    grouped = transitions.groupby(group_columns, sort=True, observed=True, dropna=False)
    patient_metrics = grouped[metrics].mean().reset_index()
    counts = grouped.size().rename("n_transitions").reset_index()
    patient_metrics = patient_metrics.merge(
        counts, on=group_columns, validate="one_to_one"
    )
    return patient_metrics[[*group_columns, "n_transitions", *metrics]]


def fold_mean_sample_std(
    fold_metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = BINARY_METRICS,
) -> pd.DataFrame:
    """Summarize fold metrics with NaN-aware mean and sample SD (``ddof=1``)."""

    metrics = _validate_metric_columns(fold_metrics, metric_columns)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = fold_metrics[metric].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "metric": metric,
                "mean": float(finite.mean()) if finite.size else np.nan,
                "sample_std": float(finite.std(ddof=1)) if finite.size >= 2 else np.nan,
                "n_valid_folds": int(finite.size),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "mean", "sample_std", "n_valid_folds"])


def summarize_transition_folds(
    transitions: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    fold_col: str = "fold",
    metric_columns: Sequence[str] = TRANSITION_METRICS,
) -> dict[str, Any]:
    """Return transition-, patient-, and fold-weighted summaries.

    ``fold_metrics`` are computed from patient means, so a patient with three
    valid transitions does not receive three times the fold weight of a patient
    with one.  ``pooled_transition_metrics`` is additionally reported because it
    is explicitly requested by the audit, but is clearly separated from the
    equal-patient pooled result.
    """

    metrics = _validate_metric_columns(transitions, metric_columns)
    patient_metrics = aggregate_transition_metrics_by_patient(
        transitions,
        patient_col=patient_col,
        fold_col=fold_col,
        metric_columns=metrics,
    )
    grouped = patient_metrics.groupby(fold_col, sort=True, observed=True, dropna=False)
    fold_metrics = grouped[metrics].mean().reset_index()
    fold_counts = grouped.agg(
        n_patients=(patient_col, "size"), n_transitions=("n_transitions", "sum")
    ).reset_index()
    fold_metrics = fold_counts.merge(fold_metrics, on=fold_col, validate="one_to_one")

    pooled_transition: dict[str, Any] = {"n_transitions": int(len(transitions))}
    pooled_patient: dict[str, Any] = {
        "n_patients": int(len(patient_metrics)),
        "n_transitions": int(patient_metrics["n_transitions"].sum()),
    }
    for metric in metrics:
        pooled_transition[metric] = float(transitions[metric].mean())
        pooled_patient[metric] = float(patient_metrics[metric].mean())
    return {
        "patient_metrics": patient_metrics,
        "fold_metrics": fold_metrics,
        "fold_summary": fold_mean_sample_std(fold_metrics, metric_columns=metrics),
        "pooled_transition_metrics": pooled_transition,
        "pooled_patient_metrics": pooled_patient,
    }


def _binary_arrays(y_true: Any, probability: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    probabilities = np.asarray(probability, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and probability must be one-dimensional")
    if labels.size != probabilities.size:
        raise ValueError("y_true and probability must have the same length")
    if labels.size:
        try:
            numeric_labels = labels.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("y_true must contain only binary 0/1 labels") from error
        if (
            not np.isfinite(numeric_labels).all()
            or not np.isin(numeric_labels, (0.0, 1.0)).all()
        ):
            raise ValueError("y_true must contain only binary 0/1 labels")
        if not np.isfinite(probabilities).all():
            raise ValueError("probability contains non-finite values")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probability values must lie in [0, 1]")
        labels = numeric_labels.astype(np.int64)
    else:
        labels = labels.astype(np.int64)
    return labels, probabilities


def _threshold_array(threshold: Any, n_rows: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(n_rows, values.item(), dtype=np.float64)
    elif values.ndim != 1 or values.size != n_rows:
        raise ValueError(
            "threshold must be scalar or one-dimensional with one value per row"
        )
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("threshold values must be finite and lie in [0, 1]")
    return values


def binary_classification_metrics(
    y_true: Any,
    probability: Any,
    *,
    threshold: Any = 0.5,
) -> dict[str, int | float]:
    """Compute binary discrimination and thresholded classification metrics.

    A fold without both classes has ``NaN`` AUROC and AUPRC by policy.  Accuracy
    remains defined; sensitivity is ``NaN`` without positives and specificity is
    ``NaN`` without negatives.  Empty inputs return ``NaN`` for every metric.
    """

    labels, probabilities = _binary_arrays(y_true, probability)
    thresholds = _threshold_array(threshold, labels.size)
    positives = labels == 1
    negatives = labels == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    result: dict[str, int | float] = {
        "n": int(labels.size),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }
    if labels.size == 0:
        result.update({metric: np.nan for metric in BINARY_METRICS})
        return result

    predicted = probabilities >= thresholds
    true_positive = int((predicted & positives).sum())
    true_negative = int((~predicted & negatives).sum())
    result["auroc"] = (
        float(roc_auc_score(labels, probabilities))
        if n_positive and n_negative
        else np.nan
    )
    result["auprc"] = (
        float(average_precision_score(labels, probabilities))
        if n_positive and n_negative
        else np.nan
    )
    result["accuracy"] = float((predicted == positives).mean())
    result["sensitivity"] = float(true_positive / n_positive) if n_positive else np.nan
    result["specificity"] = float(true_negative / n_negative) if n_negative else np.nan
    return result


def _threshold_values(
    frame: pd.DataFrame, threshold: float | str | Sequence[float]
) -> Any:
    if isinstance(threshold, str):
        if threshold not in frame:
            raise KeyError(f"missing threshold column: {threshold}")
        return frame[threshold].to_numpy(dtype=np.float64)
    return threshold


def binary_metrics_by_fold(
    predictions: pd.DataFrame,
    *,
    fold_col: str = "fold",
    label_col: str = "y_true",
    probability_col: str = "predicted_probability",
    threshold: float | str | Sequence[float] = 0.5,
) -> pd.DataFrame:
    """Compute one binary metric record per fold."""

    required = [fold_col, label_col, probability_col]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"missing prediction columns: {missing}")
    if predictions[fold_col].isna().any():
        raise ValueError("fold identifiers must not be missing")
    frame = predictions.reset_index(drop=True)
    all_thresholds = _threshold_values(frame, threshold)
    if not np.isscalar(all_thresholds):
        all_thresholds = _threshold_array(all_thresholds, len(frame))

    rows: list[dict[str, Any]] = []
    for fold, group in frame.groupby(fold_col, sort=True, observed=True, dropna=False):
        group_threshold: Any = all_thresholds
        if not np.isscalar(all_thresholds):
            group_threshold = all_thresholds[group.index.to_numpy()]
        rows.append(
            {
                fold_col: fold,
                **binary_classification_metrics(
                    group[label_col].to_numpy(),
                    group[probability_col].to_numpy(),
                    threshold=group_threshold,
                ),
            }
        )
    columns = [fold_col, "n", "n_positive", "n_negative", *BINARY_METRICS]
    return pd.DataFrame(rows, columns=columns)


def summarize_binary_folds(
    predictions: pd.DataFrame,
    *,
    fold_col: str = "fold",
    label_col: str = "y_true",
    probability_col: str = "predicted_probability",
    threshold: float | str | Sequence[float] = 0.5,
) -> dict[str, Any]:
    """Return by-fold, fold mean/sample-SD, and pooled OOF binary metrics."""

    frame = predictions.reset_index(drop=True)
    folds = binary_metrics_by_fold(
        frame,
        fold_col=fold_col,
        label_col=label_col,
        probability_col=probability_col,
        threshold=threshold,
    )
    pooled_threshold = _threshold_values(frame, threshold)
    pooled = binary_classification_metrics(
        frame[label_col].to_numpy(),
        frame[probability_col].to_numpy(),
        threshold=pooled_threshold,
    )
    pooled["n_folds"] = int(frame[fold_col].nunique())
    return {
        "fold_metrics": folds,
        "fold_summary": fold_mean_sample_std(folds),
        "pooled_oof": pooled,
    }


def aggregate_patient_predictions(
    predictions: pd.DataFrame,
    probability_columns: str | Sequence[str],
    *,
    patient_col: str = "patient_id",
    label_col: str = "y_true",
    constant_columns: Sequence[str] = (),
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Collapse repeated transition/repetition rows to one row per patient.

    Labels and ``constant_columns`` must be constant within each patient.
    Probabilities are aggregated by mean (default) or median.  The original row
    count is retained as ``n_records`` for auditing.
    """

    if isinstance(probability_columns, str):
        probability_columns = [probability_columns]
    probability_columns = list(probability_columns)
    constant_columns = list(constant_columns)
    required = [patient_col, label_col, *constant_columns, *probability_columns]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"missing patient prediction columns: {missing}")
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'")
    if predictions[patient_col].isna().any():
        raise ValueError("patient identifiers must not be missing")
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                patient_col,
                label_col,
                *constant_columns,
                *probability_columns,
                "n_records",
            ]
        )

    for probability_column in probability_columns:
        _binary_arrays(
            predictions[label_col].to_numpy(),
            predictions[probability_column].to_numpy(),
        )
    grouped = predictions.groupby(patient_col, sort=True, observed=True, dropna=False)
    invariant_columns = [label_col, *constant_columns]
    for column in invariant_columns:
        counts = grouped[column].nunique(dropna=False)
        if (counts != 1).any():
            offending = counts[counts != 1].index.tolist()
            raise ValueError(
                f"{column} is not constant within patients: {offending[:5]}"
            )

    invariant = grouped[invariant_columns].first()
    probabilities = grouped[probability_columns].agg(aggregation)
    counts = grouped.size().rename("n_records")
    result = pd.concat((invariant, probabilities, counts), axis=1).reset_index()
    return result[
        [patient_col, label_col, *constant_columns, *probability_columns, "n_records"]
    ]


def paired_patient_bootstrap_difference(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    label_col: str = "y_true",
    probability_col: str = "predicted_probability",
    pair_columns: Sequence[str] = (),
    aggregation: str = "mean",
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
    require_complete_pairs: bool = True,
) -> dict[str, Any]:
    """Paired patient-block bootstrap CIs for AUROC/AUPRC differences.

    Repeated transitions or audit repetitions are first collapsed within each
    patient using ``aggregation``.  Patients are then sampled with replacement as
    paired blocks.  Every difference is ``comparison - reference``; positive
    values therefore favor the comparison.  Percentile intervals omit bootstrap
    samples that contain only one class.
    """

    if (
        isinstance(n_bootstrap, bool)
        or int(n_bootstrap) != n_bootstrap
        or n_bootstrap <= 0
    ):
        raise ValueError("n_bootstrap must be a positive integer")
    n_bootstrap = int(n_bootstrap)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    reference_patients = aggregate_patient_predictions(
        reference,
        probability_col,
        patient_col=patient_col,
        label_col=label_col,
        constant_columns=pair_columns,
        aggregation=aggregation,
    ).rename(
        columns={
            label_col: "reference_label",
            probability_col: "reference_probability",
            "n_records": "reference_n_records",
            **{column: f"reference_{column}" for column in pair_columns},
        }
    )
    comparison_patients = aggregate_patient_predictions(
        comparison,
        probability_col,
        patient_col=patient_col,
        label_col=label_col,
        constant_columns=pair_columns,
        aggregation=aggregation,
    ).rename(
        columns={
            label_col: "comparison_label",
            probability_col: "comparison_probability",
            "n_records": "comparison_n_records",
            **{column: f"comparison_{column}" for column in pair_columns},
        }
    )
    reference_ids = set(reference_patients[patient_col].tolist())
    comparison_ids = set(comparison_patients[patient_col].tolist())
    if require_complete_pairs and reference_ids != comparison_ids:
        missing_reference = sorted(map(str, comparison_ids - reference_ids))
        missing_comparison = sorted(map(str, reference_ids - comparison_ids))
        raise ValueError(
            "reference and comparison patient sets differ; "
            f"missing from reference={missing_reference[:5]}, "
            f"missing from comparison={missing_comparison[:5]}"
        )
    paired = reference_patients.merge(
        comparison_patients,
        on=patient_col,
        how="inner",
        validate="one_to_one",
        sort=True,
    )
    if not np.array_equal(
        paired["reference_label"].to_numpy(), paired["comparison_label"].to_numpy()
    ):
        raise ValueError("paired reference and comparison labels disagree")
    for column in pair_columns:
        if not paired[f"reference_{column}"].equals(paired[f"comparison_{column}"]):
            raise ValueError(
                f"paired reference and comparison values disagree for {column}"
            )
        paired[column] = paired[f"reference_{column}"]
        paired.drop(
            columns=[f"reference_{column}", f"comparison_{column}"], inplace=True
        )
    paired.rename(columns={"reference_label": label_col}, inplace=True)
    paired.drop(columns=["comparison_label"], inplace=True)

    labels = paired[label_col].to_numpy(dtype=np.int64)
    reference_probability = paired["reference_probability"].to_numpy(dtype=np.float64)
    comparison_probability = paired["comparison_probability"].to_numpy(dtype=np.float64)
    reference_metrics = binary_classification_metrics(labels, reference_probability)
    comparison_metrics = binary_classification_metrics(labels, comparison_probability)

    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, Any]] = []
    n_patients = len(paired)
    for bootstrap_index in range(n_bootstrap):
        if n_patients:
            indices = rng.integers(0, n_patients, size=n_patients)
            sampled_labels = labels[indices]
            sampled_reference = reference_probability[indices]
            sampled_comparison = comparison_probability[indices]
        else:
            sampled_labels = labels
            sampled_reference = reference_probability
            sampled_comparison = comparison_probability
        sampled_reference_metrics = binary_classification_metrics(
            sampled_labels, sampled_reference
        )
        sampled_comparison_metrics = binary_classification_metrics(
            sampled_labels, sampled_comparison
        )
        replicate_rows.append(
            {
                "bootstrap_index": bootstrap_index,
                "auroc_difference": sampled_comparison_metrics["auroc"]
                - sampled_reference_metrics["auroc"],
                "auprc_difference": sampled_comparison_metrics["auprc"]
                - sampled_reference_metrics["auprc"],
            }
        )
    replicates = pd.DataFrame(
        replicate_rows,
        columns=["bootstrap_index", "auroc_difference", "auprc_difference"],
    )
    alpha = 1.0 - confidence_level
    summary_rows: list[dict[str, Any]] = []
    for metric in ("auroc", "auprc"):
        distribution = replicates[f"{metric}_difference"].to_numpy(dtype=np.float64)
        valid = distribution[np.isfinite(distribution)]
        lower, upper = (np.nan, np.nan)
        if valid.size:
            lower, upper = np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0])
        summary_rows.append(
            {
                "metric": metric,
                "reference": reference_metrics[metric],
                "comparison": comparison_metrics[metric],
                "difference": comparison_metrics[metric] - reference_metrics[metric],
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": float(confidence_level),
                "n_patients": int(n_patients),
                "n_bootstrap": n_bootstrap,
                "n_valid_bootstrap": int(valid.size),
                "seed": int(seed),
                "difference_direction": "comparison - reference",
            }
        )
    summary = pd.DataFrame(summary_rows)
    paired.attrs.update(
        {
            "n_reference_patients": len(reference_patients),
            "n_comparison_patients": len(comparison_patients),
            "n_paired_patients": n_patients,
            "aggregation": aggregation,
        }
    )
    return {
        "summary": summary,
        "bootstrap_samples": replicates,
        "patient_predictions": paired,
    }


def difference_from_native(
    native: Mapping[str, Any] | pd.Series | float,
    comparison: Mapping[str, Any] | pd.Series | float,
    *,
    metrics: Sequence[str] | None = None,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Return signed absolute and relative differences from native metrics.

    Differences are always ``comparison - native``.  Relative differences divide
    by ``abs(native)``.  When the native magnitude is at most ``epsilon`` the
    relative result is ``NaN`` (rather than an infinite or arbitrarily huge
    value), and ``relative_defined`` is false.
    """

    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite non-negative number")
    if np.isscalar(native) and np.isscalar(comparison):
        native_values: Mapping[str, Any] = {"value": native}
        comparison_values: Mapping[str, Any] = {"value": comparison}
    elif isinstance(native, (Mapping, pd.Series)) and isinstance(
        comparison, (Mapping, pd.Series)
    ):
        native_values = native
        comparison_values = comparison
    else:
        raise TypeError(
            "native and comparison must both be scalars or both be mappings/Series"
        )

    selected_metrics = (
        list(metrics) if metrics is not None else list(native_values.keys())
    )
    if not selected_metrics:
        raise ValueError("metrics must not be empty")
    missing_native = [
        metric for metric in selected_metrics if metric not in native_values
    ]
    missing_comparison = [
        metric for metric in selected_metrics if metric not in comparison_values
    ]
    if missing_native or missing_comparison:
        raise KeyError(
            f"missing native metrics={missing_native}, comparison metrics={missing_comparison}"
        )

    rows: list[dict[str, Any]] = []
    for metric in selected_metrics:
        native_value = float(native_values[metric])
        comparison_value = float(comparison_values[metric])
        difference = comparison_value - native_value
        relative_defined = bool(
            np.isfinite(native_value)
            and np.isfinite(comparison_value)
            and abs(native_value) > epsilon
        )
        relative = difference / abs(native_value) if relative_defined else np.nan
        rows.append(
            {
                "metric": metric,
                "native": native_value,
                "comparison": comparison_value,
                "absolute_difference": difference,
                "relative_difference": relative,
                "relative_percentage_difference": 100.0 * relative,
                "relative_defined": relative_defined,
                "difference_direction": "comparison - native",
            }
        )
    return pd.DataFrame(rows)
