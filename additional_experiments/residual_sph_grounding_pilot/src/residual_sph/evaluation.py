"""Aggregate-only evaluation primitives for the residual-SPH pilot.

The functions in this module deliberately do not read files, patient identifiers,
clinical variables, or pCR labels.  Patient-level arrays are supplied by the
caller, reduced in memory, and never returned.  Formal helpers fail closed unless
the preregistered two seeds and five outer folds are present.

Two aggregation rules are important:

* regression endpoint metrics are computed after pooling the five disjoint OOF
  test folds within a seed; fold-level correlations are never averaged;
* visit/interval macros are unweighted means of those already-pooled endpoint
  metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


FORMAL_SEEDS: tuple[int, int] = (2026, 3026)
FORMAL_FOLDS: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4)
STATIC_VISITS: tuple[str, str, str, str] = ("T0", "T1", "T2", "T3")
OBSERVED_DELTA_INTERVALS: tuple[str, str, str] = (
    "T0-T1",
    "T1-T2",
    "T2-T3",
)
GATE_D_TIMINGS: tuple[str, str, str] = ("T0", "T0-T1", "T0-T2")
REGRESSION_METRICS: tuple[str, ...] = (
    "spearman",
    "pearson",
    "natural_r2",
    "rmse",
    "mae",
    "variance_ratio",
)


class EvaluationContractError(ValueError):
    """Raised when a formal evaluation input violates the frozen contract."""


def _finite_vector(values: Any, *, name: str, minimum_rows: int = 2) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise EvaluationContractError(f"{name} must be a one-dimensional vector")
    if len(array) < minimum_rows:
        raise EvaluationContractError(
            f"{name} must contain at least {minimum_rows} observations"
        )
    if not np.isfinite(array).all():
        raise EvaluationContractError(f"{name} must contain only finite values")
    return array


def _probability_vector(values: Any, *, name: str) -> np.ndarray:
    array = _finite_vector(values, name=name)
    if np.any((array < 0.0) | (array > 1.0)):
        raise EvaluationContractError(f"{name} must lie in [0, 1]")
    return array


def _binary_vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) < 2:
        raise EvaluationContractError(f"{name} must be a one-dimensional vector")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise EvaluationContractError(f"{name} must contain binary values") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise EvaluationContractError(f"{name} must contain only 0 and 1")
    return numeric.astype(np.int64)


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationContractError(f"{name} must be numeric") from error
    if not np.isfinite(scalar):
        raise EvaluationContractError(f"{name} must be finite")
    return scalar


def _integer_key(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise EvaluationContractError(f"{name} must be an integer, not boolean")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationContractError(f"{name} must be an integer") from error
    if not np.isfinite(numeric) or numeric != float(integer):
        raise EvaluationContractError(f"{name} must be an integer")
    return integer


def _normalize_integer_mapping(
    values: Mapping[Any, Any], *, name: str
) -> dict[int, Any]:
    if not isinstance(values, Mapping):
        raise EvaluationContractError(f"{name} must be a mapping")
    normalized: dict[int, Any] = {}
    for raw_key, value in values.items():
        key = _integer_key(raw_key, name=f"{name} key")
        if key in normalized:
            raise EvaluationContractError(f"{name} contains duplicate key {key}")
        normalized[key] = value
    return normalized


def _require_exact_keys(
    observed: Sequence[Any] | set[Any],
    expected: Sequence[Any],
    *,
    name: str,
) -> None:
    observed_set = set(observed)
    expected_set = set(expected)
    if observed_set != expected_set:
        missing = sorted(expected_set.difference(observed_set), key=str)
        extra = sorted(observed_set.difference(expected_set), key=str)
        raise EvaluationContractError(
            f"{name} coverage drifted; missing={missing}, extra={extra}"
        )


def regression_metrics(target: Any, prediction: Any) -> dict[str, float | int]:
    """Return natural-unit regression metrics for one pooled OOF endpoint.

    ``natural_r2`` is deliberately named to distinguish it from R2 in a
    standardized training-target space.  Population variances (``ddof=0``) are
    used for the prediction/target variance ratio.
    """

    y_true = _finite_vector(target, name="target", minimum_rows=1)
    y_pred = _finite_vector(prediction, name="prediction", minimum_rows=1)
    if y_true.shape != y_pred.shape:
        raise EvaluationContractError("target and prediction lengths differ")
    if len(y_true) < 2:
        raise EvaluationContractError(
            "target and prediction must contain at least 2 observations"
        )
    target_variance = float(np.var(y_true, ddof=0))
    prediction_variance = float(np.var(y_pred, ddof=0))
    if target_variance <= 0.0:
        raise EvaluationContractError("target variance must be positive")
    if prediction_variance <= 0.0:
        raise EvaluationContractError(
            "prediction variance must be positive for correlation metrics"
        )
    spearman = float(spearmanr(y_true, y_pred).statistic)
    pearson = float(pearsonr(y_true, y_pred).statistic)
    if not np.isfinite(spearman) or not np.isfinite(pearson):
        raise EvaluationContractError("correlation is undefined")
    return {
        "n": int(len(y_true)),
        "spearman": spearman,
        "pearson": pearson,
        "natural_r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "variance_ratio": prediction_variance / target_variance,
    }


def pooled_oof_regression(
    fold_predictions: Mapping[Any, tuple[Any, Any]],
    *,
    formal: bool = True,
) -> dict[str, Any]:
    """Pool disjoint test folds within one seed, then compute one endpoint metric.

    Each mapping value is ``(target, prediction)`` for that fold's held-out test
    patients.  Only aggregate metrics and fold sizes are returned.
    """

    folds = _normalize_integer_mapping(fold_predictions, name="fold_predictions")
    if formal:
        _require_exact_keys(folds, FORMAL_FOLDS, name="outer fold")
    elif not folds:
        raise EvaluationContractError("fold_predictions must not be empty")
    if any(fold < 0 for fold in folds):
        raise EvaluationContractError("outer fold IDs must be non-negative")

    pooled_target: list[np.ndarray] = []
    pooled_prediction: list[np.ndarray] = []
    fold_sizes: dict[str, int] = {}
    for fold in sorted(folds):
        pair = folds[fold]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise EvaluationContractError(
                f"fold {fold} must contain a (target, prediction) pair"
            )
        target = _finite_vector(pair[0], name=f"fold {fold} target")
        prediction = _finite_vector(pair[1], name=f"fold {fold} prediction")
        if target.shape != prediction.shape:
            raise EvaluationContractError(
                f"fold {fold} target and prediction lengths differ"
            )
        pooled_target.append(target)
        pooled_prediction.append(prediction)
        fold_sizes[str(fold)] = int(len(target))

    result: dict[str, Any] = regression_metrics(
        np.concatenate(pooled_target), np.concatenate(pooled_prediction)
    )
    result.update(
        {
            "aggregation": "pooled_oof_within_seed",
            "fold_count": int(len(folds)),
            "fold_sizes": fold_sizes,
        }
    )
    return result


def macro_regression_metrics(
    endpoint_metrics: Mapping[str, Mapping[str, Any]],
    *,
    expected_endpoints: Sequence[str] | None,
    formal: bool = True,
) -> dict[str, Any]:
    """Compute an unweighted macro across pooled visit/interval endpoints."""

    if not isinstance(endpoint_metrics, Mapping) or not endpoint_metrics:
        raise EvaluationContractError("endpoint_metrics must be a non-empty mapping")
    observed = {str(key) for key in endpoint_metrics}
    if formal:
        if expected_endpoints is None:
            raise EvaluationContractError(
                "formal macro evaluation requires expected_endpoints"
            )
        expected = tuple(str(value) for value in expected_endpoints)
        if len(set(expected)) != len(expected):
            raise EvaluationContractError("expected_endpoints contains duplicates")
        _require_exact_keys(observed, expected, name="endpoint")
        ordered_endpoints = list(expected)
    else:
        if expected_endpoints is not None:
            expected = tuple(str(value) for value in expected_endpoints)
            _require_exact_keys(observed, expected, name="endpoint")
            ordered_endpoints = list(expected)
        else:
            ordered_endpoints = sorted(observed)

    values: dict[str, list[float]] = {metric: [] for metric in REGRESSION_METRICS}
    n_total = 0
    for endpoint in ordered_endpoints:
        metrics = endpoint_metrics[endpoint]
        if not isinstance(metrics, Mapping):
            raise EvaluationContractError(f"metrics for {endpoint} must be a mapping")
        if formal and metrics.get("aggregation") != "pooled_oof_within_seed":
            raise EvaluationContractError(
                f"{endpoint} must be pooled across OOF folds before macro averaging"
            )
        for metric in REGRESSION_METRICS:
            if metric not in metrics:
                raise EvaluationContractError(f"{endpoint} is missing {metric}")
            values[metric].append(
                _finite_scalar(metrics[metric], name=f"{endpoint} {metric}")
            )
        n_value = _finite_scalar(metrics.get("n"), name=f"{endpoint} n")
        if n_value < 1 or not float(n_value).is_integer():
            raise EvaluationContractError(f"{endpoint} n must be a positive integer")
        n_total += int(n_value)

    result: dict[str, Any] = {
        "aggregation": "unweighted_macro_across_pooled_endpoints",
        "endpoints": ordered_endpoints,
        "endpoint_count": int(len(ordered_endpoints)),
        "n_total_across_endpoints": int(n_total),
    }
    for metric, metric_values in values.items():
        result[metric] = float(np.mean(np.asarray(metric_values, dtype=np.float64)))
    return result


def _arm_seed_values(
    source: Mapping[str, Mapping[Any, Any]],
    *,
    source_name: str,
    required_arms: Sequence[str],
    formal: bool,
) -> dict[str, dict[int, float]]:
    if not isinstance(source, Mapping):
        raise EvaluationContractError(f"{source_name} must be a mapping")
    result: dict[str, dict[int, float]] = {}
    for arm in required_arms:
        if arm not in source:
            raise EvaluationContractError(f"{source_name} is missing arm {arm}")
        seed_values = _normalize_integer_mapping(
            source[arm], name=f"{source_name}[{arm}]"
        )
        if formal:
            _require_exact_keys(seed_values, FORMAL_SEEDS, name=f"{source_name} seeds")
        elif not seed_values:
            raise EvaluationContractError(f"{source_name}[{arm}] must not be empty")
        result[arm] = {
            seed: _finite_scalar(value, name=f"{source_name}[{arm}][{seed}]")
            for seed, value in seed_values.items()
        }
    reference_seeds = set(result[required_arms[0]])
    for arm in required_arms[1:]:
        _require_exact_keys(
            result[arm], reference_seeds, name=f"{source_name} paired seeds"
        )
    return result


def seed_level_effects(
    *,
    static_ftv_macro_spearman: Mapping[str, Mapping[Any, Any]],
    delta_ftv_macro_spearman: Mapping[str, Mapping[Any, Any]],
    sph_res_t0_spearman: Mapping[str, Mapping[Any, Any]],
    formal: bool = True,
) -> dict[str, Any]:
    """Compute preregistered paired representation effects E1--E4 by seed."""

    static = _arm_seed_values(
        static_ftv_macro_spearman,
        source_name="static_ftv_macro_spearman",
        required_arms=("S0", "S2"),
        formal=formal,
    )
    delta = _arm_seed_values(
        delta_ftv_macro_spearman,
        source_name="delta_ftv_macro_spearman",
        required_arms=("S0", "S2"),
        formal=formal,
    )
    sph_res = _arm_seed_values(
        sph_res_t0_spearman,
        source_name="sph_res_t0_spearman",
        required_arms=("S0", "S1", "S2"),
        formal=formal,
    )
    seeds = sorted(static["S0"])
    for other in (delta["S0"], sph_res["S0"]):
        _require_exact_keys(other, seeds, name="paired effect seeds")

    definitions = {
        "E1": "S2-S0 static FTV macro Spearman",
        "E2": "S2-S0 observed delta-FTV macro Spearman",
        "E3": "S2-S0 T0 SPH_res Spearman",
        "E4": "S2-S1 T0 SPH_res Spearman",
    }
    by_seed: dict[str, dict[str, float]] = {}
    for seed in seeds:
        by_seed[str(seed)] = {
            "E1": static["S2"][seed] - static["S0"][seed],
            "E2": delta["S2"][seed] - delta["S0"][seed],
            "E3": sph_res["S2"][seed] - sph_res["S0"][seed],
            "E4": sph_res["S2"][seed] - sph_res["S1"][seed],
        }
    mean = {
        effect: float(np.mean([by_seed[str(seed)][effect] for seed in seeds]))
        for effect in definitions
    }
    return {
        "aggregation": "paired_seed_level_effects",
        "seeds": seeds,
        "definitions": definitions,
        "by_seed": by_seed,
        "mean": mean,
    }


def _control_matrix(control: Any, *, n_rows: int) -> np.ndarray:
    matrix = np.asarray(control, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != n_rows or matrix.shape[1] < 1:
        raise EvaluationContractError(
            "control must have one row per observation and at least one column"
        )
    if not np.isfinite(matrix).all():
        raise EvaluationContractError("control must contain only finite values")
    if n_rows < matrix.shape[1] + 3:
        raise EvaluationContractError("too few observations for partial correlation")
    return matrix


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(controls), dtype=np.float64), controls])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def _residual_pearson(left: np.ndarray, right: np.ndarray, *, name: str) -> float:
    if float(np.var(left, ddof=0)) <= 0.0 or float(np.var(right, ddof=0)) <= 0.0:
        raise EvaluationContractError(f"{name} is undefined after controlling for FTV")
    value = float(pearsonr(left, right).statistic)
    if not np.isfinite(value):
        raise EvaluationContractError(f"{name} is undefined")
    return value


def partial_correlations_controlling_ftv(
    state_derived_sph: Any,
    target_sph: Any,
    ftv: Any,
) -> dict[str, float | int]:
    """Return partial Pearson and partial Spearman correlations controlling FTV.

    Partial Spearman is defined by average-ranking both variables and each FTV
    control column, linearly residualizing the ranks, then correlating residuals.
    """

    prediction = _finite_vector(
        state_derived_sph, name="state-derived SPH", minimum_rows=4
    )
    target = _finite_vector(target_sph, name="target SPH", minimum_rows=4)
    if prediction.shape != target.shape:
        raise EvaluationContractError(
            "state-derived SPH and target SPH lengths differ"
        )
    controls = _control_matrix(ftv, n_rows=len(target))

    prediction_residual = _residualize(prediction, controls)
    target_residual = _residualize(target, controls)
    partial_pearson = _residual_pearson(
        prediction_residual, target_residual, name="partial Pearson"
    )

    ranked_prediction = rankdata(prediction, method="average")
    ranked_target = rankdata(target, method="average")
    ranked_controls = np.column_stack(
        [rankdata(controls[:, column], method="average") for column in range(controls.shape[1])]
    )
    ranked_prediction_residual = _residualize(ranked_prediction, ranked_controls)
    ranked_target_residual = _residualize(ranked_target, ranked_controls)
    partial_spearman = _residual_pearson(
        ranked_prediction_residual,
        ranked_target_residual,
        name="partial Spearman",
    )
    return {
        "n": int(len(target)),
        "control_dimension": int(controls.shape[1]),
        "partial_pearson": partial_pearson,
        "partial_spearman": partial_spearman,
    }


def paired_fold_stratified_auroc_bootstrap(
    fold: Any,
    labels: Any,
    reference_probability: Any,
    comparison_probability: Any,
    *,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 260_811,
    formal: bool = True,
) -> dict[str, Any]:
    """Paired OOF binary-metric differences with resampling within each fold.

    This reproduces the prior complementarity audit's paired bootstrap: one
    patient-resampling matrix is generated separately for every outer fold and
    the fold blocks are concatenated for each draw.  AUROC and AUPRC improvements
    are ``comparison - reference``.  Brier improvement is ``reference -
    comparison``, so positive values favor the comparison for all three metrics.
    Single-class draws are omitted from AUROC/AUPRC summaries but remain valid for
    Brier.  The caller retains all patient-level arrays; only aggregates are
    returned.
    """

    y_true = _binary_vector(labels, name="labels")
    reference = _probability_vector(
        reference_probability, name="reference_probability"
    )
    comparison = _probability_vector(
        comparison_probability, name="comparison_probability"
    )
    raw_fold = np.asarray(fold)
    if raw_fold.ndim != 1:
        raise EvaluationContractError("fold must be a one-dimensional vector")
    try:
        fold_numeric = raw_fold.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise EvaluationContractError("fold values must be integers") from error
    if (
        not np.isfinite(fold_numeric).all()
        or not np.equal(fold_numeric, np.floor(fold_numeric)).all()
        or np.any(fold_numeric < 0)
    ):
        raise EvaluationContractError("fold values must be non-negative integers")
    outer_fold = fold_numeric.astype(np.int64)
    if not (y_true.shape == reference.shape == comparison.shape == outer_fold.shape):
        raise EvaluationContractError("bootstrap vectors differ in length")
    observed_folds = tuple(sorted(int(value) for value in np.unique(outer_fold)))
    if formal:
        _require_exact_keys(observed_folds, FORMAL_FOLDS, name="outer fold")
        if int(n_bootstrap) != 2_000:
            raise EvaluationContractError(
                "formal paired bootstrap requires exactly 2,000 draws"
            )
    elif int(n_bootstrap) < 1:
        raise EvaluationContractError("n_bootstrap must be positive")
    if isinstance(n_bootstrap, (bool, np.bool_)) or int(n_bootstrap) != n_bootstrap:
        raise EvaluationContractError("n_bootstrap must be an integer")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise EvaluationContractError(
            "confidence_level must lie strictly between zero and one"
        )
    if len(np.unique(y_true)) != 2:
        raise EvaluationContractError("pooled labels must contain both outcomes")

    strata: list[np.ndarray] = []
    fold_sizes: dict[str, int] = {}
    for fold_value in observed_folds:
        fold_positions = np.flatnonzero(outer_fold == fold_value)
        if len(fold_positions) == 0:
            raise EvaluationContractError(f"outer fold {fold_value} is empty")
        fold_sizes[str(fold_value)] = int(len(fold_positions))
        strata.append(fold_positions)

    reference_auroc = float(roc_auc_score(y_true, reference))
    comparison_auroc = float(roc_auc_score(y_true, comparison))
    reference_auprc = float(average_precision_score(y_true, reference))
    comparison_auprc = float(average_precision_score(y_true, comparison))
    reference_brier = float(np.mean(np.square(reference - y_true)))
    comparison_brier = float(np.mean(np.square(comparison - y_true)))

    rng = np.random.default_rng(int(seed))
    sampled_blocks = [
        rng.choice(index, size=(int(n_bootstrap), len(index)), replace=True)
        for index in strata
    ]
    sampled_indices = np.concatenate(sampled_blocks, axis=1)
    auroc_draws = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    auprc_draws = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    brier_draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for draw, selected in enumerate(sampled_indices):
        sampled_y = y_true[selected]
        sampled_reference = reference[selected]
        sampled_comparison = comparison[selected]
        brier_draws[draw] = float(
            np.mean(np.square(sampled_reference - sampled_y))
            - np.mean(np.square(sampled_comparison - sampled_y))
        )
        if len(np.unique(sampled_y)) == 2:
            auroc_draws[draw] = float(
                roc_auc_score(sampled_y, sampled_comparison)
                - roc_auc_score(sampled_y, sampled_reference)
            )
            auprc_draws[draw] = float(
                average_precision_score(sampled_y, sampled_comparison)
                - average_precision_score(sampled_y, sampled_reference)
            )

    finite_auroc = auroc_draws[np.isfinite(auroc_draws)]
    finite_auprc = auprc_draws[np.isfinite(auprc_draws)]
    finite_brier = brier_draws[np.isfinite(brier_draws)]
    if not len(finite_auroc) or not len(finite_auprc):
        raise EvaluationContractError(
            "all paired bootstrap AUROC/AUPRC draws were single-class"
        )
    if len(finite_brier) != int(n_bootstrap):
        raise EvaluationContractError("paired bootstrap produced invalid Brier draws")

    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "metrics": ["auroc", "auprc", "brier"],
        "orientation": "comparison_minus_reference",
        "auroc_orientation": "comparison_minus_reference",
        "auprc_orientation": "comparison_minus_reference",
        "brier_orientation": "reference_minus_comparison_lower_is_better",
        "aggregation": "pooled_oof_paired_patient_bootstrap",
        "stratification": "patient_within_outer_fold",
        "bootstrap_unit": "patient_within_outer_fold",
        "ci_method": "percentile",
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "fold_count": int(len(observed_folds)),
        "fold_sizes": fold_sizes,
        "n_bootstrap": int(n_bootstrap),
        # Compatibility alias: historically this count referred to AUROC.
        "n_valid_bootstrap": int(len(finite_auroc)),
        "n_valid_auroc_bootstrap": int(len(finite_auroc)),
        "n_valid_auprc_bootstrap": int(len(finite_auprc)),
        "n_valid_brier_bootstrap": int(len(finite_brier)),
        "confidence_level": float(confidence_level),
        "bootstrap_seed": int(seed),
        "reference_auroc": reference_auroc,
        "comparison_auroc": comparison_auroc,
        "delta_auroc": comparison_auroc - reference_auroc,
        "delta_auroc_bootstrap_mean": float(np.mean(finite_auroc)),
        "delta_auroc_ci_lower": float(np.quantile(finite_auroc, alpha)),
        "delta_auroc_ci_upper": float(np.quantile(finite_auroc, 1.0 - alpha)),
        "delta_auroc_bootstrap_probability_positive": float(
            np.mean(finite_auroc > 0.0)
        ),
        # Compatibility alias used by the existing report.
        "bootstrap_probability_positive": float(np.mean(finite_auroc > 0.0)),
        "reference_auprc": reference_auprc,
        "comparison_auprc": comparison_auprc,
        "delta_auprc": comparison_auprc - reference_auprc,
        "delta_auprc_bootstrap_mean": float(np.mean(finite_auprc)),
        "delta_auprc_ci_lower": float(np.quantile(finite_auprc, alpha)),
        "delta_auprc_ci_upper": float(np.quantile(finite_auprc, 1.0 - alpha)),
        "delta_auprc_bootstrap_probability_positive": float(
            np.mean(finite_auprc > 0.0)
        ),
        "reference_brier": reference_brier,
        "comparison_brier": comparison_brier,
        "brier_improvement": reference_brier - comparison_brier,
        "brier_improvement_bootstrap_mean": float(np.mean(finite_brier)),
        "brier_improvement_ci_lower": float(np.quantile(finite_brier, alpha)),
        "brier_improvement_ci_upper": float(
            np.quantile(finite_brier, 1.0 - alpha)
        ),
        "brier_improvement_bootstrap_probability_positive": float(
            np.mean(finite_brier > 0.0)
        ),
    }


def _optimization_safety_flags(
    values: Mapping[Any, Mapping[Any, Any]], *, formal: bool
) -> tuple[list[bool], dict[str, dict[str, bool]]]:
    seeds = _normalize_integer_mapping(values, name="optimization_safety")
    if formal:
        _require_exact_keys(seeds, FORMAL_SEEDS, name="optimization safety seeds")
    elif not seeds:
        raise EvaluationContractError("optimization_safety must not be empty")
    flags: list[bool] = []
    normalized: dict[str, dict[str, bool]] = {}
    for seed in sorted(seeds):
        folds = _normalize_integer_mapping(
            seeds[seed], name=f"optimization_safety[{seed}]"
        )
        if formal:
            _require_exact_keys(
                folds, FORMAL_FOLDS, name=f"optimization safety folds for {seed}"
            )
        elif not folds:
            raise EvaluationContractError(
                f"optimization_safety[{seed}] must not be empty"
            )
        normalized[str(seed)] = {}
        for fold in sorted(folds):
            value = folds[fold]
            if not isinstance(value, (bool, np.bool_)):
                raise EvaluationContractError(
                    f"optimization_safety[{seed}][{fold}] must be boolean"
                )
            flag = bool(value)
            flags.append(flag)
            normalized[str(seed)][str(fold)] = flag
    return flags, normalized


def evaluate_decision_gates(
    *,
    effects: Mapping[str, Any],
    optimization_safety: Mapping[Any, Mapping[Any, Any]],
    downstream_delta_auroc: Mapping[str, Mapping[Any, Any]],
    formal: bool = True,
) -> dict[str, Any]:
    """Evaluate Gates A--D and the attachment's final classifications.

    The otherwise qualitative Gate-A phrase "systematic degradation" is made
    auditable as degradation (E2 < 0) in *both* preregistered seeds.  No new
    magnitude threshold is introduced for observed delta-FTV.
    """

    if not isinstance(effects, Mapping) or not isinstance(
        effects.get("by_seed"), Mapping
    ):
        raise EvaluationContractError("effects must be seed_level_effects output")
    raw_by_seed = _normalize_integer_mapping(
        effects["by_seed"], name="effects.by_seed"
    )
    if formal:
        _require_exact_keys(raw_by_seed, FORMAL_SEEDS, name="effect seeds")
    elif not raw_by_seed:
        raise EvaluationContractError("effects.by_seed must not be empty")
    by_seed: dict[int, dict[str, float]] = {}
    for seed, values in raw_by_seed.items():
        if not isinstance(values, Mapping):
            raise EvaluationContractError(f"effects for seed {seed} must be a mapping")
        by_seed[seed] = {
            effect: _finite_scalar(values.get(effect), name=f"seed {seed} {effect}")
            for effect in ("E1", "E2", "E3", "E4")
        }

    safety_flags, normalized_safety = _optimization_safety_flags(
        optimization_safety, formal=formal
    )
    if set(by_seed) != {int(seed) for seed in normalized_safety}:
        raise EvaluationContractError("effect and optimization-safety seeds differ")

    seeds = sorted(by_seed)
    e1 = {seed: by_seed[seed]["E1"] for seed in seeds}
    e2 = {seed: by_seed[seed]["E2"] for seed in seeds}
    e3 = {seed: by_seed[seed]["E3"] for seed in seeds}
    e4 = {seed: by_seed[seed]["E4"] for seed in seeds}

    static_safe = all(value >= -0.03 for value in e1.values())
    delta_systematic_degradation = all(value < 0.0 for value in e2.values())
    safety_pass_count = int(sum(safety_flags))
    required_safety_count = 9 if formal else int(np.ceil(0.9 * len(safety_flags)))
    optimization_safe = safety_pass_count >= required_safety_count
    gate_a_pass = static_safe and not delta_systematic_degradation and optimization_safe

    e3_mean = float(np.mean(list(e3.values())))
    residual_positive_both = all(value > 0.0 for value in e3.values())
    gate_b_pass = residual_positive_both and e3_mean >= 0.05
    gate_b_strong = residual_positive_both and e3_mean >= 0.10

    gate_c_pass = all(value > 0.0 for value in e4.values())

    if not isinstance(downstream_delta_auroc, Mapping):
        raise EvaluationContractError("downstream_delta_auroc must be a mapping")
    observed_timings = {str(key) for key in downstream_delta_auroc}
    if formal:
        _require_exact_keys(observed_timings, GATE_D_TIMINGS, name="Gate D timing")
        timings = list(GATE_D_TIMINGS)
    else:
        if not observed_timings:
            raise EvaluationContractError("downstream_delta_auroc must not be empty")
        timings = sorted(observed_timings)
    normalized_downstream: dict[str, dict[str, float]] = {}
    qualifying_timings: list[str] = []
    strong_timings: list[str] = []
    for timing in timings:
        seed_values = _normalize_integer_mapping(
            downstream_delta_auroc[timing],
            name=f"downstream_delta_auroc[{timing}]",
        )
        _require_exact_keys(seed_values, seeds, name=f"Gate D seeds at {timing}")
        values = {
            seed: _finite_scalar(value, name=f"Gate D {timing} seed {seed}")
            for seed, value in seed_values.items()
        }
        normalized_downstream[timing] = {
            str(seed): values[seed] for seed in sorted(values)
        }
        if all(value > 0.0 for value in values.values()):
            qualifying_timings.append(timing)
            if float(np.mean(list(values.values()))) >= 0.03:
                strong_timings.append(timing)
    gate_d_pass = bool(qualifying_timings)
    gate_d_strong = bool(strong_timings)

    if not gate_a_pass or not gate_b_pass:
        representation_classification = "SPH_GROUNDING_NOT_USEFUL"
    elif not gate_c_pass:
        representation_classification = "RAW_SPH_SUFFICIENT"
    else:
        representation_classification = "RESIDUAL_SPH_GROUNDING_VALIDATED"

    downstream_classification: str | None = None
    if gate_a_pass and gate_b_pass and gate_c_pass:
        downstream_classification = (
            "RESIDUAL_SPH_COMPLEMENTARITY_SUPPORTED"
            if gate_d_pass
            else "MORPHOLOGY_ORGANIZED_BUT_NO_PCR_COMPLEMENTARITY"
        )

    labels = [representation_classification]
    if downstream_classification is not None:
        labels.append(downstream_classification)

    return {
        "gates": {
            "A": {
                "name": "RESPONSE SAFETY",
                "passed": gate_a_pass,
                "pass_label": "RESPONSE_STATE_PRESERVED",
                "static_ftv_both_seeds_ge_minus_0_03": static_safe,
                "delta_ftv_systematic_degradation": delta_systematic_degradation,
                "delta_ftv_systematic_degradation_rule": "E2 < 0 in both seeds",
                "optimization_safety_pass_count": safety_pass_count,
                "optimization_safety_total": int(len(safety_flags)),
                "optimization_safety_required": required_safety_count,
                "optimization_safety_passed": optimization_safe,
                "effects_by_seed": {
                    str(seed): {"E1": e1[seed], "E2": e2[seed]} for seed in seeds
                },
            },
            "B": {
                "name": "RESIDUAL SPH ORGANIZATION",
                "passed": gate_b_pass,
                "strong_form_passed": gate_b_strong,
                "pass_label": "RESIDUAL_SPH_GROUNDING_WORKS",
                "both_seed_gains_positive": residual_positive_both,
                "mean_gain": e3_mean,
                "minimum_mean_gain": 0.05,
                "strong_mean_gain": 0.10,
                "effects_by_seed": {str(seed): e3[seed] for seed in seeds},
            },
            "C": {
                "name": "RESIDUAL BENEFIT OVER RAW SPH",
                "passed": gate_c_pass,
                "pass_label": "RESIDUAL_TARGET_IS_PREFERABLE",
                "effects_by_seed": {str(seed): e4[seed] for seed in seeds},
            },
            "D": {
                "name": "DOWNSTREAM COMPLEMENTARITY",
                "passed": gate_d_pass,
                "strong_form_passed": gate_d_strong,
                "pass_label": "SPH_GROUNDED_STATE_ADDS_BEYOND_FTV",
                "effects_by_timing_and_seed": normalized_downstream,
                "qualifying_timings": qualifying_timings,
                "strong_timings": strong_timings,
                "strong_mean_delta_auroc": 0.03,
            },
        },
        "optimization_safety": normalized_safety,
        "classification": {
            "representation": representation_classification,
            "downstream": downstream_classification,
            "labels": labels,
            "five_seed_confirmation_justified": bool(
                gate_a_pass and gate_b_pass and gate_c_pass
            ),
        },
    }


__all__ = [
    "EvaluationContractError",
    "FORMAL_FOLDS",
    "FORMAL_SEEDS",
    "GATE_D_TIMINGS",
    "OBSERVED_DELTA_INTERVALS",
    "REGRESSION_METRICS",
    "STATIC_VISITS",
    "evaluate_decision_gates",
    "macro_regression_metrics",
    "paired_fold_stratified_auroc_bootstrap",
    "partial_correlations_controlling_ftv",
    "pooled_oof_regression",
    "regression_metrics",
    "seed_level_effects",
]
