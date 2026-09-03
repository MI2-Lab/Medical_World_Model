from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_sph.evaluation import (  # noqa: E402
    EvaluationContractError,
    FORMAL_FOLDS,
    FORMAL_SEEDS,
    GATE_D_TIMINGS,
    STATIC_VISITS,
    evaluate_decision_gates,
    macro_regression_metrics,
    paired_fold_stratified_auroc_bootstrap,
    partial_correlations_controlling_ftv,
    pooled_oof_regression,
    regression_metrics,
    seed_level_effects,
)


def test_regression_metrics_perfect_natural_unit_prediction() -> None:
    target = np.array([1.0, 2.0, 4.0, 8.0])
    result = regression_metrics(target, target.copy())

    assert result == pytest.approx(
        {
            "n": 4,
            "spearman": 1.0,
            "pearson": 1.0,
            "natural_r2": 1.0,
            "rmse": 0.0,
            "mae": 0.0,
            "variance_ratio": 1.0,
        }
    )


def test_regression_metrics_keep_natural_r2_and_variance_ratio_distinct() -> None:
    target = np.array([-2.0, -1.0, 1.0, 2.0])
    prediction = 2.0 * target
    result = regression_metrics(target, prediction)

    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson"] == pytest.approx(1.0)
    assert result["variance_ratio"] == pytest.approx(4.0)
    assert result["natural_r2"] < 1.0
    assert result["rmse"] > 0.0


@pytest.mark.parametrize(
    ("target", "prediction", "message"),
    [
        ([1.0, 2.0], [1.0], "lengths differ"),
        ([1.0, np.nan], [1.0, 2.0], "finite"),
        ([1.0, 1.0], [1.0, 2.0], "target variance"),
        ([1.0, 2.0], [1.0, 1.0], "prediction variance"),
    ],
)
def test_regression_metrics_fail_closed(
    target: list[float], prediction: list[float], message: str
) -> None:
    with pytest.raises(EvaluationContractError, match=message):
        regression_metrics(target, prediction)


def _five_fold_regression() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in FORMAL_FOLDS:
        target = np.arange(4, dtype=np.float64) + 10.0 * fold
        prediction = target.copy() if fold != 4 else target[::-1].copy()
        folds[fold] = target, prediction
    return folds


def test_pooled_oof_metrics_are_computed_after_concatenating_five_folds() -> None:
    folds = _five_fold_regression()
    result = pooled_oof_regression(folds)
    target = np.concatenate([folds[fold][0] for fold in FORMAL_FOLDS])
    prediction = np.concatenate([folds[fold][1] for fold in FORMAL_FOLDS])
    mean_fold_spearman = np.mean(
        [spearmanr(*folds[fold]).statistic for fold in FORMAL_FOLDS]
    )

    assert result["aggregation"] == "pooled_oof_within_seed"
    assert result["fold_count"] == 5
    assert result["n"] == 20
    assert result["spearman"] == pytest.approx(
        spearmanr(target, prediction).statistic
    )
    assert result["spearman"] != pytest.approx(mean_fold_spearman)
    assert "fold_metrics" not in result


def test_pooled_oof_formal_mode_requires_all_and_only_five_folds() -> None:
    folds = _five_fold_regression()
    folds.pop(4)
    with pytest.raises(EvaluationContractError, match="coverage drifted"):
        pooled_oof_regression(folds)


def _pooled_metric(spearman: float, *, n: int = 75) -> dict[str, object]:
    return {
        "aggregation": "pooled_oof_within_seed",
        "n": n,
        "spearman": spearman,
        "pearson": spearman - 0.01,
        "natural_r2": spearman - 0.02,
        "rmse": 1.0 - spearman,
        "mae": 0.8 - spearman,
        "variance_ratio": 1.0 + spearman,
    }


def test_macro_is_unweighted_across_already_pooled_visit_endpoints() -> None:
    endpoint_metrics = {
        visit: _pooled_metric(value)
        for visit, value in zip(STATIC_VISITS, (0.1, 0.2, 0.3, 0.4), strict=True)
    }
    result = macro_regression_metrics(
        endpoint_metrics, expected_endpoints=STATIC_VISITS
    )

    assert result["aggregation"] == "unweighted_macro_across_pooled_endpoints"
    assert result["endpoints"] == list(STATIC_VISITS)
    assert result["spearman"] == pytest.approx(0.25)
    assert result["n_total_across_endpoints"] == 300


def test_formal_macro_rejects_fold_metric_or_missing_visit() -> None:
    endpoint_metrics = {visit: _pooled_metric(0.2) for visit in STATIC_VISITS}
    endpoint_metrics["T0"] = dict(endpoint_metrics["T0"])
    endpoint_metrics["T0"]["aggregation"] = "mean_across_folds"
    with pytest.raises(EvaluationContractError, match="pooled across OOF folds"):
        macro_regression_metrics(endpoint_metrics, expected_endpoints=STATIC_VISITS)

    endpoint_metrics = {visit: _pooled_metric(0.2) for visit in STATIC_VISITS[:-1]}
    with pytest.raises(EvaluationContractError, match="coverage drifted"):
        macro_regression_metrics(endpoint_metrics, expected_endpoints=STATIC_VISITS)


def _effects(
    *,
    e1: tuple[float, float] = (-0.01, 0.02),
    e2: tuple[float, float] = (0.01, 0.02),
    e3: tuple[float, float] = (0.12, 0.10),
    e4: tuple[float, float] = (0.08, 0.06),
) -> dict[str, object]:
    by_seed = {
        str(seed): {
            "E1": e1[index],
            "E2": e2[index],
            "E3": e3[index],
            "E4": e4[index],
        }
        for index, seed in enumerate(FORMAL_SEEDS)
    }
    return {"by_seed": by_seed}


def test_seed_level_effects_compute_e1_through_e4_without_seed_averaging() -> None:
    result = seed_level_effects(
        static_ftv_macro_spearman={
            "S0": {2026: 0.50, 3026: 0.45},
            "S2": {2026: 0.49, 3026: 0.48},
        },
        delta_ftv_macro_spearman={
            "S0": {2026: 0.30, 3026: 0.25},
            "S2": {2026: 0.32, 3026: 0.24},
        },
        sph_res_t0_spearman={
            "S0": {2026: 0.20, 3026: 0.19},
            "S1": {2026: 0.23, 3026: 0.20},
            "S2": {2026: 0.32, 3026: 0.27},
        },
    )

    assert result["seeds"] == [2026, 3026]
    assert result["by_seed"]["2026"] == pytest.approx(
        {"E1": -0.01, "E2": 0.02, "E3": 0.12, "E4": 0.09}
    )
    assert result["by_seed"]["3026"] == pytest.approx(
        {"E1": 0.03, "E2": -0.01, "E3": 0.08, "E4": 0.07}
    )
    assert result["mean"]["E3"] == pytest.approx(0.10)


def test_seed_level_effects_require_both_formal_seeds() -> None:
    with pytest.raises(EvaluationContractError, match="coverage drifted"):
        seed_level_effects(
            static_ftv_macro_spearman={
                "S0": {2026: 0.5},
                "S2": {2026: 0.5},
            },
            delta_ftv_macro_spearman={
                "S0": {2026: 0.3},
                "S2": {2026: 0.3},
            },
            sph_res_t0_spearman={
                "S0": {2026: 0.2},
                "S1": {2026: 0.2},
                "S2": {2026: 0.3},
            },
        )


def test_partial_correlations_remove_ftv_and_recover_shared_morphology() -> None:
    ftv = np.linspace(-2.0, 2.0, 101)
    morphology = np.sin(np.linspace(0.0, 8.0 * np.pi, 101))
    state_derived_sph = 5.0 * ftv + morphology
    target_sph = -2.0 * ftv + 3.0 * morphology

    result = partial_correlations_controlling_ftv(
        state_derived_sph, target_sph, ftv
    )

    assert result["n"] == 101
    assert result["control_dimension"] == 1
    assert result["partial_pearson"] == pytest.approx(1.0)
    # Rank residualization is intentionally not the same operation as ranking
    # the linear residuals; it still recovers a positive morphology association.
    assert result["partial_spearman"] > 0.5


def _bootstrap_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folds = np.repeat(np.asarray(FORMAL_FOLDS), 4)
    labels = np.tile(np.array([0, 0, 1, 1]), len(FORMAL_FOLDS))
    reference = np.tile(np.array([0.5, 0.5, 0.5, 0.5]), len(FORMAL_FOLDS))
    comparison = np.where(labels == 1, 0.9, 0.1)
    return folds, labels, reference, comparison


def test_formal_paired_bootstrap_is_seeded_stratified_and_aggregate_only() -> None:
    folds, labels, reference, comparison = _bootstrap_vectors()
    first = paired_fold_stratified_auroc_bootstrap(
        folds, labels, reference, comparison
    )
    second = paired_fold_stratified_auroc_bootstrap(
        folds, labels, reference, comparison
    )

    assert first == second
    assert first["n_bootstrap"] == 2_000
    assert first["bootstrap_seed"] == 260_811
    assert first["stratification"] == "patient_within_outer_fold"
    assert first["n_valid_bootstrap"] == 2_000
    assert first["n_valid_auroc_bootstrap"] == 2_000
    assert first["n_valid_auprc_bootstrap"] == 2_000
    assert first["n_valid_brier_bootstrap"] == 2_000
    assert first["orientation"] == "comparison_minus_reference"
    assert first["auprc_orientation"] == "comparison_minus_reference"
    assert first["brier_orientation"] == (
        "reference_minus_comparison_lower_is_better"
    )
    assert first["delta_auroc"] == pytest.approx(0.5)
    assert first["delta_auroc_ci_lower"] == pytest.approx(0.5)
    assert first["delta_auroc_ci_upper"] == pytest.approx(0.5)
    assert first["reference_auprc"] == pytest.approx(0.5)
    assert first["comparison_auprc"] == pytest.approx(1.0)
    assert first["delta_auprc"] == pytest.approx(0.5)
    assert first["reference_brier"] == pytest.approx(0.25)
    assert first["comparison_brier"] == pytest.approx(0.01)
    assert first["brier_improvement"] == pytest.approx(0.24)
    assert first["brier_improvement_ci_lower"] == pytest.approx(0.24)
    assert first["brier_improvement_ci_upper"] == pytest.approx(0.24)
    assert "draws" not in first
    assert not any(isinstance(value, np.ndarray) for value in first.values())


def test_paired_bootstrap_matches_prior_fold_block_sampling_and_omits_rank_draws() -> None:
    folds = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 0, 1])
    reference = np.array([0.30, 0.60, 0.45, 0.70])
    comparison = np.array([0.10, 0.80, 0.30, 0.90])
    n_bootstrap = 128
    confidence_level = 0.90
    seed = 37

    result = paired_fold_stratified_auroc_bootstrap(
        folds,
        labels,
        reference,
        comparison,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
        formal=False,
    )
    repeated = paired_fold_stratified_auroc_bootstrap(
        folds,
        labels,
        reference,
        comparison,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
        formal=False,
    )
    next_comparison = paired_fold_stratified_auroc_bootstrap(
        folds,
        labels,
        reference,
        comparison,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed + 1,
        formal=False,
    )

    rng = np.random.default_rng(seed)
    blocks = [
        rng.choice(index, size=(n_bootstrap, len(index)), replace=True)
        for index in (np.array([0, 1]), np.array([2, 3]))
    ]
    expected_auroc: list[float] = []
    expected_auprc: list[float] = []
    expected_brier: list[float] = []
    for selected in np.concatenate(blocks, axis=1):
        sampled_labels = labels[selected]
        sampled_reference = reference[selected]
        sampled_comparison = comparison[selected]
        expected_brier.append(
            float(
                np.mean(np.square(sampled_reference - sampled_labels))
                - np.mean(np.square(sampled_comparison - sampled_labels))
            )
        )
        if len(np.unique(sampled_labels)) == 2:
            expected_auroc.append(
                float(
                    roc_auc_score(sampled_labels, sampled_comparison)
                    - roc_auc_score(sampled_labels, sampled_reference)
                )
            )
            expected_auprc.append(
                float(
                    average_precision_score(sampled_labels, sampled_comparison)
                    - average_precision_score(sampled_labels, sampled_reference)
                )
            )

    alpha = (1.0 - confidence_level) / 2.0
    assert result == repeated
    assert result["bootstrap_seed"] == seed
    assert next_comparison["bootstrap_seed"] == seed + 1
    assert next_comparison["brier_improvement_bootstrap_mean"] != pytest.approx(
        result["brier_improvement_bootstrap_mean"]
    )
    assert 0 < len(expected_auroc) < n_bootstrap
    assert result["n_valid_auroc_bootstrap"] == len(expected_auroc)
    assert result["n_valid_auprc_bootstrap"] == len(expected_auprc)
    assert result["n_valid_brier_bootstrap"] == n_bootstrap
    assert result["delta_auroc_bootstrap_mean"] == pytest.approx(
        np.mean(expected_auroc)
    )
    assert result["delta_auprc_ci_lower"] == pytest.approx(
        np.quantile(expected_auprc, alpha)
    )
    assert result["delta_auprc_ci_upper"] == pytest.approx(
        np.quantile(expected_auprc, 1.0 - alpha)
    )
    assert result["brier_improvement_bootstrap_mean"] == pytest.approx(
        np.mean(expected_brier)
    )
    assert result["brier_improvement_ci_lower"] == pytest.approx(
        np.quantile(expected_brier, alpha)
    )
    assert result["brier_improvement_ci_upper"] == pytest.approx(
        np.quantile(expected_brier, 1.0 - alpha)
    )
    assert not any(isinstance(value, np.ndarray) for value in result.values())


def test_formal_paired_bootstrap_requires_exact_draw_and_fold_contract() -> None:
    folds, labels, reference, comparison = _bootstrap_vectors()
    with pytest.raises(EvaluationContractError, match="exactly 2,000"):
        paired_fold_stratified_auroc_bootstrap(
            folds,
            labels,
            reference,
            comparison,
            n_bootstrap=1_999,
        )
    with pytest.raises(EvaluationContractError, match="coverage drifted"):
        keep = folds != 4
        paired_fold_stratified_auroc_bootstrap(
            folds[keep], labels[keep], reference[keep], comparison[keep]
        )


def _all_safe() -> dict[int, dict[int, bool]]:
    return {
        seed: {fold: True for fold in FORMAL_FOLDS} for seed in FORMAL_SEEDS
    }


def _downstream(
    *, t0: tuple[float, float] = (0.04, 0.03),
    t01: tuple[float, float] = (-0.01, 0.02),
    t02: tuple[float, float] = (0.00, 0.01),
) -> dict[str, dict[int, float]]:
    values = {"T0": t0, "T0-T1": t01, "T0-T2": t02}
    return {
        timing: {seed: pair[index] for index, seed in enumerate(FORMAL_SEEDS)}
        for timing, pair in values.items()
    }


def test_all_decision_gates_and_strong_forms_pass() -> None:
    result = evaluate_decision_gates(
        effects=_effects(),
        optimization_safety=_all_safe(),
        downstream_delta_auroc=_downstream(),
    )

    assert set(result["gates"]) == {"A", "B", "C", "D"}
    assert all(result["gates"][gate]["passed"] for gate in "ABCD")
    assert result["gates"]["B"]["strong_form_passed"]
    assert result["gates"]["D"]["strong_form_passed"]
    assert result["classification"]["representation"] == (
        "RESIDUAL_SPH_GROUNDING_VALIDATED"
    )
    assert result["classification"]["downstream"] == (
        "RESIDUAL_SPH_COMPLEMENTARITY_SUPPORTED"
    )
    assert result["classification"]["five_seed_confirmation_justified"]


def test_gate_a_requires_nine_of_ten_and_no_two_seed_delta_degradation() -> None:
    safety = _all_safe()
    safety[2026][0] = False
    exactly_nine = evaluate_decision_gates(
        effects=_effects(e2=(0.01, -0.01)),
        optimization_safety=safety,
        downstream_delta_auroc=_downstream(),
    )
    assert exactly_nine["gates"]["A"]["passed"]

    safety[3026][0] = False
    only_eight = evaluate_decision_gates(
        effects=_effects(e2=(0.01, -0.01)),
        optimization_safety=safety,
        downstream_delta_auroc=_downstream(),
    )
    assert not only_eight["gates"]["A"]["passed"]

    systematic_delta_loss = evaluate_decision_gates(
        effects=_effects(e2=(-0.001, -0.02)),
        optimization_safety=_all_safe(),
        downstream_delta_auroc=_downstream(),
    )
    assert systematic_delta_loss["gates"]["A"][
        "delta_ftv_systematic_degradation"
    ]
    assert not systematic_delta_loss["gates"]["A"]["passed"]


def test_gate_b_minimum_is_distinct_from_strong_form() -> None:
    result = evaluate_decision_gates(
        effects=_effects(e3=(0.06, 0.05)),
        optimization_safety=_all_safe(),
        downstream_delta_auroc=_downstream(t0=(0.02, 0.02)),
    )
    assert result["gates"]["B"]["passed"]
    assert not result["gates"]["B"]["strong_form_passed"]
    assert result["gates"]["D"]["passed"]
    assert not result["gates"]["D"]["strong_form_passed"]


def test_raw_sph_sufficient_and_no_complementarity_classifications() -> None:
    raw_sufficient = evaluate_decision_gates(
        effects=_effects(e4=(0.01, -0.01)),
        optimization_safety=_all_safe(),
        downstream_delta_auroc=_downstream(),
    )
    assert not raw_sufficient["gates"]["C"]["passed"]
    assert raw_sufficient["classification"]["representation"] == "RAW_SPH_SUFFICIENT"
    assert raw_sufficient["classification"]["downstream"] is None

    raw_sufficient_no_pcr = evaluate_decision_gates(
        effects=_effects(e4=(0.01, -0.01)),
        optimization_safety=_all_safe(),
        downstream_delta_auroc=_downstream(
            t0=(-0.01, 0.01), t01=(0.01, -0.01), t02=(0.0, 0.02)
        ),
    )
    assert raw_sufficient_no_pcr["classification"]["representation"] == "RAW_SPH_SUFFICIENT"
    assert raw_sufficient_no_pcr["classification"]["downstream"] is None

    no_pcr = evaluate_decision_gates(
        effects=_effects(),
        optimization_safety=_all_safe(),
        downstream_delta_auroc=_downstream(
            t0=(-0.01, 0.01), t01=(0.01, -0.01), t02=(0.0, 0.02)
        ),
    )
    assert no_pcr["classification"]["representation"] == (
        "RESIDUAL_SPH_GROUNDING_VALIDATED"
    )
    assert no_pcr["classification"]["downstream"] == (
        "MORPHOLOGY_ORGANIZED_BUT_NO_PCR_COMPLEMENTARITY"
    )
    assert no_pcr["classification"]["five_seed_confirmation_justified"]


def test_failed_residual_gain_classifies_grounding_not_useful() -> None:
    result = evaluate_decision_gates(
        effects=_effects(e3=(0.03, 0.02)),
        optimization_safety=_all_safe(),
        downstream_delta_auroc=_downstream(),
    )
    assert not result["gates"]["B"]["passed"]
    assert result["classification"]["representation"] == "SPH_GROUNDING_NOT_USEFUL"


def test_gate_d_formal_mode_requires_exact_preregistered_timings() -> None:
    downstream = _downstream()
    downstream.pop(GATE_D_TIMINGS[-1])
    with pytest.raises(EvaluationContractError, match="coverage drifted"):
        evaluate_decision_gates(
            effects=_effects(),
            optimization_safety=_all_safe(),
            downstream_delta_auroc=downstream,
        )
