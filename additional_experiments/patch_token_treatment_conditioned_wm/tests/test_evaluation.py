from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from patch_token_wm.evaluation import (  # noqa: E402
    FINAL_A,
    FINAL_B,
    FINAL_C,
    FINAL_D,
    INCOMPLETE,
    INCOMPLETE_FINAL,
    LOGISTIC_CS,
    RIDGE_ALPHAS,
    FoldSafeTokenSummarizer,
    classify_final,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_gate_d,
    fit_fold_safe_logistic,
    fit_fold_safe_ridge,
    fractional_weighted_token_mean,
    paired_metric_bootstrap,
    paired_pcr_bootstrap,
    pcr_metrics,
    regression_metrics,
)


def test_fractional_weighted_mean_is_exact_and_keeps_visit_axis() -> None:
    tokens = np.zeros((1, 4, 500, 128), dtype=np.float32)
    tokens[:, :, 0, :] = 10.0
    tokens[:, :, 1, :] = 20.0
    weights = np.zeros(500, dtype=np.float64)
    weights[0] = 0.25
    weights[1] = 0.75

    result = fractional_weighted_token_mean(tokens, weights)

    assert result.shape == (1, 4, 128)
    np.testing.assert_allclose(result, 17.5, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="positive total weight"):
        fractional_weighted_token_mean(tokens, np.zeros(500))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        fractional_weighted_token_mean(tokens, np.full(500, 2.0))


def test_token_summary_pca_is_train_only_label_free_and_locked_192d() -> None:
    # 17 patients x 4 visits gives 68 centered observations, enough to identify
    # the locked 64 components without weakening the production tensor contract.
    rng = np.random.default_rng(22)
    train_tokens = rng.standard_normal((17, 4, 500, 128), dtype=np.float32)
    test_tokens = rng.standard_normal((2, 4, 500, 128), dtype=np.float32) + 40.0
    weights = np.linspace(0.05, 1.0, 500, dtype=np.float64)
    patient_ids = [f"train-{index:02d}" for index in range(17)]
    summarizer = FoldSafeTokenSummarizer(outer_fold=3, random_state=9)

    train_summary = summarizer.fit_transform(
        train_tokens, weights, train_patient_ids=patient_ids
    )
    test_parts = summarizer.transform_parts(test_tokens, weights)

    assert train_summary.shape == (17, 4, 192)
    assert test_parts.weighted_mean.shape == (2, 4, 128)
    assert test_parts.pca_scores.shape == (2, 4, 64)
    assert test_parts.primary.shape == (2, 4, 192)
    np.testing.assert_allclose(
        train_summary[..., :128],
        fractional_weighted_token_mean(train_tokens, weights),
        rtol=1e-12,
        atol=1e-12,
    )
    normalized = weights / weights.sum()
    expected_train_pca_mean = (
        (train_tokens * np.sqrt(normalized).astype(np.float32)[None, None, :, None])
        .reshape(68, -1)
        .mean(axis=0)
    )
    np.testing.assert_allclose(
        summarizer.pca_mean_, expected_train_pca_mean, rtol=1e-5, atol=1e-7
    )
    combined = np.concatenate((train_tokens, test_tokens), axis=0)
    combined_mean = (
        (combined * np.sqrt(normalized).astype(np.float32)[None, None, :, None])
        .reshape(76, -1)
        .mean(axis=0)
    )
    assert not np.allclose(summarizer.pca_mean_, combined_mean)
    provenance = summarizer.provenance
    assert provenance["fit_scope"] == "outer_train_only"
    assert provenance["labels_used"] is False
    assert provenance["attention_used"] is False
    assert provenance["pca_components"] == 64
    assert provenance["primary_summary_dim"] == 192
    assert provenance["train_patient_order_sha256"]
    signature = inspect.signature(FoldSafeTokenSummarizer.fit)
    assert not any(
        term in parameter.lower()
        for parameter in signature.parameters
        for term in ("label", "target", "outcome", "pcr")
    )
    with pytest.raises(RuntimeError, match="single-fit"):
        summarizer.fit(train_tokens, weights)


def test_token_summary_rejects_nontrain_fit_and_component_drift() -> None:
    with pytest.raises(ValueError, match="locked to 64"):
        FoldSafeTokenSummarizer(outer_fold=0, n_components=32)
    tokens = np.zeros((17, 4, 500, 128), dtype=np.float32)
    summarizer = FoldSafeTokenSummarizer(outer_fold=0)
    with pytest.raises(ValueError, match="split='train'"):
        summarizer.fit(tokens, np.ones(500), split="val")


def test_regression_and_pcr_metrics_have_known_values() -> None:
    truth = np.asarray([0.0, 1.0, 2.0, 3.0])
    prediction = 0.25 + 0.5 * truth
    metrics = regression_metrics(truth, prediction)

    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["calibration_slope"] == pytest.approx(0.5)
    assert metrics["prediction_target_variance_ratio"] == pytest.approx(0.25)
    expected_r2 = (
        1.0
        - np.square(prediction - truth).sum() / np.square(truth - truth.mean()).sum()
    )
    assert metrics["natural_r2"] == pytest.approx(expected_r2)
    assert metrics["rmse"] == pytest.approx(
        np.sqrt(np.square(prediction - truth).mean())
    )
    assert metrics["mae"] == pytest.approx(np.abs(prediction - truth).mean())

    binary = pcr_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert binary["auroc"] == pytest.approx(1.0)
    assert binary["auprc"] == pytest.approx(1.0)
    assert binary["brier"] == pytest.approx(0.085)


def test_ridge_uses_train_scaler_validation_tie_break_and_test_once() -> None:
    train_x = np.arange(24, dtype=float).reshape(8, 3)
    validation_x = np.arange(12, dtype=float).reshape(4, 3) + 100.0
    fitted = fit_fold_safe_ridge(
        train_x,
        np.full(8, 0.25),
        validation_x,
        np.full(4, 0.25),
        outer_fold=2,
    )

    np.testing.assert_allclose(fitted.scaler.mean_, train_x.mean(axis=0))
    assert not np.allclose(
        fitted.scaler.mean_, np.concatenate((train_x, validation_x)).mean(axis=0)
    )
    assert tuple(score.alpha for score in fitted.candidate_scores) == RIDGE_ALPHAS
    assert fitted.selected_alpha == pytest.approx(1e-4)
    assert fitted.provenance["refit_after_selection"] is False
    prediction = fitted.predict_test_once(np.ones((3, 3)))
    assert prediction.shape == (3,)
    assert fitted.test_prediction_calls == 1
    with pytest.raises(RuntimeError, match="single-use"):
        fitted.predict_test_once(np.ones((3, 3)))
    with pytest.raises(ValueError, match="locked"):
        fit_fold_safe_ridge(
            train_x,
            np.full(8, 0.25),
            validation_x,
            np.full(4, 0.25),
            alphas=[1.0],
        )


def test_logistic_is_l2_liblinear_train_scaled_smallest_c_and_test_once() -> None:
    train_x = np.zeros((8, 2), dtype=float)
    train_y = np.asarray([0, 1] * 4)
    validation_x = np.full((6, 2), 20.0)
    validation_y = np.asarray([0, 1] * 3)
    fitted = fit_fold_safe_logistic(
        train_x,
        train_y,
        validation_x,
        validation_y,
        outer_fold=4,
        random_state=7,
    )

    np.testing.assert_allclose(fitted.scaler.mean_, train_x.mean(axis=0))
    assert tuple(score.c_value for score in fitted.candidate_scores) == LOGISTIC_CS
    assert fitted.selected_c == pytest.approx(1e-4)
    assert fitted.validation_auroc == pytest.approx(0.5)
    assert fitted.model.penalty == "l2"
    assert fitted.model.solver == "liblinear"
    assert fitted.model.class_weight is None
    probability = fitted.predict_test_once(np.ones((4, 2)))
    assert probability.shape == (4,)
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    with pytest.raises(RuntimeError, match="single-use"):
        fitted.predict_test_once(np.ones((4, 2)))


def _pcr_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.tile(np.asarray([0, 1, 0, 1, 0, 1]), 2)
    common = {
        "patient_id": [f"P{index:02d}" for index in range(12)],
        "fold": np.repeat([0, 1], 6),
        "y_true": labels,
    }
    reference = pd.DataFrame({**common, "predicted_probability": np.full(12, 0.5)})
    comparison = pd.DataFrame(
        {
            **common,
            "predicted_probability": np.where(labels == 1, 0.9, 0.1),
        }
    )
    return reference, comparison


def test_pcr_bootstrap_is_paired_patient_within_fold_and_deterministic() -> None:
    reference, comparison = _pcr_frames()
    first = paired_pcr_bootstrap(reference, comparison, n_bootstrap=2_000, seed=77)
    second = paired_pcr_bootstrap(
        reference,
        comparison.sample(frac=1.0, random_state=8),
        n_bootstrap=2_000,
        seed=77,
    )

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.draws, second.draws)
    assert first.n_patients == 12
    assert first.fold_sizes == {"0": 6, "1": 6}
    assert set(first.summary["bootstrap_unit"]) == {"patient_within_outer_fold"}
    assert set(first.summary["metric_aggregation"]) == {
        "pooled_patients_not_fold_replicates"
    }
    assert (first.summary["n_bootstrap"] == 2_000).all()
    assert (first.summary["improvement"] > 0.0).all()
    brier = first.summary.set_index("metric").loc["brier"]
    assert brier["improvement"] == pytest.approx(0.24)
    assert brier["orientation"] == "reference - comparison (lower is better)"


def test_generic_bootstrap_supports_lower_metrics_and_rejects_weak_protocol() -> None:
    ids = [f"P{index:02d}" for index in range(10)]
    truth = np.linspace(0.0, 9.0, 10)
    common = {"patient_id": ids, "fold": np.repeat([0, 1], 5), "y_true": truth}
    reference = pd.DataFrame({**common, "y_pred": truth + 1.0})
    comparison = pd.DataFrame({**common, "y_pred": truth + 0.1})
    result = paired_metric_bootstrap(
        reference,
        comparison,
        metric_functions={"mae": lambda y, p: float(np.mean(np.abs(p - y)))},
        metric_directions={"mae": "lower"},
        n_bootstrap=2_000,
        seed=2,
    )
    row = result.summary.iloc[0]
    assert row["improvement"] == pytest.approx(0.9)
    assert row["ci_lower"] == pytest.approx(0.9)
    assert row["ci_upper"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="at least 2000"):
        paired_metric_bootstrap(
            reference,
            comparison,
            metric_functions={"mae": lambda y, p: float(np.mean(np.abs(p - y)))},
            metric_directions={"mae": "lower"},
            n_bootstrap=1_999,
        )
    broken = comparison.copy()
    broken.loc[0, "patient_id"] = "UNMATCHED"
    with pytest.raises(ValueError, match="patient sets must match exactly"):
        paired_metric_bootstrap(
            reference,
            broken,
            metric_functions={"mae": lambda y, p: float(np.mean(np.abs(p - y)))},
            metric_directions={"mae": "lower"},
        )


def _passing_dynamics() -> dict[int, dict[str, object]]:
    return {
        seed: {
            "folds": 5,
            "actual_cosine": 0.50,
            "shuffled_cosine": 0.20,
            "cosine_gain": 0.30,
            "normalized_mse_relative_improvement": 0.20,
            "target_std": 0.15,
            "prediction_std": 0.12,
        }
        for seed in (2026, 3026)
    }


def _effects(
    static: tuple[float, float],
    delta: tuple[float, float],
    pcr: tuple[float, float],
) -> dict[int, dict[str, float]]:
    return {
        seed: {
            "static_ftv_spearman_delta": static[index],
            "delta_ftv_spearman_delta": delta[index],
            "mri_pcr_auroc_delta": pcr[index],
        }
        for index, seed in enumerate((2026, 3026))
    }


def _passing_complementarity() -> dict[str, object]:
    return {
        "T0-T1": {
            "seed_effects": {2026: 0.02, 3026: 0.01},
            "bootstrap_ci_lower": 0.001,
        },
        "T0-T2": {"seed_effects": {2026: -0.01, 3026: 0.02}},
    }


def _failing_complementarity() -> dict[str, object]:
    return {
        "T0-T1": {"seed_effects": {2026: -0.01, 3026: 0.01}},
        "T0-T2": {"seed_effects": {2026: 0.01, 3026: -0.01}},
    }


def test_gates_and_final_a_and_b_follow_preregistered_hierarchy() -> None:
    effects = _effects((0.04, 0.04), (0.02, 0.03), (0.01, 0.02))
    gates_a = {
        "A": evaluate_gate_a(_passing_dynamics()),
        "B": evaluate_gate_b(effects),
        "C": evaluate_gate_c(effects),
        "D": evaluate_gate_d(_passing_complementarity()),
    }
    assert {letter: decision.status for letter, decision in gates_a.items()} == {
        "A": "PASS",
        "B": "PASS",
        "C": "PASS",
        "D": "PASS",
    }
    assert gates_a["A"].success_label == "PATCH_DYNAMICS_VALID"
    assert gates_a["C"].success_label == "PATCH_STATE_ADDS_INFORMATION"
    assert gates_a["D"].success_label == "PATCH_STATE_COMPLEMENTARITY_SUPPORTED"
    assert classify_final(gates_a).label == FINAL_A

    pcr_gain = _effects((0.01, 0.01), (0.01, 0.01), (0.04, 0.04))
    gates_b = {
        "A": evaluate_gate_a(_passing_dynamics()),
        "B": evaluate_gate_b(pcr_gain),
        "C": evaluate_gate_c(pcr_gain),
        "D": evaluate_gate_d(_failing_complementarity()),
    }
    assert gates_b["D"].status == "FAIL"
    assert classify_final(gates_b).label == FINAL_B


def test_response_only_and_pooled_sufficient_labels_are_not_fabricated() -> None:
    strong_response = _effects((0.04, 0.04), (0.02, 0.03), (0.01, 0.02))
    strong_response_gates = {
        "A": evaluate_gate_a(_passing_dynamics()),
        "B": evaluate_gate_b(strong_response),
        "C": evaluate_gate_c(strong_response),
        "D": evaluate_gate_d(_failing_complementarity()),
    }
    assert strong_response_gates["C"].status == "PASS"
    assert classify_final(strong_response_gates).label == FINAL_C

    response_only = _effects((0.01, 0.01), (0.02, 0.02), (-0.01, -0.02))
    response_gates = {
        "A": evaluate_gate_a(_passing_dynamics()),
        "B": evaluate_gate_b(response_only),
        "C": evaluate_gate_c(response_only),
        "D": evaluate_gate_d(_failing_complementarity()),
    }
    assert response_gates["B"].status == "PASS"
    assert response_gates["C"].status == "FAIL"
    assert classify_final(response_gates).label == FINAL_C

    no_gain = _effects((-0.01, -0.01), (-0.01, -0.02), (-0.01, -0.02))
    no_gain_gates = {
        "A": evaluate_gate_a(_passing_dynamics()),
        "B": evaluate_gate_b(no_gain),
        "C": evaluate_gate_c(no_gain),
        "D": evaluate_gate_d(_failing_complementarity()),
    }
    assert classify_final(no_gain_gates).label == FINAL_D

    incomplete_a = evaluate_gate_a({2026: _passing_dynamics()[2026]})
    assert incomplete_a.status == INCOMPLETE
    final = classify_final({**response_gates, "A": incomplete_a})
    assert final.status == INCOMPLETE
    assert final.label == INCOMPLETE_FINAL


def test_gate_threshold_edges_and_incomplete_timing_are_explicit() -> None:
    # Gate B is strict at -0.03 for both static and DeltaFTV in every seed.
    edge = _effects((-0.03, 0.0), (0.01, 0.01), (0.0, 0.0))
    assert evaluate_gate_b(edge).status == "FAIL"
    tolerated = _effects((0.0, 0.0), (-0.029, -0.02), (0.0, 0.0))
    assert evaluate_gate_b(tolerated).status == "PASS"
    delta_edge = _effects((0.0, 0.0), (-0.03, 0.0), (0.0, 0.0))
    assert evaluate_gate_b(delta_edge).status == "FAIL"

    missing_timing = {"T0-T1": {"seed_effects": {2026: -0.01, 3026: 0.01}}}
    assert evaluate_gate_d(missing_timing).status == INCOMPLETE

    contradictory = _passing_dynamics()
    contradictory[2026] = {
        **contradictory[2026],
        "cosine_gain": 0.01,
        "normalized_mse_relative_improvement": 0.01,
    }
    assert evaluate_gate_a(contradictory).status == "FAIL"
