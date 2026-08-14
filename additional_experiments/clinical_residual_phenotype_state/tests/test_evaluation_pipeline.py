from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from crps import evaluation  # noqa: E402
from crps.evaluation import build_decision_summary  # noqa: E402
from crps.evaluation_contracts import (  # noqa: E402
    EvaluationContractError,
    file_sha256,
    load_factorized_asset,
)
from crps.evaluation_modeling import (  # noqa: E402
    ClinicalEncoder,
    binary_metrics,
    fit_binary_logistic,
    fit_multiclass_logistic,
    fit_ridge,
    paired_stratified_bootstrap,
)
from crps import reporting  # noqa: E402


def _factorized_fixture(tmp_path: Path, *, pcr_used: bool = False) -> tuple[dict, pd.DataFrame, Path]:
    n = 12
    rng = np.random.default_rng(7)
    root = tmp_path / "formal_primary"
    cell = root / "seed_2026" / "F1" / "fold_0"
    cell.mkdir(parents=True)
    z_r = rng.normal(size=(n, 4, 96)).astype(np.float32)
    z_p = rng.normal(size=(n, 4, 96)).astype(np.float32)
    path = cell / "factorized_state.private.npz"
    patients = np.asarray([f"P{i:02d}" for i in range(n)], dtype="U")
    split = np.asarray(["train"] * 6 + ["val"] * 3 + ["test"] * 3, dtype="U")
    np.savez_compressed(
        path,
        patient_id=patients,
        split=split,
        z_R=z_r,
        z_P=z_p,
        full=np.concatenate((z_r, z_p), axis=-1),
        z_P_aug=z_p.copy(),
        z_P_future_pred=z_p[:, 1:].copy(),
        z_P_future_target=z_p[:, 1:].copy(),
        z_P_future_context=z_p[:, :-1].copy(),
        arm=np.asarray("F1"),
        seed_base=np.asarray(2026, dtype=np.int64),
        fold=np.asarray(0, dtype=np.int64),
    )
    metadata = {
        "experiment": "clinical_residual_phenotype_state",
        "arm": "F1",
        "seed_base": 2026,
        "fold": 0,
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "pcr_labels_used": pcr_used,
        "export_completed": True,
        "feature_sha256": file_sha256(path),
        "representation_frozen_before_export": True,
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    folds = pd.DataFrame({"patient_id": patients, "fold": 0, "split": split, "label_pcr": [0, 1] * 6})
    config = {"frozen_inputs": {"factorized_feature_root": str(root), "expected_primary_patient_count": n}}
    return config, folds, path


def test_factorized_asset_rejects_noncanonical_feature_root(tmp_path: Path) -> None:
    bad_config, bad_folds, _ = _factorized_fixture(tmp_path / "bad", pcr_used=True)
    with pytest.raises(EvaluationContractError, match="not canonical"):
        load_factorized_asset(bad_config, bad_folds, "F1", 2026, 0)


def test_factorized_fixture_constructs_nonexact_concatenation_for_boundary_tests(tmp_path: Path) -> None:
    config, folds, path = _factorized_fixture(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["full"][0, 0, 0] += 1.0
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        assert not np.array_equal(
            archive["full"], np.concatenate((archive["z_R"], archive["z_P"]), axis=-1)
        )


def test_load_all_assets_requests_exact_twenty_factorized_and_ten_f0_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factorized_calls: list[tuple[str, int, int]] = []
    f0_calls: list[tuple[int, int]] = []

    def factorized(_config: dict, _folds: pd.DataFrame, arm: str, seed: int, fold: int) -> tuple:
        factorized_calls.append((arm, seed, fold))
        return arm, seed, fold

    def f0(_config: dict, _folds: pd.DataFrame, seed: int, fold: int) -> tuple:
        f0_calls.append((seed, fold))
        return seed, fold

    monkeypatch.setattr(evaluation, "load_factorized_asset", factorized)
    monkeypatch.setattr(evaluation, "load_f0_asset", f0)
    factorized_assets, f0_assets = evaluation.load_all_assets({}, pd.DataFrame())
    assert len(factorized_assets) == len(set(factorized_calls)) == 20
    assert len(f0_assets) == len(set(f0_calls)) == 10
    assert set(factorized_calls) == {
        (arm, seed, fold)
        for arm in ("F1", "F2")
        for seed in (2026, 3026)
        for fold in range(5)
    }


def test_train_only_linear_models_select_from_validation_and_predict_finite() -> None:
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(80, 5))
    x_val = rng.normal(size=(40, 5))
    x_test = rng.normal(size=(30, 5))
    y_train = (x_train[:, 0] + rng.normal(scale=0.4, size=80) > 0).astype(int)
    y_val = (x_val[:, 0] + rng.normal(scale=0.4, size=40) > 0).astype(int)
    fit = fit_binary_logistic(x_train, y_train, x_val, y_val, (0.01, 0.1, 1.0))
    probability = fit.predict(x_test)
    assert fit.selected_c in (0.01, 0.1, 1.0)
    assert probability.shape == (30,)
    assert np.all((probability >= 0) & (probability <= 1))

    target_train = 2.0 * x_train[:, 1] + rng.normal(size=80)
    target_val = 2.0 * x_val[:, 1] + rng.normal(size=40)
    ridge = fit_ridge(x_train, target_train, x_val, target_val, (0.01, 1.0, 100.0))
    assert ridge.selected_alpha in (0.01, 1.0, 100.0)
    assert np.isfinite(ridge.predict(x_test)).all()


def test_clinical_encoder_fits_train_vocabulary_and_maps_unknown_to_zero() -> None:
    train = pd.DataFrame(
        {
            "label_hr": [0, 1], "label_her2": [0, 1], "label_mp": [1, 0],
            "age_at_screening": [40.0, np.nan], "arm": ["A", "B"],
        }
    )
    test = pd.DataFrame(
        {
            "label_hr": [1], "label_her2": [0], "label_mp": [1],
            "age_at_screening": [np.nan], "arm": ["UNSEEN"],
        }
    )
    encoder = ClinicalEncoder().fit(train)
    transformed = encoder.transform(test)
    assert transformed.shape == (1, 6)
    np.testing.assert_array_equal(transformed[0, -2:], [0.0, 0.0])
    assert transformed[0, 3] == pytest.approx(40.0)


def test_paired_bootstrap_is_deterministic_fold_outcome_stratified_and_minimum_2000() -> None:
    rng = np.random.default_rng(19)
    folds = np.repeat(np.arange(5), 20)
    labels = np.tile(np.repeat([0, 1], 10), 5)
    baseline = np.clip(0.35 + 0.3 * labels + rng.normal(scale=0.12, size=100), 0, 1)
    augmented = np.clip(baseline + 0.08 * (2 * labels - 1), 0, 1)
    ids = np.asarray([f"P{i:03d}" for i in range(100)])
    first, first_draws = paired_stratified_bootstrap(
        ids, folds, labels, baseline, augmented, n_bootstrap=2000, random_state=77
    )
    second, second_draws = paired_stratified_bootstrap(
        ids, folds, labels, baseline, augmented, n_bootstrap=2000, random_state=77
    )
    assert first == second
    pd.testing.assert_frame_equal(first_draws, second_draws)
    assert first["stratification"] == "outer_fold_x_outcome"
    assert first["delta_auroc"] > 0
    with pytest.raises(ValueError, match="at least 2000"):
        paired_stratified_bootstrap(ids, folds, labels, baseline, augmented, n_bootstrap=1999)


def _decision_inputs() -> tuple[pd.DataFrame, ...]:
    diagnostics = pd.DataFrame(
        [
            {
                "arm": arm, "seed_base": seed, "fold": 0,
                "z_P_effective_rank": 20.0 if arm != "F0" else np.nan,
                "z_P_mean_std": 0.2 if arm != "F0" else np.nan,
                "standardized_crosscov_rms": 0.1 if arm == "F1" else 0.09 if arm == "F2" else 0.2,
                "optional_diagnostic_tensors_present": arm != "F0",
                "augmentation_mean_cosine": 0.8 if arm != "F0" else np.nan,
                "future_phenotype_mse": 0.2 if arm != "F0" else np.nan,
                "future_persistence_mse": 0.3 if arm != "F0" else np.nan,
                "future_mse_improvement_over_persistence": 0.1 if arm != "F0" else np.nan,
            }
            for arm in ("F0", "F1", "F2") for seed in (2026, 3026)
        ]
    )
    nearest = pd.DataFrame(
        [
            {"arm": "F1", "fold": -1, "mean_jaccard": 0.3},
            {"arm": "F2", "fold": -1, "mean_jaccard": 0.25},
        ]
    )
    response_rows = []
    for arm, state, shift in (("F0", "F0", 0.0), ("F1", "z_R", 0.01), ("F2", "z_R", 0.005)):
        for seed in (2026, 3026):
            response_rows.extend(
                [
                    {"arm": arm, "state": state, "seed_base": seed, "task": "static", "endpoint": "macro", "spearman": 0.5 + shift},
                    {"arm": arm, "state": state, "seed_base": seed, "task": "delta", "endpoint": "macro", "spearman": 0.3 + shift},
                ]
            )
    response = pd.DataFrame(response_rows)
    profile = pd.DataFrame(
        [
            {
                "arm": arm, "state": "z_P", "target": target,
                "endpoint": "T0_T1_T2_macro", "seed_base": seed,
                "auroc": (0.70 if target == "label_hr" else 0.65) - (0.08 if arm == "F2" else 0.0),
                "flip_invariant_decodability": 0.5
                + abs(
                    (0.70 if target == "label_hr" else 0.65)
                    - (0.08 if arm == "F2" else 0.0)
                    - 0.5
                ),
            }
            for arm in ("F1", "F2") for target in ("label_hr", "label_her2") for seed in (2026, 3026)
        ]
    )
    effects = []
    for arm in ("F1", "F2"):
        for seed in (2026, 3026):
            effects.extend(
                [
                    {"arm": arm, "seed_base": seed, "comparison": "MRI_full_vs_zR", "timing": "T0-T2", "delta_auroc": 0.02},
                    {"arm": arm, "seed_base": seed, "comparison": "beyond_C_full_vs_zR", "timing": "T0-T2", "delta_auroc": 0.02},
                ]
            )
    for timing in ("T0-T1", "T0-T2"):
        for seed in (2026, 3026):
            effects.append(
                {"arm": "F2", "seed_base": seed, "comparison": "beyond_C_F_full_vs_zR", "timing": timing, "delta_auroc": 0.04}
            )
    return diagnostics, nearest, response, profile, pd.DataFrame(), pd.DataFrame(effects)


def _evaluation_config() -> dict:
    return {
        "diagnostics": {"effective_rank_floor": 10.0, "phenotype_mean_std_floor": 0.05, "augmentation_cosine_floor": 0.5},
        "gates": {
            "response_static_ftv_spearman_degradation_floor": -0.03,
            "phenotype_complementarity_timings": ["T0-T1", "T0-T2"],
            "phenotype_complementarity_strong_mean": 0.03,
        },
        "bootstrap": {"replicates": 2000, "confidence_level": 0.95},
    }


def test_decision_gates_require_preservation_noncollapse_redundancy_and_complementarity() -> None:
    decision = build_decision_summary(*_decision_inputs(), _evaluation_config())
    assert decision["classification"]["code"] == "A"
    assert all(value["pass"] for value in decision["gates"].values())


def test_gate_d_strong_form_requires_both_seeds_positive() -> None:
    diagnostics, nearest, response, profile, pcr, effects = _decision_inputs()
    mask = (
        effects.arm.eq("F2")
        & effects.comparison.eq("beyond_C_F_full_vs_zR")
        & effects.timing.eq("T0-T1")
    )
    effects.loc[mask & effects.seed_base.eq(2026), "delta_auroc"] = -0.01
    effects.loc[mask & effects.seed_base.eq(3026), "delta_auroc"] = 0.08
    decision = build_decision_summary(
        diagnostics, nearest, response, profile, pcr, effects, _evaluation_config()
    )
    gate = decision["gates"]["D_PHENOTYPE_COMPLEMENTARITY"]
    assert "T0-T1" not in gate["positive_both_seed_timings"]
    assert "T0-T1" not in gate["strong_mean_ge_0_03_timings"]


def test_gate_a_fails_closed_on_nonfinite_delta_response_effect() -> None:
    diagnostics, nearest, response, profile, pcr, effects = _decision_inputs()
    response.loc[
        response.arm.eq("F2")
        & response.seed_base.eq(2026)
        & response.task.eq("delta"),
        "spearman",
    ] = np.nan
    decision = build_decision_summary(
        diagnostics, nearest, response, profile, pcr, effects, _evaluation_config()
    )
    gate = decision["gates"]["A_RESPONSE_PRESERVED"]
    assert gate["pass"] is False
    assert gate["all_response_effects_finite"] is False


def test_gate_c_uses_flip_invariant_decodability_not_raw_auroc_decrease() -> None:
    diagnostics, nearest, response, profile, pcr, effects = _decision_inputs()
    profile.loc[profile.arm.eq("F1"), "auroc"] = 0.45
    profile.loc[profile.arm.eq("F2"), "auroc"] = 0.40
    profile.loc[profile.arm.eq("F1"), "flip_invariant_decodability"] = 0.55
    profile.loc[profile.arm.eq("F2"), "flip_invariant_decodability"] = 0.60
    decision = build_decision_summary(
        diagnostics, nearest, response, profile, pcr, effects, _evaluation_config()
    )
    gate = decision["gates"]["C_CLINICAL_REDUNDANCY_REDUCED"]
    assert gate["pass"] is False
    assert gate["profile_auroc"]["label_hr"]["2026"]["F2_minus_F1_decodability"] > 0


def test_gate_c_uses_mean_endpoint_decodability_without_raw_auroc_cancellation() -> None:
    diagnostics, nearest, response, profile, pcr, effects = _decision_inputs()
    # A raw macro AUROC of 0.5 can hide endpoints at 0.4 and 0.6.  The
    # evaluator supplies their mean symmetric association score instead.
    profile["auroc"] = 0.5
    profile.loc[profile.arm.eq("F1"), "flip_invariant_decodability"] = 0.60
    profile.loc[profile.arm.eq("F2"), "flip_invariant_decodability"] = 0.55
    decision = build_decision_summary(
        diagnostics, nearest, response, profile, pcr, effects, _evaluation_config()
    )
    gate = decision["gates"]["C_CLINICAL_REDUNDANCY_REDUCED"]
    assert gate["pass"] is True
    assert gate["profile_auroc"]["label_hr"]["2026"]["F2_minus_F1_decodability"] == pytest.approx(-0.05)


def test_binary_metrics_rejects_nonprobabilities() -> None:
    with pytest.raises(ValueError, match="probability"):
        binary_metrics([0, 1], [-0.1, 1.1])


def test_logistic_solver_contracts_are_explicit() -> None:
    rng = np.random.default_rng(41)
    x_train = rng.normal(size=(80, 4))
    x_val = rng.normal(size=(40, 4))
    y_train = np.tile([0, 1], 40)
    y_val = np.tile([0, 1], 20)
    with pytest.raises(ValueError, match="liblinear"):
        fit_binary_logistic(
            x_train, y_train, x_val, y_val, (0.1,), solver="lbfgs"
        )


def test_subtype_probe_requires_exact_four_class_coverage_per_fit_split() -> None:
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(60, 4))
    x_val = rng.normal(size=(30, 4))
    required = ("A", "B", "C", "D")
    y_train = np.tile(["A", "B", "C"], 20)
    y_val = np.tile(["A", "B", "C"], 10)
    with pytest.raises(ValueError, match="exact required classes"):
        fit_multiclass_logistic(
            x_train,
            y_train,
            x_val,
            y_val,
            (0.1,),
            expected_classes=required,
        )


def test_cached_clinical_baseline_realigns_different_asset_row_orders() -> None:
    first_ids = np.asarray(["P1", "P2", "P3"])
    first_labels = np.asarray([0, 1, 0])
    first_probability = np.asarray([0.1, 0.8, 0.3])
    entry = evaluation._make_clinical_cache_entry(
        first_ids, first_labels, first_probability, 0.1, 0.7
    )
    second_ids = np.asarray(["P3", "P1", "P2"])
    second_labels = np.asarray([0, 0, 1])
    aligned, selected_c, validation_auroc = evaluation._read_clinical_cache_entry(
        entry, second_ids, second_labels
    )
    np.testing.assert_array_equal(aligned, [0.3, 0.1, 0.8])
    assert selected_c == pytest.approx(0.1)
    assert validation_auroc == pytest.approx(0.7)


def test_future_safeguard_uses_only_same_space_ema_target_persistence() -> None:
    target = np.zeros((2, 3, 96), dtype=np.float32)
    target[:, 1] = 1.0
    target[:, 2] = 3.0
    prediction = target.copy()
    # T0->T1 prediction is outside the valid comparison because EMA-target T0
    # context was not exported; make it arbitrarily wrong to prove exclusion.
    prediction[:, 0] = 10_000.0
    future_mse, persistence_mse = evaluation._future_prediction_diagnostic(
        prediction, target
    )
    assert future_mse == pytest.approx(0.0)
    assert persistence_mse == pytest.approx(2.5)


def test_chinese_report_answers_all_twelve_questions_from_aggregates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics, nearest, response, profile, _, effects_minimal = _decision_inputs()
    decision = build_decision_summary(
        diagnostics, nearest, response, profile, pd.DataFrame(), effects_minimal, _evaluation_config()
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    diagnostics = diagnostics.assign(
        z_R_mean_std=0.2, z_R_effective_rank=30.0, z_R_cov_trace=1.0,
        z_P_cov_trace=1.0, z_R_eigen_top1_fraction=0.2, z_P_eigen_top1_fraction=0.2,
        z_R_collapsed_dimensions=0, z_P_collapsed_dimensions=0,
        standardized_crosscov_frobenius=1.0, cca_mean_top10=0.2, cca_max=0.3,
        future_mse_improvement_over_persistence=0.1,
        n_patient_visits=100, pcr_labels_used=False,
    )
    diagnostics.to_csv(metrics / "state_diagnostics.csv", index=False)
    nearest.assign(state="z_P", timing="T0-T2").to_csv(metrics / "nearest_neighbor_stability.csv", index=False)
    response.assign(n=100, rmse=1.0, mae=0.8, r2=0.1).to_csv(metrics / "response_metrics.csv", index=False)
    profile.assign(n=100, auprc=0.6, brier=0.2, n_positive=50, auroc_macro_ovr=np.nan, accuracy=np.nan).to_csv(
        metrics / "phenotype_probes.csv", index=False
    )
    pd.DataFrame(
        [{"seed_base": 2026, "arm": "F1", "population": "full_808", "timing": "T0", "model": "C", "auroc": 0.7, "auprc": 0.5, "brier": 0.2}]
    ).to_csv(metrics / "pcr_metrics.csv", index=False)
    effect_rows = []
    for comparison in (
        "MRI_full_vs_zR", "beyond_C_full_vs_zR", "beyond_C_F_full_vs_zR",
        "beyond_C_F_zP_vs_C_F", "adversarial_F2_vs_F1_full",
    ):
        for timing in ("T0-T1", "T0-T2"):
            for arm in ("F1", "F2"):
                for seed in (2026, 3026):
                    effect_rows.append(
                        {
                            "arm": arm, "seed_base": seed,
                            "population": "ftv_complete_375" if "F_" in comparison or "adversarial" in comparison else "full_808",
                            "timing": timing, "comparison": comparison,
                            "delta_auroc": 0.04, "delta_auroc_ci_lower": 0.01,
                            "delta_auroc_ci_upper": 0.07, "n_bootstrap": 2000,
                        }
                    )
    pd.DataFrame(effect_rows).to_csv(metrics / "paired_bootstrap_effects.csv", index=False)
    (metrics / "decision_summary.json").write_text(json.dumps(decision), encoding="utf-8")
    report_config = _evaluation_config() | {
        "gates": _evaluation_config()["gates"] | {"clinical_redundancy_primary_view": "static_T0_T1_T2_macro"}
    }
    monkeypatch.setattr(reporting, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(reporting, "load_evaluation_config", lambda **_kwargs: report_config)
    monkeypatch.setattr(reporting, "require_before_outcome_access", lambda: {})
    output = reporting.generate_report()
    text = output.read_text(encoding="utf-8")
    for number in range(1, 13):
        assert f"{number}. **" in text
    assert "最终分类" in text
    assert "patient-level OOF" in text
