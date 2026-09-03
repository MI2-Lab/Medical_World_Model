from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_sph.evaluation import regression_metrics  # noqa: E402
from residual_sph.probes import (  # noqa: E402
    select_state_reconstruction_ridge,
    state_reconstruction_metrics,
)


def _load_aggregate_module():
    path = EXPERIMENT_ROOT / "scripts" / "aggregate_representation.py"
    spec = importlib.util.spec_from_file_location("residual_sph_aggregate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGGREGATE = _load_aggregate_module()


def _load_report_module():
    path = EXPERIMENT_ROOT / "scripts" / "generate_report.py"
    spec = importlib.util.spec_from_file_location("residual_sph_report_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = _load_report_module()


def test_multivariate_state_r2_has_explicit_variance_weighted_definition() -> None:
    truth = np.asarray(
        [
            [-2.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0],
            [1.0, -1.0, 5.0],
            [2.0, 1.0, 5.0],
        ],
        dtype=np.float64,
    )
    prediction = truth.copy()
    prediction[:, 0] += np.asarray([0.2, -0.1, 0.1, -0.2])
    prediction[:, 1] *= 0.5
    metrics = state_reconstruction_metrics(truth, prediction)

    centered = truth - truth.mean(axis=0, keepdims=True)
    expected = 1.0 - np.sum((truth - prediction) ** 2) / np.sum(centered**2)
    assert metrics["state_variance_weighted_r2"] == pytest.approx(expected)
    assert metrics["state_dimension"] == 3
    assert metrics["nonconstant_test_state_dimensions"] == 2
    assert metrics["state_standardized_rmse"] == pytest.approx(
        np.sqrt(np.mean((truth - prediction) ** 2))
    )


def test_target_to_state_selector_fits_both_scalers_on_train_only() -> None:
    train_target = np.linspace(-2.0, 2.0, 40)
    val_target = np.linspace(20.0, 30.0, 12)
    coefficients = np.asarray([0.5, -1.2, 2.0, 0.25])
    train_state = train_target[:, None] * coefficients[None, :] + np.asarray(
        [2.0, -3.0, 0.5, 4.0]
    )
    val_state = val_target[:, None] * coefficients[None, :] + np.asarray(
        [2.0, -3.0, 0.5, 4.0]
    )
    selected = select_state_reconstruction_ridge(
        train_target, train_state, val_target, val_state
    )

    assert selected.target_scaler.mean_[0] == pytest.approx(np.mean(train_target))
    np.testing.assert_allclose(selected.state_scaler.mean_, np.mean(train_state, axis=0))
    assert selected.target_scaler.mean_[0] != pytest.approx(np.mean(val_target))
    predicted = selected.model.predict(
        selected.target_scaler.transform(val_target[:, None])
    )
    truth = selected.state_scaler.transform(val_state)
    assert state_reconstruction_metrics(truth, predicted)[
        "state_variance_weighted_r2"
    ] > 0.99


def _prediction_frame(task: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = np.asarray([-1.2, -0.2, 0.4, 1.3], dtype=np.float64)
    error = np.asarray([0.15, -0.1, 0.2, -0.15], dtype=np.float64)
    for fold in range(5):
        # Analysis coordinates deliberately have fold-specific offsets.  A
        # pooled metric would be spuriously excellent from between-fold scale.
        analysis_true = base + 100.0 * fold
        analysis_pred = 0.55 * base + 100.0 * fold + error
        if task == "sph_res" and fold == 4:
            analysis_pred = analysis_pred[::-1].copy()
        natural_true = 0.20 + 0.015 * (base + fold)
        natural_pred = natural_true + 0.003 * error
        for index in range(len(base)):
            rows.append(
                {
                    "arm": "S2",
                    "seed_base": 2026,
                    "fold": fold,
                    "task": task,
                    "endpoint": "T0",
                    "y_true_analysis": analysis_true[index],
                    "y_pred_analysis": analysis_pred[index],
                    "y_true_natural": natural_true[index],
                    "y_pred_natural": natural_pred[index],
                }
            )
    return pd.DataFrame(rows)


def test_residual_ranks_are_pooled_scale_metrics_foldwise_and_reconstruction_separate() -> None:
    frame = _prediction_frame("sph_res")
    residual_row, reconstructed_row = AGGREGATE._residual_metric_rows(frame)
    fold_metrics = [
        regression_metrics(
            fold_frame["y_true_analysis"], fold_frame["y_pred_analysis"]
        )
        for _, fold_frame in frame.groupby("fold", sort=True)
    ]
    expected_r2 = np.mean([value["natural_r2"] for value in fold_metrics])
    expected_fold_spearman = np.mean([value["spearman"] for value in fold_metrics])
    expected_rmse = np.sqrt(np.mean([value["rmse"] ** 2 for value in fold_metrics]))
    pooled_residual = regression_metrics(
        frame["y_true_analysis"], frame["y_pred_analysis"]
    )

    assert residual_row["space"] == "residual"
    assert residual_row["rank_aggregation"] == (
        "pooled_5fold_oof_analysis_coordinate_within_seed"
    )
    assert residual_row["scale_metric_aggregation"] == (
        "outer_test_n_weighted_fold_metric_with_rmse_from_weighted_fold_mse"
    )
    assert residual_row["residual_space_r2"] == pytest.approx(expected_r2)
    assert residual_row["residual_space_spearman"] == pytest.approx(
        pooled_residual["spearman"]
    )
    assert residual_row["spearman"] == pytest.approx(
        pooled_residual["spearman"]
    )
    assert residual_row["residual_space_pearson"] == pytest.approx(
        pooled_residual["pearson"]
    )
    assert residual_row["fold_weighted_residual_space_spearman"] == pytest.approx(
        expected_fold_spearman
    )
    assert residual_row["residual_space_rmse"] == pytest.approx(expected_rmse)
    assert residual_row["residual_space_r2"] != pytest.approx(
        pooled_residual["natural_r2"]
    )
    assert residual_row["residual_space_spearman"] != pytest.approx(
        expected_fold_spearman
    )
    assert "natural_r2" not in residual_row

    expected_reconstruction = regression_metrics(
        frame["y_true_natural"], frame["y_pred_natural"]
    )
    assert reconstructed_row["space"] == "reconstructed_natural"
    assert reconstructed_row["reconstructed_natural_r2"] == pytest.approx(
        expected_reconstruction["natural_r2"]
    )
    assert reconstructed_row["reconstructed_natural_rmse"] == pytest.approx(
        expected_reconstruction["rmse"]
    )
    assert "residual_space_r2" not in reconstructed_row


def test_invalid_cross_fold_analysis_pooling_is_never_published() -> None:
    static_rows = AGGREGATE._aggregate_probe_group(
        _prediction_frame("static_ftv")
    )
    assert len(static_rows) == 1
    assert static_rows[0]["space"] == "natural"

    raw_rows = AGGREGATE._aggregate_probe_group(_prediction_frame("raw_sph"))
    assert [row["space"] for row in raw_rows] == ["natural", "transformed"]
    assert raw_rows[1]["aggregation"] == (
        "outer_test_n_weighted_fold_transformed_metrics"
    )
    assert "natural_r2" not in raw_rows[1]


def _diagnostic_fold_rows(seed: int, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for fold, n_test in enumerate((10, 20, 30, 40, 50)):
        rows.append(
            {
                "arm": "S2",
                "seed_base": seed,
                "fold": fold,
                "task": "sph_res_to_state",
                "endpoint": "T0",
                "target_coordinate": "fold_train_fitted_residual_SPH_z",
                "state_coordinate": "train_standardized_response_state",
                "selected_alpha": 1.0,
                "n_test": n_test,
                "state_variance_weighted_r2": 0.01 * fold + offset,
                "state_uniform_average_r2": -0.02 + 0.01 * fold + offset,
                "state_standardized_rmse": 1.0 - 0.05 * fold,
                "state_standardized_mae": 0.8 - 0.04 * fold,
                "nonconstant_test_state_dimensions": 192,
            }
        )
    return pd.DataFrame(rows)


def test_state_redundancy_is_fold_weighted_and_has_seed_consistency() -> None:
    first = AGGREGATE._aggregate_state_diagnostics(_diagnostic_fold_rows(2026))
    weights = np.asarray([10, 20, 30, 40, 50], dtype=np.float64)
    assert first.iloc[0]["state_variance_weighted_r2"] == pytest.approx(
        np.average(np.arange(5) * 0.01, weights=weights)
    )
    assert first.iloc[0]["state_standardized_rmse"] == pytest.approx(
        np.sqrt(np.average((1.0 - 0.05 * np.arange(5)) ** 2, weights=weights))
    )
    assert not bool(first.iloc[0]["cross_fold_state_vectors_pooled"])

    diagnostics = AGGREGATE._aggregate_state_diagnostics(
        pd.concat(
            [_diagnostic_fold_rows(2026), _diagnostic_fold_rows(3026, 0.02)],
            ignore_index=True,
        )
    )
    metric_rows: list[dict[str, object]] = []
    for seed, change in ((2026, 0.0), (3026, 0.02)):
        metric_rows.extend(
            [
                {
                    "arm": "S2",
                    "seed_base": seed,
                    "task": "raw_sph",
                    "endpoint": "T0",
                    "space": "natural",
                    "spearman": 0.2 + change,
                    "natural_r2": 0.1 + change,
                },
                {
                    "arm": "S2",
                    "seed_base": seed,
                    "task": "sph_res",
                    "endpoint": "T0",
                    "space": "residual",
                    "residual_space_spearman": 0.3 + change,
                    "residual_space_r2": 0.05 + change,
                },
                {
                    "arm": "S2",
                    "seed_base": seed,
                    "task": "sph_res",
                    "endpoint": "T0",
                    "space": "reconstructed_natural",
                    "reconstructed_natural_r2": 0.15 + change,
                },
            ]
        )
    consistency = AGGREGATE._seed_consistency_rows(
        pd.DataFrame(metric_rows), diagnostics
    )
    assert set(consistency["metric"]) == {
        "natural_spearman",
        "natural_r2",
        "residual_space_spearman",
        "residual_space_r2",
        "reconstructed_natural_r2",
        "state_variance_weighted_r2",
    }
    assert consistency["seed_count"].eq(2).all()
    np.testing.assert_allclose(
        consistency["absolute_seed_difference"].to_numpy(dtype=np.float64),
        0.02,
        rtol=0.0,
        atol=1e-14,
    )


def test_report_consumes_only_explicit_residual_and_reconstruction_fields() -> None:
    rows: list[dict[str, object]] = []
    aggregations = {
        ("raw_sph", "natural"): "pooled_5fold_oof_within_seed",
        ("raw_sph", "transformed"): "outer_test_n_weighted_fold_transformed_metrics",
        ("sph_res", "residual"): (
            "pooled_5fold_oof_residual_rank_and_fold_weighted_scale_metrics"
        ),
        ("sph_res", "reconstructed_natural"): (
            "pooled_5fold_oof_conditional_target_reconstruction"
        ),
    }
    for arm_index, arm in enumerate(REPORT.ARMS):
        for seed in REPORT.SEEDS:
            for visit_index, endpoint in enumerate(REPORT.VISITS):
                base = 0.1 + 0.01 * arm_index + 0.001 * visit_index
                for (task, space), aggregation in aggregations.items():
                    row: dict[str, object] = {
                        "arm": arm,
                        "seed_base": seed,
                        "task": task,
                        "endpoint": endpoint,
                        "space": space,
                        "aggregation": aggregation,
                        "n": 375,
                    }
                    if (task, space) == ("raw_sph", "natural"):
                        row.update(
                            {
                                "raw_natural_spearman": base,
                                "raw_natural_pearson": base - 0.01,
                                "raw_natural_r2": base - 0.02,
                                "raw_natural_rmse": 0.2,
                                "raw_natural_mae": 0.1,
                            }
                        )
                    elif (task, space) == ("sph_res", "residual"):
                        row.update(
                            {
                                "rank_aggregation": (
                                    "pooled_5fold_oof_analysis_coordinate_within_seed"
                                ),
                                "scale_metric_aggregation": (
                                    "outer_test_n_weighted_fold_metric_with_rmse_from_weighted_fold_mse"
                                ),
                                "residual_space_spearman": base + 0.1,
                                "residual_space_pearson": base + 0.09,
                                "residual_space_r2": base + 0.08,
                                "residual_space_rmse": 0.8,
                                "residual_space_mae": 0.6,
                                "fold_weighted_residual_space_spearman": base + 0.07,
                                "fold_weighted_residual_space_pearson": base + 0.06,
                            }
                        )
                    elif (task, space) == ("sph_res", "reconstructed_natural"):
                        row.update(
                            {
                                "reconstructed_natural_spearman": base + 0.2,
                                "reconstructed_natural_pearson": base + 0.19,
                                "reconstructed_natural_r2": base + 0.18,
                                "reconstructed_natural_rmse": 0.02,
                                "reconstructed_natural_mae": 0.01,
                                "reconstructed_natural_variance_ratio": 0.9,
                            }
                        )
                    rows.append(row)
    combined = REPORT._sph_combined_table(pd.DataFrame(rows))
    assert len(combined) == 32
    assert {
        "residual_space_r2",
        "fold_weighted_residual_space_spearman",
        "reconstructed_sph_natural_r2",
        "raw_sph_natural_r2",
    }.issubset(combined.columns)
    selected = combined.loc[
        combined["arm"].astype(str).eq("S2")
        & combined["seed_base"].eq(2026)
        & combined["endpoint"].astype(str).eq("T0")
    ].iloc[0]
    assert selected["residual_space_r2"] != selected["reconstructed_sph_natural_r2"]

    with pytest.raises(ValueError, match="exact arm/seed/visit/space coverage"):
        REPORT._sph_combined_table(pd.DataFrame(rows).iloc[:-1])


def _formal_bootstrap_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 0
    for comparison in REPORT.PCR_COMPARISONS:
        for timing in REPORT.PCR_TIMINGS:
            for seed in REPORT.SEEDS:
                effect = 0.01 + counter * 1e-5
                rows.append(
                    {
                        "comparison": comparison,
                        "timing": timing,
                        "seed_base": seed,
                        "orientation": "comparison_minus_reference",
                        "auroc_orientation": "comparison_minus_reference",
                        "auprc_orientation": "comparison_minus_reference",
                        "brier_orientation": "reference_minus_comparison_lower_is_better",
                        "aggregation": "pooled_oof_paired_patient_bootstrap",
                        "stratification": "patient_within_outer_fold",
                        "bootstrap_unit": "patient_within_outer_fold",
                        "ci_method": "percentile",
                        "n": 375,
                        "fold_count": 5,
                        "n_bootstrap": 2000,
                        "confidence_level": 0.95,
                        "reference_auroc": 0.60,
                        "comparison_auroc": 0.60 + effect,
                        "delta_auroc": effect,
                        "delta_auroc_ci_lower": effect - 0.02,
                        "delta_auroc_ci_upper": effect + 0.02,
                        "delta_auroc_bootstrap_probability_positive": 0.75,
                        "reference_auprc": 0.40,
                        "comparison_auprc": 0.40 + effect,
                        "delta_auprc": effect,
                        "delta_auprc_ci_lower": effect - 0.03,
                        "delta_auprc_ci_upper": effect + 0.03,
                        "delta_auprc_bootstrap_probability_positive": 0.70,
                        "reference_brier": 0.20,
                        "comparison_brier": 0.20 - effect,
                        "brier_improvement": effect,
                        "brier_improvement_ci_lower": effect - 0.01,
                        "brier_improvement_ci_upper": effect + 0.01,
                        "brier_improvement_bootstrap_probability_positive": 0.80,
                    }
                )
                counter += 1
    return pd.DataFrame(rows)


def test_report_audits_and_renders_three_metric_paired_bootstrap() -> None:
    audited = REPORT._paired_bootstrap_audit(_formal_bootstrap_frame())
    assert len(audited) == 40
    assert set(audited["comparison"].astype(str)) == set(REPORT.PCR_COMPARISONS)

    q9 = REPORT._bootstrap_effect_text(
        audited,
        comparison="S2_CM_minus_C",
        point="delta_auroc",
        lower="delta_auroc_ci_lower",
        upper="delta_auroc_ci_upper",
    )
    assert all(timing in q9 for timing in REPORT.PCR_TIMINGS)
    assert "seed 2026" in q9 and "seed 3026" in q9

    source = (
        EXPERIMENT_ROOT / "scripts" / "generate_report.py"
    ).read_text(encoding="utf-8")
    for column in (
        "delta_auroc_bootstrap_probability_positive",
        "delta_auprc_ci_lower",
        "delta_auprc_ci_upper",
        "delta_auprc_bootstrap_probability_positive",
        "brier_improvement_ci_lower",
        "brier_improvement_ci_upper",
        "brier_improvement_bootstrap_probability_positive",
    ):
        assert column in source

    invalid = _formal_bootstrap_frame().iloc[:-1]
    with pytest.raises(ValueError, match="incomplete formal coverage"):
        REPORT._paired_bootstrap_audit(invalid)


def test_report_question_9_consumes_paired_effect_summary() -> None:
    effects = {
        "S2_C_plus_M_minus_C": {
            "T0": {"2026": 0.03, "3026": 0.01},
        },
        "paired_metric_effect_summaries": {
            "S2_C_plus_M_minus_C": {
                "T0": {
                    "by_seed": {
                        "2026": {
                            "delta_auroc": 0.03,
                            "delta_auprc": 0.02,
                            "brier_improvement": 0.01,
                        },
                        "3026": {
                            "delta_auroc": 0.01,
                            "delta_auprc": -0.01,
                            "brier_improvement": 0.00,
                        },
                    },
                    "two_seed_mean": {
                        "delta_auroc": 0.02,
                        "delta_auprc": 0.005,
                        "brier_improvement": 0.005,
                    },
                    "both_seeds_positive": {
                        "delta_auroc": True,
                        "delta_auprc": False,
                        "brier_improvement": False,
                    },
                }
            }
        },
    }
    assert REPORT._paired_metric_values(
        effects, "S2_C_plus_M_minus_C", "T0", "delta_auroc"
    ) == {2026: 0.03, 3026: 0.01}
    assert REPORT._paired_metric_values(
        effects, "S2_C_plus_M_minus_C", "T0", "delta_auprc"
    ) == {2026: 0.02, 3026: -0.01}

    effects["S2_C_plus_M_minus_C"]["T0"]["3026"] = 0.02
    with pytest.raises(ValueError, match="legacy and paired AUROC effects disagree"):
        REPORT._paired_metric_values(
            effects, "S2_C_plus_M_minus_C", "T0", "delta_auroc"
        )


def test_public_trajectory_rows_are_complete_and_identifier_free() -> None:
    epochs = []
    for epoch in range(1, 4):
        epochs.append(
            {
                "epoch": epoch,
                "finite": True,
                "checkpoint_eligible": epoch >= 2,
                "base_gate_pass": True,
                "noncollapse": True,
                "train_loss": 1.0 / epoch,
                "train_base_loss": 0.7 / epoch,
                "train_state_loss": 0.6 / epoch,
                "train_ftv_loss": 0.5 / epoch,
                "train_sph_loss": 0.4 / epoch,
                "train_representation_std": 0.2,
                "val_loss": 1.1 / epoch,
                "val_base_objective": 0.8 / epoch,
                "val_state_loss": 0.7 / epoch,
                "val_ftv_loss": 0.6 / epoch,
                "val_sph_loss": 0.5 / epoch,
                "val_representation_std": 0.2,
            }
        )
    selection = {
        "arm": "S2",
        "seed_base": 2026,
        "fold": 0,
        "effective_seed": 2026,
        "selected_epoch": 2,
        "selection_mode": "primary",
        "experiment_pass": True,
        "test_data_used": False,
        "pcr_used": False,
        "clinical_used": False,
        "treatment_used": False,
        "epochs": epochs,
    }
    rows = AGGREGATE._trajectory_rows_from_selection(
        selection, arm="S2", seed=2026, fold=0
    )
    assert len(rows) == 3
    assert sum(bool(row["is_selected_epoch"]) for row in rows) == 1
    assert not any("patient" in key for row in rows for key in row)


def test_report_links_all_aggregate_artifacts_and_fails_closed_on_figures(
    tmp_path: Path,
) -> None:
    expected_artifacts = {
        "residualizer_fits.csv",
        "residualizer_inventory.json",
        "representation_metrics.csv",
        "table_static_ftv.csv",
        "table_observed_delta_ftv.csv",
        "table_sph_and_residual.csv",
        "table_state_redundancy.csv",
        "table_seed_consistency.csv",
        "table_partial_correlations.csv",
        "optimization_safety.csv",
        "optimization_trajectories.csv",
        "representation_effects.json",
        "table_pcr_complementarity.csv",
        "paired_bootstrap.csv",
        "pcr_effects.json",
        "decision.json",
        "execution_status.json",
    }
    assert {path.name for _, path in REPORT.AGGREGATE_ARTIFACTS} == expected_artifacts
    assert {path.name for _, path in REPORT.AGGREGATE_FIGURES} == {
        "representation_effects.svg",
        "sph_res_organization.svg",
        "pcr_effects.svg",
    }

    report_directory = tmp_path / "reports"
    metrics_directory = tmp_path / "metrics"
    figures_directory = tmp_path / "figures"
    report_directory.mkdir()
    metrics_directory.mkdir()
    figures_directory.mkdir()
    aggregate = metrics_directory / "representation_effects.json"
    figure = figures_directory / "representation_effects.svg"
    aggregate.write_text("{}\n", encoding="utf-8")
    figure.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")

    artifact_links = REPORT._artifact_markdown_links(
        (("Representation effects", aggregate),),
        report_directory=report_directory,
        label="aggregate CSV/JSON artifact",
    )
    figure_links = REPORT._artifact_markdown_links(
        (("Primary representation effects", figure),),
        report_directory=report_directory,
        label="aggregate figure",
        require_svg=True,
    )
    assert artifact_links == "- [Representation effects](../metrics/representation_effects.json)"
    assert figure_links == "- [Primary representation effects](../figures/representation_effects.svg)"
    assert str(tmp_path) not in artifact_links + figure_links

    missing = figures_directory / "pcr_effects.svg"
    with pytest.raises(FileNotFoundError, match="missing aggregate figure"):
        REPORT._artifact_markdown_links(
            (("Post-freeze pCR effects", missing),),
            report_directory=report_directory,
            label="aggregate figure",
            require_svg=True,
        )
