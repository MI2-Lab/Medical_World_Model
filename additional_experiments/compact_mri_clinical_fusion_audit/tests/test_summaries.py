from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from summaries import (  # noqa: E402
    ComparisonSpec,
    SUBTYPE_CLASSES,
    SUBTYPE_PROBABILITY_COLUMNS,
    SummaryContractError,
    aggregate_pcr_predictions,
    aggregate_profile_predictions,
    paired_bootstrap_effects,
    paired_point_effects,
    summarize_paired_comparisons,
)


POPULATION_SIZES = {"full_808": 8, "ftv_complete_375": 4}
FOLDS = (0, 1)


def _pcr_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, count in POPULATION_SIZES.items():
        for index in range(count):
            patient_id = f"P{index:02d}"
            fold = (index // 2) % 2
            label = index % 2
            for model_key, family, representation in (
                ("C", "C", "clinical"),
                ("C+Mk:selected", "C+M", "pca_selected"),
            ):
                probability = 0.5 if model_key == "C" else (0.9 if label else 0.1)
                rows.append(
                    {
                        "patient_id": patient_id,
                        "fold": fold,
                        "population": population,
                        "seed": 2026,
                        "arm": "LOCAL0",
                        "timing": "T0",
                        "model_family": family,
                        "representation": representation,
                        # Selected compact dimensions may legitimately differ by fold.
                        "dimension": 0 if model_key == "C" else (8 if fold == 0 else 16),
                        "model_key": model_key,
                        "y_true": label,
                        "predicted_probability": probability,
                        "predicted_label": int(probability >= 0.5),
                        "threshold": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _comparison() -> ComparisonSpec:
    return ComparisonSpec(
        name="compact_vs_clinical",
        reference={"model_key": "C"},
        comparison={"model_key": "C+Mk:selected"},
    )


def test_pcr_aggregation_is_strict_oof_and_retains_populations() -> None:
    predictions = _pcr_predictions().sample(frac=1.0, random_state=7)
    metrics = aggregate_pcr_predictions(
        predictions,
        expected_population_sizes=POPULATION_SIZES,
        expected_folds=FOLDS,
    )

    assert len(metrics) == 4
    assert set(metrics["population"]) == set(POPULATION_SIZES)
    selected = metrics.loc[
        metrics["population"].eq("full_808")
        & metrics["model_key"].eq("C+Mk:selected")
    ].iloc[0]
    assert selected["n"] == 8
    assert selected["n_folds"] == 2
    assert selected["auroc"] == pytest.approx(1.0)
    assert selected["auprc"] == pytest.approx(1.0)
    assert selected["brier"] == pytest.approx(0.01)
    assert pd.isna(selected["dimension"])
    assert set(json.loads(selected["dimension_values"])) == {8, 16}
    assert json.loads(selected["fold_sizes"]) == {"0": 4, "1": 4}


def test_pcr_aggregation_rejects_duplicates_and_cross_model_label_drift() -> None:
    predictions = _pcr_predictions()
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(SummaryContractError, match="repeats patients"):
        aggregate_pcr_predictions(
            duplicated,
            expected_population_sizes=POPULATION_SIZES,
            expected_folds=FOLDS,
        )

    drifted = predictions.copy()
    row = drifted.index[
        drifted["population"].eq("full_808")
        & drifted["model_key"].eq("C+Mk:selected")
        & drifted["patient_id"].eq("P00")
    ][0]
    drifted.loc[row, "y_true"] = 1
    drifted.loc[row, "predicted_label"] = int(
        drifted.loc[row, "predicted_probability"] >= drifted.loc[row, "threshold"]
    )
    with pytest.raises(SummaryContractError, match="patient/fold/label coverage differs"):
        aggregate_pcr_predictions(
            drifted,
            expected_population_sizes=POPULATION_SIZES,
            expected_folds=FOLDS,
        )


def test_paired_effects_and_goal2_bootstrap_have_both_brier_orientations() -> None:
    predictions = _pcr_predictions()
    point = paired_point_effects(
        predictions,
        [_comparison()],
        expected_population_sizes=POPULATION_SIZES,
        expected_folds=FOLDS,
    )
    assert len(point) == 2
    assert set(point["population"]) == set(POPULATION_SIZES)
    assert np.allclose(point["delta_auroc"], 0.5)
    assert np.allclose(point["delta_brier"], -0.24)
    assert np.allclose(point["brier_improvement"], 0.24)

    summary, draws = paired_bootstrap_effects(
        predictions.sample(frac=1.0, random_state=9),
        [_comparison()],
        n_bootstrap=40,
        random_seed=77,
        expected_population_sizes=POPULATION_SIZES,
        expected_folds=FOLDS,
    )
    assert len(summary) == 2 * 3
    assert len(draws) == 2 * 40
    assert set(summary["bootstrap_unit"]) == {"patient_within_outer_fold"}
    brier = summary.loc[summary["metric"].eq("brier")]
    assert np.allclose(brier["delta_brier"], -brier["improvement"])
    assert (brier["delta_brier"] < 0.0).all()
    assert np.allclose(draws["delta_brier"], -draws["brier_improvement"])

    combined = summarize_paired_comparisons(
        predictions,
        [_comparison()],
        n_bootstrap=20,
        random_seed=77,
        expected_population_sizes=POPULATION_SIZES,
        expected_folds=FOLDS,
    )
    pd.testing.assert_frame_equal(combined.point_effects, point)
    assert len(combined.bootstrap_draws) == 40


def test_paired_comparison_cannot_cross_population_estimands() -> None:
    cross_population = ComparisonSpec(
        name="invalid_cross_population",
        reference={"population": "full_808", "model_key": "C"},
        comparison={"population": "ftv_complete_375", "model_key": "C+Mk:selected"},
    )
    with pytest.raises(SummaryContractError, match="cells differ"):
        paired_point_effects(
            _pcr_predictions(),
            [cross_population],
            expected_population_sizes=POPULATION_SIZES,
            expected_folds=FOLDS,
        )


def _profile_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    class_cycle = list(SUBTYPE_CLASSES) * 4
    for representation in ("raw", "pca16", "pca32"):
        for target in ("HR", "HER2", "subtype_4class"):
            for index, subtype in enumerate(class_cycle):
                common: dict[str, object] = {
                    "patient_id": f"P{index:02d}",
                    "fold": index % 2,
                    "seed": 2026,
                    "arm": "LOCAL0",
                    "timing": "T0",
                    "representation": representation,
                    "target": target,
                }
                subtype_probability = {
                    name: (0.8 if name == subtype else 0.2 / 3.0)
                    for name in SUBTYPE_CLASSES
                }
                for name, column in SUBTYPE_PROBABILITY_COLUMNS.items():
                    common[column] = subtype_probability[name]
                if target == "HR":
                    label = int(subtype.startswith("HR+"))
                    probability = 0.8 if label else 0.2
                    common.update(
                        {
                            "y_true": label,
                            "predicted_probability": probability,
                            "predicted_label": int(probability >= 0.5),
                            "threshold": 0.5,
                        }
                    )
                elif target == "HER2":
                    label = int(subtype.endswith("HER2+"))
                    probability = 0.8 if label else 0.2
                    common.update(
                        {
                            "y_true": label,
                            "predicted_probability": probability,
                            "predicted_label": int(probability >= 0.5),
                            "threshold": 0.5,
                        }
                    )
                else:
                    common.update(
                        {
                            "y_true": subtype,
                            "predicted_probability": np.nan,
                            "predicted_label": subtype,
                            "threshold": np.nan,
                        }
                    )
                rows.append(common)
    return pd.DataFrame(rows)


def test_profile_aggregation_supports_binary_and_fixed_four_class_outputs() -> None:
    metrics = aggregate_profile_predictions(
        _profile_predictions().sample(frac=1.0, random_state=11),
        expected_patient_count=16,
        expected_folds=FOLDS,
    )
    assert len(metrics) == 9
    assert set(metrics["representation"]) == {"raw", "pca16", "pca32"}
    assert set(metrics["target"]) == {"HR", "HER2", "subtype_4class"}
    assert metrics["auroc"].eq(1.0).all()
    assert metrics["auprc"].eq(1.0).all()
    subtype = metrics.loc[metrics["target"].eq("subtype_4class")]
    assert subtype["n_classes"].eq(4).all()
    assert subtype["brier"].isna().all()
    assert all(len(json.loads(value)) == 4 for value in subtype["class_counts"])


def test_profile_aggregation_rejects_missing_representation_and_bad_subtype_rows() -> None:
    predictions = _profile_predictions()
    missing = predictions.loc[~predictions["representation"].eq("pca32")]
    with pytest.raises(SummaryContractError, match="lacks exact representations"):
        aggregate_profile_predictions(
            missing,
            expected_patient_count=16,
            expected_folds=FOLDS,
        )

    broken = predictions.copy()
    row = broken.index[broken["target"].eq("subtype_4class")][0]
    broken.loc[row, SUBTYPE_PROBABILITY_COLUMNS[SUBTYPE_CLASSES[0]]] += 0.1
    with pytest.raises(SummaryContractError, match="sum to one"):
        aggregate_profile_predictions(
            broken,
            expected_patient_count=16,
            expected_folds=FOLDS,
        )
