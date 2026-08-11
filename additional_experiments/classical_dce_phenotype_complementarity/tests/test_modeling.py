"""Focused leakage and reproducibility tests for modeling.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "modeling.py"
SPEC = importlib.util.spec_from_file_location("classical_dce_modeling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
modeling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = modeling
SPEC.loader.exec_module(modeling)


def test_secondary_transform_uses_train_only_clipping_imputation_and_indicators() -> None:
    train = pd.DataFrame(
        {
            "FTV": [0.0, 2.0, 4.0],
            "LD": [np.nan, 10.0, 20.0],
        }
    )
    transformer = modeling.NumericRadiomicsTransformer(
        winsor_quantiles=(0.25, 0.75),
        missing_strategy="median_indicator",
    ).fit(train)

    np.testing.assert_allclose(transformer.clip_lower_, [1.0, 12.5])
    np.testing.assert_allclose(transformer.clip_upper_, [3.0, 17.5])
    # Medians are learned after the train-only clip, never from held-out rows.
    np.testing.assert_allclose(transformer.medians_, [2.0, 15.0])

    learned_state = tuple(
        value.copy()
        for value in (
            transformer.clip_lower_,
            transformer.clip_upper_,
            transformer.medians_,
            transformer.mean_,
            transformer.scale_,
        )
    )
    held_out = pd.DataFrame({"FTV": [1_000_000.0], "LD": [np.nan]})
    transformed = transformer.transform(held_out)

    # Extreme held-out FTV is clipped at the training upper bound. Missing LD
    # is imputed to the training median and identified in its indicator column.
    reference = transformer.transform(pd.DataFrame({"FTV": [3.0], "LD": [15.0]}))
    np.testing.assert_allclose(transformed[:, :2], reference[:, :2])
    np.testing.assert_array_equal(transformed[:, 2:], [[0.0, 1.0]])
    assert transformer.get_feature_names_out().tolist() == [
        "FTV",
        "LD",
        "FTV__missing",
        "LD__missing",
    ]

    # Calling transform on test data cannot alter any fitted statistic.
    for before, after in zip(
        learned_state,
        (
            transformer.clip_lower_,
            transformer.clip_upper_,
            transformer.medians_,
            transformer.mean_,
            transformer.scale_,
        ),
    ):
        np.testing.assert_array_equal(before, after)


def test_primary_strict_mode_enforces_complete_case_rows() -> None:
    incomplete = np.asarray([[1.0, np.nan], [2.0, 3.0]])
    np.testing.assert_array_equal(modeling.complete_case_mask(incomplete), [False, True])
    with pytest.raises(ValueError, match="complete-case training"):
        modeling.NumericRadiomicsTransformer(missing_strategy="strict").fit(incomplete)

    transformer = modeling.NumericRadiomicsTransformer(
        winsor_quantiles=None, missing_strategy="strict"
    ).fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="complete-case transform"):
        transformer.transform([[5.0, np.nan]])


def test_multioutput_ftv_residualization_is_train_fitted_and_directionally_correct() -> None:
    x = np.linspace(-3.0, 3.0, 40)
    ftv_train = np.column_stack((x, x**2))
    nonftv_train = np.column_stack(
        (
            1.0 + 2.0 * ftv_train[:, 0] - 0.3 * ftv_train[:, 1],
            -2.0 + 0.5 * ftv_train[:, 0] + 1.2 * ftv_train[:, 1],
        )
    )
    residualizer = modeling.FTVResidualizer(
        alpha=0.0,
        preprocessor=modeling.NumericRadiomicsTransformer(
            winsor_quantiles=None,
            missing_strategy="strict",
            with_scaling=True,
        ),
    ).fit(ftv_train, nonftv_train)

    training_residuals = residualizer.transform(ftv_train, nonftv_train)
    assert training_residuals.shape == nonftv_train.shape
    assert np.max(np.abs(training_residuals)) < 1e-10

    coefficient_before = residualizer.estimator_.coef_.copy()
    held_out_ftv = np.asarray([[5.0, 25.0], [-4.0, 16.0]])
    held_out_nonftv = np.column_stack(
        (
            1.0 + 2.0 * held_out_ftv[:, 0] - 0.3 * held_out_ftv[:, 1],
            -2.0 + 0.5 * held_out_ftv[:, 0] + 1.2 * held_out_ftv[:, 1],
        )
    )
    held_out_residuals = residualizer.transform(held_out_ftv, held_out_nonftv)
    assert np.max(np.abs(held_out_residuals)) < 1e-10
    np.testing.assert_array_equal(coefficient_before, residualizer.estimator_.coef_)


def test_paired_fold_stratified_bootstrap_is_deterministic() -> None:
    y = np.tile([0, 1], 12)
    folds = np.repeat(np.arange(4), 6)
    baseline = np.where(y == 1, 0.62, 0.38)
    augmented = np.where(y == 1, 0.82, 0.18)
    kwargs = dict(
        patient_ids=np.arange(len(y)),
        n_bootstrap=2000,
        random_state=20260811,
        return_distributions=True,
    )

    first = modeling.paired_fold_stratified_bootstrap(
        y, baseline, augmented, folds, **kwargs
    )
    second = modeling.paired_fold_stratified_bootstrap(
        y, baseline, augmented, folds, **kwargs
    )

    assert first["stratification"] == "outer_fold+outcome"
    assert first["n_bootstrap"] == 2000
    assert first["brier_improvement"]["estimate"] > 0.0
    for metric in ("delta_auroc", "delta_auprc", "brier_improvement"):
        assert first[metric] == second[metric]
        np.testing.assert_array_equal(
            first["distributions"][metric], second["distributions"][metric]
        )


def test_multiclass_metrics_for_perfect_probabilities() -> None:
    y = np.asarray(["HR+/HER2-", "HR-/HER2+", "TN"] * 2)
    labels = ["HR+/HER2-", "HR-/HER2+", "TN"]
    lookup = {label: index for index, label in enumerate(labels)}
    probability = np.zeros((len(y), len(labels)))
    for row, label in enumerate(y):
        probability[row, lookup[label]] = 1.0

    metrics = modeling.multiclass_metrics(y, probability, labels=labels)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.0)
