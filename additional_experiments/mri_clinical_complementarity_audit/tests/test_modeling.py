from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT / "scripts"))

from modeling import (  # noqa: E402
    binary_metrics,
    clinical_probability_error,
    fit_binary_logistic,
    fit_clinical_error_ridge,
    fit_ftv_mri_residualizer,
    fit_multiclass_logistic,
    multiclass_metrics,
    paired_fold_stratified_bootstrap,
    select_validation_balanced_threshold,
)


def test_binary_logistic_uses_train_scaler_and_validation_selection() -> None:
    train_x = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    train_y = np.asarray([0, 0, 0, 1, 1, 1])
    validation_x = np.asarray([[10.0], [11.0], [12.0], [13.0]])
    validation_y = np.asarray([0, 0, 1, 1])

    fitted = fit_binary_logistic(
        train_x,
        train_y,
        validation_x,
        validation_y,
        c_grid=[10.0, 0.1, 1.0],
    )

    np.testing.assert_allclose(fitted.scaler.mean_, train_x.mean(axis=0))
    assert not np.allclose(
        fitted.scaler.mean_, np.concatenate([train_x, validation_x]).mean(axis=0)
    )
    assert fitted.selected_c == pytest.approx(0.1)
    assert fitted.validation_auroc == pytest.approx(1.0)
    assert fitted.model.penalty == "l2"
    assert fitted.threshold_selection.balanced_accuracy == pytest.approx(1.0)
    assert fitted.threshold_selection.threshold > 0.5
    probabilities = fitted.predict_proba(np.asarray([[-4.0], [14.0]]))
    assert probabilities.shape == (2,)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_binary_threshold_and_metrics_have_known_values() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])

    selection = select_validation_balanced_threshold(labels, probabilities)
    metrics = binary_metrics(labels, probabilities, threshold=selection.threshold)

    assert selection.threshold == pytest.approx(0.5)
    assert selection.balanced_accuracy == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.085)


@pytest.mark.parametrize(
    ("train_x", "train_y", "validation_x", "validation_y", "error_match"),
    [
        (
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 0, 0],
            [[4.0], [5.0]],
            [0, 1],
            "both binary classes",
        ),
        (
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 1, 1],
            [[4.0], [np.nan]],
            [0, 1],
            "NaN or infinity",
        ),
        (
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 1, 1],
            [[4.0, 1.0], [5.0, 1.0]],
            [0, 1],
            "features; expected",
        ),
    ],
)
def test_binary_logistic_fails_fast_on_invalid_fold_data(
    train_x: list[list[float]],
    train_y: list[int],
    validation_x: list[list[float]],
    validation_y: list[int],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        fit_binary_logistic(
            train_x,
            train_y,
            validation_x,
            validation_y,
            c_grid=[1.0],
        )


def test_multiclass_logistic_is_balanced_train_scaled_and_smaller_c_tied() -> None:
    train_x = np.asarray(
        [
            [-4.0, -4.0],
            [-3.0, -3.0],
            [-2.0, -3.0],
            [4.0, -4.0],
            [3.0, -3.0],
            [2.0, -3.0],
            [0.0, 4.0],
            [-1.0, 3.0],
            [1.0, 3.0],
        ]
    )
    train_y = np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    validation_x = np.asarray(
        [
            [-4.5, -3.5],
            [-2.5, -2.5],
            [4.5, -3.5],
            [2.5, -2.5],
            [0.0, 4.5],
            [0.5, 2.5],
        ]
    )
    validation_y = np.asarray(["a", "a", "b", "b", "c", "c"])

    fitted = fit_multiclass_logistic(
        train_x,
        train_y,
        validation_x,
        validation_y,
        c_grid=[100.0, 0.01, 1.0],
    )

    np.testing.assert_allclose(fitted.scaler.mean_, train_x.mean(axis=0))
    assert fitted.model.class_weight == "balanced"
    assert fitted.model.penalty == "l2"
    assert fitted.selected_c == pytest.approx(0.01)
    assert fitted.validation_macro_ovr_auroc == pytest.approx(1.0)
    assert fitted.validation_macro_ovr_auprc == pytest.approx(1.0)
    metrics = multiclass_metrics(
        validation_y,
        fitted.predict_proba(validation_x),
        classes=fitted.classes,
    )
    assert metrics["macro_ovr_auroc"] == pytest.approx(1.0)
    assert metrics["macro_ovr_auprc"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_multiclass_checks_class_coverage_and_probability_rows() -> None:
    train_x = np.arange(18, dtype=float).reshape(9, 2)
    train_y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    validation_x = np.arange(8, dtype=float).reshape(4, 2)

    with pytest.raises(ValueError, match="exactly the train classes"):
        fit_multiclass_logistic(
            train_x,
            train_y,
            validation_x,
            np.asarray([0, 0, 1, 1]),
            c_grid=[1.0],
        )
    with pytest.raises(ValueError, match="rows must sum to one"):
        multiclass_metrics(
            np.asarray([0, 1, 2]),
            np.asarray([[0.8, 0.1, 0.0], [0.2, 0.8, 0.2], [0.1, 0.1, 0.8]]),
            classes=[0, 1, 2],
        )


def test_ftv_to_mri_residualizer_is_train_fit_and_recovers_linear_map() -> None:
    train_ftv = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, -1.0]]
    )
    weights = np.asarray([[2.0, -1.0, 0.5], [3.0, 4.0, -2.0]])
    intercept = np.asarray([1.0, -2.0, 5.0])
    train_mri = train_ftv @ weights + intercept
    validation_ftv = np.asarray([[10.0, 4.0], [-3.0, 2.0]])
    validation_mri = validation_ftv @ weights + intercept

    fitted = fit_ftv_mri_residualizer(train_ftv, train_mri)

    np.testing.assert_allclose(fitted.scaler.mean_, train_ftv.mean(axis=0))
    np.testing.assert_allclose(fitted.transform(train_ftv, train_mri), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        fitted.transform(validation_ftv, validation_mri), 0.0, atol=1e-12
    )
    with pytest.raises(ValueError, match="rows; expected"):
        fitted.transform(validation_ftv, validation_mri[:1])


def test_clinical_error_ridge_uses_validation_mse_and_smaller_alpha_tie() -> None:
    labels = np.asarray([0, 1, 1, 0])
    probabilities = np.asarray([0.2, 0.7, 0.9, 0.4])
    np.testing.assert_allclose(
        clinical_probability_error(labels, probabilities),
        np.asarray([-0.2, 0.3, 0.1, -0.4]),
    )

    train_x = np.asarray([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    validation_x = np.asarray([[20.0, 10.0], [21.0, 11.0]])
    fitted = fit_clinical_error_ridge(
        train_x,
        np.full(4, 0.25),
        validation_x,
        np.full(2, 0.25),
        alphas=[10.0, 0.01, 1.0],
    )

    np.testing.assert_allclose(fitted.scaler.mean_, train_x.mean(axis=0))
    assert fitted.selected_alpha == pytest.approx(0.01)
    assert fitted.validation_mse == pytest.approx(0.0, abs=1e-24)
    np.testing.assert_allclose(fitted.predict(validation_x), 0.25)
    with pytest.raises(ValueError, match="NaN or infinity"):
        fit_clinical_error_ridge(
            train_x,
            np.asarray([0.1, 0.2, np.nan, 0.4]),
            validation_x,
            np.asarray([0.1, 0.2]),
            alphas=[1.0],
        )


def _paired_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def test_paired_fold_bootstrap_is_deterministic_and_orients_improvement() -> None:
    reference, comparison = _paired_frames()

    first = paired_fold_stratified_bootstrap(
        reference, comparison, n_bootstrap=250, seed=77
    )
    second = paired_fold_stratified_bootstrap(
        reference,
        comparison.sample(frac=1.0, random_state=4),
        n_bootstrap=250,
        seed=77,
    )

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.draws, second.draws)
    assert first.n_patients == 12
    assert first.fold_sizes == {0: 6, 1: 6}
    assert set(first.summary["bootstrap_unit"]) == {"patient_within_outer_fold"}
    assert (first.summary["improvement"] > 0.0).all()
    brier = first.summary.set_index("metric").loc["brier"]
    assert brier["improvement"] == pytest.approx(0.24)
    assert brier["orientation"] == "reference - comparison (lower Brier is better)"
    assert (first.draws["brier_improvement"] > 0.0).all()


@pytest.mark.parametrize("mutation", ["patient", "fold", "label", "duplicate"])
def test_paired_fold_bootstrap_requires_exact_patient_pairing(mutation: str) -> None:
    reference, comparison = _paired_frames()
    if mutation == "patient":
        comparison.loc[0, "patient_id"] = "unmatched"
        error = "patient sets must match exactly"
    elif mutation == "fold":
        comparison.loc[0, "fold"] = 1
        error = "fold assignments must match exactly"
    elif mutation == "label":
        comparison.loc[0, "y_true"] = 1
        error = "labels disagree"
    else:
        comparison.loc[0, "patient_id"] = comparison.loc[1, "patient_id"]
        error = "exactly one row per patient"

    with pytest.raises(ValueError, match=error):
        paired_fold_stratified_bootstrap(reference, comparison, n_bootstrap=10, seed=1)
