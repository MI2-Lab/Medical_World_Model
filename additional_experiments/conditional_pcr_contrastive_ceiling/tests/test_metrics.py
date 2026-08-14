from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from conditional_ceiling.metrics import (  # noqa: E402
    binary_metrics,
    calibration_slope,
    ece10,
    expected_calibration_error,
    paired_fold_stratified_bootstrap,
)


def test_registered_metrics_include_calibration_and_ece10() -> None:
    probabilities = np.repeat([0.1, 0.3, 0.7, 0.9], 10)
    labels = np.concatenate(
        (
            [1] + [0] * 9,
            [1] * 3 + [0] * 7,
            [1] * 7 + [0] * 3,
            [1] * 9 + [0],
        )
    )
    result = binary_metrics(labels, probabilities)

    assert set(result) == {
        "n",
        "n_positive",
        "n_negative",
        "auroc",
        "auprc",
        "brier",
        "calibration_slope",
        "ece10",
    }
    assert result["n"] == 40
    assert result["brier"] == pytest.approx(np.mean((probabilities - labels) ** 2))
    assert result["calibration_slope"] == pytest.approx(1.0, abs=1e-7)
    assert result["ece10"] == pytest.approx(0.0, abs=1e-12)
    assert ece10(labels, probabilities) == expected_calibration_error(
        labels, probabilities, n_bins=10
    )


def test_ece_bins_include_probability_one_and_slope_handles_degenerate_predictions() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.1, 0.9, 1.0])
    assert ece10(labels, probabilities) == pytest.approx(0.05)
    assert np.isnan(calibration_slope(labels, np.full(4, 0.5)))

    with pytest.raises(ValueError, match=r"\[0,1\]"):
        binary_metrics(labels, [-0.1, 0.2, 0.8, 0.9])
    with pytest.raises(ValueError, match="both binary classes"):
        binary_metrics(np.zeros(4, dtype=int), np.full(4, 0.2))


def _paired_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.tile([0, 1, 0, 1, 0, 1], 2)
    common = {
        "patient_id": [f"P{index:02d}" for index in range(12)],
        "fold": np.repeat([0, 1], 6),
        "y_true": labels,
    }
    reference = pd.DataFrame(
        {**common, "predicted_probability": np.where(labels == 1, 0.55, 0.45)}
    )
    comparison = pd.DataFrame(
        {**common, "predicted_probability": np.where(labels == 1, 0.9, 0.1)}
    )
    return reference, comparison


def test_paired_bootstrap_is_fold_stratified_deterministic_and_aggregate_only() -> None:
    reference, comparison = _paired_frames()
    first = paired_fold_stratified_bootstrap(
        reference,
        comparison,
        n_bootstrap=250,
        seed=77,
        metrics=("auroc", "auprc", "brier", "ece10"),
    )
    second = paired_fold_stratified_bootstrap(
        reference,
        comparison.sample(frac=1.0, random_state=4),
        n_bootstrap=250,
        seed=77,
        metrics=("auroc", "auprc", "brier", "ece10"),
    )
    pd.testing.assert_frame_equal(first, second)

    assert set(first["metric"]) == {"auroc", "auprc", "brier", "ece10"}
    assert set(first["bootstrap_unit"]) == {"patient_within_outer_fold"}
    assert set(first["n_bootstrap"]) == {250}
    assert set(first["n_folds"]) == {2}
    assert "patient_id" not in first.columns
    brier = first.set_index("metric").loc["brier"]
    assert brier["delta"] < 0.0
    assert brier["improvement"] > 0.0
    assert brier["point"] == pytest.approx(brier["comparison"] - brier["reference"])


@pytest.mark.parametrize("mutation", ["patient", "fold", "label", "duplicate"])
def test_paired_bootstrap_requires_exact_patient_pairing(mutation: str) -> None:
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
        error = "exactly one row"
    with pytest.raises(ValueError, match=error):
        paired_fold_stratified_bootstrap(
            reference, comparison, n_bootstrap=10, metrics=("auroc",)
        )
