from __future__ import annotations

import numpy as np
import pytest

from patch_token_wm.diagnostics import (
    cyclic_shuffled_targets,
    spatial_band_labels,
    spatial_error_summary,
    summarize_dynamics,
)


def test_cyclic_time_shuffle_is_locked() -> None:
    target = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    shuffled = cyclic_shuffled_targets(target)
    np.testing.assert_array_equal(shuffled[:, 0], target[:, 1])
    np.testing.assert_array_equal(shuffled[:, 1], target[:, 2])
    np.testing.assert_array_equal(shuffled[:, 2], target[:, 0])


def test_spatial_bands_use_fixed_physical_boundaries() -> None:
    coords = np.asarray([[0, 0, 0], [17, 0, 0], [25, 0, 0]], dtype=float)
    assert spatial_band_labels(coords).tolist() == [
        "central",
        "inner_local",
        "outer_local",
    ]


def test_dynamics_prefers_matched_targets() -> None:
    rng = np.random.default_rng(12)
    target = rng.normal(size=(5, 3, 4, 8))
    prediction = target + rng.normal(scale=0.01, size=target.shape)
    shuffled = rng.normal(size=target.shape)
    summary = summarize_dynamics(prediction, target, shuffled)
    assert summary.cosine_gain > 0.5
    assert summary.normalized_mse_relative_improvement > 0.9
    assert summary.target_std > 0
    assert summary.prediction_std > 0


def test_spatial_error_summary_is_aggregate() -> None:
    rng = np.random.default_rng(7)
    coords = np.asarray([[0, 0, 0], [17, 0, 0], [25, 0, 0]], dtype=float)
    target = rng.normal(size=(2, 3, 3, 4))
    prediction = target + 0.1
    indices = np.broadcast_to(np.arange(3), (2, 3, 3)).copy()
    output = spatial_error_summary(prediction, target, indices, coords)
    assert output["T1_central_token_predictions"] == 2
    assert output["T3_outer_local_token_predictions"] == 2
    assert all("patient" not in key for key in output)


def test_nonfinite_is_rejected() -> None:
    values = np.ones((2, 3, 2, 4))
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        summarize_dynamics(values, np.ones_like(values), np.ones_like(values))
