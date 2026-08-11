from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pooling import (  # noqa: E402
    local_statistics,
    pooling_variant,
    weighted_mean,
    weighted_population_std,
    weighted_quantiles,
)


def _spatial(values: list[float]) -> torch.Tensor:
    base = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1)
    return base.expand(1, 128, 1, 1, len(values)).clone()


def test_weighted_moments_match_manual_population_definition() -> None:
    spatial = _spatial([1.0, 2.0, 10.0])
    weights = torch.tensor([0.5, 1.0, 0.5]).reshape(1, 1, 1, 1, 3)
    mean = weighted_mean(spatial, weights)
    std = weighted_population_std(spatial, weights)
    expected_mean = (1.0 + 4.0 + 10.0) / 4.0
    expected_std = np.sqrt(
        (
            (1.0 - expected_mean) ** 2
            + 2 * (2.0 - expected_mean) ** 2
            + (10.0 - expected_mean) ** 2
        )
        / 4.0
    )
    assert torch.allclose(mean, torch.full_like(mean, expected_mean))
    assert torch.allclose(std, torch.full_like(std, expected_std))


def test_population_std_constant_map_has_exact_zero_and_finite_zero_gradient() -> None:
    spatial = torch.full(
        (2, 128, 2, 2, 2), 3.25, dtype=torch.float32, requires_grad=True
    )
    weights = torch.tensor(
        [0.0, 0.25, 0.5, 1.0, 0.75, 0.0, 1.0, 0.5], dtype=torch.float32
    ).reshape(1, 1, 2, 2, 2)
    std = weighted_population_std(spatial, weights)
    assert torch.equal(std, torch.zeros_like(std))
    std.sum().backward()
    assert spatial.grad is not None
    assert torch.isfinite(spatial.grad).all()
    assert torch.equal(spatial.grad, torch.zeros_like(spatial.grad))


def test_population_std_nonconstant_map_backward_is_finite() -> None:
    spatial = _spatial([1.0, 2.0, 4.0]).requires_grad_(True)
    weights = torch.tensor([0.25, 0.5, 1.0]).reshape(1, 1, 1, 1, 3)
    weighted_population_std(spatial, weights).sum().backward()
    assert spatial.grad is not None
    assert torch.isfinite(spatial.grad).all()
    # Translation invariance of SD makes each channel's spatial gradient sum 0.
    assert torch.allclose(
        spatial.grad.sum(dim=(-3, -2, -1)),
        torch.zeros(1, 128),
        rtol=0.0,
        atol=1e-6,
    )


def test_weighted_quantile_uses_left_continuous_inverse_cdf() -> None:
    spatial = _spatial([10.0, 1.0, 5.0, 100.0])
    weights = torch.tensor([0.5, 0.5, 1.0, 0.0]).reshape(1, 1, 1, 1, 4)
    result = weighted_quantiles(spatial, weights)
    assert torch.equal(result[0, 0], torch.tensor([1.0, 5.0, 5.0]))


def test_weighted_quantile_zero_support_filter_is_rowwise_equivalent() -> None:
    spatial = torch.stack(
        (_spatial([7.0, 1.0, 4.0, 9.0])[0], _spatial([3.0, 8.0, 2.0, 6.0])[0]),
        dim=0,
    )
    weights = torch.tensor(
        [[1.0, 0.0, 0.5, 0.0], [0.0, 0.25, 0.0, 1.0]], dtype=torch.float32
    ).reshape(2, 1, 1, 1, 4)
    result = weighted_quantiles(spatial, weights)
    assert torch.equal(result[0, 0], torch.tensor([4.0, 7.0, 7.0]))
    assert torch.equal(result[1, 0], torch.tensor([6.0, 6.0, 6.0]))


def test_registered_variant_dimensions() -> None:
    statistics = local_statistics(_spatial([1.0, 2.0, 3.0]), torch.ones(1, 1, 1, 1, 3))
    expected = {"P1": 128, "P2": 128, "P3": 256, "P4": 384, "P5": 640}
    for name, dimension in expected.items():
        assert pooling_variant(statistics, name).shape == (1, dimension)


def test_empty_support_and_unknown_variant_fail_closed() -> None:
    spatial = _spatial([1.0, 2.0])
    with pytest.raises(ValueError, match="positive spatial support"):
        weighted_mean(spatial, torch.zeros(1, 1, 1, 1, 2))
    with pytest.raises(ValueError, match="unknown pooling"):
        pooling_variant(local_statistics(spatial, torch.ones(1, 1, 1, 1, 2)), "P6")
