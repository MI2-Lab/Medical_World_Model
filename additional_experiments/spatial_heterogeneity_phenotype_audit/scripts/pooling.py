"""Deterministic raw-channel statistics for frozen LOCAL spatial maps."""

from __future__ import annotations

from typing import Iterable

import torch


CHANNELS = 128
QUANTILES = (0.25, 0.5, 0.75)


class _PopulationStdSqrt(torch.autograd.Function):
    """Exact nonnegative square root with the registered zero subgradient.

    The forward is exactly ``sqrt(clamp_min(variance, 0))``.  Its mathematical
    derivative is singular at zero, so Stage B explicitly uses subgradient zero
    there (and for negative roundoff clamped to zero) to keep encoder gradients
    finite without adding an epsilon or changing the population-SD value.
    """

    @staticmethod
    def forward(ctx: object, variance: torch.Tensor) -> torch.Tensor:
        result = torch.sqrt(variance.clamp_min(0.0))
        ctx.save_for_backward(result)  # type: ignore[attr-defined]
        return result

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (result,) = ctx.saved_tensors  # type: ignore[attr-defined]
        positive = result > 0
        safe_result = torch.where(positive, result, torch.ones_like(result))
        derivative = torch.where(
            positive,
            gradient / (2.0 * safe_result),
            torch.zeros_like(gradient),
        )
        return (derivative,)


def _validated(
    spatial: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(spatial, torch.Tensor) or spatial.ndim != 5:
        raise ValueError("spatial must be a tensor [N,C,D,H,W]")
    if spatial.shape[0] <= 0 or spatial.shape[1] != CHANNELS:
        raise ValueError(f"spatial must have nonempty N and C={CHANNELS}")
    if not spatial.dtype.is_floating_point or not bool(torch.isfinite(spatial).all()):
        raise ValueError("spatial must contain finite floating values")
    if (
        not isinstance(weights, torch.Tensor)
        or weights.ndim != 5
        or weights.shape[1] != 1
    ):
        raise ValueError("weights must be [N|1,1,D,H,W]")
    if weights.shape[0] not in {1, spatial.shape[0]}:
        raise ValueError("weights batch must be one or match spatial")
    if tuple(weights.shape[-3:]) != tuple(spatial.shape[-3:]):
        raise ValueError("weights and spatial grids differ")
    if weights.device != spatial.device:
        raise ValueError("weights and spatial must share a device")
    if not weights.dtype.is_floating_point or not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must contain finite floating values")
    if bool((weights < 0).any()) or bool((weights > 1).any()):
        raise ValueError("weights must lie in [0,1]")
    expanded = weights.expand(spatial.shape[0], -1, -1, -1, -1).to(spatial.dtype)
    denominator = expanded.sum(dim=(-3, -2, -1))
    if not bool((denominator > 0).all()):
        raise ValueError("every row must have positive spatial support")
    return expanded, denominator


def weighted_mean(spatial: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted first moment with the fixed physical-overlap denominator."""

    expanded, denominator = _validated(spatial, weights)
    result = (spatial * expanded).sum(dim=(-3, -2, -1)) / denominator
    if tuple(result.shape) != tuple(spatial.shape[:2]) or not bool(
        torch.isfinite(result).all()
    ):
        raise AssertionError("weighted mean produced an invalid result")
    return result


def weighted_population_std(
    spatial: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Weighted population SD (ddof=0), with stable nonnegative roundoff."""

    expanded, denominator = _validated(spatial, weights)
    mean = (spatial * expanded).sum(dim=(-3, -2, -1)) / denominator
    centered = spatial - mean[..., None, None, None]
    variance = (expanded * centered.square()).sum(dim=(-3, -2, -1)) / denominator
    result = _PopulationStdSqrt.apply(variance)
    if tuple(result.shape) != tuple(spatial.shape[:2]) or not bool(
        torch.isfinite(result).all()
    ):
        raise AssertionError("weighted SD produced an invalid result")
    return result


def weighted_quantiles(
    spatial: torch.Tensor,
    weights: torch.Tensor,
    quantiles: Iterable[float] = QUANTILES,
) -> torch.Tensor:
    """Left-continuous weighted empirical inverse CDF for each channel.

    Returns `[N,C,Q]` in the exact supplied quantile order. Zero-weight cells do
    not affect the cumulative distribution.
    """

    expanded, _ = _validated(spatial, weights)
    requested = tuple(float(value) for value in quantiles)
    if not requested or any(not 0.0 < value < 1.0 for value in requested):
        raise ValueError("quantiles must be a nonempty sequence strictly inside (0,1)")
    flat = spatial.flatten(start_dim=2)
    row_weights = expanded[:, 0].flatten(start_dim=1)
    # Zero-weight cells have no mass in the registered inverse CDF. Removing
    # their union before sorting is exactly equivalent and reduces the formal
    # shared LOCAL sort from 6,160 to 500 values per channel.
    positive_union = torch.any(row_weights > 0, dim=0)
    flat = flat[..., positive_union]
    weight_flat = row_weights[:, None, positive_union].expand(-1, spatial.shape[1], -1)
    sorted_values, order = torch.sort(flat, dim=-1, stable=True)
    sorted_weights = torch.gather(weight_flat, dim=-1, index=order)
    cumulative = torch.cumsum(sorted_weights, dim=-1)
    total = cumulative[..., -1]
    outputs: list[torch.Tensor] = []
    for quantile in requested:
        threshold = total * quantile
        index = torch.searchsorted(
            cumulative.contiguous(), threshold.unsqueeze(-1).contiguous(), right=False
        ).squeeze(-1)
        index = index.clamp(max=flat.shape[-1] - 1)
        outputs.append(torch.gather(sorted_values, -1, index.unsqueeze(-1)).squeeze(-1))
    result = torch.stack(outputs, dim=-1)
    if tuple(result.shape) != (spatial.shape[0], spatial.shape[1], len(requested)):
        raise AssertionError("weighted quantiles produced an invalid shape")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("weighted quantiles produced nonfinite values")
    return result


def local_statistics(
    spatial: torch.Tensor, weights: torch.Tensor
) -> dict[str, torch.Tensor]:
    mean = weighted_mean(spatial, weights)
    std = weighted_population_std(spatial, weights)
    quantile = weighted_quantiles(spatial, weights, QUANTILES)
    return {
        "mean": mean,
        "std": std,
        "q25": quantile[..., 0],
        "q50": quantile[..., 1],
        "q75": quantile[..., 2],
    }


def pooling_variant(statistics: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    components = {
        "P1": ("mean",),
        "P2": ("std",),
        "P3": ("mean", "std"),
        "P4": ("q25", "q50", "q75"),
        "P5": ("mean", "std", "q25", "q50", "q75"),
    }
    normalized = str(name).upper()
    if normalized not in components:
        raise ValueError(f"unknown pooling variant: {name}")
    values = [statistics[component] for component in components[normalized]]
    if any(value.ndim != 2 or value.shape != values[0].shape for value in values):
        raise ValueError("pooling statistic component shapes differ")
    return torch.cat(values, dim=-1)


__all__ = [
    "CHANNELS",
    "QUANTILES",
    "local_statistics",
    "pooling_variant",
    "weighted_mean",
    "weighted_population_std",
    "weighted_quantiles",
]
