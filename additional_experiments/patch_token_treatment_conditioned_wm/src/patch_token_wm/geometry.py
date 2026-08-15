"""Exact fixed-LOCAL support, physical coordinates, and token gathering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .contracts import (
    C1B_INPUT_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    FINAL_CENTER_OFFSET_ZYX,
    FINAL_CHANNELS,
    FINAL_STRIDE_ZYX,
    FORMAL_LOCAL_TOKEN_COUNT,
    POSITION_NORMALIZATION_MM,
    audited_expected_feature_shape,
    audited_fixed_physical_local_weights,
    validate_geometry_values,
)


@dataclass(frozen=True)
class LocalTokenGeometry:
    """Outcome-free geometry shared by every patient and visit."""

    input_shape_zyx: tuple[int, int, int]
    feature_shape_zyx: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    dense_weights: torch.Tensor
    flat_indices: torch.Tensor
    weights: torch.Tensor
    coordinates_xyz_mm: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.flat_indices.numel())

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "LocalTokenGeometry":
        if not dtype.is_floating_point:
            raise TypeError("geometry floating tensors require a floating dtype")
        target = torch.device(device)
        return LocalTokenGeometry(
            input_shape_zyx=self.input_shape_zyx,
            feature_shape_zyx=self.feature_shape_zyx,
            spacing_xyz_mm=self.spacing_xyz_mm,
            dense_weights=self.dense_weights.to(device=target, dtype=dtype),
            flat_indices=self.flat_indices.to(device=target),
            weights=self.weights.to(device=target, dtype=dtype),
            coordinates_xyz_mm=self.coordinates_xyz_mm.to(device=target, dtype=dtype),
        )


def derived_feature_shape(
    input_shape_zyx: Sequence[int] = C1B_INPUT_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """Derive final grid dimensions from the audited convolution geometry."""

    shape, _ = validate_geometry_values(input_shape_zyx, C1B_SPACING_XYZ_MM)
    return tuple(
        int(value) for value in audited_expected_feature_shape(shape, stage="final")
    )


def feature_cell_coordinates_xyz_mm(
    input_shape_zyx: Sequence[int],
    feature_shape_zyx: Sequence[int],
    spacing_xyz_mm: Sequence[float],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return crop-centered physical cell centers as ``[D,H,W,3]`` XYZ mm.

    This follows exactly the center formula used by the audited fractional
    overlap routine: ``(offset + j*stride - (input_size-1)/2) * spacing``.
    Tensor axes are ZYX, while the final coordinate component order is XYZ.
    """

    input_shape, spacing_xyz = validate_geometry_values(input_shape_zyx, spacing_xyz_mm)
    feature_shape = tuple(int(value) for value in feature_shape_zyx)
    expected = tuple(
        int(value)
        for value in audited_expected_feature_shape(input_shape, stage="final")
    )
    if feature_shape != expected:
        raise ValueError(
            "feature shape disagrees with audited final geometry: "
            f"expected {expected}, got {feature_shape}"
        )
    if not dtype.is_floating_point:
        raise TypeError("physical coordinates require a floating dtype")
    output_device = torch.device("cpu" if device is None else device)
    spacing_zyx = tuple(reversed(spacing_xyz))
    axes: list[torch.Tensor] = []
    for input_size, feature_size, stride, offset, spacing in zip(
        input_shape,
        feature_shape,
        FINAL_STRIDE_ZYX,
        FINAL_CENTER_OFFSET_ZYX,
        spacing_zyx,
    ):
        index = torch.arange(feature_size, dtype=torch.float64, device=output_device)
        center = 0.5 * (input_size - 1.0)
        axes.append((float(offset) + index * float(stride) - center) * float(spacing))
    z_mm, y_mm, x_mm = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((x_mm, y_mm, z_mm), dim=-1).to(dtype=dtype)


def build_local_token_geometry(
    input_shape_zyx: Sequence[int] = C1B_INPUT_SHAPE_ZYX,
    spacing_xyz_mm: Sequence[float] = C1B_SPACING_XYZ_MM,
    *,
    require_formal_count: bool | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> LocalTokenGeometry:
    """Build strictly-positive audited 64-mm support in stable ZYX-flat order.

    Formal C1B-H geometry is required to contain exactly 500 positions.  Small
    synthetic geometries may opt out (the default outside formal geometry) so
    model contracts can be exercised without allocating the full MRI tensor.
    """

    input_shape, spacing = validate_geometry_values(input_shape_zyx, spacing_xyz_mm)
    feature_shape = tuple(
        int(value)
        for value in audited_expected_feature_shape(input_shape, stage="final")
    )
    weights = audited_fixed_physical_local_weights(
        input_shape,
        feature_shape,
        spacing,
        stage="final",
        device=device,
        dtype=dtype,
    )
    flat = weights.reshape(-1)
    flat_indices = torch.nonzero(flat > 0, as_tuple=False).squeeze(1)
    selected_weights = flat.index_select(0, flat_indices)
    dense_coordinates = feature_cell_coordinates_xyz_mm(
        input_shape,
        feature_shape,
        spacing,
        device=weights.device,
        dtype=dtype,
    )
    selected_coordinates = dense_coordinates.reshape(-1, 3).index_select(
        0, flat_indices
    )
    is_formal = input_shape == C1B_INPUT_SHAPE_ZYX and spacing == C1B_SPACING_XYZ_MM
    enforce = is_formal if require_formal_count is None else bool(require_formal_count)
    if enforce and int(flat_indices.numel()) != FORMAL_LOCAL_TOKEN_COUNT:
        raise ValueError(
            "formal C1B-H positive-overlap support must contain exactly "
            f"{FORMAL_LOCAL_TOKEN_COUNT} cells; got {flat_indices.numel()}"
        )
    if bool((selected_weights <= 0).any()) or not bool(
        torch.isfinite(selected_coordinates).all()
    ):
        raise AssertionError("invalid positive LOCAL token geometry")
    return LocalTokenGeometry(
        input_shape_zyx=input_shape,
        feature_shape_zyx=feature_shape,
        spacing_xyz_mm=spacing,
        dense_weights=weights,
        flat_indices=flat_indices,
        weights=selected_weights,
        coordinates_xyz_mm=selected_coordinates,
    )


def validate_runtime_spatial(
    spatial: torch.Tensor, geometry: LocalTokenGeometry
) -> None:
    if not isinstance(spatial, torch.Tensor) or spatial.ndim != 5:
        raise ValueError("encoder final feature must have shape [N,128,D,H,W]")
    if int(spatial.shape[1]) != FINAL_CHANNELS:
        raise ValueError(f"encoder final feature must have {FINAL_CHANNELS} channels")
    actual = tuple(int(value) for value in spatial.shape[-3:])
    if actual != geometry.feature_shape_zyx:
        raise ValueError(
            "runtime encoder final shape disagrees with dynamically derived geometry: "
            f"expected {geometry.feature_shape_zyx}, got {actual}"
        )
    if spatial.device != geometry.flat_indices.device:
        raise ValueError("spatial features and local geometry must share a device")
    if not spatial.dtype.is_floating_point or not bool(torch.isfinite(spatial).all()):
        raise ValueError("spatial features must contain finite floating values")


def gather_local_tokens(
    spatial: torch.Tensor, geometry: LocalTokenGeometry
) -> torch.Tensor:
    """Gather all positive-overlap cells as ``[N,K,128]`` without pooling."""

    validate_runtime_spatial(spatial, geometry)
    flattened = spatial.flatten(start_dim=2).transpose(1, 2)
    tokens = flattened.index_select(1, geometry.flat_indices)
    expected = (spatial.shape[0], geometry.token_count, FINAL_CHANNELS)
    if tuple(tokens.shape) != expected:
        raise AssertionError(
            f"LOCAL token gather produced {tuple(tokens.shape)}, expected {expected}"
        )
    return tokens


def weighted_local_mean(
    local_tokens: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Canonical fractional LOCAL mean, with no epsilon or empty fallback."""

    if local_tokens.ndim != 3:
        raise ValueError("local_tokens must have shape [N,K,C]")
    if weights.ndim != 1 or weights.shape[0] != local_tokens.shape[1]:
        raise ValueError("weights must have shape [K] matching local_tokens")
    if weights.device != local_tokens.device:
        raise ValueError("weights and local_tokens must share a device")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("selected LOCAL weights must be finite and strictly positive")
    numerator = (local_tokens * weights.to(local_tokens.dtype)[None, :, None]).sum(
        dim=1
    )
    return numerator / weights.to(local_tokens.dtype).sum()


def sinusoidal_physical_position_encoding(
    coordinates_xyz_mm: torch.Tensor,
    embedding_dim: int = 128,
    *,
    normalization_mm: float = POSITION_NORMALIZATION_MM,
) -> torch.Tensor:
    """Deterministic XYZ sinusoidal encoding used only for source/query tokens.

    JEPA target values remain position-free, preventing a shared positional
    component from creating an artificial prediction shortcut.
    """

    if coordinates_xyz_mm.ndim != 2 or coordinates_xyz_mm.shape[-1] != 3:
        raise ValueError("coordinates_xyz_mm must have shape [K,3]")
    if embedding_dim <= 0 or normalization_mm <= 0:
        raise ValueError("embedding_dim and normalization_mm must be positive")
    if not coordinates_xyz_mm.dtype.is_floating_point or not bool(
        torch.isfinite(coordinates_xyz_mm).all()
    ):
        raise ValueError("physical coordinates must contain finite floating values")
    # Interleave sin/cos at geometrically spaced frequencies for each XYZ axis,
    # then truncate/pad deterministically to the exact model width.
    frequencies_per_axis = math.ceil(embedding_dim / 6)
    exponent = torch.arange(
        frequencies_per_axis,
        device=coordinates_xyz_mm.device,
        dtype=coordinates_xyz_mm.dtype,
    )
    if frequencies_per_axis == 1:
        frequencies = torch.ones_like(exponent)
    else:
        frequencies = torch.exp(
            -math.log(10_000.0) * exponent / float(frequencies_per_axis - 1)
        )
    phase = (coordinates_xyz_mm / float(normalization_mm)).unsqueeze(-1) * frequencies
    encoded = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(start_dim=1)
    if encoded.shape[1] < embedding_dim:
        encoded = torch.nn.functional.pad(
            encoded, (0, embedding_dim - encoded.shape[1])
        )
    return encoded[:, :embedding_dim]


__all__ = [
    "LocalTokenGeometry",
    "build_local_token_geometry",
    "derived_feature_shape",
    "feature_cell_coordinates_xyz_mm",
    "gather_local_tokens",
    "sinusoidal_physical_position_encoding",
    "validate_runtime_spatial",
    "weighted_local_mean",
]
