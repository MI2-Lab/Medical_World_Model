"""Deterministic spatial pooling primitives for the frozen C1B audit.

This module deliberately contains no data loading, target access, probe fitting, or
trainable readout.  It maps already-extracted spatial features and outcome-free
geometry masks to preregistered pooled response states.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


_TRIPLE = tuple[int, int, int]
LOCAL_WINDOW_MM_XYZ = (64.0, 64.0, 64.0)
FINAL_CHANNELS = 128
RESPONSE_DIM = 192


def _positive_int_triple(values: Sequence[int], *, name: str) -> _TRIPLE:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three ZYX values")
    output: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain positive integers")
        try:
            parsed = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain positive integers") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must contain positive integers")
        output.append(parsed)
    return tuple(output)  # type: ignore[return-value]


@dataclass(frozen=True)
class StageGeometry:
    """Frozen input-to-feature geometry for one allowed encoder stage."""

    name: str
    channels: int
    receptive_field_zyx: _TRIPLE
    stride_zyx: _TRIPLE
    padding_zyx: _TRIPLE
    center_offset_zyx: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.name not in {"final", "s3"}:
            raise ValueError("only the preregistered final and s3 stages are allowed")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        receptive_field = _positive_int_triple(
            self.receptive_field_zyx, name="receptive_field_zyx"
        )
        stride = _positive_int_triple(self.stride_zyx, name="stride_zyx")
        padding = _positive_int_triple(self.padding_zyx, name="padding_zyx")
        if len(self.center_offset_zyx) != 3 or any(
            not torch.isfinite(torch.tensor(float(value)))
            for value in self.center_offset_zyx
        ):
            raise ValueError("center_offset_zyx must contain three finite values")
        if any(kernel % 2 != 1 for kernel in receptive_field):
            raise ValueError("receptive fields must be odd")
        expected_padding = tuple((kernel - 1) // 2 for kernel in receptive_field)
        if padding != expected_padding:
            raise ValueError("padding must preserve the frozen centered RF geometry")
        object.__setattr__(self, "receptive_field_zyx", receptive_field)
        object.__setattr__(self, "stride_zyx", stride)
        object.__setattr__(self, "padding_zyx", padding)


FINAL_STAGE_GEOMETRY = StageGeometry(
    name="final",
    channels=FINAL_CHANNELS,
    receptive_field_zyx=(47, 47, 47),
    stride_zyx=(8, 8, 8),
    padding_zyx=(23, 23, 23),
    center_offset_zyx=(0.0, 0.0, 0.0),
)
S3_STAGE_GEOMETRY = StageGeometry(
    name="s3",
    channels=64,
    receptive_field_zyx=(23, 23, 23),
    stride_zyx=(4, 4, 4),
    padding_zyx=(11, 11, 11),
    center_offset_zyx=(0.0, 0.0, 0.0),
)


def _stage_geometry(stage: str) -> StageGeometry:
    normalized = str(stage).lower()
    if normalized == "final":
        return FINAL_STAGE_GEOMETRY
    if normalized == "s3":
        return S3_STAGE_GEOMETRY
    raise ValueError(f"unknown/unregistered spatial stage: {stage!r}")


def expected_feature_shape(
    input_spatial_shape_zyx: Sequence[int], *, stage: str = "final"
) -> _TRIPLE:
    """Return the exact Conv3d/avg-pool output shape for a frozen stage."""

    input_shape = _positive_int_triple(
        input_spatial_shape_zyx, name="input_spatial_shape_zyx"
    )
    geometry = _stage_geometry(stage)
    output = tuple(
        (size + 2 * padding - kernel) // stride + 1
        for size, kernel, stride, padding in zip(
            input_shape,
            geometry.receptive_field_zyx,
            geometry.stride_zyx,
            geometry.padding_zyx,
        )
    )
    if any(size <= 0 for size in output):
        raise ValueError("input is too small for the preregistered stage geometry")
    return output  # type: ignore[return-value]


def _validate_spatial_feature(spatial: torch.Tensor) -> None:
    if not isinstance(spatial, torch.Tensor):
        raise TypeError("spatial must be a torch.Tensor")
    if spatial.ndim != 5:
        raise ValueError(
            f"spatial must have shape [N,C,D,H,W], got {tuple(spatial.shape)}"
        )
    if any(size <= 0 for size in spatial.shape):
        raise ValueError("spatial dimensions must all be nonempty")
    if not spatial.dtype.is_floating_point:
        raise TypeError("spatial must have a floating dtype")
    if not bool(torch.isfinite(spatial).all()):
        raise ValueError("spatial must contain only finite values")


def _validate_weights(weights: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if weights.ndim != 5 or weights.shape[1] != 1:
        raise ValueError(
            f"weights must have shape [N|1,1,D,H,W], got {tuple(weights.shape)}"
        )
    if weights.shape[0] not in {1, spatial.shape[0]}:
        raise ValueError("weights batch must be one or exactly match spatial batch")
    if tuple(weights.shape[-3:]) != tuple(spatial.shape[-3:]):
        raise ValueError("weights and spatial feature grids must match exactly")
    if weights.device != spatial.device:
        raise ValueError("weights and spatial must be on the same device")
    if not weights.dtype.is_floating_point:
        raise TypeError("weights must have a floating dtype")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must contain only finite values")
    if bool((weights < 0).any()) or bool((weights > 1).any()):
        raise ValueError("weights must lie in the closed interval [0,1]")
    support = weights.sum(dim=(-3, -2, -1))
    if not bool((support > 0).all()):
        raise ValueError("every weight map must have nonempty support")
    return weights.to(dtype=spatial.dtype)


def receptive_field_occupancy(
    input_mask: torch.Tensor,
    feature_spatial_shape_zyx: Sequence[int],
    *,
    stage: str = "final",
) -> torch.Tensor:
    """Map an input mask to exact theoretical-RF fractional occupancy.

    External neural padding is included in the fixed denominator as zero.  This
    is therefore exactly ``avg_pool3d(mask, k, s, p, count_include_pad=True)``
    with the preregistered final or S3 geometry, not an arbitrary resize.
    """

    if not isinstance(input_mask, torch.Tensor):
        raise TypeError("input_mask must be a torch.Tensor")
    if input_mask.ndim != 5 or input_mask.shape[1] != 1:
        raise ValueError(
            f"input_mask must have shape [N,1,Z,Y,X], got {tuple(input_mask.shape)}"
        )
    if any(size <= 0 for size in input_mask.shape):
        raise ValueError("input_mask dimensions must all be nonempty")
    if input_mask.dtype.is_complex:
        raise TypeError("input_mask must be real-valued")
    if input_mask.dtype.is_floating_point and not bool(torch.isfinite(input_mask).all()):
        raise ValueError("input_mask must contain only finite values")
    if bool((input_mask < 0).any()) or bool((input_mask > 1).any()):
        raise ValueError("input_mask must lie in the closed interval [0,1]")

    requested_shape = _positive_int_triple(
        feature_spatial_shape_zyx, name="feature_spatial_shape_zyx"
    )
    exact_shape = expected_feature_shape(input_mask.shape[-3:], stage=stage)
    if requested_shape != exact_shape:
        raise ValueError(
            "feature shape disagrees with frozen convolution geometry: "
            f"expected {exact_shape}, got {requested_shape}"
        )

    work_dtype = torch.float64 if input_mask.dtype == torch.float64 else torch.float32
    mask = input_mask.to(dtype=work_dtype)
    input_support = mask.sum(dim=(-3, -2, -1))
    if not bool((input_support > 0).all()):
        raise ValueError("every input mask must have nonempty support")

    geometry = _stage_geometry(stage)
    occupancy = F.avg_pool3d(
        mask,
        kernel_size=geometry.receptive_field_zyx,
        stride=geometry.stride_zyx,
        padding=geometry.padding_zyx,
        count_include_pad=True,
    )
    if tuple(occupancy.shape[-3:]) != exact_shape:
        raise AssertionError("PyTorch occupancy output violated frozen geometry")
    if not bool(torch.isfinite(occupancy).all()):
        raise ValueError("feature occupancy contains nonfinite values")
    if bool((occupancy < 0).any()) or bool((occupancy > 1).any()):
        raise ValueError("feature occupancy escaped [0,1]")
    if not bool((occupancy.sum(dim=(-3, -2, -1)) > 0).all()):
        raise ValueError("an input mask has no support on the feature grid")
    return occupancy


def global_average_pool(spatial: torch.Tensor) -> torch.Tensor:
    """P0: exact unweighted mean over the final three spatial axes."""

    _validate_spatial_feature(spatial)
    pooled = spatial.mean(dim=(-3, -2, -1))
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("global pooled features contain nonfinite values")
    return pooled


def weighted_average_pool(spatial: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """PVALID/PLOCAL/PORACLE normalized weighted spatial mean.

    A singleton weight batch is explicitly allowed for a shared fixed local
    window.  Empty support is an error; there is no GAP fallback or epsilon
    denominator that could silently change the preregistered population.
    """

    _validate_spatial_feature(spatial)
    normalized_weights = _validate_weights(weights, spatial)
    denominator = normalized_weights.sum(dim=(-3, -2, -1))
    pooled = (spatial * normalized_weights).sum(dim=(-3, -2, -1)) / denominator
    if tuple(pooled.shape) != tuple(spatial.shape[:2]):
        raise AssertionError("weighted pooling produced an unexpected shape")
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("weighted pooled features contain nonfinite values")
    return pooled


def _spacing_batch_xyz(
    spacing_xyz_mm: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(spacing_xyz_mm, torch.Tensor):
        if spacing_xyz_mm.requires_grad:
            raise ValueError("physical spacing must not require gradients")
        if spacing_xyz_mm.dtype.is_complex:
            raise TypeError("physical spacing must be real-valued")
    try:
        spacing = torch.as_tensor(
            spacing_xyz_mm, dtype=torch.float64, device=device
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing_xyz_mm must be a triplet or batch of triplets") from exc
    if spacing.ndim == 1:
        spacing = spacing.unsqueeze(0)
    if spacing.ndim != 2 or spacing.shape[0] <= 0 or spacing.shape[1] != 3:
        raise ValueError("spacing_xyz_mm must have shape [3] or [N,3]")
    if not bool(torch.isfinite(spacing).all()) or not bool((spacing > 0).all()):
        raise ValueError("spacing_xyz_mm must contain finite positive values")
    return spacing


def fixed_physical_local_weights(
    input_spatial_shape_zyx: Sequence[int],
    feature_spatial_shape_zyx: Sequence[int],
    spacing_xyz_mm: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    *,
    stage: str = "final",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return sampling-cell fractional overlap with the fixed central 64-mm cube.

    Feature location ``j`` is centered at frozen input voxel-center index
    ``offset + j*stride``.  Its local sampling cell is exactly ``stride`` input
    voxels wide per axis.  The returned weight is the separable physical-volume
    overlap fraction with the central 64x64x64-mm cube.  The receptive-field
    extent is intentionally *not* used for PLOCAL.

    Spacing is supplied in XYZ order while tensor shapes are ZYX.  A ``[N,3]``
    spacing batch supports visit-specific legacy geometry; a single triplet
    returns one broadcastable weight map.
    """

    input_shape = _positive_int_triple(
        input_spatial_shape_zyx, name="input_spatial_shape_zyx"
    )
    feature_shape = _positive_int_triple(
        feature_spatial_shape_zyx, name="feature_spatial_shape_zyx"
    )
    exact_shape = expected_feature_shape(input_shape, stage=stage)
    if feature_shape != exact_shape:
        raise ValueError(
            "feature shape disagrees with frozen convolution geometry: "
            f"expected {exact_shape}, got {feature_shape}"
        )
    if not dtype.is_floating_point:
        raise TypeError("local weight dtype must be floating")
    output_device = torch.device(device) if device is not None else (
        spacing_xyz_mm.device
        if isinstance(spacing_xyz_mm, torch.Tensor)
        else torch.device("cpu")
    )
    spacing_xyz = _spacing_batch_xyz(spacing_xyz_mm, device=output_device)
    spacing_zyx = torch.flip(spacing_xyz, dims=(1,))
    geometry = _stage_geometry(stage)
    half_window_mm = 0.5 * LOCAL_WINDOW_MM_XYZ[0]

    axis_fractions: list[torch.Tensor] = []
    for axis, (input_size, feature_size, stride, offset) in enumerate(
        zip(
            input_shape,
            feature_shape,
            geometry.stride_zyx,
            geometry.center_offset_zyx,
        )
    ):
        spacing_mm = spacing_zyx[:, axis : axis + 1]
        feature_indices = torch.arange(
            feature_size, dtype=torch.float64, device=output_device
        ).unsqueeze(0)
        crop_center_index = 0.5 * (input_size - 1.0)
        centers_mm = (
            float(offset) + feature_indices * float(stride) - crop_center_index
        ) * spacing_mm
        half_cell_mm = 0.5 * float(stride) * spacing_mm
        lower = torch.maximum(
            centers_mm - half_cell_mm,
            torch.full_like(centers_mm, -half_window_mm),
        )
        upper = torch.minimum(
            centers_mm + half_cell_mm,
            torch.full_like(centers_mm, half_window_mm),
        )
        overlap_mm = (upper - lower).clamp_min(0.0)
        axis_fractions.append(overlap_mm / (2.0 * half_cell_mm))

    z_fraction, y_fraction, x_fraction = axis_fractions
    weights = (
        z_fraction[:, :, None, None]
        * y_fraction[:, None, :, None]
        * x_fraction[:, None, None, :]
    ).unsqueeze(1)
    if tuple(weights.shape[2:]) != feature_shape:
        raise AssertionError("local overlap construction produced a wrong feature shape")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("local overlap weights contain nonfinite values")
    roundoff_tolerance = 16.0 * torch.finfo(weights.dtype).eps
    if bool((weights < -roundoff_tolerance).any()) or bool(
        (weights > 1.0 + roundoff_tolerance).any()
    ):
        raise ValueError("local overlap weights escaped [0,1]")
    weights = weights.clamp(0.0, 1.0)
    if not bool((weights.sum(dim=(-3, -2, -1)) > 0).all()):
        raise ValueError("the fixed 64-mm window has no feature-grid support")
    return weights.to(dtype=dtype)


def apply_frozen_response_projection(
    pooled: torch.Tensor, response_projection: nn.Module
) -> torch.Tensor:
    """Apply only the eval-mode frozen online 128->192 response projection."""

    if not isinstance(pooled, torch.Tensor):
        raise TypeError("pooled must be a torch.Tensor")
    if pooled.ndim != 2 or pooled.shape[0] <= 0 or pooled.shape[1] != FINAL_CHANNELS:
        raise ValueError(
            f"pooled must have shape [N,{FINAL_CHANNELS}], got {tuple(pooled.shape)}"
        )
    if not pooled.dtype.is_floating_point or not bool(torch.isfinite(pooled).all()):
        raise ValueError("pooled features must be finite floating values")
    if not isinstance(response_projection, nn.Module):
        raise TypeError("response_projection must be a torch.nn.Module")
    if response_projection.training:
        raise ValueError("response_projection must be in eval mode for frozen export")
    parameters = tuple(response_projection.parameters())
    if not parameters:
        raise ValueError("response_projection has no checkpoint parameters")
    if any(parameter.device != pooled.device for parameter in parameters):
        raise ValueError("pooled features and response_projection must share a device")
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise ValueError("response_projection parameters must be finite")

    parameter_versions = tuple(parameter._version for parameter in parameters)
    with torch.inference_mode():
        response = response_projection(pooled)
    if tuple(parameter._version for parameter in parameters) != parameter_versions:
        raise RuntimeError("response_projection mutated during frozen inference")
    if tuple(response.shape) != (pooled.shape[0], RESPONSE_DIM):
        raise ValueError(
            "response_projection must produce the frozen 192-D state; "
            f"got {tuple(response.shape)}"
        )
    if not response.dtype.is_floating_point or not bool(torch.isfinite(response).all()):
        raise ValueError("projected response must contain finite floating values")
    if response.requires_grad:
        raise AssertionError("frozen response projection unexpectedly retained gradients")
    return response


def concatenate_local_global(
    local_response: torch.Tensor, global_response: torch.Tensor
) -> torch.Tensor:
    """Primary PLOCAL+GLOBAL state: direct [local; global] concatenation."""

    for name, response in (
        ("local_response", local_response),
        ("global_response", global_response),
    ):
        if not isinstance(response, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if response.ndim < 2 or response.shape[-1] != RESPONSE_DIM:
            raise ValueError(f"{name} must end in the frozen {RESPONSE_DIM}-D state")
        if not response.dtype.is_floating_point or not bool(torch.isfinite(response).all()):
            raise ValueError(f"{name} must contain finite floating values")
    if local_response.shape != global_response.shape:
        raise ValueError("local and global response shapes must match exactly")
    if local_response.device != global_response.device:
        raise ValueError("local and global responses must share a device")
    if local_response.dtype != global_response.dtype:
        raise ValueError("local and global responses must share a dtype")
    combined = torch.cat((local_response, global_response), dim=-1)
    if combined.shape[-1] != 2 * RESPONSE_DIM:
        raise AssertionError("local-global concatenation produced a wrong dimension")
    return combined
