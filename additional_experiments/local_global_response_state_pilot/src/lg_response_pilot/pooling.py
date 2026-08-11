"""Frozen C1B pooling contract backed by the audited implementation.

The numerical primitives below are aliases of the hash-locked spatial-audit
functions, not local reimplementations.  This module only binds those
primitives to the preregistered C1B geometry and validates the actual encoder
tensor before it is pooled.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .contracts import (
    C1B_INPUT_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    FINAL_FEATURE_CHANNELS,
    LOCAL_WINDOW_MM_XYZ,
    validate_c1b_geometry,
)
from .upstream import (
    AUDITED_POOLING_SHA256,
    audited_expected_feature_shape,
    fixed_physical_local_weights as _fixed_physical_local_weights,
    weighted_average_pool as _weighted_average_pool,
)


# Public aliases retain object identity with the audited source functions.
expected_feature_shape = audited_expected_feature_shape
fixed_physical_local_weights = _fixed_physical_local_weights
weighted_average_pool = _weighted_average_pool


def derived_final_feature_shape(
    input_shape_zyx: Sequence[int] = C1B_INPUT_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """Derive, rather than spell out, the frozen encoder's final grid."""

    shape, _, _ = validate_c1b_geometry(
        input_shape_zyx, C1B_SPACING_XYZ_MM, LOCAL_WINDOW_MM_XYZ
    )
    return expected_feature_shape(shape, stage="final")


def build_fixed_c1b_local_weights(
    *,
    input_shape_zyx: Sequence[int] = C1B_INPUT_SHAPE_ZYX,
    spacing_xyz_mm: Sequence[float] = C1B_SPACING_XYZ_MM,
    local_window_mm_xyz: Sequence[float] = LOCAL_WINDOW_MM_XYZ,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the one shared 64-mm fractional sampling-cell weight map."""

    shape, spacing, _ = validate_c1b_geometry(
        input_shape_zyx, spacing_xyz_mm, local_window_mm_xyz
    )
    feature_shape = expected_feature_shape(shape, stage="final")
    return fixed_physical_local_weights(
        shape,
        feature_shape,
        spacing,
        stage="final",
        device=device,
        dtype=dtype,
    )


def validate_actual_final_feature(
    spatial: torch.Tensor,
    *,
    input_shape_zyx: Sequence[int] = C1B_INPUT_SHAPE_ZYX,
    local_weights: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    """Validate the actual full final encoder map against derived geometry."""

    if not isinstance(spatial, torch.Tensor) or spatial.ndim != 5:
        raise ValueError("encoder final map must be [N,128,D,H,W]")
    if int(spatial.shape[1]) != FINAL_FEATURE_CHANNELS:
        raise ValueError("encoder final map must contain exactly 128 channels")
    actual_shape = tuple(int(value) for value in spatial.shape[-3:])
    derived_shape = derived_final_feature_shape(input_shape_zyx)
    if actual_shape != derived_shape:
        raise ValueError(
            "actual encoder.features[3] shape disagrees with audited runtime "
            f"geometry: expected {derived_shape}, got {actual_shape}"
        )
    if local_weights is not None:
        if tuple(int(value) for value in local_weights.shape) != (
            1,
            1,
            *actual_shape,
        ):
            raise ValueError(
                "fixed local-weight buffer differs from the actual final map"
            )
        if local_weights.device != spatial.device:
            raise ValueError(
                "model/local buffer and spatial feature must share a device"
            )
    return actual_shape


def pooling_contract() -> dict[str, Any]:
    """Return the JSON-safe preregistered geometry and weighting contract."""

    return {
        "schema_version": 1,
        "input_shape_zyx": list(C1B_INPUT_SHAPE_ZYX),
        "spacing_xyz_mm": list(C1B_SPACING_XYZ_MM),
        "final_feature_shape_policy": "derived_then_checked_against_actual_runtime_tensor",
        "derived_feature_shape_zyx": list(derived_final_feature_shape()),
        "final_feature_channels": FINAL_FEATURE_CHANNELS,
        "local_window_mm_xyz": list(LOCAL_WINDOW_MM_XYZ),
        "local_center": "frozen_c1b_crop_physical_center",
        "coordinate_convention": "tensor_ZYX_spacing_XYZ",
        "local_weights": "fractional_feature_sampling_cell_overlap",
        "local_uses_receptive_field_occupancy": False,
        "local_weight_shared_across_patients_and_visits": True,
        "audited_pooling_sha256": AUDITED_POOLING_SHA256,
    }


__all__ = [
    "AUDITED_POOLING_SHA256",
    "build_fixed_c1b_local_weights",
    "derived_final_feature_shape",
    "expected_feature_shape",
    "fixed_physical_local_weights",
    "pooling_contract",
    "validate_actual_final_feature",
    "weighted_average_pool",
]
