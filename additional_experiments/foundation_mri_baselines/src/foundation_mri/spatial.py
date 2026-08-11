"""Frozen spatial and channel adapters for the C1B-H DCE7 tensor.

The input is always the audited C1B-H tensor ``[7,112,176,160]`` for one
visit.  Its axis order is channel/ZYX and its spacing is XYZ=(0.9,0.9,2.0)
mm.  LOCAL is a mask-free, fixed 64-mm cube about the frozen C1B physical
centre.  It therefore inherits the upstream T0-localisation prior but never
loads a lesion mask or outcome.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


C1B_CHANNELS = (
    "pre",
    "early",
    "late",
    "early_minus_pre",
    "late_minus_pre",
    "peak_relative_enhancement",
    "late_minus_peak_relative_enhancement",
)
C1B_VISIT_SHAPE = (7, 112, 176, 160)
C1B_SPACING_ZYX_MM = (2.0, 0.9, 0.9)
LOCAL_CUBE_MM = 64.0

# Physics-selected before any test evaluation: early, late, late-pre.
DINO_CHANNEL_INDICES = (1, 2, 4)
DINO_CHANNEL_NAMES = tuple(C1B_CHANNELS[index] for index in DINO_CHANNEL_INDICES)
DINO_AXIAL_SLICES = 32
DINO_LOCAL_NATIVE_SHAPE = (32, 72, 72)
DINO_IMAGE_SIZE = 224
MEDICALNET_FEATURE_SHAPE = (14, 22, 20)


def validate_visit(volume: torch.Tensor) -> torch.Tensor:
    """Validate one finite, clipped C1B-H visit without changing its values."""

    if not isinstance(volume, torch.Tensor) or tuple(volume.shape) != C1B_VISIT_SHAPE:
        raise ValueError(
            f"C1B visit must have shape {C1B_VISIT_SHAPE}, got "
            f"{getattr(volume, 'shape', None)}"
        )
    if not volume.is_floating_point():
        raise TypeError("C1B visit must be floating point")
    if not torch.isfinite(volume).all():
        raise FloatingPointError("C1B visit contains NaN/Inf")
    if bool(torch.any(volume < -5.000001)) or bool(torch.any(volume > 5.000001)):
        raise ValueError("C1B visit violates the frozen [-5,5] clip")
    return volume


def _target_offsets(count: int, extent_mm: float, reference: torch.Tensor) -> torch.Tensor:
    """Return target-cell centres spanning exactly ``extent_mm``."""

    if int(count) <= 0 or float(extent_mm) <= 0:
        raise ValueError("target count and extent must be positive")
    return (
        (torch.arange(count, device=reference.device, dtype=torch.float32) + 0.5)
        / float(count)
        - 0.5
    ) * float(extent_mm)


def fixed_physical_center_crop(
    volume: torch.Tensor,
    *,
    target_shape_zyx: Sequence[int],
    cube_mm: float = LOCAL_CUBE_MM,
    spacing_zyx_mm: Sequence[float] = C1B_SPACING_ZYX_MM,
) -> torch.Tensor:
    """Trilinearly sample the exact central physical cube.

    ``grid_sample(..., align_corners=True)`` is used with explicit source
    voxel-centre coordinates.  Target sample centres span the 64-mm support;
    no mask, bounding box, label, or per-patient parameter is accepted.
    """

    validate_visit(volume)
    target = tuple(int(value) for value in target_shape_zyx)
    spacing = tuple(float(value) for value in spacing_zyx_mm)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError("target_shape_zyx must contain three positive integers")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing_zyx_mm must contain three positive values")

    source_shape = tuple(int(value) for value in volume.shape[-3:])
    coordinates: list[torch.Tensor] = []
    for count, source_count, source_spacing in zip(
        target, source_shape, spacing, strict=True
    ):
        offsets = _target_offsets(count, cube_mm, volume)
        indices = (float(source_count) - 1.0) / 2.0 + offsets / source_spacing
        normalized = 2.0 * indices / float(source_count - 1) - 1.0
        coordinates.append(normalized)
    zz, yy, xx = torch.meshgrid(*coordinates, indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        volume.float().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    if tuple(sampled.shape) != (C1B_VISIT_SHAPE[0], *target):
        raise AssertionError("physical crop produced an unexpected shape")
    if not torch.isfinite(sampled).all():
        raise FloatingPointError("physical crop produced NaN/Inf")
    return sampled


def global_resample(volume: torch.Tensor, target_shape_zyx: Sequence[int]) -> torch.Tensor:
    """Resize the full, fixed C1B field of view to an architecture input grid."""

    validate_visit(volume)
    target = tuple(int(value) for value in target_shape_zyx)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError("target_shape_zyx must contain three positive integers")
    output = F.interpolate(
        volume.float().unsqueeze(0),
        size=target,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    if tuple(output.shape) != (C1B_VISIT_SHAPE[0], *target):
        raise AssertionError("global resize produced an unexpected shape")
    return output


def dino_slice_stack(volume: torch.Tensor, spatial_axis: str) -> torch.Tensor:
    """Create 32 RGB-like axial images as ``[slice,3,224,224]``.

    Channel semantics are fixed to early/late/late-minus-pre.  C1B values are
    not normalised here; the DINO adapter applies the separately frozen
    ``[-5,5] -> [0,1] -> ImageNet normalisation`` mapping.
    """

    validate_visit(volume)
    selected = volume[list(DINO_CHANNEL_INDICES)]
    axis = str(spatial_axis).upper()
    if axis == "GLOBAL":
        # Preserve in-plane aspect ratio: uniformly resample the full Z axis,
        # symmetrically pad the fixed 176x160 C1B plane to a square, then resize.
        axial = F.interpolate(
            selected.float().unsqueeze(0),
            size=(DINO_AXIAL_SLICES, C1B_VISIT_SHAPE[-2], C1B_VISIT_SHAPE[-1]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        slices = axial.permute(1, 0, 2, 3)
        pad_total = int(C1B_VISIT_SHAPE[-2] - C1B_VISIT_SHAPE[-1])
        slices = F.pad(slices, (pad_total // 2, pad_total - pad_total // 2, 0, 0))
        slices = F.interpolate(
            slices,
            size=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        stack = slices.permute(1, 0, 2, 3)
    elif axis == "LOCAL":
        local = fixed_physical_center_crop(
            volume, target_shape_zyx=DINO_LOCAL_NATIVE_SHAPE
        )[list(DINO_CHANNEL_INDICES)]
        slices = local.permute(1, 0, 2, 3)
        slices = F.interpolate(
            slices,
            size=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        stack = slices.permute(1, 0, 2, 3)
    else:
        raise ValueError("spatial_axis must be GLOBAL or LOCAL")
    output = stack.permute(1, 0, 2, 3).contiguous()
    if tuple(output.shape) != (DINO_AXIAL_SLICES, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE):
        raise AssertionError("DINO slice adapter produced an unexpected shape")
    return output


def medicalnet_volume_batch(volume: torch.Tensor) -> torch.Tensor:
    """Create seven native-grid shared-encoder inputs.

    GLOBAL and LOCAL are pooled from the same layer4 map.  Keeping the native
    C1B grid both avoids a second geometry adapter and yields the exact
    ``[14,22,20]`` output grid used by the audited 64-mm overlap primitive.
    """

    validate_visit(volume)
    output = volume.float().unsqueeze(1).contiguous()
    if tuple(output.shape) != (7, 1, *C1B_VISIT_SHAPE[-3:]):
        raise AssertionError("MedicalNet channel adapter produced an unexpected shape")
    return output


def spatial_contract() -> dict[str, object]:
    """Return the JSON-safe, pre-test-frozen input contract."""

    return {
        "c1b_visit_shape_czyx": list(C1B_VISIT_SHAPE),
        "c1b_spacing_zyx_mm": list(C1B_SPACING_ZYX_MM),
        "c1b_channel_order": list(C1B_CHANNELS),
        "local_cube_mm_xyz": [LOCAL_CUBE_MM] * 3,
        "local_center": "frozen_c1b_crop_physical_center",
        "local_uses_lesion_mask": False,
        "local_inherits_frozen_t0_localisation_prior": True,
        "dino_channels": list(DINO_CHANNEL_NAMES),
        "dino_channel_indices": list(DINO_CHANNEL_INDICES),
        "dino_axial_slices": DINO_AXIAL_SLICES,
        "dino_image_size": DINO_IMAGE_SIZE,
        "dino_global_sampling": (
            "trilinear_Z_to_32_keep_176x160_then_symmetric_square_pad_"
            "and_bicubic_resize_224"
        ),
        "dino_local_native_shape_zyx": list(DINO_LOCAL_NATIVE_SHAPE),
        "medicalnet_adapter": (
            "native_C1B_grid_shared_1channel_encoder_per_DCE7_channel;_"
            "GLOBAL_and_fixed64mm_LOCAL_pool_from_same_layer4_map_then_concat"
        ),
        "medicalnet_input_shape_zyx": list(C1B_VISIT_SHAPE[-3:]),
        "medicalnet_feature_shape_zyx": list(MEDICALNET_FEATURE_SHAPE),
    }


__all__ = [
    "C1B_CHANNELS",
    "C1B_SPACING_ZYX_MM",
    "C1B_VISIT_SHAPE",
    "DINO_AXIAL_SLICES",
    "DINO_CHANNEL_INDICES",
    "DINO_CHANNEL_NAMES",
    "DINO_IMAGE_SIZE",
    "LOCAL_CUBE_MM",
    "MEDICALNET_FEATURE_SHAPE",
    "dino_slice_stack",
    "fixed_physical_center_crop",
    "global_resample",
    "medicalnet_volume_batch",
    "spatial_contract",
    "validate_visit",
]
