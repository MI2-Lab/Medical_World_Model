"""Outcome-blind physical regions for the frozen C1B final feature grid.

This module intentionally has no patient-data loader and no mask, bounding-box,
clinical, FTV, treatment, phenotype, or outcome input.  Every weight is a
deterministic function of the frozen image geometry and the *runtime* encoder
feature shape.  A feature location represents its stride-sized sampling cell;
theoretical receptive-field extent is not used for these regions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CHANNELS = 128
INPUT_SHAPE_ZYX = (112, 176, 160)
SPACING_XYZ_MM = (0.9, 0.9, 2.0)
STAGE_STRIDE_ZYX = (8, 8, 8)
STAGE_RECEPTIVE_FIELD_ZYX = (47, 47, 47)
STAGE_PADDING_ZYX = (23, 23, 23)
STAGE_CENTER_OFFSET_ZYX = (0.0, 0.0, 0.0)
PRIMARY_BOUNDARIES_MM = (32.0, 48.0, 64.0)
SECONDARY_BOUNDARIES_MM = (24.0, 40.0, 64.0)
REGION_MEAN_KEYS = (
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R5_RP192",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
)
REGION_DIMENSIONS = {
    "R0": 128,
    "R1": 128,
    "R2": 128,
    "R3": 128,
    "R4": 256,
    "R5": 384,
    "R5_RP192": 192,
    "S1": 128,
    "S2": 128,
    "S3": 128,
    "S4": 256,
    "S5": 384,
}
WEIGHT_KEYS = ("R0", "R1", "R2", "R3", "S1", "S2", "S3")
PROJECTION_SEED = 260812
PROJECTION_INPUT_DIM = 384
PROJECTION_OUTPUT_DIM = 192


def _positive_int_triple(values: Sequence[int], *, name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three ZYX values")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain positive integers")
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise ValueError(f"{name} must contain positive integers") from error
        if integer <= 0:
            raise ValueError(f"{name} must contain positive integers")
        parsed.append(integer)
    return tuple(parsed)  # type: ignore[return-value]


def _finite_positive_triple(
    values: Sequence[float], *, name: str
) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    parsed = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError(f"{name} must contain finite positive values")
    return parsed  # type: ignore[return-value]


def expected_feature_shape(
    input_shape_zyx: Sequence[int] = INPUT_SHAPE_ZYX,
    *,
    stride_zyx: Sequence[int] = STAGE_STRIDE_ZYX,
    receptive_field_zyx: Sequence[int] = STAGE_RECEPTIVE_FIELD_ZYX,
    padding_zyx: Sequence[int] = STAGE_PADDING_ZYX,
) -> tuple[int, int, int]:
    """Derive the final Conv3d grid; no final-map dimensions are hard-coded."""

    input_shape = _positive_int_triple(input_shape_zyx, name="input_shape_zyx")
    stride = _positive_int_triple(stride_zyx, name="stride_zyx")
    receptive_field = _positive_int_triple(
        receptive_field_zyx, name="receptive_field_zyx"
    )
    padding = _positive_int_triple(padding_zyx, name="padding_zyx")
    output = tuple(
        (size + 2 * pad - kernel) // step + 1
        for size, step, kernel, pad in zip(
            input_shape, stride, receptive_field, padding, strict=True
        )
    )
    if any(size <= 0 for size in output):
        raise ValueError("input is too small for the frozen final-stage geometry")
    return output  # type: ignore[return-value]


def cube_sampling_cell_weights(
    cube_size_mm: float,
    feature_shape_zyx: Sequence[int],
    *,
    input_shape_zyx: Sequence[int] = INPUT_SHAPE_ZYX,
    spacing_xyz_mm: Sequence[float] = SPACING_XYZ_MM,
    stride_zyx: Sequence[int] = STAGE_STRIDE_ZYX,
    center_offset_zyx: Sequence[float] = STAGE_CENTER_OFFSET_ZYX,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return `[1,1,D,H,W]` fractional sampling-cell overlap with a cube.

    The cube is centered at the frozen C1B crop physical center.  Tensor axes
    are ZYX while supplied physical spacing is XYZ.
    """

    input_shape = _positive_int_triple(input_shape_zyx, name="input_shape_zyx")
    feature_shape = _positive_int_triple(
        feature_shape_zyx, name="feature_shape_zyx"
    )
    exact_shape = expected_feature_shape(input_shape)
    if feature_shape != exact_shape:
        raise ValueError(
            "runtime feature shape disagrees with frozen convolution geometry: "
            f"expected {exact_shape}, got {feature_shape}"
        )
    spacing_xyz = _finite_positive_triple(spacing_xyz_mm, name="spacing_xyz_mm")
    stride = _positive_int_triple(stride_zyx, name="stride_zyx")
    if len(center_offset_zyx) != 3:
        raise ValueError("center_offset_zyx must contain exactly three values")
    offsets = tuple(float(value) for value in center_offset_zyx)
    if any(not math.isfinite(value) for value in offsets):
        raise ValueError("center_offset_zyx must contain finite values")
    size = float(cube_size_mm)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("cube_size_mm must be finite and positive")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("weight dtype must be floating")

    output_device = torch.device("cpu" if device is None else device)
    spacing_zyx = tuple(reversed(spacing_xyz))
    half_window = 0.5 * size
    fractions: list[torch.Tensor] = []
    # Construct in float64 so shell subtraction and volume checks are stable.
    for input_size, feature_size, step, offset, spacing in zip(
        input_shape,
        feature_shape,
        stride,
        offsets,
        spacing_zyx,
        strict=True,
    ):
        indices = torch.arange(feature_size, dtype=torch.float64, device=output_device)
        crop_center_index = 0.5 * (input_size - 1.0)
        centers_mm = (offset + indices * step - crop_center_index) * spacing
        half_cell_mm = 0.5 * step * spacing
        lower = torch.maximum(
            centers_mm - half_cell_mm,
            torch.full_like(centers_mm, -half_window),
        )
        upper = torch.minimum(
            centers_mm + half_cell_mm,
            torch.full_like(centers_mm, half_window),
        )
        fractions.append((upper - lower).clamp_min(0.0) / (2.0 * half_cell_mm))
    z_fraction, y_fraction, x_fraction = fractions
    weights = (
        z_fraction[:, None, None]
        * y_fraction[None, :, None]
        * x_fraction[None, None, :]
    ).unsqueeze(0).unsqueeze(0)
    if tuple(weights.shape) != (1, 1, *feature_shape):
        raise AssertionError("cube overlap construction produced a wrong shape")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("cube overlap contains nonfinite values")
    tolerance = 16.0 * torch.finfo(torch.float64).eps
    if bool((weights < -tolerance).any()) or bool((weights > 1.0 + tolerance).any()):
        raise ValueError("cube overlap escaped [0,1]")
    weights = weights.clamp(0.0, 1.0)
    if not bool((weights.sum() > 0)):
        raise ValueError("cube has no support on the runtime feature grid")
    return weights.to(dtype=dtype)


@dataclass(frozen=True)
class RegionWeights:
    """The seven unique pooled supports used by R0-R5 and S1-S5."""

    feature_shape_zyx: tuple[int, int, int]
    weights: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if set(self.weights) != set(WEIGHT_KEYS):
            raise ValueError("region weight keys drifted")
        expected = (1, 1, *self.feature_shape_zyx)
        for name, weight in self.weights.items():
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != expected:
                raise ValueError(f"{name} weight shape drifted")
            if not weight.dtype.is_floating_point or not bool(torch.isfinite(weight).all()):
                raise ValueError(f"{name} weights must be finite floating values")
            if bool((weight < 0).any()) or bool((weight > 1).any()):
                raise ValueError(f"{name} weights escaped [0,1]")
            if not bool((weight.sum() > 0)):
                raise ValueError(f"{name} has empty support")

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.weights[name]


def build_region_weights(
    feature_shape_zyx: Sequence[int],
    *,
    input_shape_zyx: Sequence[int] = INPUT_SHAPE_ZYX,
    spacing_xyz_mm: Sequence[float] = SPACING_XYZ_MM,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> RegionWeights:
    """Construct exact nested 32/48/64 and 24/40/64 cube/shell weights."""

    shape = _positive_int_triple(feature_shape_zyx, name="feature_shape_zyx")
    unique_sizes = sorted(set(PRIMARY_BOUNDARIES_MM + SECONDARY_BOUNDARIES_MM))
    cubes64 = {
        size: cube_sampling_cell_weights(
            size,
            shape,
            input_shape_zyx=input_shape_zyx,
            spacing_xyz_mm=spacing_xyz_mm,
            device=device,
            dtype=torch.float64,
        )
        for size in unique_sizes
    }

    def shell(outer: float, inner: float) -> torch.Tensor:
        difference = cubes64[outer] - cubes64[inner]
        tolerance = 64.0 * torch.finfo(torch.float64).eps
        if bool((difference < -tolerance).any()):
            raise AssertionError("nested cube construction produced negative shell weight")
        return difference.clamp_min(0.0)

    raw = {
        "R0": cubes64[64.0],
        "R1": cubes64[32.0],
        "R2": shell(48.0, 32.0),
        "R3": shell(64.0, 48.0),
        "S1": cubes64[24.0],
        "S2": shell(40.0, 24.0),
        "S3": shell(64.0, 40.0),
    }
    if not torch.equal(raw["R1"] + raw["R2"] + raw["R3"], raw["R0"]):
        # Associative floating arithmetic need not be bitwise telescoping, but
        # float64 error must remain at machine precision.
        if not torch.allclose(
            raw["R1"] + raw["R2"] + raw["R3"],
            raw["R0"],
            rtol=0.0,
            atol=64.0 * torch.finfo(torch.float64).eps,
        ):
            raise AssertionError("primary region shells do not partition R0")
    if not torch.allclose(
        raw["S1"] + raw["S2"] + raw["S3"],
        raw["R0"],
        rtol=0.0,
        atol=64.0 * torch.finfo(torch.float64).eps,
    ):
        raise AssertionError("secondary region shells do not partition R0")
    cast = {name: value.to(dtype=dtype) for name, value in raw.items()}
    return RegionWeights(shape, cast)


def weighted_mean(spatial: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Channel-wise normalized weighted mean for `[N,128,D,H,W]`."""

    if not isinstance(spatial, torch.Tensor) or spatial.ndim != 5:
        raise ValueError("spatial must be [N,C,D,H,W]")
    if spatial.shape[0] <= 0 or spatial.shape[1] != CHANNELS:
        raise ValueError(f"spatial must have nonempty N and C={CHANNELS}")
    if not spatial.dtype.is_floating_point or not bool(torch.isfinite(spatial).all()):
        raise ValueError("spatial must contain finite floating values")
    if (
        not isinstance(weights, torch.Tensor)
        or weights.ndim != 5
        or weights.shape[1] != 1
        or weights.shape[0] not in {1, spatial.shape[0]}
        or tuple(weights.shape[-3:]) != tuple(spatial.shape[-3:])
    ):
        raise ValueError("weights must be [N|1,1,D,H,W] on the spatial grid")
    if weights.device != spatial.device:
        raise ValueError("weights and spatial must share a device")
    if not weights.dtype.is_floating_point or not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must contain finite floating values")
    if bool((weights < 0).any()) or bool((weights > 1).any()):
        raise ValueError("weights must lie in [0,1]")
    expanded = weights.expand(spatial.shape[0], -1, -1, -1, -1).to(spatial.dtype)
    denominator = expanded.sum(dim=(-3, -2, -1))
    if not bool((denominator > 0).all()):
        raise ValueError("each weight map must have positive support")
    result = (spatial * expanded).sum(dim=(-3, -2, -1)) / denominator
    if tuple(result.shape) != tuple(spatial.shape[:2]) or not bool(
        torch.isfinite(result).all()
    ):
        raise AssertionError("regional weighted mean produced an invalid result")
    return result


def fixed_qr_projection(
    *,
    seed: int = PROJECTION_SEED,
    input_dim: int = PROJECTION_INPUT_DIM,
    output_dim: int = PROJECTION_OUTPUT_DIM,
) -> np.ndarray:
    """Return the preregistered fixed Gaussian reduced-QR projection matrix."""

    if input_dim <= 0 or output_dim <= 0 or output_dim > input_dim:
        raise ValueError("projection dimensions must satisfy input >= output > 0")
    gaussian = np.random.default_rng(int(seed)).standard_normal(
        (int(input_dim), int(output_dim))
    )
    q, r = np.linalg.qr(gaussian, mode="reduced")
    # QR column signs are otherwise an implementation-dependent convention.
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    if q.shape != (input_dim, output_dim) or not np.isfinite(q).all():
        raise AssertionError("fixed QR projection construction failed")
    if not np.allclose(q.T @ q, np.eye(output_dim), rtol=0.0, atol=1e-12):
        raise AssertionError("fixed QR projection columns are not orthonormal")
    return np.asarray(q, dtype=np.float32)


def projection_sha256(matrix: np.ndarray | None = None) -> str:
    value = fixed_qr_projection() if matrix is None else np.asarray(matrix)
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def extract_region_features(
    spatial: torch.Tensor,
    region_weights: RegionWeights,
    *,
    projection: torch.Tensor | np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Pool one streamed spatial batch into all preregistered feature variants."""

    if tuple(spatial.shape[-3:]) != region_weights.feature_shape_zyx:
        raise ValueError("spatial and region-weight runtime grids differ")
    means = {
        name: weighted_mean(spatial, region_weights[name])
        for name in WEIGHT_KEYS
    }
    result = {
        "R0": means["R0"],
        "R1": means["R1"],
        "R2": means["R2"],
        "R3": means["R3"],
        "R4": torch.cat((means["R1"], means["R2"]), dim=-1),
        "R5": torch.cat((means["R1"], means["R2"], means["R3"]), dim=-1),
        "S1": means["S1"],
        "S2": means["S2"],
        "S3": means["S3"],
        "S4": torch.cat((means["S1"], means["S2"]), dim=-1),
        "S5": torch.cat((means["S1"], means["S2"], means["S3"]), dim=-1),
    }
    matrix = fixed_qr_projection() if projection is None else projection
    matrix_tensor = torch.as_tensor(matrix, dtype=spatial.dtype, device=spatial.device)
    if tuple(matrix_tensor.shape) != (PROJECTION_INPUT_DIM, PROJECTION_OUTPUT_DIM):
        raise ValueError("R5 random-projection matrix shape drifted")
    if not bool(torch.isfinite(matrix_tensor).all()):
        raise ValueError("R5 random-projection matrix contains nonfinite values")
    result["R5_RP192"] = result["R5"] @ matrix_tensor
    if tuple(result) != REGION_MEAN_KEYS:
        # Reorder explicitly to the frozen archive order.
        result = {name: result[name] for name in REGION_MEAN_KEYS}
    for name, value in result.items():
        if tuple(value.shape) != (spatial.shape[0], REGION_DIMENSIONS[name]):
            raise AssertionError(f"{name} feature shape drifted")
    return result


def geometry_contract(region_weights: RegionWeights) -> dict[str, Any]:
    """Build a de-identified public geometry/occupancy contract."""

    spacing_zyx = tuple(reversed(SPACING_XYZ_MM))
    cell_volume = float(np.prod(np.asarray(STAGE_STRIDE_ZYX) * spacing_zyx))
    weight_sum = {
        name: float(weight.detach().double().cpu().sum().item())
        for name, weight in region_weights.weights.items()
    }
    volume = {name: value * cell_volume for name, value in weight_sum.items()}
    expected_volume = {
        "R0": 64.0**3,
        "R1": 32.0**3,
        "R2": 48.0**3 - 32.0**3,
        "R3": 64.0**3 - 48.0**3,
        "S1": 24.0**3,
        "S2": 40.0**3 - 24.0**3,
        "S3": 64.0**3 - 40.0**3,
    }
    max_volume_error = max(
        abs(volume[name] - expected_volume[name]) for name in WEIGHT_KEYS
    )
    primary_residual = (
        region_weights["R1"]
        + region_weights["R2"]
        + region_weights["R3"]
        - region_weights["R0"]
    )
    secondary_residual = (
        region_weights["S1"]
        + region_weights["S2"]
        + region_weights["S3"]
        - region_weights["R0"]
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "GEOMETRY_VALID",
        "feature_shape_zyx": list(region_weights.feature_shape_zyx),
        "input_shape_zyx": list(INPUT_SHAPE_ZYX),
        "spacing_xyz_mm": list(SPACING_XYZ_MM),
        "sampling_cell_stride_zyx": list(STAGE_STRIDE_ZYX),
        "sampling_cell_volume_mm3": cell_volume,
        "center": "frozen_c1b_crop_physical_center",
        "weight_definition": "fractional_feature_sampling_cell_physical_volume_overlap",
        "receptive_field_used_for_region_weight": False,
        "primary_boundaries_mm": list(PRIMARY_BOUNDARIES_MM),
        "secondary_boundaries_mm": list(SECONDARY_BOUNDARIES_MM),
        "weight_sum_cells": weight_sum,
        "physical_volume_mm3": volume,
        "expected_physical_volume_mm3": expected_volume,
        "maximum_physical_volume_error_mm3": max_volume_error,
        "primary_partition_max_abs": float(primary_residual.abs().max().item()),
        "secondary_partition_max_abs": float(secondary_residual.abs().max().item()),
        "minimum_weight": min(
            float(value.min().item()) for value in region_weights.weights.values()
        ),
        "maximum_weight": max(
            float(value.max().item()) for value in region_weights.weights.values()
        ),
        "nonzero_cells": {
            name: int(torch.count_nonzero(value).item())
            for name, value in region_weights.weights.items()
        },
        "fractional_cells": {
            name: int(torch.count_nonzero((value > 0) & (value < 1)).item())
            for name, value in region_weights.weights.items()
        },
        "projection": {
            "variant": "R5_RP192",
            "method": "fixed_gaussian_reduced_qr_orthonormal_columns",
            "seed": PROJECTION_SEED,
            "input_dim": PROJECTION_INPUT_DIM,
            "output_dim": PROJECTION_OUTPUT_DIM,
            "matrix_float32_sha256": projection_sha256(),
        },
        "contains_patient_data": False,
        "uses_mask_bbox_ftv_clinical_treatment_phenotype_or_outcome": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["contract_canonical_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


__all__ = [
    "CHANNELS",
    "INPUT_SHAPE_ZYX",
    "PRIMARY_BOUNDARIES_MM",
    "PROJECTION_INPUT_DIM",
    "PROJECTION_OUTPUT_DIM",
    "PROJECTION_SEED",
    "REGION_DIMENSIONS",
    "REGION_MEAN_KEYS",
    "SECONDARY_BOUNDARIES_MM",
    "SPACING_XYZ_MM",
    "STAGE_CENTER_OFFSET_ZYX",
    "STAGE_PADDING_ZYX",
    "STAGE_RECEPTIVE_FIELD_ZYX",
    "STAGE_STRIDE_ZYX",
    "WEIGHT_KEYS",
    "RegionWeights",
    "build_region_weights",
    "cube_sampling_cell_weights",
    "expected_feature_shape",
    "extract_region_features",
    "fixed_qr_projection",
    "geometry_contract",
    "projection_sha256",
    "weighted_mean",
]
