"""Outcome-blind physical-geometry primitives for the BPE FOV audit.

The frozen repository convention is preserved: a spatial affine maps XYZ
voxel *centres* into RAS millimetres, while physical support includes the full
outer footprint from -0.5 through shape-0.5.  No function in this module reads
clinical labels or chooses a crop from target values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


VISITS = ("T0", "T1", "T2", "T3")
INTERVALS = ("T0->T1", "T1->T2", "T2->T3")
CLASSIFICATIONS = {
    "A": "FULL_C1B_CONTEXT_SUFFICIENT",
    "B": "BROADER_BILATERAL_CONTEXT_REQUIRED",
    "C": "BPE_ALREADY_LOCAL_OBSERVABLE",
    "D": "BPE_SOURCE_NOT_RELIABLY_AUDITABLE",
    "E": "SOURCE_ACQUISITION_LIMITED",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_case_sample(
    patient_ids: Iterable[str], sample_size: int, *, salt: str
) -> tuple[list[str], str]:
    """Select cases without outcome, target magnitude, or filesystem order."""

    unique = sorted({str(value) for value in patient_ids})
    size = int(sample_size)
    if size < 1 or size > len(unique):
        raise ValueError("sample_size must be within the unique patient count")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest(),
    )
    selected = ranked[:size]
    selection_digest = hashlib.sha256(
        (salt + "\0" + "\0".join(selected)).encode("utf-8")
    ).hexdigest()
    return selected, selection_digest


def validate_affine(affine: np.ndarray, *, name: str = "affine") -> np.ndarray:
    result = np.asarray(affine, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} is not homogeneous")
    singular = np.linalg.svd(result[:3, :3], compute_uv=False)
    if singular[-1] <= np.finfo(np.float64).eps * max(singular[0], 1.0) * 16.0:
        raise ValueError(f"{name} is singular")
    return result


def validate_shape_xyz(shape_xyz: Sequence[int]) -> np.ndarray:
    shape = np.asarray(shape_xyz, dtype=np.int64)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError(f"shape_xyz must contain three positive integers, got {shape}")
    return shape


def voxel_footprint_corners_xyz(shape_xyz: Sequence[int]) -> np.ndarray:
    """Return the eight outer voxel-footprint corners in voxel coordinates."""

    shape = validate_shape_xyz(shape_xyz).astype(np.float64)
    return np.asarray(
        list(itertools.product(*[(-0.5, length - 0.5) for length in shape])),
        dtype=np.float64,
    )


def apply_affine(affine: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    spatial = validate_affine(affine)
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N,3]")
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (homogeneous @ spatial.T)[:, :3]


def physical_bounds_xyz(
    shape_xyz: Sequence[int], affine_ras: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    corners = apply_affine(affine_ras, voxel_footprint_corners_xyz(shape_xyz))
    return corners.min(axis=0), corners.max(axis=0)


def physical_extent_xyz(shape_xyz: Sequence[int], affine_ras: np.ndarray) -> np.ndarray:
    low, high = physical_bounds_xyz(shape_xyz, affine_ras)
    return high - low


def grid_center_from_affine(
    shape_xyz: Sequence[int], affine_ras: np.ndarray
) -> np.ndarray:
    shape = validate_shape_xyz(shape_xyz).astype(np.float64)
    center_voxel = 0.5 * (shape - 1.0)
    return apply_affine(affine_ras, center_voxel[None, :])[0]


def is_axis_aligned_ras(affine_ras: np.ndarray, *, tolerance: float = 1e-6) -> bool:
    affine = validate_affine(affine_ras)
    linear = affine[:3, :3]
    diagonal = np.diag(np.diag(linear))
    return bool(
        np.all(np.diag(linear) > 0)
        and np.allclose(linear, diagonal, atol=tolerance, rtol=0.0)
    )


@dataclass(frozen=True)
class ROIFOVAudit:
    source_positive_voxels: int
    source_volume_mm3: float
    intersection_volume_mm3: float
    occupancy: float
    boundary_touch_x: bool
    boundary_touch_y: bool
    boundary_touch_z: bool
    boundary_touch_any: bool
    centroid_x_ras_mm: float
    centroid_y_ras_mm: float
    centroid_z_ras_mm: float
    centroid_inside: bool
    physical_margin_mm: float


def audit_axis_aligned_roi_against_fov(
    roi_mask_xyz: np.ndarray,
    roi_affine_ras: np.ndarray,
    fov_shape_xyz: Sequence[int],
    fov_affine_ras: np.ndarray,
    *,
    tolerance_mm: float = 1e-6,
) -> ROIFOVAudit:
    """Compute exact box intersection for a truly RAS+, axis-aligned ROI grid.

    The current audit cannot call this routine because the true source ROI is
    absent.  It is nevertheless implemented and synthetically tested so a
    future authoritative mask can be audited without changing gate semantics.
    Oblique/sheared ROI grids fail closed instead of using an axis-index proxy.
    """

    mask = np.asarray(roi_mask_xyz, dtype=bool)
    if mask.ndim != 3 or any(length < 1 for length in mask.shape):
        raise ValueError("roi_mask_xyz must be a non-empty three-dimensional array")
    if not mask.any():
        raise ValueError("source ROI is empty")
    roi_affine = validate_affine(roi_affine_ras, name="ROI affine")
    fov_affine = validate_affine(fov_affine_ras, name="FOV affine")
    if not is_axis_aligned_ras(roi_affine) or not is_axis_aligned_ras(fov_affine):
        raise ValueError("exact audit requires truly RAS+, axis-aligned grids")

    coordinates = np.argwhere(mask).astype(np.float64)
    centers = apply_affine(roi_affine, coordinates)
    roi_spacing = np.diag(roi_affine[:3, :3])
    half = 0.5 * roi_spacing
    voxel_low = centers - half[None, :]
    voxel_high = centers + half[None, :]
    fov_low, fov_high = physical_bounds_xyz(fov_shape_xyz, fov_affine)
    overlap_low = np.maximum(voxel_low, fov_low[None, :])
    overlap_high = np.minimum(voxel_high, fov_high[None, :])
    intersection = np.clip(overlap_high - overlap_low, 0.0, None)
    intersection_volume = float(np.prod(intersection, axis=1).sum())
    voxel_volume = float(np.prod(roi_spacing))
    source_volume = float(len(coordinates) * voxel_volume)
    occupancy = float(intersection_volume / source_volume)

    roi_low = voxel_low.min(axis=0)
    roi_high = voxel_high.max(axis=0)
    overlaps = np.all(
        (voxel_high >= fov_low[None, :] - tolerance_mm)
        & (voxel_low <= fov_high[None, :] + tolerance_mm),
        axis=1,
    )
    touches = []
    for axis in range(3):
        low_touch = np.any(
            overlaps
            & (voxel_low[:, axis] <= fov_low[axis] + tolerance_mm)
            & (voxel_high[:, axis] >= fov_low[axis] - tolerance_mm)
        )
        high_touch = np.any(
            overlaps
            & (voxel_low[:, axis] <= fov_high[axis] + tolerance_mm)
            & (voxel_high[:, axis] >= fov_high[axis] - tolerance_mm)
        )
        touches.append(bool(low_touch or high_touch))

    centroid = centers.mean(axis=0)
    centroid_inside = bool(
        np.all(centroid >= fov_low - tolerance_mm)
        and np.all(centroid <= fov_high + tolerance_mm)
    )
    margins = np.concatenate((roi_low - fov_low, fov_high - roi_high))
    return ROIFOVAudit(
        source_positive_voxels=int(len(coordinates)),
        source_volume_mm3=source_volume,
        intersection_volume_mm3=intersection_volume,
        occupancy=occupancy,
        boundary_touch_x=touches[0],
        boundary_touch_y=touches[1],
        boundary_touch_z=touches[2],
        boundary_touch_any=bool(any(touches)),
        centroid_x_ras_mm=float(centroid[0]),
        centroid_y_ras_mm=float(centroid[1]),
        centroid_z_ras_mm=float(centroid[2]),
        centroid_inside=centroid_inside,
        physical_margin_mm=float(np.min(margins)),
    )


def summarize_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "n": 0,
            "minimum": None,
            "q01": None,
            "q05": None,
            "q50": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "maximum": None,
            "mean": None,
        }
    quantiles = np.quantile(finite, [0.01, 0.05, 0.5, 0.9, 0.95, 0.99])
    return {
        "n": int(len(finite)),
        "minimum": float(finite.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
    }


def roi_audit_to_dict(audit: ROIFOVAudit) -> dict:
    return asdict(audit)


def decide_scientific_classification(
    *,
    source_roi_available: bool,
    local_gate_pass: bool | None,
    c1b_gate_pass: bool | None,
    acquisition_source_complete: bool | None,
) -> tuple[str, str]:
    """Apply the preregistered, fail-closed classification priority."""

    if not source_roi_available:
        return "D", CLASSIFICATIONS["D"]
    if local_gate_pass is None:
        raise ValueError("LOCAL gate cannot be unknown when source ROI is available")
    if local_gate_pass:
        return "C", CLASSIFICATIONS["C"]
    if c1b_gate_pass is None:
        raise ValueError("C1B gate cannot be unknown when LOCAL fails")
    if c1b_gate_pass:
        return "A", CLASSIFICATIONS["A"]
    if acquisition_source_complete is None:
        raise ValueError("acquisition completeness cannot be unknown when C1B fails")
    if acquisition_source_complete:
        return "B", CLASSIFICATIONS["B"]
    return "E", CLASSIFICATIONS["E"]
