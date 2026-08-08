"""Minimal NIfTI-1 header geometry reader with explicit affine validation.

The repository's historical reader intentionally ignores qform/sform.  Stage A
needs header geometry but should not add a heavyweight runtime dependency, so
this module parses the 348-byte NIfTI-1 header directly.  Coordinates are RAS+
world millimetres, matching the affine written by dcm2niix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import struct
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NiftiGeometry:
    path: Path
    shape: tuple[int, ...]
    spacing_xyz_mm: tuple[float, float, float]
    qform_code: int
    sform_code: int
    qform: np.ndarray | None
    sform: np.ndarray | None
    selected_affine: np.ndarray | None
    affine_source: str
    sform_valid: bool
    qform_valid: bool
    sform_failure_reason: str | None
    orientation: str
    max_obliquity_deg: float | None
    determinant: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "qform_code": self.qform_code,
            "sform_code": self.sform_code,
            "affine_source": self.affine_source,
            "sform_valid": self.sform_valid,
            "qform_valid": self.qform_valid,
            "sform_failure_reason": self.sform_failure_reason,
            "orientation": self.orientation,
            "max_obliquity_deg": self.max_obliquity_deg,
            "determinant": self.determinant,
        }


def _qform_affine(
    pixdim: np.ndarray,
    quaternion_bcd: tuple[float, float, float],
    offset_xyz: tuple[float, float, float],
) -> np.ndarray | None:
    b, c, d = (float(value) for value in quaternion_bcd)
    squared = b * b + c * c + d * d
    if not np.isfinite(squared):
        return None
    if squared > 1.0 + 1e-5:
        return None
    if squared > 1.0:
        scale = 1.0 / math.sqrt(squared)
        b, c, d = b * scale, c * scale, d * scale
        squared = 1.0
    a = math.sqrt(max(0.0, 1.0 - squared))
    rotation = np.asarray(
        [
            [
                a * a + b * b - c * c - d * d,
                2.0 * b * c - 2.0 * a * d,
                2.0 * b * d + 2.0 * a * c,
            ],
            [
                2.0 * b * c + 2.0 * a * d,
                a * a + c * c - b * b - d * d,
                2.0 * c * d - 2.0 * a * b,
            ],
            [
                2.0 * b * d - 2.0 * a * c,
                2.0 * c * d + 2.0 * a * b,
                a * a + d * d - c * c - b * b,
            ],
        ],
        dtype=np.float64,
    )
    zooms = np.abs(np.asarray(pixdim[1:4], dtype=np.float64))
    if zooms.shape != (3,) or not np.all(np.isfinite(zooms)) or np.any(zooms <= 0):
        return None
    qfac = -1.0 if float(pixdim[0]) < 0 else 1.0
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation @ np.diag([zooms[0], zooms[1], zooms[2] * qfac])
    affine[:3, 3] = np.asarray(offset_xyz, dtype=np.float64)
    return affine if np.all(np.isfinite(affine)) else None


def validate_sform(
    affine: np.ndarray | None,
    spacing_xyz_mm: tuple[float, float, float],
    *,
    rtol: float = 1e-4,
    atol_mm: float = 1e-4,
) -> tuple[bool, str | None]:
    if affine is None:
        return False, "SFORM_CODE_ZERO_OR_UNAVAILABLE"
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        return False, "SFORM_NONFINITE"
    linear = affine[:3, :3]
    if np.linalg.matrix_rank(linear) < 3:
        return False, "SFORM_SINGULAR"
    norms = np.linalg.norm(linear, axis=0)
    if not np.allclose(
        norms,
        np.asarray(spacing_xyz_mm, dtype=np.float64),
        rtol=rtol,
        atol=atol_mm,
    ):
        return False, "SFORM_SPACING_MISMATCH"
    directions = linear / norms[None, :]
    gram = directions.T @ directions
    if not np.allclose(gram, np.eye(3), rtol=1e-4, atol=1e-4):
        return False, "SFORM_NONORTHOGONAL"
    return True, None


def _axis_codes(affine: np.ndarray | None) -> str:
    if affine is None:
        return "UNKNOWN"
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    norms = np.linalg.norm(linear, axis=0)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        return "UNKNOWN"
    directions = linear / norms[None, :]
    world_axes = np.argmax(np.abs(directions), axis=0)
    if len(set(int(value) for value in world_axes)) != 3:
        return "UNKNOWN"
    labels = (("L", "R"), ("P", "A"), ("I", "S"))
    return "".join(
        labels[int(axis)][1 if directions[int(axis), index] >= 0 else 0]
        for index, axis in enumerate(world_axes)
    )


def _max_obliquity_deg(affine: np.ndarray | None) -> float | None:
    if affine is None:
        return None
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    norms = np.linalg.norm(linear, axis=0)
    if np.any(norms <= 0):
        return None
    directions = linear / norms[None, :]
    closest_cardinal_cosine = np.max(np.abs(directions), axis=0)
    closest_cardinal_cosine = np.clip(closest_cardinal_cosine, 0.0, 1.0)
    return float(np.degrees(np.arccos(closest_cardinal_cosine)).max())


def read_nifti_geometry(
    path: str | Path,
    *,
    sform_spacing_rtol: float = 1e-4,
    sform_spacing_atol_mm: float = 1e-4,
) -> NiftiGeometry:
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as stream:
        header = stream.read(348)
    if len(header) != 348:
        raise ValueError(f"NIfTI-1 header incomplete: {path}")
    endian = "<"
    if struct.unpack("<i", header[:4])[0] != 348:
        endian = ">"
    if struct.unpack(endian + "i", header[:4])[0] != 348:
        raise ValueError(f"Not a NIfTI-1 file: {path}")

    dims = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dims[0])
    if ndim < 3 or ndim > 7:
        raise ValueError(f"Unsupported NIfTI dimension {ndim}: {path}")
    shape = tuple(int(value) for value in dims[1 : ndim + 1])
    pixdim = np.asarray(struct.unpack(endian + "8f", header[76:108]), dtype=np.float64)
    spacing = tuple(float(abs(value)) for value in pixdim[1:4])
    if not np.all(np.isfinite(spacing)) or min(spacing) <= 0:
        raise ValueError(f"Invalid spatial spacing {spacing}: {path}")

    qform_code, sform_code = struct.unpack(endian + "2h", header[252:256])
    qform = None
    if qform_code > 0:
        qform = _qform_affine(
            pixdim,
            struct.unpack(endian + "3f", header[256:268]),
            struct.unpack(endian + "3f", header[268:280]),
        )
    sform = None
    if sform_code > 0:
        sform = np.eye(4, dtype=np.float64)
        sform[:3, :] = np.asarray(
            [
                struct.unpack(endian + "4f", header[280:296]),
                struct.unpack(endian + "4f", header[296:312]),
                struct.unpack(endian + "4f", header[312:328]),
            ],
            dtype=np.float64,
        )
    sform_valid, failure_reason = validate_sform(
        sform,
        spacing,
        rtol=sform_spacing_rtol,
        atol_mm=sform_spacing_atol_mm,
    )
    qform_valid = bool(
        qform is not None
        and np.linalg.matrix_rank(qform[:3, :3]) == 3
        and np.all(np.isfinite(qform))
    )
    if sform_valid:
        selected, source = sform, "sform"
    elif qform_valid:
        selected, source = qform, "qform_fallback"
    else:
        selected, source = None, "unavailable"
    determinant = (
        float(np.linalg.det(selected[:3, :3])) if selected is not None else None
    )
    return NiftiGeometry(
        path=path,
        shape=shape,
        spacing_xyz_mm=spacing,
        qform_code=int(qform_code),
        sform_code=int(sform_code),
        qform=qform,
        sform=sform,
        selected_affine=selected,
        affine_source=source,
        sform_valid=sform_valid,
        qform_valid=qform_valid,
        sform_failure_reason=failure_reason,
        orientation=_axis_codes(selected),
        max_obliquity_deg=_max_obliquity_deg(selected),
        determinant=determinant,
    )


def affine_max_corner_disagreement_mm(
    left: np.ndarray,
    right: np.ndarray,
    shape_xyz: tuple[int, int, int],
) -> float:
    """Maximum world-space disagreement across the eight voxel-edge corners."""

    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("affines must be 4x4")
    bounds = [(-0.5, float(length) - 0.5) for length in shape_xyz]
    corners = np.asarray(
        [
            (x, y, z, 1.0)
            for x in bounds[0]
            for y in bounds[1]
            for z in bounds[2]
        ],
        dtype=np.float64,
    )
    world_left = (left @ corners.T).T[:, :3]
    world_right = (right @ corners.T).T[:, :3]
    return float(np.linalg.norm(world_left - world_right, axis=1).max())


def read_nifti_array(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an uncompressed NIfTI-1 array without importing the torch package."""

    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as stream:
        header = stream.read(348)
    endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
    if struct.unpack(endian + "i", header[:4])[0] != 348:
        raise ValueError(f"Not a NIfTI-1 file: {path}")
    dims = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dims[0])
    shape = tuple(int(value) for value in dims[1 : ndim + 1])
    datatype = int(struct.unpack(endian + "h", header[70:72])[0])
    offset = int(struct.unpack(endian + "f", header[108:112])[0])
    dtypes = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        512: np.uint16,
        768: np.uint32,
    }
    if datatype not in dtypes:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}: {path}")
    dtype = np.dtype(dtypes[datatype]).newbyteorder(endian)
    data = np.fromfile(path, dtype=dtype, offset=offset).reshape(shape, order="F")
    pixdim = [float(value) for value in struct.unpack(endian + "8f", header[76:108])]
    return data, {
        "shape": list(shape),
        "datatype": datatype,
        "pixdim": pixdim,
        "vox_offset": offset,
    }


def read_spatial_nifti_array(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a spatial image in the repository's canonical index XYZ order."""

    data, meta = read_nifti_array(path)
    shape = tuple(int(value) for value in data.shape)
    spacing = [abs(float(value)) for value in meta["pixdim"][1:4]]
    slice_first = bool(
        len(shape) >= 3
        and min(shape[:3]) > 0
        and min(spacing) > 0
        and shape[0] < shape[1]
        and shape[0] < shape[2]
        and shape[1] >= 128
        and shape[2] >= 128
        and spacing[0] > 1.5 * max(spacing[1], spacing[2])
    )
    if not slice_first:
        return data, {**meta, "axis_canonicalization": "none"}
    axes = (1, 2, 0) + tuple(range(3, data.ndim))
    output = np.transpose(data, axes)
    pixdim = list(meta["pixdim"])
    pixdim = [pixdim[0], pixdim[2], pixdim[3], pixdim[1], *pixdim[4:]]
    return output, {
        **meta,
        "shape": list(output.shape),
        "pixdim": pixdim,
        "axis_canonicalization": "slice_first_to_xyz",
    }
