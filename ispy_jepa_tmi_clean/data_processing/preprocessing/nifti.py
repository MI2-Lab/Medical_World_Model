from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np


def read_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a simple uncompressed NIfTI-1 file into a NumPy array.

    This intentionally mirrors the lightweight reader used during preprocessing and
    avoids an extra nibabel dependency.
    """
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(348)

    endian = "<"
    sizeof_hdr = struct.unpack("<i", header[:4])[0]
    if sizeof_hdr != 348:
        sizeof_hdr = struct.unpack(">i", header[:4])[0]
        endian = ">"
    if sizeof_hdr != 348:
        raise ValueError(f"{path} does not look like a NIfTI-1 file")

    dims = struct.unpack(endian + "8h", header[40:56])
    datatype = struct.unpack(endian + "h", header[70:72])[0]
    bitpix = struct.unpack(endian + "h", header[72:74])[0]
    pixdim = struct.unpack(endian + "8f", header[76:108])
    vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])

    dtype_by_code = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        512: np.uint16,
        768: np.uint32,
    }
    dtype = dtype_by_code.get(datatype)
    if dtype is None:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
    dtype = np.dtype(dtype).newbyteorder(endian)

    shape = tuple(int(x) for x in dims[1 : 1 + dims[0]])
    data = np.fromfile(path, dtype=dtype, offset=vox_offset).reshape(shape, order="F")
    meta = {
        "dim": [int(x) for x in dims],
        "shape": list(shape),
        "datatype": int(datatype),
        "bitpix": int(bitpix),
        "pixdim": [float(x) for x in pixdim],
        "vox_offset": int(vox_offset),
    }
    return data, meta


def needs_spatial_axis_canonicalization(
    shape: tuple[int, ...] | list[int],
    pixdim: tuple[float, ...] | list[float] | None,
) -> bool:
    """Detect old I-SPY1 NIfTIs stored as slice,row,col instead of row,col,slice."""
    if len(shape) < 3:
        return False
    spatial = tuple(int(v) for v in shape[:3])
    if pixdim is None or len(pixdim) < 4:
        return False
    px = [abs(float(v)) for v in pixdim[1:4]]
    if min(spatial) <= 0 or min(px) <= 0:
        return False
    return (
        spatial[0] < spatial[1]
        and spatial[0] < spatial[2]
        and spatial[1] >= 128
        and spatial[2] >= 128
        and px[0] > 1.5 * max(px[1], px[2])
    )


def _slice_first_spatial(meta: dict[str, Any], shape: tuple[int, ...]) -> bool:
    return needs_spatial_axis_canonicalization(shape, meta.get("pixdim", []))


def _canonical_meta(meta: dict[str, Any], original_shape: tuple[int, ...], canonical_shape: tuple[int, ...]) -> dict[str, Any]:
    out = dict(meta)
    out["original_shape"] = list(original_shape)
    out["shape"] = list(canonical_shape)
    pixdim = list(out.get("pixdim", []))
    if len(pixdim) >= 4:
        out["pixdim_original"] = list(pixdim)
        out["pixdim"] = [pixdim[0], pixdim[2], pixdim[3], pixdim[1], *pixdim[4:]]
    out["axis_canonicalization"] = "slice_first_to_xyz"
    return out


def canonicalize_dce_xyz_time(data: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return DCE as X,Y,Z,T.

    Most I-SPY2 NIfTIs are already X,Y,Z,T. Some I-SPY1 phase-stack conversions
    are slice,row,col,T, which breaks center crops by treating the slice axis as X.
    The pixdim/shape heuristic is intentionally conservative and leaves standard
    row,col,slice,T volumes untouched.
    """
    original_shape = tuple(int(v) for v in data.shape)
    if _slice_first_spatial(meta, original_shape):
        axes = (1, 2, 0) + tuple(range(3, data.ndim))
        canonical = np.transpose(data, axes)
        return canonical, _canonical_meta(meta, original_shape, tuple(int(v) for v in canonical.shape))
    out = dict(meta)
    out.setdefault("original_shape", list(original_shape))
    out.setdefault("axis_canonicalization", "none")
    return data, out


def canonicalize_spatial_xyz(data: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a spatial mask/volume as X,Y,Z using the same axis rule as DCE."""
    original_shape = tuple(int(v) for v in data.shape)
    if _slice_first_spatial(meta, original_shape):
        axes = (1, 2, 0) + tuple(range(3, data.ndim))
        canonical = np.transpose(data, axes)
        return canonical, _canonical_meta(meta, original_shape, tuple(int(v) for v in canonical.shape))
    out = dict(meta)
    out.setdefault("original_shape", list(original_shape))
    out.setdefault("axis_canonicalization", "none")
    return data, out


def read_dce_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    data, meta = read_nifti(path)
    return canonicalize_dce_xyz_time(data, meta)


def read_spatial_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    data, meta = read_nifti(path)
    return canonicalize_spatial_xyz(data, meta)
