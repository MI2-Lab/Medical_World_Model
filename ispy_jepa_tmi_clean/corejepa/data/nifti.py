from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np


def read_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an uncompressed NIfTI-1 file without a nibabel dependency."""

    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(348)
    endian = "<"
    size = struct.unpack("<i", header[:4])[0]
    if size != 348:
        size = struct.unpack(">i", header[:4])[0]
        endian = ">"
    if size != 348:
        raise ValueError(f"Not a NIfTI-1 file: {path}")
    dims = struct.unpack(endian + "8h", header[40:56])
    datatype = struct.unpack(endian + "h", header[70:72])[0]
    bitpix = struct.unpack(endian + "h", header[72:74])[0]
    pixdim = struct.unpack(endian + "8f", header[76:108])
    offset = int(struct.unpack(endian + "f", header[108:112])[0])
    dtypes = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16, 768: np.uint32}
    if datatype not in dtypes:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}: {path}")
    shape = tuple(int(value) for value in dims[1 : 1 + dims[0]])
    dtype = np.dtype(dtypes[datatype]).newbyteorder(endian)
    data = np.fromfile(path, dtype=dtype, offset=offset).reshape(shape, order="F")
    return data, {
        "shape": list(shape),
        "datatype": int(datatype),
        "bitpix": int(bitpix),
        "pixdim": [float(value) for value in pixdim],
        "vox_offset": offset,
    }


def _slice_first(shape: tuple[int, ...], pixdim: list[float]) -> bool:
    if len(shape) < 3 or len(pixdim) < 4:
        return False
    spatial = shape[:3]
    spacing = [abs(float(value)) for value in pixdim[1:4]]
    return (
        min(spatial) > 0
        and min(spacing) > 0
        and spatial[0] < spatial[1]
        and spatial[0] < spatial[2]
        and spatial[1] >= 128
        and spatial[2] >= 128
        and spacing[0] > 1.5 * max(spacing[1], spacing[2])
    )


def _canonicalize(data: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    original_shape = tuple(int(value) for value in data.shape)
    if not _slice_first(original_shape, meta.get("pixdim", [])):
        return data, {**meta, "original_shape": list(original_shape), "axis_canonicalization": "none"}
    axes = (1, 2, 0) + tuple(range(3, data.ndim))
    output = np.transpose(data, axes)
    pixdim = list(meta.get("pixdim", []))
    if len(pixdim) >= 4:
        pixdim = [pixdim[0], pixdim[2], pixdim[3], pixdim[1], *pixdim[4:]]
    return output, {
        **meta,
        "original_shape": list(original_shape),
        "shape": list(output.shape),
        "pixdim": pixdim,
        "axis_canonicalization": "slice_first_to_xyz",
    }


def read_dce_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return DCE as ``float-compatible [X,Y,Z,T]``."""

    return _canonicalize(*read_nifti(path))


def read_spatial_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a spatial image as ``[X,Y,Z]``."""

    return _canonicalize(*read_nifti(path))
