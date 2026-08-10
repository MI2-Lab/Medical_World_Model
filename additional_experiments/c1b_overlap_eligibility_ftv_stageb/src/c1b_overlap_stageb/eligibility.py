"""Pure geometry implementation of four-visit valid-source eligibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import nibabel as nib
from nibabel.orientations import (
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)
import numpy as np
import pandas as pd


VISITS: tuple[str, ...] = ("T0", "T1", "T2", "T3")


def _validate_affine(value: np.ndarray, *, name: str) -> np.ndarray:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name} must be a finite 4x4 affine")
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must be homogeneous")
    singular_values = np.linalg.svd(affine[:3, :3], compute_uv=False)
    if singular_values[-1] <= np.finfo(np.float64).eps * max(singular_values[0], 1.0) * 16:
        raise ValueError(f"{name} is singular")
    return affine


def canonical_header_geometry(path: str | Path) -> tuple[tuple[int, int, int], np.ndarray]:
    """Return true-RAS spatial shape/affine without reading image intensities."""

    image = nib.load(str(Path(path)), mmap=True)
    source_affine = _validate_affine(image.affine, name="source affine")
    spatial_shape = tuple(int(value) for value in image.shape[:3])
    if len(spatial_shape) != 3 or any(value < 1 for value in spatial_shape):
        raise ValueError("Source image has an invalid spatial shape")
    source_orientation = io_orientation(source_affine)
    if source_orientation.shape != (3, 2) or np.isnan(source_orientation).any():
        raise ValueError("Source affine does not define three orientations")
    transform = ornt_transform(source_orientation, axcodes2ornt(("R", "A", "S")))
    permutation = np.argsort(transform[:, 0].astype(int))
    canonical_shape = tuple(spatial_shape[int(index)] for index in permutation)
    canonical_affine = _validate_affine(
        source_affine @ inv_ornt_aff(transform, spatial_shape),
        name="canonical RAS affine",
    )
    if tuple(nib.aff2axcodes(canonical_affine)) != ("R", "A", "S"):
        raise ValueError("Canonical header is not RAS+")
    return canonical_shape, canonical_affine


def count_valid_source_voxels(
    input_from_output: np.ndarray,
    output_shape_xyz: Sequence[int],
    source_shape_xyz: Sequence[int],
    *,
    x_slab: int = 16,
) -> int:
    """Count target centres inside source voxel footprints exactly.

    This is the builder's frozen rule: a target centre is valid iff its mapped
    source coordinate is within ``[-0.5, source_length - 0.5]`` on every axis.
    Slabbing only bounds memory and does not change the result.
    """

    mapping = _validate_affine(input_from_output, name="input-from-output mapping")
    output_shape = tuple(int(value) for value in output_shape_xyz)
    source_shape = tuple(int(value) for value in source_shape_xyz)
    if len(output_shape) != 3 or any(value < 1 for value in output_shape):
        raise ValueError("output_shape_xyz must contain three positive values")
    if len(source_shape) != 3 or any(value < 1 for value in source_shape):
        raise ValueError("source_shape_xyz must contain three positive values")
    slab = int(x_slab)
    if slab < 1:
        raise ValueError("x_slab must be positive")

    y = np.arange(output_shape[1], dtype=np.float64)[None, :, None]
    z = np.arange(output_shape[2], dtype=np.float64)[None, None, :]
    total = 0
    for first in range(0, output_shape[0], slab):
        x = np.arange(first, min(first + slab, output_shape[0]), dtype=np.float64)[
            :, None, None
        ]
        valid = np.ones((len(x), output_shape[1], output_shape[2]), dtype=bool)
        for source_axis, source_length in enumerate(source_shape):
            coordinate = (
                mapping[source_axis, 0] * x
                + mapping[source_axis, 1] * y
                + mapping[source_axis, 2] * z
                + mapping[source_axis, 3]
            )
            valid &= (coordinate >= -0.5) & (
                coordinate <= float(source_length) - 0.5
            )
        total += int(np.count_nonzero(valid))
    return total


def geometry_contract_sha256(
    *,
    source_shape_xyz: Sequence[int],
    source_affine_ras: np.ndarray,
    grid_shape_xyz: Sequence[int],
    grid_affine_ras: np.ndarray,
) -> str:
    payload = {
        "grid_affine_ras": np.asarray(grid_affine_ras, dtype=np.float64).tolist(),
        "grid_shape_xyz": [int(value) for value in grid_shape_xyz],
        "source_affine_ras": np.asarray(source_affine_ras, dtype=np.float64).tolist(),
        "source_shape_xyz": [int(value) for value in source_shape_xyz],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"c1b-four-visit-overlap-v1\0" + encoded).hexdigest()


def frozen_grid_contract_sha256(
    *,
    patient_id: str,
    cohort: str,
    grid_shape_zyx: Sequence[int],
    grid_spacing_xyz_mm: Sequence[float],
    grid_affine_ras: np.ndarray,
) -> str:
    """Fingerprint one private, already-frozen patient grid."""

    payload = {
        "cohort": str(cohort),
        "grid_affine_ras": np.asarray(grid_affine_ras, dtype=np.float64).tolist(),
        "grid_shape_zyx": [int(value) for value in grid_shape_zyx],
        "grid_spacing_xyz_mm": [float(value) for value in grid_spacing_xyz_mm],
        "patient_id": str(patient_id),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"frozen-c1b-h-grid-v1\0" + encoded).hexdigest()


def build_patient_eligibility(visits: pd.DataFrame) -> pd.DataFrame:
    """Apply the preregistered four-visit AND without patient-specific logic."""

    required = {
        "patient_id",
        "cohort",
        "visit",
        "valid_source_voxels",
        "target_grid_voxels",
    }
    if not required.issubset(visits.columns):
        raise ValueError(f"Visit eligibility table lacks {sorted(required - set(visits.columns))}")
    frame = visits.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["visit"] = frame["visit"].astype(str)
    if frame.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Visit eligibility table has duplicate patient/visit rows")
    if set(frame["visit"]) != set(VISITS):
        raise ValueError("Visit eligibility table contains a noncanonical visit set")
    if (pd.to_numeric(frame["valid_source_voxels"], errors="raise") < 0).any():
        raise ValueError("valid_source_voxels cannot be negative")
    if (pd.to_numeric(frame["target_grid_voxels"], errors="raise") < 1).any():
        raise ValueError("target_grid_voxels must be positive")

    records: list[dict[str, object]] = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        if len(group) != len(VISITS) or set(group["visit"]) != set(VISITS):
            raise ValueError("Every candidate patient must have exactly T0-T3")
        cohorts = set(group["cohort"].astype(str))
        if len(cohorts) != 1:
            raise ValueError("A patient's cohort changes across visits")
        counts = pd.to_numeric(group["valid_source_voxels"], errors="raise").astype(np.int64)
        zero_count = int(counts.eq(0).sum())
        eligible = zero_count == 0
        records.append(
            {
                "patient_id": str(patient_id),
                "cohort": next(iter(cohorts)),
                "candidate_visit_count": int(len(group)),
                "valid_visit_count": int(counts.gt(0).sum()),
                "zero_overlap_visit_count": zero_count,
                "minimum_valid_source_voxels": int(counts.min()),
                "eligible": eligible,
                "exclusion_reason": "" if eligible else "ZERO_VALID_SOURCE_OVERLAP_IN_REQUIRED_VISIT",
            }
        )
    return pd.DataFrame(records).sort_values("patient_id", kind="stable").reset_index(drop=True)
