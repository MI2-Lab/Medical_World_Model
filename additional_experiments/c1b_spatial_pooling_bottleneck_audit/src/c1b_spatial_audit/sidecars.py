"""Build outcome-blind geometry sidecars for the frozen spatial audit.

The C1B valid-source mask is read from the immutable cache.  Oracle support is
reconstructed only from the hash-bound source mask and is checked against every
stored count and physical-volume summary before it can become a pooling weight.
Legacy PVALID/PORACLE are intentionally absent: the historical cache has no
source-authoritative valid mask and incomplete source-authoritative support.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from c1b_sanity.geometry import (
    PhysicalGrid,
    audit_support_containment,
    canonical_volume_sha256,
    load_nifti_ras,
    resample_support_nearest,
)

from .contracts import (
    REPO_ROOT,
    TIMEPOINTS,
    UPSTREAM_COMPLETION_SHA256,
    cell_key,
    cells,
    file_sha256,
    relative,
)
from .pooling import (
    expected_feature_shape,
    fixed_physical_local_weights,
    receptive_field_occupancy,
)


C1B_INPUT_SHAPE_ZYX = (112, 176, 160)
C1B_SPACING_XYZ_MM = (0.9, 0.9, 2.0)
C1B_FINAL_SHAPE_ZYX = (14, 22, 20)
LEGACY_INPUT_SHAPE_ZYX = (32, 96, 96)
LEGACY_FINAL_SHAPE_ZYX = (4, 12, 12)

LEGACY_PVALID_STATUS = "NA_no_source_authoritative_mask"
LEGACY_PORACLE_STATUS = "NA_incomplete_source_authoritative_support_1488_of_1500"

SIDECAR_KEYS = (
    "patient_id",
    "c1b_valid_weight_final",
    "c1b_oracle_weight_final",
    "c1b_oracle_valid",
    "c1b_local_weight_final",
    "legacy_local_weight_final",
)
NUISANCE_COLUMNS = (
    "patient_id",
    "visit",
    "padding_fraction",
    "valid_source_fraction",
    "native_spacing_x_mm",
    "native_spacing_y_mm",
    "native_spacing_z_mm",
    "acquisition_fov_x_mm",
    "acquisition_fov_y_mm",
    "acquisition_fov_z_mm",
    "max_resample_factor",
    "resize_anisotropy",
)
OCCUPANCY_COLUMNS = (
    "patient_id",
    "visit",
    "support_source_positive_voxels",
    "support_retained_positive_voxels",
    "support_nn_target_positive_voxels",
    "support_source_volume_mm3",
    "support_retained_source_volume_mm3",
    "valid_source_voxels",
    "valid_source_volume_mm3",
    "lesion_occupancy",
    "occupancy_quartile",
)

_CACHE_MEMBERS = (
    "patient_id",
    "visits",
    "formal_ftv_overlap",
    "registration_strategy",
    "grid_affine_ras",
    "grid_center_ras_mm",
    "grid_shape_zyx",
    "grid_spacing_xyz_mm",
    "source_to_anchor_ras",
    "source_samples_per_output_axis",
    "valid_source_mask",
    "support_available",
    "support_canonical_sha256",
    "support_source_positive_voxels",
    "support_retained_positive_voxels",
    "support_nn_target_positive_voxels",
    "support_retained_positive_voxel_fraction",
    "support_physical_volume_retention",
    "support_exact_full_support_containment",
    "support_source_boundary_touch",
    "support_target_boundary_touch",
    "support_minimum_margin_mm",
    "support_source_volume_mm3",
    "support_retained_source_volume_mm3",
)


@dataclass(frozen=True)
class AuditSidecars:
    """In-memory private sidecars, ready for one atomic publication step."""

    patient_id: np.ndarray
    c1b_valid_weight_final: np.ndarray
    c1b_oracle_weight_final: np.ndarray
    c1b_oracle_valid: np.ndarray
    c1b_local_weight_final: np.ndarray
    legacy_local_weight_final: np.ndarray
    nuisance: pd.DataFrame
    occupancy: pd.DataFrame

    def npz_arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in SIDECAR_KEYS}


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], *, label: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_bool(value: object, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{label} is not a strict boolean: {value!r}")


def _ordered_patient_sha256(patient_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(patient_ids).encode("utf-8")).hexdigest()


def locked_population(preregistration_lock: str | Path) -> np.ndarray:
    """Return the sorted 808-patient set after validating all 40 P0 assets."""

    lock = json.loads(Path(preregistration_lock).read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_BEFORE_NEW_FEATURE_OR_PROBE":
        raise ValueError("preregistration lock is not in the frozen state")
    references = lock.get("formal_p0_references")
    if not isinstance(references, Mapping):
        raise ValueError("preregistration lock has no formal P0 inventory")
    expected_keys = {cell_key(seed, arm, fold) for seed, arm, fold in cells()}
    if set(references) != expected_keys or len(references) != 40:
        raise ValueError("formal P0 inventory is not the exact frozen 40-cell matrix")

    common_set: set[str] | None = None
    for key in sorted(expected_keys):
        record = references[key]
        if not isinstance(record, Mapping):
            raise ValueError(f"malformed P0 inventory record at {key}")
        feature = REPO_ROOT / str(record.get("feature_path", ""))
        expected_sha = str(record.get("feature_sha256", ""))
        if not feature.is_file() or file_sha256(feature) != expected_sha:
            raise ValueError(f"frozen P0 feature drift at {key}")
        with np.load(feature, allow_pickle=False) as archive:
            if "patient_id" not in archive or "response_state" not in archive:
                raise ValueError(f"frozen P0 feature schema drift at {key}")
            patient_ids = [str(value) for value in archive["patient_id"]]
            response_shape = tuple(archive["response_state"].shape)
        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError(f"duplicate patient identity in frozen P0 feature at {key}")
        if response_shape != (len(patient_ids), len(TIMEPOINTS), 192):
            raise ValueError(f"frozen P0 feature shape drift at {key}")
        if _ordered_patient_sha256(patient_ids) != record.get("patient_order_sha256"):
            raise ValueError(f"frozen P0 patient order hash drift at {key}")
        observed_set = set(patient_ids)
        if common_set is None:
            common_set = observed_set
        elif observed_set != common_set:
            raise ValueError("the 40 frozen P0 assets do not share one patient set")

    if common_set is None or len(common_set) != 808:
        raise ValueError("frozen formal population is not exactly 808 unique patients")
    return np.asarray(sorted(common_set))


def verify_source_locks(
    *,
    stage_a_go: str | Path,
    data_contract: str | Path,
    cache_manifest: str | Path,
    geometry_inventory: str | Path,
    support_inventory: str | Path,
    support_reference: str | Path,
) -> None:
    """Verify that every private source is bound by an immutable upstream gate."""

    go_path = Path(stage_a_go).resolve()
    go = json.loads(go_path.read_text(encoding="utf-8"))
    if go.get("status") != "GO" or go.get("stage_b_authorized") is not True:
        raise ValueError("upstream Stage A is not an authorized immutable GO")
    provenance = go.get("provenance_sha256")
    if not isinstance(provenance, Mapping):
        raise ValueError("upstream Stage A has no provenance map")
    for path in (geometry_inventory, support_inventory, support_reference):
        source = Path(path).resolve()
        key = relative(source)
        if provenance.get(key) != file_sha256(source):
            raise ValueError(f"upstream Stage A provenance drift at {key}")

    contract_path = Path(data_contract).resolve()
    expected_contract_sha = UPSTREAM_COMPLETION_SHA256[
        "manifests/stage_b_data_contract.private.json"
    ]
    if file_sha256(contract_path) != expected_contract_sha:
        raise ValueError("frozen Stage B data contract has drifted")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    cache_path = Path(cache_manifest).resolve()
    if Path(str(contract.get("c1b_cache_manifest", ""))).resolve() != cache_path:
        raise ValueError("C1B cache manifest path disagrees with the data contract")
    if contract.get("c1b_cache_manifest_sha256") != file_sha256(cache_path):
        raise ValueError("C1B cache manifest hash disagrees with the data contract")


def _visit_index(
    frame: pd.DataFrame,
    patient_ids: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    selected = frame.loc[frame["patient_id"].astype(str).isin(patient_ids)].copy()
    selected["patient_id"] = selected["patient_id"].astype(str)
    selected["visit"] = selected["visit"].astype(str)
    if selected.duplicated(["patient_id", "visit"]).any():
        raise ValueError(f"{label} has duplicate patient/visit rows")
    expected = {(patient, visit) for patient in patient_ids for visit in TIMEPOINTS}
    observed = set(zip(selected["patient_id"], selected["visit"]))
    if observed != expected:
        raise ValueError(f"{label} does not cover the exact four-visit population")
    return selected.set_index(["patient_id", "visit"], verify_integrity=True)


def _parse_geometry(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    try:
        shape = np.asarray(json.loads(str(row["source_shape_xyz_json"])), dtype=np.int64)
        affine = np.asarray(json.loads(str(row["source_affine_ras_json"])), dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid source geometry JSON") from exc
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("source shape must be three positive XYZ values")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("source affine must be finite 4x4")
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0):
        raise ValueError("source affine is not homogeneous")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("source affine has invalid voxel spacing")
    return shape, spacing


def nuisance_row(
    patient_id: str,
    visit: str,
    geometry_row: pd.Series,
    valid_source_voxels: int,
    target_grid_voxels: int,
    source_samples_per_output_axis: Sequence[float],
) -> dict[str, object]:
    """Derive the preregistered nuisance allow-list without outcome access."""

    valid = int(valid_source_voxels)
    target = int(target_grid_voxels)
    if target <= 0 or valid <= 0 or valid > target:
        raise ValueError("valid-source counts violate the frozen eligibility contract")
    shape, spacing = _parse_geometry(geometry_row)
    factors = np.asarray(source_samples_per_output_axis, dtype=np.float64)
    if factors.shape != (3,) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("source resampling factors must be three finite positive values")
    valid_fraction = float(valid / target)
    fov = shape.astype(np.float64) * spacing
    return {
        "patient_id": str(patient_id),
        "visit": str(visit),
        "padding_fraction": float(1.0 - valid_fraction),
        "valid_source_fraction": valid_fraction,
        "native_spacing_x_mm": float(spacing[0]),
        "native_spacing_y_mm": float(spacing[1]),
        "native_spacing_z_mm": float(spacing[2]),
        "acquisition_fov_x_mm": float(fov[0]),
        "acquisition_fov_y_mm": float(fov[1]),
        "acquisition_fov_z_mm": float(fov[2]),
        "max_resample_factor": float(factors.max()),
        "resize_anisotropy": float(factors.max() / factors.min()),
    }


def _serialized_float_equal(value: float, observed: object) -> bool:
    array = np.asarray(observed)
    if array.shape != () or array.dtype.kind != "f":
        return False
    return bool(np.asarray(value, dtype=array.dtype) == array)


def _close_reference(value: float, observed: object) -> bool:
    return bool(np.isclose(value, float(observed), rtol=1e-12, atol=1e-9))


def _assert_equal(observed: object, expected: object, *, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def validate_valid_source_mask(
    mask: np.ndarray,
    expected_counts: Sequence[int],
    expected_target_voxels: Sequence[int],
) -> np.ndarray:
    """Validate the source-authoritative C1B mask and return exact visit counts."""

    array = np.asarray(mask)
    expected_shape = (len(TIMEPOINTS), 1, *C1B_INPUT_SHAPE_ZYX)
    if array.shape != expected_shape:
        raise ValueError(f"C1B valid-source mask shape drift: {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("C1B valid-source mask must be an integer binary array")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("C1B valid-source mask is not binary")
    counts = array.reshape(len(TIMEPOINTS), -1).sum(axis=1, dtype=np.int64)
    expected = np.asarray(expected_counts, dtype=np.int64)
    targets = np.asarray(expected_target_voxels, dtype=np.int64)
    if expected.shape != (len(TIMEPOINTS),) or targets.shape != (len(TIMEPOINTS),):
        raise ValueError("valid-source count vectors must have four visits")
    if not np.array_equal(counts, expected):
        raise ValueError("cache valid-source mask count disagrees with eligibility")
    if np.any(targets != int(np.prod(C1B_INPUT_SHAPE_ZYX))):
        raise ValueError("eligibility target-grid voxel count drift")
    if np.any(counts <= 0) or np.any(counts > targets):
        raise ValueError("C1B valid-source mask has invalid support")
    return counts


def _cache_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(_CACHE_MEMBERS) - set(archive.files))
        if missing:
            raise ValueError(f"C1B cache is missing audit sidecars: {missing}")
        return {name: np.asarray(archive[name]).copy() for name in _CACHE_MEMBERS}


def _validate_cache_identity(
    arrays: Mapping[str, np.ndarray], patient_id: str
) -> PhysicalGrid:
    _assert_equal(str(arrays["patient_id"].item()), patient_id, label="cache patient")
    _assert_equal(
        tuple(str(value) for value in arrays["visits"]),
        tuple(TIMEPOINTS),
        label="cache visit order",
    )
    _assert_equal(
        str(arrays["registration_strategy"].item()), "C1B-H", label="cache strategy"
    )
    shape = tuple(int(value) for value in arrays["grid_shape_zyx"])
    spacing = tuple(float(value) for value in arrays["grid_spacing_xyz_mm"])
    if shape != C1B_INPUT_SHAPE_ZYX or spacing != C1B_SPACING_XYZ_MM:
        raise ValueError("cache is not on the frozen C1B grid")
    grid = PhysicalGrid(
        shape_zyx=shape,
        spacing_xyz_mm=spacing,
        center_ras_mm=tuple(float(value) for value in arrays["grid_center_ras_mm"]),
    )
    if not np.array_equal(grid.affine_ras, arrays["grid_affine_ras"]):
        raise ValueError("cache grid affine disagrees with its center/shape/spacing")
    transforms = np.asarray(arrays["source_to_anchor_ras"], dtype=np.float64)
    if transforms.shape != (len(TIMEPOINTS), 4, 4) or not np.allclose(
        transforms,
        np.broadcast_to(np.eye(4), transforms.shape),
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("frozen C1B-H cache contains a non-identity transform")
    return grid


def validate_reconstructed_support(
    *,
    support: object,
    sampled_support_zyx: np.ndarray,
    audit: object,
    cache: Mapping[str, np.ndarray],
    visit_index: int,
    reference_row: pd.Series,
) -> None:
    """Fail closed unless source, cache, and prior physical audit all agree."""

    index = int(visit_index)
    if not bool(np.asarray(cache["support_available"])[index]):
        raise ValueError("formal C1B support is marked unavailable")
    observed_hash = canonical_volume_sha256(support)  # type: ignore[arg-type]
    _assert_equal(
        observed_hash,
        str(np.asarray(cache["support_canonical_sha256"])[index]),
        label="canonical support hash",
    )
    exact_integer_checks = {
        "source positive count": (
            int(audit.full_positive_voxels),  # type: ignore[attr-defined]
            int(np.asarray(cache["support_source_positive_voxels"])[index]),
        ),
        "retained positive count": (
            int(audit.retained_positive_voxels),  # type: ignore[attr-defined]
            int(np.asarray(cache["support_retained_positive_voxels"])[index]),
        ),
        "NN target positive count": (
            int(np.count_nonzero(sampled_support_zyx)),
            int(np.asarray(cache["support_nn_target_positive_voxels"])[index]),
        ),
    }
    for label, (observed, expected) in exact_integer_checks.items():
        _assert_equal(observed, expected, label=label)

    cache_float_checks = {
        "retained positive fraction": (
            audit.retained_positive_voxel_fraction,  # type: ignore[attr-defined]
            np.asarray(cache["support_retained_positive_voxel_fraction"])[index],
        ),
        "physical retention": (
            audit.physical_volume_retention,  # type: ignore[attr-defined]
            np.asarray(cache["support_physical_volume_retention"])[index],
        ),
        "source physical volume": (
            audit.full_physical_volume_mm3,  # type: ignore[attr-defined]
            np.asarray(cache["support_source_volume_mm3"])[index],
        ),
        "retained physical volume": (
            audit.retained_physical_volume_mm3,  # type: ignore[attr-defined]
            np.asarray(cache["support_retained_source_volume_mm3"])[index],
        ),
        "minimum physical margin": (
            audit.minimum_margin_mm,  # type: ignore[attr-defined]
            np.asarray(cache["support_minimum_margin_mm"])[index],
        ),
    }
    for label, (value, serialized) in cache_float_checks.items():
        if not _serialized_float_equal(float(value), serialized):
            raise ValueError(f"{label} disagrees with serialized cache sidecar")

    cache_bool_checks = {
        "exact containment": (
            audit.exact_full_support_containment,  # type: ignore[attr-defined]
            np.asarray(cache["support_exact_full_support_containment"])[index],
        ),
        "source boundary touch": (
            audit.source_boundary_touch,  # type: ignore[attr-defined]
            np.asarray(cache["support_source_boundary_touch"])[index],
        ),
        "target boundary touch": (
            audit.target_boundary_touch,  # type: ignore[attr-defined]
            np.asarray(cache["support_target_boundary_touch"])[index],
        ),
    }
    for label, (value, serialized) in cache_bool_checks.items():
        _assert_equal(bool(value), bool(serialized), label=label)

    reference_checks = {
        "prior source count": (
            int(audit.full_positive_voxels),  # type: ignore[attr-defined]
            int(reference_row["source_positive_voxels"]),
            False,
        ),
        "prior retained count": (
            int(audit.retained_positive_voxels),  # type: ignore[attr-defined]
            int(reference_row["retained_positive_voxels"]),
            False,
        ),
        "prior source volume": (
            float(audit.full_physical_volume_mm3),  # type: ignore[attr-defined]
            reference_row["full_physical_volume_mm3"],
            True,
        ),
        "prior retained volume": (
            float(audit.retained_physical_volume_mm3),  # type: ignore[attr-defined]
            reference_row["retained_physical_volume_mm3"],
            True,
        ),
        "prior physical retention": (
            float(audit.physical_volume_retention),  # type: ignore[attr-defined]
            reference_row["physical_volume_retention"],
            True,
        ),
        "prior minimum margin": (
            float(audit.minimum_margin_mm),  # type: ignore[attr-defined]
            reference_row["minimum_margin_mm"],
            True,
        ),
    }
    for label, (value, expected, floating) in reference_checks.items():
        matches = _close_reference(float(value), expected) if floating else value == expected
        if not matches:
            raise ValueError(f"{label} disagrees with immutable containment table")
    for field, value in (
        ("exact_full_support_containment", audit.exact_full_support_containment),  # type: ignore[attr-defined]
        ("source_boundary_touch", audit.source_boundary_touch),  # type: ignore[attr-defined]
        ("target_boundary_touch", audit.target_boundary_touch),  # type: ignore[attr-defined]
    ):
        if bool(value) != _as_bool(reference_row[field], label=field):
            raise ValueError(f"prior {field} disagrees with reconstruction")


def assign_occupancy_quartiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen pooled qcut rule, stopping on non-unique boundaries."""

    values = frame["lesion_occupancy"].to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("lesion occupancy must be finite, positive, and nonempty")
    boundaries = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 1.0))
    if len(np.unique(boundaries)) != 5:
        raise ValueError("occupancy qcut boundaries are not unique; frozen rule STOP")
    result = frame.copy()
    result["occupancy_quartile"] = pd.qcut(
        result["lesion_occupancy"], q=4, labels=("Q1", "Q2", "Q3", "Q4")
    ).astype(str)
    if result["occupancy_quartile"].isna().any():
        raise ValueError("occupancy qcut produced missing quartiles")
    return result


def _validate_cache_file(
    row: pd.Series, *, verify_archive_sha256: bool
) -> Path:
    if str(row["input_kind"]) != "c1b":
        raise ValueError("cache manifest contains a non-C1B row")
    path = Path(str(row["cache_path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if stat.st_size != int(row["cache_size_bytes"]):
        raise ValueError("C1B cache size drift")
    if stat.st_mtime_ns != int(row["cache_mtime_ns"]):
        raise ValueError("C1B cache mtime drift")
    if verify_archive_sha256 and file_sha256(path) != str(row["cache_sha256"]):
        raise ValueError("C1B cache archive SHA-256 drift")
    return path


def build_audit_sidecars(
    *,
    preregistration_lock: str | Path,
    stage_a_go: str | Path,
    data_contract: str | Path,
    cache_manifest: str | Path,
    geometry_inventory: str | Path,
    support_inventory: str | Path,
    support_reference: str | Path,
    verify_cache_archive_sha256: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> AuditSidecars:
    """Build the exact private NPZ/CSV payloads without reading outcomes."""

    if expected_feature_shape(C1B_INPUT_SHAPE_ZYX) != C1B_FINAL_SHAPE_ZYX:
        raise AssertionError("C1B final feature geometry drift")
    if expected_feature_shape(LEGACY_INPUT_SHAPE_ZYX) != LEGACY_FINAL_SHAPE_ZYX:
        raise AssertionError("legacy final feature geometry drift")
    verify_source_locks(
        stage_a_go=stage_a_go,
        data_contract=data_contract,
        cache_manifest=cache_manifest,
        geometry_inventory=geometry_inventory,
        support_inventory=support_inventory,
        support_reference=support_reference,
    )
    patients = locked_population(preregistration_lock)
    patient_list = [str(value) for value in patients]

    geometry_frame = pd.read_csv(geometry_inventory)
    _require_columns(
        geometry_frame,
        (
            "patient_id",
            "visit",
            "source_shape_xyz_json",
            "source_affine_ras_json",
            "valid_source_voxels",
            "target_grid_voxels",
            "valid_source_fraction",
        ),
        label="outcome-free geometry inventory",
    )
    geometry = _visit_index(
        geometry_frame, patient_list, label="outcome-free geometry inventory"
    )

    cache_frame = pd.read_csv(cache_manifest)
    _require_columns(
        cache_frame,
        (
            "patient_id",
            "cache_path",
            "cache_sha256",
            "cache_size_bytes",
            "cache_mtime_ns",
            "input_kind",
        ),
        label="C1B cache manifest",
    )
    cache_frame["patient_id"] = cache_frame["patient_id"].astype(str)
    if cache_frame["patient_id"].duplicated().any():
        raise ValueError("C1B cache manifest has duplicate patients")
    cache_index = cache_frame.set_index("patient_id", verify_integrity=True)
    if not set(patient_list).issubset(cache_index.index):
        raise ValueError("C1B cache manifest does not cover the frozen population")

    source_frame = pd.read_csv(support_inventory)
    _require_columns(
        source_frame,
        ("patient_id", "visit", "formal_ftv_overlap", "ftv_mask_nifti"),
        label="support source inventory",
    )
    source_frame["patient_id"] = source_frame["patient_id"].astype(str)
    source_frame["visit"] = source_frame["visit"].astype(str)
    source_frame["_formal"] = [
        _as_bool(value, label="formal_ftv_overlap")
        for value in source_frame["formal_ftv_overlap"]
    ]
    formal_sources = source_frame.loc[
        source_frame["_formal"] & source_frame["patient_id"].isin(patient_list)
    ].copy()
    if formal_sources.duplicated(["patient_id", "visit"]).any():
        raise ValueError("formal support inventory has duplicate patient/visit rows")
    formal_ids = sorted(formal_sources["patient_id"].unique())
    if len(formal_ids) != 375 or len(formal_sources) != 1500:
        raise ValueError("formal support population is not the frozen 375/1500 cohort")
    expected_formal_keys = {
        (patient, visit) for patient in formal_ids for visit in TIMEPOINTS
    }
    observed_formal_keys = set(
        zip(formal_sources["patient_id"], formal_sources["visit"])
    )
    if observed_formal_keys != expected_formal_keys:
        raise ValueError("formal support inventory is not complete over T0-T3")
    source_index = formal_sources.set_index(
        ["patient_id", "visit"], verify_integrity=True
    )

    reference_frame = pd.read_csv(support_reference)
    _require_columns(
        reference_frame,
        (
            "patient_id",
            "visit",
            "strategy",
            "source_positive_voxels",
            "retained_positive_voxels",
            "full_physical_volume_mm3",
            "retained_physical_volume_mm3",
            "physical_volume_retention",
            "exact_full_support_containment",
            "source_boundary_touch",
            "target_boundary_touch",
            "minimum_margin_mm",
        ),
        label="immutable support containment reference",
    )
    reference_frame["patient_id"] = reference_frame["patient_id"].astype(str)
    reference_frame["visit"] = reference_frame["visit"].astype(str)
    reference_frame = reference_frame.loc[
        reference_frame["patient_id"].isin(formal_ids)
    ].copy()
    if len(reference_frame) != 1500 or set(reference_frame["strategy"]) != {"C1B-H"}:
        raise ValueError("immutable support reference is not the frozen C1B-H cohort")
    reference_index = reference_frame.set_index(
        ["patient_id", "visit"], verify_integrity=True
    )

    patient_count = len(patient_list)
    valid_weights = np.empty(
        (patient_count, len(TIMEPOINTS), *C1B_FINAL_SHAPE_ZYX), dtype=np.float32
    )
    oracle_weights = np.zeros_like(valid_weights)
    oracle_valid = np.zeros((patient_count, len(TIMEPOINTS)), dtype=bool)
    nuisance_rows: list[dict[str, object]] = []
    occupancy_rows: list[dict[str, object]] = []
    legacy_spacings: list[np.ndarray] = []

    for patient_index, patient_id in enumerate(patient_list):
        cache_path = _validate_cache_file(
            cache_index.loc[patient_id],
            verify_archive_sha256=verify_cache_archive_sha256,
        )
        cache = _cache_arrays(cache_path)
        grid = _validate_cache_identity(cache, patient_id)
        formal = patient_id in set(formal_ids)
        _assert_equal(
            bool(cache["formal_ftv_overlap"].item()), formal, label="formal support scope"
        )

        patient_geometry = [geometry.loc[(patient_id, visit)] for visit in TIMEPOINTS]
        expected_counts = [int(row["valid_source_voxels"]) for row in patient_geometry]
        target_counts = [int(row["target_grid_voxels"]) for row in patient_geometry]
        observed_counts = validate_valid_source_mask(
            cache["valid_source_mask"], expected_counts, target_counts
        )
        mask_tensor = torch.from_numpy(cache["valid_source_mask"]).to(torch.float32)
        valid_weight = receptive_field_occupancy(
            mask_tensor, C1B_FINAL_SHAPE_ZYX, stage="final"
        )
        valid_weights[patient_index] = valid_weight[:, 0].cpu().numpy()

        factors = np.asarray(cache["source_samples_per_output_axis"], dtype=np.float64)
        if factors.shape != (len(TIMEPOINTS), 3):
            raise ValueError("cache source resampling factor shape drift")
        patient_spacings: list[np.ndarray] = []
        for visit_index, (visit, row) in enumerate(zip(TIMEPOINTS, patient_geometry)):
            nuisance = nuisance_row(
                patient_id,
                visit,
                row,
                int(observed_counts[visit_index]),
                int(target_counts[visit_index]),
                factors[visit_index],
            )
            manifest_fraction = float(row["valid_source_fraction"])
            if not np.isclose(
                nuisance["valid_source_fraction"], manifest_fraction, rtol=0, atol=5e-13
            ):
                raise ValueError("valid-source fraction disagrees with eligibility")
            nuisance_rows.append(nuisance)
            _, spacing = _parse_geometry(row)
            patient_spacings.append(spacing)

            if not formal:
                continue
            source_row = source_index.loc[(patient_id, visit)]
            support_path = Path(str(source_row["ftv_mask_nifti"])).resolve()
            if not support_path.is_file():
                raise FileNotFoundError(support_path)
            support = load_nifti_ras(support_path)
            transform = np.asarray(cache["source_to_anchor_ras"])[visit_index]
            sampled = resample_support_nearest(
                support, grid, source_to_anchor_ras=transform
            )
            audit = audit_support_containment(
                support, grid, source_to_anchor_ras=transform
            )
            validate_reconstructed_support(
                support=support,
                sampled_support_zyx=sampled,
                audit=audit,
                cache=cache,
                visit_index=visit_index,
                reference_row=reference_index.loc[(patient_id, visit)],
            )
            oracle_weight = receptive_field_occupancy(
                torch.from_numpy(sampled[None, None]).to(torch.float32),
                C1B_FINAL_SHAPE_ZYX,
                stage="final",
            )
            oracle_weights[patient_index, visit_index] = oracle_weight[0, 0].numpy()
            oracle_valid[patient_index, visit_index] = True

            grid_voxel_volume = float(abs(np.linalg.det(grid.affine_ras[:3, :3])))
            valid_volume = float(observed_counts[visit_index] * grid_voxel_volume)
            # The preregistered occupancy numerator is the cache sidecar value.
            # It is used only after the float32 serialization has been proved to
            # equal the independently reconstructed physical audit above.
            source_volume = float(
                np.asarray(cache["support_source_volume_mm3"])[visit_index]
            )
            retained_volume = float(
                np.asarray(cache["support_retained_source_volume_mm3"])[visit_index]
            )
            occupancy = retained_volume / valid_volume
            if not np.isfinite(occupancy) or occupancy <= 0 or occupancy > 1:
                raise ValueError("lesion occupancy escaped the physical [0,1] contract")
            occupancy_rows.append(
                {
                    "patient_id": patient_id,
                    "visit": visit,
                    "support_source_positive_voxels": int(
                        audit.full_positive_voxels
                    ),
                    "support_retained_positive_voxels": int(
                        audit.retained_positive_voxels
                    ),
                    "support_nn_target_positive_voxels": int(
                        np.count_nonzero(sampled)
                    ),
                    "support_source_volume_mm3": source_volume,
                    "support_retained_source_volume_mm3": retained_volume,
                    "valid_source_voxels": int(observed_counts[visit_index]),
                    "valid_source_volume_mm3": valid_volume,
                    "lesion_occupancy": float(occupancy),
                    "occupancy_quartile": "PENDING_QCUT",
                }
            )
        legacy_spacings.extend(patient_spacings)
        if progress is not None:
            progress(patient_index + 1, patient_count)

    if int(oracle_valid.sum()) != 1500 or not np.array_equal(
        oracle_valid.any(axis=1), np.isin(patients, np.asarray(formal_ids))
    ):
        raise ValueError("oracle availability is not exactly the frozen formal cohort")
    if np.any(oracle_weights[~oracle_valid] != 0):
        raise AssertionError("non-formal oracle weights must remain absent/zero")

    c1b_local = fixed_physical_local_weights(
        C1B_INPUT_SHAPE_ZYX,
        C1B_FINAL_SHAPE_ZYX,
        C1B_SPACING_XYZ_MM,
        stage="final",
        dtype=torch.float32,
    )[0, 0].cpu().numpy()
    spacing_array = np.asarray(legacy_spacings, dtype=np.float64)
    if spacing_array.shape != (patient_count * len(TIMEPOINTS), 3):
        raise AssertionError("legacy spacing assembly shape drift")
    legacy_local = fixed_physical_local_weights(
        LEGACY_INPUT_SHAPE_ZYX,
        LEGACY_FINAL_SHAPE_ZYX,
        spacing_array,
        stage="final",
        dtype=torch.float32,
    )[:, 0].cpu().numpy()
    legacy_local = legacy_local.reshape(
        patient_count, len(TIMEPOINTS), *LEGACY_FINAL_SHAPE_ZYX
    )

    nuisance_frame = pd.DataFrame(nuisance_rows, columns=NUISANCE_COLUMNS)
    if len(nuisance_frame) != patient_count * len(TIMEPOINTS):
        raise AssertionError("nuisance sidecar row count drift")
    occupancy_frame = assign_occupancy_quartiles(
        pd.DataFrame(occupancy_rows, columns=OCCUPANCY_COLUMNS)
    )
    occupancy_frame = occupancy_frame.loc[:, OCCUPANCY_COLUMNS]
    if len(occupancy_frame) != 1500:
        raise AssertionError("occupancy sidecar row count drift")

    bundle = AuditSidecars(
        patient_id=patients,
        c1b_valid_weight_final=np.ascontiguousarray(valid_weights),
        c1b_oracle_weight_final=np.ascontiguousarray(oracle_weights),
        c1b_oracle_valid=np.ascontiguousarray(oracle_valid),
        c1b_local_weight_final=np.ascontiguousarray(c1b_local, dtype=np.float32),
        legacy_local_weight_final=np.ascontiguousarray(legacy_local, dtype=np.float32),
        nuisance=nuisance_frame,
        occupancy=occupancy_frame,
    )
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: AuditSidecars) -> None:
    patient_count = len(bundle.patient_id)
    if patient_count <= 0 or len(set(bundle.patient_id.astype(str))) != patient_count:
        raise ValueError("sidecar patient identities must be unique and nonempty")
    expected_shapes = {
        "c1b_valid_weight_final": (
            patient_count,
            len(TIMEPOINTS),
            *C1B_FINAL_SHAPE_ZYX,
        ),
        "c1b_oracle_weight_final": (
            patient_count,
            len(TIMEPOINTS),
            *C1B_FINAL_SHAPE_ZYX,
        ),
        "c1b_oracle_valid": (patient_count, len(TIMEPOINTS)),
        "c1b_local_weight_final": C1B_FINAL_SHAPE_ZYX,
        "legacy_local_weight_final": (
            patient_count,
            len(TIMEPOINTS),
            *LEGACY_FINAL_SHAPE_ZYX,
        ),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(getattr(bundle, name))
        if array.shape != shape:
            raise ValueError(f"{name} shape drift: expected {shape}, got {array.shape}")
        if name != "c1b_oracle_valid":
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite float32")
            if np.any(array < 0) or np.any(array > 1):
                raise ValueError(f"{name} escaped [0,1]")
    if np.asarray(bundle.c1b_oracle_valid).dtype != np.bool_:
        raise ValueError("c1b_oracle_valid must be boolean")
    if tuple(bundle.nuisance.columns) != NUISANCE_COLUMNS:
        raise ValueError("nuisance sidecar schema drift")
    if tuple(bundle.occupancy.columns) != OCCUPANCY_COLUMNS:
        raise ValueError("occupancy sidecar schema drift")


def _private_path(path: str | Path, suffix: str) -> Path:
    output = Path(path).resolve()
    if not output.name.endswith(suffix):
        raise ValueError(f"private output must end in {suffix}: {output.name}")
    return output


def require_outputs_absent(
    sidecar_output: str | Path,
    nuisance_output: str | Path,
    occupancy_output: str | Path,
) -> tuple[Path, Path, Path]:
    outputs = (
        _private_path(sidecar_output, ".private.npz"),
        _private_path(nuisance_output, ".private.csv"),
        _private_path(occupancy_output, ".private.csv"),
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("private output paths must be distinct")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite private sidecars: {existing}")
    return outputs


def _temporary_path(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    path = Path(name)
    path.chmod(0o600)
    return path


def write_private_sidecars(
    bundle: AuditSidecars,
    *,
    sidecar_output: str | Path,
    nuisance_output: str | Path,
    occupancy_output: str | Path,
) -> tuple[Path, Path, Path]:
    """Atomically publish exactly the three owner-only identifier-bearing files."""

    validate_bundle(bundle)
    outputs = require_outputs_absent(sidecar_output, nuisance_output, occupancy_output)
    temporaries: list[Path] = []
    published: list[Path] = []
    try:
        npz_temp = _temporary_path(outputs[0], ".npz")
        nuisance_temp = _temporary_path(outputs[1], ".csv")
        occupancy_temp = _temporary_path(outputs[2], ".csv")
        temporaries.extend((npz_temp, nuisance_temp, occupancy_temp))
        np.savez_compressed(npz_temp, **bundle.npz_arrays())
        bundle.nuisance.to_csv(nuisance_temp, index=False)
        bundle.occupancy.to_csv(occupancy_temp, index=False)
        for path in temporaries:
            path.chmod(0o600)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        for temporary, destination in zip(temporaries, outputs):
            temporary.replace(destination)
            destination.chmod(0o600)
            published.append(destination)
        return outputs
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporaries:
            path.unlink(missing_ok=True)


__all__ = [
    "AuditSidecars",
    "C1B_FINAL_SHAPE_ZYX",
    "C1B_INPUT_SHAPE_ZYX",
    "C1B_SPACING_XYZ_MM",
    "LEGACY_FINAL_SHAPE_ZYX",
    "LEGACY_INPUT_SHAPE_ZYX",
    "LEGACY_PORACLE_STATUS",
    "LEGACY_PVALID_STATUS",
    "NUISANCE_COLUMNS",
    "OCCUPANCY_COLUMNS",
    "SIDECAR_KEYS",
    "assign_occupancy_quartiles",
    "build_audit_sidecars",
    "locked_population",
    "nuisance_row",
    "require_outputs_absent",
    "validate_bundle",
    "validate_reconstructed_support",
    "validate_valid_source_mask",
    "verify_source_locks",
    "write_private_sidecars",
]
