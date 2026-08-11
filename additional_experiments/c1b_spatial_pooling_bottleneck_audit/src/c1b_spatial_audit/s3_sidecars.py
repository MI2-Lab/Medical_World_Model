"""Authoritative, trigger-gated geometry sidecars for the conditional S3 audit.

S3 weights are reconstructed directly from the same hash-bound source geometry,
C1B caches, and source lesion masks used by the primary audit.  They are never
resized or inferred from the final-stage sidecar.  Legacy oracle support remains
absent by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from c1b_sanity.geometry import (
    audit_support_containment,
    load_nifti_ras,
    resample_support_nearest,
)

from .contracts import TIMEPOINTS, file_sha256
from .pooling import (
    expected_feature_shape,
    fixed_physical_local_weights,
    receptive_field_occupancy,
)
from .s3_trigger import require_s3_trigger_authorization
from .sidecars import (
    C1B_INPUT_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    LEGACY_INPUT_SHAPE_ZYX,
    LEGACY_PORACLE_STATUS,
    _as_bool,
    _cache_arrays,
    _parse_geometry,
    _require_columns,
    _validate_cache_file,
    _validate_cache_identity,
    _visit_index,
    locked_population,
    validate_reconstructed_support,
    verify_source_locks,
)


C1B_S3_SHAPE_ZYX = (28, 44, 40)
LEGACY_S3_SHAPE_ZYX = (8, 24, 24)
FORMAL_PATIENT_COUNT = 808
FORMAL_ORACLE_VISIT_COUNT = 1500
S3_SIDECAR_FILENAME = "audit_sidecars_s3.private.npz"
S3_SIDECAR_METADATA_FILENAME = "audit_sidecars_s3.private.metadata.json"
S3_SIDECAR_KEYS = (
    "patient_id",
    "c1b_oracle_weight_s3",
    "c1b_oracle_valid",
    "c1b_local_weight_s3",
    "legacy_local_weight_s3",
)
S3_SIDECAR_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "stage",
        "sidecar_path",
        "sidecar_sha256",
        "sidecar_keys",
        "patient_count",
        "oracle_valid_visit_count",
        "c1b_feature_shape_zyx",
        "legacy_feature_shape_zyx",
        "legacy_poracle",
        "trigger_gate_path",
        "trigger_gate_sha256",
        "trigger_status",
        "source_sha256",
        "implementation_sha256",
        "derived_from_final_stage_sidecar",
        "legacy_oracle_fabricated",
        "new_training_performed",
        "patient_identifiers_private",
    }
)


@dataclass(frozen=True)
class S3AuditSidecars:
    patient_id: np.ndarray
    c1b_oracle_weight_s3: np.ndarray
    c1b_oracle_valid: np.ndarray
    c1b_local_weight_s3: np.ndarray
    legacy_local_weight_s3: np.ndarray

    def npz_arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in S3_SIDECAR_KEYS}


@dataclass(frozen=True)
class LoadedS3Sidecars:
    path: Path
    metadata_path: Path
    sha256: str
    metadata_sha256: str
    patient_id: tuple[str, ...]
    c1b_oracle_weight_s3: np.ndarray
    c1b_oracle_valid: np.ndarray
    c1b_local_weight_s3: np.ndarray
    legacy_local_weight_s3: np.ndarray


def _formal_support_sources(
    support_inventory: str | Path,
    support_reference: str | Path,
    patients: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    source = pd.read_csv(support_inventory)
    _require_columns(
        source,
        ("patient_id", "visit", "formal_ftv_overlap", "ftv_mask_nifti"),
        label="support source inventory",
    )
    source["patient_id"] = source["patient_id"].astype(str)
    source["visit"] = source["visit"].astype(str)
    source["_formal"] = [
        _as_bool(value, label="formal_ftv_overlap")
        for value in source["formal_ftv_overlap"]
    ]
    selected = source.loc[
        source["_formal"] & source["patient_id"].isin(patients)
    ].copy()
    if selected.duplicated(["patient_id", "visit"]).any():
        raise ValueError("formal support inventory duplicates a patient/visit")
    formal_ids = tuple(sorted(selected["patient_id"].unique()))
    expected = {
        (patient_id, visit) for patient_id in formal_ids for visit in TIMEPOINTS
    }
    observed = set(zip(selected["patient_id"], selected["visit"], strict=True))
    if len(formal_ids) != 375 or len(selected) != FORMAL_ORACLE_VISIT_COUNT:
        raise ValueError("formal support inventory is not the frozen 375/1500 cohort")
    if observed != expected:
        raise ValueError("formal support inventory is incomplete over T0-T3")

    reference = pd.read_csv(support_reference)
    _require_columns(
        reference,
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
    reference["patient_id"] = reference["patient_id"].astype(str)
    reference["visit"] = reference["visit"].astype(str)
    reference = reference.loc[reference["patient_id"].isin(formal_ids)].copy()
    if (
        len(reference) != FORMAL_ORACLE_VISIT_COUNT
        or set(reference["strategy"].astype(str)) != {"C1B-H"}
        or reference.duplicated(["patient_id", "visit"]).any()
    ):
        raise ValueError("immutable C1B-H support reference is incomplete")
    return (
        selected.set_index(["patient_id", "visit"], verify_integrity=True),
        reference.set_index(["patient_id", "visit"], verify_integrity=True),
        formal_ids,
    )


def build_s3_audit_sidecars(
    *,
    trigger_gate: str | Path,
    preregistration_lock: str | Path,
    stage_a_go: str | Path,
    data_contract: str | Path,
    cache_manifest: str | Path,
    geometry_inventory: str | Path,
    support_inventory: str | Path,
    support_reference: str | Path,
    verify_cache_archive_sha256: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> S3AuditSidecars:
    """Reconstruct all S3 weights from authoritative sources after a weak gate."""

    # This is deliberately the first data-bearing action.
    require_s3_trigger_authorization(trigger_gate, verify_live=True)
    if expected_feature_shape(C1B_INPUT_SHAPE_ZYX, stage="s3") != C1B_S3_SHAPE_ZYX:
        raise AssertionError("C1B S3 convolution geometry drifted")
    if expected_feature_shape(LEGACY_INPUT_SHAPE_ZYX, stage="s3") != LEGACY_S3_SHAPE_ZYX:
        raise AssertionError("legacy S3 convolution geometry drifted")
    verify_source_locks(
        stage_a_go=stage_a_go,
        data_contract=data_contract,
        cache_manifest=cache_manifest,
        geometry_inventory=geometry_inventory,
        support_inventory=support_inventory,
        support_reference=support_reference,
    )
    patients = locked_population(preregistration_lock)
    patient_ids = tuple(str(value) for value in patients)
    if len(patient_ids) != FORMAL_PATIENT_COUNT:
        raise ValueError("S3 sidecar population is not the frozen 808 patients")

    geometry_frame = pd.read_csv(geometry_inventory)
    _require_columns(
        geometry_frame,
        (
            "patient_id",
            "visit",
            "source_shape_xyz_json",
            "source_affine_ras_json",
        ),
        label="outcome-free geometry inventory",
    )
    geometry = _visit_index(
        geometry_frame, patient_ids, label="outcome-free geometry inventory"
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
        raise ValueError("C1B cache manifest duplicates a patient")
    cache_index = cache_frame.set_index("patient_id", verify_integrity=True)
    if not set(patient_ids).issubset(cache_index.index):
        raise ValueError("C1B cache manifest does not cover the frozen population")

    source_index, reference_index, formal_ids = _formal_support_sources(
        support_inventory, support_reference, patient_ids
    )
    patient_lookup = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    oracle_weights = np.zeros(
        (FORMAL_PATIENT_COUNT, len(TIMEPOINTS), *C1B_S3_SHAPE_ZYX),
        dtype=np.float32,
    )
    oracle_valid = np.zeros((FORMAL_PATIENT_COUNT, len(TIMEPOINTS)), dtype=bool)

    for completed, patient_id in enumerate(formal_ids, start=1):
        cache_path = _validate_cache_file(
            cache_index.loc[patient_id],
            verify_archive_sha256=verify_cache_archive_sha256,
        )
        cache = _cache_arrays(cache_path)
        grid = _validate_cache_identity(cache, patient_id)
        if not bool(np.asarray(cache["formal_ftv_overlap"]).item()):
            raise ValueError("formal support patient is not marked formal in C1B cache")
        patient_index = patient_lookup[patient_id]
        for visit_index, visit in enumerate(TIMEPOINTS):
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
            weight = receptive_field_occupancy(
                torch.from_numpy(sampled[None, None]).to(torch.float32),
                C1B_S3_SHAPE_ZYX,
                stage="s3",
            )
            oracle_weights[patient_index, visit_index] = weight[0, 0].cpu().numpy()
            oracle_valid[patient_index, visit_index] = True
        if progress is not None:
            progress(completed, len(formal_ids))

    if int(oracle_valid.sum()) != FORMAL_ORACLE_VISIT_COUNT:
        raise ValueError("S3 oracle validity is not the exact formal 1500 visits")
    expected_formal = np.isin(patients, np.asarray(formal_ids))
    if not np.array_equal(oracle_valid.all(axis=1), expected_formal):
        raise ValueError("S3 oracle patient availability differs from the formal cohort")
    if np.any(oracle_weights[~oracle_valid] != 0):
        raise AssertionError("non-formal S3 oracle weights must remain explicit zero")

    c1b_local = fixed_physical_local_weights(
        C1B_INPUT_SHAPE_ZYX,
        C1B_S3_SHAPE_ZYX,
        C1B_SPACING_XYZ_MM,
        stage="s3",
        dtype=torch.float32,
    )[0, 0].cpu().numpy()
    legacy_spacings = np.asarray(
        [
            _parse_geometry(geometry.loc[(patient_id, visit)])[1]
            for patient_id in patient_ids
            for visit in TIMEPOINTS
        ],
        dtype=np.float64,
    )
    if legacy_spacings.shape != (FORMAL_PATIENT_COUNT * len(TIMEPOINTS), 3):
        raise AssertionError("legacy S3 spacing assembly shape drifted")
    legacy_local = fixed_physical_local_weights(
        LEGACY_INPUT_SHAPE_ZYX,
        LEGACY_S3_SHAPE_ZYX,
        legacy_spacings,
        stage="s3",
        dtype=torch.float32,
    )[:, 0].cpu().numpy()
    legacy_local = legacy_local.reshape(
        FORMAL_PATIENT_COUNT, len(TIMEPOINTS), *LEGACY_S3_SHAPE_ZYX
    )
    bundle = S3AuditSidecars(
        patient_id=np.asarray(patient_ids),
        c1b_oracle_weight_s3=np.ascontiguousarray(oracle_weights),
        c1b_oracle_valid=np.ascontiguousarray(oracle_valid),
        c1b_local_weight_s3=np.ascontiguousarray(c1b_local, dtype=np.float32),
        legacy_local_weight_s3=np.ascontiguousarray(legacy_local, dtype=np.float32),
    )
    validate_s3_bundle(bundle)
    return bundle


def _validate_weight(
    array: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    require_support: np.ndarray | bool,
) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.float32 or value.shape != shape:
        raise ValueError(f"{name} must be float32 {shape}; got {value.dtype}/{value.shape}")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError(f"{name} must contain finite [0,1] values")
    if value.ndim == 3:
        support = np.asarray(value.sum(dtype=np.float64) > 0)
    else:
        support = value.reshape(*value.shape[:2], -1).sum(axis=-1, dtype=np.float64) > 0
    required = np.broadcast_to(np.asarray(require_support, dtype=bool), support.shape)
    if not bool(np.all(support[required])) or bool(np.any(support[~required])):
        raise ValueError(f"{name} support does not match declared validity")
    return value


def validate_s3_bundle(bundle: S3AuditSidecars) -> None:
    patient_ids = np.asarray(bundle.patient_id).astype(str)
    if (
        patient_ids.shape != (FORMAL_PATIENT_COUNT,)
        or len(set(patient_ids)) != FORMAL_PATIENT_COUNT
        or np.any(patient_ids == "")
    ):
        raise ValueError("S3 sidecar requires 808 unique nonempty patient identities")
    valid = np.asarray(bundle.c1b_oracle_valid)
    if valid.dtype != np.bool_ or valid.shape != (FORMAL_PATIENT_COUNT, len(TIMEPOINTS)):
        raise ValueError("c1b_oracle_valid must be bool [808,4]")
    if int(valid.sum()) != FORMAL_ORACLE_VISIT_COUNT:
        raise ValueError("S3 oracle validity must contain exactly 1500 visits")
    _validate_weight(
        bundle.c1b_oracle_weight_s3,
        name="c1b_oracle_weight_s3",
        shape=(FORMAL_PATIENT_COUNT, len(TIMEPOINTS), *C1B_S3_SHAPE_ZYX),
        require_support=valid,
    )
    _validate_weight(
        bundle.c1b_local_weight_s3,
        name="c1b_local_weight_s3",
        shape=C1B_S3_SHAPE_ZYX,
        require_support=True,
    )
    _validate_weight(
        bundle.legacy_local_weight_s3,
        name="legacy_local_weight_s3",
        shape=(FORMAL_PATIENT_COUNT, len(TIMEPOINTS), *LEGACY_S3_SHAPE_ZYX),
        require_support=True,
    )


def _private_path(path: str | Path, *, filename: str) -> Path:
    source = Path(path).resolve()
    if source.name != filename:
        raise ValueError(f"S3 private artifact must be named {filename}")
    return source


def _temporary(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    path = Path(name)
    path.chmod(0o600)
    return path


def write_s3_sidecars(
    bundle: S3AuditSidecars,
    *,
    sidecar_output: str | Path,
    metadata_output: str | Path,
    trigger_gate: str | Path,
    source_paths: Mapping[str, str | Path],
) -> tuple[Path, Path]:
    """Atomically publish a new S3 NPZ plus its owner-only provenance record."""

    validate_s3_bundle(bundle)
    # Revalidate the trigger immediately before publication.
    gate = require_s3_trigger_authorization(trigger_gate, verify_live=True)
    sidecar = _private_path(sidecar_output, filename=S3_SIDECAR_FILENAME)
    metadata = _private_path(metadata_output, filename=S3_SIDECAR_METADATA_FILENAME)
    if sidecar == metadata or sidecar.exists() or metadata.exists():
        raise FileExistsError("refusing to overwrite conditional S3 sidecars")
    trigger_path = Path(trigger_gate).resolve()
    source_hashes = {
        str(name): file_sha256(Path(path).resolve())
        for name, path in sorted(source_paths.items())
    }
    npz_temp = _temporary(sidecar, ".npz")
    metadata_temp = _temporary(metadata, ".json")
    published: list[Path] = []
    try:
        np.savez_compressed(npz_temp, **bundle.npz_arrays())
        npz_temp.chmod(0o600)
        payload = {
            "schema_version": 1,
            "status": "COMPLETE",
            "stage": "s3",
            "sidecar_path": str(sidecar),
            "sidecar_sha256": file_sha256(npz_temp),
            "sidecar_keys": list(S3_SIDECAR_KEYS),
            "patient_count": FORMAL_PATIENT_COUNT,
            "oracle_valid_visit_count": FORMAL_ORACLE_VISIT_COUNT,
            "c1b_feature_shape_zyx": list(C1B_S3_SHAPE_ZYX),
            "legacy_feature_shape_zyx": list(LEGACY_S3_SHAPE_ZYX),
            "legacy_poracle": LEGACY_PORACLE_STATUS,
            "trigger_gate_path": str(trigger_path),
            "trigger_gate_sha256": file_sha256(trigger_path),
            "trigger_status": gate["status"],
            "source_sha256": source_hashes,
            "implementation_sha256": {
                "s3_sidecars.py": file_sha256(Path(__file__)),
                "sidecars.py": file_sha256(Path(__file__).with_name("sidecars.py")),
                "pooling.py": file_sha256(Path(__file__).with_name("pooling.py")),
                "s3_trigger.py": file_sha256(Path(__file__).with_name("s3_trigger.py")),
            },
            "derived_from_final_stage_sidecar": False,
            "legacy_oracle_fabricated": False,
            "new_training_performed": False,
            "patient_identifiers_private": True,
        }
        metadata_temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metadata_temp.chmod(0o600)
        for path in (npz_temp, metadata_temp):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        npz_temp.replace(sidecar)
        published.append(sidecar)
        metadata_temp.replace(metadata)
        published.append(metadata)
        for path in published:
            path.chmod(0o600)
        return sidecar, metadata
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        npz_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)


def _metadata_for_sidecar(path: Path) -> Path:
    if path.name != S3_SIDECAR_FILENAME:
        raise ValueError(f"S3 sidecar must be named {S3_SIDECAR_FILENAME}")
    return path.with_name(S3_SIDECAR_METADATA_FILENAME)


def load_s3_sidecars(
    path: str | Path,
    expected_patient_ids: Iterable[str],
    *,
    verify_live: bool = True,
) -> LoadedS3Sidecars:
    """Load, reindex, and provenance-check the immutable S3 sidecar."""

    source = Path(path).resolve()
    metadata_path = _metadata_for_sidecar(source)
    if not source.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("conditional S3 sidecar or metadata is absent")
    if source.stat().st_mode & 0o077 or metadata_path.stat().st_mode & 0o077:
        raise PermissionError("S3 sidecar and metadata must be owner-only")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != set(S3_SIDECAR_KEYS):
            raise ValueError("S3 sidecar NPZ schema drifted")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    stored_ids = tuple(np.asarray(arrays["patient_id"]).astype(str).tolist())
    expected_ids = tuple(str(value) for value in expected_patient_ids)
    if (
        len(expected_ids) != FORMAL_PATIENT_COUNT
        or len(set(expected_ids)) != FORMAL_PATIENT_COUNT
        or set(stored_ids) != set(expected_ids)
    ):
        raise ValueError("S3 sidecar patient set differs from the formal fold population")
    lookup = {patient_id: index for index, patient_id in enumerate(stored_ids)}
    reorder = np.asarray([lookup[patient_id] for patient_id in expected_ids], dtype=np.int64)
    for name in ("c1b_oracle_weight_s3", "c1b_oracle_valid", "legacy_local_weight_s3"):
        arrays[name] = arrays[name][reorder]
    bundle = S3AuditSidecars(
        patient_id=np.asarray(expected_ids),
        c1b_oracle_weight_s3=arrays["c1b_oracle_weight_s3"],
        c1b_oracle_valid=arrays["c1b_oracle_valid"],
        c1b_local_weight_s3=arrays["c1b_local_weight_s3"],
        legacy_local_weight_s3=arrays["legacy_local_weight_s3"],
    )
    validate_s3_bundle(bundle)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("S3 sidecar metadata is not an object")
    if set(metadata) != S3_SIDECAR_METADATA_FIELDS:
        raise ValueError("S3 sidecar metadata schema drifted")
    identity = {
        "schema_version": 1,
        "status": "COMPLETE",
        "stage": "s3",
        "sidecar_path": str(source),
        "sidecar_sha256": file_sha256(source),
        "sidecar_keys": list(S3_SIDECAR_KEYS),
        "patient_count": FORMAL_PATIENT_COUNT,
        "oracle_valid_visit_count": FORMAL_ORACLE_VISIT_COUNT,
        "c1b_feature_shape_zyx": list(C1B_S3_SHAPE_ZYX),
        "legacy_feature_shape_zyx": list(LEGACY_S3_SHAPE_ZYX),
        "derived_from_final_stage_sidecar": False,
        "legacy_oracle_fabricated": False,
        "new_training_performed": False,
        "patient_identifiers_private": True,
    }
    for field, expected in identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f"S3 sidecar metadata drifted at {field}")
    if metadata.get("legacy_poracle") != LEGACY_PORACLE_STATUS:
        raise ValueError("S3 metadata fabricated or changed legacy oracle availability")
    trigger_path = Path(str(metadata.get("trigger_gate_path", ""))).resolve()
    if not trigger_path.is_file() or metadata.get("trigger_gate_sha256") != file_sha256(
        trigger_path
    ):
        raise ValueError("S3 sidecar trigger binding drifted")
    if verify_live:
        trigger_gate = require_s3_trigger_authorization(trigger_path, verify_live=True)
        if metadata.get("trigger_status") != trigger_gate["status"]:
            raise ValueError("S3 sidecar trigger status differs from authorization gate")
        implementation = metadata.get("implementation_sha256")
        expected_implementation = {
            "s3_sidecars.py": Path(__file__),
            "sidecars.py": Path(__file__).with_name("sidecars.py"),
            "pooling.py": Path(__file__).with_name("pooling.py"),
            "s3_trigger.py": Path(__file__).with_name("s3_trigger.py"),
        }
        if not isinstance(implementation, Mapping) or set(implementation) != set(
            expected_implementation
        ):
            raise ValueError("S3 sidecar implementation hash map drifted")
        for name, implementation_path in expected_implementation.items():
            if implementation.get(name) != file_sha256(implementation_path):
                raise ValueError(f"S3 sidecar implementation changed at {name}")
        source_hashes = metadata.get("source_sha256")
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise ValueError("S3 sidecar source hash inventory is absent")
    return LoadedS3Sidecars(
        path=source,
        metadata_path=metadata_path,
        sha256=file_sha256(source),
        metadata_sha256=file_sha256(metadata_path),
        patient_id=expected_ids,
        c1b_oracle_weight_s3=np.asarray(bundle.c1b_oracle_weight_s3),
        c1b_oracle_valid=np.asarray(bundle.c1b_oracle_valid),
        c1b_local_weight_s3=np.asarray(bundle.c1b_local_weight_s3),
        legacy_local_weight_s3=np.asarray(bundle.legacy_local_weight_s3),
    )


__all__ = [
    "C1B_S3_SHAPE_ZYX",
    "FORMAL_ORACLE_VISIT_COUNT",
    "FORMAL_PATIENT_COUNT",
    "LEGACY_S3_SHAPE_ZYX",
    "LoadedS3Sidecars",
    "S3AuditSidecars",
    "S3_SIDECAR_FILENAME",
    "S3_SIDECAR_KEYS",
    "S3_SIDECAR_METADATA_FIELDS",
    "S3_SIDECAR_METADATA_FILENAME",
    "build_s3_audit_sidecars",
    "load_s3_sidecars",
    "validate_s3_bundle",
    "write_s3_sidecars",
]
