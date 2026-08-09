"""Private source-resolution and raw-DICOM provenance helpers.

The functions in this module deliberately keep identifying values in private
data structures.  Public writers in :mod:`zero_overlap_audit.runner` consume
only the explicitly anonymized summaries returned alongside those structures.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pydicom


CASE_ALIAS = "CASE_ZERO_OVERLAP_001"
VISITS = ("T0", "T1", "T2", "T3")
SPATIAL_REGISTRATION_UIDS = {
    "1.2.840.10008.5.1.4.1.1.66.1",
    "1.2.840.10008.5.1.4.1.1.66.3",
}

PROVENANCE_TAGS = (
    "SOPClassUID",
    "SOPInstanceUID",
    "PatientPosition",
    "FrameOfReferenceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ImageType",
    "Modality",
    "BodyPartExamined",
    "Laterality",
    "ImageLaterality",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceLocation",
    "InstanceNumber",
    "TemporalPositionIdentifier",
    "AcquisitionNumber",
    "AcquisitionTime",
    "ContentTime",
    "TriggerTime",
    "NumberOfTemporalPositions",
    "MRAcquisitionType",
    "ScanningSequence",
    "SequenceVariant",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "Manufacturer",
    "ManufacturerModelName",
    "StationName",
    "MagneticFieldStrength",
    "ReceiveCoilName",
    "TransmitCoilName",
    "TableHeight",
    "TableTraverse",
    "TableMotion",
    "TableVerticalIncrement",
    "TableLongitudinalIncrement",
    "TableLateralIncrement",
    "TableAngle",
    "TableType",
    "TableSpeed",
    "ReconstructionDiameter",
    "ReconstructionMethod",
    "ReconstructionAlgorithm",
    "SeriesNumber",
)


class ProvenanceAuditError(RuntimeError):
    """A privacy-safe error containing a stable code but no source value."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def resolve_private_case(prior_root: Path) -> dict[str, Any]:
    """Resolve exactly one failure and its four visits without logging values."""

    failure_path = prior_root / "metrics/model_input_pipeline_h_validation_failures.private.csv"
    gate_path = prior_root / "metrics/model_input_pipeline_h_validation_gate.json"
    inventory_path = prior_root / "manifests/model_input_inventory.private.csv"
    eligibility_path = prior_root / "manifests/ispy1_base_eligibility_visits.private.csv"
    orientation_path = prior_root / "metrics/orientation_resampling_patient_visit.private.csv"
    for path in (
        failure_path,
        gate_path,
        inventory_path,
        eligibility_path,
        orientation_path,
    ):
        if not path.is_file():
            raise ProvenanceAuditError("REQUIRED_PRIVATE_INPUT_MISSING")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_hash = str(gate.get("private_evidence_sha256", {}).get("failure_table", ""))
    if not expected_hash or sha256_file(failure_path) != expected_hash:
        raise ProvenanceAuditError("PRIVATE_FAILURE_HASH_MISMATCH")
    failures = [
        row
        for row in _rows(failure_path)
        if row.get("failure_code") == "ZERO_VALID_SOURCE_OVERLAP"
    ]
    if len(failures) != 1:
        raise ProvenanceAuditError("ZERO_OVERLAP_CASE_NOT_UNIQUE")
    failure = failures[0]
    patient_id = str(failure["patient_id"])
    failed_visit = str(failure["visit"])
    if failed_visit not in VISITS:
        raise ProvenanceAuditError("FAILED_VISIT_INVALID")

    inventory = [
        row for row in _rows(inventory_path) if str(row.get("patient_id")) == patient_id
    ]
    eligibility = [
        row
        for row in _rows(eligibility_path)
        if str(row.get("patient_id")) == patient_id
    ]
    orientation = [
        row
        for row in _rows(orientation_path)
        if str(row.get("patient_id")) == patient_id
    ]
    for collection in (inventory, eligibility, orientation):
        if len(collection) != 4 or {row.get("visit") for row in collection} != set(VISITS):
            raise ProvenanceAuditError("FOUR_VISIT_JOIN_INCOMPLETE")
    if {str(row.get("cohort")) for row in inventory} != {"I-SPY1"}:
        raise ProvenanceAuditError("UNSUPPORTED_CASE_COHORT")
    if any(str(row.get("status")) != "PASS" for row in eligibility):
        raise ProvenanceAuditError("PRIOR_RAW_ELIGIBILITY_NOT_PASS")
    anchor_values = {str(row.get("anchor_provenance")) for row in orientation}
    if anchor_values != {"t0_acquisition_physical_center_fallback"}:
        raise ProvenanceAuditError("NON_SOURCE_ONLY_T0_ANCHOR")

    inventory_by_visit = {str(row["visit"]): row for row in inventory}
    eligibility_by_visit = {str(row["visit"]): row for row in eligibility}
    orientation_by_visit = {str(row["visit"]): row for row in orientation}
    visits: dict[str, dict[str, Any]] = {}
    for visit in VISITS:
        selected = json.loads(eligibility_by_visit[visit]["selected_source_series"])
        if not isinstance(selected, list) or not selected:
            raise ProvenanceAuditError("SELECTED_SOURCE_SERIES_INVALID")
        selected_paths = [str(Path(value).resolve()) for value in selected]
        if not all(Path(value).is_dir() for value in selected_paths):
            raise ProvenanceAuditError("RAW_SOURCE_SERIES_MISSING")
        study_dir = Path(eligibility_by_visit[visit]["study_dir"]).resolve()
        rebuilt_nifti = Path(eligibility_by_visit[visit]["rebuilt_nifti"]).resolve()
        if not study_dir.is_dir() or not rebuilt_nifti.is_file():
            raise ProvenanceAuditError("RAW_STUDY_OR_REBUILD_MISSING")
        visits[visit] = {
            "visit": visit,
            "study_dir": str(study_dir),
            "selected_source_series": selected_paths,
            "rebuilt_nifti": str(rebuilt_nifti),
            "inventory": inventory_by_visit[visit],
            "prior_orientation": orientation_by_visit[visit],
            "prior_eligibility": eligibility_by_visit[visit],
        }

    return {
        "schema_version": 1,
        "case_alias": CASE_ALIAS,
        "patient_id": patient_id,
        "cohort": "I-SPY1",
        "failed_visit": failed_visit,
        "failure": failure,
        "visits": visits,
        "source_artifact_sha256": {
            "failure_table": sha256_file(failure_path),
            "model_input_inventory": sha256_file(inventory_path),
            "strict_eligibility_visits": sha256_file(eligibility_path),
            "orientation_header_audit": sha256_file(orientation_path),
        },
    }


def load_frozen_ispy1_contract(prior_root: Path) -> Any:
    """Load the already frozen outcome-free I-SPY1 source contract."""

    source = prior_root / "scripts/run_ispy1_base_eligibility.py"
    if not source.is_file():
        raise ProvenanceAuditError("FROZEN_SOURCE_CONTRACT_MISSING")
    module_name = "_zero_overlap_frozen_ispy1_contract"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ProvenanceAuditError("FROZEN_SOURCE_CONTRACT_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        numeric = value.item() if hasattr(value, "item") else value
        if isinstance(numeric, float) and not np.isfinite(numeric):
            return None
        return numeric
    if isinstance(value, Iterable):
        return [_plain(item) for item in value]
    return str(value)


def _value_key(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False)


def read_selected_series_provenance(series_audit: Any) -> dict[str, Any]:
    """Read every requested tag from every file in a selected raw series."""

    if not getattr(series_audit, "headers", None):
        raise ProvenanceAuditError("SELECTED_SERIES_HEADERS_EMPTY")
    tag_values: dict[str, list[Any]] = {tag: [] for tag in PROVENANCE_TAGS}
    missing: Counter[str] = Counter()
    private_files: list[dict[str, Any]] = []
    for header in series_audit.headers:
        try:
            dataset = pydicom.dcmread(
                str(header.path),
                stop_before_pixels=True,
                force=True,
                specific_tags=list(PROVENANCE_TAGS),
            )
        except Exception as exc:
            raise ProvenanceAuditError("DICOM_PROVENANCE_HEADER_READ_FAILED") from exc
        record: dict[str, Any] = {"source_path": str(Path(header.path).resolve())}
        for tag in PROVENANCE_TAGS:
            raw = getattr(dataset, tag, None)
            value = _plain(raw)
            if raw in (None, ""):
                missing[tag] += 1
            else:
                tag_values[tag].append(value)
            record[tag] = value
        private_files.append(record)

    unique: dict[str, list[Any]] = {}
    for tag, values in tag_values.items():
        by_key = {_value_key(value): value for value in values}
        unique[tag] = [by_key[key] for key in sorted(by_key)]
    requested_complete = {
        tag: int(missing[tag]) == 0 for tag in PROVENANCE_TAGS
    }
    return {
        "file_count": len(private_files),
        "unique_values": unique,
        "missing_counts": dict(sorted(missing.items())),
        "requested_tag_complete": requested_complete,
        "files": private_files,
    }


def representative_value(provenance: Mapping[str, Any], tag: str) -> Any:
    values = list(provenance.get("unique_values", {}).get(tag, []))
    return values[0] if len(values) == 1 else None


def semantic_public_summary(series_audit: Any) -> dict[str, Any]:
    flags = set(str(value) for value in getattr(series_audit, "semantic_flags", []))
    return {
        "geometry_status": str(getattr(series_audit, "status", "FAIL")),
        "geometry_reason": str(getattr(series_audit, "reason", "UNVALIDATED")),
        "modality_mr": "NOT_MR" not in flags,
        "original_primary": not bool(
            flags & {"NOT_ORIGINAL_PRIMARY", "NOT_EXPLICIT_ORIGINAL_PRIMARY"}
        ),
        "breast": "BODY_PART_MISMATCH" not in flags,
        "native_dce_semantics": not bool(
            flags
            & {
                "SUBTRACTION",
                "BOLD",
                "PELVIC",
                "NON_DCE_SEQUENCE",
                "NO_POSITIVE_T1_DCE_EVIDENCE",
            }
        ),
        "derived_or_subtraction": bool(
            flags
            & {
                "NOT_ORIGINAL_PRIMARY",
                "NOT_EXPLICIT_ORIGINAL_PRIMARY",
                "SUBTRACTION",
            }
        ),
        "temporal_stack_valid": bool(
            getattr(series_audit, "status", "") == "PASS"
            and (
                getattr(series_audit, "is_dynamic", False)
                or getattr(series_audit, "is_single_volume", False)
            )
        ),
        "dynamic": bool(getattr(series_audit, "is_dynamic", False)),
        "single_volume": bool(getattr(series_audit, "is_single_volume", False)),
        "phase_count": len(getattr(series_audit, "phase_times_seconds", [])),
        "semantic_flags": "|".join(sorted(flags)) if flags else "NONE",
    }


def audit_failed_study_candidates(
    frozen_contract: Any,
    *,
    study_dir: Path,
    selected_paths: Sequence[str],
) -> dict[str, Any]:
    """Enumerate every source-semantic candidate before any overlap is read."""

    audits = frozen_contract.audit_study(study_dir)
    if not audits:
        raise ProvenanceAuditError("FAILED_STUDY_SERIES_EMPTY")
    candidates = frozen_contract.alternative_candidates(audits)
    selected_key = tuple(sorted(str(Path(value)) for value in selected_paths))
    candidate_keys = {candidate.source_key for candidate in candidates}
    alternatives = [candidate for candidate in candidates if candidate.source_key != selected_key]
    current_valid = selected_key in candidate_keys

    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for index, audit in enumerate(sorted(audits, key=lambda value: str(value.path)), start=1):
        label = f"SERIES_{index:03d}"
        summary = semantic_public_summary(audit)
        source_path = str(Path(audit.path))
        member_keys = [key for key in candidate_keys if source_path in key]
        public = {
            "case_alias": CASE_ALIAS,
            "series_alias": label,
            **summary,
            "strict_candidate_member": bool(member_keys),
            "selected_current": source_path in selected_key,
            "alternate_candidate_member": any(
                source_path in candidate.source_key for candidate in alternatives
            ),
            "raw_pixel_rebuild": "PENDING"
            if source_path in selected_key
            else "NOT_REQUIRED_NO_UNIQUE_REPLACEMENT",
        }
        rows.append(public)
        private_rows.append(
            {
                **public,
                "source_path": str(Path(audit.path).resolve()),
                "series_instance_uid": str(audit.headers[0].series_uid)
                if getattr(audit, "headers", None)
                else "",
            }
        )
    return {
        "audits": audits,
        "candidates": candidates,
        "alternatives": alternatives,
        "selected_key": selected_key,
        "current_valid": current_valid,
        "strict_candidate_count": len(candidates),
        "strict_alternate_count": len(alternatives),
        "public_rows": rows,
        "private_rows": private_rows,
    }


def rebuild_selected_series(frozen_contract: Any, series_audit: Any) -> dict[str, Any]:
    """Decode and independently compare every raw PixelData cell twice."""

    try:
        volume, cells = frozen_contract.decode_series(series_audit)
    except Exception as exc:
        raise ProvenanceAuditError("RAW_PIXEL_REBUILD_FAILED") from exc
    array = np.asarray(volume, dtype=np.float32)
    finite = bool(np.isfinite(array).all())
    nonconstant = bool(finite and float(np.std(array, dtype=np.float64)) > 0.0)
    expected_cells = int(len(series_audit.headers))
    exact_second = bool(
        len(cells) == expected_cells
        and all(bool(item.get("second_comparison_exact")) for item in cells)
    )
    if not finite or not nonconstant or not exact_second:
        raise ProvenanceAuditError("RAW_PIXEL_REBUILD_QC_FAILED")
    return {
        "volume_xyzt": array,
        "affine_ras": np.asarray(series_audit.affine_ras, dtype=np.float64),
        "private_cells": cells,
        "public": {
            "status": "PASS",
            "file_count": expected_cells,
            "verified_cell_count": len(cells),
            "second_pixel_comparison_exact": exact_second,
            "finite": finite,
            "nonconstant": nonconstant,
            "shape_xyzt": [int(value) for value in array.shape],
            "volume_min": float(np.min(array)),
            "volume_max": float(np.max(array)),
            "volume_std": float(np.std(array, dtype=np.float64)),
        },
    }


def scan_study_objects(study_dir: Path) -> dict[str, Any]:
    """Inventory SOP classes and registration objects for one study."""

    series_uids: set[str] = set()
    study_uids: set[str] = set()
    sop_class_counts: Counter[str] = Counter()
    registration_objects: list[dict[str, Any]] = []
    readable = 0
    for path in sorted(item for item in study_dir.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "SOPClassUID",
                    "SOPInstanceUID",
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "FrameOfReferenceUID",
                    "RegistrationSequence",
                    "ReferencedSeriesSequence",
                ],
            )
        except Exception:
            continue
        readable += 1
        study_uid = str(getattr(dataset, "StudyInstanceUID", "") or "")
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "") or "")
        sop_uid = str(getattr(dataset, "SOPClassUID", "") or "")
        if study_uid:
            study_uids.add(study_uid)
        if series_uid:
            series_uids.add(series_uid)
        sop_name = pydicom.uid.UID(sop_uid).name if sop_uid else "MISSING"
        sop_class_counts[str(sop_name)] += 1
        if sop_uid in SPATIAL_REGISTRATION_UIDS or hasattr(dataset, "RegistrationSequence"):
            registration_objects.append(
                {
                    "source_path": str(path.resolve()),
                    "sop_class_uid": sop_uid,
                    "sop_instance_uid": str(
                        getattr(dataset, "SOPInstanceUID", "") or ""
                    ),
                    "study_instance_uid": study_uid,
                    "series_instance_uid": series_uid,
                    "frame_of_reference_uid": str(
                        getattr(dataset, "FrameOfReferenceUID", "") or ""
                    ),
                    "registration_sequence_present": hasattr(
                        dataset, "RegistrationSequence"
                    ),
                }
            )
    return {
        "readable_instances": readable,
        "study_instance_uids": sorted(study_uids),
        "series_instance_uids": sorted(series_uids),
        "sop_class_counts": dict(sorted(sop_class_counts.items())),
        "registration_objects": registration_objects,
        "public": {
            "readable_instances": readable,
            "study_uid_count": len(study_uids),
            "series_uid_count": len(series_uids),
            "sop_class_counts": dict(sorted(sop_class_counts.items())),
            "spatial_registration_object_count": len(registration_objects),
            "authoritative_registration_relationship_found": bool(
                registration_objects
            ),
        },
    }


def atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a mode-0600 private JSON without leaking values to stdout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Some private JSONs intentionally live beside public summaries in
    # manifests/ or metrics/.  Restrict the artifact itself, but never chmod a
    # mixed public directory.  A dedicated directory literally named
    # ``private`` (and any descendants) remains owner-only.
    private_boundary = next(
        (
            directory
            for directory in (path.parent, *path.parent.parents)
            if directory.name.casefold() == "private"
        ),
        None,
    )
    if private_boundary is not None:
        directory = path.parent
        while True:
            os.chmod(directory, 0o700)
            if directory == private_boundary:
                break
            directory = directory.parent
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CASE_ALIAS",
    "PROVENANCE_TAGS",
    "ProvenanceAuditError",
    "VISITS",
    "atomic_private_json",
    "audit_failed_study_candidates",
    "load_frozen_ispy1_contract",
    "read_selected_series_provenance",
    "rebuild_selected_series",
    "representative_value",
    "resolve_private_case",
    "scan_study_objects",
    "sha256_file",
]
