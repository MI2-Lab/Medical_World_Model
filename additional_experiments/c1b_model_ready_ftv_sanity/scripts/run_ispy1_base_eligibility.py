#!/usr/bin/env python3
"""Build a strict, outcome-free I-SPY1 base-eligibility cohort.

This runner intentionally lives outside the shared C1B preprocessing modules.  It
reads only the raw imaging references in the existing I-SPY1 manifests, validates
the complete raw DICOM grid, decodes PixelData twice, and emits a rebuilt 4-D
NIfTI only when the visit passes every contract check.  A patient is eligible
only when T0, T1, T2, and T3 all pass.

Public artifacts contain aggregate counts only.  Patient identifiers, paths,
UIDs, and per-cell hashes are written exclusively to ``*.private.*`` artifacts.
No clinical, pathology, response, FTV, or outcome table is read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import affine_transform


SCHEMA_VERSION = 1
VISITS = ("T0", "T1", "T2", "T3")
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
POSITION_ATOL_MM = 0.001
IN_PLANE_ATOL_MM = 0.10
CORNER_ATOL_MM = 0.10
IOP_ATOL = 1e-4
SPACING_ATOL_MM = 1e-3
RESAMPLE_MAX_ANGLE_DEG = 1.0
RESAMPLE_MAX_SPACING_RELATIVE_ERROR = 0.02
RESAMPLE_MIN_VALID_FRACTION = 0.99

HEADER_TAGS = (
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "ImageType",
    "SeriesDescription",
    "ProtocolName",
    "BodyPartExamined",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "MRAcquisitionType",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "EchoTrainLength",
    "AcquisitionDate",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "TemporalPositionIdentifier",
    "AcquisitionNumber",
    "AcquisitionTime",
    "ContentTime",
    "InstanceNumber",
    "SeriesNumber",
    "RescaleSlope",
    "RescaleIntercept",
)

NEGATIVE_TEXT_RE = re.compile(
    r"(?:BOLD|\bPELV(?:IS|IC)?\b|\bT2\b|SCOUT|LOCALI[ZS]ER|\bLOC\.?\b|"
    r"DIFFUS|\bDWI\b|DWSSFSE|FIESTA|SEGMENT|BREAST\s*TISSUE|\bPJ[NM]\b|"
    r"SURVEY|B0.?MAP|POST.?NEEDLE|LABEL.?CONTROL|CONTROL.?LABEL|"
    r"(?:^|[^A-Z])PE1(?:[^A-Z]|$)|(?:^|[^A-Z])SER(?:[^A-Z]|$))",
    re.IGNORECASE,
)
SUBTRACTION_RE = re.compile(
    r"(?:SUBTRACTION|SUBTRACTED|(?:^|[^A-Z])SUBT?(?:RACT(?:ION)?)?(?:[^A-Z]|$))",
    re.IGNORECASE,
)
POSITIVE_DCE_RE = re.compile(
    r"(?:DYNAMIC|(?:^|[^A-Z0-9])(?:\d+\s*)?DYN(?:AMIC)?(?:[^A-Z0-9]|$)|"
    r"3D\s*F?GRE|3DFGRE|FL[_ -]?3D|IR[-_ ]?SPGR|F[MS]?P?SPGR|SPGR|VIBRANT|"
    r"3D\s*SAG|SAG\s*3D|3DSAG|(?:^|[^A-Z])PASS\s*\d*(?:[^A-Z]|$)|"
    r"PRE.?POST|PRE.?CONTRAST|POST.?CONTRAST|CONTRAST|(?:^|[^A-Z])CE\d*(?:[^A-Z]|$)|"
    r"S[AG]{1,2}T1CE|(?:^|[^A-Z])T1(?:[^A-Z]|$))",
    re.IGNORECASE,
)


class ContractFailure(RuntimeError):
    """A fail-closed visit error with a stable, privacy-safe reason code."""

    def __init__(self, code: str, private_detail: str = "") -> None:
        self.code = str(code)
        self.private_detail = str(private_detail)
        super().__init__(self.code)


@dataclass(frozen=True)
class SliceHeader:
    path: str
    sop_uid: str
    series_uid: str
    modality: str
    image_type: tuple[str, ...]
    description: str
    protocol: str
    body_part: str
    sequence_name: str
    scanning_sequence: str
    sequence_variant: str
    mr_acquisition_type: str
    repetition_time_ms: float | None
    echo_time_ms: float | None
    flip_angle_degrees: float | None
    echo_train_length: int | None
    acquisition_date: str
    rows: int
    columns: int
    iop: tuple[float, ...]
    ipp: tuple[float, ...]
    pixel_spacing: tuple[float, float]
    temporal_position: int | None
    acquisition_number: int | None
    acquisition_time_seconds: float | None
    instance_number: int | None
    series_number: int | None
    slope: float
    intercept: float


@dataclass
class SeriesAudit:
    path: str
    headers: list[SliceHeader]
    status: str = "FAIL"
    reason: str = "UNVALIDATED"
    semantic_ok: bool = False
    semantic_flags: list[str] = field(default_factory=list)
    description: str = ""
    protocol: str = ""
    laterality: str = ""
    rows: int = 0
    columns: int = 0
    affine_ras: np.ndarray | None = None
    slice_labels: list[float] = field(default_factory=list)
    slice_index_by_path: dict[str, int] = field(default_factory=dict)
    time_index_by_path: dict[str, int] = field(default_factory=dict)
    phase_times_seconds: list[float | None] = field(default_factory=list)
    temporal_method: str = ""
    is_single_volume: bool = False
    is_dynamic: bool = False
    spacing_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    in_plane_max_deviation_mm: float = math.inf
    slice_spacing_max_deviation_mm: float = math.inf

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return (self.columns, self.rows, len(self.slice_labels))


@dataclass
class Candidate:
    mode: str
    series: list[SeriesAudit]
    provenance: str
    phase_times_seconds: list[float | None]
    max_corner_mismatch_mm: float = 0.0
    min_valid_fraction: float = 1.0
    needs_resampling: list[bool] = field(default_factory=list)

    @property
    def source_key(self) -> tuple[str, ...]:
        return tuple(sorted(item.path for item in self.series))


def parse_args() -> argparse.Namespace:
    experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("/data/data/Preprocessed/I-SPY1"),
        help="Existing manifest root. Only manifest imaging fields are read.",
    )
    parser.add_argument("--experiment-root", type=Path, default=experiment_root)
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--smoke", action="store_true", help="Run five deterministic audit strata.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def fail(code: str, detail: str = "") -> None:
    raise ContractFailure(code, detail)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def patient_token(patient_id: str) -> str:
    return sha256_bytes(f"ispy1-base-v1:{patient_id}".encode("utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def finite_tuple(value: Any, length: int, tag: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        fail("INVALID_REQUIRED_HEADER", f"{tag}: {exc}")
    if len(result) != length or not np.all(np.isfinite(result)):
        fail("INVALID_REQUIRED_HEADER", tag)
    return result


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def positive_int(value: Any, tag: str) -> int:
    numeric = optional_int(value)
    if numeric is None or numeric <= 0:
        fail("INVALID_REQUIRED_HEADER", tag)
    return numeric


def dicom_time_seconds(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace(":", "")
    try:
        if len(text) < 6:
            text = text.zfill(6)
        hours = int(text[:2])
        minutes = int(text[2:4])
        seconds = float(text[4:])
    except (TypeError, ValueError):
        return None
    result = hours * 3600.0 + minutes * 60.0 + seconds
    if hours > 23 or minutes > 59 or seconds >= 60 or not np.isfinite(result):
        return None
    return result


def unwrap_times(values: Sequence[float | None]) -> list[float | None]:
    if not values or any(value is None for value in values):
        return list(values)
    result: list[float | None] = []
    day_offset = 0.0
    previous: float | None = None
    for value in values:
        assert value is not None
        current = value + day_offset
        if previous is not None and current < previous - 12 * 3600:
            day_offset += 24 * 3600
            current = value + day_offset
        result.append(current)
        previous = current
    return result


def discover_dicom_files(series_dir: Path) -> list[Path]:
    files = sorted(path for path in series_dir.glob("*.dcm") if path.is_file())
    if not files:
        files = sorted(path for path in series_dir.iterdir() if path.is_file())
    if not files:
        fail("EMPTY_SERIES", str(series_dir))
    return files


def read_slice_header(path: Path) -> SliceHeader:
    try:
        ds = pydicom.dcmread(
            str(path), stop_before_pixels=True, force=True, specific_tags=list(HEADER_TAGS)
        )
    except Exception as exc:  # noqa: BLE001 - converted to a stable private failure.
        fail("DICOM_HEADER_READ_FAILED", f"{path}: {exc}")
    frames = optional_int(getattr(ds, "NumberOfFrames", None))
    if frames not in (None, 1):
        fail("ENHANCED_MULTIFRAME_UNSUPPORTED", str(path))
    slope_raw = getattr(ds, "RescaleSlope", None)
    intercept_raw = getattr(ds, "RescaleIntercept", None)
    slope = 1.0 if slope_raw is None or str(slope_raw).strip() == "" else float(slope_raw)
    intercept = (
        0.0
        if intercept_raw is None or str(intercept_raw).strip() == ""
        else float(intercept_raw)
    )
    if not np.isfinite(slope) or slope == 0.0 or not np.isfinite(intercept):
        fail("INVALID_RESCALE", str(path))
    acq_time = dicom_time_seconds(getattr(ds, "AcquisitionTime", None))
    if acq_time is None:
        acq_time = dicom_time_seconds(getattr(ds, "ContentTime", None))
    image_type_raw = getattr(ds, "ImageType", ()) or ()
    if isinstance(image_type_raw, str):
        image_type_raw = image_type_raw.split("\\")
    return SliceHeader(
        path=str(path),
        sop_uid=str(getattr(ds, "SOPInstanceUID", "") or ""),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "") or ""),
        modality=str(getattr(ds, "Modality", "") or "").upper(),
        image_type=tuple(str(item).upper().strip() for item in image_type_raw),
        description=str(getattr(ds, "SeriesDescription", "") or ""),
        protocol=str(getattr(ds, "ProtocolName", "") or ""),
        body_part=str(getattr(ds, "BodyPartExamined", "") or ""),
        sequence_name=str(getattr(ds, "SequenceName", "") or ""),
        scanning_sequence=str(getattr(ds, "ScanningSequence", "") or ""),
        sequence_variant=str(getattr(ds, "SequenceVariant", "") or ""),
        mr_acquisition_type=str(getattr(ds, "MRAcquisitionType", "") or ""),
        repetition_time_ms=optional_float(getattr(ds, "RepetitionTime", None)),
        echo_time_ms=optional_float(getattr(ds, "EchoTime", None)),
        flip_angle_degrees=optional_float(getattr(ds, "FlipAngle", None)),
        echo_train_length=optional_int(getattr(ds, "EchoTrainLength", None)),
        acquisition_date=str(getattr(ds, "AcquisitionDate", "") or ""),
        rows=positive_int(getattr(ds, "Rows", None), "Rows"),
        columns=positive_int(getattr(ds, "Columns", None), "Columns"),
        iop=finite_tuple(getattr(ds, "ImageOrientationPatient", None), 6, "IOP"),
        ipp=finite_tuple(getattr(ds, "ImagePositionPatient", None), 3, "IPP"),
        pixel_spacing=finite_tuple(getattr(ds, "PixelSpacing", None), 2, "PixelSpacing"),
        temporal_position=optional_int(getattr(ds, "TemporalPositionIdentifier", None)),
        acquisition_number=optional_int(getattr(ds, "AcquisitionNumber", None)),
        acquisition_time_seconds=acq_time,
        instance_number=optional_int(getattr(ds, "InstanceNumber", None)),
        series_number=optional_int(getattr(ds, "SeriesNumber", None)),
        slope=slope,
        intercept=intercept,
    )


def infer_laterality(text: str) -> str:
    upper = f" {text.upper().replace('_', ' ')} "
    left = bool(re.search(r"(?:^|\s)(?:LEFT|LT|L)(?:\s|$)", upper))
    right = bool(re.search(r"(?:^|\s)(?:RIGHT|RT|R)(?:\s|$)", upper))
    if left and not right:
        return "L"
    if right and not left:
        return "R"
    return ""


def semantic_contract(headers: Sequence[SliceHeader]) -> tuple[bool, list[str], str]:
    flags: set[str] = set()
    text = " | ".join(
        f"{header.description} {header.protocol} {header.body_part} {header.sequence_name}"
        for header in headers
    )
    if any(header.modality != "MR" for header in headers):
        flags.add("NOT_MR")
    if any(
        any(token in header.image_type for token in ("DERIVED", "SECONDARY", "MIP", "MPR", "PROJECTION", "LOCALIZER"))
        for header in headers
    ):
        flags.add("NOT_ORIGINAL_PRIMARY")
    if any(len(header.image_type) < 2 or header.image_type[:2] != ("ORIGINAL", "PRIMARY") for header in headers):
        flags.add("NOT_EXPLICIT_ORIGINAL_PRIMARY")
    if any("SUB" in item for header in headers for item in header.image_type):
        flags.add("SUBTRACTION")
    if SUBTRACTION_RE.search(text):
        flags.add("SUBTRACTION")
    # Historical I-SPY descriptions commonly join BOLD to another token with
    # underscores, which are word characters to regular expressions.
    if "BOLD" in text.upper():
        flags.add("BOLD")
    if re.search(r"\bPELV(?:IS|IC)?\b", text, flags=re.IGNORECASE):
        flags.add("PELVIC")
    if NEGATIVE_TEXT_RE.search(text):
        flags.add("NON_DCE_SEQUENCE")
    body_parts = {header.body_part.upper().strip() for header in headers if header.body_part.strip()}
    if body_parts != {"BREAST"}:
        flags.add("BODY_PART_MISMATCH")
    first = headers[0]
    sequence_name_positive = bool(
        re.match(r"^[* ]*(?:FL3|FGRE3D)", first.sequence_name, flags=re.IGNORECASE)
    )
    bare_3d_gradient = first.description.strip().upper() == "3D" and bool(
        re.search(r"(?:^|\\)(?:GR|RM)(?:\\|$)", first.scanning_sequence.upper())
    )
    if not POSITIVE_DCE_RE.search(text) and not sequence_name_positive and not bare_3d_gradient:
        flags.add("NO_POSITIVE_T1_DCE_EVIDENCE")
    return (not flags), sorted(flags), infer_laterality(text)


def cluster_positions(values: Sequence[float], atol: float) -> tuple[list[float], list[int]]:
    if not values:
        fail("EMPTY_SERIES")
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    clusters: list[list[float]] = []
    assignment = [-1] * len(values)
    for raw_index in order:
        value = float(values[int(raw_index)])
        if not clusters or abs(value - float(np.mean(clusters[-1]))) > atol:
            clusters.append([value])
        else:
            clusters[-1].append(value)
        assignment[int(raw_index)] = len(clusters) - 1
    centers = [float(np.mean(cluster)) for cluster in clusters]
    return centers, assignment


def complete_grid(
    headers: Sequence[SliceHeader],
    slice_indices: Sequence[int],
    keys: Sequence[int | None],
    method: str,
) -> tuple[dict[str, int], list[float | None]] | None:
    if any(key is None for key in keys):
        return None
    unique_keys = sorted({int(key) for key in keys if key is not None})
    slice_count = max(slice_indices) + 1
    if len(unique_keys) < 2 or len(headers) != len(unique_keys) * slice_count:
        return None
    cells: dict[tuple[int, int], int] = Counter(
        (int(key), int(slice_index))
        for key, slice_index in zip(keys, slice_indices, strict=True)
        if key is not None
    )
    if len(cells) != len(headers) or any(count != 1 for count in cells.values()):
        return None
    group_times: dict[int, list[float]] = defaultdict(list)
    for header, key in zip(headers, keys, strict=True):
        assert key is not None
        if header.acquisition_time_seconds is not None:
            group_times[int(key)].append(header.acquisition_time_seconds)
    medians: list[float | None] = [
        float(np.median(group_times[key])) if group_times[key] else None for key in unique_keys
    ]
    unwrapped = unwrap_times(medians)
    if all(value is not None for value in unwrapped):
        numeric = [float(value) for value in unwrapped if value is not None]
        if len(set(numeric)) == 1:
            # Some classic series write one series-level AcquisitionTime into
            # every phase.  The verified integer temporal key remains valid,
            # but the time values contain no phase timing information.
            unwrapped = [None] * len(unwrapped)
        elif any(later <= earlier for earlier, later in zip(numeric, numeric[1:])):
            return None
    time_map = {key: index for index, key in enumerate(unique_keys)}
    return (
        {
            header.path: time_map[int(key)]
            for header, key in zip(headers, keys, strict=True)
            if key is not None
        },
        unwrapped,
    )


def instance_block_grid(
    headers: Sequence[SliceHeader], slice_indices: Sequence[int]
) -> tuple[dict[str, int], list[float | None]] | None:
    if any(header.instance_number is None for header in headers):
        return None
    ordered = sorted(headers, key=lambda header: (int(header.instance_number or 0), header.path))
    instance_numbers = [int(header.instance_number or 0) for header in ordered]
    if len(set(instance_numbers)) != len(instance_numbers):
        return None
    slice_by_path = {header.path: index for header, index in zip(headers, slice_indices, strict=True)}
    slice_count = max(slice_indices) + 1
    if len(ordered) % slice_count != 0 or len(ordered) // slice_count < 2:
        return None
    time_count = len(ordered) // slice_count
    mapping: dict[str, int] = {}
    times: list[float | None] = []
    expected_slices = set(range(slice_count))
    for time_index in range(time_count):
        chunk = ordered[time_index * slice_count : (time_index + 1) * slice_count]
        if {slice_by_path[item.path] for item in chunk} != expected_slices:
            return None
        for item in chunk:
            mapping[item.path] = time_index
        available = [item.acquisition_time_seconds for item in chunk if item.acquisition_time_seconds is not None]
        times.append(float(np.median(available)) if available else None)
    times = unwrap_times(times)
    if all(value is not None for value in times):
        numeric = [float(value) for value in times if value is not None]
        if len(set(numeric)) == 1:
            times = [None] * len(times)
        elif any(later <= earlier for earlier, later in zip(numeric, numeric[1:])):
            return None
    return mapping, times


def audit_series(series_dir: Path) -> SeriesAudit:
    files = discover_dicom_files(series_dir)
    headers = [read_slice_header(path) for path in files]
    audit = SeriesAudit(path=str(series_dir), headers=headers)
    audit.description = headers[0].description
    audit.protocol = headers[0].protocol
    audit.semantic_ok, audit.semantic_flags, audit.laterality = semantic_contract(headers)
    if len({header.sop_uid for header in headers}) != len(headers) or any(not header.sop_uid for header in headers):
        audit.reason = "DUPLICATE_OR_MISSING_SOP_UID"
        return audit
    if len({header.series_uid for header in headers}) != 1 or not headers[0].series_uid:
        audit.reason = "MIXED_OR_MISSING_SERIES_UID"
        return audit
    if len({(header.rows, header.columns) for header in headers}) != 1:
        audit.reason = "INCONSISTENT_MATRIX"
        return audit
    audit.rows, audit.columns = headers[0].rows, headers[0].columns
    reference_iop = np.asarray(headers[0].iop, dtype=np.float64)
    if max(float(np.max(np.abs(np.asarray(header.iop) - reference_iop))) for header in headers) > IOP_ATOL:
        audit.reason = "INCONSISTENT_IOP"
        return audit
    column_direction = reference_iop[:3]
    row_direction = reference_iop[3:]
    if not np.isclose(np.linalg.norm(column_direction), 1.0, atol=IOP_ATOL) or not np.isclose(
        np.linalg.norm(row_direction), 1.0, atol=IOP_ATOL
    ) or not np.isclose(float(np.dot(column_direction, row_direction)), 0.0, atol=IOP_ATOL):
        audit.reason = "NON_ORTHONORMAL_IOP"
        return audit
    normal = np.cross(column_direction, row_direction)
    normal /= np.linalg.norm(normal)
    reference_spacing = np.asarray(headers[0].pixel_spacing, dtype=np.float64)
    if np.any(reference_spacing <= 0) or max(
        float(np.max(np.abs(np.asarray(header.pixel_spacing) - reference_spacing))) for header in headers
    ) > SPACING_ATOL_MM:
        audit.reason = "INCONSISTENT_PIXEL_SPACING"
        return audit
    positions = np.asarray([header.ipp for header in headers], dtype=np.float64)
    projected = positions @ normal
    slice_centers, slice_indices = cluster_positions(projected.tolist(), POSITION_ATOL_MM)
    residuals = positions - np.outer(projected, normal)
    residual_center = np.median(residuals, axis=0)
    audit.in_plane_max_deviation_mm = float(np.max(np.linalg.norm(residuals - residual_center, axis=1)))
    if audit.in_plane_max_deviation_mm > IN_PLANE_ATOL_MM:
        audit.reason = "IN_PLANE_POSITION_DRIFT"
        return audit
    audit.slice_labels = slice_centers
    audit.slice_index_by_path = {
        header.path: int(index) for header, index in zip(headers, slice_indices, strict=True)
    }
    if len(slice_centers) < 2:
        audit.reason = "INSUFFICIENT_SLICES"
        return audit
    slice_diffs = np.diff(np.asarray(slice_centers, dtype=np.float64))
    slice_spacing = float(np.median(slice_diffs))
    audit.slice_spacing_max_deviation_mm = float(np.max(np.abs(slice_diffs - slice_spacing)))
    if slice_spacing <= 0 or audit.slice_spacing_max_deviation_mm > POSITION_ATOL_MM:
        audit.reason = "IRREGULAR_SLICE_SPACING"
        return audit
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = column_direction * reference_spacing[1]
    affine_lps[:3, 1] = row_direction * reference_spacing[0]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = residual_center + normal * slice_centers[0]
    audit.affine_ras = LPS_TO_RAS @ affine_lps
    audit.spacing_xyz_mm = (float(reference_spacing[1]), float(reference_spacing[0]), slice_spacing)
    counts = Counter(slice_indices)
    if set(counts.values()) == {1}:
        audit.is_single_volume = True
        audit.time_index_by_path = {header.path: 0 for header in headers}
        available = [header.acquisition_time_seconds for header in headers if header.acquisition_time_seconds is not None]
        audit.phase_times_seconds = [float(np.median(available)) if available else None]
        audit.temporal_method = "single_volume"
        audit.status = "PASS"
        audit.reason = "PASS"
        return audit
    if len(set(counts.values())) != 1:
        audit.reason = "INCOMPLETE_REPEATED_POSITION_GRID"
        return audit
    temporal_results: list[tuple[str, dict[str, int], list[float | None]]] = []
    for method, keys in (
        ("TemporalPositionIdentifier", [header.temporal_position for header in headers]),
        ("AcquisitionNumber", [header.acquisition_number for header in headers]),
    ):
        result = complete_grid(headers, slice_indices, keys, method)
        if result is not None:
            temporal_results.append((method, result[0], result[1]))
    if temporal_results:
        reference_mapping = temporal_results[0][1]
        if any(mapping != reference_mapping for _, mapping, _ in temporal_results[1:]):
            audit.reason = "TEMPORAL_KEYS_DISAGREE"
            return audit
        audit.temporal_method = "+".join(method for method, _, _ in temporal_results)
        audit.time_index_by_path = reference_mapping
        audit.phase_times_seconds = temporal_results[0][2]
    else:
        result = instance_block_grid(headers, slice_indices)
        if result is None:
            audit.reason = "NO_VERIFIED_TEMPORAL_GRID"
            return audit
        audit.time_index_by_path, audit.phase_times_seconds = result
        audit.temporal_method = "InstanceNumberPhaseBlocks"
    if audit.time_index_by_path:
        if len(audit.phase_times_seconds) > 20:
            audit.reason = "IMPLAUSIBLE_DYNAMIC_PHASE_COUNT"
            return audit
        audit.is_dynamic = True
        audit.status = "PASS"
        audit.reason = "PASS"
        return audit
    audit.reason = "NO_VERIFIED_TEMPORAL_GRID"
    return audit


def voxel_corners(shape: Sequence[int], footprint: bool = False) -> np.ndarray:
    limits = [(-0.5, float(size) - 0.5) if footprint else (0.0, float(size) - 1.0) for size in shape]
    return np.asarray(
        [[x, y, z, 1.0] for x in limits[0] for y in limits[1] for z in limits[2]],
        dtype=np.float64,
    )


def corner_hausdorff_mm(
    first_affine: np.ndarray, first_shape: Sequence[int], second_affine: np.ndarray, second_shape: Sequence[int]
) -> float:
    first = (first_affine @ voxel_corners(first_shape).T).T[:, :3]
    second = (second_affine @ voxel_corners(second_shape).T).T[:, :3]
    pairwise = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(max(np.max(np.min(pairwise, axis=1)), np.max(np.min(pairwise, axis=0))))


def direction_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_dir = first[:3, :3] / np.linalg.norm(first[:3, :3], axis=0, keepdims=True)
    second_dir = second[:3, :3] / np.linalg.norm(second[:3, :3], axis=0, keepdims=True)
    dots = np.clip(np.abs(np.sum(first_dir * second_dir, axis=0)), -1.0, 1.0)
    return float(np.max(np.degrees(np.arccos(dots))))


def valid_source_fraction(
    reference_affine: np.ndarray,
    reference_shape: Sequence[int],
    source_affine: np.ndarray,
    source_shape: Sequence[int],
) -> float:
    transform = np.linalg.inv(source_affine) @ reference_affine
    x = np.arange(int(reference_shape[0]), dtype=np.float64)[:, None, None]
    y = np.arange(int(reference_shape[1]), dtype=np.float64)[None, :, None]
    z = np.arange(int(reference_shape[2]), dtype=np.float64)[None, None, :]
    valid = np.ones(tuple(int(item) for item in reference_shape), dtype=bool)
    for axis in range(3):
        mapped = transform[axis, 0] * x + transform[axis, 1] * y + transform[axis, 2] * z + transform[axis, 3]
        valid &= (mapped >= -0.5) & (mapped <= float(source_shape[axis]) - 0.5)
    return float(np.mean(valid))


def validate_phase_stack(series: Sequence[SeriesAudit], provenance: str) -> Candidate:
    if len(series) < 2:
        fail("INSUFFICIENT_VALID_PHASES")
    if any(item.status != "PASS" or not item.is_single_volume or not item.semantic_ok for item in series):
        fail("INVALID_PHASE_SERIES")
    known_lateralities = {item.laterality for item in series if item.laterality}
    if len(known_lateralities) > 1:
        fail("PHASE_LATERALITY_MISMATCH")
    ordered = sorted(
        series,
        key=lambda item: (
            item.headers[0].acquisition_date,
            item.phase_times_seconds[0] if item.phase_times_seconds[0] is not None else 10**9,
            item.headers[0].series_number if item.headers[0].series_number is not None else 10**9,
            item.path,
        ),
    )
    times = unwrap_times([item.phase_times_seconds[0] for item in ordered])
    if all(value is not None for value in times):
        numeric = [float(value) for value in times if value is not None]
        if any(later <= earlier for earlier, later in zip(numeric, numeric[1:])):
            fail("NON_MONOTONIC_PHASE_TIME")
    reference = ordered[0]
    assert reference.affine_ras is not None
    needs_resampling = [False]
    max_corner = 0.0
    min_valid = 1.0
    for item in ordered[1:]:
        assert item.affine_ras is not None
        corner = corner_hausdorff_mm(reference.affine_ras, reference.shape_xyz, item.affine_ras, item.shape_xyz)
        max_corner = max(max_corner, corner)
        if corner <= CORNER_ATOL_MM and item.shape_xyz == reference.shape_xyz:
            needs_resampling.append(False)
            continue
        angle = direction_angle_degrees(reference.affine_ras, item.affine_ras)
        spacing_error = float(
            np.max(
                np.abs(np.asarray(reference.spacing_xyz_mm) - np.asarray(item.spacing_xyz_mm))
                / np.maximum(np.asarray(reference.spacing_xyz_mm), 1e-12)
            )
        )
        if angle > RESAMPLE_MAX_ANGLE_DEG or spacing_error > RESAMPLE_MAX_SPACING_RELATIVE_ERROR:
            fail("UNSAFE_PHASE_GEOMETRY")
        fraction = valid_source_fraction(
            reference.affine_ras, reference.shape_xyz, item.affine_ras, item.shape_xyz
        )
        min_valid = min(min_valid, fraction)
        if fraction < RESAMPLE_MIN_VALID_FRACTION:
            fail("UNSAFE_PHASE_FOV_OVERLAP")
        needs_resampling.append(True)
    return Candidate(
        mode="phase_stack",
        series=ordered,
        provenance=provenance,
        phase_times_seconds=times,
        max_corner_mismatch_mm=max_corner,
        min_valid_fraction=min_valid,
        needs_resampling=needs_resampling,
    )


def normalized_dicom_text(value: str) -> str:
    return " ".join(str(value).upper().replace("_", " ").split())


def rounded_optional(value: float | None, quantum: float) -> float | None:
    return None if value is None else round(value / quantum) * quantum


def acquisition_family_key(item: SeriesAudit) -> tuple[Any, ...]:
    """Exact outcome-free fingerprint for grouping 3-D phases into one run."""

    header = item.headers[0]
    return (
        item.laterality,
        normalized_dicom_text(header.sequence_name),
        normalized_dicom_text(header.scanning_sequence),
        normalized_dicom_text(header.sequence_variant),
        normalized_dicom_text(header.mr_acquisition_type),
        rounded_optional(header.repetition_time_ms, 0.01),
        rounded_optional(header.echo_time_ms, 0.01),
        rounded_optional(header.flip_angle_degrees, 0.1),
        header.echo_train_length,
    )


def validate_replacement_phase_run(series: Sequence[SeriesAudit]) -> Candidate:
    candidate = validate_phase_stack(series, "same_study_original_primary_replacement")
    if any(candidate.needs_resampling):
        fail("REPLACEMENT_PHASES_NOT_COMMON_FRAME")
    if any(value is None for value in candidate.phase_times_seconds):
        fail("REPLACEMENT_PHASE_TIME_MISSING")
    times = [float(value) for value in candidate.phase_times_seconds if value is not None]
    if times[-1] - times[0] > 30.0 * 60.0:
        fail("REPLACEMENT_PHASE_RUN_TOO_LONG")
    series_numbers = [item.headers[0].series_number for item in candidate.series]
    if any(value is None for value in series_numbers):
        fail("REPLACEMENT_SERIES_NUMBER_MISSING")
    numeric_series = [int(value) for value in series_numbers if value is not None]
    if any(later <= earlier for earlier, later in zip(numeric_series, numeric_series[1:])):
        fail("REPLACEMENT_SERIES_ORDER_DISAGREES")
    for first_index, first in enumerate(candidate.series):
        assert first.affine_ras is not None
        for second in candidate.series[first_index + 1 :]:
            assert second.affine_ras is not None
            if corner_hausdorff_mm(first.affine_ras, first.shape_xyz, second.affine_ras, second.shape_xyz) > CORNER_ATOL_MM:
                fail("REPLACEMENT_PHASES_NOT_COMMON_FRAME")
    return candidate


def audit_study(study_dir: Path) -> list[SeriesAudit]:
    results: list[SeriesAudit] = []
    for series_dir in sorted(path for path in study_dir.iterdir() if path.is_dir()):
        try:
            results.append(audit_series(series_dir))
        except ContractFailure as exc:
            results.append(SeriesAudit(path=str(series_dir), headers=[], reason=exc.code))
    return results


def current_candidate(visit: Mapping[str, Any], by_path: Mapping[str, SeriesAudit]) -> Candidate:
    raw_paths = [str(Path(path)) for path in visit.get("raw_dce_series", [])]
    selected = [by_path[path] for path in raw_paths if path in by_path]
    if len(selected) != len(raw_paths) or not selected:
        fail("CURRENT_SOURCE_NOT_FOUND")
    mode = str(visit.get("dce_selection_mode", ""))
    if mode == "dynamic":
        item = selected[0]
        if item.status != "PASS":
            fail(item.reason)
        if not item.semantic_ok:
            fail("CURRENT_SEMANTIC_REJECTION", ",".join(item.semantic_flags))
        if not item.is_dynamic:
            fail("CURRENT_NOT_VERIFIED_DYNAMIC")
        return Candidate(
            mode="dynamic",
            series=[item],
            provenance="current_raw_verified",
            phase_times_seconds=item.phase_times_seconds,
            needs_resampling=[False],
        )
    if mode == "phase_stack":
        valid = [item for item in selected if item.status == "PASS" and item.is_single_volume and item.semantic_ok]
        provenance = "current_raw_verified" if len(valid) == len(selected) else "current_filtered_original_primary"
        return validate_phase_stack(valid, provenance)
    fail("UNKNOWN_CURRENT_SELECTION_MODE")


def alternative_candidates(all_series: Sequence[SeriesAudit]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for item in all_series:
        if item.status == "PASS" and item.semantic_ok and item.is_dynamic:
            candidates.append(
                Candidate(
                    mode="dynamic",
                    series=[item],
                    provenance="same_study_original_primary_replacement",
                    phase_times_seconds=item.phase_times_seconds,
                    needs_resampling=[False],
                )
            )
    groups: dict[tuple[Any, ...], list[SeriesAudit]] = defaultdict(list)
    for item in all_series:
        if item.status != "PASS" or not item.semantic_ok or not item.is_single_volume:
            continue
        groups[acquisition_family_key(item)].append(item)
    for group in groups.values():
        if len(group) < 2:
            continue
        try:
            candidates.append(validate_replacement_phase_run(group))
        except ContractFailure:
            continue
    deduplicated: dict[tuple[str, ...], Candidate] = {}
    for candidate in candidates:
        deduplicated[candidate.source_key] = candidate
    return list(deduplicated.values())


def choose_candidate(visit: Mapping[str, Any], study_audits: Sequence[SeriesAudit]) -> Candidate:
    by_path = {str(Path(item.path)): item for item in study_audits}
    current_error: ContractFailure | None = None
    try:
        return current_candidate(visit, by_path)
    except ContractFailure as exc:
        current_error = exc
    alternatives = alternative_candidates(study_audits)
    current_key = tuple(sorted(str(Path(path)) for path in visit.get("raw_dce_series", [])))
    alternatives = [candidate for candidate in alternatives if candidate.source_key != current_key]
    if not alternatives:
        fail("NO_CLEAR_SAME_STUDY_REPLACEMENT", current_error.code if current_error else "")
    if len(alternatives) != 1:
        fail("AMBIGUOUS_SAME_STUDY_REPLACEMENT", f"candidate_count={len(alternatives)}")
    return alternatives[0]


def decode_scaled_xy(header: SliceHeader) -> np.ndarray:
    try:
        ds = pydicom.dcmread(header.path, force=True)
        pixels = np.asarray(ds.pixel_array)
    except Exception as exc:  # noqa: BLE001
        fail("PIXELDATA_DECODE_FAILED", f"{header.path}: {exc}")
    if pixels.ndim != 2 or pixels.shape != (header.rows, header.columns):
        fail("PIXELDATA_MATRIX_MISMATCH", header.path)
    scaled = pixels.astype(np.float32) * np.float32(header.slope) + np.float32(header.intercept)
    if not np.all(np.isfinite(scaled)):
        fail("NONFINITE_PIXELDATA", header.path)
    return np.ascontiguousarray(scaled.T, dtype=np.float32)


def decode_series(
    audit: SeriesAudit,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    time_count = max(audit.time_index_by_path.values()) + 1
    volume = np.empty((*audit.shape_xyz, time_count), dtype=np.float32)
    cells: list[dict[str, Any]] = []
    occupied: set[tuple[int, int]] = set()
    for header in audit.headers:
        z_index = audit.slice_index_by_path[header.path]
        time_index = audit.time_index_by_path[header.path]
        cell = (z_index, time_index)
        if cell in occupied:
            fail("DUPLICATE_DECODE_CELL", header.path)
        pixels = decode_scaled_xy(header)
        volume[:, :, z_index, time_index] = pixels
        occupied.add(cell)
        cells.append(
            {
                "source_path": header.path,
                "sop_instance_uid": header.sop_uid,
                "z_index": z_index,
                "time_index": time_index,
                "scaled_pixel_sha256": sha256_array(pixels),
            }
        )
    expected = len(audit.slice_labels) * time_count
    if len(occupied) != expected:
        fail("MISSING_DECODE_CELL", f"expected={expected},actual={len(occupied)}")
    for header, record in zip(audit.headers, cells, strict=True):
        second = decode_scaled_xy(header)
        expected_pixels = volume[:, :, record["z_index"], record["time_index"]]
        if not np.array_equal(second, expected_pixels):
            fail("SECOND_PIXELDATA_COMPARISON_FAILED", header.path)
        if sha256_array(second) != record["scaled_pixel_sha256"]:
            fail("SECOND_PIXELDATA_HASH_MISMATCH", header.path)
        record["second_comparison_exact"] = True
    return volume, cells


def resample_to_reference(
    source_xyz: np.ndarray,
    source_affine: np.ndarray,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray,
) -> np.ndarray:
    mapping = np.linalg.inv(source_affine) @ reference_affine
    result = affine_transform(
        source_xyz,
        matrix=mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=tuple(int(item) for item in reference_shape),
        order=1,
        # A geometry-safe phase stack can miss a sub-voxel sliver of the
        # reference lattice.  Constant-zero extrapolation would turn that
        # sliver into an acquisition/source sentinel.  Mirror the frozen C1B
        # padding policy instead; the separately audited overlap fraction is
        # still retained in the private phase contract.
        mode="reflect",
        prefilter=False,
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def rebuild_candidate(candidate: Candidate) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    cell_audits: list[dict[str, Any]] = []
    if candidate.mode == "dynamic":
        item = candidate.series[0]
        volume, cells = decode_series(item)
        assert item.affine_ras is not None
        return volume, item.affine_ras, cells, 0
    reference = candidate.series[0]
    assert reference.affine_ras is not None
    phases: list[np.ndarray] = []
    resampled_count = 0
    for time_index, (item, needs_resampling) in enumerate(
        zip(candidate.series, candidate.needs_resampling, strict=True)
    ):
        source, cells = decode_series(item)
        if source.shape[-1] != 1:
            fail("PHASE_SERIES_NOT_3D")
        phase = source[..., 0]
        source_hash = sha256_array(phase)
        if needs_resampling:
            assert item.affine_ras is not None
            phase = resample_to_reference(
                phase, item.affine_ras, reference.shape_xyz, reference.affine_ras
            )
            resampled_count += 1
        for record in cells:
            record["time_index"] = time_index
            record["source_volume_sha256"] = source_hash
            record["post_resample_volume_sha256"] = sha256_array(phase)
            record["resampled"] = bool(needs_resampling)
        cell_audits.extend(cells)
        phases.append(phase)
    return np.stack(phases, axis=3).astype(np.float32), reference.affine_ras, cell_audits, resampled_count


def sanitized_phase_contract(times: Sequence[float | None]) -> dict[str, Any]:
    time_count = len(times)
    if time_count < 2:
        fail("INSUFFICIENT_PHASES")
    unwrapped = unwrap_times(times)
    if all(value is not None for value in unwrapped):
        base = float(unwrapped[0])  # type: ignore[arg-type]
        relative = [round(float(value) - base, 6) for value in unwrapped if value is not None]
        if any(later <= earlier for earlier, later in zip(relative, relative[1:])):
            fail("NON_MONOTONIC_PHASE_TIME")
        timing_source = "dicom_acquisition_or_content_time"
    else:
        relative = []
        timing_source = "indices_only_missing_dicom_time"
    if time_count <= 4:
        early_index = 1
        late_index = time_count - 1
    else:
        early_index = 2
        late_index = min(5, time_count - 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": timing_source,
        "outcome_fields_read": [],
        "time_count": time_count,
        "relative_phase_seconds": relative,
        "pre_index": 0,
        "early_index": early_index,
        "late_index": late_index,
        "index_rule": "T<=4:(0,1,T-1);T>4:(0,2,min(5,T-1))",
    }


def write_nifti_atomic(path: Path, volume: np.ndarray, affine: np.ndarray, phase_seconds: Sequence[float]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.all(np.isfinite(volume)):
        fail("NONFINITE_REBUILT_VOLUME")
    if float(np.std(volume, dtype=np.float64)) == 0.0:
        fail("CONSTANT_REBUILT_VOLUME")
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    spatial = nib.affines.voxel_sizes(affine)
    temporal_spacing = 1.0
    if len(phase_seconds) >= 2:
        diffs = np.diff(np.asarray(phase_seconds, dtype=np.float64))
        if np.all(np.isfinite(diffs)) and np.all(diffs > 0):
            temporal_spacing = float(np.median(diffs))
    image.header.set_xyzt_units("mm", "sec")
    image.header.set_zooms((*[float(value) for value in spatial], temporal_spacing))
    image.set_data_dtype(np.float32)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".nii", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        nib.save(image, str(tmp_path))
        reloaded = nib.load(str(tmp_path))
        reloaded_data = np.asanyarray(reloaded.dataobj).astype(np.float32)
        if not np.array_equal(reloaded_data, volume):
            fail("NIFTI_READBACK_DATA_MISMATCH")
        qform, qcode = reloaded.get_qform(coded=True)
        sform, scode = reloaded.get_sform(coded=True)
        if int(qcode) != 1 or int(scode) != 1:
            fail("NIFTI_FORM_CODE_MISMATCH")
        if qform is None or sform is None or not np.allclose(qform, affine, atol=1e-4) or not np.allclose(
            sform, affine, atol=1e-4
        ):
            fail("NIFTI_AFFINE_READBACK_MISMATCH")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    final = nib.load(str(path))
    return {
        "shape_xyzt": [int(item) for item in final.shape],
        "dtype": str(final.get_data_dtype()),
        "orientation_ras": "".join(nib.aff2axcodes(final.affine)),
        "qform_code": int(final.header["qform_code"]),
        "sform_code": int(final.header["sform_code"]),
        "nifti_sha256": sha256_file(path),
        "volume_sha256": sha256_array(volume),
    }


def private_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)


def process_visit(job: Mapping[str, Any]) -> dict[str, Any]:
    patient_id = str(job["patient_id"])
    token = patient_token(patient_id)
    visit = dict(job["visit"])
    visit_name = str(visit["visit"])
    cache_root = Path(str(job["cache_root"]))
    private_cell_root = Path(str(job["private_cell_root"]))
    output_path = cache_root / token / f"{visit_name}.nii"
    cell_path = private_cell_root / f"{sha256_bytes(f'{token}:{visit_name}'.encode())}.private.json"
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "patient_id": patient_id,
        "patient_token": token,
        "visit": visit_name,
        "smoke_stratum": str(job.get("smoke_stratum", "")),
        "study_dir": str(visit.get("study_dir", "")),
        "current_source_series": json.dumps(visit.get("raw_dce_series", [])),
        "status": "FAIL",
        "failure_reason": "UNEXPECTED_FAILURE",
        "selection_provenance": "",
        "selected_mode": "",
        "selected_source_series": "[]",
        "phase_count": 0,
        "pre_index": "",
        "early_index": "",
        "late_index": "",
        "timing_source": "",
        "relative_phase_seconds": "[]",
        "raw_pixel_cells_verified": 0,
        "resampled_phase_count": 0,
        "max_source_corner_mismatch_mm": "",
        "minimum_resample_valid_fraction": "",
        "rebuilt_nifti": "",
        "nifti_sha256": "",
        "volume_sha256": "",
        "shape_xyzt": "[]",
        "orientation_ras": "",
        "private_cell_audit": "",
        "private_detail": "",
    }
    try:
        if output_path.exists():
            if not bool(job["overwrite"]):
                fail("OUTPUT_EXISTS_REQUIRES_OVERWRITE", str(output_path))
            # The exact visit-level target is known.  Remove it before the
            # audit so a newly failed visit can never leave a stale NIfTI that
            # appears canonical after an overwrite run.
            output_path.unlink()
        if cell_path.exists() and bool(job["overwrite"]):
            cell_path.unlink()
        study_dir = Path(str(visit.get("study_dir", "")))
        if not study_dir.is_dir():
            fail("STUDY_DIRECTORY_MISSING", str(study_dir))
        audits = audit_study(study_dir)
        candidate = choose_candidate(visit, audits)
        volume, affine, cells, resampled_count = rebuild_candidate(candidate)
        phase_contract = sanitized_phase_contract(candidate.phase_times_seconds)
        nifti = write_nifti_atomic(
            output_path,
            volume,
            affine,
            phase_contract["relative_phase_seconds"],
        )
        cell_payload = {
            "schema_version": SCHEMA_VERSION,
            "patient_id": patient_id,
            "patient_token": token,
            "visit": visit_name,
            "study_dir": str(study_dir),
            "selected_source_series": [item.path for item in candidate.series],
            "temporal_methods": [item.temporal_method for item in candidate.series],
            "phase_contract": phase_contract,
            "safe_phase_resample_boundary_mode": "reflect",
            "cells": cells,
        }
        private_json_atomic(cell_path, cell_payload)
        base.update(
            {
                "status": "PASS",
                "failure_reason": "",
                "selection_provenance": candidate.provenance,
                "selected_mode": candidate.mode,
                "selected_source_series": json.dumps([item.path for item in candidate.series]),
                "phase_count": phase_contract["time_count"],
                "pre_index": phase_contract["pre_index"],
                "early_index": phase_contract["early_index"],
                "late_index": phase_contract["late_index"],
                "timing_source": phase_contract["source"],
                "relative_phase_seconds": json.dumps(phase_contract["relative_phase_seconds"]),
                "raw_pixel_cells_verified": len(cells),
                "resampled_phase_count": resampled_count,
                "max_source_corner_mismatch_mm": round(candidate.max_corner_mismatch_mm, 6),
                "minimum_resample_valid_fraction": round(candidate.min_valid_fraction, 9),
                "rebuilt_nifti": str(output_path),
                "nifti_sha256": nifti["nifti_sha256"],
                "volume_sha256": nifti["volume_sha256"],
                "shape_xyzt": json.dumps(nifti["shape_xyzt"]),
                "orientation_ras": nifti["orientation_ras"],
                "private_cell_audit": str(cell_path),
            }
        )
    except ContractFailure as exc:
        base["failure_reason"] = exc.code
        base["private_detail"] = exc.private_detail
    except Exception as exc:  # noqa: BLE001 - batch survives; detail remains private.
        base["failure_reason"] = "UNEXPECTED_FAILURE"
        base["private_detail"] = f"{type(exc).__name__}: {exc}"
    return base


def manifest_jobs(preprocessed_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for manifest_path in sorted(preprocessed_root.glob("ISPY1_*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        patient_id = str(manifest.get("patient_id", ""))
        visits = {str(item.get("visit", "")): item for item in manifest.get("visits", [])}
        if not patient_id or set(visits) != set(VISITS):
            continue
        for visit_name in VISITS:
            jobs.append({"patient_id": patient_id, "visit": visits[visit_name]})
    return jobs


def smoke_signature(job: Mapping[str, Any]) -> str:
    visit = job["visit"]
    raw_paths = [Path(path) for path in visit.get("raw_dce_series", [])]
    mode = str(visit.get("dce_selection_mode", ""))
    audits: list[SeriesAudit] = []
    for path in raw_paths:
        try:
            audits.append(audit_series(path))
        except ContractFailure:
            pass
    flags = {flag for item in audits for flag in item.semantic_flags}
    if "BOLD" in flags:
        return "selected_bold"
    if "SUBTRACTION" in flags or "NOT_ORIGINAL_PRIMARY" in flags:
        return "selected_derived_or_subtraction"
    if mode == "dynamic" and audits and audits[0].is_dynamic and len(audits[0].phase_times_seconds) != int(
        visit.get("n_times", 0) or 0
    ):
        return "dynamic_raw_grid_disagrees_with_prior_nifti"
    if mode == "phase_stack" and len(audits) >= 2:
        try:
            candidate = validate_phase_stack(audits, "smoke")
            if candidate.max_corner_mismatch_mm > CORNER_ATOL_MM:
                return "phase_stack_safe_resample"
        except ContractFailure:
            return "phase_stack_unsafe_geometry"
        return "clean_phase_stack"
    if mode == "dynamic" and audits and audits[0].is_dynamic and audits[0].semantic_ok:
        return "clean_dynamic"
    return "other"


def choose_smoke_jobs(
    jobs: Sequence[dict[str, Any]], workers: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    wanted = (
        "clean_dynamic",
        "clean_phase_stack",
        "selected_bold",
        "selected_derived_or_subtraction",
        "dynamic_raw_grid_disagrees_with_prior_nifti",
    )
    chosen: dict[str, dict[str, Any]] = {}
    patient_tokens: set[str] = set()
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        signatures = pool.map(smoke_signature, jobs, chunksize=4)
        inspected = zip(jobs, signatures, strict=True)
        for job, signature in inspected:
            if len(chosen) == len(wanted):
                break
            token = patient_token(str(job["patient_id"]))
            if signature in wanted and signature not in chosen and token not in patient_tokens:
                smoke_job = dict(job)
                smoke_job["smoke_stratum"] = signature
                chosen[signature] = smoke_job
                patient_tokens.add(token)
    if len(chosen) < len(wanted):
        missing = sorted(set(wanted) - set(chosen))
        fail("SMOKE_STRATA_NOT_FOUND", ",".join(missing))
    return [chosen[name] for name in wanted], {name: 1 for name in wanted}


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fail("EMPTY_OUTPUT_ROWS", str(path))
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", mode="w", encoding="utf-8", newline="", delete=False
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    if private:
        os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)


def public_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def aggregate_results(
    visit_rows: Sequence[Mapping[str, Any]], patient_rows: Sequence[Mapping[str, Any]], smoke_strata: Mapping[str, int]
) -> dict[str, Any]:
    pass_rows = [row for row in visit_rows if row["status"] == "PASS"]
    failed_rows = [row for row in visit_rows if row["status"] != "PASS"]
    reason_counts = Counter(str(row["failure_reason"]) for row in failed_rows)
    mode_counts = Counter(str(row["selected_mode"]) for row in pass_rows)
    provenance_counts = Counter(str(row["selection_provenance"]) for row in pass_rows)
    phase_counts = Counter(str(row["phase_count"]) for row in pass_rows)
    timing_counts = Counter(str(row["timing_source"]) for row in pass_rows)
    orientation_counts = Counter(str(row["orientation_ras"]) for row in pass_rows)
    blocked_source_reasons = Counter(
        detail if re.fullmatch(r"[A-Z0-9_]+", detail) else "UNSPECIFIED_PRIVATE_DETAIL"
        for row in failed_rows
        if (detail := str(row.get("private_detail", "")))
    )
    visit_status_counts = {
        visit: dict(sorted(Counter(str(row["status"]) for row in visit_rows if row["visit"] == visit).items()))
        for visit in VISITS
    }
    failing_visit_distribution = Counter(
        str(4 - int(row["passing_visit_count"])) for row in patient_rows
    )
    smoke_outcomes: dict[str, dict[str, int]] = {}
    for row in visit_rows:
        stratum = str(row.get("smoke_stratum", ""))
        if stratum:
            smoke_outcomes.setdefault(stratum, {"PASS": 0, "FAIL": 0})[str(row["status"])] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "contract": "strict_outcome_free_ispy1_base_eligibility_v1",
        "patient_eligible_rule": "all four visits T0,T1,T2,T3 must PASS",
        "outcome_fields_read": [],
        "clinical_or_outcome_tables_read": [],
        "public_artifact_contains_identifiers_or_paths": False,
        "thresholds": {
            "source_corner_tolerance_mm": CORNER_ATOL_MM,
            "in_plane_position_tolerance_mm": IN_PLANE_ATOL_MM,
            "safe_resample_max_direction_angle_degrees": RESAMPLE_MAX_ANGLE_DEG,
            "safe_resample_max_spacing_relative_error": RESAMPLE_MAX_SPACING_RELATIVE_ERROR,
            "safe_resample_min_valid_source_fraction": RESAMPLE_MIN_VALID_FRACTION,
            "safe_resample_boundary_mode": "reflect",
        },
        "counts": {
            "patients_total": len(patient_rows),
            "patients_eligible": sum(int(row["eligible"]) for row in patient_rows),
            "patients_ineligible": sum(not int(row["eligible"]) for row in patient_rows),
            "visits_total": len(visit_rows),
            "visits_pass": len(pass_rows),
            "visits_fail": len(failed_rows),
            "raw_pixel_cells_verified": sum(int(row["raw_pixel_cells_verified"]) for row in pass_rows),
            "visits_with_same_study_replacement": sum(
                row["selection_provenance"] == "same_study_original_primary_replacement" for row in pass_rows
            ),
            "visits_with_filtered_current_stack": sum(
                row["selection_provenance"] == "current_filtered_original_primary" for row in pass_rows
            ),
            "visits_with_safe_phase_resampling": sum(int(row["resampled_phase_count"]) > 0 for row in pass_rows),
            "resampled_phases": sum(int(row["resampled_phase_count"]) for row in pass_rows),
        },
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "blocked_current_source_reason_counts": dict(sorted(blocked_source_reasons.items())),
        "visit_status_counts": visit_status_counts,
        "patient_failing_visit_count_distribution": dict(
            sorted(failing_visit_distribution.items(), key=lambda item: int(item[0]))
        ),
        "selected_mode_counts": dict(sorted(mode_counts.items())),
        "selection_provenance_counts": dict(sorted(provenance_counts.items())),
        "phase_count_distribution": dict(sorted(phase_counts.items(), key=lambda item: int(item[0]))),
        "timing_source_counts": dict(sorted(timing_counts.items())),
        "orientation_counts": dict(sorted(orientation_counts.items())),
        "smoke_strata": dict(sorted(smoke_strata.items())),
        "smoke_case_outcomes": dict(sorted(smoke_outcomes.items())),
    }


def patient_results(visit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in visit_rows:
        grouped[str(row["patient_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for patient_id, visits in sorted(grouped.items()):
        by_visit = {str(row["visit"]): row for row in visits}
        eligible = set(by_visit) == set(VISITS) and all(by_visit[name]["status"] == "PASS" for name in VISITS)
        failures = [
            f"{name}:{by_visit[name]['failure_reason']}" if name in by_visit else f"{name}:NOT_RUN"
            for name in VISITS
            if name not in by_visit or by_visit[name]["status"] != "PASS"
        ]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "patient_id": patient_id,
                "patient_token": patient_token(patient_id),
                "eligible": int(eligible),
                "passing_visit_count": sum(row["status"] == "PASS" for row in visits),
                "failure_reasons_by_visit": json.dumps(failures),
            }
        )
    return rows


def phase_contract_rows(visit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "patient_id": row["patient_id"],
            "patient_token": row["patient_token"],
            "visit": row["visit"],
            "time_count": row["phase_count"],
            "timing_source": row["timing_source"],
            "relative_phase_seconds": row["relative_phase_seconds"],
            "pre_index": row["pre_index"],
            "early_index": row["early_index"],
            "late_index": row["late_index"],
            "index_rule": "T<=4:(0,1,T-1);T>4:(0,2,min(5,T-1))",
            "outcome_fields_read": "[]",
        }
        for row in visit_rows
        if row["status"] == "PASS"
    ]


def reason_rows(visit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in visit_rows:
        if row["status"] != "PASS":
            grouped[str(row["failure_reason"])].append(row)
    return [
        {
            "failure_reason": reason,
            "visit_count": len(rows),
            "patient_count": len({row["patient_token"] for row in rows}),
        }
        for reason, rows in sorted(grouped.items())
    ] or [{"failure_reason": "NONE", "visit_count": 0, "patient_count": 0}]


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.progress_every < 1:
        raise SystemExit("--workers and --progress-every must be positive")
    jobs = manifest_jobs(args.preprocessed_root)
    if not jobs:
        raise SystemExit("No complete four-visit imaging manifests found")
    smoke_strata: dict[str, int] = {}
    prefix = "ispy1_base_eligibility"
    if args.smoke:
        jobs, smoke_strata = choose_smoke_jobs(jobs, args.workers)
        prefix = "ispy1_base_eligibility_smoke"
    elif args.limit is not None:
        jobs = jobs[: args.limit]
        prefix = "ispy1_base_eligibility_limited"

    experiment_root = args.experiment_root.resolve()
    cache_root = experiment_root / "cache" / ("ispy1_validated_dce_smoke" if args.smoke else "ispy1_validated_dce")
    private_cell_root = experiment_root / "manifests" / f"{prefix}_cells.private"
    for job in jobs:
        job.update(
            {
                "cache_root": str(cache_root),
                "private_cell_root": str(private_cell_root),
                "overwrite": bool(args.overwrite),
            }
        )

    print(json.dumps({"event": "start", "visits": len(jobs), "mode": "smoke" if args.smoke else "full"}))
    visit_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(process_visit, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            visit_rows.append(future.result())
            if completed % args.progress_every == 0 or completed == len(futures):
                counts = Counter(row["status"] for row in visit_rows)
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "completed": completed,
                            "total": len(futures),
                            "pass": counts["PASS"],
                            "fail": counts["FAIL"],
                        }
                    )
                )
    visit_rows.sort(key=lambda row: (row["patient_token"], VISITS.index(str(row["visit"]))))
    patient_rows = patient_results(visit_rows)
    phase_rows = phase_contract_rows(visit_rows)
    summary = aggregate_results(visit_rows, patient_rows, smoke_strata)

    private_visit_path = experiment_root / "manifests" / f"{prefix}_visits.private.csv"
    private_patient_path = experiment_root / "manifests" / f"{prefix}_patients.private.csv"
    private_phase_path = experiment_root / "manifests" / f"{prefix}_phase_contract.private.csv"
    if args.smoke:
        public_root = experiment_root / "logs" / "ispy1_base_smoke"
    elif args.limit is not None:
        public_root = experiment_root / "logs" / "ispy1_base_limited_debug"
    else:
        public_root = experiment_root / "metrics"
    public_summary_path = public_root / f"{prefix}_summary.json"
    public_reason_path = public_root / f"{prefix}_failure_reasons.csv"
    write_csv_atomic(private_visit_path, visit_rows, private=True)
    write_csv_atomic(private_patient_path, patient_rows, private=True)
    write_csv_atomic(private_phase_path, phase_rows, private=True)
    write_csv_atomic(public_reason_path, reason_rows(visit_rows), private=False)
    summary["private_manifest_sha256"] = {
        "visits": sha256_file(private_visit_path),
        "patients": sha256_file(private_patient_path),
        "phase_contract": sha256_file(private_phase_path),
    }
    public_json_atomic(public_summary_path, summary)
    print(json.dumps({"event": "complete", "counts": summary["counts"], "failure_reason_counts": summary["failure_reason_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
