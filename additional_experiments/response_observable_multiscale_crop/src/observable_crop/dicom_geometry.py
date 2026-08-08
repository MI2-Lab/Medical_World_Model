"""Header-only DICOM geometry audit for the Stage A input contract.

This module intentionally never requests or decodes ``PixelData``.  It can
establish whether a DCE series describes a complete Cartesian space/time grid,
construct that grid's canonical RAS+ affine, and compare the physical volume
with a mask sform.  It cannot establish how an already converted NIfTI array's
voxels were ordered.  Consequently a mask-sform match is reported as a
geometry repair candidate, not as proof that changing a NIfTI header is safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import pydicom
except ImportError as exc:  # pragma: no cover - exercised only in a lean env
    raise ImportError(
        "observable_crop.dicom_geometry requires pydicom; use the project "
        "environment declared in requirements.txt"
    ) from exc


_HEADER_TAGS = (
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "Rows",
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "TemporalPositionIdentifier",
    "AcquisitionTime",
)

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class DicomAuditTolerances:
    """Numerical tolerances in DICOM patient coordinates (millimetres)."""

    iop_atol: float = 1e-5
    direction_norm_atol: float = 1e-5
    direction_dot_atol: float = 1e-5
    pixel_spacing_atol_mm: float = 1e-5
    slice_position_cluster_atol_mm: float = 1e-3
    slice_spacing_atol_mm: float = 1e-3
    in_plane_position_atol_mm: float = 0.1
    nifti_spacing_atol_mm: float = 1e-3
    affine_corner_atol_mm: float = 0.1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TimeGroupingAudit:
    """Completeness of one DICOM temporal grouping tag."""

    tag: str
    present_count: int
    header_count: int
    group_count: int
    expected_group_count: int | None
    expected_slice_count: int
    occupied_cells: int
    expected_cells: int | None
    missing_cells: int | None
    duplicate_cells: int
    complete: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DicomGeometryAudit:
    """Privacy-safe aggregate result of a header-only series audit."""

    schema_version: int
    headers_only: bool
    pixel_data_read: bool
    file_count: int
    readable_header_count: int
    required_header_complete_count: int
    unique_sop_instance_uid_count: int
    unique_series_instance_uid_count: int
    rows: int | None
    columns: int | None
    expected_shape_xyz_t: tuple[int, int, int, int] | None
    expected_file_count: int | None
    iop_max_abs_delta: float | None
    iop_orthonormal: bool
    pixel_spacing_row_col_mm: tuple[float, float] | None
    pixel_spacing_max_abs_delta_mm: float | None
    slice_count: int
    slice_spacing_mm: float | None
    slice_spacing_max_deviation_mm: float | None
    in_plane_position_max_deviation_mm: float | None
    temporal_position: TimeGroupingAudit
    acquisition_time: TimeGroupingAudit
    temporal_grouping_source: str | None
    temporal_groupings_agree: bool | None
    dicom_shape_xyz: tuple[int, int, int] | None
    dicom_spacing_xyz_mm: tuple[float, float, float] | None
    dicom_affine_ras: np.ndarray | None
    dicom_orientation_ras: str
    mask_shape_matches: bool | None
    mask_affine_valid: bool
    mask_center_corner_hausdorff_mm: float | None
    mask_footprint_corner_hausdorff_mm: float | None
    dce_sform_valid: bool
    dce_qform_valid: bool
    dce_sform_mask_index_corner_max_mm: float | None
    dce_qform_mask_index_corner_max_mm: float | None
    dce_sform_dicom_corner_hausdorff_mm: float | None
    dce_qform_dicom_corner_hausdorff_mm: float | None
    series_geometry_valid: bool
    mask_geometry_consistent: bool
    audit_pass: bool
    decision: str
    selected_affine_source: str | None
    selected_affine_ras: np.ndarray | None
    geometry_auto_repairable: bool
    header_only_safe: bool
    pixel_order_verified: bool
    recommended_action: str
    quarantine_reason_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]

    def to_record(self, *, include_affines: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable record with no file path or patient ID."""

        record = asdict(self)
        record["temporal_position"] = self.temporal_position.to_record()
        record["acquisition_time"] = self.acquisition_time.to_record()
        for key in ("dicom_affine_ras", "selected_affine_ras"):
            value = getattr(self, key)
            record[key] = (
                value.astype(float).tolist()
                if include_affines and value is not None
                else None
            )
        record["quarantine_reason_codes"] = list(self.quarantine_reason_codes)
        record["warning_codes"] = list(self.warning_codes)
        return record


@dataclass(frozen=True)
class _Header:
    sop_uid: str | None
    series_uid: str | None
    rows: int | None
    columns: int | None
    iop: np.ndarray | None
    ipp: np.ndarray | None
    pixel_spacing: np.ndarray | None
    temporal_position: str | None
    acquisition_time_seconds: float | None

    @property
    def required_complete(self) -> bool:
        return bool(
            self.rows is not None
            and self.columns is not None
            and self.iop is not None
            and self.ipp is not None
            and self.pixel_spacing is not None
        )


def _float_vector(value: Any, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        return None
    return result


def _integer(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _temporal_position(value: Any) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if not np.isfinite(numeric):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, ".12g")


def _dicom_time_seconds(value: Any) -> float | None:
    """Normalize DICOM TM to seconds after midnight without using dates."""

    if value is None:
        return None
    text = str(value).strip().replace(":", "")
    if not text:
        return None
    try:
        if "." in text:
            whole, fraction = text.split(".", 1)
            fractional_seconds = float(f"0.{fraction}")
        else:
            whole, fractional_seconds = text, 0.0
        whole = whole.ljust(6, "0")
        if len(whole) < 6:
            return None
        hours = int(whole[:2])
        minutes = int(whole[2:4])
        seconds = int(whole[4:6]) + fractional_seconds
    except (ValueError, OverflowError):
        return None
    if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds < 60):
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def _read_header(path: Path) -> _Header | None:
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=list(_HEADER_TAGS),
            force=False,
        )
    except Exception:  # pydicom exposes several parser-specific exception types
        return None
    return _Header(
        sop_uid=(
            str(dataset.SOPInstanceUID)
            if getattr(dataset, "SOPInstanceUID", None) is not None
            else None
        ),
        series_uid=(
            str(dataset.SeriesInstanceUID)
            if getattr(dataset, "SeriesInstanceUID", None) is not None
            else None
        ),
        rows=_integer(getattr(dataset, "Rows", None)),
        columns=_integer(getattr(dataset, "Columns", None)),
        iop=_float_vector(getattr(dataset, "ImageOrientationPatient", None), 6),
        ipp=_float_vector(getattr(dataset, "ImagePositionPatient", None), 3),
        pixel_spacing=_float_vector(getattr(dataset, "PixelSpacing", None), 2),
        temporal_position=_temporal_position(
            getattr(dataset, "TemporalPositionIdentifier", None)
        ),
        acquisition_time_seconds=_dicom_time_seconds(
            getattr(dataset, "AcquisitionTime", None)
        ),
    )


def _finite_affine(affine: np.ndarray | Sequence[Sequence[float]] | None) -> bool:
    if affine is None:
        return False
    array = np.asarray(affine, dtype=np.float64)
    return bool(
        array.shape == (4, 4)
        and np.all(np.isfinite(array))
        and np.linalg.matrix_rank(array[:3, :3]) == 3
        and np.allclose(array[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8)
    )


def _shape_xyz(values: Iterable[int] | None, name: str) -> tuple[int, int, int] | None:
    if values is None:
        return None
    shape = tuple(int(value) for value in values)
    if len(shape) < 3 or any(value <= 0 for value in shape[:3]):
        raise ValueError(f"{name} must contain at least three positive integers")
    return shape[:3]


def _shape_xyzt(values: Iterable[int] | None) -> tuple[int, int, int, int] | None:
    if values is None:
        return None
    shape = tuple(int(value) for value in values)
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError("expected_shape_xyz_t must contain four positive integers")
    return shape


def _spacing_xyz(values: Iterable[float] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    spacing = tuple(float(value) for value in values)
    if (
        len(spacing) != 3
        or not np.all(np.isfinite(spacing))
        or any(value <= 0 for value in spacing)
    ):
        raise ValueError("expected_spacing_xyz_mm must be a positive finite triplet")
    return spacing


def _cluster_scalars(values: np.ndarray, atol: float) -> tuple[np.ndarray, np.ndarray]:
    """Cluster sorted scalar positions and return centers plus input labels."""

    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite vector")
    if values.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    clusters: list[list[int]] = []
    for index in order:
        if not clusters:
            clusters.append([int(index)])
            continue
        center = float(np.mean(values[clusters[-1]]))
        if abs(float(values[index]) - center) <= atol:
            clusters[-1].append(int(index))
        else:
            clusters.append([int(index)])
    labels = np.empty(values.size, dtype=np.int64)
    centers = np.empty(len(clusters), dtype=np.float64)
    for label, members in enumerate(clusters):
        labels[members] = label
        centers[label] = float(np.mean(values[members]))
    return centers, labels


def _time_group_audit(
    *,
    tag: str,
    keys: Sequence[str | float | None],
    slice_labels: np.ndarray,
    expected_groups: int | None,
    n_slices: int,
) -> TimeGroupingAudit:
    present = [index for index, key in enumerate(keys) if key is not None]
    groups = sorted({keys[index] for index in present}, key=str)
    occupancy: dict[tuple[Any, int], int] = {}
    for index in present:
        cell = (keys[index], int(slice_labels[index]))
        occupancy[cell] = occupancy.get(cell, 0) + 1
    duplicates = sum(max(0, count - 1) for count in occupancy.values())
    expected_cells = expected_groups * n_slices if expected_groups is not None else None
    missing: int | None = None
    if expected_groups is not None and len(groups) == expected_groups:
        missing = sum(
            (group, slice_index) not in occupancy
            for group in groups
            for slice_index in range(n_slices)
        )
    complete = bool(
        len(present) == len(keys)
        and expected_groups is not None
        and len(groups) == expected_groups
        and missing == 0
        and duplicates == 0
        and len(occupancy) == expected_cells
    )
    return TimeGroupingAudit(
        tag=tag,
        present_count=len(present),
        header_count=len(keys),
        group_count=len(groups),
        expected_group_count=expected_groups,
        expected_slice_count=n_slices,
        occupied_cells=len(occupancy),
        expected_cells=expected_cells,
        missing_cells=missing,
        duplicate_cells=duplicates,
        complete=complete,
    )


def _temporal_groupings_agree(
    temporal: Sequence[str | None], acquisition: Sequence[float | None]
) -> bool | None:
    if any(value is None for value in temporal) or any(
        value is None for value in acquisition
    ):
        return None
    forward: dict[str, set[float]] = {}
    reverse: dict[float, set[str]] = {}
    for temporal_key, acquisition_key in zip(temporal, acquisition, strict=True):
        assert temporal_key is not None and acquisition_key is not None
        forward.setdefault(temporal_key, set()).add(acquisition_key)
        reverse.setdefault(acquisition_key, set()).add(temporal_key)
    return bool(
        all(len(values) == 1 for values in forward.values())
        and all(len(values) == 1 for values in reverse.values())
        and len(forward) == len(reverse)
    )


def _corner_points(
    affine: np.ndarray, shape_xyz: tuple[int, int, int], *, footprint: bool
) -> np.ndarray:
    if footprint:
        bounds = [(-0.5, float(length) - 0.5) for length in shape_xyz]
    else:
        bounds = [(0.0, float(length) - 1.0) for length in shape_xyz]
    index_corners = np.asarray(
        [(*corner, 1.0) for corner in product(*bounds)], dtype=np.float64
    )
    return (np.asarray(affine, dtype=np.float64) @ index_corners.T).T[:, :3]


def _corner_hausdorff_mm(
    left_affine: np.ndarray,
    left_shape: tuple[int, int, int],
    right_affine: np.ndarray,
    right_shape: tuple[int, int, int],
    *,
    footprint: bool,
) -> float:
    left = _corner_points(left_affine, left_shape, footprint=footprint)
    right = _corner_points(right_affine, right_shape, footprint=footprint)
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _index_corner_max_mm(
    left_affine: np.ndarray,
    right_affine: np.ndarray,
    shape_xyz: tuple[int, int, int],
) -> float:
    left = _corner_points(left_affine, shape_xyz, footprint=False)
    right = _corner_points(right_affine, shape_xyz, footprint=False)
    return float(np.linalg.norm(left - right, axis=1).max())


def _axis_codes(affine: np.ndarray | None) -> str:
    if affine is None:
        return "UNKNOWN"
    linear = affine[:3, :3]
    norms = np.linalg.norm(linear, axis=0)
    if np.any(norms <= 0):
        return "UNKNOWN"
    directions = linear / norms[None, :]
    world_axes = np.argmax(np.abs(directions), axis=0)
    if len(set(int(value) for value in world_axes)) != 3:
        return "OBLIQUE"
    labels = (("L", "R"), ("P", "A"), ("I", "S"))
    return "".join(
        labels[int(axis)][1 if directions[int(axis), column] >= 0 else 0]
        for column, axis in enumerate(world_axes)
    )


def audit_dicom_geometry(
    series_dir: str | Path,
    *,
    expected_shape_xyz_t: Iterable[int] | None,
    expected_spacing_xyz_mm: Iterable[float] | None,
    mask_affine_ras: np.ndarray | Sequence[Sequence[float]] | None,
    mask_shape_xyz: Iterable[int] | None,
    dce_sform_ras: np.ndarray | Sequence[Sequence[float]] | None = None,
    dce_qform_ras: np.ndarray | Sequence[Sequence[float]] | None = None,
    tolerances: DicomAuditTolerances | None = None,
) -> DicomGeometryAudit:
    """Audit one classic single-frame DCE DICOM series without reading pixels.

    ``expected_shape_xyz_t`` follows the NIfTI order ``X,Y,Z,T``.  The canonical
    DICOM affine maps ``(column,row,slice)`` indices to RAS+ millimetres, with
    slices sorted along the IOP cross-product normal.  Corner agreement is an
    unordered physical-volume comparison so that a valid dcm2niix row/slice
    flip does not look like a registration failure.
    """

    tolerances = tolerances or DicomAuditTolerances()
    expected_shape = _shape_xyzt(expected_shape_xyz_t)
    expected_spacing = _spacing_xyz(expected_spacing_xyz_mm)
    mask_shape = _shape_xyz(mask_shape_xyz, "mask_shape_xyz")
    root = Path(series_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("DICOM series input is not a directory")
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    headers = [header for path in files if (header := _read_header(path)) is not None]
    complete = [header for header in headers if header.required_complete]

    reasons: list[str] = []
    warnings: list[str] = []
    if not files:
        reasons.append("NO_DICOM_FILES")
    if len(headers) != len(files):
        reasons.append("UNREADABLE_DICOM_HEADER")
    if len(complete) != len(headers):
        reasons.append("MISSING_REQUIRED_DICOM_TAG")

    sop_uids = [header.sop_uid for header in headers if header.sop_uid]
    series_uids = [header.series_uid for header in headers if header.series_uid]
    unique_sop_count = len(set(sop_uids))
    unique_series_count = len(set(series_uids))
    if len(sop_uids) != len(headers) or unique_sop_count != len(headers):
        reasons.append("MISSING_OR_DUPLICATE_SOP_INSTANCE_UID")
    if len(series_uids) != len(headers) or unique_series_count != 1:
        reasons.append("MISSING_OR_MIXED_SERIES_INSTANCE_UID")

    row_values = sorted({header.rows for header in complete})
    column_values = sorted({header.columns for header in complete})
    rows = row_values[0] if len(row_values) == 1 else None
    columns = column_values[0] if len(column_values) == 1 else None
    if complete and (rows is None or columns is None):
        reasons.append("INCONSISTENT_ROWS_COLUMNS")
    if expected_shape is not None and (
        columns != expected_shape[0] or rows != expected_shape[1]
    ):
        reasons.append("DICOM_IN_PLANE_SHAPE_MISMATCH")

    reference_iop: np.ndarray | None = None
    iop_delta: float | None = None
    iop_orthonormal = False
    if complete:
        iops = np.stack([header.iop for header in complete])
        reference_iop = np.median(iops, axis=0)
        iop_delta = float(np.max(np.abs(iops - reference_iop)))
        if iop_delta > tolerances.iop_atol:
            reasons.append("INCONSISTENT_IMAGE_ORIENTATION_PATIENT")
        first = reference_iop[:3]
        second = reference_iop[3:]
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        iop_orthonormal = bool(
            abs(first_norm - 1.0) <= tolerances.direction_norm_atol
            and abs(second_norm - 1.0) <= tolerances.direction_norm_atol
            and abs(float(np.dot(first, second))) <= tolerances.direction_dot_atol
        )
        if not iop_orthonormal:
            reasons.append("NONORTHONORMAL_IMAGE_ORIENTATION_PATIENT")

    pixel_spacing: tuple[float, float] | None = None
    pixel_spacing_delta: float | None = None
    if complete:
        spacings = np.stack([header.pixel_spacing for header in complete])
        representative = np.median(spacings, axis=0)
        pixel_spacing_delta = float(np.max(np.abs(spacings - representative)))
        if pixel_spacing_delta > tolerances.pixel_spacing_atol_mm:
            reasons.append("INCONSISTENT_PIXEL_SPACING")
        if np.any(representative <= 0):
            reasons.append("INVALID_PIXEL_SPACING")
        else:
            pixel_spacing = (float(representative[0]), float(representative[1]))

    normal: np.ndarray | None = None
    slice_centers = np.empty(0, dtype=np.float64)
    slice_labels = np.empty(len(complete), dtype=np.int64)
    slice_spacing: float | None = None
    spacing_deviation: float | None = None
    in_plane_deviation: float | None = None
    dicom_affine: np.ndarray | None = None
    if complete and reference_iop is not None and iop_orthonormal:
        row_direction = reference_iop[:3]
        row_direction = row_direction / np.linalg.norm(row_direction)
        column_direction = reference_iop[3:]
        column_direction = column_direction / np.linalg.norm(column_direction)
        normal = np.cross(row_direction, column_direction)
        normal = normal / np.linalg.norm(normal)
        positions = np.stack([header.ipp for header in complete])
        projected = positions @ normal
        slice_centers, slice_labels = _cluster_scalars(
            projected, tolerances.slice_position_cluster_atol_mm
        )
        residuals = positions - projected[:, None] * normal[None, :]
        residual_center = np.median(residuals, axis=0)
        in_plane_deviation = float(
            np.linalg.norm(residuals - residual_center[None, :], axis=1).max()
        )
        if in_plane_deviation > tolerances.in_plane_position_atol_mm:
            reasons.append("IN_PLANE_IMAGE_POSITION_DRIFT")
        if slice_centers.size < 2:
            reasons.append("INSUFFICIENT_SLICE_POSITIONS")
        else:
            differences = np.diff(slice_centers)
            slice_spacing = float(np.median(differences))
            spacing_deviation = float(np.max(np.abs(differences - slice_spacing)))
            if (
                slice_spacing <= 0
                or spacing_deviation > tolerances.slice_spacing_atol_mm
            ):
                reasons.append("IRREGULAR_SLICE_SPACING")
            elif pixel_spacing is not None:
                affine_lps = np.eye(4, dtype=np.float64)
                # DICOM IOP first triplet advances image columns; PixelSpacing
                # is ordered (row spacing, column spacing).
                affine_lps[:3, 0] = row_direction * pixel_spacing[1]
                affine_lps[:3, 1] = column_direction * pixel_spacing[0]
                affine_lps[:3, 2] = normal * slice_spacing
                affine_lps[:3, 3] = residual_center + normal * slice_centers[0]
                dicom_affine = _LPS_TO_RAS @ affine_lps

    n_slices = int(slice_centers.size)
    expected_slices = expected_shape[2] if expected_shape is not None else n_slices
    expected_times: int | None = (
        expected_shape[3] if expected_shape is not None else None
    )
    if expected_shape is not None and n_slices != expected_slices:
        reasons.append("UNEXPECTED_SLICE_COUNT")
    expected_file_count = (
        expected_slices * expected_times if expected_times is not None else None
    )
    if expected_file_count is not None and len(files) != expected_file_count:
        reasons.append("UNEXPECTED_DICOM_FILE_COUNT")

    if expected_times is None and n_slices > 0 and len(complete) % n_slices == 0:
        expected_times = len(complete) // n_slices
    temporal_keys = [header.temporal_position for header in complete]
    acquisition_keys = [header.acquisition_time_seconds for header in complete]
    temporal_audit = _time_group_audit(
        tag="TemporalPositionIdentifier",
        keys=temporal_keys,
        slice_labels=slice_labels,
        expected_groups=expected_times,
        n_slices=n_slices,
    )
    acquisition_audit = _time_group_audit(
        tag="AcquisitionTime",
        keys=acquisition_keys,
        slice_labels=slice_labels,
        expected_groups=expected_times,
        n_slices=n_slices,
    )
    if temporal_audit.complete:
        temporal_source = "TemporalPositionIdentifier"
    elif acquisition_audit.complete:
        temporal_source = "AcquisitionTime"
    else:
        temporal_source = None
        reasons.append("INCOMPLETE_TEMPORAL_GROUPING")
    groupings_agree = _temporal_groupings_agree(temporal_keys, acquisition_keys)
    if temporal_audit.complete and acquisition_audit.complete and not groupings_agree:
        reasons.append("TEMPORAL_GROUPINGS_DISAGREE")
    elif temporal_audit.complete and not acquisition_audit.complete:
        warnings.append("ACQUISITION_TIME_NOT_VOLUME_GROUPED")
    elif acquisition_audit.complete and not temporal_audit.complete:
        warnings.append("TEMPORAL_POSITION_IDENTIFIER_NOT_VOLUME_GROUPED")

    dicom_shape = (
        (columns, rows, n_slices)
        if columns is not None and rows is not None and n_slices > 0
        else None
    )
    dicom_spacing = (
        (pixel_spacing[1], pixel_spacing[0], slice_spacing)
        if pixel_spacing is not None and slice_spacing is not None
        else None
    )
    if expected_spacing is not None and dicom_spacing is not None and not np.allclose(
        np.asarray(dicom_spacing),
        np.asarray(expected_spacing),
        rtol=0.0,
        atol=tolerances.nifti_spacing_atol_mm,
    ):
        reasons.append("DICOM_NIFTI_SPACING_MISMATCH")

    mask_valid = _finite_affine(mask_affine_ras)
    mask_array = np.asarray(mask_affine_ras, dtype=np.float64) if mask_valid else None
    if not mask_valid:
        reasons.append("MASK_AFFINE_INVALID_OR_MISSING")
    mask_shape_matches: bool | None = None
    if mask_shape is not None and dicom_shape is not None:
        mask_shape_matches = mask_shape == dicom_shape
        if not mask_shape_matches:
            reasons.append("MASK_DICOM_SHAPE_MISMATCH")

    mask_center_error: float | None = None
    mask_footprint_error: float | None = None
    if (
        dicom_affine is not None
        and dicom_shape is not None
        and mask_array is not None
        and mask_shape is not None
    ):
        mask_center_error = _corner_hausdorff_mm(
            dicom_affine,
            dicom_shape,
            mask_array,
            mask_shape,
            footprint=False,
        )
        mask_footprint_error = _corner_hausdorff_mm(
            dicom_affine,
            dicom_shape,
            mask_array,
            mask_shape,
            footprint=True,
        )
        if (
            mask_center_error > tolerances.affine_corner_atol_mm
            or mask_footprint_error > tolerances.affine_corner_atol_mm
        ):
            reasons.append("MASK_DICOM_CORNER_MISMATCH")

    dce_sform_valid = _finite_affine(dce_sform_ras)
    dce_qform_valid = _finite_affine(dce_qform_ras)
    dce_sform = (
        np.asarray(dce_sform_ras, dtype=np.float64) if dce_sform_valid else None
    )
    dce_qform = (
        np.asarray(dce_qform_ras, dtype=np.float64) if dce_qform_valid else None
    )

    def affine_errors(
        affine: np.ndarray | None,
    ) -> tuple[float | None, float | None]:
        index_error = None
        dicom_error = None
        if affine is not None and mask_array is not None and dicom_shape is not None:
            index_error = _index_corner_max_mm(affine, mask_array, dicom_shape)
        if affine is not None and dicom_affine is not None and dicom_shape is not None:
            dicom_error = _corner_hausdorff_mm(
                affine,
                dicom_shape,
                dicom_affine,
                dicom_shape,
                footprint=False,
            )
        return index_error, dicom_error

    sform_mask_error, sform_dicom_error = affine_errors(dce_sform)
    qform_mask_error, qform_dicom_error = affine_errors(dce_qform)

    non_affine_reasons = tuple(dict.fromkeys(reasons))
    series_reason_prefixes = (
        "NO_DICOM",
        "UNREADABLE",
        "MISSING_REQUIRED",
        "MISSING_OR_DUPLICATE",
        "MISSING_OR_MIXED",
        "INCONSISTENT",
        "NONORTHONORMAL",
        "INVALID_PIXEL",
        "IN_PLANE",
        "INSUFFICIENT",
        "IRREGULAR",
        "UNEXPECTED",
        "INCOMPLETE_TEMPORAL",
        "TEMPORAL_GROUPINGS",
        "DICOM_IN_PLANE",
        "DICOM_NIFTI",
    )
    series_valid = bool(
        dicom_affine is not None
        and not any(
            reason.startswith(series_reason_prefixes) for reason in non_affine_reasons
        )
    )
    mask_consistent = bool(
        mask_valid
        and mask_shape_matches
        and mask_center_error is not None
        and mask_footprint_error is not None
        and mask_center_error <= tolerances.affine_corner_atol_mm
        and mask_footprint_error <= tolerances.affine_corner_atol_mm
    )
    audit_pass = series_valid and mask_consistent

    sform_trusted = bool(
        audit_pass
        and dce_sform_valid
        and sform_mask_error is not None
        and sform_dicom_error is not None
        and sform_mask_error <= tolerances.affine_corner_atol_mm
        and sform_dicom_error <= tolerances.affine_corner_atol_mm
    )
    qform_trusted = bool(
        audit_pass
        and dce_qform_valid
        and qform_mask_error is not None
        and qform_dicom_error is not None
        and qform_mask_error <= tolerances.affine_corner_atol_mm
        and qform_dicom_error <= tolerances.affine_corner_atol_mm
    )
    if sform_trusted:
        decision = "TRUST_DCE_SFORM"
        selected_source = "dce_sform"
        selected_affine = dce_sform
        repairable = False
        header_only_safe = True
        action = "KEEP_DCE_SFORM"
    elif qform_trusted:
        decision = "TRUST_DCE_QFORM"
        selected_source = "dce_qform"
        selected_affine = dce_qform
        repairable = True
        header_only_safe = True
        action = "USE_DCE_QFORM_AS_VALIDATED_AFFINE"
    elif audit_pass:
        decision = "MASK_SFORM_GEOMETRY_CANDIDATE"
        selected_source = "mask_sform_authoritative_target"
        selected_affine = mask_array
        repairable = True
        header_only_safe = False
        action = "REBUILD_DICOM_PIXELS_THEN_USE_MASK_SFORM"
        warnings.append("NIFTI_PIXEL_ORDER_NOT_VERIFIED_BY_HEADER_AUDIT")
    else:
        decision = "QUARANTINE"
        selected_source = None
        selected_affine = None
        repairable = False
        header_only_safe = False
        action = "QUARANTINE_SERIES"

    return DicomGeometryAudit(
        schema_version=1,
        headers_only=True,
        pixel_data_read=False,
        file_count=len(files),
        readable_header_count=len(headers),
        required_header_complete_count=len(complete),
        unique_sop_instance_uid_count=unique_sop_count,
        unique_series_instance_uid_count=unique_series_count,
        rows=rows,
        columns=columns,
        expected_shape_xyz_t=expected_shape,
        expected_file_count=expected_file_count,
        iop_max_abs_delta=iop_delta,
        iop_orthonormal=iop_orthonormal,
        pixel_spacing_row_col_mm=pixel_spacing,
        pixel_spacing_max_abs_delta_mm=pixel_spacing_delta,
        slice_count=n_slices,
        slice_spacing_mm=slice_spacing,
        slice_spacing_max_deviation_mm=spacing_deviation,
        in_plane_position_max_deviation_mm=in_plane_deviation,
        temporal_position=temporal_audit,
        acquisition_time=acquisition_audit,
        temporal_grouping_source=temporal_source,
        temporal_groupings_agree=groupings_agree,
        dicom_shape_xyz=dicom_shape,
        dicom_spacing_xyz_mm=dicom_spacing,
        dicom_affine_ras=dicom_affine,
        dicom_orientation_ras=_axis_codes(dicom_affine),
        mask_shape_matches=mask_shape_matches,
        mask_affine_valid=mask_valid,
        mask_center_corner_hausdorff_mm=mask_center_error,
        mask_footprint_corner_hausdorff_mm=mask_footprint_error,
        dce_sform_valid=dce_sform_valid,
        dce_qform_valid=dce_qform_valid,
        dce_sform_mask_index_corner_max_mm=sform_mask_error,
        dce_qform_mask_index_corner_max_mm=qform_mask_error,
        dce_sform_dicom_corner_hausdorff_mm=sform_dicom_error,
        dce_qform_dicom_corner_hausdorff_mm=qform_dicom_error,
        series_geometry_valid=series_valid,
        mask_geometry_consistent=mask_consistent,
        audit_pass=audit_pass,
        decision=decision,
        selected_affine_source=selected_source,
        selected_affine_ras=selected_affine,
        geometry_auto_repairable=repairable,
        header_only_safe=header_only_safe,
        pixel_order_verified=False,
        recommended_action=action,
        quarantine_reason_codes=non_affine_reasons,
        warning_codes=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "DicomAuditTolerances",
    "DicomGeometryAudit",
    "TimeGroupingAudit",
    "audit_dicom_geometry",
]
