"""Strict raw-PixelData rebuild for classic single-frame DCE-MRI series.

The public API deliberately separates privacy-safe aggregate metrics from
optional private provenance.  A successful rebuild has independently checked
the header grid, decoded every DICOM cell, constructed an ``[X,Y,Z,T]``
float32 volume, and then re-read and re-compared every constructed cell against
the scaled raw pixel array.

This module supports classic single-frame DICOM only.  Enhanced/multiframe
DICOM must use a separately validated implementation rather than being
silently interpreted as this layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import product
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset


_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
_HEADER_TAGS = (
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "TemporalPositionIdentifier",
    "AcquisitionTime",
    "RescaleSlope",
    "RescaleIntercept",
)


class DicomPixelRebuildError(ValueError):
    """Fail-closed, privacy-safe rebuild error with a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class RebuildTolerances:
    """Numerical tolerances in DICOM patient coordinates (millimetres)."""

    iop_atol: float = 1e-5
    direction_norm_atol: float = 1e-5
    direction_dot_atol: float = 1e-5
    pixel_spacing_atol_mm: float = 1e-5
    slice_position_cluster_atol_mm: float = 1e-3
    slice_spacing_atol_mm: float = 1e-3
    in_plane_position_atol_mm: float = 0.1
    expected_spacing_atol_mm: float = 1e-3
    corner_atol_mm: float = 0.1
    nifti_affine_atol_mm: float = 1e-4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class NiftiWriteAudit:
    """Privacy-safe NIfTI write/read-back audit."""

    written: bool
    shape_xyzt: tuple[int, int, int, int]
    dtype: str
    qform_code: int
    sform_code: int
    qform_max_abs_error_mm: float
    sform_max_abs_error_mm: float
    selected_affine_max_abs_error_mm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PixelRebuildMetrics:
    """Aggregate metrics safe for reports that must not expose identifiers."""

    schema_version: int
    status: str
    classic_single_frame: bool
    pixel_data_read: bool
    pixel_rebuild_executed: bool
    pixel_order_verified: bool
    pixel_rebuild_ready: bool
    file_count: int
    unique_sop_instance_uid_count: int
    unique_series_instance_uid_count: int
    rows: int
    columns: int
    slice_count: int
    timepoint_count: int
    expected_cell_count: int
    missing_cell_count: int
    duplicate_cell_count: int
    decoded_cell_count: int
    verified_cell_count: int
    temporal_position_complete: bool
    acquisition_time_complete: bool
    temporal_groupings_agree: bool
    temporal_order_agrees: bool
    iop_max_abs_delta: float
    iop_orthonormal: bool
    pixel_spacing_max_abs_delta_mm: float
    slice_spacing_max_deviation_mm: float
    in_plane_position_max_deviation_mm: float
    volume_shape_xyzt: tuple[int, int, int, int]
    volume_dtype: str
    finite_fraction: float
    nonconstant: bool
    volume_min: float
    volume_max: float
    volume_std: float
    spacing_xyz_mm: tuple[float, float, float]
    orientation_ras: str
    rescale_slope_min: float
    rescale_slope_max: float
    rescale_intercept_min: float
    rescale_intercept_max: float
    cell_recomparison_max_abs_error: float
    reference_center_corner_hausdorff_mm: float
    reference_footprint_corner_hausdorff_mm: float
    corner_tolerance_mm: float
    nifti_write: NiftiWriteAudit | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DicomPixelRebuildResult:
    """In-memory rebuilt volume plus public and optional private provenance."""

    volume_xyzt: np.ndarray
    affine_ras: np.ndarray
    metrics: PixelRebuildMetrics
    private: Mapping[str, Any] | None = None

    def public_metrics(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with no paths, UIDs, or hashes."""

        return self.metrics.to_dict()


@dataclass(frozen=True)
class _HeaderCell:
    path: Path
    sop_instance_uid: str
    series_instance_uid: str
    rows: int
    columns: int
    iop: np.ndarray
    ipp: np.ndarray
    pixel_spacing_row_col_mm: np.ndarray
    temporal_position: int
    acquisition_time_seconds: Decimal
    rescale_slope: float
    rescale_intercept: float


def _error(code: str, message: str) -> DicomPixelRebuildError:
    return DicomPixelRebuildError(code, message)


def _finite_vector(value: Any, length: int, tag: str) -> np.ndarray:
    if value is None:
        raise _error("MISSING_REQUIRED_DICOM_TAG", f"{tag} is required")
    try:
        vector = np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _error("INVALID_DICOM_TAG", f"{tag} is not numeric") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise _error("INVALID_DICOM_TAG", f"{tag} must be a finite length-{length} vector")
    return vector


def _positive_integer(value: Any, tag: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise _error("INVALID_DICOM_TAG", f"{tag} must be a positive integer") from exc
    if numeric <= 0:
        raise _error("INVALID_DICOM_TAG", f"{tag} must be a positive integer")
    return numeric


def _temporal_position(value: Any) -> int:
    if value is None:
        raise _error(
            "MISSING_TEMPORAL_POSITION_IDENTIFIER",
            "TemporalPositionIdentifier is required for every file",
        )
    try:
        decimal = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise _error(
            "INVALID_TEMPORAL_POSITION_IDENTIFIER",
            "TemporalPositionIdentifier must be an integer",
        ) from exc
    integral = decimal.to_integral_value()
    if decimal != integral or integral < 0:
        raise _error(
            "INVALID_TEMPORAL_POSITION_IDENTIFIER",
            "TemporalPositionIdentifier must be a non-negative integer",
        )
    return int(integral)


def _acquisition_time_seconds(value: Any) -> Decimal:
    """Parse DICOM TM exactly enough to use it as a grouping key."""

    if value is None:
        raise _error(
            "MISSING_ACQUISITION_TIME", "AcquisitionTime is required for every file"
        )
    text = str(value).strip().replace(":", "")
    if not text:
        raise _error("INVALID_ACQUISITION_TIME", "AcquisitionTime is empty")
    if text.count(".") > 1:
        raise _error("INVALID_ACQUISITION_TIME", "AcquisitionTime has invalid syntax")
    whole, dot, fraction = text.partition(".")
    if len(whole) not in (2, 4, 6) or not whole.isdigit() or (fraction and not fraction.isdigit()):
        raise _error("INVALID_ACQUISITION_TIME", "AcquisitionTime has invalid TM syntax")
    padded = whole.ljust(6, "0")
    hours, minutes, seconds = int(padded[:2]), int(padded[2:4]), int(padded[4:6])
    if hours > 23 or minutes > 59 or seconds > 59:
        raise _error("INVALID_ACQUISITION_TIME", "AcquisitionTime is out of range")
    try:
        fractional = Decimal(f"0.{fraction}") if dot else Decimal(0)
    except InvalidOperation as exc:  # pragma: no cover - guarded by digit check
        raise _error("INVALID_ACQUISITION_TIME", "AcquisitionTime is invalid") from exc
    return Decimal(hours * 3600 + minutes * 60 + seconds) + fractional


def _scaling(dataset: Dataset) -> tuple[float, float]:
    try:
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    except (TypeError, ValueError) as exc:
        raise _error("INVALID_PIXEL_SCALING", "rescale slope/intercept must be numeric") from exc
    if not np.isfinite(slope) or slope == 0.0 or not np.isfinite(intercept):
        raise _error(
            "INVALID_PIXEL_SCALING",
            "rescale slope must be finite/nonzero and intercept finite",
        )
    return slope, intercept


def _read_header(path: Path) -> _HeaderCell:
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=list(_HEADER_TAGS),
            force=False,
        )
    except Exception as exc:
        raise _error("UNREADABLE_DICOM", "a source file is not a readable DICOM") from exc

    sop = str(getattr(dataset, "SOPInstanceUID", "")).strip()
    series = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
    if not sop or not series:
        raise _error(
            "MISSING_DICOM_UID", "SOPInstanceUID and SeriesInstanceUID are required"
        )
    number_of_frames = getattr(dataset, "NumberOfFrames", 1)
    try:
        number_of_frames = int(number_of_frames)
    except (TypeError, ValueError) as exc:
        raise _error("INVALID_NUMBER_OF_FRAMES", "NumberOfFrames must be an integer") from exc
    if number_of_frames != 1:
        raise _error(
            "MULTIFRAME_DICOM_NOT_SUPPORTED",
            "only classic single-frame DICOM is supported",
        )
    slope, intercept = _scaling(dataset)
    return _HeaderCell(
        path=path,
        sop_instance_uid=sop,
        series_instance_uid=series,
        rows=_positive_integer(getattr(dataset, "Rows", None), "Rows"),
        columns=_positive_integer(getattr(dataset, "Columns", None), "Columns"),
        iop=_finite_vector(
            getattr(dataset, "ImageOrientationPatient", None),
            6,
            "ImageOrientationPatient",
        ),
        ipp=_finite_vector(
            getattr(dataset, "ImagePositionPatient", None),
            3,
            "ImagePositionPatient",
        ),
        pixel_spacing_row_col_mm=_finite_vector(
            getattr(dataset, "PixelSpacing", None), 2, "PixelSpacing"
        ),
        temporal_position=_temporal_position(
            getattr(dataset, "TemporalPositionIdentifier", None)
        ),
        acquisition_time_seconds=_acquisition_time_seconds(
            getattr(dataset, "AcquisitionTime", None)
        ),
        rescale_slope=slope,
        rescale_intercept=intercept,
    )


def _shape4(values: Iterable[int], name: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be four positive integers") from exc
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(f"{name} must be four positive integers")
    return shape


def _shape3(values: Iterable[int], name: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be three positive integers") from exc
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"{name} must be three positive integers")
    return shape


def _reference_affine(value: Any) -> np.ndarray:
    affine = np.asarray(value, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.all(np.isfinite(affine))
        or np.linalg.matrix_rank(affine[:3, :3]) < 3
        or not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise ValueError("reference_affine_ras must be a finite nonsingular 4x4 affine")
    return affine


def _expected_spacing(values: Iterable[float] | None) -> tuple[float, float, float] | None:
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


def _cluster_positions(values: np.ndarray, atol: float) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise _error("INVALID_SLICE_POSITIONS", "slice positions must be finite")
    order = np.argsort(values, kind="stable")
    clusters: list[list[int]] = []
    for raw_index in order:
        index = int(raw_index)
        if not clusters:
            clusters.append([index])
            continue
        center = float(np.mean(values[clusters[-1]]))
        if abs(float(values[index]) - center) <= atol:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    centers = np.asarray(
        [float(np.mean(values[members])) for members in clusters], dtype=np.float64
    )
    labels = np.empty(values.size, dtype=np.int64)
    for label, members in enumerate(clusters):
        labels[members] = label
    return centers, labels


def _unwrap_acquisition_times(times: Sequence[Decimal]) -> tuple[float, ...]:
    """Require TPI order and AcquisitionTime order to agree, allowing midnight."""

    unwrapped: list[float] = []
    day_offset = 0.0
    for value in times:
        candidate = float(value) + day_offset
        if unwrapped and candidate <= unwrapped[-1]:
            if unwrapped[-1] - candidate > 12.0 * 3600.0:
                day_offset += 24.0 * 3600.0
                candidate = float(value) + day_offset
        if unwrapped and candidate <= unwrapped[-1]:
            raise _error(
                "TEMPORAL_ORDER_DISAGREEMENT",
                "AcquisitionTime order does not strictly agree with TemporalPositionIdentifier",
            )
        unwrapped.append(candidate)
    return tuple(unwrapped)


def _corner_points(
    affine: np.ndarray, shape_xyz: tuple[int, int, int], *, footprint: bool
) -> np.ndarray:
    bounds = (
        [(-0.5, float(length) - 0.5) for length in shape_xyz]
        if footprint
        else [(0.0, float(length) - 1.0) for length in shape_xyz]
    )
    indices = np.asarray([(*corner, 1.0) for corner in product(*bounds)], dtype=float)
    return (affine @ indices.T).T[:, :3]


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _read_full_dataset(cell: _HeaderCell) -> Dataset:
    try:
        dataset = pydicom.dcmread(cell.path, stop_before_pixels=False, force=False)
    except Exception as exc:
        raise _error("PIXELDATA_READ_FAILED", "a DICOM cell could not be re-read") from exc
    if str(getattr(dataset, "SOPInstanceUID", "")).strip() != cell.sop_instance_uid:
        raise _error("DICOM_CHANGED_DURING_REBUILD", "SOPInstanceUID changed during rebuild")
    if str(getattr(dataset, "SeriesInstanceUID", "")).strip() != cell.series_instance_uid:
        raise _error("DICOM_CHANGED_DURING_REBUILD", "SeriesInstanceUID changed during rebuild")
    return dataset


def _decoded_scaled_xy(dataset: Dataset, cell: _HeaderCell) -> np.ndarray:
    number_of_frames = getattr(dataset, "NumberOfFrames", 1)
    try:
        number_of_frames = int(number_of_frames)
    except (TypeError, ValueError) as exc:
        raise _error("INVALID_NUMBER_OF_FRAMES", "NumberOfFrames changed or is invalid") from exc
    if number_of_frames != 1:
        raise _error("MULTIFRAME_DICOM_NOT_SUPPORTED", "a source file is not single-frame")
    try:
        raw = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise _error("PIXELDATA_DECODE_FAILED", "a DICOM pixel array could not be decoded") from exc
    if raw.ndim != 2 or raw.shape != (cell.rows, cell.columns):
        raise _error(
            "PIXEL_ARRAY_DIMENSION_MISMATCH",
            "decoded pixels do not match DICOM Rows/Columns",
        )
    slope, intercept = _scaling(dataset)
    if slope != cell.rescale_slope or intercept != cell.rescale_intercept:
        raise _error("DICOM_CHANGED_DURING_REBUILD", "pixel scaling changed during rebuild")
    scaled = raw.astype(np.float64, copy=False) * slope + intercept
    if not np.all(np.isfinite(scaled)):
        raise _error("NONFINITE_SCALED_PIXELS", "scaled raw pixels contain non-finite values")
    scaled32 = scaled.astype(np.float32)
    if not np.all(np.isfinite(scaled32)):
        raise _error("FLOAT32_PIXEL_OVERFLOW", "scaled raw pixels overflow float32")
    return np.ascontiguousarray(scaled32.T)


def _write_nifti(
    volume_xyzt: np.ndarray,
    affine_ras: np.ndarray,
    output_path: Path,
    *,
    temporal_spacing_seconds: float,
    tolerance_mm: float,
    overwrite: bool,
) -> NiftiWriteAudit:
    target = output_path.expanduser().resolve()
    if not (target.name.endswith(".nii") or target.name.endswith(".nii.gz")):
        raise ValueError("output_nifti must end in .nii or .nii.gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError("output_nifti already exists; set overwrite_output=True")
    suffix = ".nii.gz" if target.name.endswith(".nii.gz") else ".nii"
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=suffix, dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        image = nib.Nifti1Image(
            np.asarray(volume_xyzt, dtype=np.float32), affine_ras, dtype=np.float32
        )
        image.set_qform(affine_ras, code=1)
        image.set_sform(affine_ras, code=1)
        image.header.set_xyzt_units("mm", "sec")
        spatial_zooms = tuple(
            float(value) for value in np.linalg.norm(affine_ras[:3, :3], axis=0)
        )
        image.header.set_zooms((*spatial_zooms, float(temporal_spacing_seconds)))
        image.header["descrip"] = b"C1B raw DICOM PixelData rebuild"
        nib.save(image, temporary)

        reloaded = nib.load(temporary)
        qform, qform_code = reloaded.get_qform(coded=True)
        sform, sform_code = reloaded.get_sform(coded=True)
        if tuple(int(value) for value in reloaded.shape) != tuple(volume_xyzt.shape):
            raise _error("NIFTI_ROUNDTRIP_SHAPE_MISMATCH", "saved NIfTI shape changed")
        if np.dtype(reloaded.get_data_dtype()) != np.dtype(np.float32):
            raise _error("NIFTI_ROUNDTRIP_DTYPE_MISMATCH", "saved NIfTI is not float32")
        if qform is None or int(qform_code) <= 0 or sform is None or int(sform_code) <= 0:
            raise _error("NIFTI_AFFINE_CODE_INVALID", "qform and sform must both be valid")
        q_error = float(np.max(np.abs(np.asarray(qform) - affine_ras)))
        s_error = float(np.max(np.abs(np.asarray(sform) - affine_ras)))
        selected_error = float(np.max(np.abs(np.asarray(reloaded.affine) - affine_ras)))
        if max(q_error, s_error, selected_error) > tolerance_mm:
            raise _error(
                "NIFTI_AFFINE_ROUNDTRIP_MISMATCH",
                "saved qform/sform changed beyond tolerance",
            )
        os.replace(temporary, target)
        return NiftiWriteAudit(
            written=True,
            shape_xyzt=tuple(int(value) for value in volume_xyzt.shape),
            dtype="float32",
            qform_code=int(qform_code),
            sform_code=int(sform_code),
            qform_max_abs_error_mm=q_error,
            sform_max_abs_error_mm=s_error,
            selected_affine_max_abs_error_mm=selected_error,
        )
    finally:
        temporary.unlink(missing_ok=True)


def rebuild_classic_dce_series(
    series_dir: str | Path,
    *,
    expected_shape_xyzt: Iterable[int],
    reference_affine_ras: Any,
    reference_shape_xyz: Iterable[int],
    expected_spacing_xyz_mm: Iterable[float] | None = None,
    output_nifti: str | Path | None = None,
    overwrite_output: bool = False,
    include_private: bool = False,
    tolerances: RebuildTolerances | None = None,
) -> DicomPixelRebuildResult:
    """Rebuild and verify one classic single-frame DCE series.

    Parameters use NIfTI spatial order: expected shape is ``[X,Y,Z,T]`` and the
    returned volume has that same order.  ``reference_affine_ras`` and
    ``reference_shape_xyz`` are required so a successful result always includes
    the physical-corner gate used by C1B Stage A.

    The default return contains no source path, DICOM UID, acquisition time, or
    hash.  Set ``include_private=True`` only for an access-controlled sidecar.
    """

    tolerances = tolerances or RebuildTolerances()
    expected_shape = _shape4(expected_shape_xyzt, "expected_shape_xyzt")
    reference_shape = _shape3(reference_shape_xyz, "reference_shape_xyz")
    reference_affine = _reference_affine(reference_affine_ras)
    expected_spacing = _expected_spacing(expected_spacing_xyz_mm)
    if reference_shape != expected_shape[:3]:
        raise ValueError("reference_shape_xyz must equal expected_shape_xyzt[:3]")
    if expected_shape[3] < 2:
        raise ValueError("a DCE rebuild requires at least two timepoints")

    root = Path(series_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("series_dir is not a directory")
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    if not files:
        raise _error("NO_DICOM_FILES", "series directory contains no files")
    headers = tuple(_read_header(path) for path in files)

    sop_uids = [cell.sop_instance_uid for cell in headers]
    if len(set(sop_uids)) != len(headers):
        raise _error("DUPLICATE_SOP_INSTANCE_UID", "SOPInstanceUID values must be unique")
    series_uids = {cell.series_instance_uid for cell in headers}
    if len(series_uids) != 1:
        raise _error("MIXED_SERIES_INSTANCE_UID", "exactly one SeriesInstanceUID is required")
    rows_values = {cell.rows for cell in headers}
    column_values = {cell.columns for cell in headers}
    if len(rows_values) != 1 or len(column_values) != 1:
        raise _error("INCONSISTENT_ROWS_COLUMNS", "Rows/Columns must agree in all files")
    rows, columns = next(iter(rows_values)), next(iter(column_values))

    iops = np.stack([cell.iop for cell in headers])
    reference_iop = np.median(iops, axis=0)
    iop_delta = float(np.max(np.abs(iops - reference_iop)))
    if iop_delta > tolerances.iop_atol:
        raise _error(
            "INCONSISTENT_IMAGE_ORIENTATION_PATIENT",
            "ImageOrientationPatient differs across files",
        )
    column_direction = reference_iop[:3]
    row_direction = reference_iop[3:]
    column_norm = float(np.linalg.norm(column_direction))
    row_norm = float(np.linalg.norm(row_direction))
    orthonormal = bool(
        abs(column_norm - 1.0) <= tolerances.direction_norm_atol
        and abs(row_norm - 1.0) <= tolerances.direction_norm_atol
        and abs(float(np.dot(column_direction, row_direction)))
        <= tolerances.direction_dot_atol
    )
    if not orthonormal:
        raise _error(
            "NONORTHONORMAL_IMAGE_ORIENTATION_PATIENT",
            "ImageOrientationPatient directions must be orthonormal",
        )
    column_direction = column_direction / column_norm
    row_direction = row_direction / row_norm
    normal = np.cross(column_direction, row_direction)
    normal = normal / np.linalg.norm(normal)

    pixel_spacings = np.stack([cell.pixel_spacing_row_col_mm for cell in headers])
    representative_pixel_spacing = np.median(pixel_spacings, axis=0)
    pixel_spacing_delta = float(
        np.max(np.abs(pixel_spacings - representative_pixel_spacing))
    )
    if pixel_spacing_delta > tolerances.pixel_spacing_atol_mm or np.any(
        representative_pixel_spacing <= 0
    ):
        raise _error("INCONSISTENT_PIXEL_SPACING", "PixelSpacing is invalid or inconsistent")

    positions = np.stack([cell.ipp for cell in headers])
    projected = positions @ normal
    slice_centers, slice_labels = _cluster_positions(
        projected, tolerances.slice_position_cluster_atol_mm
    )
    if len(slice_centers) < 2:
        raise _error("INSUFFICIENT_SLICE_POSITIONS", "at least two slices are required")
    slice_differences = np.diff(slice_centers)
    slice_spacing = float(np.median(slice_differences))
    slice_spacing_deviation = float(
        np.max(np.abs(slice_differences - slice_spacing))
    )
    if slice_spacing <= 0 or slice_spacing_deviation > tolerances.slice_spacing_atol_mm:
        raise _error("IRREGULAR_SLICE_SPACING", "slice positions are not a regular grid")
    residuals = positions - projected[:, None] * normal[None, :]
    residual_center = np.median(residuals, axis=0)
    in_plane_deviation = float(
        np.linalg.norm(residuals - residual_center[None, :], axis=1).max()
    )
    if in_plane_deviation > tolerances.in_plane_position_atol_mm:
        raise _error("IN_PLANE_IMAGE_POSITION_DRIFT", "IPP has in-plane drift")

    by_tpi: dict[int, set[Decimal]] = {}
    by_time: dict[Decimal, set[int]] = {}
    for cell in headers:
        by_tpi.setdefault(cell.temporal_position, set()).add(
            cell.acquisition_time_seconds
        )
        by_time.setdefault(cell.acquisition_time_seconds, set()).add(
            cell.temporal_position
        )
    if any(len(values) != 1 for values in by_tpi.values()) or any(
        len(values) != 1 for values in by_time.values()
    ):
        raise _error(
            "TEMPORAL_GROUPING_DISAGREEMENT",
            "TemporalPositionIdentifier and AcquisitionTime must define identical groups",
        )
    temporal_positions = tuple(sorted(by_tpi))
    acquisition_times = tuple(next(iter(by_tpi[value])) for value in temporal_positions)
    unwrapped_times = _unwrap_acquisition_times(acquisition_times)
    time_index = {value: index for index, value in enumerate(temporal_positions)}

    cells: dict[tuple[int, int], _HeaderCell] = {}
    duplicate_cells = 0
    for cell, slice_index_raw in zip(headers, slice_labels, strict=True):
        key = (cell.temporal_position, int(slice_index_raw))
        if key in cells:
            duplicate_cells += 1
        else:
            cells[key] = cell
    if duplicate_cells:
        raise _error(
            "DUPLICATE_TIME_SLICE_CELL",
            "more than one file occupies a (time,slice) cell",
        )
    expected_keys = {
        (temporal, slice_index)
        for temporal in temporal_positions
        for slice_index in range(len(slice_centers))
    }
    missing_keys = expected_keys - set(cells)
    if missing_keys:
        raise _error("MISSING_TIME_SLICE_CELL", "the DICOM time/slice grid is incomplete")

    detected_shape = (columns, rows, len(slice_centers), len(temporal_positions))
    if detected_shape != expected_shape:
        raise _error(
            "EXPECTED_DIMENSION_MISMATCH",
            "DICOM-derived [X,Y,Z,T] dimensions do not match expected dimensions",
        )
    if len(files) != int(np.prod(detected_shape[2:])):
        raise _error("UNEXPECTED_FILE_COUNT", "file count does not equal Z times T")

    affine_lps = np.eye(4, dtype=np.float64)
    # DICOM PixelSpacing is (row, column).  [X,Y] are [columns, rows].
    affine_lps[:3, 0] = column_direction * representative_pixel_spacing[1]
    affine_lps[:3, 1] = row_direction * representative_pixel_spacing[0]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = residual_center + normal * slice_centers[0]
    affine_ras = _LPS_TO_RAS @ affine_lps
    spacing_xyz = tuple(
        float(value) for value in np.linalg.norm(affine_ras[:3, :3], axis=0)
    )
    if expected_spacing is not None and not np.allclose(
        spacing_xyz,
        expected_spacing,
        rtol=0.0,
        atol=tolerances.expected_spacing_atol_mm,
    ):
        raise _error("EXPECTED_SPACING_MISMATCH", "DICOM spacing does not match expected spacing")

    center_corner_error = _corner_hausdorff_mm(
        affine_ras,
        detected_shape[:3],
        reference_affine,
        reference_shape,
        footprint=False,
    )
    footprint_corner_error = _corner_hausdorff_mm(
        affine_ras,
        detected_shape[:3],
        reference_affine,
        reference_shape,
        footprint=True,
    )
    if max(center_corner_error, footprint_corner_error) > tolerances.corner_atol_mm:
        raise _error(
            "REFERENCE_CORNER_MISMATCH",
            "rebuilt DICOM volume does not match reference physical corners",
        )

    volume = np.full(detected_shape, np.nan, dtype=np.float32)
    ordered_records: list[dict[str, Any]] = []
    for temporal in temporal_positions:
        for slice_index in range(len(slice_centers)):
            cell = cells[(temporal, slice_index)]
            decoded_xy = _decoded_scaled_xy(_read_full_dataset(cell), cell)
            target_time = time_index[temporal]
            volume[:, :, slice_index, target_time] = decoded_xy
            if include_private:
                ordered_records.append(
                    {
                        "time_index": target_time,
                        "slice_index": slice_index,
                        "temporal_position_identifier": temporal,
                        "acquisition_time_seconds": str(
                            cell.acquisition_time_seconds
                        ),
                        "sop_instance_uid": cell.sop_instance_uid,
                        "source_path": str(cell.path),
                        "source_file_sha256": _sha256_file(cell.path),
                        "scaled_pixel_sha256": _sha256_array(decoded_xy),
                    }
                )

    if tuple(volume.shape) != expected_shape:
        raise _error("CONSTRUCTED_DIMENSION_MISMATCH", "constructed volume shape is invalid")
    finite_fraction = float(np.mean(np.isfinite(volume)))
    if finite_fraction != 1.0:
        raise _error("NONFINITE_REBUILT_VOLUME", "rebuilt volume contains non-finite values")
    volume_min = float(np.min(volume))
    volume_max = float(np.max(volume))
    volume_std = float(np.std(volume, dtype=np.float64))
    nonconstant = bool(volume_max > volume_min and volume_std > 0.0)
    if not nonconstant:
        raise _error("CONSTANT_REBUILT_VOLUME", "rebuilt volume is constant")

    # Independent second PixelData pass: every constructed target cell must be
    # exactly equal to a freshly decoded and per-file-scaled raw pixel array.
    verified_cells = 0
    max_recomparison_error = 0.0
    for temporal in temporal_positions:
        for slice_index in range(len(slice_centers)):
            cell = cells[(temporal, slice_index)]
            expected_xy = _decoded_scaled_xy(_read_full_dataset(cell), cell)
            observed_xy = volume[:, :, slice_index, time_index[temporal]]
            difference = float(
                np.max(np.abs(observed_xy.astype(np.float64) - expected_xy.astype(np.float64)))
            )
            max_recomparison_error = max(max_recomparison_error, difference)
            if not np.array_equal(observed_xy, expected_xy):
                raise _error(
                    "PIXEL_CELL_RECOMPARISON_FAILED",
                    "a constructed cell differs from freshly decoded scaled PixelData",
                )
            verified_cells += 1
    expected_cells = len(expected_keys)
    pixel_order_verified = verified_cells == expected_cells
    if not pixel_order_verified:
        raise _error("PIXEL_ORDER_INCOMPLETE", "not every time/slice cell was verified")

    nifti_audit: NiftiWriteAudit | None = None
    output_path: Path | None = None
    if output_nifti is not None:
        output_path = Path(output_nifti)
        time_differences = np.diff(np.asarray(unwrapped_times, dtype=np.float64))
        temporal_spacing = (
            float(np.median(time_differences))
            if time_differences.size and np.all(time_differences > 0)
            else 1.0
        )
        nifti_audit = _write_nifti(
            volume,
            affine_ras,
            output_path,
            temporal_spacing_seconds=temporal_spacing,
            tolerance_mm=tolerances.nifti_affine_atol_mm,
            overwrite=overwrite_output,
        )

    slopes = np.asarray([cell.rescale_slope for cell in headers], dtype=float)
    intercepts = np.asarray([cell.rescale_intercept for cell in headers], dtype=float)
    orientation = "".join(str(value) for value in nib.aff2axcodes(affine_ras))
    metrics = PixelRebuildMetrics(
        schema_version=1,
        status="PASS",
        classic_single_frame=True,
        pixel_data_read=True,
        pixel_rebuild_executed=True,
        pixel_order_verified=pixel_order_verified,
        pixel_rebuild_ready=True,
        file_count=len(files),
        unique_sop_instance_uid_count=len(set(sop_uids)),
        unique_series_instance_uid_count=len(series_uids),
        rows=rows,
        columns=columns,
        slice_count=len(slice_centers),
        timepoint_count=len(temporal_positions),
        expected_cell_count=expected_cells,
        missing_cell_count=0,
        duplicate_cell_count=0,
        decoded_cell_count=expected_cells,
        verified_cell_count=verified_cells,
        temporal_position_complete=True,
        acquisition_time_complete=True,
        temporal_groupings_agree=True,
        temporal_order_agrees=True,
        iop_max_abs_delta=iop_delta,
        iop_orthonormal=orthonormal,
        pixel_spacing_max_abs_delta_mm=pixel_spacing_delta,
        slice_spacing_max_deviation_mm=slice_spacing_deviation,
        in_plane_position_max_deviation_mm=in_plane_deviation,
        volume_shape_xyzt=detected_shape,
        volume_dtype="float32",
        finite_fraction=finite_fraction,
        nonconstant=nonconstant,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_std=volume_std,
        spacing_xyz_mm=spacing_xyz,
        orientation_ras=orientation,
        rescale_slope_min=float(slopes.min()),
        rescale_slope_max=float(slopes.max()),
        rescale_intercept_min=float(intercepts.min()),
        rescale_intercept_max=float(intercepts.max()),
        cell_recomparison_max_abs_error=max_recomparison_error,
        reference_center_corner_hausdorff_mm=center_corner_error,
        reference_footprint_corner_hausdorff_mm=footprint_corner_error,
        corner_tolerance_mm=tolerances.corner_atol_mm,
        nifti_write=nifti_audit,
    )

    private: dict[str, Any] | None = None
    if include_private:
        private = {
            "source_series_directory": str(root),
            "series_instance_uid": next(iter(series_uids)),
            "ordered_temporal_position_identifiers": list(temporal_positions),
            "ordered_acquisition_times_seconds": [str(value) for value in acquisition_times],
            "ordered_cells": ordered_records,
            "rebuilt_volume_sha256": _sha256_array(volume),
            "output_nifti": str(output_path.expanduser().resolve())
            if output_path is not None
            else None,
            "output_nifti_sha256": _sha256_file(output_path.expanduser().resolve())
            if output_path is not None
            else None,
        }
    return DicomPixelRebuildResult(
        volume_xyzt=volume,
        affine_ras=affine_ras,
        metrics=metrics,
        private=private,
    )


__all__ = [
    "DicomPixelRebuildError",
    "DicomPixelRebuildResult",
    "NiftiWriteAudit",
    "PixelRebuildMetrics",
    "RebuildTolerances",
    "rebuild_classic_dce_series",
]
