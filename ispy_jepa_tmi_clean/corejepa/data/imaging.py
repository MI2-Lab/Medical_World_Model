from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .contracts import DCE8_CHANNELS, VISITS
from .nifti import read_dce_nifti, read_spatial_nifti
from .records import PatientRecord


def load_phase_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_csv(path)
    if "pid" not in frame:
        raise ValueError(f"BreastDCEDL metadata has no 'pid' column: {path}")
    return frame.set_index("pid").to_dict(orient="index")


def _safe_index(value: Any, default: int, n_phases: int) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = default
        return int(np.clip(round(float(value)), 0, max(n_phases - 1, 0)))
    except (TypeError, ValueError):
        return int(np.clip(default, 0, max(n_phases - 1, 0)))


def select_phase_indices(
    n_phases: int,
    metadata: dict[str, Any] | None,
    policy: str = "adaptive_early_late",
) -> tuple[int, int, int, tuple[int, ...]]:
    """Return ``pre, early, late, peak_window`` acquisition indices."""

    if n_phases < 1:
        raise ValueError("A DCE visit must contain at least one phase")
    metadata = metadata or {}
    pre = _safe_index(metadata.get("pre"), 0, n_phases)
    metadata_early = _safe_index(metadata.get("post_early"), min(2, n_phases - 1), n_phases)
    metadata_late = _safe_index(metadata.get("post_late"), min(5, n_phases - 1), n_phases)
    if policy == "adaptive_early_late":
        early = _safe_index(pre + 1, 1, n_phases) if n_phases <= 4 else metadata_early
        late = n_phases - 1 if n_phases <= 4 else metadata_late
    elif policy == "first_last":
        early, late = min(1, n_phases - 1), n_phases - 1
    elif policy == "breastdcedl":
        early, late = metadata_early, metadata_late
    else:
        raise ValueError(f"Unknown DCE phase policy: {policy}")
    peak_window = tuple(range(1, n_phases)) if n_phases > 1 else (pre,)
    return pre, early, late, peak_window


def crop_or_pad_cxyz(
    volume: np.ndarray,
    center_xyz: tuple[float, float, float],
    crop_size_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Crop ``[C,X,Y,Z]`` and return ``float32 [C,Z,Y,X]``."""

    size_z, size_y, size_x = crop_size_zyx
    output_xyz = np.zeros((volume.shape[0], size_x, size_y, size_z), dtype=np.float32)
    center = [int(round(value)) for value in center_xyz]
    starts = [center[0] - size_x // 2, center[1] - size_y // 2, center[2] - size_z // 2]
    sizes = [size_x, size_y, size_z]
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for axis, (start, size) in enumerate(zip(starts, sizes), start=1):
        source_start = max(start, 0)
        source_end = min(start + size, volume.shape[axis])
        if source_end <= source_start:
            return np.transpose(output_xyz, (0, 3, 2, 1))
        target_start = source_start - start
        source_slices.append(slice(source_start, source_end))
        target_slices.append(slice(target_start, target_start + source_end - source_start))
    output_xyz[(slice(None), *target_slices)] = volume[(slice(None), *source_slices)]
    return np.transpose(output_xyz, (0, 3, 2, 1)).astype(np.float32, copy=False)


def _bbox_mask(visit: dict[str, Any], shape_xyz: tuple[int, int, int]) -> np.ndarray | None:
    bbox = visit.get("bbox_nii_xyz_inclusive")
    if not bbox:
        return None
    mask = np.zeros(shape_xyz, dtype=bool)
    x0, x1 = max(0, int(bbox["x_min"])), min(shape_xyz[0], int(bbox["x_max"]) + 1)
    y0, y1 = max(0, int(bbox["y_min"])), min(shape_xyz[1], int(bbox["y_max"]) + 1)
    z0, z1 = max(0, int(bbox["z_min"])), min(shape_xyz[2], int(bbox["z_max"]) + 1)
    mask[x0:x1, y0:y1, z0:z1] = True
    return mask if mask.any() else None


def _automatic_enhancement_roi(dce: np.ndarray) -> np.ndarray:
    """Create an automatic localization ROI when no released ROI is available."""

    if dce.ndim == 3:
        dce = dce[..., None]
    if dce.shape[-1] <= 1:
        return np.ones(dce.shape[:3], dtype=bool)
    dce = dce.astype(np.float32, copy=False)
    pre = dce[..., 0]
    peak = dce[..., 1:].max(axis=-1)
    absolute = peak - pre
    relative = absolute / np.maximum(np.abs(pre), 1.0)
    finite_pre = pre[np.isfinite(pre)]
    positive_pre = finite_pre[finite_pre > 0]
    threshold = np.percentile(positive_pre, 10.0) if positive_pre.size >= 128 else np.percentile(finite_pre, 25.0)
    valid = (pre > threshold) & np.isfinite(relative) & np.isfinite(absolute) & (absolute > 0)
    values = relative[valid]
    if values.size < 128:
        return np.ones(dce.shape[:3], dtype=bool)
    candidate: np.ndarray | None = None
    for percentile in (99.7, 99.5, 99.0, 98.5):
        relative_threshold = max(float(np.percentile(values, percentile)), 0.03)
        absolute_threshold = max(float(np.percentile(absolute[valid], 85.0)), 1.0)
        test = valid & (relative >= relative_threshold) & (absolute >= absolute_threshold)
        if int(test.sum()) >= 32:
            candidate = test
            break
    if candidate is None:
        candidate = valid & (relative >= np.percentile(values, 99.0))
    labels, n_labels = ndi.label(candidate, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_labels < 1:
        return candidate
    best_label, best_score = 0, -np.inf
    for label in range(1, n_labels + 1):
        component = labels == label
        count = int(component.sum())
        if count < 16:
            continue
        positions = np.nonzero(component)
        dimensions = [int(axis.max() - axis.min() + 1) for axis in positions]
        bbox_volume = max(int(np.prod(dimensions)), 1)
        if bbox_volume / max(float(component.size), 1.0) > 0.05:
            continue
        score = float(np.nanmean(relative[component]) + 0.01 * np.nanmean(absolute[component]))
        score *= math.sqrt(count) * (0.25 + min(count / bbox_volume, 1.0))
        if score > best_score:
            best_label, best_score = label, score
    selected = labels == best_label if best_label else candidate
    return ndi.binary_dilation(selected, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)


def load_visit_roi(
    visit: dict[str, Any],
    dce: np.ndarray,
    automatic_fallback: bool,
    legacy_empty_ftv_full_field: bool = False,
) -> tuple[np.ndarray, str]:
    """Return ``binary [X,Y,Z]`` ROI and its provenance string."""

    shape_xyz = tuple(int(value) for value in dce.shape[:3])
    ftv_path = visit.get("ftv_mask_nifti")
    if ftv_path and Path(ftv_path).exists():
        try:
            ftv, _ = read_spatial_nifti(ftv_path)
            if tuple(ftv.shape[:3]) == shape_xyz and np.count_nonzero(ftv > 0) > 0:
                return ftv > 0, "ftv_inclusion_region"
            if tuple(ftv.shape[:3]) == shape_xyz and legacy_empty_ftv_full_field and not visit.get("bbox_nii_xyz_inclusive"):
                return np.ones(shape_xyz, dtype=bool), "legacy_full_field_empty_ftv"
        except (OSError, ValueError):
            pass
    bbox = _bbox_mask(visit, shape_xyz)
    if bbox is not None:
        return bbox, "released_bbox"
    if automatic_fallback:
        return _automatic_enhancement_roi(dce), "automatic_enhancement_roi"
    return np.ones(shape_xyz, dtype=bool), "full_field_fallback"


def _mask_center(mask: np.ndarray) -> tuple[float, float, float]:
    positions = np.nonzero(mask)
    if positions[0].size == 0:
        return tuple(float(length - 1) / 2.0 for length in mask.shape)
    return tuple(float(axis.mean()) for axis in positions)


def _paper_crop_center(
    visit: dict[str, Any],
    roi: np.ndarray,
    source: str,
) -> tuple[float, float, float]:
    """Reproduce bbox-center crops for released I-SPY2 ROIs."""

    bbox = visit.get("bbox_nii_xyz_inclusive")
    if source != "automatic_enhancement_roi" and bbox:
        return (
            0.5 * (float(bbox["x_min"]) + float(bbox["x_max"])),
            0.5 * (float(bbox["y_min"]) + float(bbox["y_max"])),
            0.5 * (float(bbox["z_min"]) + float(bbox["z_max"])),
        )
    if source == "legacy_full_field_empty_ftv":
        return tuple(float(length) / 2.0 for length in roi.shape)
    return _mask_center(roi)


def _project_center(
    center_xyz: tuple[float, float, float],
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    output = []
    for center, source_length, target_length in zip(center_xyz, source_shape, target_shape):
        fraction = center / max(source_length - 1, 1)
        output.append(float(np.clip(fraction * max(target_length - 1, 0), 0, max(target_length - 1, 0))))
    return tuple(output)


def _robust_normalize(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.0))
    clipped = np.clip(channel.astype(np.float32, copy=False), low, high)
    median = float(np.median(clipped))
    q1, q3 = np.percentile(clipped, (25.0, 75.0))
    scale = float((q3 - q1) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(clipped) + 1e-6)
    return np.clip((clipped - median) / scale, -5.0, 5.0).astype(np.float32)


def dce8_visit(
    dce: np.ndarray,
    roi: np.ndarray,
    center_xyz: tuple[float, float, float],
    crop_size_zyx: tuple[int, int, int],
    phase_metadata: dict[str, Any] | None,
    phase_policy: str,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Construct one visit tensor.

    Inputs:
        ``dce [X,Y,Z,T]``, ``roi [X,Y,Z]``.
    Outputs:
        ``x [8,Z,Y,X]`` and ``(pre, early, late)`` indices.
    """

    if dce.ndim == 3:
        dce = dce[..., None]
    dce = dce.astype(np.float32, copy=False)
    pre_index, early_index, late_index, peak_window = select_phase_indices(dce.shape[-1], phase_metadata, phase_policy)
    pre, early, late = dce[..., pre_index], dce[..., early_index], dce[..., late_index]
    peak = dce[..., peak_window].max(axis=-1)
    denominator = np.maximum(np.abs(pre), 1.0)
    image_channels = np.stack(
        (pre, early, late, early - pre, late - pre, (peak - pre) / denominator, (late - peak) / denominator),
        axis=0,
    )
    image_crop = crop_or_pad_cxyz(image_channels, center_xyz, crop_size_zyx)
    image_crop = np.stack([_robust_normalize(channel) for channel in image_crop], axis=0)
    roi_crop = crop_or_pad_cxyz(roi.astype(np.float32)[None], center_xyz, crop_size_zyx)
    return np.concatenate((image_crop, roi_crop), axis=0), (pre_index, early_index, late_index)


def mask_geometry(mask_zyx: np.ndarray) -> np.ndarray:
    """Map a cropped ROI ``[Z,Y,X]`` to a normalized 9-D lesion descriptor."""

    mask = np.asarray(mask_zyx > 0.5)
    shape = np.asarray(mask.shape, dtype=np.float32)
    positions = np.nonzero(mask)
    if positions[0].size == 0:
        return np.zeros(9, dtype=np.float32)
    minimum = np.asarray([axis.min() for axis in positions], dtype=np.float32)
    maximum = np.asarray([axis.max() + 1 for axis in positions], dtype=np.float32)
    dimensions = np.maximum(maximum - minimum, 1.0)
    count = float(positions[0].size)
    total = float(mask.size)
    bbox_volume = float(np.prod(dimensions))
    center = (minimum + maximum - 1.0) / 2.0
    center = (center / np.maximum(shape - 1.0, 1.0) - 0.5) * 2.0
    return np.asarray(
        (count / total, *(dimensions / shape), bbox_volume / total, count / bbox_volume, *center),
        dtype=np.float32,
    )


def build_patient_tensor(
    record: PatientRecord,
    cache_dir: str | Path,
    crop_size_zyx: tuple[int, int, int],
    phase_policy: str,
    phase_metadata: dict[str, Any] | None,
    automatic_roi_fallback: bool,
    minimum_roi_capture: float,
    legacy_empty_ftv_full_field: bool = True,
    overwrite: bool = False,
) -> Path:
    """Build and save one patient cache.

    Saved arrays:
        ``image [4,8,Z,Y,X]``, ``geometry [4,9]``,
        ``phase_indices [4,3]``, and ``roi_sources [4]``.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{record.patient_id}.npz"
    if output.exists() and not overwrite:
        return output
    manifest = json.loads(record.manifest_path.read_text())
    visits = {item["visit"]: item for item in manifest["visits"]}
    dce_by_visit: dict[str, np.ndarray] = {}
    roi_by_visit: dict[str, np.ndarray] = {}
    roi_sources: list[str] = []
    for visit_name in VISITS:
        dce, _ = read_dce_nifti(visits[visit_name]["dce_nifti"])
        dce_by_visit[visit_name] = dce
        roi, source = load_visit_roi(
            visits[visit_name],
            dce,
            automatic_roi_fallback,
            legacy_empty_ftv_full_field=legacy_empty_ftv_full_field,
        )
        roi_by_visit[visit_name] = roi
        roi_sources.append(source)
    t0_roi = roi_by_visit["T0"]
    t0_center = _paper_crop_center(visits["T0"], t0_roi, roi_sources[0])
    t0_shape = tuple(int(value) for value in t0_roi.shape)
    tensors: list[np.ndarray] = []
    phases: list[tuple[int, int, int]] = []
    geometry: list[np.ndarray] = []
    for visit_name, roi_source in zip(VISITS, roi_sources):
        dce, roi = dce_by_visit[visit_name], roi_by_visit[visit_name]
        center = _project_center(t0_center, t0_shape, tuple(int(value) for value in roi.shape))
        if roi_source == "automatic_enhancement_roi":
            captured = crop_or_pad_cxyz(roi.astype(np.float32)[None], center, crop_size_zyx).sum()
            if captured < 32 or captured / max(float(roi.sum()), 1.0) < minimum_roi_capture:
                center = _mask_center(roi)
        tensor, phase = dce8_visit(dce, roi, center, crop_size_zyx, phase_metadata, phase_policy)
        tensors.append(tensor)
        phases.append(phase)
        geometry.append(mask_geometry(tensor[-1]))
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        image=np.stack(tensors).astype(np.float32),
        geometry=np.stack(geometry).astype(np.float32),
        phase_indices=np.asarray(phases, dtype=np.int16),
        roi_sources=np.asarray(roi_sources),
        channel_names=np.asarray(DCE8_CHANNELS),
        patient_id=np.asarray(record.patient_id),
    )
    temporary.replace(output)
    return output
