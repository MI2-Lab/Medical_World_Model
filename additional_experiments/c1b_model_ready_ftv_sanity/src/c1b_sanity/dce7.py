"""Outcome-free DCE7 construction on a fixed physical C1B grid."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from .geometry import (
    CanonicalVolume,
    PhysicalGrid,
    input_from_output_affine,
)


DCE7_CHANNEL_NAMES: tuple[str, ...] = (
    "pre",
    "early",
    "late",
    "early_minus_pre",
    "late_minus_pre",
    "peak_relative_enhancement",
    "late_minus_peak_relative_enhancement",
)
ANTI_ALIAS_FACTOR_THRESHOLD = 1.5
PADDING_MODE = "reflect"
PHASE_METADATA_FIELDS: tuple[str, ...] = ("pre", "post_early", "post_late")


def normalized_phase_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    """Return the complete acquisition-only phase allowlist in canonical form."""

    supplied: Mapping[str, Any] = {} if metadata is None else metadata
    unexpected = set(supplied).difference(PHASE_METADATA_FIELDS)
    if unexpected:
        raise ValueError(
            f"production phase metadata contains forbidden fields: {sorted(unexpected)}"
        )
    output: dict[str, int | float | None] = {}
    for field in PHASE_METADATA_FIELDS:
        value = supplied.get(field)
        if value is None:
            output[field] = None
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"phase metadata {field!r} is not numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"phase metadata {field!r} is non-finite")
        output[field] = int(numeric) if numeric.is_integer() else numeric
    return output


def phase_metadata_sha256(metadata: Mapping[str, Any] | None) -> str:
    payload = normalized_phase_metadata(metadata)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"c1b-phase-metadata-v1\0" + encoded).hexdigest()


def _safe_index(value: Any, default: int, n_phases: int) -> int:
    """Legacy-compatible numeric rounding and clipping."""

    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = default
        return int(np.clip(round(float(value)), 0, max(n_phases - 1, 0)))
    except (TypeError, ValueError, OverflowError):
        return int(np.clip(default, 0, max(n_phases - 1, 0)))


@dataclass(frozen=True)
class PhaseSelection:
    pre: int
    early: int
    late: int
    peak_window: tuple[int, ...]

    @property
    def indices(self) -> tuple[int, int, int]:
        return self.pre, self.early, self.late


def select_phase_indices(
    n_phases: int,
    metadata: Mapping[str, Any] | None = None,
) -> PhaseSelection:
    """Apply the frozen legacy adaptive early/late policy.

    Only the acquisition fields ``pre``, ``post_early``, and ``post_late`` are
    read.  All other metadata—including outcomes, lesion measurements, or
    image-derived scores—is structurally unable to affect this function.
    """

    if n_phases < 1:
        raise ValueError("a DCE visit must contain at least one phase")
    acquisition: Mapping[str, Any] = {} if metadata is None else metadata
    pre = _safe_index(acquisition.get("pre"), 0, n_phases)
    metadata_early = _safe_index(
        acquisition.get("post_early"), min(2, n_phases - 1), n_phases
    )
    metadata_late = _safe_index(
        acquisition.get("post_late"), min(5, n_phases - 1), n_phases
    )
    if n_phases <= 4:
        early = _safe_index(pre + 1, 1, n_phases)
        late = n_phases - 1
    else:
        early = metadata_early
        late = metadata_late
    peak_window = tuple(range(1, n_phases)) if n_phases > 1 else (pre,)
    return PhaseSelection(pre=pre, early=early, late=late, peak_window=peak_window)


def construct_dce7_xyzt(
    dce_xyzt: np.ndarray,
    selection: PhaseSelection | None = None,
    *,
    phase_metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, PhaseSelection]:
    """Construct exact, unnormalised DCE7 channels as ``[7, X, Y, Z]``."""

    dce = np.asarray(dce_xyzt, dtype=np.float32)
    if dce.ndim == 3:
        dce = dce[..., None]
    if dce.ndim != 4:
        raise ValueError(f"DCE data must have shape [X,Y,Z,T], got {dce.shape}")
    if dce.shape[-1] < 1:
        raise ValueError("DCE data contains no phases")
    if not np.isfinite(dce).all():
        raise ValueError("DCE data contains non-finite values")
    chosen = selection or select_phase_indices(dce.shape[-1], phase_metadata)
    all_indices = (*chosen.indices, *chosen.peak_window)
    if any(index < 0 or index >= dce.shape[-1] for index in all_indices):
        raise ValueError("phase selection contains an out-of-range index")

    pre = dce[..., chosen.pre]
    early = dce[..., chosen.early]
    late = dce[..., chosen.late]
    peak = np.max(dce[..., chosen.peak_window], axis=-1)
    denominator = np.maximum(np.abs(pre), np.float32(1.0))
    channels = np.stack(
        (
            pre,
            early,
            late,
            early - pre,
            late - pre,
            (peak - pre) / denominator,
            (late - peak) / denominator,
        ),
        axis=0,
    )
    return np.ascontiguousarray(channels, dtype=np.float32), chosen


@dataclass(frozen=True)
class NormalizationStats:
    p01: np.ndarray
    p99: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    scale_source: tuple[str, ...]
    valid_voxels: int


def normalize_dce7(
    channels_czyx: np.ndarray,
    valid_source_mask_zyx: np.ndarray,
    *,
    output_clip: Sequence[float] = (-5.0, 5.0),
) -> tuple[np.ndarray, NormalizationStats]:
    """Normalize each post-crop channel using only valid-source voxels."""

    channels = np.asarray(channels_czyx, dtype=np.float32)
    valid = np.asarray(valid_source_mask_zyx, dtype=bool)
    if channels.ndim != 4 or channels.shape[0] != len(DCE7_CHANNEL_NAMES):
        raise ValueError(f"channels must have shape [7,Z,Y,X], got {channels.shape}")
    if valid.shape != channels.shape[1:]:
        raise ValueError(
            f"valid-source mask shape {valid.shape} does not match {channels.shape[1:]}"
        )
    if not np.isfinite(channels).all():
        raise ValueError("DCE7 channels contain non-finite values")
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 1:
        raise ValueError("target grid contains no valid source voxels")
    clip_min, clip_max = (float(value) for value in output_clip)
    if not np.isfinite((clip_min, clip_max)).all() or clip_min >= clip_max:
        raise ValueError(f"invalid output clip range {tuple(output_clip)}")

    output = np.empty_like(channels, dtype=np.float32)
    p01 = np.empty(len(DCE7_CHANNEL_NAMES), dtype=np.float32)
    p99 = np.empty(len(DCE7_CHANNEL_NAMES), dtype=np.float32)
    medians = np.empty(len(DCE7_CHANNEL_NAMES), dtype=np.float32)
    scales = np.empty(len(DCE7_CHANNEL_NAMES), dtype=np.float32)
    sources: list[str] = []
    for index, channel in enumerate(channels):
        valid_values = channel[valid]
        low, high = np.percentile(valid_values, (1.0, 99.0))
        clipped = np.clip(channel, low, high)
        clipped_valid = clipped[valid]
        median = float(np.median(clipped_valid))
        q1, q3 = np.percentile(clipped_valid, (25.0, 75.0))
        scale = float((q3 - q1) / 1.349)
        scale_source = "iqr_div_1.349"
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(clipped_valid))
            scale_source = "std_fallback"
        if not np.isfinite(scale) or scale < 1e-6:
            # A constant channel is valid.  Unit scale keeps it exactly zero
            # after centering without inventing cross-patient statistics.
            scale = 1.0
            scale_source = "unit_constant_fallback"
        output[index] = np.clip((clipped - median) / scale, clip_min, clip_max)
        p01[index], p99[index] = low, high
        medians[index], scales[index] = median, scale
        sources.append(scale_source)

    return np.ascontiguousarray(output, dtype=np.float32), NormalizationStats(
        p01=p01,
        p99=p99,
        median=medians,
        scale=scales,
        scale_source=tuple(sources),
        valid_voxels=valid_count,
    )


@dataclass(frozen=True)
class ResamplingAudit:
    input_from_output: np.ndarray
    source_spacing_xyz_mm: np.ndarray
    source_samples_per_output_axis: np.ndarray
    anti_alias_sigma_source_voxels: np.ndarray
    anti_alias_applied: bool
    interpolation: str = "linear"
    padding_mode: str = PADDING_MODE


def _valid_source_footprint_mask_xyz(
    mapping: np.ndarray,
    output_shape_xyz: tuple[int, int, int],
    source_shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    """Mark output centres inside the source voxel-footprint boundaries."""

    x = np.arange(output_shape_xyz[0], dtype=np.float64)[:, None, None]
    y = np.arange(output_shape_xyz[1], dtype=np.float64)[None, :, None]
    z = np.arange(output_shape_xyz[2], dtype=np.float64)[None, None, :]
    valid = np.ones(output_shape_xyz, dtype=bool)
    for source_axis, source_length in enumerate(source_shape_xyz):
        coordinate = (
            mapping[source_axis, 0] * x
            + mapping[source_axis, 1] * y
            + mapping[source_axis, 2] * z
            + mapping[source_axis, 3]
        )
        valid &= (coordinate >= -0.5) & (coordinate <= float(source_length) - 0.5)
    return valid


def resample_dce_to_grid(
    volume: CanonicalVolume,
    grid: PhysicalGrid,
    *,
    source_to_anchor_ras: np.ndarray | None = None,
    anti_alias_factor_threshold: float = ANTI_ALIAS_FACTOR_THRESHOLD,
    padding_mode: str = PADDING_MODE,
) -> tuple[np.ndarray, np.ndarray, ResamplingAudit]:
    """Resample every raw phase together in one physical interpolation pass.

    Returns ``(DCE [X,Y,Z,T], valid_source [Z,Y,X], audit)``.  Anti-aliasing is
    a source-domain prefilter; the subsequent affine interpolation is a single
    call over the complete 4-D intensity array and never interpolates phases.
    """

    dce = np.asarray(volume.data, dtype=np.float32)
    if dce.ndim == 3:
        dce = dce[..., None]
    if dce.ndim != 4 or dce.shape[-1] < 1:
        raise ValueError(f"DCE volume must have shape [X,Y,Z,T], got {dce.shape}")
    if not np.isfinite(dce).all():
        raise ValueError("DCE volume contains non-finite values")
    if padding_mode != PADDING_MODE:
        raise ValueError(f"C1B padding mode is frozen to {PADDING_MODE!r}")
    threshold = float(anti_alias_factor_threshold)
    if not np.isfinite(threshold) or threshold <= 1.0:
        raise ValueError("anti-alias threshold must be finite and greater than one")
    if threshold != ANTI_ALIAS_FACTOR_THRESHOLD:
        raise ValueError(
            f"C1B anti-alias threshold is frozen to {ANTI_ALIAS_FACTOR_THRESHOLD}"
        )

    mapping = input_from_output_affine(
        volume.affine_ras,
        grid,
        source_to_anchor_ras=source_to_anchor_ras,
    )
    source_spacing = np.linalg.norm(volume.affine_ras[:3, :3], axis=0)
    # Each row describes the target-grid step footprint measured along one
    # source voxel axis.  This remains meaningful when an external rigid hook
    # rotates the follow-up into the anchor frame.
    samples_per_output = np.linalg.norm(mapping[:3, :3], axis=1)
    sigma = np.zeros(3, dtype=np.float64)
    downsampled = samples_per_output > threshold
    sigma[downsampled] = 0.5 * np.sqrt(
        np.maximum(samples_per_output[downsampled] ** 2 - 1.0, 0.0)
    )
    filtered = dce
    if np.any(downsampled):
        filtered = ndi.gaussian_filter(
            dce,
            sigma=(*sigma.tolist(), 0.0),
            order=0,
            mode=PADDING_MODE,
        ).astype(np.float32, copy=False)

    # scipy maps output indices to input indices.  Extending the spatial map
    # with an identity phase axis performs one 4-D interpolation call while
    # guaranteeing no temporal interpolation or phase mixing.
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = mapping[:3, :3]
    offset = np.zeros(4, dtype=np.float64)
    offset[:3] = mapping[:3, 3]
    sampled = ndi.affine_transform(
        filtered,
        matrix=matrix,
        offset=offset,
        output_shape=(*grid.shape_xyz, dce.shape[-1]),
        order=1,
        mode=PADDING_MODE,
        prefilter=False,
    ).astype(np.float32, copy=False)
    valid_xyz = _valid_source_footprint_mask_xyz(
        mapping,
        grid.shape_xyz,
        tuple(int(value) for value in dce.shape[:3]),
    )
    valid_zyx = np.ascontiguousarray(valid_xyz.transpose(2, 1, 0))
    audit = ResamplingAudit(
        input_from_output=np.asarray(mapping, dtype=np.float64),
        source_spacing_xyz_mm=np.asarray(source_spacing, dtype=np.float64),
        source_samples_per_output_axis=np.asarray(samples_per_output, dtype=np.float64),
        anti_alias_sigma_source_voxels=np.asarray(sigma, dtype=np.float64),
        anti_alias_applied=bool(np.any(downsampled)),
    )
    return np.ascontiguousarray(sampled), valid_zyx, audit


@dataclass(frozen=True)
class VisitDCE7:
    """One model tensor plus strictly separate, non-model sidecars."""

    tensor_czyx: np.ndarray
    valid_source_mask_zyx: np.ndarray
    phase_selection: PhaseSelection
    phase_count: int
    normalization: NormalizationStats
    grid: PhysicalGrid
    resampling: ResamplingAudit

    def __post_init__(self) -> None:
        expected = (len(DCE7_CHANNEL_NAMES), *self.grid.shape_zyx)
        if self.tensor_czyx.shape != expected or self.tensor_czyx.dtype != np.float32:
            raise ValueError(
                f"model tensor must be float32 with shape {expected}, got "
                f"{self.tensor_czyx.shape}/{self.tensor_czyx.dtype}"
            )
        if self.valid_source_mask_zyx.shape != self.grid.shape_zyx:
            raise ValueError("valid-source sidecar does not match the physical grid")


def build_visit_dce7(
    volume: CanonicalVolume,
    grid: PhysicalGrid,
    *,
    phase_metadata: Mapping[str, Any] | None = None,
    source_to_anchor_ras: np.ndarray | None = None,
) -> VisitDCE7:
    """Build the frozen seven-channel model input for one visit."""

    raw_phase_count = 1 if volume.data.ndim == 3 else int(volume.data.shape[-1])
    # Freeze the outcome-free acquisition choice before any physical image
    # processing.  All raw phases are still resampled together because the
    # frozen peak window spans phases 1..last.
    selection = select_phase_indices(raw_phase_count, phase_metadata)
    resampled_xyzt, valid_zyx, audit = resample_dce_to_grid(
        volume,
        grid,
        source_to_anchor_ras=source_to_anchor_ras,
    )
    channels_cxyz, selection = construct_dce7_xyzt(
        resampled_xyzt,
        selection=selection,
    )
    channels_czyx = np.ascontiguousarray(channels_cxyz.transpose(0, 3, 2, 1))
    normalized, stats = normalize_dce7(channels_czyx, valid_zyx)
    return VisitDCE7(
        tensor_czyx=normalized,
        valid_source_mask_zyx=valid_zyx,
        phase_selection=selection,
        phase_count=int(resampled_xyzt.shape[-1]),
        normalization=stats,
        grid=grid,
        resampling=audit,
    )
