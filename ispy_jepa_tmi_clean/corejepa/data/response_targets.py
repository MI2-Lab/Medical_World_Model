from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .condition import ConditionEncoder, ROUTING_FAMILIES
from .contracts import RESPONSE_VECTOR_FEATURES, VISITS
from .imaging import load_visit_roi, select_phase_indices
from .nifti import read_dce_nifti
from .records import PatientRecord


STATISTICS = ("mean", "std", "p10", "p25", "p50", "p75", "p90", "iqr", "positive_fraction")
KINETIC_MAPS = (
    "pre_raw",
    "early_absolute",
    "peak_absolute",
    "late_absolute",
    "washout_absolute",
    "early_relative",
    "peak_relative",
    "late_relative",
    "washout_relative",
    "auc_relative",
    "time_to_peak",
)
SHAPE_FEATURES = (
    "shape_voxels",
    "shape_volume_mm3",
    "shape_bbox_dx",
    "shape_bbox_dy",
    "shape_bbox_dz",
    "shape_bbox_volume",
    "shape_bbox_fill",
)


def response_feature_names() -> list[str]:
    names = list(SHAPE_FEATURES)
    for map_name in KINETIC_MAPS:
        names.extend(f"{map_name}_{statistic}" for statistic in STATISTICS)
    return names


def _statistics(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan")] * len(STATISTICS)
    p10, p25, p50, p75, p90 = np.percentile(values, (10, 25, 50, 75, 90))
    return [
        float(values.mean()),
        float(values.std()),
        float(p10),
        float(p25),
        float(p50),
        float(p75),
        float(p90),
        float(p75 - p25),
        float((values > 0).mean()),
    ]


def extract_visit_response_features(
    visit: dict[str, Any],
    automatic_roi_fallback: bool,
    legacy_empty_ftv_full_field: bool,
    phase_metadata: dict[str, Any] | None,
    phase_policy: str,
) -> tuple[np.ndarray, str]:
    """Extract 106 pCR-free descriptors from one original DCE visit."""

    dce, metadata = read_dce_nifti(visit["dce_nifti"])
    if dce.ndim == 3:
        dce = dce[..., None]
    dce = dce.astype(np.float32, copy=False)
    roi, source = load_visit_roi(
        visit,
        dce,
        automatic_roi_fallback,
        legacy_empty_ftv_full_field=legacy_empty_ftv_full_field,
    )
    positions = np.nonzero(roi)
    if positions[0].size == 0:
        return np.full(len(response_feature_names()), np.nan, dtype=np.float32), "empty"
    minimum = [int(axis.min()) for axis in positions]
    maximum = [int(axis.max() + 1) for axis in positions]
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    count = int(roi.sum())
    bbox_volume = max(int(np.prod(dimensions)), 1)
    spacing = metadata.get("pixdim", [1.0, 1.0, 1.0, 1.0])
    voxel_volume = abs(float(spacing[1]) * float(spacing[2]) * float(spacing[3]))
    features: list[float] = [
        float(count),
        float(count * voxel_volume),
        float(dimensions[0]),
        float(dimensions[1]),
        float(dimensions[2]),
        float(bbox_volume),
        float(count / bbox_volume),
    ]
    pre_index, early_index, late_index, peak_window = select_phase_indices(
        dce.shape[-1], phase_metadata, phase_policy
    )
    phase_values = dce[roi].astype(np.float32, copy=False)
    pre = phase_values[:, pre_index]
    early = phase_values[:, early_index]
    late = phase_values[:, late_index]
    window_indices = np.asarray(peak_window, dtype=np.int64)
    post = phase_values[:, window_indices]
    peak_offset = post.argmax(axis=1)
    peak = post[np.arange(post.shape[0]), peak_offset]
    denominator = np.maximum(np.abs(pre), 1.0)
    relative_curve = (post - pre[:, None]) / denominator[:, None]
    maps = {
        "pre_raw": pre,
        "early_absolute": early - pre,
        "peak_absolute": peak - pre,
        "late_absolute": late - pre,
        "washout_absolute": late - peak,
        "early_relative": (early - pre) / denominator,
        "peak_relative": (peak - pre) / denominator,
        "late_relative": (late - pre) / denominator,
        "washout_relative": (late - peak) / denominator,
        "auc_relative": relative_curve.mean(axis=1),
        "time_to_peak": window_indices[peak_offset].astype(np.float32),
    }
    for map_name in KINETIC_MAPS:
        features.extend(_statistics(maps[map_name]))
    return np.asarray(features, dtype=np.float32), source


def build_response_feature_cache(
    records: list[PatientRecord],
    output_path: str | Path,
    automatic_roi_fallback: bool = True,
    legacy_empty_ftv_full_field: bool = True,
    phase_metadata: dict[str, dict[str, Any]] | None = None,
    phase_policy: str = "breastdcedl",
    overwrite: bool = False,
) -> Path:
    """Save raw response descriptors as ``x_visit [N,4,106]``."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    rows: list[np.ndarray] = []
    sources: list[list[str]] = []
    names = response_feature_names()
    name_to_index = {name: index for index, name in enumerate(names)}
    for record in records:
        patient_rows, patient_sources = [], []
        if record.cohort.lower() == "ispy1" and record.longest_diameter is not None:
            diameter = np.asarray(record.longest_diameter, dtype=np.float32)
            diameter = np.where(np.isfinite(diameter) & (diameter > 0), diameter, np.nan)
            volume = diameter**3
            for visit_index in range(4):
                features = np.full(len(names), np.nan, dtype=np.float32)
                for name in ("shape_voxels", "shape_volume_mm3", "shape_bbox_volume"):
                    features[name_to_index[name]] = volume[visit_index]
                for name in ("shape_bbox_dx", "shape_bbox_dy", "shape_bbox_dz"):
                    features[name_to_index[name]] = diameter[visit_index]
                features[name_to_index["shape_bbox_fill"]] = 1.0
                patient_rows.append(features)
                patient_sources.append("released_longest_diameter_proxy")
        else:
            manifest = json.loads(record.manifest_path.read_text())
            visits = {item["visit"]: item for item in manifest["visits"]}
            for visit_name in VISITS:
                features, source = extract_visit_response_features(
                    visits[visit_name],
                    automatic_roi_fallback,
                    legacy_empty_ftv_full_field,
                    (phase_metadata or {}).get(record.patient_id),
                    phase_policy,
                )
                patient_rows.append(features)
                patient_sources.append(source)
        rows.append(np.stack(patient_rows))
        sources.append(patient_sources)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_visit=np.stack(rows).astype(np.float32),
        patient_ids=np.asarray([record.patient_id for record in records]),
        feature_names=np.asarray(names),
        roi_sources=np.asarray(sources),
    )
    return output_path


def _column(names: list[str], name: str) -> int:
    legacy_aliases = {
        "early_relative_mean": "early_rel_mean",
        "peak_relative_mean": "peak_rel_mean",
        "washout_relative_mean": "washout_rel_mean",
        "auc_relative_mean": "auc_rel_mean",
    }
    try:
        return names.index(name)
    except ValueError as error:
        alias = legacy_aliases.get(name)
        if alias is not None and alias in names:
            return names.index(alias)
        raise ValueError(f"Response cache is missing required feature: {name}") from error


def _log_ratio(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(value, 0.0) + 1.0) - np.log(np.maximum(reference, 0.0) + 1.0)


def response_vector(x_visit: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Create raw ``v [N,3,18]`` targets for T1, T2, and T3."""

    def values(name: str) -> np.ndarray:
        return x_visit[:, :, _column(feature_names, name)].astype(np.float32)

    blocks: list[np.ndarray] = []

    def add_logratio(value: np.ndarray) -> None:
        blocks.extend((_log_ratio(value[:, 1:], value[:, :1]), _log_ratio(value[:, 1:], value[:, :-1])))

    def add_delta(value: np.ndarray) -> None:
        blocks.extend((value[:, 1:] - value[:, :1], value[:, 1:] - value[:, :-1]))

    add_logratio(values("shape_volume_mm3"))
    add_logratio(values("shape_bbox_volume"))
    bbox_dimensions = np.stack(
        (values("shape_bbox_dx"), values("shape_bbox_dy"), values("shape_bbox_dz")),
        axis=-1,
    )
    longest_diameter = np.full(bbox_dimensions.shape[:2], np.nan, dtype=np.float32)
    has_dimension = np.isfinite(bbox_dimensions).any(axis=-1)
    longest_diameter[has_dimension] = np.nanmax(bbox_dimensions[has_dimension], axis=-1)
    add_logratio(longest_diameter)
    add_delta(values("shape_bbox_fill"))
    for name in (
        "early_relative_mean",
        "peak_relative_mean",
        "washout_relative_mean",
        "auc_relative_mean",
        "time_to_peak_mean",
    ):
        add_delta(values(name))
    return np.stack(blocks, axis=-1).astype(np.float32)


@dataclass
class ResponseTargetTransform:
    """Training-split transform for pCR-free response targets."""

    vector_median: np.ndarray
    vector_mean: np.ndarray
    vector_std: np.ndarray
    score_median: np.ndarray
    score_low: np.ndarray
    score_high: np.ndarray
    score_mean: np.ndarray
    score_std: np.ndarray
    family_mean: np.ndarray
    family_std: np.ndarray

    @classmethod
    def fit(
        cls,
        raw_vector: np.ndarray,
        records: list[PatientRecord],
        train_indices: list[int],
    ) -> "ResponseTargetTransform":
        train = raw_vector[np.asarray(train_indices)].reshape(-1, raw_vector.shape[-1])
        vector_median = np.nanmedian(train, axis=0)
        vector_median = np.where(np.isfinite(vector_median), vector_median, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(raw_vector), raw_vector, vector_median[None, None])
        train_filled = filled[np.asarray(train_indices)].reshape(-1, raw_vector.shape[-1])
        vector_mean = train_filled.mean(axis=0).astype(np.float32)
        vector_std = train_filled.std(axis=0).astype(np.float32)
        vector_std = np.where(vector_std > 1e-6, vector_std, 1.0).astype(np.float32)

        score_columns = np.asarray((0, 2, 4, 8, 10, 14), dtype=np.int64)
        raw_score_features = raw_vector[:, :, score_columns]
        score_train = raw_score_features[np.asarray(train_indices)].reshape(-1, len(score_columns))
        score_median = np.nanmedian(score_train, axis=0)
        score_median = np.where(np.isfinite(score_median), score_median, 0.0).astype(np.float32)
        score_filled = np.where(np.isfinite(raw_score_features), raw_score_features, score_median[None, None])
        score_train = score_filled[np.asarray(train_indices)].reshape(-1, len(score_columns))
        score_low, score_high = np.percentile(score_train, (2.0, 98.0), axis=0).astype(np.float32)
        clipped = np.clip(score_filled, score_low[None, None], score_high[None, None])
        clipped_train = clipped[np.asarray(train_indices)].reshape(-1, len(score_columns))
        score_mean = clipped_train.mean(axis=0).astype(np.float32)
        score_std = clipped_train.std(axis=0).astype(np.float32)
        score_std = np.where(score_std > 1e-6, score_std, 1.0).astype(np.float32)
        global_score = -((clipped - score_mean[None, None]) / score_std[None, None]).mean(axis=-1)
        family_mean = np.zeros(len(ROUTING_FAMILIES), dtype=np.float32)
        family_std = np.ones(len(ROUTING_FAMILIES), dtype=np.float32)
        targets = np.asarray([ConditionEncoder.routing_target(record) for record in records], dtype=np.int64)
        global_train = global_score[np.asarray(train_indices)].reshape(-1)
        global_mean, global_std = float(global_train.mean()), float(global_train.std())
        global_std = global_std if global_std > 1e-6 else 1.0
        for family in range(len(ROUTING_FAMILIES)):
            rows = np.asarray(train_indices)[targets[np.asarray(train_indices)] == family]
            values = global_score[rows].reshape(-1)
            if values.size >= 6:
                family_mean[family] = float(values.mean())
                family_std[family] = max(float(values.std()), 1e-6)
            else:
                family_mean[family], family_std[family] = global_mean, global_std
        return cls(
            vector_median,
            vector_mean,
            vector_std,
            score_median,
            score_low,
            score_high,
            score_mean,
            score_std,
            family_mean,
            family_std,
        )

    def transform(
        self,
        raw_vector: np.ndarray,
        records: list[PatientRecord],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``response_vector [N,3,18]`` and ``score [N,3,1]``."""

        filled = np.where(np.isfinite(raw_vector), raw_vector, self.vector_median[None, None])
        vector = np.clip(
            (filled - self.vector_mean[None, None]) / self.vector_std[None, None],
            -8.0,
            8.0,
        ).astype(np.float32)
        score_columns = np.asarray((0, 2, 4, 8, 10, 14), dtype=np.int64)
        score_features = raw_vector[:, :, score_columns]
        score_features = np.where(np.isfinite(score_features), score_features, self.score_median[None, None])
        score_features = np.clip(score_features, self.score_low[None, None], self.score_high[None, None])
        score = -((score_features - self.score_mean[None, None]) / self.score_std[None, None]).mean(axis=-1)
        family = np.asarray([ConditionEncoder.routing_target(record) for record in records], dtype=np.int64)
        score = (score - self.family_mean[family, None]) / self.family_std[family, None]
        return vector, score[:, :, None].astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {name: getattr(self, name).astype(float).tolist() for name in self.__dataclass_fields__}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "ResponseTargetTransform":
        return cls(**{name: np.asarray(state[name], dtype=np.float32) for name in cls.__dataclass_fields__})


def load_response_vectors(
    cache_path: str | Path,
    records: list[PatientRecord],
) -> np.ndarray:
    cache = np.load(cache_path, allow_pickle=False)
    ids = cache["patient_ids"].astype(str).tolist()
    index = {patient_id: row for row, patient_id in enumerate(ids)}
    missing = [record.patient_id for record in records if record.patient_id not in index]
    if missing:
        raise RuntimeError(f"Response cache misses {len(missing)} records; first={missing[0]}")
    ordered = cache["x_visit"][[index[record.patient_id] for record in records]].astype(np.float32)
    names = cache["feature_names"].astype(str).tolist()
    vector = response_vector(ordered, names)
    if vector.shape[-1] != len(RESPONSE_VECTOR_FEATURES):
        raise RuntimeError(f"Expected 18 response targets, got {vector.shape}")
    return vector
