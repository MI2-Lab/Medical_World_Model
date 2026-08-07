"""Observed-state audit 的 fold-train-only static 与 change target。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .common import AUDIT_ROOT, atomic_json, file_sha256, load_yaml, refuse_existing
from .extraction import TIMEPOINTS, import_source_modules


FEATURE_NAMES = ("ftv", "sphericity", "ld", "bpe")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")


@dataclass(frozen=True)
class StaticFeatureTransform:
    timepoint: str
    feature_name: str
    value_transform: str
    epsilon: float
    winsor_low: float
    winsor_high: float
    center: float
    scale: float
    n_train: int

    def analysis_value(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if self.value_transform == "log_epsilon":
            if np.any(raw + self.epsilon <= 0):
                raise ValueError(f"{self.feature_name}/{self.timepoint} 出现 log 非法值")
            return np.log(raw + self.epsilon)
        if self.value_transform == "identity":
            return raw
        raise ValueError(self.value_transform)

    def standardize(self, raw: np.ndarray) -> np.ndarray:
        analysis = self.analysis_value(raw)
        return (np.clip(analysis, self.winsor_low, self.winsor_high) - self.center) / self.scale

    def inverse_prediction(self, standardized: np.ndarray) -> np.ndarray:
        analysis = np.asarray(standardized, dtype=np.float64) * self.scale + self.center
        if self.value_transform == "log_epsilon":
            return np.exp(analysis) - self.epsilon
        return analysis


@dataclass(frozen=True)
class StaticTargetTransform:
    schema_version: int
    fold: int
    raw_targets_sha256: str
    train_patient_hash: str
    train_patient_count: int
    paired_train_patient_count: int
    quantiles: tuple[float, float]
    specs: tuple[StaticFeatureTransform, ...]

    def spec(self, timepoint: str, feature_name: str) -> StaticFeatureTransform:
        matches = [
            item
            for item in self.specs
            if item.timepoint == timepoint and item.feature_name == feature_name
        ]
        if len(matches) != 1:
            raise KeyError(f"static transform cell 不唯一: {timepoint}/{feature_name}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fold": self.fold,
            "fit_scope": "每个 timepoint×feature 仅使用该 fold train paired patients",
            "raw_targets_sha256": self.raw_targets_sha256,
            "train_patient_hash": self.train_patient_hash,
            "train_patient_count": self.train_patient_count,
            "paired_train_patient_count": self.paired_train_patient_count,
            "quantiles": list(self.quantiles),
            "feature_order": list(FEATURE_NAMES),
            "timepoints": list(TIMEPOINTS),
            "specs": [item.__dict__ for item in self.specs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticTargetTransform":
        if payload.get("schema_version") != 1:
            raise ValueError("static target transform schema 不兼容")
        if tuple(payload["feature_order"]) != FEATURE_NAMES:
            raise ValueError("static target feature order 漂移")
        if tuple(payload["timepoints"]) != TIMEPOINTS:
            raise ValueError("static target timepoint order 漂移")
        specs = tuple(StaticFeatureTransform(**item) for item in payload["specs"])
        if len(specs) != len(FEATURE_NAMES) * len(TIMEPOINTS):
            raise ValueError("static transform cell 数不完整")
        return cls(
            schema_version=1,
            fold=int(payload["fold"]),
            raw_targets_sha256=str(payload["raw_targets_sha256"]),
            train_patient_hash=str(payload["train_patient_hash"]),
            train_patient_count=int(payload["train_patient_count"]),
            paired_train_patient_count=int(payload["paired_train_patient_count"]),
            quantiles=tuple(float(value) for value in payload["quantiles"]),
            specs=specs,
        )

    @classmethod
    def load(cls, path: Path) -> "StaticTargetTransform":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def reconstruct_static_values(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 `[3 transition,4 feature,(start,end,valid)]` 恢复为四访绝对值。"""

    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (3, 4, 3):
        raise ValueError(f"raw target shape 非法: {raw.shape}")
    values = np.empty((4, 4), dtype=np.float64)
    valid = np.empty((4, 4), dtype=bool)
    values[0] = raw[0, :, 0]
    values[1] = raw[0, :, 1]
    values[2] = raw[1, :, 1]
    values[3] = raw[2, :, 1]
    valid[0] = raw[0, :, 2].astype(bool)
    valid[1] = raw[0, :, 2].astype(bool) & raw[1, :, 2].astype(bool)
    valid[2] = raw[1, :, 2].astype(bool) & raw[2, :, 2].astype(bool)
    valid[3] = raw[2, :, 2].astype(bool)
    for transition in (0, 1):
        left = raw[transition, :, 1]
        right = raw[transition + 1, :, 0]
        both = raw[transition, :, 2].astype(bool) & raw[transition + 1, :, 2].astype(bool)
        if not np.allclose(left[both], right[both], rtol=0.0, atol=1e-10):
            raise ValueError(f"共享 visit endpoint 不一致: {TIMEPOINTS[transition + 1]}")
    valid &= np.isfinite(values)
    return values, valid


def load_target_assets(config_path: Path) -> tuple[Any, dict[str, np.ndarray], Any, Any]:
    config = load_yaml(config_path)
    source = import_source_modules(config)
    source_config = source.load_config(source.source_root / config["models"]["m0"]["config"])
    from rnc.training import build_bundle  # type: ignore
    from rnc.transforms import raw_targets_hash  # type: ignore

    bundle = build_bundle(source_config)
    return bundle, bundle.raw_radiomics, raw_targets_hash, source


def fit_static_transform(
    raw_targets: Mapping[str, np.ndarray],
    train_ids: Iterable[str],
    fold: int,
    change_transform: Any,
    raw_targets_hash: Any,
    patient_hash: Any,
    quantiles: tuple[float, float] = (0.01, 0.99),
) -> StaticTargetTransform:
    train_ids = tuple(sorted(str(value) for value in train_ids))
    paired_ids = [patient_id for patient_id in train_ids if patient_id in raw_targets]
    if not paired_ids:
        raise ValueError(f"fold {fold} train 无 paired radiomics")
    static = {patient_id: reconstruct_static_values(raw_targets[patient_id]) for patient_id in paired_ids}
    change_specs = {item.name: item for item in change_transform.features}
    specs: list[StaticFeatureTransform] = []
    transform_names = {"ftv": "log_epsilon", "ld": "log_epsilon", "sphericity": "identity", "bpe": "identity"}
    for time_index, timepoint in enumerate(TIMEPOINTS):
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            values = np.asarray(
                [static[patient_id][0][time_index, feature_index] for patient_id in paired_ids],
                dtype=np.float64,
            )
            valid = np.asarray(
                [static[patient_id][1][time_index, feature_index] for patient_id in paired_ids],
                dtype=bool,
            ) & np.isfinite(values)
            observed = values[valid]
            transform_name = transform_names[feature_name]
            epsilon = float(change_specs[feature_name].epsilon) if transform_name == "log_epsilon" else 0.0
            if transform_name == "log_epsilon":
                if np.any(observed + epsilon <= 0):
                    raise ValueError(f"fold {fold}/{timepoint}/{feature_name} log 非法")
                analysis = np.log(observed + epsilon)
            else:
                analysis = observed
            low, high = (float(np.quantile(analysis, q)) for q in quantiles)
            clipped = np.clip(analysis, low, high)
            center = float(np.median(clipped))
            q1, q3 = (float(np.quantile(clipped, q)) for q in (0.25, 0.75))
            scale = max(q3 - q1, 1e-6)
            specs.append(
                StaticFeatureTransform(
                    timepoint=timepoint,
                    feature_name=feature_name,
                    value_transform=transform_name,
                    epsilon=epsilon,
                    winsor_low=low,
                    winsor_high=high,
                    center=center,
                    scale=scale,
                    n_train=int(observed.size),
                )
            )
    return StaticTargetTransform(
        schema_version=1,
        fold=fold,
        raw_targets_sha256=raw_targets_hash(raw_targets),
        train_patient_hash=patient_hash(train_ids),
        train_patient_count=len(train_ids),
        paired_train_patient_count=len(paired_ids),
        quantiles=quantiles,
        specs=tuple(specs),
    )


def prepare_static_transforms(config_path: Path, output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    bundle, raw_targets, raw_targets_hash, source = load_target_assets(config_path)
    from rnc.data import patient_hash  # type: ignore
    from rnc.transforms import RadiomicsChangeTransform  # type: ignore

    output_paths = [output_dir / f"static_target_transform_fold_{fold}.json" for fold in range(5)]
    summary_path = AUDIT_ROOT / "metrics" / "target_transform_validation.json"
    refuse_existing([*output_paths, summary_path], overwrite)
    fold_rows: list[dict[str, Any]] = []
    for fold, output_path in enumerate(output_paths):
        subset = bundle.folds[bundle.folds["fold"] == fold]
        train_ids = subset.loc[subset["split"] == "train", "patient_id"].astype(str).tolist()
        change_path = source.source_root / "configs" / f"radiomics_transform_fold_{fold}.json"
        change = RadiomicsChangeTransform.load(change_path)
        transform = fit_static_transform(
            raw_targets,
            train_ids,
            fold,
            change,
            raw_targets_hash,
            patient_hash,
        )
        if transform.raw_targets_sha256 != change.raw_targets_sha256:
            raise ValueError("static/change transform raw target hash 不一致")
        atomic_json(output_path, transform.to_dict())
        fold_rows.append(
            {
                "fold": fold,
                "static_transform": str(output_path.resolve()),
                "static_transform_sha256": file_sha256(output_path),
                "change_transform": str(change_path.resolve()),
                "change_transform_sha256": file_sha256(change_path),
                "train_patient_hash": transform.train_patient_hash,
                "train_patient_count": transform.train_patient_count,
                "paired_train_patient_count": transform.paired_train_patient_count,
                "cells": len(transform.specs),
            }
        )
    summary = {
        "status": "static/change target transform validation complete",
        "raw_targets_sha256": raw_targets_hash(raw_targets),
        "radiomics_patients": len(raw_targets),
        "static_endpoint_equality_verified": True,
        "folds": fold_rows,
        "test_used_for_fit": False,
    }
    atomic_json(summary_path, summary)
    return summary


def static_target(
    raw: np.ndarray,
    transform: StaticTargetTransform,
    time_index: int,
    feature_index: int,
) -> tuple[float, float, float, bool]:
    values, valid = reconstruct_static_values(raw)
    value = float(values[time_index, feature_index])
    is_valid = bool(valid[time_index, feature_index])
    spec = transform.spec(TIMEPOINTS[time_index], FEATURE_NAMES[feature_index])
    standardized = float(spec.standardize(np.asarray([value]))[0]) if is_valid else float("nan")
    analysis = float(spec.analysis_value(np.asarray([value]))[0]) if is_valid else float("nan")
    return standardized, value, analysis, is_valid


def change_target(
    raw: np.ndarray,
    change_transform: Any,
    transition_index: int,
    feature_index: int,
) -> tuple[float, float, bool]:
    transformed, mask = change_transform.transform_one(raw)
    valid = bool(mask[transition_index, feature_index])
    spec = change_transform.features[feature_index]
    values = np.asarray(raw, dtype=np.float64)[transition_index, feature_index]
    natural = float(
        change_transform._raw_change(values[None, :], spec.value_transform, spec.epsilon)[0]
    ) if valid else float("nan")
    return float(transformed[transition_index, feature_index]), natural, valid

