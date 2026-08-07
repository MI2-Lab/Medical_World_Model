"""Fold 内拟合、可序列化的 radiomics 相邻变化变换。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .data import FEATURE_NAMES, patient_hash


TRANSFORM_SPEC_VERSION = "adjacent_v2_ftv_ld_logepsilon_sphericity_bpe_absolute_winsor01_99_robust"


def raw_targets_hash(raw_targets: Mapping[str, np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for patient_id in sorted(raw_targets):
        digest.update(patient_id.encode("utf-8"))
        digest.update(np.asarray(raw_targets[patient_id], dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureTransform:
    name: str
    value_transform: str
    epsilon: float
    winsor_low: float
    winsor_high: float
    center: float
    scale: float
    n_train_values: int

    def transform_values(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        if self.value_transform == "log_epsilon":
            change = np.log(end + self.epsilon) - np.log(start + self.epsilon)
        elif self.value_transform == "log1p":
            change = np.log1p(end) - np.log1p(start)
        elif self.value_transform == "identity":
            change = end - start
        else:
            raise ValueError(f"未知 value transform: {self.value_transform}")
        clipped = np.clip(change, self.winsor_low, self.winsor_high)
        return (clipped - self.center) / self.scale

    def inverse_change(self, standardized: np.ndarray) -> np.ndarray:
        return np.asarray(standardized) * self.scale + self.center


@dataclass(frozen=True)
class RadiomicsChangeTransform:
    spec_version: str
    raw_targets_sha256: str
    fold: int
    train_patient_hash: str
    train_patient_count: int
    paired_train_patient_count: int
    quantiles: tuple[float, float]
    features: tuple[FeatureTransform, ...]

    @staticmethod
    def _raw_change(values: np.ndarray, transform_name: str, epsilon: float) -> np.ndarray:
        start, end = values[..., 0], values[..., 1]
        if transform_name == "log_epsilon":
            return np.log(end + epsilon) - np.log(start + epsilon)
        if transform_name == "log1p":
            return np.log1p(end) - np.log1p(start)
        if transform_name == "identity":
            return end - start
        raise ValueError(transform_name)

    @classmethod
    def fit(
        cls,
        raw_targets: Mapping[str, np.ndarray],
        train_patient_ids: Iterable[str],
        fold: int,
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> "RadiomicsChangeTransform":
        train_ids = tuple(sorted(str(value) for value in train_patient_ids))
        if not train_ids:
            raise ValueError("fold train patient 列表为空")
        paired = [raw_targets[patient_id] for patient_id in train_ids if patient_id in raw_targets]
        if not paired:
            raise ValueError("fold train 中没有 radiomics 配对患者")
        stacked = np.stack(paired).astype(np.float64)
        feature_transforms: list[FeatureTransform] = []
        transform_names = {
            "ftv": "log_epsilon",
            "sphericity": "identity",
            "ld": "log_epsilon",
            "bpe": "identity",
        }
        for feature_index, feature in enumerate(FEATURE_NAMES):
            values = stacked[:, :, feature_index]
            valid = values[..., 2].astype(bool) & np.isfinite(values[..., 0]) & np.isfinite(values[..., 1])
            if not valid.any():
                raise ValueError(f"fold {fold} 的 {feature} 没有有效训练 target")
            transform_name = transform_names[feature]
            epsilon = 0.0
            if transform_name == "log_epsilon":
                observed = values[..., :2][np.repeat(valid[..., None], 2, axis=-1)]
                positive = observed[observed > 0]
                if positive.size == 0:
                    raise ValueError(f"fold {fold} 的 {feature} 没有正值")
                epsilon = max(1e-6, float(positive.min()) * 0.5)
            change = cls._raw_change(values, transform_name, epsilon)[valid]
            finite = change[np.isfinite(change)]
            if finite.size == 0:
                raise ValueError(f"fold {fold} 的 {feature} change 非有限")
            low, high = (float(np.quantile(finite, q)) for q in quantiles)
            clipped = np.clip(finite, low, high)
            center = float(np.median(clipped))
            q1, q3 = (float(np.quantile(clipped, q)) for q in (0.25, 0.75))
            scale = max(q3 - q1, 1e-6)
            feature_transforms.append(
                FeatureTransform(feature, transform_name, epsilon, low, high, center, scale, int(finite.size))
            )
        return cls(
            spec_version=TRANSFORM_SPEC_VERSION,
            raw_targets_sha256=raw_targets_hash(raw_targets),
            fold=int(fold),
            train_patient_hash=patient_hash(train_ids),
            train_patient_count=len(train_ids),
            paired_train_patient_count=len(paired),
            quantiles=tuple(float(value) for value in quantiles),
            features=tuple(feature_transforms),
        )

    def transform_one(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (3, len(FEATURE_NAMES), 3):
            raise ValueError(f"radiomics raw shape 应为 [3,4,3]，实际 {raw.shape}")
        target = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros_like(target, dtype=bool)
        for feature_index, spec in enumerate(self.features):
            values = raw[:, feature_index]
            valid = values[:, 2].astype(bool) & np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
            safe_start = np.where(valid, values[:, 0], 0.0)
            safe_end = np.where(valid, values[:, 1], 0.0)
            transformed = spec.transform_values(safe_start, safe_end)
            target[:, feature_index] = np.where(valid, transformed, 0.0).astype(np.float32)
            mask[:, feature_index] = valid
        return target, mask

    def transform_all(self, raw_targets: Mapping[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {patient_id: self.transform_one(raw) for patient_id, raw in raw_targets.items()}

    def inverse_feature(self, feature_index: int, standardized: np.ndarray) -> np.ndarray:
        return self.features[feature_index].inverse_change(standardized)

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            "raw_targets_sha256": self.raw_targets_sha256,
            "fold": self.fold,
            "fit_scope": "仅该 fold 的 I-SPY2 train patients；未使用 val/test",
            "train_patient_hash": self.train_patient_hash,
            "train_patient_count": self.train_patient_count,
            "paired_train_patient_count": self.paired_train_patient_count,
            "quantiles": list(self.quantiles),
            "feature_order": list(FEATURE_NAMES),
            "features": [spec.__dict__ for spec in self.features],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RadiomicsChangeTransform":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("spec_version") != TRANSFORM_SPEC_VERSION:
            raise ValueError(f"radiomics transform 版本陈旧: {path}")
        if tuple(payload["feature_order"]) != FEATURE_NAMES:
            raise ValueError("radiomics transform feature order 不一致")
        if tuple(item.get("name") for item in payload.get("features", [])) != FEATURE_NAMES:
            raise ValueError("radiomics transform 内部 feature 名称/顺序不一致")
        return cls(
            spec_version=str(payload["spec_version"]),
            raw_targets_sha256=str(payload["raw_targets_sha256"]),
            fold=int(payload["fold"]),
            train_patient_hash=str(payload["train_patient_hash"]),
            train_patient_count=int(payload["train_patient_count"]),
            paired_train_patient_count=int(payload["paired_train_patient_count"]),
            quantiles=tuple(float(value) for value in payload["quantiles"]),
            features=tuple(FeatureTransform(**item) for item in payload["features"]),
        )
