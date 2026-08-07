"""仅由 outer-fold train 患者拟合的四访 pooled FTV transform。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import atomic_json


TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")


def patient_hash(patient_ids: Iterable[str]) -> str:
    content = "\n".join(sorted(str(value) for value in patient_ids)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def raw_ftv_hash(raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> str:
    """对 patient-sorted raw FTV/value-valid 对生成平台无关的 canonical hash。"""

    digest = hashlib.sha256()
    for patient_id in sorted(raw_ftv):
        values, valid = raw_ftv[patient_id]
        values = np.asarray(values, dtype="<f8")
        valid = np.asarray(valid, dtype=np.uint8)
        if values.shape != (4,) or valid.shape != (4,):
            raise ValueError(f"{patient_id} 的 FTV shape 非法: {values.shape}/{valid.shape}")
        digest.update(str(patient_id).encode("utf-8"))
        digest.update(values.tobytes())
        digest.update(valid.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PooledFTVTransform:
    """四个 observed visits 共用的 log/winsor/robust transform。"""

    schema_version: int
    fold: int
    train_patient_hash: str
    raw_targets_sha256: str
    train_patient_count: int
    paired_train_patient_count: int
    valid_visit_count: int
    epsilon: float
    winsor_quantiles: tuple[float, float]
    winsor_low: float
    winsor_high: float
    center_median: float
    scale_iqr: float

    @classmethod
    def fit(
        cls,
        raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]],
        train_ids: Iterable[str],
        fold: int,
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> "PooledFTVTransform":
        train_ids = tuple(str(value) for value in train_ids)
        if fold not in range(5):
            raise ValueError("fold 必须为 0–4")
        if not (0.0 <= quantiles[0] < quantiles[1] <= 1.0):
            raise ValueError("winsor quantiles 非法")
        pooled: list[np.ndarray] = []
        paired = 0
        for patient_id in train_ids:
            if patient_id not in raw_ftv:
                continue
            values, valid = raw_ftv[patient_id]
            values = np.asarray(values, dtype=np.float64)
            valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
            if values.shape != (4,) or valid.shape != (4,):
                raise ValueError(f"{patient_id} 的 FTV shape 非法")
            if valid.any():
                paired += 1
                pooled.append(values[valid])
        if not pooled:
            raise ValueError(f"fold {fold} train 中没有有效 FTV")
        observed = np.concatenate(pooled)
        if np.any(observed < 0):
            raise ValueError("FTV 必须非负")
        positive = observed[observed > 0]
        if not positive.size:
            raise ValueError("训练 FTV 没有正值，无法定义 epsilon")
        epsilon = max(1e-6, 0.5 * float(positive.min()))
        analysis = np.log(observed + epsilon)
        low, high = (float(np.quantile(analysis, q)) for q in quantiles)
        clipped = np.clip(analysis, low, high)
        center = float(np.median(clipped))
        q1, q3 = (float(np.quantile(clipped, q)) for q in (0.25, 0.75))
        scale = max(q3 - q1, 1e-6)
        return cls(
            schema_version=1,
            fold=int(fold),
            train_patient_hash=patient_hash(train_ids),
            raw_targets_sha256=raw_ftv_hash(raw_ftv),
            train_patient_count=len(train_ids),
            paired_train_patient_count=paired,
            valid_visit_count=int(observed.size),
            epsilon=epsilon,
            winsor_quantiles=tuple(float(value) for value in quantiles),
            winsor_low=low,
            winsor_high=high,
            center_median=center,
            scale_iqr=scale,
        )

    def transform_values(
        self, values: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        if valid is None:
            valid = np.isfinite(values)
        else:
            valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
        if values.shape != valid.shape:
            raise ValueError("FTV values/valid shape 不一致")
        if np.any(values[valid] < 0):
            raise ValueError("FTV 必须非负")
        output = np.zeros(values.shape, dtype=np.float32)
        if valid.any():
            analysis = np.log(values[valid] + self.epsilon)
            analysis = np.clip(analysis, self.winsor_low, self.winsor_high)
            output[valid] = ((analysis - self.center_median) / self.scale_iqr).astype(np.float32)
        return output, valid.astype(bool, copy=False)

    def inverse(self, standardized: np.ndarray) -> np.ndarray:
        analysis = np.asarray(standardized, dtype=np.float64) * self.scale_iqr + self.center_median
        return np.maximum(np.exp(analysis) - self.epsilon, 0.0)

    def transform_all(
        self, raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if raw_ftv_hash(raw_ftv) != self.raw_targets_sha256:
            raise ValueError("raw FTV hash 与 transform 不一致")
        return {
            patient_id: self.transform_values(values, valid)
            for patient_id, (values, valid) in raw_ftv.items()
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "fit_scope": "outer-fold train paired patients pooled across T0/T1/T2/T3",
                "value_transform": "log(ftv + epsilon)",
                "standardization": "winsor_01_99_then_median_iqr",
                "timepoints": list(TIMEPOINTS),
                "winsor_quantiles": list(self.winsor_quantiles),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PooledFTVTransform":
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("FTV transform schema 不兼容")
        if tuple(payload.get("timepoints", TIMEPOINTS)) != TIMEPOINTS:
            raise ValueError("FTV transform timepoint 顺序漂移")
        return cls(
            schema_version=1,
            fold=int(payload["fold"]),
            train_patient_hash=str(payload["train_patient_hash"]),
            raw_targets_sha256=str(payload["raw_targets_sha256"]),
            train_patient_count=int(payload["train_patient_count"]),
            paired_train_patient_count=int(payload["paired_train_patient_count"]),
            valid_visit_count=int(payload["valid_visit_count"]),
            epsilon=float(payload["epsilon"]),
            winsor_quantiles=tuple(float(value) for value in payload["winsor_quantiles"]),
            winsor_low=float(payload["winsor_low"]),
            winsor_high=float(payload["winsor_high"]),
            center_median=float(payload["center_median"]),
            scale_iqr=float(payload["scale_iqr"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PooledFTVTransform":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        atomic_json(path, self.to_dict())
