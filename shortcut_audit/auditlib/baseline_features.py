"""审计重训练的简化输入特征；所有统计量只能由 fold-train 拟合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord


DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
N_GEOMETRY = 9


def _decision_code(n_patients: int) -> np.ndarray:
    return np.broadcast_to(
        np.eye(3, dtype=np.float32)[None, :, :], (n_patients, 3, 3)
    ).copy()


@dataclass(frozen=True)
class ClinicalFeatureSpec:
    """仅由 fold-train baseline covariates 拟合的 arm/age metadata。"""

    arms: tuple[str, ...]
    age_mean: float
    age_std: float

    @classmethod
    def fit(cls, records: Sequence[PatientRecord]) -> "ClinicalFeatureSpec":
        if not records:
            raise ValueError("至少需要一名 fold-train 患者拟合 clinical feature spec")
        arms = tuple(sorted({str(record.arm) for record in records}))
        ages = np.asarray([record.age for record in records], dtype=np.float64)
        finite = ages[np.isfinite(ages)]
        mean = float(finite.mean()) if finite.size else 0.0
        std = float(finite.std(ddof=0)) if finite.size else 1.0
        if not np.isfinite(std) or std <= 1e-6:
            std = 1.0
        return cls(arms=arms, age_mean=mean, age_std=std)

    @property
    def feature_names_without_decision(self) -> tuple[str, ...]:
        return (
            "HR",
            "HER2",
            "MammaPrint",
            "age_z",
            *(f"arm={arm}" for arm in self.arms),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *self.feature_names_without_decision,
            *(f"decision={name}" for name in DECISION_POINTS),
        )


def clinical_features(
    records: Sequence[PatientRecord], spec: ClinicalFeatureSpec
) -> np.ndarray:
    """返回 ``[N,3,F]`` clinical/treatment + nominal decision features。"""

    arm_index = {arm: index for index, arm in enumerate(spec.arms)}
    base = np.zeros((len(records), 4 + len(spec.arms)), dtype=np.float32)
    for row, record in enumerate(records):
        if record.arm not in arm_index:
            raise ValueError(f"测试患者 {record.patient_id} 出现 fold-train 未见 arm：{record.arm}")
        age_z = (
            0.0
            if not np.isfinite(record.age)
            else (float(record.age) - spec.age_mean) / spec.age_std
        )
        base[row, :4] = (record.hr, record.her2, record.mp, age_z)
        base[row, 4 + arm_index[record.arm]] = 1.0
    repeated = np.broadcast_to(base[:, None, :], (len(records), 3, base.shape[-1]))
    return np.concatenate((repeated, _decision_code(len(records))), axis=-1).astype(
        np.float32
    )


def geometry_features(geometry: np.ndarray, *, relative_epsilon: float = 1e-6) -> np.ndarray:
    """构造只使用当前 decision 可见 q 的 ``[N,3,57]`` 特征。

    每行依次包含 baseline、current、相对 baseline 差、最近差、相对变化、当前
    prefix 均值（各 9 维）和 3 维 nominal decision code。T0 的两个差分为零。
    """

    values = np.asarray(geometry, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (4, N_GEOMETRY):
        raise ValueError(f"geometry 必须为 [N,4,9]，实际为 {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("geometry 含非有限值")
    if not np.isfinite(relative_epsilon) or relative_epsilon <= 0:
        raise ValueError("relative_epsilon 必须为有限正数")

    baseline = np.repeat(values[:, :1, :], 3, axis=1)
    current = values[:, :3, :]
    from_baseline = current - baseline
    recent = np.zeros_like(current)
    recent[:, 1:, :] = current[:, 1:, :] - current[:, :-1, :]
    relative = from_baseline / (np.abs(baseline) + relative_epsilon)
    prefix_mean = np.stack(
        [values[:, : decision + 1, :].mean(axis=1) for decision in range(3)], axis=1
    )
    return np.concatenate(
        (
            baseline,
            current,
            from_baseline,
            recent,
            relative,
            prefix_mean,
            _decision_code(len(values)),
        ),
        axis=-1,
    ).astype(np.float32)


def clinical_geometry_features(
    records: Sequence[PatientRecord],
    geometry: np.ndarray,
    spec: ClinicalFeatureSpec,
) -> np.ndarray:
    """组合 clinical base 与 geometry；nominal decision code 只保留一份。"""

    clinical = clinical_features(records, spec)
    geometric = geometry_features(geometry)
    if len(clinical) != len(geometric):
        raise ValueError("records 与 geometry 患者数不一致")
    return np.concatenate((clinical[:, :, :-3], geometric), axis=-1).astype(np.float32)


def timepoint_only_features(n_patients: int) -> np.ndarray:
    """返回 intercept 由 logistic estimator 提供时所需的 nominal timepoint one-hot。"""

    if isinstance(n_patients, bool) or int(n_patients) != n_patients or n_patients < 0:
        raise ValueError("n_patients 必须为非负整数")
    return _decision_code(int(n_patients))


def static_t0_features(t0_state: np.ndarray) -> np.ndarray:
    """将冻结的 T0 state 重复到三个 decision point，不读取 follow-up state。"""

    state = np.asarray(t0_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 1:
        raise ValueError(f"t0_state 必须为 [N,D] 且 D>0，实际为 {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("t0_state 含非有限值")
    repeated = np.broadcast_to(state[:, None, :], (len(state), 3, state.shape[-1]))
    return np.concatenate((repeated, _decision_code(len(state))), axis=-1).astype(
        np.float32
    )


__all__ = [
    "ClinicalFeatureSpec",
    "DECISION_POINTS",
    "clinical_features",
    "clinical_geometry_features",
    "geometry_features",
    "static_t0_features",
    "timepoint_only_features",
]
