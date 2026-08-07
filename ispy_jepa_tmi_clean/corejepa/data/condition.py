from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import TEMPORAL_CONDITION_FEATURES
from .records import PatientRecord, treatment_family


ROUTING_FAMILIES = (
    "her2_targeted",
    "io",
    "ispy1_nact",
    "platinum_parp",
    "targeted_other",
    "taxane",
)


@dataclass(frozen=True)
class ConditionSpec:
    feature_names: tuple[str, ...]
    arm_vocab: dict[str, int]
    age_mean: float
    age_std: float

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    @property
    def temporal_dim(self) -> int:
        return len(TEMPORAL_CONDITION_FEATURES)


class ConditionEncoder:
    """Create transition conditions.

    Input:
        ``PatientRecord``.
    Output:
        ``condition: float32 [3,Cc]``. Rows forecast T1, T2, and T3. In the
        paper cohort, ``Cc = 7 temporal + 14 arms + 3 biomarkers + 1 age = 25``.
    """

    def __init__(self, records: list[PatientRecord]):
        arms = sorted({record.arm for record in records})
        arm_vocab = {arm: index for index, arm in enumerate(arms)}
        ages = np.asarray([record.age for record in records], dtype=np.float32)
        finite = ages[np.isfinite(ages)]
        age_mean = float(finite.mean()) if finite.size else 0.0
        age_std = float(finite.std()) if finite.size else 1.0
        if age_std <= 1e-6:
            age_std = 1.0
        names = list(TEMPORAL_CONDITION_FEATURES)
        names.extend(f"arm={arm}" for arm in arms)
        names.extend(("HR", "HER2", "MammaPrint", "age_z"))
        self.spec = ConditionSpec(tuple(names), arm_vocab, age_mean, age_std)

    def encode(self, record: PatientRecord) -> np.ndarray:
        output = np.zeros((3, self.spec.dim), dtype=np.float32)
        arm_offset = 7
        clinical_offset = arm_offset + len(self.spec.arm_vocab)
        age = 0.0 if not np.isfinite(record.age) else (record.age - self.spec.age_mean) / self.spec.age_std
        for transition in range(3):
            output[transition, transition] = 1.0
            output[transition, 3 : 4 + transition] = 1.0
            output[transition, arm_offset + self.spec.arm_vocab[record.arm]] = 1.0
            output[transition, clinical_offset : clinical_offset + 4] = (
                float(record.hr),
                float(record.her2),
                float(record.mp),
                float(age),
            )
        return output

    @staticmethod
    def routing_target(record: PatientRecord) -> int:
        return ROUTING_FAMILIES.index(treatment_family(record))

    @staticmethod
    def routing_class_weights(records: list[PatientRecord], train_indices: list[int]) -> np.ndarray:
        targets = np.asarray([ConditionEncoder.routing_target(record) for record in records], dtype=np.int64)
        counts = np.bincount(targets[np.asarray(train_indices)], minlength=len(ROUTING_FAMILIES)).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (len(ROUTING_FAMILIES) * counts)
        return (weights / weights.mean()).astype(np.float32)
