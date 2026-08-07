"""从 clean checkpoint metadata 恢复只读评估 runtime。"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np
import torch

from ispy_jepa_tmi_clean.corejepa.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    ReadoutConfig,
    TrainConfig,
)
from ispy_jepa_tmi_clean.corejepa.data.condition import ConditionEncoder
from ispy_jepa_tmi_clean.corejepa.data.contracts import TEMPORAL_CONDITION_FEATURES
from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from ispy_jepa_tmi_clean.corejepa.models import CoReJEPA

from .provenance import validate_checkpoint_payload


ConfigType = TypeVar("ConfigType")


def _restore_dataclass(cls: type[ConfigType], values: Mapping[str, Any]) -> ConfigType:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"checkpoint config {cls.__name__} 含未知字段：{unknown}")
    defaults = cls()
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(getattr(defaults, key), tuple) and isinstance(value, list):
            value = tuple(value)
        normalized[key] = value
    return cls(**normalized)


def experiment_config_from_payload(payload: Mapping[str, Any]) -> ExperimentConfig:
    """从 checkpoint 内冻结的 config 恢复 dataclass，拒绝静默字段漂移。"""

    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config 缺失或类型错误")
    sections = ("data", "model", "loss", "train", "readout")
    if any(not isinstance(config.get(name), Mapping) for name in sections):
        raise ValueError(f"checkpoint config 必须包含 mapping sections：{sections}")
    return ExperimentConfig(
        data=_restore_dataclass(DataConfig, config["data"]),
        model=_restore_dataclass(ModelConfig, config["model"]),
        loss=_restore_dataclass(LossConfig, config["loss"]),
        train=_restore_dataclass(TrainConfig, config["train"]),
        readout=_restore_dataclass(ReadoutConfig, config["readout"]),
    )


class StoredConditionEncoder:
    """严格使用 checkpoint 保存的 arm vocabulary 与年龄统计。"""

    def __init__(self, metadata: Mapping[str, Any]):
        required = ("feature_names", "arm_vocab", "age_mean", "age_std")
        if any(name not in metadata for name in required):
            raise ValueError(f"condition metadata 必须包含 {required}")
        self.feature_names = tuple(str(value) for value in metadata["feature_names"])
        self.arm_vocab = {str(key): int(value) for key, value in metadata["arm_vocab"].items()}
        self.age_mean = float(metadata["age_mean"])
        self.age_std = float(metadata["age_std"])
        if not np.isfinite(self.age_mean) or not np.isfinite(self.age_std) or self.age_std <= 1e-6:
            raise ValueError("checkpoint age normalization metadata 无效")
        expected = list(TEMPORAL_CONDITION_FEATURES)
        expected.extend(
            f"arm={arm}" for arm, _ in sorted(self.arm_vocab.items(), key=lambda item: item[1])
        )
        expected.extend(("HR", "HER2", "MammaPrint", "age_z"))
        if tuple(expected) != self.feature_names:
            raise ValueError("checkpoint feature_names 与 clean ConditionEncoder contract 不一致")
        indices = sorted(self.arm_vocab.values())
        if indices != list(range(len(indices))):
            raise ValueError("checkpoint arm_vocab 索引必须从 0 连续排列")

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def encode(self, record: PatientRecord) -> np.ndarray:
        if record.arm not in self.arm_vocab:
            raise ValueError(f"患者 {record.patient_id} 的 treatment arm 不在 checkpoint vocabulary")
        output = np.zeros((3, self.dim), dtype=np.float32)
        arm_offset = len(TEMPORAL_CONDITION_FEATURES)
        clinical_offset = arm_offset + len(self.arm_vocab)
        age = 0.0 if not np.isfinite(record.age) else (record.age - self.age_mean) / self.age_std
        for transition in range(3):
            output[transition, transition] = 1.0
            output[transition, 3 : 4 + transition] = 1.0
            output[transition, arm_offset + self.arm_vocab[record.arm]] = 1.0
            output[transition, clinical_offset : clinical_offset + 4] = (
                float(record.hr),
                float(record.her2),
                float(record.mp),
                float(age),
            )
        return output

    @staticmethod
    def routing_target(record: PatientRecord) -> int:
        return ConditionEncoder.routing_target(record)


def records_in_checkpoint_order(
    records: Sequence[PatientRecord],
    patient_ids: Sequence[str],
) -> list[PatientRecord]:
    """按 checkpoint patient order 排列 records，拒绝缺失、额外或重复 ID。"""

    by_id = {record.patient_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("输入 records 含重复 patient_id")
    requested = [str(value) for value in patient_ids]
    missing = [value for value in requested if value not in by_id]
    extra = sorted(set(by_id).difference(requested))
    if missing or extra:
        raise ValueError(f"records 与 checkpoint patient order 不一致：missing={len(missing)}, extra={len(extra)}")
    return [by_id[value] for value in requested]


def restore_model_for_evaluation(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> tuple[CoReJEPA, ExperimentConfig, StoredConditionEncoder]:
    """恢复模型并强制 ``eval``；不创建 optimizer，也不改写 checkpoint。"""

    validate_checkpoint_payload(payload)
    config = experiment_config_from_payload(payload)
    condition_encoder = StoredConditionEncoder(payload["condition"])
    model = CoReJEPA(config.model, condition_encoder.dim).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, condition_encoder
