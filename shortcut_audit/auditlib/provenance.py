"""Checkpoint、split 与 frozen readout 的不可变 provenance 校验。"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CHECKPOINT_FIELDS = (
    "model",
    "config",
    "condition",
    "response_transform",
    "patient_ids",
    "n_primary",
    "splits",
    "epoch",
    "validation",
)


def file_sha256(path: str | Path) -> str:
    """流式计算文件哈希，不加载 checkpoint tensor。"""

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indices(values: Sequence[object], *, name: str, size: int) -> list[int]:
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)):
        raise ValueError(f"checkpoint split {name} 含重复索引")
    if any(index < 0 or index >= size for index in indices):
        raise ValueError(f"checkpoint split {name} 含越界索引")
    return indices


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_primary_ids: Sequence[str] | None = None,
    expected_fold_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """验证 clean checkpoint schema、patient order 和 split 隔离。

    ``expected_fold_ids`` 使用规范键 ``train/val/test``，只比较 primary
    I-SPY2 IDs；I-SPY1 extra records 仅允许出现在 ``pretrain_train``。
    """

    missing = [field for field in CHECKPOINT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"checkpoint 缺少 clean schema 字段：{missing}")
    patient_ids = [str(value) for value in payload["patient_ids"]]
    if not patient_ids or len(patient_ids) != len(set(patient_ids)):
        raise ValueError("checkpoint patient_ids 为空或含重复值")
    n_primary = int(payload["n_primary"])
    if n_primary <= 0 or n_primary > len(patient_ids):
        raise ValueError("checkpoint n_primary 无效")
    primary_ids = patient_ids[:n_primary]
    if expected_primary_ids is not None and set(primary_ids) != {str(value) for value in expected_primary_ids}:
        raise ValueError("checkpoint primary patient 集合与 clean cohort 不一致")

    raw_splits = payload["splits"]
    required_splits = ("primary_train", "pretrain_train", "validation", "test")
    if not isinstance(raw_splits, Mapping) or any(name not in raw_splits for name in required_splits):
        raise ValueError(f"checkpoint splits 必须包含 {required_splits}")
    splits = {
        name: _indices(raw_splits[name], name=name, size=len(patient_ids))
        for name in required_splits
    }
    primary_train = set(splits["primary_train"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    if primary_train & validation or primary_train & test or validation & test:
        raise ValueError("checkpoint primary train/validation/test 发生患者重叠")
    if any(index >= n_primary for index in primary_train | validation | test):
        raise ValueError("primary split 中出现 I-SPY1 extra record")
    if primary_train | validation | test != set(range(n_primary)):
        raise ValueError("checkpoint primary split 未完整覆盖 I-SPY2 patient order")
    pretrain = set(splits["pretrain_train"])
    if not primary_train.issubset(pretrain):
        raise ValueError("pretrain_train 未包含全部 primary_train")
    if pretrain & validation or pretrain & test:
        raise ValueError("pretrain_train 与 primary validation/test 重叠")
    expected_pretrain = primary_train | set(range(n_primary, len(patient_ids)))
    if pretrain != expected_pretrain:
        raise ValueError("pretrain_train 的 extra-record 范围与 clean runner 不一致")

    if expected_fold_ids is not None:
        key_map = {"train": "primary_train", "val": "validation", "test": "test"}
        for canonical, checkpoint_key in key_map.items():
            expected = {str(value) for value in expected_fold_ids[canonical]}
            observed = {patient_ids[index] for index in splits[checkpoint_key]}
            if observed != expected:
                raise ValueError(
                    f"checkpoint {checkpoint_key} 与 fold manifest 不一致："
                    f"missing={len(expected - observed)}, extra={len(observed - expected)}"
                )

    condition = payload["condition"]
    condition_fields = ("feature_names", "arm_vocab", "age_mean", "age_std")
    if not isinstance(condition, Mapping) or any(name not in condition for name in condition_fields):
        raise ValueError(f"checkpoint condition metadata 必须包含 {condition_fields}")
    model_state = payload["model"]
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("checkpoint model state 为空或类型错误")
    return {
        "n_records": len(patient_ids),
        "n_primary": n_primary,
        "n_extra": len(patient_ids) - n_primary,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "epoch": int(payload["epoch"]),
        "condition_dim": len(condition["feature_names"]),
        "model_tensor_count": len(model_state),
    }


def inspect_checkpoint(
    path: str | Path,
    **validation_kwargs: object,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """在 CPU 上读取本地 clean checkpoint 并返回校验摘要。"""

    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint 顶层必须为 mapping")
    summary = validate_checkpoint_payload(payload, **validation_kwargs)
    return payload, {
        **summary,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def load_frozen_readout(path: str | Path) -> Any:
    """只加载既有 readout；拒绝没有 ``predict_proba`` 的对象。"""

    path = Path(path).resolve()
    with path.open("rb") as stream:
        readout = pickle.load(stream)  # noqa: S301 - 仅加载项目方本地可信 artifact
    if not callable(getattr(readout, "predict_proba", None)):
        raise ValueError("frozen readout 不提供 predict_proba")
    return readout
