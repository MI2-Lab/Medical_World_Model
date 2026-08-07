"""冻结 image-only world model 的五折 readout、grounding 与扰动评估。

本模块只从 checkpoint 锁定的 I-SPY2 train/val/test 患者读取 DCE cache。
图像模型始终冻结；pCR 只在训练 logistic readout 时使用，test 标签不会参与
模型、超参数或 threshold 的选择。
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import EXPERIMENT_ROOT, resolve_path
from .data import (
    FEATURE_NAMES,
    TRANSITIONS,
    CohortBundle,
    LongitudinalCacheDataset,
    patient_hash,
    split_ids,
)
from .model import ImageOnlyWorldModel
from .training import build_bundle, implementation_sha256, records_for_ids
from .transforms import RadiomicsChangeTransform, raw_targets_hash


DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
AVAILABLE_VISITS = (1, 2, 3)
READOUT_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
READOUT_PENALTIES = ("l2", "l1")
FEATURE_SCHEMA = "concat(current_target_state, predicted_next_state, predicted_delta)"
FEATURE_SCHEMA_SHA256 = hashlib.sha256(FEATURE_SCHEMA.encode("utf-8")).hexdigest()
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class LoadedEvaluation:
    """经 contract 验证后可用于评估的冻结对象。"""

    checkpoint_path: Path
    checkpoint_sha256: str
    payload: Mapping[str, Any]
    model: ImageOnlyWorldModel
    bundle: CohortBundle
    splits: Mapping[str, list[str]]
    radiomics_transform: RadiomicsChangeTransform
    fold: int
    run_name: str
    mode: str


@dataclass(frozen=True)
class ExtractedSplit:
    """一个 patient split 的原生 image-derived 表征。"""

    patient_ids: tuple[str, ...]
    labels: np.ndarray
    has_radiomics: np.ndarray
    features: Mapping[str, np.ndarray]
    current_state: np.ndarray
    target_next: np.ndarray
    predicted_next: np.ndarray
    target_delta: np.ndarray
    predicted_delta: np.ndarray
    radiomics_prediction: np.ndarray | None


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA256，不修改源文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_implementation_sha256() -> str:
    """锁定 evaluator 与 CLI 源码；训练实现另由 checkpoint 自身锁定。"""

    digest = hashlib.sha256()
    paths = (Path(__file__).resolve(), EXPERIMENT_ROOT / "scripts" / "evaluate_fold.py")
    for path in paths:
        digest.update(str(path.relative_to(EXPERIMENT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 mapping")
    return value


def _validated_model_config(value: Any) -> dict[str, Any]:
    config = dict(_require_mapping(value, "checkpoint.model_config"))
    required = {
        "mode",
        "image_channels",
        "base_channels",
        "latent_dim",
        "predictor_depth",
        "predictor_heads",
        "predictor_mlp_dim",
        "dropout",
        "radiomics_dim",
    }
    if set(config) != required:
        raise ValueError(
            "checkpoint.model_config 字段不符合白名单；"
            f"缺少={sorted(required - set(config))}，额外={sorted(set(config) - required)}"
        )
    mode = config["mode"]
    if not isinstance(mode, str) or mode not in ImageOnlyWorldModel.VALID_MODES:
        raise ValueError(f"checkpoint model mode 非法: {mode!r}")
    integer_fields = (
        "image_channels",
        "base_channels",
        "latent_dim",
        "predictor_depth",
        "predictor_heads",
        "predictor_mlp_dim",
        "radiomics_dim",
    )
    for name in integer_fields:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint.model_config.{name} 必须为正整数")
    if config["image_channels"] not in (7, 8):
        raise ValueError("checkpoint image_channels 只能是 7 或 8")
    if config["radiomics_dim"] != len(FEATURE_NAMES):
        raise ValueError("checkpoint radiomics_dim 与固定 feature schema 不一致")
    dropout = config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise TypeError("checkpoint dropout 必须为数值")
    if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("checkpoint dropout 必须位于 [0,1)")
    config["dropout"] = float(dropout)
    return config


def _safe_load_checkpoint(path: Path) -> Mapping[str, Any]:
    """只用默认安全白名单的 weights-only unpickler 加载 checkpoint。

    不额外 allowlist 任何 pickle global，也绝不回退到 ``weights_only=False``。
    因此旧 payload 若把版本保存成 ``TorchVersion`` 会被明确拒绝；当前训练代码
    已把版本强制保存为普通 ``str``。
    """

    if path.name != "best.pt":
        raise ValueError(f"评估入口只接受 best.pt，实际为: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # 错误信息保留原异常类型，禁止不安全回退。
        raise RuntimeError(f"checkpoint 无法通过安全 weights-only 加载: {path}") from error
    payload = _require_mapping(payload, "checkpoint")
    required = {
        "schema_version",
        "run_name",
        "fold",
        "model_state",
        "model_config",
        "data_contract",
        "splits",
    }
    if missing := required.difference(payload):
        raise ValueError(f"checkpoint 缺少必要字段: {sorted(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError(f"不支持的 checkpoint schema: {payload['schema_version']!r}")
    return payload


def _resolved_existing_path(value: str | Path, name: str, *, directory: bool = False) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if directory and not path.is_dir():
        raise NotADirectoryError(f"{name} 不是目录: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{name} 不是文件: {path}")
    return path


def _validate_checkpoint_splits(
    payload: Mapping[str, Any], bundle: CohortBundle, fold: int
) -> dict[str, list[str]]:
    stored = _require_mapping(payload["splits"], "checkpoint.splits")
    expected = split_ids(bundle, fold)
    output: dict[str, list[str]] = {}
    for split in ("train", "val", "test", "pretrain_train"):
        values = stored.get(split)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"checkpoint.splits.{split} 必须是 patient ID sequence")
        ids = [str(value) for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"checkpoint.splits.{split} 含重复 patient_id")
        if ids != expected[split]:
            raise ValueError(f"checkpoint 锁定的 {split} IDs/顺序与 fold manifest 不一致")
        output[split] = ids

    primary_sets = [set(output[name]) for name in ("train", "val", "test")]
    if any(primary_sets[i] & primary_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("checkpoint 的 train/val/test 发生 patient overlap")
    if set.union(*primary_sets) != {record.patient_id for record in bundle.primary}:
        raise ValueError("checkpoint 的 train/val/test 未恰好覆盖 808 名 primary 患者")
    return output


def _validate_data_contract(
    payload: Mapping[str, Any], config: Mapping[str, Any], bundle: CohortBundle, splits: Mapping[str, list[str]]
) -> RadiomicsChangeTransform:
    contract = _require_mapping(payload["data_contract"], "checkpoint.data_contract")
    data_config = _require_mapping(config.get("data"), "config.data")
    manifest_from_config = resolve_path(data_config["fold_manifest"]).resolve(strict=True)
    manifest_from_checkpoint = _resolved_existing_path(contract["fold_manifest"], "checkpoint fold manifest")
    if manifest_from_config != manifest_from_checkpoint:
        raise ValueError("config 与 checkpoint 指向不同 fold manifest")
    actual_manifest_hash = file_sha256(manifest_from_config)
    if actual_manifest_hash != str(contract.get("fold_manifest_sha256", "")):
        raise ValueError("fold manifest SHA256 与 checkpoint 锁定值不一致")

    cache_from_config = resolve_path(data_config["cache_root"]).resolve(strict=True)
    cache_from_checkpoint = _resolved_existing_path(
        contract["cache_root"], "checkpoint cache root", directory=True
    )
    if cache_from_config != cache_from_checkpoint:
        raise ValueError("config 与 checkpoint 指向不同 DCE cache")
    if not cache_from_config.is_dir():
        raise NotADirectoryError(f"DCE cache root 非目录: {cache_from_config}")

    for split in ("train", "val", "test"):
        expected = patient_hash(splits[split])
        if str(contract.get(f"{split}_patient_hash", "")) != expected:
            raise ValueError(f"checkpoint 的 {split}_patient_hash 不一致")
    extra_ids = set(splits["pretrain_train"]) - set(splits["train"])
    if str(contract.get("extra_pretrain_patient_hash", "")) != patient_hash(extra_ids):
        raise ValueError("checkpoint 的 extra_pretrain_patient_hash 不一致")

    transform_path = _resolved_existing_path(contract["radiomics_transform"], "radiomics transform")
    if file_sha256(transform_path) != str(contract.get("radiomics_transform_sha256", "")):
        raise ValueError("radiomics transform SHA256 与 checkpoint 锁定值不一致")
    transform = RadiomicsChangeTransform.load(transform_path)
    if transform.fold != int(payload["fold"]):
        raise ValueError("radiomics transform fold 与 checkpoint 不一致")
    if transform.train_patient_hash != patient_hash(splits["train"]):
        raise ValueError("radiomics transform 并非仅由锁定 fold-train 拟合")
    if transform.train_patient_count != len(splits["train"]):
        raise ValueError("radiomics transform train patient 数与 checkpoint 不一致")
    if transform.raw_targets_sha256 != raw_targets_hash(bundle.raw_radiomics):
        raise ValueError("当前 radiomics raw targets 与 checkpoint transform 拟合来源不一致")

    # bundle 已严格校验 808 cache 与 manifest；这里再确保评估集合不含 I-SPY1。
    primary_ids = {record.patient_id for record in bundle.primary}
    if set(splits["train"] + splits["val"] + splits["test"]) != primary_ids:
        raise ValueError("评估 split 不是锁定的 808 名 I-SPY2 primary cohort")
    return transform


def load_evaluation(
    checkpoint_path: Path, config: Mapping[str, Any], device: torch.device
) -> LoadedEvaluation:
    """安全重建冻结模型并验证 config、manifest、split 与 transform contract。"""

    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    payload = _safe_load_checkpoint(checkpoint_path)
    fold = payload["fold"]
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError(f"checkpoint fold 非法: {fold!r}")
    run_name = payload["run_name"]
    if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
        raise ValueError(f"checkpoint run_name 不可作为安全输出目录: {run_name!r}")

    model_config = _validated_model_config(payload["model_config"])
    configured_model = _validated_model_config(_require_mapping(config.get("model"), "config.model"))
    if model_config != configured_model:
        raise ValueError("传入 config.model 与 checkpoint 锁定 model_config 不一致")
    configured_train = dict(_require_mapping(config.get("train"), "config.train"))
    checkpoint_train = dict(_require_mapping(payload.get("train_config"), "checkpoint.train_config"))
    for name, value in configured_train.items():
        if name != "epochs" and checkpoint_train.get(name) != value:
            raise ValueError(f"config.train.{name} 与 checkpoint 不一致")
    configured_loss = dict(_require_mapping(config.get("loss"), "config.loss"))
    checkpoint_loss = dict(_require_mapping(payload.get("loss_config"), "checkpoint.loss_config"))
    for name, value in configured_loss.items():
        if name != "lambda_rad" and checkpoint_loss.get(name) != value:
            raise ValueError(f"config.loss.{name} 与 checkpoint 不一致")
    # 训练 CLI 合法地覆盖 epochs/lambda_rad；用 checkpoint 中保存的 resolved
    # train/loss 重建训练时完整 config，再核验其哈希。
    resolved_config = copy.deepcopy(dict(config))
    resolved_config["model"] = dict(_require_mapping(payload["model_config"], "model_config"))
    resolved_config["train"] = checkpoint_train
    resolved_config["loss"] = checkpoint_loss
    expected_config_hash = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if str(payload.get("resolved_config_sha256", "")) != expected_config_hash:
        raise ValueError("传入 resolved config 与 checkpoint SHA256 不一致")
    if str(payload.get("implementation_sha256", "")) != implementation_sha256():
        raise ValueError("当前训练实现与 checkpoint implementation_sha256 不一致")

    bundle = build_bundle(dict(config))
    splits = _validate_checkpoint_splits(payload, bundle, fold)
    transform = _validate_data_contract(payload, config, bundle, splits)

    mode = str(model_config.pop("mode"))
    model = ImageOnlyWorldModel(mode=mode, **model_config)
    state = _require_mapping(payload["model_state"], "checkpoint.model_state")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("checkpoint.model_state 必须只包含 string→Tensor")
    for key, value in state.items():
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise FloatingPointError(f"checkpoint 参数含 NaN/Inf: {key}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError("checkpoint state_dict 与白名单模型架构不一致") from error
    model.requires_grad_(False).eval().to(device)

    contract = model.architecture_contract()
    stored_architecture = dict(
        _require_mapping(payload.get("architecture_contract"), "checkpoint.architecture_contract")
    )
    if stored_architecture != contract:
        raise ValueError("checkpoint architecture_contract 与当前白名单模型不一致")
    if contract["readout_feature"] != FEATURE_SCHEMA:
        raise AssertionError("模型 readout feature contract 发生漂移")
    if contract["forbidden_inputs_absent"] != [
        "clinical",
        "treatment",
        "geometry_descriptor",
        "radiomics",
    ]:
        raise AssertionError("模型禁止输入 contract 发生漂移")
    return LoadedEvaluation(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=file_sha256(checkpoint_path),
        payload=payload,
        model=model,
        bundle=bundle,
        splits=splits,
        radiomics_transform=transform,
        fold=fold,
        run_name=run_name,
        mode=mode,
    )


def _make_loader(
    evaluation: LoadedEvaluation,
    split: str,
    batch_size: int,
    workers: int,
) -> DataLoader:
    records = records_for_ids(evaluation.bundle, evaluation.splits[split])
    if any(record.source != "ispy2" or record.pcr is None for record in records):
        raise ValueError(f"{split} loader 意外包含非 I-SPY2/无标签患者")
    dataset = LongitudinalCacheDataset(
        records,
        transformed_radiomics=None,
        image_channels=evaluation.model.image_channels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=evaluation.model.encoder.features[0].main[0].weight.device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _components_for_prefix(
    model: ImageOnlyWorldModel,
    online: torch.Tensor,
    target: torch.Tensor,
    visits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    transition_output = model.transition(online[:, :visits])[:, -1]
    current = target[:, visits - 1]
    if model.mode == "m0":
        predicted_next = transition_output
        predicted_delta = predicted_next - current
    else:
        predicted_delta = transition_output
        predicted_next = current + predicted_delta
    return current, predicted_next, predicted_delta


def extract_native_split(
    evaluation: LoadedEvaluation,
    split: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> ExtractedSplit:
    """按 checkpoint 中固定顺序提取一个 split；不读取真实未来作为 readout 输入。"""

    if split not in ("train", "val", "test"):
        raise ValueError(f"未知 split: {split}")
    loader = _make_loader(evaluation, split, batch_size, workers)
    feature_parts: dict[str, list[np.ndarray]] = {point: [] for point in DECISION_POINTS}
    current_parts: list[np.ndarray] = []
    target_next_parts: list[np.ndarray] = []
    predicted_next_parts: list[np.ndarray] = []
    target_delta_parts: list[np.ndarray] = []
    predicted_delta_parts: list[np.ndarray] = []
    radiomics_parts: list[np.ndarray] = []
    patient_ids: list[str] = []
    has_radiomics: list[bool] = []

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            online = evaluation.model.encode_online(image)
            target = evaluation.model.encode_target(image)
            batch_current: list[torch.Tensor] = []
            batch_target_next: list[torch.Tensor] = []
            batch_predicted_next: list[torch.Tensor] = []
            batch_target_delta: list[torch.Tensor] = []
            batch_predicted_delta: list[torch.Tensor] = []
            batch_radiomics: list[torch.Tensor] = []
            for index, (point, visits) in enumerate(zip(DECISION_POINTS, AVAILABLE_VISITS)):
                current, predicted_next, predicted_delta = _components_for_prefix(
                    evaluation.model, online, target, visits
                )
                target_next = target[:, index + 1]
                target_delta = target_next - current
                feature = torch.cat((current, predicted_next, predicted_delta), dim=-1)
                feature_parts[point].append(feature.float().cpu().numpy())
                batch_current.append(current)
                batch_target_next.append(target_next)
                batch_predicted_next.append(predicted_next)
                batch_target_delta.append(target_delta)
                batch_predicted_delta.append(predicted_delta)
                if evaluation.model.radiomics_head is not None:
                    batch_radiomics.append(evaluation.model.radiomics_head(predicted_delta))
            current_parts.append(torch.stack(batch_current, dim=1).float().cpu().numpy())
            target_next_parts.append(torch.stack(batch_target_next, dim=1).float().cpu().numpy())
            predicted_next_parts.append(torch.stack(batch_predicted_next, dim=1).float().cpu().numpy())
            target_delta_parts.append(torch.stack(batch_target_delta, dim=1).float().cpu().numpy())
            predicted_delta_parts.append(torch.stack(batch_predicted_delta, dim=1).float().cpu().numpy())
            if batch_radiomics:
                radiomics_parts.append(torch.stack(batch_radiomics, dim=1).float().cpu().numpy())
            batch_patient_ids = [str(value) for value in batch["patient_id"]]
            patient_ids.extend(batch_patient_ids)
            # Dataset 在没有传入 transformed target 时会返回全零 mask；评估阶段的
            # availability 必须来自已审计的 patient overlap，而不是该占位 mask。
            has_radiomics.extend(
                patient_id in evaluation.bundle.raw_radiomics for patient_id in batch_patient_ids
            )

    expected_ids = evaluation.splits[split]
    if patient_ids != expected_ids:
        raise RuntimeError(f"{split} DataLoader patient 顺序与 checkpoint 锁定顺序不一致")
    label_lookup = {record.patient_id: int(record.pcr) for record in evaluation.bundle.primary}
    labels = np.asarray([label_lookup[patient_id] for patient_id in patient_ids], dtype=np.int64)
    features = {point: np.concatenate(parts, axis=0) for point, parts in feature_parts.items()}
    for point, values in features.items():
        expected_shape = (len(patient_ids), evaluation.model.latent_dim * 3)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise FloatingPointError(f"{split}/{point} readout feature 非有限或 shape 错误: {values.shape}")
    radiomics_prediction = np.concatenate(radiomics_parts, axis=0) if radiomics_parts else None
    return ExtractedSplit(
        patient_ids=tuple(patient_ids),
        labels=labels,
        has_radiomics=np.asarray(has_radiomics, dtype=bool),
        features=features,
        current_state=np.concatenate(current_parts, axis=0),
        target_next=np.concatenate(target_next_parts, axis=0),
        predicted_next=np.concatenate(predicted_next_parts, axis=0),
        target_delta=np.concatenate(target_delta_parts, axis=0),
        predicted_delta=np.concatenate(predicted_delta_parts, axis=0),
        radiomics_prediction=radiomics_prediction,
    )


def _make_logistic(penalty: str, c_value: float, random_state: int) -> Any:
    if penalty not in READOUT_PENALTIES or c_value not in READOUT_C_GRID:
        raise ValueError("readout penalty/C 不在预注册 grid")
    arguments: dict[str, Any] = {
        "C": float(c_value),
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": int(random_state),
    }
    # sklearn 1.8 将 penalty= 标为 deprecated；兼容 repo 允许的旧版本。
    if inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated":
        arguments["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        arguments["penalty"] = penalty
    return make_pipeline(StandardScaler(), LogisticRegression(**arguments))


def _binary_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size != 2:
        raise ValueError("validation readout 选择要求两个 pCR 类别")
    return float(roc_auc_score(labels, probabilities))


def select_and_fit_readouts(
    train: ExtractedSplit,
    validation: ExtractedSplit,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """每个 decision point 独立拟合；C/penalty 只看该点 validation AUROC。"""

    readouts: dict[str, Any] = {}
    selections: dict[str, dict[str, Any]] = {}
    if np.unique(train.labels).size != 2:
        raise ValueError("fold-train readout labels 必须同时包含 0/1")
    for point_index, point in enumerate(DECISION_POINTS):
        candidates: list[dict[str, Any]] = []
        selected_model: Any | None = None
        selected_record: dict[str, Any] | None = None
        for penalty in READOUT_PENALTIES:
            for c_value in READOUT_C_GRID:
                model = _make_logistic(penalty, c_value, random_state + point_index)
                model.fit(train.features[point], train.labels)
                probability = model.predict_proba(validation.features[point])[:, 1]
                validation_auroc = _binary_auroc(validation.labels, probability)
                record = {
                    "penalty": penalty,
                    "C": float(c_value),
                    "validation_auroc": validation_auroc,
                }
                candidates.append(record)
                # Grid 顺序是预注册 tie-break：先 l2，再 l1；同 penalty 先较小 C。
                if selected_record is None or validation_auroc > float(
                    selected_record["validation_auroc"]
                ) + 1e-12:
                    selected_record = record
                    selected_model = model
        if selected_model is None or selected_record is None:
            raise RuntimeError(f"{point} readout grid 未产生候选模型")
        readouts[point] = selected_model
        selections[point] = {
            "selected_penalty": selected_record["penalty"],
            "selected_C": selected_record["C"],
            "selected_validation_auroc": selected_record["validation_auroc"],
            "selection_scope": f"{point} 的 C/penalty 仅使用 fold validation；模型仅在 fold train 拟合",
            "tie_break": "validation AUROC 在 1e-12 内并列时，按预注册 grid 顺序：l2 优先，再选择更小 C",
            "candidates": candidates,
        }
    return readouts, selections


def select_youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """只用 validation 概率选择最大 Youden J；确定性处理并列。"""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("threshold labels/probabilities 必须是一维、等长且非空")
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
        raise ValueError("threshold validation labels 必须同时包含 0/1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("threshold probabilities 必须是 [0,1] 内有限值")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    positive = labels == 1
    negative = ~positive
    records: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        prediction = probabilities >= threshold
        sensitivity = float(prediction[positive].mean())
        specificity = float((~prediction[negative]).mean())
        youden = sensitivity + specificity - 1.0
        records.append((float(threshold), youden, sensitivity, specificity))
    best = max(record[1] for record in records)
    ties = [record for record in records if math.isclose(record[1], best, abs_tol=1e-12)]
    distance = min(abs(record[0] - 0.5) for record in ties)
    distance_ties = [
        record for record in ties if math.isclose(abs(record[0] - 0.5), distance, abs_tol=1e-12)
    ]
    chosen = min(distance_ties, key=lambda record: record[0])
    return {
        "threshold": chosen[0],
        "youden_j": chosen[1],
        "sensitivity": chosen[2],
        "specificity": chosen[3],
        "selection_scope": "仅 fold validation",
        "candidate_rule": "validation unique probabilities 加 0/1；prediction = probability >= threshold",
        "tie_break": "Youden J 在 1e-12 内并列时取最接近 0.5，再取较小 threshold",
        "n_candidates": len(candidates),
    }


def fit_thresholds(
    readouts: Mapping[str, Any], validation: ExtractedSplit
) -> dict[str, dict[str, Any]]:
    return {
        point: select_youden_threshold(
            validation.labels, readouts[point].predict_proba(validation.features[point])[:, 1]
        )
        for point in DECISION_POINTS
    }


def prediction_frame(
    evaluation: LoadedEvaluation,
    split_name: str,
    split: ExtractedSplit,
    readouts: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """生成一行一患者一 decision point 的必要 prediction 字段。"""

    rows: list[dict[str, Any]] = []
    for point, visits in zip(DECISION_POINTS, AVAILABLE_VISITS):
        probabilities = np.asarray(
            readouts[point].predict_proba(split.features[point])[:, 1], dtype=np.float64
        )
        threshold = float(thresholds[point]["threshold"])
        for index, patient_id in enumerate(split.patient_ids):
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": evaluation.fold,
                    "split": split_name,
                    "model_name": evaluation.mode,
                    "run_name": evaluation.run_name,
                    "decision_point": point,
                    "y_true": int(split.labels[index]),
                    "predicted_probability": float(probabilities[index]),
                    "predicted_label": int(probabilities[index] >= threshold),
                    "predicted_label_0_5": int(probabilities[index] >= 0.5),
                    "threshold": threshold,
                    "checkpoint": str(evaluation.checkpoint_path),
                    "checkpoint_sha256": evaluation.checkpoint_sha256,
                    "fold_manifest_sha256": evaluation.payload["data_contract"][
                        "fold_manifest_sha256"
                    ],
                    "resolved_config_sha256": evaluation.payload["resolved_config_sha256"],
                    "implementation_sha256": evaluation.payload["implementation_sha256"],
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "has_radiomics": bool(split.has_radiomics[index]),
                    "available_visits": visits,
                    "observed_visits": point,
                    "feature_schema": FEATURE_SCHEMA,
                    "selected_penalty": selections[point]["selected_penalty"],
                    "selected_C": selections[point]["selected_C"],
                }
            )
    return pd.DataFrame(rows)


def _classification_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = probability >= threshold
    positive = labels == 1
    negative = labels == 0
    return {
        "n": int(len(labels)),
        "positive": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "auroc": float(roc_auc_score(labels, probability)) if np.unique(labels).size == 2 else None,
        "auprc": float(average_precision_score(labels, probability)) if positive.any() else None,
        "accuracy": float((prediction == labels).mean()),
        "sensitivity": float(prediction[positive].mean()) if positive.any() else None,
        "specificity": float((~prediction[negative]).mean()) if negative.any() else None,
        "threshold": float(threshold),
    }


def classification_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        output[split_name] = {}
        for point in DECISION_POINTS:
            subset = predictions.loc[
                predictions["split"].eq(split_name) & predictions["decision_point"].eq(point)
            ]
            selected = _classification_metrics(
                subset["y_true"].to_numpy(),
                subset["predicted_probability"].to_numpy(),
                float(subset["threshold"].iloc[0]),
            )
            selected["fixed_0_5_sensitivity_analysis"] = _classification_metrics(
                subset["y_true"].to_numpy(),
                subset["predicted_probability"].to_numpy(),
                0.5,
            )
            output[split_name][point] = selected
    return output


def _featurewise_layer_norm(values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.square(values - mean).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon)


def transition_metric_frame(evaluation: LoadedEvaluation, test: ExtractedSplit) -> pd.DataFrame:
    """保存 test patient×transition 的 learned-vs-copy 记录。"""

    learned_error = np.square(
        _featurewise_layer_norm(test.predicted_next) - _featurewise_layer_norm(test.target_next)
    ).mean(axis=-1)
    copy_error = np.square(
        _featurewise_layer_norm(test.current_state) - _featurewise_layer_norm(test.target_next)
    ).mean(axis=-1)
    learned_raw = np.square(test.predicted_next - test.target_next).mean(axis=-1)
    copy_raw = np.square(test.current_state - test.target_next).mean(axis=-1)
    gain = (copy_error - learned_error) / np.maximum(copy_error, 1e-8)
    predicted_norm = np.linalg.norm(test.predicted_delta, axis=-1)
    target_norm = np.linalg.norm(test.target_delta, axis=-1)
    denominator = predicted_norm * target_norm
    cosine = np.full_like(denominator, np.nan, dtype=np.float64)
    np.divide(
        (test.predicted_delta * test.target_delta).sum(axis=-1),
        denominator,
        out=cosine,
        where=denominator > 0,
    )
    cosine[np.isfinite(cosine)] = np.clip(cosine[np.isfinite(cosine)], -1.0, 1.0)
    rows: list[dict[str, Any]] = []
    for patient_index, patient_id in enumerate(test.patient_ids):
        for transition_index, transition in enumerate(TRANSITIONS):
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": evaluation.fold,
                    "split": "test",
                    "model_name": evaluation.mode,
                    "run_name": evaluation.run_name,
                    "transition": transition,
                    "y_true": int(test.labels[patient_index]),
                    "has_radiomics": bool(test.has_radiomics[patient_index]),
                    "learned_error": float(learned_error[patient_index, transition_index]),
                    "copy_error": float(copy_error[patient_index, transition_index]),
                    "gain": float(gain[patient_index, transition_index]),
                    "learned_raw_mse": float(learned_raw[patient_index, transition_index]),
                    "copy_raw_mse": float(copy_raw[patient_index, transition_index]),
                    "predicted_delta_norm": float(predicted_norm[patient_index, transition_index]),
                    "target_delta_norm": float(target_norm[patient_index, transition_index]),
                    "delta_norm_ratio": float(
                        predicted_norm[patient_index, transition_index]
                        / max(target_norm[patient_index, transition_index], 1e-8)
                    ),
                    "delta_cosine_similarity": float(cosine[patient_index, transition_index]),
                    "error_definition": "independent feature-wise LayerNorm MSE",
                    "checkpoint_sha256": evaluation.checkpoint_sha256,
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                }
            )
    return pd.DataFrame(rows)


def transition_metric_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    """以误差和的比值报告稳定 aggregate gain，避免平均极小分母 ratio。"""

    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("全部", transitions)]
    groups.extend((str(name), group) for name, group in transitions.groupby("transition", sort=False))
    for transition, group in groups:
        learned_sum = float(group["learned_error"].sum())
        copy_sum = float(group["copy_error"].sum())
        learned_raw_sum = float(group["learned_raw_mse"].sum())
        copy_raw_sum = float(group["copy_raw_mse"].sum())
        rows.append(
            {
                "transition": transition,
                "n": int(len(group)),
                "learned_error_sum": learned_sum,
                "copy_error_sum": copy_sum,
                "aggregate_gain": (copy_sum - learned_sum) / max(copy_sum, 1e-8),
                "learned_raw_mse_sum": learned_raw_sum,
                "copy_raw_mse_sum": copy_raw_sum,
                "aggregate_raw_gain": (copy_raw_sum - learned_raw_sum) / max(copy_raw_sum, 1e-8),
                "mean_patient_step_gain": float(group["gain"].mean()),
                "median_patient_step_gain": float(group["gain"].median()),
                "positive_patient_step_gain_fraction": float(group["gain"].gt(0).mean()),
                "aggregate_gain_definition": "(sum(copy_error)-sum(learned_error))/max(sum(copy_error),1e-8)",
            }
        )
    return pd.DataFrame(rows)


def _feature_from_observed(model: ImageOnlyWorldModel, image: torch.Tensor) -> torch.Tensor:
    online = model.encode_online(image)
    target = model.encode_target(image)
    current, predicted_next, predicted_delta = _components_for_prefix(
        model, online, target, image.size(1)
    )
    return torch.cat((current, predicted_next, predicted_delta), dim=-1)


def extract_test_perturbation_features(
    evaluation: LoadedEvaluation,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[tuple[str, ...], dict[tuple[str, str], np.ndarray]]:
    """只改变 test 的可观察 prefix；模型与 readout 均保持原生冻结状态。"""

    loader = _make_loader(evaluation, "test", batch_size, workers)
    parts: dict[tuple[str, str], list[np.ndarray]] = {
        ("repeated_t0", "T0-T1"): [],
        ("repeated_t0", "T0-T2"): [],
        ("temporal_shuffle_t1_t2", "T0-T2"): [],
    }
    patient_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            repeated_two = torch.cat((image[:, :1], image[:, :1]), dim=1)
            repeated_three = torch.cat((image[:, :1], image[:, :1], image[:, :1]), dim=1)
            shuffled_three = image[:, (0, 2, 1)]
            parts[("repeated_t0", "T0-T1")].append(
                _feature_from_observed(evaluation.model, repeated_two).float().cpu().numpy()
            )
            parts[("repeated_t0", "T0-T2")].append(
                _feature_from_observed(evaluation.model, repeated_three).float().cpu().numpy()
            )
            parts[("temporal_shuffle_t1_t2", "T0-T2")].append(
                _feature_from_observed(evaluation.model, shuffled_three).float().cpu().numpy()
            )
            patient_ids.extend(str(value) for value in batch["patient_id"])
    if patient_ids != evaluation.splits["test"]:
        raise RuntimeError("perturbation test patient 顺序与 checkpoint 不一致")
    output = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    if any(not np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("perturbation readout feature 含 NaN/Inf")
    return tuple(patient_ids), output


def perturbation_prediction_frame(
    evaluation: LoadedEvaluation,
    test: ExtractedSplit,
    readouts: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, Any]],
    features: Mapping[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    """用同一个 native readout/validation threshold 生成 test 扰动预测。"""

    native_probability = {
        point: np.asarray(
            readouts[point].predict_proba(test.features[point])[:, 1], dtype=np.float64
        )
        for point in DECISION_POINTS
    }
    rows: list[dict[str, Any]] = []
    for (perturbation, point), values in features.items():
        probability = np.asarray(readouts[point].predict_proba(values)[:, 1], dtype=np.float64)
        threshold = float(thresholds[point]["threshold"])
        native = native_probability[point]
        for index, patient_id in enumerate(test.patient_ids):
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": evaluation.fold,
                    "split": "test",
                    "model_name": evaluation.mode,
                    "run_name": evaluation.run_name,
                    "decision_point": point,
                    "perturbation": perturbation,
                    "y_true": int(test.labels[index]),
                    "predicted_probability": float(probability[index]),
                    "predicted_label": int(probability[index] >= threshold),
                    "predicted_label_0_5": int(probability[index] >= 0.5),
                    "threshold": threshold,
                    "native_predicted_probability": float(native[index]),
                    "native_predicted_label": int(native[index] >= threshold),
                    "probability_change": float(probability[index] - native[index]),
                    "absolute_probability_change": float(abs(probability[index] - native[index])),
                    "same_native_readout": True,
                    "checkpoint": str(evaluation.checkpoint_path),
                    "checkpoint_sha256": evaluation.checkpoint_sha256,
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "has_radiomics": bool(test.has_radiomics[index]),
                    "feature_schema": FEATURE_SCHEMA,
                }
            )
    return pd.DataFrame(rows)


def radiomics_prediction_frame(
    evaluation: LoadedEvaluation, test: ExtractedSplit
) -> pd.DataFrame | None:
    """M2 test paired head 输出；inverse 后仍处于 log/absolute transformed-change 单位。"""

    if evaluation.mode != "m2":
        return None
    prediction = test.radiomics_prediction
    if prediction is None or prediction.shape != (len(test.patient_ids), 3, len(FEATURE_NAMES)):
        raise RuntimeError("M2 radiomics prediction shape 错误")
    transformed = evaluation.radiomics_transform.transform_all(evaluation.bundle.raw_radiomics)
    rows: list[dict[str, Any]] = []
    for patient_index, patient_id in enumerate(test.patient_ids):
        if patient_id not in transformed:
            continue
        target, mask = transformed[patient_id]
        for transition_index, transition in enumerate(TRANSITIONS):
            for feature_index, feature_name in enumerate(FEATURE_NAMES):
                target_standardized = float(target[transition_index, feature_index])
                predicted_standardized = float(prediction[patient_index, transition_index, feature_index])
                spec = evaluation.radiomics_transform.features[feature_index]
                target_change = float(
                    evaluation.radiomics_transform.inverse_feature(feature_index, target_standardized)
                )
                predicted_change = float(
                    evaluation.radiomics_transform.inverse_feature(feature_index, predicted_standardized)
                )
                rows.append(
                    {
                        "patient_id": patient_id,
                        "fold": evaluation.fold,
                        "split": "test",
                        "model_name": evaluation.mode,
                        "run_name": evaluation.run_name,
                        "transition": transition,
                        "feature_name": feature_name,
                        "target_standardized": target_standardized,
                        "predicted_standardized": predicted_standardized,
                        "target_change": target_change,
                        "predicted_change": predicted_change,
                        "valid_mask": bool(mask[transition_index, feature_index]),
                        "transformed_change_unit": spec.value_transform,
                        "epsilon": spec.epsilon,
                        "checkpoint_sha256": evaluation.checkpoint_sha256,
                        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    }
                )
    if not rows:
        raise RuntimeError("M2 test split 没有 paired radiomics 患者")
    return pd.DataFrame(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        frame.to_csv(stream, index=False)


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_evaluation(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device_name: str,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    """执行单 checkpoint/fold 核心评估，并返回新建输出路径。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if workers < 0:
        raise ValueError("workers 不得小于 0")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 评估，但当前 CUDA 不可用")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index 越界: {device}")

    evaluation = load_evaluation(checkpoint_path, config, device)
    evaluator_hash = evaluation_implementation_sha256()
    namespace = (
        f"{evaluation.checkpoint_sha256[:12]}_eval{evaluator_hash[:12]}_"
        f"epoch{int(evaluation.payload.get('epoch', 0))}"
    )
    prediction_dir = (
        EXPERIMENT_ROOT / "predictions" / evaluation.run_name / f"fold_{evaluation.fold}" / namespace
    )
    metric_dir = EXPERIMENT_ROOT / "metrics" / "evaluation" / evaluation.run_name / f"fold_{evaluation.fold}" / namespace
    if prediction_dir.exists() or metric_dir.exists():
        raise FileExistsError(
            "该 checkpoint 的评估输出已存在；为避免覆盖，未运行。"
            f" predictions={prediction_dir}, metrics={metric_dir}"
        )
    # 先完成 train 拟合与 validation-only 选择，再首次读取 test cache。这样不仅
    # 数学上不引用 test，执行顺序本身也留下清晰的 leakage barrier。
    extracted: dict[str, ExtractedSplit] = {
        split: extract_native_split(evaluation, split, device, batch_size, workers)
        for split in ("train", "val")
    }
    random_state = int(
        _require_mapping(evaluation.payload.get("train_config"), "train_config").get("seed", 2026)
    ) + evaluation.fold * 1009
    readouts, selections = select_and_fit_readouts(
        extracted["train"], extracted["val"], random_state
    )
    thresholds = fit_thresholds(readouts, extracted["val"])
    extracted["test"] = extract_native_split(
        evaluation, "test", device, batch_size, workers
    )
    prediction_frames = {
        split: prediction_frame(
            evaluation, split, extracted[split], readouts, selections, thresholds
        )
        for split in ("train", "val", "test")
    }
    native_predictions = pd.concat(prediction_frames.values(), ignore_index=True)
    transition_metrics = transition_metric_frame(evaluation, extracted["test"])
    transition_summary = transition_metric_summary(transition_metrics)
    perturbation_ids, perturbation_features = extract_test_perturbation_features(
        evaluation, device, batch_size, workers
    )
    if perturbation_ids != extracted["test"].patient_ids:
        raise RuntimeError("native 与 perturbation test patient 顺序不一致")
    perturbation_predictions = perturbation_prediction_frame(
        evaluation, extracted["test"], readouts, thresholds, perturbation_features
    )
    radiomics_predictions = radiomics_prediction_frame(evaluation, extracted["test"])

    prediction_dir.mkdir(parents=True, exist_ok=False)
    metric_dir.mkdir(parents=True, exist_ok=False)

    for split, frame in prediction_frames.items():
        _write_csv_new(prediction_dir / f"{split}_predictions.csv", frame)
    _write_csv_new(prediction_dir / "native_predictions.csv", native_predictions)
    _write_csv_new(prediction_dir / "test_perturbation_predictions.csv", perturbation_predictions)
    _write_csv_new(metric_dir / "test_transition_metrics.csv", transition_metrics)
    _write_csv_new(metric_dir / "test_transition_summary.csv", transition_summary)
    if radiomics_predictions is not None:
        _write_csv_new(prediction_dir / "test_paired_radiomics_predictions.csv", radiomics_predictions)

    parameters: dict[str, Any] = {
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": int(extracted["train"].features["T0"].shape[1]),
        "by_decision_point": {},
    }
    for point in DECISION_POINTS:
        classifier = readouts[point].steps[-1][1]
        scaler = readouts[point].steps[0][1]
        parameters["by_decision_point"][point] = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "logistic_coef": classifier.coef_.tolist(),
            "logistic_intercept": classifier.intercept_.tolist(),
            "classes": classifier.classes_.tolist(),
            "selected_penalty": selections[point]["selected_penalty"],
            "selected_C": selections[point]["selected_C"],
        }
    _write_json_new(metric_dir / "readout_parameters.json", parameters)
    metrics_payload = {
        "schema_version": 1,
        "run_name": evaluation.run_name,
        "model_name": evaluation.mode,
        "fold": evaluation.fold,
        "checkpoint": str(evaluation.checkpoint_path),
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "fold_manifest_sha256": evaluation.payload["data_contract"]["fold_manifest_sha256"],
        "resolved_config_sha256": evaluation.payload["resolved_config_sha256"],
        "implementation_sha256": evaluation.payload["implementation_sha256"],
        "evaluation_implementation_sha256": evaluator_hash,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "radiomics_raw_targets_sha256": evaluation.radiomics_transform.raw_targets_sha256,
        "architecture_contract": evaluation.model.architecture_contract(),
        "config_model_matches_checkpoint": True,
        "split_counts": {name: len(extracted[name].patient_ids) for name in ("train", "val", "test")},
        "primary_patient_count": sum(len(extracted[name].patient_ids) for name in ("train", "val", "test")),
        "cache_access": "只读锁定 808 名 I-SPY2 primary cache；未读取 I-SPY1 cache",
        "feature_schema": FEATURE_SCHEMA,
        "readout_selection": selections,
        "validation_thresholds": thresholds,
        "classification": classification_summary(native_predictions),
        "test_transition_rows": len(transition_metrics),
        "test_transition_aggregate": transition_summary.to_dict(orient="records"),
        "test_perturbation_rows": len(perturbation_predictions),
        "test_paired_radiomics_rows": 0 if radiomics_predictions is None else len(radiomics_predictions),
        "leakage_guards": {
            "image_model_frozen": True,
            "readout_fit_scope": "fold train only",
            "penalty_C_selection_scope": "fold validation only",
            "threshold_selection_scope": "fold validation only",
            "test_used_for_selection": False,
            "clinical_treatment_geometry_radiomics_in_readout": False,
            "perturbation_uses_native_readout": True,
        },
        "outputs": {
            "prediction_dir": str(prediction_dir),
            "metric_dir": str(metric_dir),
        },
    }
    _write_json_new(metric_dir / "evaluation_summary.json", metrics_payload)
    return {
        "status": "完成",
        "run_name": evaluation.run_name,
        "mode": evaluation.mode,
        "fold": evaluation.fold,
        "prediction_dir": str(prediction_dir),
        "metric_dir": str(metric_dir),
        "selected_readouts": {
            point: {
                "penalty": selections[point]["selected_penalty"],
                "C": selections[point]["selected_C"],
            }
            for point in DECISION_POINTS
        },
    }
