#!/usr/bin/env python3
"""仅用 fold 0 validation 为 M2 选择预注册的 radiomics loss 权重。

本脚本故意不提供 test split 参数，也不调用完整 evaluator。它只通过
``extract_native_split(..., "val", ...)`` 提取四个 M2 候选的 validation
表征，并用对应 fold-train-only radiomics transform 计算 standardized 指标。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rnc.config import load_config  # noqa: E402
from rnc.data import FEATURE_NAMES, TRANSITIONS, patient_hash  # noqa: E402
from rnc.evaluation import extract_native_split, load_evaluation  # noqa: E402
from rnc.transforms import RadiomicsChangeTransform, raw_targets_hash  # noqa: E402


FOLD = 0
M2_LAMBDA_GRID = (0.05, 0.1, 0.25, 0.5)
IMAGE_DEGRADATION_LIMIT = 0.05
MAE_TIE_RELATIVE_TOLERANCE = 0.01
NUMERICAL_ZERO = 1e-12
DELTA_NORM_NUMERICAL_ZERO = 1e-8
HISTORY_RECOMPUTE_RTOL = 1e-6
HISTORY_RECOMPUTE_ATOL = 5e-7

DEFAULT_CSV = EXPERIMENT_ROOT / "metrics" / "m2_lambda_selection.csv"
DEFAULT_JSON = EXPERIMENT_ROOT / "metrics" / "m2_lambda_selection.json"
DEFAULT_REPORT = EXPERIMENT_ROOT / "reports" / "m2_lambda_selection.md"

HISTORY_FIELDS = (
    "val_raw_next_mse",
    "val_copy_mse",
    "val_aggregate_transition_gain",
    "val_normalized_next_mse",
    "val_normalized_copy_mse",
    "val_normalized_error_aggregate_gain",
    "val_state_loss",
    "val_delta_loss",
    "val_predicted_delta_norm",
    "val_target_delta_norm",
    "val_delta_cosine",
    "val_visit_state_std",
    "val_visit_feature_std",
    "train_first_batch_image_task_gradient_norm",
    "train_first_batch_radiomics_shared_gradient_norm_raw",
    "train_first_batch_radiomics_head_gradient_norm_raw",
    "train_first_batch_radiomics_shared_gradient_norm_weighted",
    "train_first_batch_radiomics_head_gradient_norm_weighted",
    "train_first_batch_weighted_radiomics_to_image_gradient_ratio",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} 不是文件: {resolved}")
    return resolved


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 mapping")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} 不能是 bool")
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} 必须是数值，实际为 {value!r}") from error
    if not math.isfinite(output):
        raise FloatingPointError(f"{name} 非有限: {output}")
    return output


def _history_path(run_name: str) -> Path:
    return EXPERIMENT_ROOT / "metrics" / "training" / run_name / f"fold_{FOLD}.csv"


def read_best_history(
    path: Path,
    *,
    evaluation: Any,
    best_epoch: int,
    expected_mode: str,
    expected_lambda: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """读取 checkpoint 所指 best epoch，而不是重新从 history 选择 epoch。"""

    path = _resolved_file(path, "training history")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"training history 为空: {path}")
    parsed_rows: list[tuple[int, dict[str, str]]] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            epoch = int(float(row.get("epoch", "nan")))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"history 第 {row_index} 行 epoch 非法: {row.get('epoch')!r}")
        parsed_rows.append((epoch, row))
    epochs = [epoch for epoch, _ in parsed_rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"history epoch 必须从 1 连续且唯一，实际为 {epochs}")
    matches = [row for epoch, row in parsed_rows if epoch == best_epoch]
    if len(matches) != 1:
        raise ValueError(
            f"history 中 best epoch={best_epoch} 必须恰好一行，实际 {len(matches)} 行: {path}"
        )
    row = matches[0]
    required_columns = set(HISTORY_FIELDS) | {"validation_selection_metric"}
    missing = sorted(name for name in required_columns if name not in row)
    if missing:
        raise ValueError(f"history 缺少预注册指标字段: {missing}")
    minimum_latent_std = _finite_number(
        _as_mapping(evaluation.payload.get("train_config"), "checkpoint.train_config").get(
            "min_latent_std"
        ),
        "checkpoint.train_config.min_latent_std",
    )
    eligible_selection: list[tuple[float, int]] = []
    for epoch, candidate in parsed_rows:
        if candidate.get("mode") != expected_mode:
            raise ValueError(
                f"history epoch {epoch} mode 与 checkpoint 不一致: "
                f"{candidate.get('mode')!r} != {expected_mode!r}"
            )
        fold = int(_finite_number(candidate.get("fold"), f"history[{epoch}].fold"))
        if fold != FOLD:
            raise ValueError(f"history epoch {epoch} fold 必须为 0，实际为 {fold}")
        history_lambda = _finite_number(
            candidate.get("lambda_rad"), f"history[{epoch}].lambda_rad"
        )
        if not math.isclose(history_lambda, expected_lambda, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"history epoch {epoch} lambda_rad 与 checkpoint 不一致: "
                f"{history_lambda} != {expected_lambda}"
            )
        selection_metric = _finite_number(
            candidate.get("validation_selection_metric"),
            f"history[{epoch}].validation_selection_metric",
        )
        aggregate_gain = _finite_number(
            candidate.get("val_aggregate_transition_gain"),
            f"history[{epoch}].val_aggregate_transition_gain",
        )
        if not math.isclose(selection_metric, -aggregate_gain, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"history epoch {epoch} selection metric 不是 -val aggregate gain"
            )
        feature_std = _finite_number(
            candidate.get("val_visit_feature_std"), f"history[{epoch}].val_visit_feature_std"
        )
        if feature_std >= minimum_latent_std:
            eligible_selection.append((selection_metric, epoch))

    if not eligible_selection:
        raise ValueError("history 没有满足 minimum latent std 的 epoch")
    derived_metric, derived_epoch = min(eligible_selection, key=lambda item: (item[0], item[1]))
    checkpoint_objective = _finite_number(
        evaluation.payload.get("best_validation_objective"),
        "checkpoint.best_validation_objective",
    )
    best_row_metric = _finite_number(
        row.get("validation_selection_metric"), "history.best.validation_selection_metric"
    )
    if derived_epoch != best_epoch:
        raise ValueError(
            f"checkpoint best epoch 与完整 history 的预注册选择不一致: {best_epoch} != {derived_epoch}"
        )
    if not math.isclose(best_row_metric, checkpoint_objective, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("checkpoint best_validation_objective 与 best history row 不一致")
    if not math.isclose(derived_metric, checkpoint_objective, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("checkpoint objective 与由完整 history 重算的最优值不一致")

    # 缺字段属于 provenance/格式错误；NaN/Inf 则保留给候选排除规则处理。
    output: dict[str, float] = {}
    for name in HISTORY_FIELDS:
        try:
            output[name] = float(row[name])
        except (TypeError, ValueError) as error:
            raise TypeError(f"history.{name} 不是数值: {row[name]!r}") from error
    output["validation_selection_metric"] = best_row_metric
    audit = {
        "row_count": len(rows),
        "last_epoch": epochs[-1],
        "derived_best_epoch": derived_epoch,
        "derived_best_validation_objective": derived_metric,
        "checkpoint_best_validation_objective": checkpoint_objective,
        "selection_metric_contract": "-val_aggregate_transition_gain",
    }
    return output, audit


def _safe_spearman(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if target.size < 2 or np.ptp(target) <= NUMERICAL_ZERO or np.ptp(prediction) <= NUMERICAL_ZERO:
        return None
    value = float(spearmanr(target, prediction).statistic)
    return value if math.isfinite(value) else None


def _validate_resolved_run(
    evaluation: Any, config: Mapping[str, Any], expected_lambda: float
) -> dict[str, Any]:
    """把 checkpoint 绑定到同目录 resolved_run.json 与运行时有效样本。"""

    path = _resolved_file(evaluation.checkpoint_path.parent / "resolved_run.json", "resolved_run")
    payload = _as_mapping(json.loads(path.read_text(encoding="utf-8")), "resolved_run")
    runtime = _as_mapping(evaluation.payload.get("runtime"), "checkpoint.runtime")
    train_config = _as_mapping(evaluation.payload.get("train_config"), "checkpoint.train_config")
    effective_train_ids = [str(value) for value in runtime.get("effective_train_ids", [])]
    expected = {
        "run_name": evaluation.run_name,
        "fold": evaluation.fold,
        "mode": evaluation.mode,
        "seed": int(train_config["seed"]) + evaluation.fold,
        "lambda_rad": expected_lambda,
        "train_samples": len(effective_train_ids),
        "train_ispy2_samples": sum(
            patient_id.startswith(("ISPY2-", "ACRIN-6698-"))
            for patient_id in effective_train_ids
        ),
        "train_ispy1_samples": sum(
            patient_id.startswith("ISPY1_") for patient_id in effective_train_ids
        ),
        "validation_samples": len(runtime.get("effective_validation_ids", [])),
        "smoke_patients": None,
        "epochs_requested": int(runtime.get("epochs_requested")),
        "device": runtime.get("device"),
        "architecture_contract": dict(
            _as_mapping(evaluation.payload.get("architecture_contract"), "architecture_contract")
        ),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"resolved_run.{field} 与 checkpoint/runtime 不一致: {payload.get(field)!r} != {value!r}"
            )
    if not math.isclose(
        _finite_number(payload.get("lambda_rad"), "resolved_run.lambda_rad"),
        expected_lambda,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("resolved_run lambda_rad 与 checkpoint 不一致")
    return {"path": str(path), "sha256": file_sha256(path), "validated_fields": sorted(expected)}


def _validate_runtime_contract(evaluation: Any) -> dict[str, Any]:
    runtime = _as_mapping(evaluation.payload.get("runtime"), "checkpoint.runtime")
    data_contract = _as_mapping(evaluation.payload.get("data_contract"), "checkpoint.data_contract")
    train_ids = [str(value) for value in runtime.get("effective_train_ids", [])]
    val_ids = [str(value) for value in runtime.get("effective_validation_ids", [])]
    transform_fit_ids = [str(value) for value in runtime.get("transform_fit_ids", [])]
    if train_ids != evaluation.splits["pretrain_train"]:
        raise ValueError("runtime.effective_train_ids 与锁定 pretrain_train IDs/顺序不一致")
    if val_ids != evaluation.splits["val"]:
        raise ValueError("runtime.effective_validation_ids 与锁定 val IDs/顺序不一致")
    if transform_fit_ids != evaluation.splits["train"]:
        raise ValueError("runtime.transform_fit_ids 与锁定 fold train IDs/顺序不一致")
    expected_train_hash = patient_hash(train_ids)
    expected_val_hash = patient_hash(val_ids)
    if runtime.get("effective_train_patient_hash") != expected_train_hash:
        raise ValueError("runtime effective train hash 与 IDs 不一致")
    if runtime.get("effective_validation_patient_hash") != expected_val_hash:
        raise ValueError("runtime effective validation hash 与 IDs 不一致")
    transform_fit_hash = patient_hash(transform_fit_ids)
    if data_contract.get("train_patient_hash") != transform_fit_hash:
        raise ValueError("runtime transform-fit IDs 与 data contract train hash 不一致")
    extra_ids = set(train_ids) - set(transform_fit_ids)
    if data_contract.get("extra_pretrain_patient_hash") != patient_hash(extra_ids):
        raise ValueError("runtime effective train 的 extra IDs 与 data contract hash 不一致")
    checkpoint_lambda = _checkpoint_lambda(evaluation)
    if not math.isclose(
        _finite_number(runtime.get("lambda_rad"), "runtime.lambda_rad"),
        checkpoint_lambda,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("runtime.lambda_rad 与 checkpoint loss_config 不一致")
    return {
        "effective_train_count": len(train_ids),
        "effective_train_patient_hash": expected_train_hash,
        "effective_validation_count": len(val_ids),
        "effective_validation_patient_hash": expected_val_hash,
        "transform_fit_patient_count": len(transform_fit_ids),
        "transform_fit_patient_hash": transform_fit_hash,
        "extra_pretrain_patient_count": len(extra_ids),
        "extra_pretrain_patient_hash": patient_hash(extra_ids),
    }


def _validate_transform_refit(evaluation: Any) -> dict[str, Any]:
    """在 selector 内独立按 fold-train IDs 重拟合并逐字段比较 artifact。"""

    raw = evaluation.bundle.raw_radiomics
    train_raw = {
        patient_id: raw[patient_id]
        for patient_id in evaluation.splits["train"]
        if patient_id in raw
    }
    refit = RadiomicsChangeTransform.fit(
        train_raw, evaluation.splits["train"], evaluation.fold
    )
    stored = evaluation.radiomics_transform
    refit_payload = refit.to_dict()
    stored_payload = stored.to_dict()
    # 训练时 artifact 的 raw_targets_sha256 锁定完整 375 人 raw source；这里的
    # selector refit 明确只接收 train 子集，故 source-universe hash 预期不同。
    # 除该来源范围 hash 外，所有拟合字段/feature 参数必须逐字段完全相等；两个
    # source hash 又分别独立验证，避免用 val/test 值重拟合参数。
    refit_train_raw_hash = str(refit_payload.pop("raw_targets_sha256"))
    stored_full_raw_hash = str(stored_payload.pop("raw_targets_sha256"))
    if refit_payload != stored_payload:
        raise ValueError("selector 独立 train-subset 重拟合结果与保存 transform 拟合参数不一致")
    if refit_train_raw_hash != raw_targets_hash(train_raw):
        raise ValueError("selector train-subset raw hash 重算不一致")
    contract = _as_mapping(evaluation.payload.get("data_contract"), "checkpoint.data_contract")
    transform_path = _resolved_file(Path(str(contract.get("radiomics_transform"))), "transform artifact")
    artifact_payload = _as_mapping(
        json.loads(transform_path.read_text(encoding="utf-8")), "transform artifact"
    )
    if dict(artifact_payload) != stored.to_dict():
        raise ValueError("transform artifact JSON 与 load 后的全部字段不一致")
    actual_hash = file_sha256(transform_path)
    if actual_hash != contract.get("radiomics_transform_sha256"):
        raise ValueError("transform artifact hash 与 checkpoint contract 不一致")
    if stored_full_raw_hash != raw_targets_hash(raw):
        raise ValueError("transform raw targets hash 与锁定 raw radiomics 数据不一致")
    return {
        "artifact_path": str(transform_path),
        "artifact_sha256": actual_hash,
        "spec_version": stored.spec_version,
        "raw_targets_sha256": stored.raw_targets_sha256,
        "refit_train_subset_raw_targets_sha256": refit_train_raw_hash,
        "train_patient_hash": stored.train_patient_hash,
        "train_patient_count": stored.train_patient_count,
        "paired_train_patient_count": stored.paired_train_patient_count,
        "quantiles": list(stored.quantiles),
        "all_fit_parameters_refit_equal": True,
        "artifact_all_fields_equal": True,
    }


def recompute_validation_image_metrics(extracted: Any) -> dict[str, Any]:
    """只由 extract_native_split 的 val tensor 独立重算 image gate/坍塌量。"""

    arrays = {
        "current_state": np.asarray(extracted.current_state, dtype=np.float32),
        "target_next": np.asarray(extracted.target_next, dtype=np.float32),
        "predicted_next": np.asarray(extracted.predicted_next, dtype=np.float32),
        "target_delta": np.asarray(extracted.target_delta, dtype=np.float32),
        "predicted_delta": np.asarray(extracted.predicted_delta, dtype=np.float32),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"validation image arrays shape 不一致: {shapes}")
    shape = next(iter(shapes.values()))
    if len(shape) != 3 or shape[1] != len(TRANSITIONS) or shape[0] != len(
        extracted.patient_ids
    ):
        raise ValueError(f"validation image array 不是 [patient,3,latent]: {shape}")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("validation extracted image array 含 NaN/Inf")

    current = torch.from_numpy(arrays["current_state"])
    target_next = torch.from_numpy(arrays["target_next"])
    predicted_next = torch.from_numpy(arrays["predicted_next"])
    target_delta = torch.from_numpy(arrays["target_delta"])
    predicted_delta = torch.from_numpy(arrays["predicted_delta"])
    latent_dim = int(shape[-1])
    normalized_prediction = F.layer_norm(predicted_next, (latent_dim,))
    normalized_target = F.layer_norm(target_next, (latent_dim,))
    normalized_current = F.layer_norm(current, (latent_dim,))
    normalized_error_per_step = (normalized_prediction - normalized_target).square().mean(-1)
    normalized_copy_per_step = (normalized_current - normalized_target).square().mean(-1)
    raw_error_per_step = (predicted_next - target_next).square().mean(-1)
    raw_copy_per_step = (current - target_next).square().mean(-1)
    step_weights = torch.tensor((2.0, 1.0, 0.5), dtype=torch.float32)
    step_weights = step_weights / step_weights.mean()
    delta_per_step = F.smooth_l1_loss(
        predicted_delta, target_delta, reduction="none"
    ).mean(-1)

    # current 依次是 T0/T1/T2；target_next 依次是 T1/T2/T3。相邻重复状态
    # 应 bitwise 一致，否则 extraction schema 已漂移。
    if not torch.equal(current[:, 1:], target_next[:, :-1]):
        maximum = float((current[:, 1:] - target_next[:, :-1]).abs().max())
        raise ValueError(f"validation target-state 邻接重复不一致，max abs diff={maximum}")
    target_visits = torch.cat((current[:, :1], target_next), dim=1)

    coordinate_variance = predicted_delta.var(dim=0, unbiased=False)
    per_transition: dict[str, dict[str, Any]] = {}
    for transition_index, transition in enumerate(TRANSITIONS):
        values = coordinate_variance[transition_index]
        collapsed = torch.nonzero(values <= NUMERICAL_ZERO, as_tuple=False).flatten().tolist()
        per_transition[transition] = {
            "latent_coordinates": latent_dim,
            "minimum_patient_variance": float(values.min()),
            "maximum_patient_variance": float(values.max()),
            "mean_patient_variance": float(values.mean()),
            "collapsed_coordinate_count": len(collapsed),
            "collapsed_coordinate_indices": [int(value) for value in collapsed],
        }
    variance_bytes = coordinate_variance.numpy().astype("<f4", copy=False).tobytes()
    variance_hash = hashlib.sha256(variance_bytes).hexdigest()

    raw_next = float(raw_error_per_step.mean())
    raw_copy = float(raw_copy_per_step.mean())
    normalized_next = float(normalized_error_per_step.mean())
    normalized_copy = float(normalized_copy_per_step.mean())
    output: dict[str, Any] = {
        "patients": int(shape[0]),
        "latent_dim": latent_dim,
        "transition_cells": int(shape[0] * shape[1]),
        "raw_next_mse": raw_next,
        "copy_mse": raw_copy,
        "aggregate_transition_gain": (raw_copy - raw_next) / max(raw_copy, 1e-8),
        "normalized_next_mse": normalized_next,
        "normalized_copy_mse": normalized_copy,
        "normalized_error_aggregate_gain": (normalized_copy - normalized_next)
        / max(normalized_copy, 1e-8),
        "state_loss": float((normalized_error_per_step * step_weights).mean()),
        "delta_loss": float((delta_per_step * step_weights).mean()),
        "predicted_delta_norm": float(predicted_delta.norm(dim=-1).mean()),
        "target_delta_norm": float(target_delta.norm(dim=-1).mean()),
        "delta_cosine": float(F.cosine_similarity(predicted_delta, target_delta, dim=-1).mean()),
        "predicted_delta_pooled_variance": float(predicted_delta.var(unbiased=False)),
        "predicted_delta_transition_latent_variance_sha256": variance_hash,
        "predicted_delta_collapsed_coordinate_count": sum(
            value["collapsed_coordinate_count"] for value in per_transition.values()
        ),
        "predicted_delta_variance_by_transition": per_transition,
        # History 的 visit_* std 是 online_state；ExtractedSplit 未返回 online_state。
        # 这里重算的是可观测 EMA target-state 版本，不能伪装成同一统计量。
        "target_visit_state_std": float(target_visits.std(unbiased=False)),
        "target_visit_feature_std": float(target_visits.std(dim=0, unbiased=False).mean()),
        "online_visit_state_std_recomputable_from_extracted_split": False,
        "online_visit_feature_std_recomputable_from_extracted_split": False,
    }
    return output


def compare_recomputed_image_history(
    recomputed: Mapping[str, Any], history: Mapping[str, float]
) -> dict[str, Any]:
    """严格交叉核验所有可由 frozen val outputs 闭合的 history 量。"""

    fields = (
        "raw_next_mse",
        "copy_mse",
        "aggregate_transition_gain",
        "normalized_next_mse",
        "normalized_copy_mse",
        "normalized_error_aggregate_gain",
        "state_loss",
        "delta_loss",
        "predicted_delta_norm",
        "target_delta_norm",
        "delta_cosine",
    )
    comparisons: dict[str, Any] = {}
    for name in fields:
        recomputed_value = _finite_number(recomputed.get(name), f"recomputed.{name}")
        history_value = _finite_number(history.get(f"val_{name}"), f"history.val_{name}")
        absolute_difference = abs(recomputed_value - history_value)
        allowed = HISTORY_RECOMPUTE_ATOL + HISTORY_RECOMPUTE_RTOL * abs(history_value)
        if absolute_difference > allowed:
            raise ValueError(
                f"recomputed val {name} 与 best history 超出严格容差: "
                f"{recomputed_value} vs {history_value}; diff={absolute_difference}, allowed={allowed}"
            )
        comparisons[name] = {
            "recomputed": recomputed_value,
            "history": history_value,
            "absolute_difference": absolute_difference,
            "allowed_absolute_difference": allowed,
            "within_tolerance": True,
        }
    return {
        "rtol": HISTORY_RECOMPUTE_RTOL,
        "atol": HISTORY_RECOMPUTE_ATOL,
        "fields": comparisons,
        "all_recomputable_fields_match": True,
        "not_recomputable_from_extracted_split": {
            "val_visit_state_std": "history 使用 online_state；ExtractedSplit 只返回 EMA target states",
            "val_visit_feature_std": "history 使用 online_state；ExtractedSplit 只返回 EMA target states",
        },
    }


def radiomics_validation_metrics(evaluation: Any, extracted: Any) -> dict[str, Any]:
    """只在 validation 配对患者/有效元素上计算 standardized head 指标。"""

    prediction = extracted.radiomics_prediction
    if prediction is None:
        raise ValueError("M2 checkpoint 没有 radiomics_prediction")
    prediction = np.asarray(prediction, dtype=np.float64)
    expected_shape = (len(extracted.patient_ids), 3, len(FEATURE_NAMES))
    if prediction.shape != expected_shape:
        raise ValueError(f"radiomics prediction shape 错误: {prediction.shape} != {expected_shape}")

    # 只构造 extract_native_split 实际返回的 validation ID 子集；禁止对 375 人
    # raw dict 调用 transform_all，也不对 train/test 目标计算候选指标。
    validation_raw = {
        patient_id: evaluation.bundle.raw_radiomics[patient_id]
        for patient_id in extracted.patient_ids
        if patient_id in evaluation.bundle.raw_radiomics
    }
    transformed = {
        patient_id: evaluation.radiomics_transform.transform_one(raw)
        for patient_id, raw in validation_raw.items()
    }
    target = np.zeros(expected_shape, dtype=np.float64)
    mask = np.zeros(expected_shape, dtype=bool)
    for patient_index, patient_id in enumerate(extracted.patient_ids):
        if patient_id not in transformed:
            continue
        patient_target, patient_mask = transformed[patient_id]
        target[patient_index] = np.asarray(patient_target, dtype=np.float64)
        mask[patient_index] = np.asarray(patient_mask, dtype=bool)

    paired_from_mask = mask.reshape(mask.shape[0], -1).any(axis=1)
    if not np.array_equal(paired_from_mask, np.asarray(extracted.has_radiomics, dtype=bool)):
        raise ValueError("validation radiomics mask 与已审计 patient overlap 不一致")
    if not mask.any():
        raise ValueError("fold 0 validation 没有配对 radiomics 元素")
    if not np.isfinite(target[mask]).all():
        raise FloatingPointError("validation standardized radiomics target 含 NaN/Inf")

    finite_prediction = np.isfinite(prediction[mask])
    nonfinite_prediction_count = int((~finite_prediction).sum())
    result: dict[str, Any] = {
        "paired_patients": int(paired_from_mask.sum()),
        "valid_elements": int(mask.sum()),
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "predicted_delta_variance": float(
            np.var(np.asarray(extracted.predicted_delta, dtype=np.float64))
        ),
        "target_patient_ids": list(validation_raw),
        "target_patient_hash": patient_hash(validation_raw),
        "features": {},
        "feature_transitions": {},
    }
    if nonfinite_prediction_count:
        result.update(
            {
                "standardized_mae": None,
                "standardized_rmse": None,
                "prediction_variance": None,
                "target_variance": float(np.var(target[mask])),
                "prediction_target_variance_ratio": None,
            }
        )
        for feature in FEATURE_NAMES:
            result["features"][feature] = {
                "n": 0,
                "standardized_mae": None,
                "standardized_rmse": None,
                "spearman": None,
                "prediction_variance": None,
                "target_variance": None,
                "prediction_target_variance_ratio": None,
                "prediction_collapsed": None,
            }
        for transition in TRANSITIONS:
            result["feature_transitions"][transition] = {}
            for feature in FEATURE_NAMES:
                result["feature_transitions"][transition][feature] = {
                    "n": 0,
                    "standardized_mae": None,
                    "standardized_rmse": None,
                    "spearman": None,
                    "prediction_variance": None,
                    "target_variance": None,
                    "prediction_target_variance_ratio": None,
                    "prediction_collapsed": None,
                }
        return result

    valid_prediction = prediction[mask]
    valid_target = target[mask]
    error = valid_prediction - valid_target
    target_variance = float(np.var(valid_target))
    prediction_variance = float(np.var(valid_prediction))
    result.update(
        {
            "standardized_mae": float(np.mean(np.abs(error))),
            "standardized_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "prediction_variance": prediction_variance,
            "target_variance": target_variance,
            "prediction_target_variance_ratio": prediction_variance
            / max(target_variance, NUMERICAL_ZERO),
        }
    )
    for feature_index, feature in enumerate(FEATURE_NAMES):
        feature_mask = mask[:, :, feature_index]
        feature_prediction = prediction[:, :, feature_index][feature_mask]
        feature_target = target[:, :, feature_index][feature_mask]
        feature_error = feature_prediction - feature_target
        feature_prediction_variance = float(np.var(feature_prediction))
        feature_target_variance = float(np.var(feature_target))
        result["features"][feature] = {
            "n": int(feature_mask.sum()),
            "standardized_mae": float(np.mean(np.abs(feature_error))),
            "standardized_rmse": float(np.sqrt(np.mean(np.square(feature_error)))),
            "spearman": _safe_spearman(feature_target, feature_prediction),
            "prediction_variance": feature_prediction_variance,
            "target_variance": feature_target_variance,
            "prediction_target_variance_ratio": feature_prediction_variance
            / max(feature_target_variance, NUMERICAL_ZERO),
            "prediction_collapsed": feature_prediction_variance <= NUMERICAL_ZERO,
        }
    for transition_index, transition in enumerate(TRANSITIONS):
        result["feature_transitions"][transition] = {}
        for feature_index, feature in enumerate(FEATURE_NAMES):
            cell_mask = mask[:, transition_index, feature_index]
            cell_prediction = prediction[:, transition_index, feature_index][cell_mask]
            cell_target = target[:, transition_index, feature_index][cell_mask]
            cell_error = cell_prediction - cell_target
            prediction_variance = float(np.var(cell_prediction))
            target_variance = float(np.var(cell_target))
            result["feature_transitions"][transition][feature] = {
                "n": int(cell_mask.sum()),
                "standardized_mae": float(np.mean(np.abs(cell_error))),
                "standardized_rmse": float(np.sqrt(np.mean(np.square(cell_error)))),
                "spearman": _safe_spearman(cell_target, cell_prediction),
                "prediction_variance": prediction_variance,
                "target_variance": target_variance,
                "prediction_target_variance_ratio": prediction_variance
                / max(target_variance, NUMERICAL_ZERO),
                "prediction_collapsed": prediction_variance <= NUMERICAL_ZERO,
            }
    return result


def _same_contract(reference: Any, candidate: Any) -> None:
    """确保 lambda 之外的 patient/data/model/training 比较边界一致。"""

    if candidate.fold != FOLD or reference.fold != FOLD:
        raise ValueError("M1/M2 lambda pilot 只允许 fold 0")
    if reference.mode != "m1":
        raise ValueError(f"M1 reference mode 必须为 m1，实际为 {reference.mode!r}")
    if candidate.mode != "m2":
        raise ValueError(f"M2 candidate mode 必须为 m2，实际为 {candidate.mode!r}")
    if candidate.splits["train"] != reference.splits["train"]:
        raise ValueError("M2 与 M1 的 fold-train patient IDs/顺序不一致")
    if candidate.splits["val"] != reference.splits["val"]:
        raise ValueError("M2 与 M1 的 validation patient IDs/顺序不一致")
    if candidate.splits["pretrain_train"] != reference.splits["pretrain_train"]:
        raise ValueError("M2 与 M1 的 image pretraining patient IDs/顺序不一致")

    reference_payload = reference.payload
    candidate_payload = candidate.payload
    if candidate_payload.get("implementation_sha256") != reference_payload.get("implementation_sha256"):
        raise ValueError("M2 与 M1 的训练 implementation_sha256 不一致")
    reference_contract = _as_mapping(reference_payload.get("data_contract"), "M1 data_contract")
    candidate_contract = _as_mapping(candidate_payload.get("data_contract"), "M2 data_contract")
    contract_fields = (
        "fold_manifest",
        "fold_manifest_sha256",
        "cache_root",
        "radiomics_transform",
        "radiomics_transform_sha256",
        "train_patient_hash",
        "fold_train_patient_hash",
        "val_patient_hash",
        "test_patient_hash",
        "extra_pretrain_patient_hash",
    )
    for field in contract_fields:
        if candidate_contract.get(field) != reference_contract.get(field):
            raise ValueError(f"M2 与 M1 的 data contract 不一致: {field}")

    reference_model = dict(_as_mapping(reference_payload.get("model_config"), "M1 model_config"))
    candidate_model = dict(_as_mapping(candidate_payload.get("model_config"), "M2 model_config"))
    reference_model.pop("mode", None)
    candidate_model.pop("mode", None)
    if candidate_model != reference_model:
        raise ValueError("M2 与 M1 除 mode 外的 model config 不一致")
    if dict(_as_mapping(candidate_payload.get("train_config"), "M2 train_config")) != dict(
        _as_mapping(reference_payload.get("train_config"), "M1 train_config")
    ):
        raise ValueError("M2 与 M1 的 train config 不一致")
    reference_loss = dict(_as_mapping(reference_payload.get("loss_config"), "M1 loss_config"))
    candidate_loss = dict(_as_mapping(candidate_payload.get("loss_config"), "M2 loss_config"))
    reference_loss.pop("lambda_rad", None)
    candidate_loss.pop("lambda_rad", None)
    if candidate_loss != reference_loss:
        raise ValueError("M2 与 M1 除 lambda_rad 外的 loss config 不一致")
    if reference_payload.get("fold_provenance_status") != candidate_payload.get(
        "fold_provenance_status"
    ):
        raise ValueError("M2 与 M1 的 fold provenance status 不一致")
    if dict(_as_mapping(reference_payload.get("git"), "M1 git provenance")) != dict(
        _as_mapping(candidate_payload.get("git"), "M2 git provenance")
    ):
        raise ValueError("M2 与 M1 的 git provenance 不一致")

    reference_runtime = _as_mapping(reference_payload.get("runtime"), "M1 runtime")
    candidate_runtime = _as_mapping(candidate_payload.get("runtime"), "M2 runtime")
    for field in (
        "effective_train_ids",
        "effective_train_patient_hash",
        "effective_validation_ids",
        "effective_validation_patient_hash",
        "transform_fit_ids",
        "epochs_requested",
    ):
        if candidate_runtime.get(field) != reference_runtime.get(field):
            raise ValueError(f"M2 与 M1 runtime effective contract 不一致: {field}")


def _contract_summary(evaluation: Any) -> dict[str, Any]:
    contract = _as_mapping(evaluation.payload.get("data_contract"), "checkpoint.data_contract")
    git = _as_mapping(evaluation.payload.get("git"), "checkpoint.git")
    return {
        "resolved_config_sha256": evaluation.payload.get("resolved_config_sha256"),
        "implementation_sha256": evaluation.payload.get("implementation_sha256"),
        "fold_provenance_status": evaluation.payload.get("fold_provenance_status"),
        "git_branch": git.get("branch"),
        "git_commit": git.get("commit"),
        "git_experiment_status": git.get("experiment_status"),
        "fold_manifest": contract.get("fold_manifest"),
        "fold_manifest_sha256": contract.get("fold_manifest_sha256"),
        "cache_root": contract.get("cache_root"),
        "radiomics_transform": contract.get("radiomics_transform"),
        "radiomics_transform_sha256": contract.get("radiomics_transform_sha256"),
        "train_patient_hash": contract.get("train_patient_hash"),
        "val_patient_hash": contract.get("val_patient_hash"),
        "test_patient_hash": contract.get("test_patient_hash"),
        "extra_pretrain_patient_hash": contract.get("extra_pretrain_patient_hash"),
    }


def _checkpoint_epoch(evaluation: Any) -> int:
    value = evaluation.payload.get("epoch")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"checkpoint epoch 非法: {value!r}")
    return value


def _checkpoint_lambda(evaluation: Any) -> float:
    loss = _as_mapping(evaluation.payload.get("loss_config"), "checkpoint.loss_config")
    return _finite_number(loss.get("lambda_rad"), "checkpoint.loss_config.lambda_rad")


def _assert_full_run(evaluation: Any, name: str) -> None:
    runtime = _as_mapping(evaluation.payload.get("runtime"), f"{name}.runtime")
    if runtime.get("smoke") is not False:
        raise ValueError(f"{name} 必须是正式全量 run；smoke checkpoint 不能产生正式 lambda 选择")


def _relative_degradation(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return math.nan
    if reference <= NUMERICAL_ZERO:
        raise ValueError(f"M1 reference image loss 必须为正，实际为 {reference}")
    return (candidate - reference) / reference


def _candidate_row(
    *,
    evaluation: Any,
    config_path: Path,
    history_path: Path,
    history: Mapping[str, float],
    history_audit: Mapping[str, Any],
    recomputed_image: Mapping[str, Any],
    recomputed_history_audit: Mapping[str, Any],
    radiomics: Mapping[str, Any],
    resolved_run_audit: Mapping[str, Any],
    runtime_audit: Mapping[str, Any],
    transform_refit_audit: Mapping[str, Any],
    m1_recomputed_image: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    required_finite = tuple(HISTORY_FIELDS)
    nonfinite_history = [name for name in required_finite if not math.isfinite(float(history[name]))]
    if nonfinite_history:
        reasons.append("nonfinite_history_metric:" + ",".join(nonfinite_history))

    normalized_degradation = _relative_degradation(
        float(recomputed_image["normalized_next_mse"]),
        float(m1_recomputed_image["normalized_next_mse"]),
    )
    state_degradation = _relative_degradation(
        float(recomputed_image["state_loss"]), float(m1_recomputed_image["state_loss"])
    )
    if math.isfinite(normalized_degradation) and normalized_degradation > IMAGE_DEGRADATION_LIMIT:
        reasons.append("image_normalized_next_mse_degradation_gt_5pct")
    if math.isfinite(state_degradation) and state_degradation > IMAGE_DEGRADATION_LIMIT:
        reasons.append("image_state_loss_degradation_gt_5pct")

    train_config = _as_mapping(evaluation.payload.get("train_config"), "checkpoint.train_config")
    minimum_latent_std = _finite_number(
        train_config.get("min_latent_std"), "checkpoint.train_config.min_latent_std"
    )
    target_visit_feature_std = float(recomputed_image["target_visit_feature_std"])
    if (
        not math.isfinite(target_visit_feature_std)
        or target_visit_feature_std < minimum_latent_std
    ):
        reasons.append("target_state_representation_collapse")

    image_gradient = float(history["train_first_batch_image_task_gradient_norm"])
    radiomics_shared_gradient = float(
        history["train_first_batch_radiomics_shared_gradient_norm_weighted"]
    )
    radiomics_head_gradient = float(
        history["train_first_batch_radiomics_head_gradient_norm_weighted"]
    )
    gradient_ratio = float(
        history["train_first_batch_weighted_radiomics_to_image_gradient_ratio"]
    )
    if not math.isfinite(image_gradient) or image_gradient <= NUMERICAL_ZERO:
        reasons.append("invalid_or_zero_image_gradient")
    if not math.isfinite(radiomics_shared_gradient) or radiomics_shared_gradient <= NUMERICAL_ZERO:
        reasons.append("invalid_or_zero_radiomics_shared_gradient")
    if not math.isfinite(radiomics_head_gradient) or radiomics_head_gradient <= NUMERICAL_ZERO:
        reasons.append("invalid_or_zero_radiomics_head_gradient")
    if not math.isfinite(gradient_ratio) or gradient_ratio <= NUMERICAL_ZERO:
        reasons.append("invalid_or_zero_radiomics_to_image_gradient_ratio")

    predicted_delta_norm = float(recomputed_image["predicted_delta_norm"])
    target_delta_norm = float(recomputed_image["target_delta_norm"])
    delta_norm_ratio = (
        predicted_delta_norm / max(target_delta_norm, NUMERICAL_ZERO)
        if math.isfinite(predicted_delta_norm) and math.isfinite(target_delta_norm)
        else math.nan
    )
    predicted_delta_variance = float(recomputed_image["predicted_delta_pooled_variance"])
    collapsed_delta_coordinates = int(
        recomputed_image["predicted_delta_collapsed_coordinate_count"]
    )
    if (
        not math.isfinite(predicted_delta_norm)
        or predicted_delta_norm <= DELTA_NORM_NUMERICAL_ZERO
        or collapsed_delta_coordinates > 0
    ):
        reasons.append("predicted_delta_collapse")

    if int(radiomics["nonfinite_prediction_count"]) > 0:
        reasons.append("nonfinite_radiomics_prediction")
    for transition, feature_metrics in radiomics["feature_transitions"].items():
        for feature, metrics in feature_metrics.items():
            if metrics["prediction_collapsed"] is True:
                reasons.append(f"radiomics_head_cell_collapse:{transition}:{feature}")
    if radiomics["standardized_mae"] is None or not math.isfinite(
        float(radiomics["standardized_mae"])
    ):
        reasons.append("nonfinite_standardized_radiomics_mae")

    row: dict[str, Any] = {
        "fold": FOLD,
        "lambda_rad": _checkpoint_lambda(evaluation),
        "run_name": evaluation.run_name,
        "best_epoch": _checkpoint_epoch(evaluation),
        "eligible": not reasons,
        "selected": False,
        "within_1pct_of_best_mae": False,
        "selection_rank_by_mae": None,
        "exclusion_reasons": "|".join(reasons),
        "checkpoint_path": str(evaluation.checkpoint_path),
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "history_path": str(history_path),
        "history_sha256": file_sha256(history_path),
        "resolved_run_path": resolved_run_audit["path"],
        "resolved_run_sha256": resolved_run_audit["sha256"],
        "history_contract_validated": True,
        "recomputed_image_history_contract_validated": True,
        "runtime_contract_validated": True,
        "transform_train_only_refit_equal": True,
        "paired_validation_patients": radiomics["paired_patients"],
        "radiomics_validation_target_patient_hash": radiomics["target_patient_hash"],
        "radiomics_valid_elements": radiomics["valid_elements"],
        "radiomics_nonfinite_prediction_count": radiomics["nonfinite_prediction_count"],
        "radiomics_standardized_mae": radiomics["standardized_mae"],
        "radiomics_standardized_rmse": radiomics["standardized_rmse"],
        "radiomics_prediction_variance": radiomics["prediction_variance"],
        "radiomics_target_variance": radiomics["target_variance"],
        "radiomics_prediction_target_variance_ratio": radiomics[
            "prediction_target_variance_ratio"
        ],
        "image_normalized_next_mse_relative_degradation_vs_m1": normalized_degradation,
        "image_state_loss_relative_degradation_vs_m1": state_degradation,
        "delta_norm_ratio": delta_norm_ratio,
        "predicted_delta_variance": predicted_delta_variance,
        "predicted_delta_collapsed_transition_latent_coordinates": collapsed_delta_coordinates,
        "predicted_delta_transition_latent_variance_sha256": recomputed_image[
            "predicted_delta_transition_latent_variance_sha256"
        ],
        "minimum_latent_std": minimum_latent_std,
        "test_dce_arrays_accessed": False,
        "pcr_used_for_selection": False,
    }
    for name, value in _contract_summary(evaluation).items():
        row[f"contract_{name}"] = value
    row.update({f"history_audit_{name}": value for name, value in history_audit.items()})
    for name, value in recomputed_image.items():
        if not isinstance(value, (dict, list)):
            row[f"recomputed_val_{name}"] = value
    for transition, metrics in recomputed_image[
        "predicted_delta_variance_by_transition"
    ].items():
        safe_transition = transition.replace("→", "_to_")
        for name, value in metrics.items():
            if name != "collapsed_coordinate_indices":
                row[f"recomputed_val_delta_{safe_transition}_{name}"] = value
            else:
                row[
                    f"recomputed_val_delta_{safe_transition}_collapsed_coordinate_indices_json"
                ] = json.dumps(value, separators=(",", ":"))
    row["recomputed_history_rtol"] = recomputed_history_audit["rtol"]
    row["recomputed_history_atol"] = recomputed_history_audit["atol"]
    row["recomputed_history_all_fields_match"] = recomputed_history_audit[
        "all_recomputable_fields_match"
    ]
    for name, comparison in recomputed_history_audit["fields"].items():
        for field in (
            "recomputed",
            "history",
            "absolute_difference",
            "allowed_absolute_difference",
            "within_tolerance",
        ):
            row[f"recomputed_history_{name}_{field}"] = comparison[field]
    row.update({f"runtime_audit_{name}": value for name, value in runtime_audit.items()})
    row.update(
        {
            f"transform_refit_{name}": value
            for name, value in transform_refit_audit.items()
            if not isinstance(value, (list, dict))
        }
    )
    for name in HISTORY_FIELDS:
        row[f"best_epoch_{name}"] = history[name]
    for feature in FEATURE_NAMES:
        metrics = radiomics["features"][feature]
        for metric_name in (
            "n",
            "standardized_mae",
            "standardized_rmse",
            "spearman",
            "prediction_variance",
            "target_variance",
            "prediction_target_variance_ratio",
            "prediction_collapsed",
        ):
            row[f"radiomics_{feature}_{metric_name}"] = metrics[metric_name]
    for transition in TRANSITIONS:
        for feature in FEATURE_NAMES:
            metrics = radiomics["feature_transitions"][transition][feature]
            safe_transition = transition.replace("→", "_to_")
            prefix = f"radiomics_{safe_transition}_{feature}_"
            for metric_name in (
                "n",
                "standardized_mae",
                "standardized_rmse",
                "spearman",
                "prediction_variance",
                "target_variance",
                "prediction_target_variance_ratio",
                "prediction_collapsed",
            ):
                row[prefix + metric_name] = metrics[metric_name]
    return row


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (float(row["radiomics_standardized_mae"]), row["lambda_rad"]))
    for rank, row in enumerate(eligible, start=1):
        row["selection_rank_by_mae"] = rank
    if not eligible:
        return {
            "status": "no_eligible_candidate",
            "selected_lambda": None,
            "selected_run_name": None,
            "minimum_eligible_mae": None,
        }
    minimum_mae = float(eligible[0]["radiomics_standardized_mae"])
    tolerance_limit = minimum_mae * (1.0 + MAE_TIE_RELATIVE_TOLERANCE)
    within_tolerance = [
        row
        for row in eligible
        if (
            float(row["radiomics_standardized_mae"]) - minimum_mae
        )
        / max(abs(minimum_mae), NUMERICAL_ZERO)
        < MAE_TIE_RELATIVE_TOLERANCE
    ]
    for row in within_tolerance:
        row["within_1pct_of_best_mae"] = True
    selected = min(within_tolerance, key=lambda row: float(row["lambda_rad"]))
    selected["selected"] = True
    return {
        "status": "selected",
        "selected_lambda": selected["lambda_rad"],
        "selected_run_name": selected["run_name"],
        "minimum_eligible_mae": minimum_mae,
        "one_percent_mae_limit": tolerance_limit,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _reason_chinese(value: str) -> str:
    if not value:
        return "通过"
    replacements = {
        "image_normalized_next_mse_degradation_gt_5pct": "normalized next MSE 相对 M1 恶化超过 5%",
        "image_state_loss_degradation_gt_5pct": "state loss 相对 M1 恶化超过 5%",
        "target_state_representation_collapse": "重算 EMA target-state feature std 低于 eligibility 门槛",
        "predicted_delta_collapse": "预测 delta 在数值精度下为零/常数",
        "invalid_or_zero_image_gradient": "image-task gradient 非有限或为零",
        "invalid_or_zero_radiomics_shared_gradient": "radiomics shared gradient 非有限或为零",
        "invalid_or_zero_radiomics_head_gradient": "radiomics head gradient 非有限或为零",
        "invalid_or_zero_radiomics_to_image_gradient_ratio": "radiomics/image gradient ratio 非有限或为零",
        "nonfinite_radiomics_prediction": "radiomics prediction 含 NaN/Inf",
        "nonfinite_standardized_radiomics_mae": "standardized radiomics MAE 非有限",
    }
    output: list[str] = []
    for reason in value.split("|"):
        if reason.startswith("nonfinite_history_metric:"):
            output.append("best-epoch history 含非有限指标：" + reason.split(":", 1)[1])
        elif reason.startswith("radiomics_head_cell_collapse:"):
            output.append("radiomics head transition×feature 常数坍塌：" + reason.split(":", 1)[1])
        else:
            output.append(replacements.get(reason, reason))
    return "；".join(output)


def build_report(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    m1 = payload["m1_reference"]
    if selection["status"] == "selected":
        conclusion = (
            f"按预注册规则锁定 **`lambda_rad={selection['selected_lambda']}`**"
            f"（run `{selection['selected_run_name']}`）。"
        )
    else:
        conclusion = "四个候选均未通过预注册安全筛选，因此本次没有锁定 lambda。"
    gradient_ratios = [
        float(row["best_epoch_train_first_batch_weighted_radiomics_to_image_gradient_ratio"])
        for row in payload["candidates"]
        if row["best_epoch_train_first_batch_weighted_radiomics_to_image_gradient_ratio"]
        is not None
    ]
    head_variance_ratios = [
        float(row[f"radiomics_{transition.replace('→', '_to_')}_{feature}_prediction_target_variance_ratio"])
        for row in payload["candidates"]
        for transition in TRANSITIONS
        for feature in FEATURE_NAMES
    ]

    lines = [
        "# M2 fold 0 Lambda 选择报告",
        "",
        f"生成时间（UTC）：`{payload['generated_at_utc']}`",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        "该结论只使用 fold 0 validation。脚本没有提取 test split、没有加载 test DCE array，",
        "也没有把 pCR 标签用于候选排序。必须准确说明：`load_evaluation` 为执行锁定契约会",
        "读取锁定 cohort 的全量 pCR values、radiomics 表内完整 raw measurement values，并核验",
        "train/val/test patient ID/hash 与全 raw hash；这些值只用于契约/完整性，test values",
        "绝不进入候选排序。只有 val DCE",
        "array 经 DataLoader 进入模型，也没有计算任何 test 指标。",
        "",
        "## 预注册选择规则",
        "",
        "1. 候选固定为 `lambda_rad={0.05, 0.1, 0.25, 0.5}`，且必须与同一 fold 0 M1",
        "   使用相同 train/validation IDs、transform、模型规格、训练配置与实现哈希。",
        "2. 从 M1/M2 frozen validation outputs 独立重算 `normalized_next_mse` 与加权",
        "   `state_loss`，先以严格容差和 best history 交叉核验，再用于 5% gate；任一相对",
        "   M1 恶化超过 5% 即排除。hard gate 不直接信任 history 数值。",
        "3. history/prediction 非有限、重算 EMA target-state 跨患者 feature std 低于门槛、",
        "   重算 predicted-delta norm 为零，或沿 patient 维检查任一 transition×latent",
        "   coordinate 方差小于等于 `1e-12`，或任一 radiomics feature 的",
        "   任一 transition×feature validation 预测方差小于等于 `1e-12`，或 image/radiomics",
        "   shared/head 首批 gradient 非有限或为零，均视为数值/坍塌失败；不设置无依据上限。",
        "4. 在剩余候选中选择 paired validation standardized radiomics MAE 最低者；若 MAE",
        "   与最小值的相对差严格小于 1%，取更小的 lambda。Spearman 和 prediction variance 只作",
        "   grounding/坍塌诊断，不用 pCR 或 test 打破平局。",
        "",
        "## M1 参照",
        "",
        f"- run：`{m1['run_name']}`；best epoch：{m1['best_epoch']}；checkpoint SHA-256：",
        f"  `{m1['checkpoint_sha256']}`",
        f"- 重算 raw aggregate gain：{_fmt(m1['recomputed_validation_image_metrics']['aggregate_transition_gain'])}；",
        f"  重算 normalized aggregate gain：{_fmt(m1['recomputed_validation_image_metrics']['normalized_error_aggregate_gain'])}",
        f"- 重算 normalized next MSE：{_fmt(m1['recomputed_validation_image_metrics']['normalized_next_mse'])}；",
        f"  重算 state loss：{_fmt(m1['recomputed_validation_image_metrics']['state_loss'])}；",
        f"  重算 delta loss：{_fmt(m1['recomputed_validation_image_metrics']['delta_loss'])}",
        "",
        "## 候选汇总",
        "",
        "| lambda | best epoch | 通过筛选 | standardized MAE | RMSE | normalized MSE 相对 M1 | state 相对 M1 | raw gain | normalized gain | weighted grad ratio | 原因/结论 |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        reason = "选中" if row["selected"] else _reason_chinese(row["exclusion_reasons"])
        if row["eligible"] and not row["selected"]:
            reason = "通过，但未按 MAE/1% 小 lambda 规则选中"
        lines.append(
            "| {lambda_} | {epoch} | {eligible} | {mae} | {rmse} | {norm_deg} | "
            "{state_deg} | {raw_gain} | {norm_gain} | {grad} | {reason} |".format(
                lambda_=row["lambda_rad"],
                epoch=row["best_epoch"],
                eligible="是" if row["eligible"] else "否",
                mae=_fmt(row["radiomics_standardized_mae"]),
                rmse=_fmt(row["radiomics_standardized_rmse"]),
                norm_deg=_fmt(row["image_normalized_next_mse_relative_degradation_vs_m1"]),
                state_deg=_fmt(row["image_state_loss_relative_degradation_vs_m1"]),
                raw_gain=_fmt(row["recomputed_val_aggregate_transition_gain"]),
                norm_gain=_fmt(row["recomputed_val_normalized_error_aggregate_gain"]),
                grad=_fmt(
                    row[
                        "best_epoch_train_first_batch_weighted_radiomics_to_image_gradient_ratio"
                    ]
                ),
                reason=reason,
            )
        )

    lines.extend(
        [
            "",
            "表中两个“相对 M1”字段是 loss 的相对变化，正值表示恶化；5% 对应 `0.05`。",
            "weighted grad ratio 是 best epoch 首个 training batch 的加权 radiomics shared-gradient",
            "norm 与 image-task gradient norm 之比，不是 validation/test 调参信号。",
            f"当前四候选该 ratio 范围为 `{min(gradient_ratios):.4f}–{max(gradient_ratios):.4f}`。",
            "没有预注册可信的 gradient-ratio 上限，因此 selector 不用上限排除候选；最高值",
            "提示 radiomics gradient 可能主导 shared update，这是已披露限制，需谨慎解释。",
            f"transition×feature head prediction/target variance ratio 的观察范围为",
            f"`{min(head_variance_ratios):.6f}–{max(head_variance_ratios):.6f}`；它仅作 grounding",
            "诊断，不新增未预注册的 ratio gate。",
            "",
            "## Frozen Validation 重算闭环",
            "",
            f"- 可重算的 11 个 image 指标均以 `rtol={HISTORY_RECOMPUTE_RTOL}`、",
            f"  `atol={HISTORY_RECOMPUTE_ATOL}` 与 best history 逐项通过核验。",
            "- `val_visit_state_std` 与 `val_visit_feature_std` 来自训练时 online_state；",
            "  `ExtractedSplit` 未返回 online_state，不能假装直接重算。selector 透明保留该",
            "  限制，并用可观测 EMA target visits 重算 target-state std/feature std 作 hard gate。",
            "- predicted-delta 坍塌不是全维池化判断：每个候选均沿 patient 维逐一检查",
            "  3×latent_dim 个 transition×latent coordinate，并记录各 transition 的最小方差、",
            "  collapsed count 与完整方差矩阵 SHA-256。",
            "",
            "## 各 Transition × Feature Validation Grounding",
            "",
            "所有指标均在对应 fold-train-only transform 的 standardized 空间计算；每名配对患者",
            "贡献 T0→T1、T1→T2、T2→T3 三个有效 transition。",
            "",
            "| lambda | transition | feature | N | MAE | RMSE | Spearman | prediction variance | target variance | 方差比 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["candidates"]:
        for transition in TRANSITIONS:
            for feature in FEATURE_NAMES:
                prefix = f"radiomics_{transition.replace('→', '_to_')}_{feature}_"
                lines.append(
                    "| {lambda_} | {transition} | {feature} | {n} | {mae} | {rmse} | {rho} | {pvar} | {tvar} | {ratio} |".format(
                        lambda_=row["lambda_rad"],
                        transition=transition,
                        feature=feature,
                        n=row[prefix + "n"],
                        mae=_fmt(row[prefix + "standardized_mae"]),
                        rmse=_fmt(row[prefix + "standardized_rmse"]),
                        rho=_fmt(row[prefix + "spearman"]),
                        pvar=_fmt(row[prefix + "prediction_variance"], 6),
                        tvar=_fmt(row[prefix + "target_variance"], 6),
                        ratio=_fmt(row[prefix + "prediction_target_variance_ratio"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Test-Blind 审计与可复现性",
            "",
            f"- `extract_native_split` 调用记录：`{payload['access_audit']['extract_native_split_calls']}`；",
            "  每次 split 均严格为 `val`。",
            "- test DCE arrays accessed：`false`；test metrics computed：`false`；",
            "  pCR used for selection：`false`。",
            "- `load_evaluation` 为锁定契约读取 cohort 的全量 pCR values、radiomics 表内",
            "  完整 raw measurement values，并重算全 raw hash；`extract_native_split` 返回",
            "  validation pCR label array。上述值只用于契约/完整性，selector 不读取 label",
            "  数值用于排名，任何 test pCR/raw measurement value 也不参与候选排序。",
            "- 每个 transform 在 selector 内用锁定 fold-train IDs 独立重拟合，并逐字段比较",
            "  version、raw/train hash、quantile 与全部 feature 参数；validation target 仅对",
            "  extracted val patient IDs 逐人调用 `transform_one`，从未 `transform_all(375人)`。",
            f"- 选择脚本 SHA-256：`{payload['selector_sha256']}`。checkpoint、config 和 history",
            "  的绝对路径与 SHA-256 逐候选保存在 JSON/CSV 中。",
            "- 三个正式输出均采用 create-new 写入；目标已存在时脚本在评估前拒绝覆盖。",
            "",
        ]
    )
    return "\n".join(lines)


def _assert_outputs_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有 M2 lambda 选择输出: {existing}")


def _write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_json_safe(list(rows)))


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m2-checkpoints",
        "--checkpoints",
        nargs=4,
        type=Path,
        required=True,
        metavar=("L005", "L010", "L025", "L050"),
        help="四个 fold 0 M2 best.pt；实际 lambda 由 checkpoint 验证，不依赖输入顺序",
    )
    parser.add_argument(
        "--m2-configs",
        "--m2-config",
        nargs="+",
        type=Path,
        required=True,
        help="一个共享 M2 YAML，或与四个 checkpoint 同顺序的四个 YAML",
    )
    parser.add_argument("--m1-checkpoint", type=Path, required=True, help="fold 0 M1 best.pt")
    parser.add_argument("--m1-config", type=Path, required=True, help="与 M1 checkpoint 匹配的 YAML")
    parser.add_argument(
        "--csv-output", type=Path, default=DEFAULT_CSV, help="create-new CSV 输出路径"
    )
    parser.add_argument(
        "--json-output", type=Path, default=DEFAULT_JSON, help="create-new JSON 输出路径"
    )
    parser.add_argument(
        "--report-output", type=Path, default=DEFAULT_REPORT, help="create-new 中文报告路径"
    )
    parser.add_argument("--device", default="cuda", help="仅用于 validation extraction，如 cuda:0 或 cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = tuple(path.expanduser().resolve() for path in (
        args.csv_output, args.json_output, args.report_output
    ))
    if len(set(outputs)) != 3:
        raise ValueError("CSV/JSON/report 输出路径必须互不相同")
    _assert_outputs_absent(outputs)
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size 必须为正数，workers 必须为非负数")
    if len(args.m2_configs) not in (1, 4):
        raise ValueError("--m2-configs 必须提供一个共享 config 或四个逐候选 config")
    config_arguments = args.m2_configs * 4 if len(args.m2_configs) == 1 else args.m2_configs
    checkpoint_paths = [_resolved_file(path, "M2 checkpoint") for path in args.m2_checkpoints]
    config_paths = [_resolved_file(path, "M2 config") for path in config_arguments]
    m1_checkpoint = _resolved_file(args.m1_checkpoint, "M1 checkpoint")
    m1_config_path = _resolved_file(args.m1_config, "M1 config")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 torch.cuda.is_available() 为 false")
    device = torch.device(args.device)

    m1_evaluation = load_evaluation(m1_checkpoint, load_config(m1_config_path), device)
    _assert_full_run(m1_evaluation, "M1 reference")
    if m1_evaluation.fold != FOLD or m1_evaluation.mode != "m1":
        raise ValueError("M1 reference 必须是 fold 0 / mode=m1")
    if not math.isclose(_checkpoint_lambda(m1_evaluation), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("M1 reference checkpoint.loss_config.lambda_rad 必须严格为 0")
    m1_runtime_audit = _validate_runtime_contract(m1_evaluation)
    m1_resolved_run_audit = _validate_resolved_run(
        m1_evaluation, load_config(m1_config_path), 0.0
    )
    m1_transform_refit_audit = _validate_transform_refit(m1_evaluation)
    m1_history_path = _resolved_file(_history_path(m1_evaluation.run_name), "M1 history")
    m1_history, m1_history_audit = read_best_history(
        m1_history_path,
        evaluation=m1_evaluation,
        best_epoch=_checkpoint_epoch(m1_evaluation),
        expected_mode="m1",
        expected_lambda=0.0,
    )
    if any(not math.isfinite(value) for value in m1_history.values()):
        raise FloatingPointError("M1 reference best-epoch history 含 NaN/Inf，无法执行相对筛选")

    rows: list[dict[str, Any]] = []
    extraction_calls: list[dict[str, Any]] = []
    m1_extracted = extract_native_split(
        m1_evaluation,
        "val",
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    extraction_calls.append(
        {
            "role": "M1 reference",
            "run_name": m1_evaluation.run_name,
            "split": "val",
            "patient_count": len(m1_extracted.patient_ids),
        }
    )
    m1_recomputed_image = recompute_validation_image_metrics(m1_extracted)
    m1_recomputed_history_audit = compare_recomputed_image_history(
        m1_recomputed_image, m1_history
    )
    del m1_extracted
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seen_lambdas: list[float] = []
    for index, (checkpoint_path, config_path) in enumerate(zip(checkpoint_paths, config_paths)):
        evaluation = load_evaluation(checkpoint_path, load_config(config_path), device)
        _assert_full_run(evaluation, f"M2 candidate {index + 1}")
        _same_contract(m1_evaluation, evaluation)
        lambda_rad = _checkpoint_lambda(evaluation)
        if not any(math.isclose(lambda_rad, value, rel_tol=0.0, abs_tol=1e-12) for value in M2_LAMBDA_GRID):
            raise ValueError(f"M2 lambda 不在预注册 grid: {lambda_rad}")
        seen_lambdas.append(lambda_rad)

        runtime_audit = _validate_runtime_contract(evaluation)
        resolved_run_audit = _validate_resolved_run(
            evaluation, load_config(config_path), lambda_rad
        )
        transform_refit_audit = _validate_transform_refit(evaluation)
        history_path = _resolved_file(
            _history_path(evaluation.run_name), f"M2 candidate {index + 1} history"
        )
        history, history_audit = read_best_history(
            history_path,
            evaluation=evaluation,
            best_epoch=_checkpoint_epoch(evaluation),
            expected_mode="m2",
            expected_lambda=lambda_rad,
        )

        # 唯一允许的原生 split 提取；禁止把 split 变成用户输入。
        split = "val"
        extracted = extract_native_split(
            evaluation,
            split,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        extraction_calls.append(
            {
                "role": "M2 candidate",
                "run_name": evaluation.run_name,
                "split": split,
                "patient_count": len(extracted.patient_ids),
            }
        )
        recomputed_image = recompute_validation_image_metrics(extracted)
        recomputed_history_audit = compare_recomputed_image_history(
            recomputed_image, history
        )
        radiomics = radiomics_validation_metrics(evaluation, extracted)
        rows.append(
            _candidate_row(
                evaluation=evaluation,
                config_path=config_path,
                history_path=history_path,
                history=history,
                history_audit=history_audit,
                recomputed_image=recomputed_image,
                recomputed_history_audit=recomputed_history_audit,
                radiomics=radiomics,
                resolved_run_audit=resolved_run_audit,
                runtime_audit=runtime_audit,
                transform_refit_audit=transform_refit_audit,
                m1_recomputed_image=m1_recomputed_image,
            )
        )
        del extracted, evaluation
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ordered_lambdas = sorted(seen_lambdas)
    if len(set(ordered_lambdas)) != len(M2_LAMBDA_GRID) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(ordered_lambdas, M2_LAMBDA_GRID)
    ):
        raise ValueError(
            f"必须恰好提供预注册四个 lambda 各一次；实际为 {ordered_lambdas}"
        )
    if len(extraction_calls) != 5 or any(call["split"] != "val" for call in extraction_calls):
        raise AssertionError("内部 split access audit 失败")

    rows.sort(key=lambda row: float(row["lambda_rad"]))
    selection = _select(rows)
    payload: dict[str, Any] = {
        "schema_version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fold 0 validation-only M2 lambda selection",
        "selector_path": str(Path(__file__).resolve()),
        "selector_sha256": file_sha256(Path(__file__).resolve()),
        "fold": FOLD,
        "lambda_grid": list(M2_LAMBDA_GRID),
        "rules": {
            "image_degradation_limit": IMAGE_DEGRADATION_LIMIT,
            "screened_image_metrics": ["val_normalized_next_mse", "val_state_loss"],
            "mae_tie_relative_tolerance": MAE_TIE_RELATIVE_TOLERANCE,
            "numerical_zero_variance": NUMERICAL_ZERO,
            "predicted_delta_norm_numerical_zero": DELTA_NORM_NUMERICAL_ZERO,
            "history_recompute_rtol": HISTORY_RECOMPUTE_RTOL,
            "history_recompute_atol": HISTORY_RECOMPUTE_ATOL,
            "predicted_delta_collapse_scope": "patient-axis variance for every transition×latent coordinate",
            "primary_selection_metric": "paired validation standardized radiomics MAE",
            "tie_break": "relative MAE difference strictly <1% -> smaller lambda_rad",
        },
        "m1_reference": {
            "run_name": m1_evaluation.run_name,
            "best_epoch": _checkpoint_epoch(m1_evaluation),
            "checkpoint_path": str(m1_evaluation.checkpoint_path),
            "checkpoint_sha256": m1_evaluation.checkpoint_sha256,
            "config_path": str(m1_config_path),
            "config_sha256": file_sha256(m1_config_path),
            "history_path": str(m1_history_path),
            "history_sha256": file_sha256(m1_history_path),
            "history": m1_history,
            "history_audit": m1_history_audit,
            "recomputed_validation_image_metrics": m1_recomputed_image,
            "recomputed_history_audit": m1_recomputed_history_audit,
            "resolved_run_audit": m1_resolved_run_audit,
            "runtime_audit": m1_runtime_audit,
            "transform_refit_audit": m1_transform_refit_audit,
            "contract": _contract_summary(m1_evaluation),
        },
        "selection": selection,
        "candidates": rows,
        "access_audit": {
            "extract_native_split_calls": extraction_calls,
            "allowed_extraction_splits": ["val"],
            "test_split_extracted": False,
            "test_dce_arrays_accessed": False,
            "test_metrics_computed": False,
            "test_patient_ids_and_hashes_validated_as_checkpoint_metadata": True,
            "pcr_used_for_selection": False,
            "validation_labels_returned_by_api_but_not_read_by_selector": True,
            "full_cohort_pcr_values_loaded_for_contract": True,
            "full_radiomics_table_raw_measurement_values_loaded_for_contract_hash_and_train_only_refit": True,
            "full_375_radiomics_transform_all_called": False,
            "note": "load_evaluation 读取锁定 cohort 的全量 pCR values 与 radiomics 表内完整 raw measurement values，供契约、split 与全 raw hash 完整性核验；任何 test value 均不参与排名。只有 val DCE array 进入 DataLoader，validation targets 只逐 extracted val ID transform_one。",
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "torch_version": str(torch.__version__),
        },
    }
    report = build_report(payload)
    _write_csv_new(outputs[0], rows)
    _write_json_new(outputs[1], payload)
    _write_text_new(outputs[2], report)
    print(json.dumps(_json_safe(selection), ensure_ascii=False, indent=2, allow_nan=False))
    if selection["status"] != "selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
