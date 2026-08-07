"""C0/C1/C2 公平控制与 predicted-delta 后验 radiomics grounding。

本模块执行两类严格分开的分析：

1. C0 只使用截至决策点已观察到的 radiomics 值与相邻变化；C1 在同一
   paired subset 上拼接冻结 M2 的 image-derived 表征；C2 原样读取正式
   ``evaluate_fold`` 产生的 M2 image-only 概率，再限制到同一 paired test
   subset。C0/C1 的 scaler、logistic 只拟合 fold train，超参数与阈值只看
   paired validation。
2. 对 M0/M1/M2 的 predicted delta 分别拟合 train-only MultiOutput Ridge，
   alpha 只由 paired validation 选择。它是 prediction-level 的后验 probe，
   不是 M2 训练时的原生 radiomics head。

所有模型、超参数和阈值选择函数只接收 train/validation；已提取的 test 数组
绝不传入选择过程，只用于锁定后的预测与指标。radiomics 控制特征绝不包含
决策点之后的值或变化。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import EXPERIMENT_ROOT
from .data import FEATURE_NAMES, TRANSITIONS
from .evaluation import (
    DECISION_POINTS,
    FEATURE_SCHEMA_SHA256,
    READOUT_C_GRID,
    READOUT_PENALTIES,
    ExtractedSplit,
    evaluation_implementation_sha256,
    extract_native_split,
    load_evaluation,
    select_youden_threshold,
)
from .transforms import RadiomicsChangeTransform, raw_targets_hash


RIDGE_ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
POINT_TO_INDEX = {point: index for index, point in enumerate(DECISION_POINTS)}
POINT_TO_TRANSITION = dict(zip(DECISION_POINTS, TRANSITIONS))
POINT_TO_LAST_VISIT = {"T0": 0, "T0-T1": 1, "T0-T2": 2}
_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def controls_implementation_sha256() -> str:
    """锁定 controls 模块与 CLI 源码，供输出命名和审计。"""

    digest = hashlib.sha256()
    paths = (Path(__file__).resolve(), EXPERIMENT_ROOT / "scripts" / "run_controls.py")
    for path in paths:
        digest.update(str(path.relative_to(EXPERIMENT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelExtraction:
    """一个冻结 checkpoint 的 split 表征与不可变来源信息。"""

    mode: str
    run_name: str
    fold: int
    checkpoint: str
    checkpoint_sha256: str
    epoch: int
    fold_manifest_sha256: str
    resolved_config_sha256: str
    implementation_sha256: str
    feature_schema_sha256: str
    splits: Mapping[str, ExtractedSplit]


@dataclass(frozen=True)
class PairedView:
    """某 split/decision point 中 radiomics prefix 完整的患者视图。"""

    indices: np.ndarray
    patient_ids: tuple[str, ...]
    labels: np.ndarray
    radiomics_features: np.ndarray
    radiomics_feature_names: tuple[str, ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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
        json.dump(
            _json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False
        )
        stream.write("\n")


def _raw_visit_values(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由相邻 transition 表恢复 T0–T3 值，并验证共享端点一致。"""

    raw = np.asarray(raw, dtype=np.float64)
    expected = (len(TRANSITIONS), len(FEATURE_NAMES), 3)
    if raw.shape != expected:
        raise ValueError(f"raw radiomics shape 应为 {expected}，实际 {raw.shape}")
    values = np.full((4, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    valid = np.zeros_like(values, dtype=bool)
    values[0] = raw[0, :, 0]
    valid[0] = raw[0, :, 2].astype(bool) & np.isfinite(values[0])
    for transition_index in range(3):
        end = raw[transition_index, :, 1]
        end_valid = raw[transition_index, :, 2].astype(bool) & np.isfinite(end)
        values[transition_index + 1] = end
        valid[transition_index + 1] = end_valid
        if transition_index < 2:
            next_start = raw[transition_index + 1, :, 0]
            next_valid = raw[transition_index + 1, :, 2].astype(bool) & np.isfinite(
                next_start
            )
            comparable = end_valid & next_valid
            if comparable.any() and not np.allclose(
                end[comparable], next_start[comparable], rtol=1e-6, atol=1e-8
            ):
                raise ValueError("相邻 radiomics transition 的共享 visit 值不一致")
            valid[transition_index + 1] &= next_valid
    return values, valid


def observed_radiomics_features(
    raw: np.ndarray, decision_point: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """构造截至决策点的观测特征；返回空数组表示该 prefix 不完整。

    特征顺序固定为先按 visit 排列四个观测值，再按相邻 transition 排列四个
    原始差值。T0、T0-T1、T0-T2 分别含 4、12、20 维；T3 及通往 T3 的变化
    永远不会进入任何已注册决策点。
    """

    if decision_point not in POINT_TO_LAST_VISIT:
        raise ValueError(f"未知 decision point: {decision_point}")
    last_visit = POINT_TO_LAST_VISIT[decision_point]
    values, valid = _raw_visit_values(raw)
    if not valid[: last_visit + 1].all():
        return np.empty(0, dtype=np.float64), ()

    parts: list[np.ndarray] = []
    names: list[str] = []
    for visit_index in range(last_visit + 1):
        parts.append(values[visit_index])
        names.extend(f"{feature}_T{visit_index}" for feature in FEATURE_NAMES)
    for transition_index in range(last_visit):
        parts.append(values[transition_index + 1] - values[transition_index])
        names.extend(
            f"delta_{feature}_T{transition_index}_T{transition_index + 1}"
            for feature in FEATURE_NAMES
        )
    output = np.concatenate(parts).astype(np.float64, copy=False)
    if not np.isfinite(output).all():
        return np.empty(0, dtype=np.float64), ()
    expected_dim = (last_visit + 1 + last_visit) * len(FEATURE_NAMES)
    if output.shape != (expected_dim,):
        raise AssertionError("observed radiomics feature 维度不符合固定 schema")
    return output, tuple(names)


def paired_view(
    split: ExtractedSplit,
    raw_targets: Mapping[str, np.ndarray],
    decision_point: str,
) -> PairedView:
    """按原 split 顺序筛选 radiomics prefix 完整的患者。"""

    indices: list[int] = []
    patient_ids: list[str] = []
    rows: list[np.ndarray] = []
    schema: tuple[str, ...] | None = None
    for index, patient_id in enumerate(split.patient_ids):
        raw = raw_targets.get(patient_id)
        if raw is None:
            continue
        features, names = observed_radiomics_features(raw, decision_point)
        if features.size == 0:
            continue
        if schema is None:
            schema = names
        elif schema != names:
            raise AssertionError("radiomics feature schema 在患者间漂移")
        indices.append(index)
        patient_ids.append(patient_id)
        rows.append(features)
    if not rows or schema is None:
        raise ValueError(f"{decision_point} 没有 prefix 完整的 paired 患者")
    matrix = np.stack(rows)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("paired radiomics features 含 NaN/Inf")
    index_array = np.asarray(indices, dtype=np.int64)
    return PairedView(
        indices=index_array,
        patient_ids=tuple(patient_ids),
        labels=np.asarray(split.labels[index_array], dtype=np.int64),
        radiomics_features=matrix,
        radiomics_feature_names=schema,
    )


def _make_logistic(penalty: str, c_value: float, random_state: int) -> Any:
    if penalty not in READOUT_PENALTIES or c_value not in READOUT_C_GRID:
        raise ValueError("logistic penalty/C 不在预注册 grid")
    arguments: dict[str, Any] = {
        "C": float(c_value),
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": int(random_state),
    }
    if (
        inspect.signature(LogisticRegression).parameters["penalty"].default
        == "deprecated"
    ):
        arguments["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        arguments["penalty"] = penalty
    return make_pipeline(StandardScaler(), LogisticRegression(**arguments))


def select_control_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    random_state: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """train-only 拟合 C0/C1，并用 paired validation 选择模型与阈值。"""

    train_x = np.asarray(train_x, dtype=np.float64)
    validation_x = np.asarray(validation_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.int64)
    validation_y = np.asarray(validation_y, dtype=np.int64)
    if (
        train_x.ndim != 2
        or validation_x.ndim != 2
        or train_x.shape[1] != validation_x.shape[1]
    ):
        raise ValueError("train/validation control feature 必须是同维二维矩阵")
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise FloatingPointError("control feature 含 NaN/Inf")
    if np.unique(train_y).size != 2 or np.unique(validation_y).size != 2:
        raise ValueError("paired train/validation 必须均包含两个 pCR 类别")

    candidates: list[dict[str, Any]] = []
    selected_model: Any | None = None
    selected: dict[str, Any] | None = None
    for penalty in READOUT_PENALTIES:
        for c_value in READOUT_C_GRID:
            model = _make_logistic(penalty, c_value, random_state)
            model.fit(train_x, train_y)
            probability = model.predict_proba(validation_x)[:, 1]
            validation_auroc = float(roc_auc_score(validation_y, probability))
            record = {
                "penalty": penalty,
                "C": float(c_value),
                "validation_auroc": validation_auroc,
            }
            candidates.append(record)
            if (
                selected is None
                or validation_auroc > float(selected["validation_auroc"]) + 1e-12
            ):
                selected = record
                selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("control logistic grid 未产生候选模型")
    validation_probability = selected_model.predict_proba(validation_x)[:, 1]
    threshold = select_youden_threshold(validation_y, validation_probability)
    scaler = selected_model.steps[0][1]
    classifier = selected_model.steps[-1][1]
    selection = {
        "selected_penalty": selected["penalty"],
        "selected_C": selected["C"],
        "selected_validation_auroc": selected["validation_auroc"],
        "fit_scope": "StandardScaler 与 logistic 均仅拟合 paired fold train",
        "selection_scope": "C/penalty 与 Youden threshold 均仅使用 paired fold validation",
        "test_used_for_selection": False,
        "tie_break": "validation AUROC 在 1e-12 内并列时按 l2、l1，再按较小 C",
        "n_train": int(len(train_y)),
        "n_validation": int(len(validation_y)),
        "feature_dim": int(train_x.shape[1]),
        "candidates": candidates,
        "parameters": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "logistic_coef": classifier.coef_.tolist(),
            "logistic_intercept": classifier.intercept_.tolist(),
            "classes": classifier.classes_.tolist(),
        },
    }
    return selected_model, selection, threshold


def select_multioutput_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """拟合 train-only MultiOutput Ridge，以 validation 标准化 MSE 选 alpha。"""

    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    validation_x = np.asarray(validation_x, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64)
    if (
        train_x.ndim != 2
        or validation_x.ndim != 2
        or train_x.shape[1] != validation_x.shape[1]
    ):
        raise ValueError("Ridge train/validation input 必须为同维二维矩阵")
    if (
        train_y.ndim != 2
        or validation_y.ndim != 2
        or train_y.shape[1] != validation_y.shape[1]
    ):
        raise ValueError("Ridge train/validation target 必须为同维二维矩阵")
    if (
        train_x.shape[0] != train_y.shape[0]
        or validation_x.shape[0] != validation_y.shape[0]
    ):
        raise ValueError("Ridge input/target 行数不一致")
    arrays = (train_x, train_y, validation_x, validation_y)
    if any(not np.isfinite(values).all() for values in arrays):
        raise FloatingPointError("Ridge input/target 含 NaN/Inf")

    candidates: list[dict[str, Any]] = []
    selected_model: Any | None = None
    selected: dict[str, Any] | None = None
    for alpha in RIDGE_ALPHA_GRID:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(train_x, train_y)
        predicted = model.predict(validation_x)
        mse = float(mean_squared_error(validation_y, predicted))
        record = {"alpha": float(alpha), "validation_standardized_mse": mse}
        candidates.append(record)
        if (
            selected is None
            or mse < float(selected["validation_standardized_mse"]) - 1e-12
        ):
            selected = record
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("Ridge alpha grid 未产生候选模型")
    scaler = selected_model.steps[0][1]
    ridge = selected_model.steps[-1][1]
    selection = {
        "selected_alpha": selected["alpha"],
        "selected_validation_standardized_mse": selected["validation_standardized_mse"],
        "fit_scope": "StandardScaler 与 MultiOutput Ridge 均仅拟合 paired fold train",
        "selection_scope": "alpha 仅使用 paired fold validation 的四特征平均标准化 MSE",
        "test_used_for_selection": False,
        "tie_break": "validation MSE 在 1e-12 内并列时选择较小 alpha",
        "n_train": int(train_x.shape[0]),
        "n_validation": int(validation_x.shape[0]),
        "input_dim": int(train_x.shape[1]),
        "target_dim": int(train_y.shape[1]),
        "candidates": candidates,
        "parameters": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "ridge_coef": ridge.coef_.tolist(),
            "ridge_intercept": ridge.intercept_.tolist(),
        },
    }
    return selected_model, selection


def _classification_metrics(
    labels: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = probability >= threshold
    positive = labels == 1
    negative = labels == 0
    return {
        "n": int(len(labels)),
        "positive": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "auroc": (
            float(roc_auc_score(labels, probability))
            if np.unique(labels).size == 2
            else None
        ),
        "auprc": (
            float(average_precision_score(labels, probability))
            if positive.any()
            else None
        ),
        "accuracy": float((prediction == labels).mean()),
        "sensitivity": float(prediction[positive].mean()) if positive.any() else None,
        "specificity": (
            float((~prediction[negative]).mean()) if negative.any() else None
        ),
        "threshold": float(threshold),
    }


def _ridge_targets(
    split: ExtractedSplit,
    raw_targets: Mapping[str, np.ndarray],
    transform: RadiomicsChangeTransform,
    decision_point: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """取得该决策点所预测相邻变化的完整 paired target。"""

    transition_index = POINT_TO_INDEX[decision_point]
    indices: list[int] = []
    patient_ids: list[str] = []
    targets: list[np.ndarray] = []
    for index, patient_id in enumerate(split.patient_ids):
        raw = raw_targets.get(patient_id)
        if raw is None:
            continue
        target, mask = transform.transform_one(raw)
        if not mask[transition_index].all():
            continue
        row = np.asarray(target[transition_index], dtype=np.float64)
        if not np.isfinite(row).all():
            raise FloatingPointError("transformed radiomics target 含 NaN/Inf")
        indices.append(index)
        patient_ids.append(patient_id)
        targets.append(row)
    if not targets:
        raise ValueError(f"{decision_point} 没有完整 radiomics change target")
    return np.asarray(indices, dtype=np.int64), tuple(patient_ids), np.stack(targets)


def _load_native_c2(
    path: Path,
    m2: ModelExtraction,
    paired_test: Mapping[str, PairedView],
) -> pd.DataFrame:
    """验证并提取正式 M2 image-only test 概率，绝不重拟合 C2。"""

    path = path.expanduser().resolve(strict=True)
    expected_path = _default_native_prediction_path(m2).resolve()
    if path.parent != expected_path.parent or path.name not in {
        "test_predictions.csv",
        "native_predictions.csv",
    }:
        raise ValueError(
            "M2 native prediction 必须是当前 evaluator 为该 checkpoint 生成的正式 "
            f"test_predictions.csv/native_predictions.csv；期望目录={expected_path.parent}"
        )
    frame = pd.read_csv(path)
    required = {
        "patient_id",
        "fold",
        "split",
        "model_name",
        "decision_point",
        "y_true",
        "predicted_probability",
        "threshold",
        "run_name",
        "checkpoint",
        "checkpoint_sha256",
        "fold_manifest_sha256",
        "resolved_config_sha256",
        "implementation_sha256",
        "feature_schema_sha256",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"M2 native prediction 缺列: {sorted(missing)}")
    frame = frame.loc[frame["split"].eq("test")].copy()
    if frame.empty:
        raise ValueError("M2 native prediction 没有 test 行")
    if not frame["fold"].eq(m2.fold).all() or not frame["model_name"].eq("m2").all():
        raise ValueError("M2 native prediction 的 fold/model_name 与 checkpoint 不一致")
    provenance_expectations = {
        "run_name": m2.run_name,
        "checkpoint": m2.checkpoint,
        "checkpoint_sha256": m2.checkpoint_sha256,
        "fold_manifest_sha256": m2.fold_manifest_sha256,
        "resolved_config_sha256": m2.resolved_config_sha256,
        "implementation_sha256": m2.implementation_sha256,
        "feature_schema_sha256": m2.feature_schema_sha256,
    }
    for column, expected in provenance_expectations.items():
        if not frame[column].astype(str).eq(expected).all():
            raise ValueError(
                f"M2 native prediction 的 {column} 与 checkpoint/evaluator contract 不一致"
            )
    if frame.duplicated(["patient_id", "decision_point"]).any():
        raise ValueError("M2 native test prediction 存在重复 patient/decision point")
    if not frame["decision_point"].isin(DECISION_POINTS).all():
        raise ValueError("M2 native prediction 含未知 decision point")

    rows: list[pd.DataFrame] = []
    label_lookup = dict(zip(m2.splits["test"].patient_ids, m2.splits["test"].labels))
    for point in DECISION_POINTS:
        expected_ids = paired_test[point].patient_ids
        point_frame = frame.loc[frame["decision_point"].eq(point)]
        locked_test_ids = m2.splits["test"].patient_ids
        if len(point_frame) != len(locked_test_ids) or set(
            point_frame["patient_id"].astype(str)
        ) != set(locked_test_ids):
            raise ValueError(f"C2/{point} 未恰好覆盖锁定的完整 fold-test patient IDs")
        by_id = point_frame.set_index("patient_id", drop=False)
        if missing_ids := set(expected_ids).difference(by_id.index):
            raise ValueError(
                f"C2/{point} 缺 paired test prediction: {sorted(missing_ids)[:5]}"
            )
        subset = by_id.loc[list(expected_ids)].reset_index(drop=True)
        if tuple(subset["patient_id"].astype(str)) != expected_ids:
            raise RuntimeError("C2 paired test 顺序不一致")
        expected_labels = np.asarray(
            [label_lookup[patient_id] for patient_id in expected_ids], dtype=np.int64
        )
        if not np.array_equal(
            subset["y_true"].to_numpy(dtype=np.int64), expected_labels
        ):
            raise ValueError("C2 prediction 的 y_true 与锁定 split 标签不一致")
        probability = subset["predicted_probability"].to_numpy(dtype=np.float64)
        threshold = subset["threshold"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ValueError("C2 predicted_probability 非法")
        if not np.isfinite(threshold).all() or np.unique(threshold).size != 1:
            raise ValueError(
                "C2 每个 decision point 必须有唯一有限 validation threshold"
            )
        output = pd.DataFrame(
            {
                "patient_id": subset["patient_id"].astype(str),
                "fold": m2.fold,
                "split": "test",
                "control_name": "C2_m2_image_only",
                "decision_point": point,
                "y_true": expected_labels,
                "predicted_probability": probability,
                "predicted_label": (probability >= threshold[0]).astype(int),
                "predicted_label_0_5": (probability >= 0.5).astype(int),
                "threshold": float(threshold[0]),
                "paired_subset": True,
                "radiomics_used_as_input": False,
                "feature_schema": "正式 evaluate_fold 的冻结 M2 image-only readout；未重拟合",
                "observed_visits": point,
                "latest_observed_visit": f"T{POINT_TO_LAST_VISIT[point]}",
                "source_run_name": m2.run_name,
                "source_checkpoint": m2.checkpoint,
                "source_checkpoint_sha256": m2.checkpoint_sha256,
                "probability_source": str(path),
                "selection_scope": "沿用正式 M2 image-only readout 的 fold-validation 选择；本控制不重拟合",
            }
        )
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def _default_native_prediction_path(m2: ModelExtraction) -> Path:
    evaluator_hash = evaluation_implementation_sha256()
    namespace = f"{m2.checkpoint_sha256[:12]}_eval{evaluator_hash[:12]}_epoch{m2.epoch}"
    return (
        EXPERIMENT_ROOT
        / "predictions"
        / m2.run_name
        / f"fold_{m2.fold}"
        / namespace
        / "test_predictions.csv"
    )


def _extract_checkpoint(
    expected_mode: str,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[ModelExtraction, Mapping[str, np.ndarray], RadiomicsChangeTransform]:
    """复用正式 evaluation API 安全加载并提取 train/val/test。"""

    evaluation = load_evaluation(checkpoint_path, config, device)
    if evaluation.mode != expected_mode:
        raise ValueError(
            f"期望 {expected_mode} checkpoint，实际 mode={evaluation.mode}: {checkpoint_path}"
        )
    # 与正式 evaluate_fold 相同：先 train/val，最后才读取 test cache。
    splits = {
        split: extract_native_split(evaluation, split, device, batch_size, workers)
        for split in ("train", "val")
    }
    splits["test"] = extract_native_split(
        evaluation, "test", device, batch_size, workers
    )
    extraction = ModelExtraction(
        mode=evaluation.mode,
        run_name=evaluation.run_name,
        fold=evaluation.fold,
        checkpoint=str(evaluation.checkpoint_path),
        checkpoint_sha256=evaluation.checkpoint_sha256,
        epoch=int(evaluation.payload.get("epoch", 0)),
        fold_manifest_sha256=str(
            evaluation.payload["data_contract"]["fold_manifest_sha256"]
        ),
        resolved_config_sha256=str(evaluation.payload["resolved_config_sha256"]),
        implementation_sha256=str(evaluation.payload["implementation_sha256"]),
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        splits=splits,
    )
    return (
        extraction,
        evaluation.bundle.raw_radiomics,
        evaluation.radiomics_transform,
    )


def _validate_common_contract(extractions: Mapping[str, ModelExtraction]) -> int:
    if set(extractions) != {"m0", "m1", "m2"}:
        raise ValueError("必须恰好提供 M0、M1、M2 三个 extraction")
    folds = {item.fold for item in extractions.values()}
    if len(folds) != 1:
        raise ValueError("M0/M1/M2 checkpoint 不属于同一 fold")
    for split_name in ("train", "val", "test"):
        reference = extractions["m2"].splits[split_name]
        for mode in ("m0", "m1"):
            candidate = extractions[mode].splits[split_name]
            if candidate.patient_ids != reference.patient_ids:
                raise ValueError(f"{mode}/{split_name} patient IDs/顺序与 M2 不一致")
            if not np.array_equal(candidate.labels, reference.labels):
                raise ValueError(f"{mode}/{split_name} 标签与 M2 不一致")
    return next(iter(folds))


def _control_prediction_rows(
    control_name: str,
    model: Any,
    threshold: float,
    views: Mapping[str, PairedView],
    m2: ModelExtraction,
    decision_point: str,
    include_image: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        view = views[split_name]
        components = [view.radiomics_features]
        if include_image:
            components.insert(
                0, m2.splits[split_name].features[decision_point][view.indices]
            )
        matrix = (
            np.concatenate(components, axis=1) if len(components) > 1 else components[0]
        )
        probability = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": view.patient_ids,
                    "fold": m2.fold,
                    "split": split_name,
                    "control_name": control_name,
                    "decision_point": decision_point,
                    "y_true": view.labels,
                    "predicted_probability": probability,
                    "predicted_label": (probability >= threshold).astype(int),
                    "predicted_label_0_5": (probability >= 0.5).astype(int),
                    "threshold": float(threshold),
                    "paired_subset": True,
                    "radiomics_used_as_input": True,
                    "feature_schema": (
                        "M2 image-derived concat(current,predicted_next,predicted_delta) + "
                        + ",".join(view.radiomics_feature_names)
                        if include_image
                        else ",".join(view.radiomics_feature_names)
                    ),
                    "observed_visits": decision_point,
                    "latest_observed_visit": f"T{POINT_TO_LAST_VISIT[decision_point]}",
                    "source_run_name": m2.run_name,
                    "source_checkpoint": m2.checkpoint,
                    "source_checkpoint_sha256": m2.checkpoint_sha256,
                    "probability_source": "本控制的 paired-train-only logistic",
                    "selection_scope": "C/penalty 与 threshold 仅由 paired fold validation 选择",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _ridge_probe(
    extraction: ModelExtraction,
    raw_targets: Mapping[str, np.ndarray],
    transform: RadiomicsChangeTransform,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """执行单模型三个决策点的 train-only Ridge probe。"""

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for point in DECISION_POINTS:
        point_index = POINT_TO_INDEX[point]
        prepared: dict[str, tuple[np.ndarray, tuple[str, ...], np.ndarray]] = {
            split_name: _ridge_targets(
                extraction.splits[split_name], raw_targets, transform, point
            )
            for split_name in ("train", "val")
        }
        train_indices, _, train_target = prepared["train"]
        validation_indices, _, validation_target = prepared["val"]
        train_input = extraction.splits["train"].predicted_delta[
            train_indices, point_index
        ]
        validation_input = extraction.splits["val"].predicted_delta[
            validation_indices, point_index
        ]
        model, selection = select_multioutput_ridge(
            train_input, train_target, validation_input, validation_target
        )
        selection.update(
            {
                "decision_point": point,
                "predicted_transition": POINT_TO_TRANSITION[point],
                "input_schema": "冻结 image-only world model 的 predicted_delta",
                "target_schema": "fold-train RadiomicsChangeTransform 的四维标准化相邻变化",
                "posthoc_probe_not_native_head": True,
            }
        )
        selections[point] = selection

        # 只有在 alpha 锁定以后才构建/预测 test paired subset。
        test_indices, test_ids, test_target = _ridge_targets(
            extraction.splits["test"], raw_targets, transform, point
        )
        test_input = extraction.splits["test"].predicted_delta[
            test_indices, point_index
        ]
        test_prediction = np.asarray(model.predict(test_input), dtype=np.float64)
        if (
            test_prediction.shape != test_target.shape
            or not np.isfinite(test_prediction).all()
        ):
            raise FloatingPointError("Ridge test prediction shape 非法或含 NaN/Inf")
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            target_standardized = test_target[:, feature_index]
            predicted_standardized = test_prediction[:, feature_index]
            target_change = transform.inverse_feature(
                feature_index, target_standardized
            )
            predicted_change = transform.inverse_feature(
                feature_index, predicted_standardized
            )
            for patient_index, patient_id in enumerate(test_ids):
                prediction_rows.append(
                    {
                        "patient_id": patient_id,
                        "fold": extraction.fold,
                        "split": "test",
                        "model_name": extraction.mode,
                        "run_name": extraction.run_name,
                        "decision_point": point,
                        "transition": POINT_TO_TRANSITION[point],
                        "feature_name": feature_name,
                        "posthoc_ridge_target_standardized": float(
                            target_standardized[patient_index]
                        ),
                        "posthoc_ridge_predicted_standardized": float(
                            predicted_standardized[patient_index]
                        ),
                        "posthoc_ridge_target_change": float(
                            target_change[patient_index]
                        ),
                        "posthoc_ridge_predicted_change": float(
                            predicted_change[patient_index]
                        ),
                        "transformed_change_unit": transform.features[
                            feature_index
                        ].value_transform,
                        "selected_alpha": selection["selected_alpha"],
                        "probe_type": "posthoc_train_only_multioutput_ridge",
                        "is_native_m2_head": False,
                        "input_is_predicted_delta": True,
                        "input_observed_visits": point,
                        "future_radiomics_used_as_input": False,
                        "source_checkpoint": extraction.checkpoint,
                        "source_checkpoint_sha256": extraction.checkpoint_sha256,
                    }
                )
            mse = float(mean_squared_error(target_standardized, predicted_standardized))
            metric_rows.append(
                {
                    "fold": extraction.fold,
                    "model_name": extraction.mode,
                    "decision_point": point,
                    "transition": POINT_TO_TRANSITION[point],
                    "feature_name": feature_name,
                    "n_test": len(test_ids),
                    "mae_standardized": float(
                        mean_absolute_error(target_standardized, predicted_standardized)
                    ),
                    "rmse_standardized": math.sqrt(mse),
                    "r2_standardized": float(
                        r2_score(target_standardized, predicted_standardized)
                    ),
                    "selected_alpha": selection["selected_alpha"],
                    "probe_type": "posthoc_train_only_multioutput_ridge",
                    "is_native_m2_head": False,
                }
            )
    return pd.DataFrame(prediction_rows), selections, pd.DataFrame(metric_rows)


def run_control_suite(
    checkpoint_paths: Mapping[str, Path],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    device_name: str,
    batch_size: int,
    workers: int,
    output_name: str,
    random_state: int = 2026,
    m2_native_predictions: Path | None = None,
) -> dict[str, Any]:
    """运行一个 fold 的 C0/C1/C2 与三模型 Ridge grounding。"""

    if set(checkpoint_paths) != {"m0", "m1", "m2"} or set(configs) != {
        "m0",
        "m1",
        "m2",
    }:
        raise ValueError("checkpoint_paths/configs 必须恰好包含 m0、m1、m2")
    if not _OUTPUT_NAME.fullmatch(output_name):
        raise ValueError(
            "output_name 只能含字母、数字、点、下划线、短横线，且不超过 128 字符"
        )
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size 必须为正，workers 不得小于 0")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但当前 CUDA 不可用")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index 越界: {device}")

    extractions: dict[str, ModelExtraction] = {}
    radiomics_contracts: dict[
        str, tuple[Mapping[str, np.ndarray], RadiomicsChangeTransform]
    ] = {}
    for mode in ("m0", "m1", "m2"):
        extraction, mode_raw, mode_transform = _extract_checkpoint(
            mode,
            checkpoint_paths[mode],
            configs[mode],
            device,
            batch_size,
            workers,
        )
        extractions[mode] = extraction
        radiomics_contracts[mode] = (mode_raw, mode_transform)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fold = _validate_common_contract(extractions)
    raw_hashes = {
        mode: raw_targets_hash(contract[0])
        for mode, contract in radiomics_contracts.items()
    }
    if len(set(raw_hashes.values())) != 1:
        raise ValueError("M0/M1/M2 checkpoint 锁定的 raw radiomics target 不一致")
    transform_payloads = {
        mode: contract[1].to_dict() for mode, contract in radiomics_contracts.items()
    }
    if any(
        transform_payloads[mode] != transform_payloads["m2"] for mode in ("m0", "m1")
    ):
        raise ValueError(
            "M0/M1/M2 checkpoint 锁定的 fold-train radiomics transform 不一致"
        )
    raw_targets, transform = radiomics_contracts["m2"]
    if transform.fold != fold:
        raise ValueError("M2 radiomics transform 与 checkpoint fold 不一致")

    m2 = extractions["m2"]
    paired: dict[str, dict[str, PairedView]] = {
        point: {
            split_name: paired_view(m2.splits[split_name], raw_targets, point)
            for split_name in ("train", "val")
        }
        for point in DECISION_POINTS
    }
    control_frames: list[pd.DataFrame] = []
    control_selections: dict[str, Any] = {}
    paired_counts: dict[str, Any] = {}
    for point_index, point in enumerate(DECISION_POINTS):
        views = paired[point]
        train = views["train"]
        validation = views["val"]
        paired_counts[point] = {
            "train": len(train.patient_ids),
            "val": len(validation.patient_ids),
        }
        for control_name, include_image in (
            ("C0_radiomics_only", False),
            ("C1_m2_image_plus_radiomics", True),
        ):
            train_parts = [train.radiomics_features]
            validation_parts = [validation.radiomics_features]
            if include_image:
                train_parts.insert(0, m2.splits["train"].features[point][train.indices])
                validation_parts.insert(
                    0, m2.splits["val"].features[point][validation.indices]
                )
            train_x = (
                np.concatenate(train_parts, axis=1)
                if len(train_parts) > 1
                else train_parts[0]
            )
            validation_x = (
                np.concatenate(validation_parts, axis=1)
                if len(validation_parts) > 1
                else validation_parts[0]
            )
            model, selection, threshold = select_control_logistic(
                train_x,
                train.labels,
                validation_x,
                validation.labels,
                random_state + fold * 1009 + point_index,
            )
            # 控制模型和阈值锁定后，才首次为该决策点筛选 paired test。
            if "test" not in views:
                views["test"] = paired_view(m2.splits["test"], raw_targets, point)
            paired_counts[point]["test"] = len(views["test"].patient_ids)
            selection.update(
                {
                    "control_name": control_name,
                    "decision_point": point,
                    "latest_allowed_observed_visit": f"T{POINT_TO_LAST_VISIT[point]}",
                    "radiomics_feature_names": list(train.radiomics_feature_names),
                    "future_radiomics_features_absent": True,
                }
            )
            control_selections.setdefault(point, {})[control_name] = {
                "model": selection,
                "threshold": threshold,
            }
            control_frames.append(
                _control_prediction_rows(
                    control_name,
                    model,
                    float(threshold["threshold"]),
                    views,
                    m2,
                    point,
                    include_image,
                )
            )

    native_path = (
        _default_native_prediction_path(m2)
        if m2_native_predictions is None
        else m2_native_predictions
    )
    paired_test = {point: paired[point]["test"] for point in DECISION_POINTS}
    c2 = _load_native_c2(native_path, m2, paired_test)
    controls = pd.concat([*control_frames, c2], ignore_index=True)

    control_metric_rows: list[dict[str, Any]] = []
    for (control_name, point), group in controls.loc[
        controls["split"].eq("test")
    ].groupby(["control_name", "decision_point"], sort=False):
        metrics = _classification_metrics(
            group["y_true"].to_numpy(),
            group["predicted_probability"].to_numpy(),
            float(group["threshold"].iloc[0]),
        )
        control_metric_rows.append(
            {
                "fold": fold,
                "control_name": control_name,
                "decision_point": point,
                **metrics,
                "paired_test_subset": True,
            }
        )
    control_metrics = pd.DataFrame(control_metric_rows)

    ridge_prediction_frames: list[pd.DataFrame] = []
    ridge_metric_frames: list[pd.DataFrame] = []
    ridge_selections: dict[str, Any] = {}
    for mode in ("m0", "m1", "m2"):
        predictions, selections, metrics = _ridge_probe(
            extractions[mode], raw_targets, transform
        )
        ridge_prediction_frames.append(predictions)
        ridge_metric_frames.append(metrics)
        ridge_selections[mode] = selections
    ridge_predictions = pd.concat(ridge_prediction_frames, ignore_index=True)
    ridge_metrics = pd.concat(ridge_metric_frames, ignore_index=True)

    controls_hash = controls_implementation_sha256()
    namespace = "_".join(
        [
            *(extractions[mode].checkpoint_sha256[:8] for mode in ("m0", "m1", "m2")),
            f"ctrl{controls_hash[:12]}",
        ]
    )
    prediction_dir = (
        EXPERIMENT_ROOT
        / "predictions"
        / "controls"
        / output_name
        / f"fold_{fold}"
        / namespace
    )
    metric_dir = (
        EXPERIMENT_ROOT
        / "metrics"
        / "controls"
        / output_name
        / f"fold_{fold}"
        / namespace
    )
    if prediction_dir.exists() or metric_dir.exists():
        raise FileExistsError(
            "该组 checkpoint 的 controls 输出已存在；为避免覆盖，未写入。"
            f" predictions={prediction_dir}, metrics={metric_dir}"
        )
    prediction_dir.mkdir(parents=True, exist_ok=False)
    metric_dir.mkdir(parents=True, exist_ok=False)
    _write_csv_new(prediction_dir / "paired_control_predictions.csv", controls)
    _write_csv_new(
        prediction_dir / "posthoc_ridge_test_predictions.csv", ridge_predictions
    )
    _write_csv_new(metric_dir / "paired_control_test_metrics.csv", control_metrics)
    _write_csv_new(metric_dir / "posthoc_ridge_test_metrics.csv", ridge_metrics)

    selection_payload = {
        "schema_version": 1,
        "fold": fold,
        "output_name": output_name,
        "controls_implementation_sha256": controls_hash,
        "checkpoints": {
            mode: {
                "run_name": item.run_name,
                "path": item.checkpoint,
                "sha256": item.checkpoint_sha256,
                "epoch": item.epoch,
                "fold_manifest_sha256": item.fold_manifest_sha256,
                "resolved_config_sha256": item.resolved_config_sha256,
                "implementation_sha256": item.implementation_sha256,
                "feature_schema_sha256": item.feature_schema_sha256,
            }
            for mode, item in extractions.items()
        },
        "m2_native_prediction_source": str(
            native_path.expanduser().resolve(strict=True)
        ),
        "raw_radiomics_sha256": raw_targets_hash(raw_targets),
        "radiomics_transform": transform.to_dict(),
        "paired_counts": paired_counts,
        "control_selection": control_selections,
        "posthoc_ridge_selection": ridge_selections,
        "leakage_guards": {
            "C0_C1_scaler_and_logistic_fit_scope": "paired fold train only",
            "C0_C1_hyperparameter_and_threshold_scope": "paired fold validation only",
            "C2_refit": False,
            "C2_test_subset": "与 C0/C1 相同的 paired test patient IDs",
            "radiomics_control_future_values_absent": True,
            "ridge_scaler_and_model_fit_scope": "paired fold train only",
            "ridge_alpha_scope": "paired fold validation only",
            "ridge_test_used_for_selection": False,
            "ridge_is_native_m2_head": False,
        },
    }
    _write_json_new(metric_dir / "selection.json", selection_payload)
    summary = {
        "schema_version": 1,
        "status": "完成",
        "fold": fold,
        "output_name": output_name,
        "controls_implementation_sha256": controls_hash,
        "control_prediction_rows": len(controls),
        "control_test_metric_rows": len(control_metrics),
        "posthoc_ridge_test_prediction_rows": len(ridge_predictions),
        "posthoc_ridge_test_metric_rows": len(ridge_metrics),
        "paired_counts": paired_counts,
        "prediction_dir": str(prediction_dir),
        "metric_dir": str(metric_dir),
        "interpretation": {
            "C0": "仅截至 decision point 的已观察 radiomics",
            "C1": "冻结 M2 image-derived 表征加同一 observed radiomics prefix",
            "C2": "正式 evaluate_fold 的 M2 image-only 概率在同一 paired test subset 上的切片",
            "ridge": "predicted_delta 的 train-only 后验 MultiOutput Ridge probe；不是原生 M2 head",
        },
    }
    _write_json_new(metric_dir / "summary.json", summary)
    return summary


def synthetic_self_test() -> dict[str, Any]:
    """快速验证无未来特征、train/val 选择和 MultiOutput Ridge 契约。"""

    generator = np.random.default_rng(20260806)
    raw = np.zeros((3, len(FEATURE_NAMES), 3), dtype=np.float64)
    visits = generator.uniform(0.5, 4.0, size=(4, len(FEATURE_NAMES)))
    for step in range(3):
        raw[step, :, 0] = visits[step]
        raw[step, :, 1] = visits[step + 1]
        raw[step, :, 2] = 1.0
    expected_dims = {"T0": 4, "T0-T1": 12, "T0-T2": 20}
    for point, expected_dim in expected_dims.items():
        observed, names = observed_radiomics_features(raw, point)
        if observed.shape != (expected_dim,) or len(names) != expected_dim:
            raise AssertionError(f"{point} synthetic observed feature 维度错误")

    future_modified = raw.copy()
    future_modified[2, :, 1] += 1e6  # 只改 T3；所有注册 decision point 都不应变化。
    for point in DECISION_POINTS:
        before, _ = observed_radiomics_features(raw, point)
        after, _ = observed_radiomics_features(future_modified, point)
        if not np.array_equal(before, after):
            raise AssertionError(f"{point} 意外使用未来 T3 radiomics")

    train_x = generator.normal(size=(48, 7))
    validation_x = generator.normal(size=(24, 7))
    train_y = np.tile(np.array([0, 1], dtype=np.int64), 24)
    validation_y = np.tile(np.array([0, 1], dtype=np.int64), 12)
    logistic, logistic_selection, threshold = select_control_logistic(
        train_x, train_y, validation_x, validation_y, 2026
    )
    if logistic.predict_proba(validation_x).shape != (24, 2):
        raise AssertionError("synthetic logistic prediction shape 错误")

    weight = generator.normal(size=(7, len(FEATURE_NAMES)))
    ridge_train_y = train_x @ weight + generator.normal(scale=0.05, size=(48, 4))
    ridge_validation_y = validation_x @ weight + generator.normal(
        scale=0.05, size=(24, 4)
    )
    ridge, ridge_selection = select_multioutput_ridge(
        train_x, ridge_train_y, validation_x, ridge_validation_y
    )
    if ridge.predict(validation_x).shape != (24, 4):
        raise AssertionError("synthetic MultiOutput Ridge prediction shape 错误")
    return {
        "status": "通过",
        "future_feature_guard": True,
        "observed_feature_dims": expected_dims,
        "selected_logistic_penalty": logistic_selection["selected_penalty"],
        "selected_logistic_C": logistic_selection["selected_C"],
        "selected_threshold": threshold["threshold"],
        "selected_ridge_alpha": ridge_selection["selected_alpha"],
        "test_data_used_for_selection": False,
    }
