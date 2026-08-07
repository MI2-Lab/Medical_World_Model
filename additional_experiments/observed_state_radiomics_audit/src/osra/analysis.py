"""Observed-State Radiomics Decodability Audit 的严格 OOF 聚合与绘图。

本模块只读取 probe 保存的 outer-test prediction。所有 pooled 结果都由每名患者
唯一一次 outer-test prediction 构成；bootstrap 在 fold 内按患者有放回抽样。
模块不训练、重拟合或选择任何 probe，也不读取 test 以外的 feature。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .common import AUDIT_ROOT, REPO_ROOT, atomic_csv, atomic_json, file_sha256, source_sha256


SCHEMA_VERSION = 1
MODEL_ORDER = ("m0", "m1", "m2")
PRIMARY_TARGETS = ("ftv", "ld", "sphericity")
ALL_TARGETS = (*PRIMARY_TARGETS, "bpe")
TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")
INPUT_VARIANTS = (
    "current",
    "current_only",
    "observed_pair",
    "observed_difference",
    "observed_combined",
    "predicted_next_delta",
)
REQUIRED_COLUMNS = (
    "patient_id",
    "fold",
    "split",
    "task_type",
    "model",
    "run_name",
    "encoder_stream",
    "representation",
    "input_variant",
    "target_name",
    "timepoint",
    "transition",
    "probe_type",
    "feature_dim",
    "y_true_natural",
    "y_pred_natural",
    "y_true_standardized",
    "y_pred_standardized",
    "b0_prediction_natural",
    "b0_prediction_standardized",
    "zero_change_prediction_natural",
    "zero_change_prediction_standardized",
    "selected_alpha",
    "source_feature_file",
    "source_feature_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
)
GROUP_COLUMNS = (
    "task_type",
    "model",
    "run_name",
    "encoder_stream",
    "representation",
    "representation_role",
    "input_variant",
    "target_name",
    "timepoint",
    "transition",
    "probe_type",
    "feature_dim",
)
BOOTSTRAP_METRICS = (
    "mae",
    "rmse",
    "r2",
    "spearman",
    "pearson",
    "predicted_to_target_variance_ratio",
    "fold_centered_variance_ratio",
    "sign_accuracy",
    "shrinkage_accuracy",
    "shrinkage_balanced_accuracy",
    "rmse_gain_over_b0",
    "rmse_gain_over_zero_change",
)
COMPARISON_METRICS = (
    "mae_reduction",
    "rmse_reduction",
    "r2_gain",
    "spearman_gain",
    "pearson_gain",
    "sign_accuracy_gain",
    "rmse_gain_over_b0_difference",
)
OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


MODEL_LABELS = {"m0": "M0", "m1": "M1", "m2": "M2", "baseline": "Baseline"}
TARGET_LABELS = {"ftv": "FTV", "ld": "LD", "sphericity": "Sphericity", "bpe": "BPE"}
REPRESENTATION_LABELS = {
    "projected": "Projected global",
    "preprojector": "Pre-projector global",
    "global_pool": "GAP global",
    "roi_mean": "ROI local mean",
    "mask_geometry": "B2 mask geometry",
    "raw_roi_intensity": "B3 ROI intensity",
    "current_radiomics": "B1 current radiomics",
    "transition_delta": "Transition predicted delta",
    "other": "Other",
}


@dataclass(frozen=True)
class AnalysisRunConfig:
    prediction_dir: Path
    metric_dir: Path
    figure_dir: Path
    bootstrap_replicates: int = 2000
    seed: int = 20260807
    overwrite: bool = False
    allow_partial: bool = False


def analysis_implementation_sha256() -> str:
    """返回 analysis 与 CLI 源码哈希，供 summary/figure manifest 审计。"""

    paths = [Path(__file__).resolve(), AUDIT_ROOT / "scripts" / "aggregate_results.py"]
    return source_sha256([path for path in paths if path.is_file()])


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


def _portable_path(path: Path) -> str:
    """仓库内产物写为 repo-relative；外部临时输入保留绝对路径。"""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _normalise_transition(value: Any) -> str:
    text = str(value).strip().replace("->", "→").replace("–", "-")
    aliases = {
        "T0-T1": "T0→T1",
        "T1-T2": "T1→T2",
        "T2-T3": "T2→T3",
        "T0→T1": "T0→T1",
        "T1→T2": "T1→T2",
        "T2→T3": "T2→T3",
    }
    return aliases.get(text, text)


def representation_role(value: str) -> str:
    """把固定 representation 名称映射到绘图/公平比较角色。"""

    text = str(value).lower()
    if "current_radiomics" in text:
        return "current_radiomics"
    if "transition_predicted_delta" in text:
        return "transition_delta"
    if "mask_geometry" in text:
        return "mask_geometry"
    if "raw_roi_intensity" in text:
        return "raw_roi_intensity"
    if "roi_mean" in text:
        return "roi_mean"
    if "global_pool" in text or text.endswith("gap"):
        return "global_pool"
    if "preprojector" in text or "pre_projector" in text:
        return "preprojector"
    if "projected" in text:
        return "projected"
    return "other"


def _read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"prediction CSV 缺列 {missing}: {path}")
    if frame.empty:
        raise ValueError(f"prediction CSV 为空: {path}")
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["source_prediction_file"] = str(path.resolve())
    frame["source_prediction_sha256"] = file_sha256(path)
    return frame


def discover_predictions(prediction_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """发现并严格校验所有 prediction CSV，返回合并数据与文件 manifest。"""

    paths = sorted(path for path in prediction_dir.rglob("*.csv") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"未发现 prediction CSV: {prediction_dir}")
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for path in paths:
        frame = _read_prediction_file(path)
        frames.append(frame)
        manifest_rows.append(
            {
                "path": _portable_path(path),
                "sha256": frame["source_prediction_sha256"].iloc[0],
                "rows": len(frame),
                "patients": frame["patient_id"].astype(str).nunique(),
                "folds": ",".join(map(str, sorted(pd.to_numeric(frame["fold"]).unique()))),
            }
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return validate_predictions(combined), pd.DataFrame(manifest_rows)


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """规范化并验证 outer-test prediction 契约。"""

    output = frame.copy()
    string_columns = (
        "patient_id",
        "split",
        "task_type",
        "model",
        "run_name",
        "encoder_stream",
        "representation",
        "input_variant",
        "target_name",
        "timepoint",
        "transition",
        "probe_type",
        "source_feature_file",
        "source_feature_sha256",
        "source_checkpoint",
        "source_checkpoint_sha256",
    )
    for column in string_columns:
        output[column] = output[column].astype(str).str.strip()
    output["task_type"] = output["task_type"].str.lower()
    output["model"] = output["model"].str.lower()
    output["target_name"] = output["target_name"].str.lower()
    output["input_variant"] = output["input_variant"].str.lower()
    output["split"] = output["split"].str.lower()
    output["transition"] = output["transition"].map(_normalise_transition)
    output["representation_role"] = output["representation"].map(representation_role)

    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    output["feature_dim"] = pd.to_numeric(output["feature_dim"], errors="raise").astype(int)
    numeric_columns = (
        "y_true_natural",
        "y_pred_natural",
        "y_true_standardized",
        "y_pred_standardized",
        "b0_prediction_natural",
        "b0_prediction_standardized",
        "zero_change_prediction_natural",
        "zero_change_prediction_standardized",
        "selected_alpha",
    )
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    if set(output["fold"].unique()).difference(range(5)):
        raise ValueError("prediction fold 必须在 0–4")
    if not output["split"].eq("test").all():
        invalid = sorted(output.loc[~output["split"].eq("test"), "split"].unique())
        raise ValueError(f"聚合器只接受 outer-test prediction，发现 {invalid}")
    if not output["task_type"].isin({"static", "change"}).all():
        raise ValueError("task_type 必须为 static/change")
    if not output["target_name"].isin(ALL_TARGETS).all():
        raise ValueError("target_name 超出锁定 FTV/LD/Sphericity/BPE")
    if not output["input_variant"].isin(INPUT_VARIANTS).all():
        raise ValueError("input_variant 不在锁定集合")
    static = output["task_type"].eq("static")
    change = ~static
    if not output.loc[static, "timepoint"].isin(TIMEPOINTS).all():
        raise ValueError("static prediction 的 timepoint 非法")
    if not output.loc[static, "input_variant"].eq("current").all():
        raise ValueError("static prediction 的 input_variant 必须为 current")
    if not output.loc[change, "transition"].isin(TRANSITIONS).all():
        raise ValueError("change prediction 的 transition 非法")
    if output.loc[change, "input_variant"].eq("current").any():
        raise ValueError("change prediction 不得使用 static current 标签")

    always_finite = (
        "y_true_natural",
        "y_pred_natural",
        "y_true_standardized",
        "y_pred_standardized",
        "b0_prediction_natural",
        "b0_prediction_standardized",
    )
    for column in always_finite:
        values = output[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"prediction {column} 含 NaN/Inf")
    if not np.isfinite(output.loc[change, "zero_change_prediction_natural"]).all():
        raise FloatingPointError("change zero-change natural prediction 含 NaN/Inf")
    if not np.isfinite(output.loc[change, "zero_change_prediction_standardized"]).all():
        raise FloatingPointError("change zero-change standardized prediction 含 NaN/Inf")
    if (output["feature_dim"] <= 0).any():
        raise ValueError("feature_dim 必须为正")

    patient_fold = output[["patient_id", "fold"]].drop_duplicates()
    if patient_fold["patient_id"].duplicated().any():
        offenders = patient_fold.loc[patient_fold["patient_id"].duplicated(False), "patient_id"].head()
        raise ValueError(f"同一患者出现于多个 OOF test fold: {offenders.tolist()}")

    unique_key = ["patient_id", "fold", *GROUP_COLUMNS]
    if output.duplicated(unique_key).any():
        duplicate = output.loc[output.duplicated(unique_key, False), unique_key].head()
        raise ValueError(f"prediction cell 重复:\n{duplicate.to_string(index=False)}")

    truth_keys = ["patient_id", "fold", "task_type", "target_name", "timepoint", "transition"]
    truth_spread = output.groupby(truth_keys, dropna=False)["y_true_natural"].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    if (truth_spread > 1e-9).any():
        raise ValueError("同一患者/target cell 在不同 model/representation 的 y_true 不一致")
    b0_keys = [
        *truth_keys,
        "encoder_stream",
        "representation",
        "input_variant",
    ]
    b0_spread = output.groupby(b0_keys, dropna=False)["b0_prediction_natural"].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    if (b0_spread > 1e-9).any():
        raise ValueError(
            "同一 representation/input 有效 train mask 内跨 model 的 B0 prediction 不一致"
        )

    hashes = ("source_feature_sha256", "source_checkpoint_sha256")
    for column in hashes:
        if (~output[column].str.fullmatch(r"[0-9a-f]{64}")).any():
            raise ValueError(f"{column} 必须是 SHA-256")
    return output.sort_values([*GROUP_COLUMNS, "fold", "patient_id"], kind="mergesort").reset_index(drop=True)


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y).statistic if kind == "spearman" else pearsonr(x, y).statistic
    value = float(result)
    return value if math.isfinite(value) else math.nan


def _balanced_binary_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    classes = np.unique(target)
    recalls = [float(np.mean(prediction[target == item] == item)) for item in classes]
    return float(np.mean(recalls)) if len(recalls) == 2 else math.nan


def _fold_centered_variance_ratio(
    target: np.ndarray, prediction: np.ndarray, folds: np.ndarray
) -> tuple[float, float, float]:
    target_ss = 0.0
    prediction_ss = 0.0
    degrees = 0
    for fold in np.unique(folds):
        mask = folds == fold
        if mask.sum() < 2:
            continue
        target_values = target[mask]
        prediction_values = prediction[mask]
        target_ss += float(np.square(target_values - target_values.mean()).sum())
        prediction_ss += float(np.square(prediction_values - prediction_values.mean()).sum())
        degrees += int(mask.sum() - 1)
    if degrees <= 0:
        return math.nan, math.nan, math.nan
    target_variance = target_ss / degrees
    prediction_variance = prediction_ss / degrees
    ratio = prediction_variance / max(target_variance, 1e-12)
    return target_variance, prediction_variance, ratio


def regression_metrics(group: pd.DataFrame) -> dict[str, Any]:
    """计算单个 fold 或 pooled OOF cell 的完整连续结局指标。"""

    target = group["y_true_natural"].to_numpy(dtype=float)
    prediction = group["y_pred_natural"].to_numpy(dtype=float)
    target_std = group["y_true_standardized"].to_numpy(dtype=float)
    prediction_std = group["y_pred_standardized"].to_numpy(dtype=float)
    b0 = group["b0_prediction_natural"].to_numpy(dtype=float)
    folds = group["fold"].to_numpy(dtype=int)
    n = len(group)
    if n == 0:
        raise ValueError("空 regression cell")
    target_var = float(np.var(target, ddof=1)) if n > 1 else math.nan
    prediction_var = float(np.var(prediction, ddof=1)) if n > 1 else math.nan
    target_std_var = float(np.var(target_std, ddof=1)) if n > 1 else math.nan
    prediction_std_var = float(np.var(prediction_std, ddof=1)) if n > 1 else math.nan
    variance_ratio = prediction_var / max(target_var, 1e-12) if n > 1 else math.nan
    standardized_variance_ratio = (
        prediction_std_var / max(target_std_var, 1e-12) if n > 1 else math.nan
    )
    fc_target, fc_prediction, fc_ratio = _fold_centered_variance_ratio(
        target_std, prediction_std, folds
    )
    near_constant = bool(
        math.isfinite(fc_prediction)
        and math.isfinite(fc_target)
        and fc_prediction <= max(1e-10, 0.01 * fc_target)
    )
    rmse = math.sqrt(float(mean_squared_error(target, prediction)))
    b0_rmse = math.sqrt(float(mean_squared_error(target, b0)))
    task_type = str(group["task_type"].iloc[0])
    result: dict[str, Any] = {
        "n": n,
        "n_patients": group["patient_id"].nunique(),
        "n_folds": group["fold"].nunique(),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": rmse,
        "r2": float(r2_score(target, prediction)) if n >= 2 and target_var > 0 else math.nan,
        "spearman": _correlation(target, prediction, "spearman"),
        "pearson": _correlation(target, prediction, "pearson"),
        "target_variance_ddof1": target_var,
        "predicted_variance_ddof1": prediction_var,
        "predicted_to_target_variance_ratio": variance_ratio,
        "standardized_target_variance_ddof1": target_std_var,
        "standardized_predicted_variance_ddof1": prediction_std_var,
        "standardized_variance_ratio": standardized_variance_ratio,
        "fold_centered_target_variance": fc_target,
        "fold_centered_predicted_variance": fc_prediction,
        "fold_centered_variance_ratio": fc_ratio,
        "near_constant_prediction": near_constant,
        "near_constant_definition": (
            "fold-centered standardized Var(pred) <= max(1e-10, 0.01*Var(target))"
        ),
        "b0_mae": float(mean_absolute_error(target, b0)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / max(b0_rmse, 1e-12),
        "sign_accuracy": math.nan,
        "shrinkage_accuracy": math.nan,
        "shrinkage_balanced_accuracy": math.nan,
        "zero_change_mae": math.nan,
        "zero_change_rmse": math.nan,
        "rmse_gain_over_zero_change": math.nan,
    }
    if task_type == "change":
        true_sign = np.sign(target)
        predicted_sign = np.sign(prediction)
        true_shrinkage = target < 0
        predicted_shrinkage = prediction < 0
        zero = group["zero_change_prediction_natural"].to_numpy(dtype=float)
        zero_rmse = math.sqrt(float(mean_squared_error(target, zero)))
        result.update(
            {
                "sign_accuracy": float(np.mean(true_sign == predicted_sign)),
                "shrinkage_accuracy": float(np.mean(true_shrinkage == predicted_shrinkage)),
                "shrinkage_balanced_accuracy": _balanced_binary_accuracy(
                    true_shrinkage, predicted_shrinkage
                ),
                "zero_change_mae": float(mean_absolute_error(target, zero)),
                "zero_change_rmse": zero_rmse,
                "rmse_gain_over_zero_change": (zero_rmse - rmse) / max(zero_rmse, 1e-12),
            }
        )
    return result


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """产生逐 fold 与 pooled OOF 指标。"""

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby([*GROUP_COLUMNS, "fold"], sort=False, dropna=False):
        values = dict(zip([*GROUP_COLUMNS, "fold"], keys))
        fold_rows.append({"aggregation_level": "fold_test", **values, **regression_metrics(group)})
    for keys, group in predictions.groupby(list(GROUP_COLUMNS), sort=False, dropna=False):
        values = dict(zip(GROUP_COLUMNS, keys))
        oof_rows.append({"aggregation_level": "pooled_oof_test", **values, **regression_metrics(group)})
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def _stable_seed(seed: int, *values: Any) -> int:
    payload = json.dumps([int(seed), *map(str, values)], ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _bootstrap_indices(
    group: pd.DataFrame, replicates: int, seed: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """在每个 fold 内按患者抽样；返回 [R,N] row indices 与 fold block。"""

    if group.duplicated(["patient_id", "fold"]).any():
        raise ValueError("bootstrap cell 内同一 patient/fold 重复")
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    blocks: list[tuple[int, int]] = []
    offset = 0
    for fold in sorted(group["fold"].unique()):
        positions = np.flatnonzero(group["fold"].to_numpy(dtype=int) == int(fold))
        if positions.size == 0:
            continue
        sampled = rng.choice(positions, size=(replicates, positions.size), replace=True)
        parts.append(sampled)
        blocks.append((offset, offset + positions.size))
        offset += positions.size
    if not parts:
        raise ValueError("bootstrap 无 fold block")
    return np.concatenate(parts, axis=1), blocks


def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(x_centered).sum(axis=1) * np.square(y_centered).sum(axis=1)
    )
    output = np.full(x.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    output[valid] = (
        (x_centered[valid] * y_centered[valid]).sum(axis=1) / denominator[valid]
    )
    return output


def _bootstrap_values(
    group: pd.DataFrame, replicates: int, seed: int
) -> dict[str, np.ndarray]:
    indices, blocks = _bootstrap_indices(group, replicates, seed)
    target = group["y_true_natural"].to_numpy(dtype=float)[indices]
    prediction = group["y_pred_natural"].to_numpy(dtype=float)[indices]
    target_std = group["y_true_standardized"].to_numpy(dtype=float)[indices]
    prediction_std = group["y_pred_standardized"].to_numpy(dtype=float)[indices]
    b0 = group["b0_prediction_natural"].to_numpy(dtype=float)[indices]
    error = prediction - target
    b0_error = b0 - target
    mae = np.mean(np.abs(error), axis=1)
    rmse = np.sqrt(np.mean(np.square(error), axis=1))
    b0_rmse = np.sqrt(np.mean(np.square(b0_error), axis=1))
    target_centered = target - target.mean(axis=1, keepdims=True)
    target_ss = np.square(target_centered).sum(axis=1)
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    prediction_ss = np.square(prediction_centered).sum(axis=1)
    r2 = np.full(replicates, np.nan, dtype=np.float64)
    valid_target = target_ss > 0
    r2[valid_target] = 1.0 - np.square(error[valid_target]).sum(axis=1) / target_ss[valid_target]
    variance_ratio = np.full(replicates, np.nan, dtype=np.float64)
    variance_ratio[valid_target] = prediction_ss[valid_target] / target_ss[valid_target]
    pearson = _rowwise_pearson(target, prediction)
    # rankdata(axis=1) 正确处理 bootstrap 重复患者形成的 ties。
    spearman = _rowwise_pearson(rankdata(target, axis=1), rankdata(prediction, axis=1))
    fc_target_ss = np.zeros(replicates, dtype=np.float64)
    fc_prediction_ss = np.zeros(replicates, dtype=np.float64)
    for start, stop in blocks:
        y_block = target_std[:, start:stop]
        p_block = prediction_std[:, start:stop]
        fc_target_ss += np.square(y_block - y_block.mean(axis=1, keepdims=True)).sum(axis=1)
        fc_prediction_ss += np.square(p_block - p_block.mean(axis=1, keepdims=True)).sum(axis=1)
    fold_centered_ratio = np.full(replicates, np.nan, dtype=np.float64)
    valid_fc = fc_target_ss > 0
    fold_centered_ratio[valid_fc] = fc_prediction_ss[valid_fc] / fc_target_ss[valid_fc]
    results: dict[str, np.ndarray] = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "pearson": pearson,
        "predicted_to_target_variance_ratio": variance_ratio,
        "fold_centered_variance_ratio": fold_centered_ratio,
        "rmse_gain_over_b0": (b0_rmse - rmse) / np.maximum(b0_rmse, 1e-12),
        "sign_accuracy": np.full(replicates, np.nan),
        "shrinkage_accuracy": np.full(replicates, np.nan),
        "shrinkage_balanced_accuracy": np.full(replicates, np.nan),
        "rmse_gain_over_zero_change": np.full(replicates, np.nan),
    }
    if str(group["task_type"].iloc[0]) == "change":
        results["sign_accuracy"] = np.mean(np.sign(target) == np.sign(prediction), axis=1)
        true_shrinkage = target < 0
        predicted_shrinkage = prediction < 0
        results["shrinkage_accuracy"] = np.mean(true_shrinkage == predicted_shrinkage, axis=1)
        true_positive = true_shrinkage.sum(axis=1)
        true_negative = (~true_shrinkage).sum(axis=1)
        positive_recall = np.divide(
            (true_shrinkage & predicted_shrinkage).sum(axis=1),
            true_positive,
            out=np.full(replicates, np.nan),
            where=true_positive > 0,
        )
        negative_recall = np.divide(
            ((~true_shrinkage) & (~predicted_shrinkage)).sum(axis=1),
            true_negative,
            out=np.full(replicates, np.nan),
            where=true_negative > 0,
        )
        results["shrinkage_balanced_accuracy"] = (positive_recall + negative_recall) / 2.0
        zero = group["zero_change_prediction_natural"].to_numpy(dtype=float)[indices]
        zero_rmse = np.sqrt(np.mean(np.square(zero - target), axis=1))
        results["rmse_gain_over_zero_change"] = (zero_rmse - rmse) / np.maximum(
            zero_rmse, 1e-12
        )
    return results


def bootstrap_table(
    predictions: pd.DataFrame,
    oof_metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """为每个 pooled OOF cell 产生 patient/fold-stratified percentile CI。"""

    rows: list[dict[str, Any]] = []
    metric_lookup = oof_metrics.set_index(list(GROUP_COLUMNS), drop=False)
    for keys, group in predictions.groupby(list(GROUP_COLUMNS), sort=False, dropna=False):
        values = dict(zip(GROUP_COLUMNS, keys))
        sampled = _bootstrap_values(group, replicates, _stable_seed(seed, *keys))
        point = metric_lookup.loc[keys]
        if isinstance(point, pd.DataFrame):
            raise ValueError("OOF metric key 不唯一")
        for metric in BOOTSTRAP_METRICS:
            array = np.asarray(sampled[metric], dtype=np.float64)
            finite = array[np.isfinite(array)]
            rows.append(
                {
                    **values,
                    "metric": metric,
                    "estimate": float(point[metric]) if pd.notna(point[metric]) else math.nan,
                    "ci_low": float(np.quantile(finite, 0.025)) if finite.size else math.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if finite.size else math.nan,
                    "bootstrap_replicates": replicates,
                    "valid_replicates": int(finite.size),
                    "bootstrap_unit": "patient within fold",
                    "ci_method": "percentile",
                    "seed": _stable_seed(seed, *keys),
                }
            )
    return pd.DataFrame(rows)


def core_bootstrap_mask(predictions: pd.DataFrame) -> pd.Series:
    """预注册 2,000 次 CI 范围；其余完整矩阵仍保留 fold/OOF 点估计。

    核心范围避免对约两千个探索性 cell 重复昂贵 rank bootstrap：
    static online 四层；change online observed-difference 四层、transition delta，
    以及 change 的 B2/B3。BPE、EMA、D1/D2/D4 仅保留点估计。
    """

    primary = predictions["target_name"].isin(PRIMARY_TARGETS)
    four_layers = predictions["representation_role"].isin(
        {"projected", "preprojector", "global_pool", "roi_mean"}
    )
    static_core = (
        predictions["task_type"].eq("static")
        & predictions["encoder_stream"].eq("online")
        & four_layers
        & predictions["input_variant"].eq("current")
        & predictions["model"].isin(MODEL_ORDER)
    )
    change_observed = (
        predictions["task_type"].eq("change")
        & predictions["encoder_stream"].eq("online")
        & four_layers
        & predictions["input_variant"].eq("observed_difference")
        & predictions["model"].isin(MODEL_ORDER)
    )
    transition = (
        predictions["task_type"].eq("change")
        & predictions["encoder_stream"].eq("transition")
        & predictions["representation_role"].eq("transition_delta")
        & predictions["input_variant"].eq("predicted_next_delta")
        & predictions["model"].isin(MODEL_ORDER)
    )
    change_baselines = (
        predictions["task_type"].eq("change")
        & predictions["representation_role"].isin({"mask_geometry", "raw_roi_intensity"})
        & predictions["input_variant"].eq("observed_difference")
    )
    return primary & (static_core | change_observed | transition | change_baselines)


def bootstrap_scope_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    mask = core_bootstrap_mask(predictions)
    group_count = predictions.loc[mask].groupby(list(GROUP_COLUMNS), dropna=False).ngroups
    return {
        "scope_name": "primary_core_2000",
        "included_prediction_rows": int(mask.sum()),
        "included_pooled_groups": int(group_count),
        "included_targets": list(PRIMARY_TARGETS),
        "included": [
            "static online projected/preprojector/GAP/ROI current",
            "change online projected/preprojector/GAP/ROI observed_difference",
            "change transition_predicted_delta/predicted_next_delta",
            "change B2 mask_geometry and B3 raw_roi_intensity observed_difference",
            "all included rows carry same-mask B0 train-mean comparator",
        ],
        "point_estimate_only": [
            "D1 current_only, D2 observed_pair, D4 observed_combined",
            "EMA target sensitivity",
            "BPE exploratory target",
            "B1 current-radiomics table baseline unless part of a registered paired figure",
            "static B2 mask_geometry and B3 raw_roi_intensity baselines",
        ],
        "reason_cn": "控制约两千 pooled cell 的 rank-bootstrap 成本，同时不降低核心CI的2000次重复。",
    }


def _pair_on_dimension(
    predictions: pd.DataFrame,
    *,
    dimension: str,
    left_value: str,
    right_value: str,
    comparison_type: str,
    filter_mask: pd.Series | None = None,
    require_equal_patient_sets: bool = True,
) -> pd.DataFrame:
    """按一个 method 维度构造严格 patient-paired prediction。"""

    selected = predictions if filter_mask is None else predictions.loc[filter_mask].copy()
    left = selected.loc[selected[dimension].eq(left_value)].copy()
    right = selected.loc[selected[dimension].eq(right_value)].copy()
    if left.empty or right.empty:
        return pd.DataFrame()
    excluded = {dimension}
    if dimension == "model":
        excluded.add("run_name")
    if dimension == "representation_role":
        excluded.update({"representation", "feature_dim"})
    context = [column for column in GROUP_COLUMNS if column not in excluded]
    join_keys = ["patient_id", "fold", *context]
    columns = [
        *join_keys,
        "y_true_natural",
        "y_pred_natural",
        "b0_prediction_natural",
    ]
    if left.duplicated(join_keys).any() or right.duplicated(join_keys).any():
        raise ValueError(f"{comparison_type} 在配对 key 上不唯一")
    outer = left[columns].merge(
        right[columns], on=join_keys, how="outer", suffixes=("_left", "_right"), indicator=True
    )
    if require_equal_patient_sets and not outer["_merge"].eq("both").all():
        counts = outer["_merge"].value_counts().to_dict()
        raise ValueError(f"{comparison_type} 左右 patient cell 不一致: {counts}")
    paired = outer.loc[outer["_merge"].eq("both")].drop(columns="_merge").copy()
    if paired.empty:
        return paired
    if not np.allclose(
        paired["y_true_natural_left"], paired["y_true_natural_right"], rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"{comparison_type} 左右 y_true 不一致")
    paired = paired.rename(
        columns={
            "y_true_natural_left": "y_true",
            "y_pred_natural_left": "left_prediction",
            "y_pred_natural_right": "right_prediction",
            "b0_prediction_natural_left": "left_b0_prediction",
            "b0_prediction_natural_right": "right_b0_prediction",
        }
    ).drop(columns=["y_true_natural_right"])
    paired["comparison_type"] = comparison_type
    paired["comparison_dimension"] = dimension
    paired["left_method"] = left_value
    paired["right_method"] = right_value
    paired["pair_scope"] = (
        "exact patient set" if require_equal_patient_sets else "available-patient intersection"
    )
    return paired


def _custom_transition_pair(predictions: pd.DataFrame) -> pd.DataFrame:
    """Observed projected delta 与 transition-predicted delta 的定位 comparator。

    两者不是 information-matched：observed delta 使用终点 MRI，而 transition
    predicted delta 只见当前 prefix。输出与图注必须保留此声明。
    """

    left = predictions.loc[
        predictions["task_type"].eq("change")
        & predictions["encoder_stream"].eq("online")
        & predictions["representation_role"].eq("projected")
        & predictions["input_variant"].eq("observed_difference")
    ].copy()
    right = predictions.loc[
        predictions["task_type"].eq("change")
        & predictions["encoder_stream"].eq("transition")
        & predictions["representation"].eq("transition_predicted_delta")
        & predictions["input_variant"].eq("predicted_next_delta")
    ].copy()
    if left.empty or right.empty:
        return pd.DataFrame()
    excluded = {
        "encoder_stream",
        "representation",
        "representation_role",
        "input_variant",
        "feature_dim",
    }
    context = [column for column in GROUP_COLUMNS if column not in excluded]
    join_keys = ["patient_id", "fold", *context]
    columns = [*join_keys, "y_true_natural", "y_pred_natural", "b0_prediction_natural"]
    if left.duplicated(join_keys).any() or right.duplicated(join_keys).any():
        raise ValueError("observed-vs-transition comparator key 不唯一")
    paired = left[columns].merge(
        right[columns], on=join_keys, how="outer", suffixes=("_left", "_right"), indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("observed-vs-transition comparator patient set 不一致")
    paired = paired.drop(columns="_merge")
    if not np.allclose(
        paired["y_true_natural_left"], paired["y_true_natural_right"], rtol=0.0, atol=1e-9
    ):
        raise ValueError("observed-vs-transition y_true 不一致")
    paired = paired.rename(
        columns={
            "y_true_natural_left": "y_true",
            "y_pred_natural_left": "left_prediction",
            "y_pred_natural_right": "right_prediction",
            "b0_prediction_natural_left": "left_b0_prediction",
            "b0_prediction_natural_right": "right_b0_prediction",
        }
    ).drop(columns=["y_true_natural_right"])
    paired["comparison_type"] = "observed_delta_vs_transition_predicted_delta"
    paired["comparison_dimension"] = "information_availability"
    paired["left_method"] = "online_projected/observed_difference"
    paired["right_method"] = "transition_predicted_delta/predicted_next_delta"
    paired["pair_scope"] = "exact patient set; NOT information-matched"
    paired["comparison_caveat_cn"] = (
        "observed delta 看见终点 MRI；transition predicted delta 只见当前 prefix。"
        "仅用于瓶颈定位，不是 information-matched baseline。"
    )
    return paired


def build_paired_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """构造预注册的 model、representation、B0 与 transition 配对。"""

    frames: list[pd.DataFrame] = []
    image = predictions["model"].isin(MODEL_ORDER)
    for left, right, label in (
        ("m1", "m0", "m1_minus_m0"),
        ("m2", "m1", "m2_minus_m1"),
    ):
        frame = _pair_on_dimension(
            predictions,
            dimension="model",
            left_value=left,
            right_value=right,
            comparison_type=label,
            filter_mask=image,
            require_equal_patient_sets=True,
        )
        if not frame.empty:
            frames.append(frame)

    core_variant = predictions["input_variant"].isin({"current", "observed_difference"})
    for left, right, label in (
        ("roi_mean", "global_pool", "roi_minus_gap_pooling"),
        ("preprojector", "projected", "preprojector_minus_projected"),
        ("roi_mean", "projected", "roi_minus_projected_global"),
    ):
        frame = _pair_on_dimension(
            predictions,
            dimension="representation_role",
            left_value=left,
            right_value=right,
            comparison_type=label,
            filter_mask=image & core_variant,
            require_equal_patient_sets=False if left == "roi_mean" else True,
        )
        if not frame.empty:
            frames.append(frame)

    transition = _custom_transition_pair(predictions)
    if not transition.empty:
        frames.append(transition)

    # 每个 probe 与该行自带、相同有效 patient mask 的 B0 train-mean comparator。
    b0 = predictions.copy()
    b0_context = list(GROUP_COLUMNS)
    b0 = b0.rename(
        columns={"y_true_natural": "y_true", "y_pred_natural": "left_prediction"}
    )
    b0["right_prediction"] = b0["b0_prediction_natural"]
    b0["left_b0_prediction"] = b0["b0_prediction_natural"]
    b0["right_b0_prediction"] = b0["b0_prediction_natural"]
    b0["comparison_type"] = "probe_vs_b0_train_mean"
    b0["comparison_dimension"] = "baseline"
    b0["left_method"] = b0["representation"] + "/" + b0["input_variant"]
    b0["right_method"] = "B0_train_mean"
    b0["pair_scope"] = "same-row exact patient set"
    frames.append(
        b0[
            [
                "patient_id",
                "fold",
                *b0_context,
                "y_true",
                "left_prediction",
                "right_prediction",
                "left_b0_prediction",
                "right_b0_prediction",
                "comparison_type",
                "comparison_dimension",
                "left_method",
                "right_method",
                "pair_scope",
            ]
        ]
    )
    return pd.concat(frames, ignore_index=True, sort=False)


def core_paired_bootstrap_mask(paired: pd.DataFrame) -> pd.Series:
    """限制配对 CI 到与 :func:`core_bootstrap_mask` 相同的核心科学范围。"""

    primary = paired["target_name"].isin(PRIMARY_TARGETS)
    comparison = paired["comparison_type"].astype(str)
    registered_representation = comparison.isin(
        {
            "roi_minus_gap_pooling",
            "preprojector_minus_projected",
            "roi_minus_projected_global",
        }
    ) & paired.get("encoder_stream", pd.Series("", index=paired.index)).eq("online")
    registered_transition = comparison.eq("observed_delta_vs_transition_predicted_delta")
    four_layers = paired.get("representation_role", pd.Series("", index=paired.index)).isin(
        {"projected", "preprojector", "global_pool", "roi_mean"}
    )
    static_core = (
        paired["task_type"].eq("static")
        & paired.get("encoder_stream", pd.Series("", index=paired.index)).eq("online")
        & four_layers
        & paired["input_variant"].eq("current")
    )
    change_core = (
        paired["task_type"].eq("change")
        & paired.get("encoder_stream", pd.Series("", index=paired.index)).eq("online")
        & four_layers
        & paired["input_variant"].eq("observed_difference")
    )
    transition_core = (
        paired.get("representation_role", pd.Series("", index=paired.index)).eq("transition_delta")
        & paired["input_variant"].eq("predicted_next_delta")
    )
    baseline_core = (
        paired["task_type"].eq("change")
        & paired.get("representation_role", pd.Series("", index=paired.index)).isin(
            {"mask_geometry", "raw_roi_intensity"}
        )
        & paired["input_variant"].eq("observed_difference")
    )
    model_comparison = comparison.isin({"m1_minus_m0", "m2_minus_m1"}) & (
        static_core | change_core | transition_core
    )
    b0_comparison = comparison.eq("probe_vs_b0_train_mean") & (
        static_core | change_core | transition_core | baseline_core
    )
    return primary & (
        registered_representation | registered_transition | model_comparison | b0_comparison
    )


def _comparison_point(target: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_mae = float(mean_absolute_error(target, left))
    right_mae = float(mean_absolute_error(target, right))
    left_rmse = math.sqrt(float(mean_squared_error(target, left)))
    right_rmse = math.sqrt(float(mean_squared_error(target, right)))
    target_variance = float(np.var(target, ddof=1)) if len(target) > 1 else math.nan
    left_r2 = float(r2_score(target, left)) if target_variance > 0 else math.nan
    right_r2 = float(r2_score(target, right)) if target_variance > 0 else math.nan
    task_is_change = True  # caller overwrites sign metric for static below
    return {
        "mae_reduction": right_mae - left_mae,
        "rmse_reduction": right_rmse - left_rmse,
        "r2_gain": left_r2 - right_r2,
        "spearman_gain": _correlation(target, left, "spearman")
        - _correlation(target, right, "spearman"),
        "pearson_gain": _correlation(target, left, "pearson")
        - _correlation(target, right, "pearson"),
        "sign_accuracy_gain": (
            float(np.mean(np.sign(target) == np.sign(left)))
            - float(np.mean(np.sign(target) == np.sign(right)))
            if task_is_change
            else math.nan
        ),
        "rmse_gain_over_b0_difference": math.nan,
    }


def _comparison_bootstrap_values(
    group: pd.DataFrame, replicates: int, seed: int
) -> dict[str, np.ndarray]:
    indices, _ = _bootstrap_indices(group, replicates, seed)
    target = group["y_true"].to_numpy(dtype=float)[indices]
    left = group["left_prediction"].to_numpy(dtype=float)[indices]
    right = group["right_prediction"].to_numpy(dtype=float)[indices]
    left_b0 = group["left_b0_prediction"].to_numpy(dtype=float)[indices]
    right_b0 = group["right_b0_prediction"].to_numpy(dtype=float)[indices]
    left_error = left - target
    right_error = right - target
    left_mae = np.mean(np.abs(left_error), axis=1)
    right_mae = np.mean(np.abs(right_error), axis=1)
    left_rmse = np.sqrt(np.mean(np.square(left_error), axis=1))
    right_rmse = np.sqrt(np.mean(np.square(right_error), axis=1))
    target_centered = target - target.mean(axis=1, keepdims=True)
    target_ss = np.square(target_centered).sum(axis=1)
    left_r2 = np.full(replicates, np.nan)
    right_r2 = np.full(replicates, np.nan)
    valid = target_ss > 0
    left_r2[valid] = 1.0 - np.square(left_error[valid]).sum(axis=1) / target_ss[valid]
    right_r2[valid] = 1.0 - np.square(right_error[valid]).sum(axis=1) / target_ss[valid]
    left_b0_rmse = np.sqrt(np.mean(np.square(left_b0 - target), axis=1))
    right_b0_rmse = np.sqrt(np.mean(np.square(right_b0 - target), axis=1))
    left_b0_gain = (left_b0_rmse - left_rmse) / np.maximum(left_b0_rmse, 1e-12)
    right_b0_gain = (right_b0_rmse - right_rmse) / np.maximum(right_b0_rmse, 1e-12)
    output = {
        "mae_reduction": right_mae - left_mae,
        "rmse_reduction": right_rmse - left_rmse,
        "r2_gain": left_r2 - right_r2,
        "spearman_gain": _rowwise_pearson(rankdata(target, axis=1), rankdata(left, axis=1))
        - _rowwise_pearson(rankdata(target, axis=1), rankdata(right, axis=1)),
        "pearson_gain": _rowwise_pearson(target, left) - _rowwise_pearson(target, right),
        "sign_accuracy_gain": np.mean(np.sign(target) == np.sign(left), axis=1)
        - np.mean(np.sign(target) == np.sign(right), axis=1),
        "rmse_gain_over_b0_difference": left_b0_gain - right_b0_gain,
    }
    return output


PAIR_META_COLUMNS = (
    "comparison_type",
    "comparison_dimension",
    "left_method",
    "right_method",
    "pair_scope",
)


def paired_difference_tables(
    paired: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 fold point estimates 与 pooled paired-bootstrap CI。"""

    context_candidates = [*GROUP_COLUMNS, *PAIR_META_COLUMNS]
    context = [column for column in context_candidates if column in paired.columns]
    fold_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for keys, group in paired.groupby([*context, "fold"], sort=False, dropna=False):
        values = dict(zip([*context, "fold"], keys))
        point = _comparison_point(
            group["y_true"].to_numpy(dtype=float),
            group["left_prediction"].to_numpy(dtype=float),
            group["right_prediction"].to_numpy(dtype=float),
        )
        if values.get("task_type") != "change":
            point["sign_accuracy_gain"] = math.nan
        left_b0_rmse = math.sqrt(
            mean_squared_error(group["y_true"], group["left_b0_prediction"])
        )
        right_b0_rmse = math.sqrt(
            mean_squared_error(group["y_true"], group["right_b0_prediction"])
        )
        left_rmse = math.sqrt(mean_squared_error(group["y_true"], group["left_prediction"]))
        right_rmse = math.sqrt(mean_squared_error(group["y_true"], group["right_prediction"]))
        point["rmse_gain_over_b0_difference"] = (
            (left_b0_rmse - left_rmse) / max(left_b0_rmse, 1e-12)
            - (right_b0_rmse - right_rmse) / max(right_b0_rmse, 1e-12)
        )
        fold_rows.append({**values, "n": len(group), **point})

    for keys, group in paired.groupby(context, sort=False, dropna=False):
        values = dict(zip(context, keys))
        point = _comparison_point(
            group["y_true"].to_numpy(dtype=float),
            group["left_prediction"].to_numpy(dtype=float),
            group["right_prediction"].to_numpy(dtype=float),
        )
        if values.get("task_type") != "change":
            point["sign_accuracy_gain"] = math.nan
        left_b0_rmse = math.sqrt(
            mean_squared_error(group["y_true"], group["left_b0_prediction"])
        )
        right_b0_rmse = math.sqrt(
            mean_squared_error(group["y_true"], group["right_b0_prediction"])
        )
        left_rmse = math.sqrt(mean_squared_error(group["y_true"], group["left_prediction"]))
        right_rmse = math.sqrt(mean_squared_error(group["y_true"], group["right_prediction"]))
        point["rmse_gain_over_b0_difference"] = (
            (left_b0_rmse - left_rmse) / max(left_b0_rmse, 1e-12)
            - (right_b0_rmse - right_rmse) / max(right_b0_rmse, 1e-12)
        )
        sampled = _comparison_bootstrap_values(group, replicates, _stable_seed(seed, *keys))
        for metric in COMPARISON_METRICS:
            array = np.asarray(sampled[metric], dtype=float)
            if values.get("task_type") != "change" and metric == "sign_accuracy_gain":
                array[:] = np.nan
            finite = array[np.isfinite(array)]
            bootstrap_rows.append(
                {
                    **values,
                    "n": len(group),
                    "n_patients": group["patient_id"].nunique(),
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_low": float(np.quantile(finite, 0.025)) if finite.size else math.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if finite.size else math.nan,
                    "bootstrap_replicates": replicates,
                    "valid_replicates": int(finite.size),
                    "bootstrap_unit": "paired patient within fold",
                    "ci_method": "percentile",
                    "seed": _stable_seed(seed, *keys),
                    "positive_definition": (
                        "left method better; RMSE/MAE use right-left reduction, "
                        "correlation/R2 use left-right gain"
                    ),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(bootstrap_rows)


def _setup_matplotlib() -> None:
    noto = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    family = "Noto Sans CJK JP"
    if noto.is_file():
        font_manager.fontManager.addfont(str(noto))
        family = font_manager.FontProperties(fname=str(noto)).get_name()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Droid Sans Fallback", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _empty_figure(title: str, message: str) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.axis("off")
    axis.text(0.5, 0.6, title, ha="center", va="center", fontsize=16, weight="bold")
    axis.text(0.5, 0.4, message, ha="center", va="center", fontsize=11)
    return figure


def _require_or_placeholder(frame: pd.DataFrame, title: str, allow_partial: bool) -> Figure | None:
    if not frame.empty:
        return None
    if not allow_partial:
        raise ValueError(f"生成图表缺核心数据: {title}")
    return _empty_figure(title, "allow-partial 模式：该图所需 prediction cell 尚不完整")


def _with_ci(metrics: pd.DataFrame, bootstrap: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = metrics.copy()
    ci = bootstrap.loc[bootstrap["metric"].eq(metric)].copy()
    merge_keys = [column for column in GROUP_COLUMNS if column in ci.columns and column in data.columns]
    if ci.empty:
        data["ci_low"] = np.nan
        data["ci_high"] = np.nan
        return data
    ci = ci[[*merge_keys, "ci_low", "ci_high"]]
    return data.merge(ci, on=merge_keys, how="left", validate="one_to_one")


def _errorbar(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    lower = np.where(np.isfinite(low), np.maximum(y - low, 0.0), 0.0)
    upper = np.where(np.isfinite(high), np.maximum(high - y, 0.0), 0.0)
    has_ci = np.isfinite(low) & np.isfinite(high)
    yerr: np.ndarray | None = np.vstack([lower, upper]) if has_ci.any() else None
    axis.errorbar(
        x,
        y,
        yerr=yerr,
        marker=marker,
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label=label,
        color=color,
    )


ROLE_COLORS = {
    "projected": "#2c7fb8",
    "preprojector": "#7fcdbb",
    "global_pool": "#f03b20",
    "roi_mean": "#31a354",
    "transition_delta": "#756bb1",
    "mask_geometry": "#636363",
    "raw_roi_intensity": "#e6550d",
    "current_radiomics": "#969696",
}


def figure_static_spearman(
    oof: pd.DataFrame, bootstrap: pd.DataFrame, allow_partial: bool
) -> Figure:
    data = _with_ci(oof, bootstrap, "spearman")
    data = data.loc[
        data["task_type"].eq("static")
        & data["target_name"].isin(PRIMARY_TARGETS)
        & data["model"].isin(MODEL_ORDER)
        & data["encoder_stream"].eq("online")
        & data["representation_role"].isin(
            {"projected", "preprojector", "global_pool", "roi_mean"}
        )
    ]
    placeholder = _require_or_placeholder(data, "Static measurement 线性可解码性", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    roles = ("projected", "preprojector", "global_pool", "roi_mean")
    for row, model in enumerate(MODEL_ORDER):
        for column, target in enumerate(PRIMARY_TARGETS):
            axis = axes[row, column]
            cell = data.loc[data["model"].eq(model) & data["target_name"].eq(target)]
            for role in roles:
                part = cell.loc[cell["representation_role"].eq(role)].set_index("timepoint").reindex(TIMEPOINTS)
                if part["spearman"].notna().any():
                    _errorbar(
                        axis,
                        np.arange(4),
                        part["spearman"].to_numpy(dtype=float),
                        part["ci_low"].to_numpy(dtype=float),
                        part["ci_high"].to_numpy(dtype=float),
                        label=REPRESENTATION_LABELS[role],
                        color=ROLE_COLORS[role],
                    )
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.set_xticks(range(4), TIMEPOINTS)
            axis.set_ylim(-0.45, 1.0)
            axis.set_title(f"{MODEL_LABELS[model]} · {TARGET_LABELS[target]}")
            if column == 0:
                axis.set_ylabel("Spearman ρ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Observed static measurement：global 与 ROI feature 的 OOF Spearman", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False,
    )
    figure.text(0.5, 0.01, "误差线：fold 内 patient-cluster bootstrap 95% CI（2000次）", ha="center")
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    return figure


def figure_change_spearman(
    oof: pd.DataFrame, bootstrap: pd.DataFrame, allow_partial: bool
) -> Figure:
    data = _with_ci(oof, bootstrap, "spearman")
    data = data.loc[
        data["task_type"].eq("change")
        & data["target_name"].isin(PRIMARY_TARGETS)
        & data["model"].isin(MODEL_ORDER)
        & data["encoder_stream"].eq("online")
        & data["input_variant"].eq("observed_difference")
        & data["representation_role"].isin(
            {"projected", "preprojector", "global_pool", "roi_mean"}
        )
    ]
    placeholder = _require_or_placeholder(data, "Observed delta 线性可解码性", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    roles = ("projected", "preprojector", "global_pool", "roi_mean")
    for row, model in enumerate(MODEL_ORDER):
        for column, target in enumerate(PRIMARY_TARGETS):
            axis = axes[row, column]
            cell = data.loc[data["model"].eq(model) & data["target_name"].eq(target)]
            for role in roles:
                part = cell.loc[cell["representation_role"].eq(role)].set_index("transition").reindex(TRANSITIONS)
                if part["spearman"].notna().any():
                    _errorbar(
                        axis,
                        np.arange(3),
                        part["spearman"].to_numpy(dtype=float),
                        part["ci_low"].to_numpy(dtype=float),
                        part["ci_high"].to_numpy(dtype=float),
                        label=REPRESENTATION_LABELS[role],
                        color=ROLE_COLORS[role],
                    )
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.set_xticks(range(3), TRANSITIONS)
            axis.set_ylim(-0.45, 1.0)
            axis.set_title(f"{MODEL_LABELS[model]} · Δ{TARGET_LABELS[target]}")
            if column == 0:
                axis.set_ylabel("Spearman ρ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Observed global Δlatent 与 ROI Δfeature 的 OOF Spearman", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False,
    )
    figure.text(0.5, 0.01, "所有面板按 transition 单独拟合/评估；不以 pooled stage 取代", ha="center")
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    return figure


def figure_model_comparison(paired_ci: pd.DataFrame, allow_partial: bool) -> Figure:
    data = paired_ci.loc[
        paired_ci["comparison_type"].isin({"m1_minus_m0", "m2_minus_m1"})
        & paired_ci["metric"].eq("spearman_gain")
        & paired_ci["task_type"].eq("change")
        & paired_ci["target_name"].isin(PRIMARY_TARGETS)
        & paired_ci["input_variant"].eq("observed_difference")
        & paired_ci["encoder_stream"].eq("online")
        & paired_ci["representation_role"].isin({"projected", "roi_mean"})
    ].copy()
    placeholder = _require_or_placeholder(data, "M0/M1/M2 配对差异", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(1, 3, figsize=(18, 8), sharex=True)
    colors = {"m1_minus_m0": "#2c7fb8", "m2_minus_m1": "#d95f0e"}
    labels = {"m1_minus_m0": "M1−M0", "m2_minus_m1": "M2−M1"}
    for axis, target in zip(axes, PRIMARY_TARGETS):
        cell = data.loc[data["target_name"].eq(target)].copy()
        categories = [f"{transition} | {role}" for transition in TRANSITIONS for role in ("projected", "roi_mean")]
        y_positions = np.arange(len(categories))
        for comparison, offset in (("m1_minus_m0", -0.12), ("m2_minus_m1", 0.12)):
            part = cell.loc[cell["comparison_type"].eq(comparison)].copy()
            lookup = {
                f"{row.transition} | {row.representation_role}": row
                for row in part.itertuples(index=False)
            }
            for y, category in zip(y_positions + offset, categories):
                row = lookup.get(category)
                if row is None:
                    continue
                axis.errorbar(
                    row.estimate,
                    y,
                    xerr=[[max(row.estimate - row.ci_low, 0)], [max(row.ci_high - row.estimate, 0)]],
                    fmt="o",
                    color=colors[comparison],
                    capsize=2,
                    label=labels[comparison] if y == y_positions[0] + offset else None,
                )
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_yticks(y_positions, categories)
        axis.set_title(f"Δ{TARGET_LABELS[target]}")
        axis.set_xlabel("Spearman 差（正值=左侧 model 更好）")
    handles, labels_found = axes[0].get_legend_handles_labels()
    figure.suptitle("M1/M2 是否改变 observed image representation：patient-paired bootstrap", y=0.995)
    figure.legend(
        handles,
        labels_found,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def figure_delta_scatter(predictions: pd.DataFrame, target: str, allow_partial: bool) -> Figure:
    data = predictions.loc[
        predictions["task_type"].eq("change")
        & predictions["target_name"].eq(target)
        & predictions["model"].isin(MODEL_ORDER)
        & predictions["encoder_stream"].eq("online")
        & predictions["input_variant"].eq("observed_difference")
        & predictions["representation_role"].isin({"projected", "roi_mean"})
    ]
    placeholder = _require_or_placeholder(data, f"Δ{TARGET_LABELS[target]} OOF 散点", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(3, 3, figsize=(15, 14), sharex=True, sharey=True)
    for row, transition in enumerate(TRANSITIONS):
        for column, model in enumerate(MODEL_ORDER):
            axis = axes[row, column]
            cell = data.loc[data["transition"].eq(transition) & data["model"].eq(model)]
            limits: list[float] = []
            for role, marker in (("projected", "o"), ("roi_mean", "^")):
                part = cell.loc[cell["representation_role"].eq(role)]
                if part.empty:
                    continue
                axis.scatter(
                    part["y_true_natural"],
                    part["y_pred_natural"],
                    s=10,
                    alpha=0.35,
                    marker=marker,
                    color=ROLE_COLORS[role],
                    label=REPRESENTATION_LABELS[role],
                )
                limits.extend(part["y_true_natural"].tolist())
                limits.extend(part["y_pred_natural"].tolist())
            if limits:
                low, high = np.quantile(limits, [0.01, 0.99])
                axis.plot([low, high], [low, high], linestyle="--", color="#555555", linewidth=0.8)
            axis.set_title(f"{transition} · {MODEL_LABELS[model]}")
            if column == 0:
                axis.set_ylabel("预测 natural change")
            if row == 2:
                axis.set_xlabel("真实 natural change")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        f"Δ{TARGET_LABELS[target]}：outer-test OOF 真实值 vs 线性 probe 预测值",
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    return figure


def _heatmap_data(oof: pd.DataFrame, value: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    static = oof.loc[
        oof["task_type"].eq("static")
        & oof["target_name"].isin(PRIMARY_TARGETS)
        & oof["model"].isin(MODEL_ORDER)
        & oof["encoder_stream"].eq("online")
        & oof["representation_role"].isin(
            {"projected", "preprojector", "global_pool", "roi_mean"}
        )
    ].copy()
    static["cell"] = "S:" + static["target_name"] + "@" + static["timepoint"]
    change = oof.loc[
        oof["task_type"].eq("change")
        & oof["target_name"].isin(PRIMARY_TARGETS)
        & oof["model"].isin(MODEL_ORDER)
        & oof["encoder_stream"].eq("online")
        & oof["input_variant"].eq("observed_difference")
        & oof["representation_role"].isin(
            {"projected", "preprojector", "global_pool", "roi_mean"}
        )
    ].copy()
    change["cell"] = "Δ:" + change["target_name"] + "@" + change["transition"]
    data = pd.concat([static, change], ignore_index=True)
    data["row"] = data["model"].str.upper() + " | " + data["representation_role"].map(
        REPRESENTATION_LABELS
    )
    rows = [
        f"{model.upper()} | {REPRESENTATION_LABELS[role]}"
        for model in MODEL_ORDER
        for role in ("projected", "preprojector", "global_pool", "roi_mean")
    ]
    columns = [
        f"S:{target}@{timepoint}" for target in PRIMARY_TARGETS for timepoint in TIMEPOINTS
    ] + [
        f"Δ:{target}@{transition}" for target in PRIMARY_TARGETS for transition in TRANSITIONS
    ]
    pivot = data.pivot(index="row", columns="cell", values=value).reindex(index=rows, columns=columns)
    return pivot, rows, columns


def figure_r2_heatmap(oof: pd.DataFrame, allow_partial: bool) -> Figure:
    pivot, rows, columns = _heatmap_data(oof, "r2")
    placeholder = _require_or_placeholder(
        pivot.dropna(how="all"), "Static/change OOF R² heatmap", allow_partial
    )
    if placeholder is not None:
        return placeholder
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    bound = max(0.25, min(2.0, float(np.quantile(np.abs(finite), 0.95))))
    figure, axis = plt.subplots(figsize=(22, 10))
    image = axis.imshow(np.clip(values, -bound, bound), aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_yticks(range(len(rows)), rows)
    axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right")
    axis.axvline(len(PRIMARY_TARGETS) * len(TIMEPOINTS) - 0.5, color="black", linewidth=1.2)
    axis.set_title("OOF R² heatmap：static current 与 observed adjacent difference")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("R²（极端值为了显示截断）")
    figure.tight_layout()
    return figure


def figure_variance_heatmap(oof: pd.DataFrame, allow_partial: bool) -> Figure:
    pivot, rows, columns = _heatmap_data(oof, "fold_centered_variance_ratio")
    near, _, _ = _heatmap_data(oof, "near_constant_prediction")
    placeholder = _require_or_placeholder(
        pivot.dropna(how="all"), "Prediction/target variance ratio heatmap", allow_partial
    )
    if placeholder is not None:
        return placeholder
    values = pivot.to_numpy(dtype=float)
    log_values = np.log10(np.maximum(values, 1e-6))
    figure, axis = plt.subplots(figsize=(22, 10))
    image = axis.imshow(np.clip(log_values, -4, 2), aspect="auto", cmap="viridis", vmin=-4, vmax=2)
    axis.set_yticks(range(len(rows)), rows)
    axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right")
    axis.axvline(len(PRIMARY_TARGETS) * len(TIMEPOINTS) - 0.5, color="white", linewidth=1.2)
    near_values = near.to_numpy(dtype=object) == True  # noqa: E712 - NaN 应映射为 False
    for row, column in zip(*np.where(near_values)):
        axis.text(column, row, "×", ha="center", va="center", color="white", fontsize=8, weight="bold")
    axis.set_title("Fold-centered prediction/target variance ratio（× = near-constant）")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("log10 variance ratio")
    figure.tight_layout()
    return figure


def figure_roi_global_difference(paired_ci: pd.DataFrame, allow_partial: bool) -> Figure:
    data = paired_ci.loc[
        paired_ci["comparison_type"].eq("roi_minus_gap_pooling")
        & paired_ci["task_type"].eq("change")
        & paired_ci["target_name"].isin(PRIMARY_TARGETS)
        & paired_ci["metric"].isin({"spearman_gain", "r2_gain"})
        & paired_ci["input_variant"].eq("observed_difference")
        & paired_ci["encoder_stream"].eq("online")
        & paired_ci["model"].isin(MODEL_ORDER)
    ].copy()
    placeholder = _require_or_placeholder(data, "ROI−GAP pooling 配对差异", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(1, 3, figsize=(18, 10), sharex=True)
    metric_colors = {"spearman_gain": "#238b45", "r2_gain": "#756bb1"}
    metric_labels = {"spearman_gain": "ΔSpearman", "r2_gain": "ΔR²"}
    categories = [f"{transition} | {model.upper()}" for transition in TRANSITIONS for model in MODEL_ORDER]
    for axis, target in zip(axes, PRIMARY_TARGETS):
        cell = data.loc[data["target_name"].eq(target)]
        positions = np.arange(len(categories))
        for metric, offset in (("spearman_gain", -0.12), ("r2_gain", 0.12)):
            lookup = {
                f"{row.transition} | {row.model.upper()}": row
                for row in cell.loc[cell["metric"].eq(metric)].itertuples(index=False)
            }
            for y, category in zip(positions + offset, categories):
                row = lookup.get(category)
                if row is None:
                    continue
                axis.errorbar(
                    row.estimate,
                    y,
                    xerr=[[max(row.estimate - row.ci_low, 0)], [max(row.ci_high - row.estimate, 0)]],
                    fmt="o",
                    color=metric_colors[metric],
                    capsize=2,
                    label=metric_labels[metric] if y == positions[0] + offset else None,
                )
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_yticks(positions, categories)
        axis.set_title(f"Δ{TARGET_LABELS[target]}")
        axis.set_xlabel("正值 = ROI128 优于 GAP128")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Global pooling 瓶颈诊断：同一 128-D spatial map 的 ROI mean − GAP", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def figure_transition_specific(oof: pd.DataFrame, allow_partial: bool) -> Figure:
    data = oof.loc[
        oof["task_type"].eq("change")
        & oof["target_name"].isin(PRIMARY_TARGETS)
        & oof["model"].isin(MODEL_ORDER)
        & oof["encoder_stream"].eq("online")
        & oof["input_variant"].eq("observed_difference")
        & oof["representation_role"].isin({"projected", "roi_mean"})
    ]
    placeholder = _require_or_placeholder(data, "Transition-specific 结果", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    offsets = {
        ("m0", "projected"): -0.25,
        ("m0", "roi_mean"): -0.15,
        ("m1", "projected"): -0.05,
        ("m1", "roi_mean"): 0.05,
        ("m2", "projected"): 0.15,
        ("m2", "roi_mean"): 0.25,
    }
    for axis, transition in zip(axes, TRANSITIONS):
        cell = data.loc[data["transition"].eq(transition)]
        for (model, role), offset in offsets.items():
            part = cell.loc[cell["model"].eq(model) & cell["representation_role"].eq(role)].set_index(
                "target_name"
            ).reindex(PRIMARY_TARGETS)
            axis.scatter(
                np.arange(3) + offset,
                part["spearman"],
                s=35,
                marker="o" if role == "projected" else "^",
                color={"m0": "#9ecae1", "m1": "#3182bd", "m2": "#de2d26"}[model],
                label=f"{model.upper()} {REPRESENTATION_LABELS[role]}",
            )
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set_xticks(range(3), [TARGET_LABELS[item] for item in PRIMARY_TARGETS])
        axis.set_title(transition)
        axis.set_ylabel("OOF Spearman ρ")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("不同治疗阶段的 observed-change decodability（不合并 transition）", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    return figure


def figure_baseline_comparison(oof: pd.DataFrame, allow_partial: bool) -> Figure:
    image_core = (
        oof["encoder_stream"].eq("online")
        & oof["representation_role"].isin({"projected", "roi_mean"})
    )
    simple_baseline = oof["representation_role"].isin({"mask_geometry", "raw_roi_intensity"})
    table_baseline = (
        oof["representation_role"].eq("current_radiomics")
        & oof["input_variant"].eq("current_only")
    )
    data = oof.loc[
        oof["task_type"].eq("change")
        & oof["target_name"].isin(PRIMARY_TARGETS)
        & (
            (oof["input_variant"].eq("observed_difference") & (image_core | simple_baseline))
            | table_baseline
        )
    ].copy()
    placeholder = _require_or_placeholder(data, "B0/B2/B3 baseline 比较", allow_partial)
    if placeholder is not None:
        return placeholder
    data["method"] = np.where(
        data["model"].isin(MODEL_ORDER),
        data["model"].str.upper() + " " + data["representation_role"].map(REPRESENTATION_LABELS),
        data["representation_role"].map(REPRESENTATION_LABELS),
    )
    figure, axes = plt.subplots(1, 3, figsize=(19, 7), sharey=True)
    methods = list(dict.fromkeys(data["method"].tolist()))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 1)))
    for axis, target in zip(axes, PRIMARY_TARGETS):
        cell = data.loc[data["target_name"].eq(target)]
        for method, color in zip(methods, palette):
            part = cell.loc[cell["method"].eq(method)].set_index("transition").reindex(TRANSITIONS)
            if part["rmse_gain_over_b0"].notna().any():
                axis.plot(
                    range(3), part["rmse_gain_over_b0"], marker="o", linewidth=1, label=method, color=color
                )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(range(3), TRANSITIONS)
        axis.set_title(f"Δ{TARGET_LABELS[target]}")
        axis.set_ylabel("相对 B0 train-mean RMSE gain")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Image representation 与 B1/B2/B3 baseline 的比较（B0为各 cell 同有效行 RMSE 基准）", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.76))
    return figure


def figure_observed_vs_transition(oof: pd.DataFrame, allow_partial: bool) -> Figure:
    observed = oof.loc[
        oof["task_type"].eq("change")
        & oof["target_name"].isin(PRIMARY_TARGETS)
        & oof["model"].isin(MODEL_ORDER)
        & oof["encoder_stream"].eq("online")
        & oof["representation_role"].eq("projected")
        & oof["input_variant"].eq("observed_difference")
    ].copy()
    observed["method"] = "Observed Δprojected"
    transition = oof.loc[
        oof["task_type"].eq("change")
        & oof["target_name"].isin(PRIMARY_TARGETS)
        & oof["model"].isin(MODEL_ORDER)
        & oof["encoder_stream"].eq("transition")
        & oof["representation_role"].eq("transition_delta")
        & oof["input_variant"].eq("predicted_next_delta")
    ].copy()
    transition["method"] = "Transition predicted Δ"
    data = pd.concat([observed, transition], ignore_index=True)
    placeholder = _require_or_placeholder(data, "Observed vs transition delta", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    colors = {"Observed Δprojected": "#238b45", "Transition predicted Δ": "#756bb1"}
    for row, model in enumerate(MODEL_ORDER):
        for column, target in enumerate(PRIMARY_TARGETS):
            axis = axes[row, column]
            cell = data.loc[data["model"].eq(model) & data["target_name"].eq(target)]
            for method in colors:
                part = cell.loc[cell["method"].eq(method)].set_index("transition").reindex(TRANSITIONS)
                if part["spearman"].notna().any():
                    axis.plot(range(3), part["spearman"], marker="o", color=colors[method], label=method)
            axis.axhline(0, color="#666666", linewidth=0.8)
            axis.set_xticks(range(3), TRANSITIONS)
            axis.set_title(f"{model.upper()} · Δ{TARGET_LABELS[target]}")
            axis.set_ylabel("OOF Spearman ρ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Observed Δlatent vs transition-predicted Δlatent：瓶颈定位 comparator", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.01,
        "重要：observed delta 看见终点 MRI；predicted delta 只见当前 prefix。两者不是 information-matched baseline。",
        ha="center",
        color="#8c2d04",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.925))
    return figure


def save_figures(
    predictions: pd.DataFrame,
    oof: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired_ci: pd.DataFrame,
    figure_dir: Path,
    *,
    allow_partial: bool,
) -> pd.DataFrame:
    """生成预注册的至少十类中文图，并返回带哈希 manifest。"""

    _setup_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    specifications: list[tuple[str, str, str, Any]] = [
        ("01_static_global_vs_roi_spearman.png", "Static global vs ROI Spearman", "oof_metrics + bootstrap_ci", lambda: figure_static_spearman(oof, bootstrap, allow_partial)),
        ("02_delta_global_vs_roi_spearman.png", "Observed global delta vs ROI delta Spearman", "oof_metrics + bootstrap_ci", lambda: figure_change_spearman(oof, bootstrap, allow_partial)),
        ("03_m0_m1_m2_paired_comparison.png", "M0/M1/M2 paired representation difference", "paired_difference_bootstrap_ci", lambda: figure_model_comparison(paired_ci, allow_partial)),
        ("04_delta_ftv_true_vs_predicted.png", "Delta FTV OOF scatter", "prediction CSV", lambda: figure_delta_scatter(predictions, "ftv", allow_partial)),
        ("05_delta_ld_true_vs_predicted.png", "Delta LD OOF scatter", "prediction CSV", lambda: figure_delta_scatter(predictions, "ld", allow_partial)),
        ("06_delta_sphericity_true_vs_predicted.png", "Delta Sphericity OOF scatter", "prediction CSV", lambda: figure_delta_scatter(predictions, "sphericity", allow_partial)),
        ("07_r2_heatmap.png", "Static/change R2 heatmap", "oof_metrics", lambda: figure_r2_heatmap(oof, allow_partial)),
        ("08_variance_ratio_heatmap.png", "Fold-centered variance ratio heatmap", "oof_metrics", lambda: figure_variance_heatmap(oof, allow_partial)),
        ("09_roi_minus_global_difference.png", "ROI128 minus GAP128 paired difference", "paired_difference_bootstrap_ci", lambda: figure_roi_global_difference(paired_ci, allow_partial)),
        ("10_transition_specific_results.png", "Transition-specific observed-delta results", "oof_metrics", lambda: figure_transition_specific(oof, allow_partial)),
        ("11_baseline_comparison.png", "B0/B1/B2/B3 baseline comparison", "oof_metrics", lambda: figure_baseline_comparison(oof, allow_partial)),
        ("12_observed_vs_transition_delta.png", "Observed vs transition-predicted delta comparator", "oof_metrics", lambda: figure_observed_vs_transition(oof, allow_partial)),
    ]
    rows: list[dict[str, Any]] = []
    for filename, title, source, builder in specifications:
        figure = builder()
        path = figure_dir / filename
        figure.savefig(path, bbox_inches="tight", metadata={"Title": title, "Author": "OSRA aggregate_results"})
        plt.close(figure)
        rows.append(
            {
                "figure": filename,
                "title_cn_or_en": title,
                "source": source,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "bootstrap_note": (
                    "核心 primary cell 为2000次fold内patient bootstrap；"
                    "EMA/BPE/D1/D2/D4仅点估计"
                ),
            }
        )
    return pd.DataFrame(rows)


def coverage_audit(
    predictions: pd.DataFrame, *, allow_partial: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """验证核心五折矩阵；strict 模式缺任一注册 cell 即失败。"""

    coverage = (
        predictions.groupby(list(GROUP_COLUMNS), dropna=False, sort=False)
        .agg(
            rows=("patient_id", "size"),
            patients=("patient_id", "nunique"),
            folds=(
                "fold",
                lambda values: ",".join(map(str, sorted(set(map(int, values))))),
            ),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    issues: list[dict[str, Any]] = []

    def require(description: str, mask: pd.Series, expected_combinations: Iterable[tuple[Any, ...]], columns: Sequence[str]) -> None:
        observed = set(map(tuple, predictions.loc[mask, list(columns)].drop_duplicates().itertuples(index=False, name=None)))
        expected = set(expected_combinations)
        for combination in sorted(expected.difference(observed), key=str):
            issues.append(
                {
                    "severity": "error",
                    "scope": description,
                    "issue": "missing registered cell",
                    "cell": json.dumps(dict(zip(columns, combination)), ensure_ascii=False),
                }
            )

    primary = predictions["target_name"].isin(PRIMARY_TARGETS)
    registered_representations = (
        "online_projected",
        "online_preprojector",
        "online_global_pool",
        "online_roi_mean",
        "ema_projected",
        "ema_preprojector",
        "ema_global_pool",
        "ema_roi_mean",
        "mask_geometry",
        "raw_roi_intensity",
    )
    static_full_expected = (
        (model, timepoint, target, representation)
        for model in MODEL_ORDER
        for timepoint in TIMEPOINTS
        for target in ALL_TARGETS
        for representation in registered_representations
    )
    require(
        "full static registered matrix",
        predictions["task_type"].eq("static") & predictions["input_variant"].eq("current"),
        static_full_expected,
        ("model", "timepoint", "target_name", "representation"),
    )
    change_full_expected = (
        (model, transition, target, representation, variant)
        for model in MODEL_ORDER
        for transition in TRANSITIONS
        for target in ALL_TARGETS
        for representation in registered_representations
        for variant in (
            "current_only",
            "observed_pair",
            "observed_difference",
            "observed_combined",
        )
    )
    require(
        "full change registered matrix",
        predictions["task_type"].eq("change")
        & predictions["representation"].isin(registered_representations),
        change_full_expected,
        ("model", "transition", "target_name", "representation", "input_variant"),
    )
    transition_expected = (
        (model, transition, target)
        for model in MODEL_ORDER
        for transition in TRANSITIONS
        for target in ALL_TARGETS
    )
    require(
        "transition-predicted delta comparator",
        predictions["task_type"].eq("change")
        & predictions["encoder_stream"].eq("transition")
        & predictions["representation"].eq("transition_predicted_delta")
        & predictions["input_variant"].eq("predicted_next_delta"),
        transition_expected,
        ("model", "transition", "target_name"),
    )
    b1_expected = (
        (model, transition, target)
        for model in MODEL_ORDER
        for transition in TRANSITIONS
        for target in ALL_TARGETS
    )
    require(
        "B1 current-radiomics baseline",
        predictions["task_type"].eq("change")
        & predictions["representation"].eq("current_radiomics")
        & predictions["input_variant"].eq("current_only"),
        b1_expected,
        ("model", "transition", "target_name"),
    )
    incomplete_fold = coverage.loc[coverage["fold_count"].ne(5)]
    for row in incomplete_fold.itertuples(index=False):
        issues.append(
            {
                "severity": "error",
                "scope": "full registered fivefold coverage",
                "issue": f"fold_count={int(row.fold_count)}; expected 5",
                "cell": json.dumps(
                    {column: getattr(row, column) for column in GROUP_COLUMNS},
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )
    issue_frame = pd.DataFrame(issues, columns=("severity", "scope", "issue", "cell"))
    if not allow_partial and not issue_frame.empty:
        preview = issue_frame.head(10).to_string(index=False)
        raise ValueError(f"严格覆盖审计失败（共 {len(issue_frame)} 项）:\n{preview}")
    return coverage, issue_frame


def _commit_staged_directories(
    stage_metric: Path,
    stage_figure: Path,
    metric_dir: Path,
    figure_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """以目录 rename 提交；显式 overwrite 时保留可恢复备份直到两边均成功。"""

    metric_dir.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (metric_dir, figure_dir) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出已存在，默认拒绝覆盖：" + ", ".join(map(str, existing)))
    backups: dict[Path, Path] = {}
    token = next(tempfile._get_candidate_names())  # noqa: SLF001 - 仅用作同目录唯一名
    committed: list[Path] = []
    try:
        for destination in existing:
            backup = destination.with_name(f".{destination.name}.backup.{token}")
            destination.replace(backup)
            backups[destination] = backup
        stage_figure.replace(figure_dir)
        committed.append(figure_dir)
        stage_metric.replace(metric_dir)
        committed.append(metric_dir)
    except Exception:
        for destination in reversed(committed):
            if destination.exists():
                shutil.rmtree(destination)
        for destination, backup in backups.items():
            if backup.exists():
                backup.replace(destination)
        raise
    for backup in backups.values():
        shutil.rmtree(backup)


def run_analysis(config: AnalysisRunConfig) -> dict[str, Any]:
    """执行完整严格聚合；所有最终文件由 staging 目录原子提交。"""

    if config.bootstrap_replicates < 20:
        raise ValueError("bootstrap_replicates 至少为20；正式运行必须为2000")
    if config.metric_dir.resolve() == config.figure_dir.resolve():
        raise ValueError("metric_dir 与 figure_dir 必须独立")
    existing = [path for path in (config.metric_dir, config.figure_dir) if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError("输出已存在，默认拒绝覆盖：" + ", ".join(map(str, existing)))

    predictions, source_manifest = discover_predictions(config.prediction_dir)
    coverage, issues = coverage_audit(predictions, allow_partial=config.allow_partial)
    fold_metrics, oof_metrics = metric_tables(predictions)
    core_predictions = predictions.loc[core_bootstrap_mask(predictions)].copy()
    _, core_oof_metrics = metric_tables(core_predictions)
    bootstrap = bootstrap_table(
        core_predictions,
        core_oof_metrics,
        config.bootstrap_replicates,
        config.seed,
    )
    paired = build_paired_predictions(predictions)
    core_paired = paired.loc[core_paired_bootstrap_mask(paired)].copy()
    paired_fold, paired_ci = paired_difference_tables(
        core_paired, config.bootstrap_replicates, config.seed
    )

    common_parent = Path(
        os.path.commonpath([config.metric_dir.parent.resolve(), config.figure_dir.parent.resolve()])
    )
    common_parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".osra-aggregate-", dir=common_parent))
    stage_metric = stage_root / "metrics"
    stage_figure = stage_root / "figures"
    stage_metric.mkdir()
    stage_figure.mkdir()
    try:
        tables = {
            "prediction_file_manifest.csv": source_manifest,
            "coverage.csv": coverage,
            "input_issues.csv": issues,
            "fold_metrics.csv": fold_metrics,
            "oof_metrics.csv": oof_metrics,
            "bootstrap_ci.csv": bootstrap,
            "paired_differences_fold.csv": paired_fold,
            "paired_differences_bootstrap_ci.csv": paired_ci,
        }
        for name, frame in tables.items():
            atomic_csv(stage_metric / name, frame)
        figure_manifest = save_figures(
            predictions,
            oof_metrics,
            bootstrap,
            paired_ci,
            stage_figure,
            allow_partial=config.allow_partial,
        )
        # staging 绝对路径不得写入最终 manifest。
        figure_manifest["path"] = [
            _portable_path(config.figure_dir / filename) for filename in figure_manifest["figure"]
        ]
        atomic_csv(stage_metric / "figure_manifest.csv", figure_manifest)
        scope = bootstrap_scope_summary(predictions)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if issues.empty else "partial_with_registered_issues",
            "prediction_dir": _portable_path(config.prediction_dir),
            "metric_dir": _portable_path(config.metric_dir),
            "figure_dir": _portable_path(config.figure_dir),
            "analysis_implementation_sha256": analysis_implementation_sha256(),
            "seed": config.seed,
            "bootstrap_replicates": config.bootstrap_replicates,
            "bootstrap_scope": scope,
            "bootstrap_ci_is_conditional_on_fitted_models": True,
            "bootstrap_does_not_cover_training_randomness": True,
            "prediction_files": len(source_manifest),
            "prediction_rows": len(predictions),
            "patients": predictions["patient_id"].nunique(),
            "folds": sorted(map(int, predictions["fold"].unique())),
            "fold_metric_rows": len(fold_metrics),
            "oof_metric_rows": len(oof_metrics),
            "bootstrap_ci_rows": len(bootstrap),
            "paired_fold_rows": len(paired_fold),
            "paired_bootstrap_ci_rows": len(paired_ci),
            "figures": len(figure_manifest),
            "registered_issues": len(issues),
            "near_constant_definition": (
                "fold-centered standardized Var(pred) <= max(1e-10, 0.01*Var(target))"
            ),
            "transition_comparator_caveat_cn": (
                "observed delta 看见终点 MRI，transition predicted delta 只见当前 prefix；"
                "它们用于瓶颈定位，不是 information-matched baseline。"
            ),
            "output_commit": "staged directories renamed on same filesystem; default refuses overwrite",
        }
        atomic_json(stage_metric / "aggregation_summary.json", _json_safe(summary))
        _commit_staged_directories(
            stage_metric,
            stage_figure,
            config.metric_dir,
            config.figure_dir,
            overwrite=config.overwrite,
        )
        return summary
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _synthetic_prediction_frame(seed: int = 7) -> pd.DataFrame:
    """构造仅供临时 self-test 的完整小矩阵，不写入实验目录。"""

    rng = np.random.default_rng(seed)
    patients = [(f"SYN-{fold}-{index:02d}", fold, index) for fold in range(5) for index in range(6)]
    rows: list[dict[str, Any]] = []
    feature_dims = {
        "online_projected": 192,
        "online_preprojector": 192,
        "online_global_pool": 128,
        "online_roi_mean": 128,
        "ema_projected": 192,
        "ema_preprojector": 192,
        "ema_global_pool": 128,
        "ema_roi_mean": 128,
        "transition_predicted_delta": 192,
        "mask_geometry": 9,
        "raw_roi_intensity": 14,
        "current_radiomics": 1,
    }
    role_quality = {
        "online_projected": 0.45,
        "online_preprojector": 0.42,
        "online_global_pool": 0.55,
        "online_roi_mean": 0.30,
        "ema_projected": 0.48,
        "ema_preprojector": 0.45,
        "ema_global_pool": 0.58,
        "ema_roi_mean": 0.34,
        "transition_predicted_delta": 0.80,
        "mask_geometry": 0.35,
        "raw_roi_intensity": 0.65,
        "current_radiomics": 0.25,
    }

    def base_row(
        patient_id: str,
        fold: int,
        task: str,
        model: str,
        stream: str,
        representation: str,
        variant: str,
        target: str,
        timepoint: str,
        transition: str,
        truth: float,
        prediction: float,
        b0: float,
    ) -> dict[str, Any]:
        return {
            "patient_id": patient_id,
            "fold": fold,
            "split": "test",
            "task_type": task,
            "model": model,
            "run_name": f"{model}_final" if model in MODEL_ORDER else "baseline",
            "encoder_stream": stream,
            "representation": representation,
            "input_variant": variant,
            "target_name": target,
            "timepoint": timepoint,
            "transition": transition,
            "probe_type": "single_output_ridge",
            "feature_dim": feature_dims[representation],
            "y_true_natural": truth,
            "y_pred_natural": prediction,
            "y_true_standardized": truth,
            "y_pred_standardized": prediction,
            "b0_prediction_natural": b0,
            "b0_prediction_standardized": b0,
            "zero_change_prediction_natural": 0.0 if task == "change" else "",
            "zero_change_prediction_standardized": 0.0 if task == "change" else "",
            "selected_alpha": 1.0,
            "source_feature_file": f"/synthetic/{model}/fold_{fold}.npz",
            "source_feature_sha256": "a" * 64,
            "source_checkpoint": f"/synthetic/{model}/best.pt",
            "source_checkpoint_sha256": "b" * 64,
        }

    target_scale = {"ftv": 1.0, "ld": 0.7, "sphericity": 0.4, "bpe": 0.8}
    representations = (
        "online_projected",
        "online_preprojector",
        "online_global_pool",
        "online_roi_mean",
        "ema_projected",
        "ema_preprojector",
        "ema_global_pool",
        "ema_roi_mean",
        "mask_geometry",
        "raw_roi_intensity",
    )
    change_variants = (
        "current_only",
        "observed_pair",
        "observed_difference",
        "observed_combined",
    )
    for patient_id, fold, index in patients:
        subject = (index - 2.5) / 2.0 + 0.05 * fold
        for target in ALL_TARGETS:
            scale = target_scale[target]
            for time_index, timepoint in enumerate(TIMEPOINTS):
                truth = scale * (subject - 0.18 * time_index) + 0.03 * rng.normal()
                b0 = -0.18 * time_index * scale
                for model_index, model in enumerate(MODEL_ORDER):
                    for representation in representations:
                        noise = role_quality[representation] - 0.03 * model_index
                        prediction = truth + noise * rng.normal()
                        if representation.startswith("online_"):
                            stream = "online"
                        elif representation.startswith("ema_"):
                            stream = "ema_target"
                        elif representation == "mask_geometry":
                            stream = "mask_baseline"
                        else:
                            stream = "image_baseline"
                        rows.append(
                            base_row(
                                patient_id,
                                fold,
                                "static",
                                model,
                                stream,
                                representation,
                                "current",
                                target,
                                timepoint,
                                "",
                                truth,
                                prediction,
                                b0,
                            )
                        )
            for transition_index, transition in enumerate(TRANSITIONS):
                truth = scale * (-0.45 + 0.12 * subject + 0.08 * transition_index) + 0.03 * rng.normal()
                b0 = scale * (-0.45 + 0.08 * transition_index)
                for model_index, model in enumerate(MODEL_ORDER):
                    for representation in representations:
                        if representation.startswith("online_"):
                            stream = "online"
                        elif representation.startswith("ema_"):
                            stream = "ema_target"
                        elif representation == "mask_geometry":
                            stream = "mask_baseline"
                        else:
                            stream = "image_baseline"
                        for variant in change_variants:
                            variant_penalty = {
                                "current_only": 0.20,
                                "observed_pair": 0.03,
                                "observed_difference": 0.0,
                                "observed_combined": 0.02,
                            }[variant]
                            prediction = truth + (
                                role_quality[representation] + variant_penalty - 0.03 * model_index
                            ) * rng.normal()
                            rows.append(
                                base_row(
                                    patient_id,
                                    fold,
                                    "change",
                                    model,
                                    stream,
                                    representation,
                                    variant,
                                    target,
                                    "",
                                    transition,
                                    truth,
                                    prediction,
                                    b0,
                                )
                            )
                    transition_prediction = truth + role_quality["transition_predicted_delta"] * rng.normal()
                    rows.append(
                        base_row(
                            patient_id,
                            fold,
                            "change",
                            model,
                            "transition",
                            "transition_predicted_delta",
                            "predicted_next_delta",
                            target,
                            "",
                            transition,
                            truth,
                            transition_prediction,
                            b0,
                        )
                    )
                    b1_prediction = truth + role_quality["current_radiomics"] * rng.normal()
                    rows.append(
                        base_row(
                            patient_id,
                            fold,
                            "change",
                            model,
                            "table_baseline",
                            "current_radiomics",
                            "current_only",
                            target,
                            "",
                            transition,
                            truth,
                            b1_prediction,
                            b0,
                        )
                    )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def run_self_test() -> dict[str, Any]:
    """在系统临时目录运行端到端小矩阵，确认 schema、CI、图和拒绝覆盖。"""

    with tempfile.TemporaryDirectory(prefix="osra-analysis-selftest-") as name:
        root = Path(name)
        prediction_dir = root / "predictions"
        prediction_dir.mkdir()
        frame = _synthetic_prediction_frame()
        frame.to_csv(prediction_dir / "synthetic_test_predictions.csv", index=False)
        config = AnalysisRunConfig(
            prediction_dir=prediction_dir,
            metric_dir=root / "metrics" / "smoke",
            figure_dir=root / "figures" / "smoke",
            bootstrap_replicates=25,
            seed=123,
            overwrite=False,
            allow_partial=False,
        )
        summary = run_analysis(config)
        if summary["status"] != "complete" or summary["figures"] != 12:
            raise AssertionError("synthetic analysis summary 不完整")
        if len(list(config.figure_dir.glob("*.png"))) != 12:
            raise AssertionError("synthetic analysis 未生成12张图")
        expected_tables = {
            "fold_metrics.csv",
            "oof_metrics.csv",
            "bootstrap_ci.csv",
            "paired_differences_fold.csv",
            "paired_differences_bootstrap_ci.csv",
            "aggregation_summary.json",
            "figure_manifest.csv",
        }
        if not expected_tables.issubset({path.name for path in config.metric_dir.iterdir()}):
            raise AssertionError("synthetic analysis 缺聚合表")
        refused = False
        try:
            run_analysis(config)
        except FileExistsError:
            refused = True
        if not refused:
            raise AssertionError("聚合器未默认拒绝覆盖")
        return {
            "status": "self-test passed",
            "synthetic_rows": len(frame),
            "figures_verified": 12,
            "default_refuse_overwrite_verified": True,
            "temporary_outputs_removed": True,
            "bootstrap_replicates_for_smoke_only": 25,
        }
