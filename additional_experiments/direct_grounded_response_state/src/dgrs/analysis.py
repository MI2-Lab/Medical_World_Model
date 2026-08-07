"""Direct Grounded Response State 的严格 OOF 聚合、统计门槛与绘图。

本模块只消费已经保存的 outer-test prediction 和训练 history。它不会重拟合
probe/readout，也不会用 test 数据选择 lambda、checkpoint 或超参数。所有置信
区间都以患者为抽样单位，并在五个 outer fold 内分层重采样；因此区间只条件于
当前训练 seed 和已经拟合的模型，不覆盖重新训练的不确定性。
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
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCHEMA_VERSION = 1
MODEL_ORDER = ("G0", "G1", "G2", "G3", "G4")
TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")
DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
TARGETS = ("ftv", "ld", "sphericity")
PRIMARY_COMPARISONS = (("G3", "G1"), ("G4", "G2"))
PROBE_KEY = (
    "patient_id",
    "fold",
    "model",
    "task",
    "target",
    "timepoint",
    "transition",
    "representation",
)
PCR_KEY = ("patient_id", "fold", "model", "decision_point")
REGRESSION_METRICS = (
    "spearman",
    "pearson",
    "r2",
    "mae",
    "rmse",
    "rmse_gain_over_b0",
    "prediction_target_variance_ratio",
)
CLASSIFICATION_METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
)
SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_VALUE = re.compile(r"^[0-9a-fA-F]{64}$")
FORMAL_HISTORY_KEYS = frozenset(
    (model, fold) for model in ("G1", "G2", "G3", "G4") for fold in range(5)
)
PARTIAL_OUTPUT_PREFIXES = ("dev_", "partial_", "selftest_")


class AnalysisInputError(RuntimeError):
    """输入缺失、歧义或违反预注册契约。"""


@dataclass(frozen=True)
class AnalysisConfig:
    """正式聚合配置。"""

    prediction_root: Path = EXPERIMENT_ROOT / "predictions"
    history_root: Path = EXPERIMENT_ROOT / "metrics" / "training"
    checkpoint_root: Path = EXPERIMENT_ROOT / "checkpoints"
    geometry_control_path: Path = (
        EXPERIMENT_ROOT.parent
        / "observed_state_radiomics_audit"
        / "metrics"
        / "final_analysis"
        / "oof_metrics.csv"
    )
    metric_dir: Path = EXPERIMENT_ROOT / "metrics" / "final"
    figure_dir: Path = EXPERIMENT_ROOT / "figures" / "final"
    bootstrap_replicates: int = 2000
    seed: int = 20260807
    overwrite: bool = False
    allow_partial: bool = False
    expected_probe_patients: int | None = 375
    expected_pcr_patients: int | None = 808


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *map(str, parts)], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        # 保留可区分的 provenance token，但不公开用户名、挂载点或绝对目录。
        parent_token = hashlib.sha256(str(resolved.parent).encode("utf-8")).hexdigest()[:12]
        return f"<external:{parent_token}>/{resolved.name}"


def _strict_boolean(series: pd.Series, label: str) -> pd.Series:
    """把 CSV 中常见 bool 表示严格规范化；拒绝含糊 truthiness。"""

    mapping: dict[Any, bool] = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "TRUE": True,
        "FALSE": False,
    }
    normalized = series.map(mapping)
    if normalized.isna().any():
        examples = series.loc[normalized.isna()].astype(str).head(3).tolist()
        raise AnalysisInputError(f"{label} 含非法布尔值: {examples}")
    return normalized.astype(bool)


def _require_sha256(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        values = frame[column].astype(str).str.strip()
        if not values.map(lambda value: bool(SHA256_VALUE.fullmatch(value))).all():
            raise AnalysisInputError(f"{label} 的 {column} 必须逐行为 64 位 SHA-256")


def _is_partial_output(path: Path) -> bool:
    return path.name.lower().startswith(PARTIAL_OUTPUT_PREFIXES)


def _rename_aliases(frame: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    output = frame.copy()
    lower = {str(column).strip().lower(): column for column in output.columns}
    rename: dict[Any, str] = {}
    for canonical, candidates in aliases.items():
        if canonical in output.columns:
            continue
        for candidate in candidates:
            original = lower.get(candidate.lower())
            if original is not None:
                rename[original] = canonical
                break
    return output.rename(columns=rename)


def _normalise_model(value: Any) -> str:
    text = str(value).strip().upper().replace("MODEL_", "")
    if text.startswith("G") and text[1:].isdigit():
        text = f"G{int(text[1:])}"
    return text


def _normalise_transition(value: Any) -> str:
    text = str(value).strip().upper().replace(" ", "")
    text = text.replace("->", "→").replace("–", "-").replace("—", "-")
    aliases = {
        "T0-T1": "T0→T1",
        "T1-T2": "T1→T2",
        "T2-T3": "T2→T3",
        "T0→T1": "T0→T1",
        "T1→T2": "T1→T2",
        "T2→T3": "T2→T3",
    }
    return aliases.get(text, text)


def _normalise_decision(value: Any) -> str:
    text = str(value).strip().upper().replace(" ", "")
    return text.replace("→", "-").replace("–", "-").replace("—", "-")


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AnalysisInputError(f"{label} 的 {column} 含 NaN/Inf 或非数值")


def normalise_probe_predictions(
    frame: pd.DataFrame,
    source: Path | None = None,
    *,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """兼容最小约定 schema，并规范成严格内部 schema。"""

    aliases = {
        "task": ("task_type",),
        "target": ("target_name",),
        "y_true": ("y_true_natural", "target_value"),
        "y_pred": ("y_pred_natural", "prediction"),
        "y_true_standardized": ("target_standardized", "y_true_std"),
        "y_pred_standardized": ("prediction_standardized", "y_pred_std"),
        "b0_prediction": ("b0_prediction_natural", "train_mean_prediction"),
        "b0_prediction_standardized": ("b0_prediction_std",),
        "selected_alpha": ("alpha",),
        "target_transform_hash": ("transform_hash",),
        "source_feature_sha256": ("feature_sha256", "source_hash"),
    }
    output = _rename_aliases(frame, aliases)
    required = {
        "patient_id",
        "fold",
        "split",
        "model",
        "timepoint",
        "transition",
        "representation",
        "target",
        "y_true",
        "y_pred",
    }
    missing = sorted(required.difference(output.columns))
    if missing:
        raise AnalysisInputError(f"probe CSV 缺列 {missing}: {source or '<dataframe>'}")
    if not allow_partial:
        formal_required = {
            "y_true_standardized",
            "y_pred_standardized",
            "b0_prediction",
            "b0_prediction_standardized",
            "selected_alpha",
            "target_transform",
            "source_feature_sha256",
            "feature_extractor_sha256",
            "source_checkpoint_sha256",
            "fold_manifest_sha256",
            "static_transform_sha256",
            "change_transform_sha256",
            "raw_targets_sha256",
            "feature_dim",
            "input_variant",
            "test_used_for_scaler",
            "test_used_for_alpha_selection",
            "test_predict_call_count",
        }
        formal_missing = sorted(formal_required.difference(output.columns))
        if formal_missing:
            raise AnalysisInputError(
                f"正式 probe CSV 缺 provenance/test-blind 字段 {formal_missing}: "
                f"{source or '<dataframe>'}"
            )
    if "task" not in output:
        timepoint = output["timepoint"].fillna("").astype(str).str.strip()
        output["task"] = np.where(timepoint.ne(""), "static", "change")
    defaults: dict[str, Any] = {
        "y_true_standardized": output["y_true"],
        "y_pred_standardized": output["y_pred"],
        "b0_prediction": np.nan,
        "b0_prediction_standardized": np.nan,
        "selected_alpha": np.nan,
        "target_transform_hash": "",
        "source_feature_sha256": "",
    }
    for column, value in defaults.items():
        if column not in output:
            output[column] = value
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    output["split"] = output["split"].astype(str).str.strip().str.lower()
    output["model"] = output["model"].map(_normalise_model)
    output["task"] = output["task"].astype(str).str.strip().str.lower()
    output["target"] = output["target"].astype(str).str.strip().str.lower()
    output["timepoint"] = output["timepoint"].fillna("").astype(str).str.strip().str.upper()
    output["transition"] = output["transition"].fillna("").map(_normalise_transition)
    output["representation"] = output["representation"].astype(str).str.strip()
    for column in (
        "y_true",
        "y_pred",
        "y_true_standardized",
        "y_pred_standardized",
        "b0_prediction",
        "b0_prediction_standardized",
        "selected_alpha",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output.empty:
        raise AnalysisInputError(f"probe CSV 为空: {source or '<dataframe>'}")
    if not output["split"].eq("test").all():
        raise AnalysisInputError("聚合器只接受 outer-test probe prediction")
    if not output["model"].isin(MODEL_ORDER).all():
        raise AnalysisInputError(f"probe model 非法: {sorted(output['model'].unique())}")
    if set(output["fold"]).difference(range(5)):
        raise AnalysisInputError("probe fold 必须在 0–4")
    if not output["task"].isin({"static", "change"}).all():
        raise AnalysisInputError("probe task 必须为 static/change")
    if not output["target"].isin(TARGETS).all():
        raise AnalysisInputError(f"probe target 必须为 {TARGETS}")
    static = output["task"].eq("static")
    if not output.loc[static, "timepoint"].isin(TIMEPOINTS).all():
        raise AnalysisInputError("static probe timepoint 非法")
    if not output.loc[~static, "transition"].isin(TRANSITIONS).all():
        raise AnalysisInputError("change probe transition 非法")
    _finite_numeric(
        output,
        ("y_true", "y_pred", "y_true_standardized", "y_pred_standardized"),
        "probe",
    )
    if not allow_partial:
        _finite_numeric(
            output,
            (
                "b0_prediction",
                "b0_prediction_standardized",
                "selected_alpha",
                "feature_dim",
                "test_predict_call_count",
            ),
            "formal probe",
        )
        if not output["selected_alpha"].gt(0).all():
            raise AnalysisInputError("正式 probe selected_alpha 必须为有限正数")
        if not pd.to_numeric(output["feature_dim"]).eq(192).all():
            raise AnalysisInputError("正式 probe feature_dim 必须为 192")
        if not output["representation"].eq("response_state").all():
            raise AnalysisInputError("正式 probe 只接受 response_state representation")
        expected_variant = np.where(static, "current", "observed_difference")
        if not np.array_equal(output["input_variant"].astype(str).to_numpy(), expected_variant):
            raise AnalysisInputError("正式 probe input_variant 与 static/change contract 不一致")
        for column in ("target_transform",):
            if output[column].astype(str).str.strip().eq("").any():
                raise AnalysisInputError(f"正式 probe {column} 不能为空")
        _require_sha256(
            output,
            (
                "source_feature_sha256",
                "feature_extractor_sha256",
                "source_checkpoint_sha256",
                "fold_manifest_sha256",
                "static_transform_sha256",
                "change_transform_sha256",
                "raw_targets_sha256",
            ),
            "formal probe",
        )
        for column in ("test_used_for_scaler", "test_used_for_alpha_selection"):
            output[column] = _strict_boolean(output[column], f"formal probe {column}")
            if output[column].any():
                raise AnalysisInputError(f"正式 probe 禁止 {column}=true")
        calls = pd.to_numeric(output["test_predict_call_count"], errors="coerce")
        if not calls.eq(1).all():
            raise AnalysisInputError("正式 probe 每个 outer-test row 必须声明 test_predict_call_count=1")
    # B0 为正式必需信息；开发输入允许缺失，metric 会明确为 NaN。
    present_b0 = output["b0_prediction"].notna()
    if present_b0.any() and not np.isfinite(output.loc[present_b0, "b0_prediction"]).all():
        raise AnalysisInputError("probe b0_prediction 非有限")
    if output["patient_id"].eq("").any():
        raise AnalysisInputError("probe patient_id 为空")
    if output.duplicated(list(PROBE_KEY)).any():
        duplicate = output.loc[output.duplicated(list(PROBE_KEY), False), list(PROBE_KEY)].head()
        raise AnalysisInputError(f"probe prediction 唯一键重复:\n{duplicate.to_string(index=False)}")
    if source is not None:
        output["source_prediction_file"] = _portable(source)
        output["source_prediction_sha256"] = file_sha256(source)
    return output.sort_values(list(PROBE_KEY), kind="mergesort").reset_index(drop=True)


def normalise_pcr_predictions(
    frame: pd.DataFrame,
    source: Path | None = None,
    *,
    allow_partial: bool = False,
) -> pd.DataFrame:
    aliases = {
        "probability": ("y_prob", "y_score", "prob"),
        "predicted_label": ("y_pred", "prediction"),
        "threshold": ("selected_threshold",),
        "C": ("selected_c", "c"),
        "penalty": ("selected_penalty",),
        "source_feature_sha256": ("feature_sha256", "source_hash"),
    }
    output = _rename_aliases(frame, aliases)
    required = {
        "patient_id",
        "fold",
        "split",
        "model",
        "decision_point",
        "y_true",
        "probability",
        "predicted_label",
        "threshold",
    }
    missing = sorted(required.difference(output.columns))
    if missing:
        raise AnalysisInputError(f"pCR CSV 缺列 {missing}: {source or '<dataframe>'}")
    if not allow_partial:
        formal_required = {
            "C",
            "penalty",
            "readout",
            "class_weight",
            "feature_schema",
            "feature_schema_sha256",
            "feature_dim",
            "val_auroc",
            "val_auprc",
            "source_feature_sha256",
            "feature_extractor_sha256",
            "source_checkpoint_sha256",
            "fold_manifest_sha256",
            "test_used_for_scaler",
            "test_used_for_hyperparameter_selection",
            "test_used_for_threshold_selection",
            "test_predict_proba_call_count",
        }
        formal_missing = sorted(formal_required.difference(output.columns))
        if formal_missing:
            raise AnalysisInputError(
                f"正式 pCR CSV 缺 provenance/test-blind 字段 {formal_missing}: "
                f"{source or '<dataframe>'}"
            )
    for column, default in (
        ("C", np.nan),
        ("penalty", ""),
        ("source_feature_sha256", ""),
    ):
        if column not in output:
            output[column] = default
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    output["split"] = output["split"].astype(str).str.strip().str.lower()
    output["model"] = output["model"].map(_normalise_model)
    output["decision_point"] = output["decision_point"].map(_normalise_decision)
    for column in ("y_true", "probability", "predicted_label", "threshold", "C"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output.empty:
        raise AnalysisInputError(f"pCR CSV 为空: {source or '<dataframe>'}")
    if not output["split"].eq("test").all():
        raise AnalysisInputError("聚合器只接受 outer-test pCR prediction")
    if not output["model"].isin(MODEL_ORDER).all():
        raise AnalysisInputError("pCR model 非法")
    if set(output["fold"]).difference(range(5)):
        raise AnalysisInputError("pCR fold 必须在 0–4")
    if not output["decision_point"].isin(DECISION_POINTS).all():
        raise AnalysisInputError("pCR decision_point 非法")
    _finite_numeric(output, ("y_true", "probability", "predicted_label", "threshold"), "pCR")
    if not output["y_true"].isin({0, 1}).all():
        raise AnalysisInputError("pCR y_true 必须为 0/1")
    if not output["predicted_label"].isin({0, 1}).all():
        raise AnalysisInputError("pCR predicted_label 必须为 0/1")
    if not output["probability"].between(0, 1).all():
        raise AnalysisInputError("pCR probability 必须在 [0,1]")
    if not output["threshold"].between(0, 1).all():
        raise AnalysisInputError("pCR threshold 必须在 [0,1]")
    expected_label = output["probability"].ge(output["threshold"]).astype(int)
    if not output["predicted_label"].eq(expected_label).all():
        raise AnalysisInputError("pCR predicted_label 与 probability>=threshold 不一致")
    if not allow_partial:
        _finite_numeric(
            output,
            ("C", "feature_dim", "val_auroc", "val_auprc", "test_predict_proba_call_count"),
            "formal pCR",
        )
        if not output["C"].gt(0).all() or not output["penalty"].astype(str).isin({"l1", "l2"}).all():
            raise AnalysisInputError("正式 pCR penalty/C contract 非法")
        expected_dims = output["decision_point"].map({"T0": 192, "T0-T1": 576, "T0-T2": 1152})
        if not pd.to_numeric(output["feature_dim"]).eq(expected_dims).all():
            raise AnalysisInputError("正式 pCR feature_dim 与 decision point 不一致")
        if not output["readout"].astype(str).eq("class-balanced LogisticRegression").all():
            raise AnalysisInputError("正式 pCR readout 必须为 class-balanced LogisticRegression")
        if not output["class_weight"].astype(str).eq("balanced").all():
            raise AnalysisInputError("正式 pCR class_weight 必须为 balanced")
        if output["feature_schema"].astype(str).str.strip().eq("").any():
            raise AnalysisInputError("正式 pCR feature_schema 不能为空")
        _require_sha256(
            output,
            (
                "feature_schema_sha256",
                "source_feature_sha256",
                "feature_extractor_sha256",
                "source_checkpoint_sha256",
                "fold_manifest_sha256",
            ),
            "formal pCR",
        )
        for column in (
            "test_used_for_scaler",
            "test_used_for_hyperparameter_selection",
            "test_used_for_threshold_selection",
        ):
            output[column] = _strict_boolean(output[column], f"formal pCR {column}")
            if output[column].any():
                raise AnalysisInputError(f"正式 pCR 禁止 {column}=true")
        calls = pd.to_numeric(output["test_predict_proba_call_count"], errors="coerce")
        if not calls.eq(1).all():
            raise AnalysisInputError(
                "正式 pCR 每个 outer-test row 必须声明 test_predict_proba_call_count=1"
            )
    if output.duplicated(list(PCR_KEY)).any():
        duplicate = output.loc[output.duplicated(list(PCR_KEY), False), list(PCR_KEY)].head()
        raise AnalysisInputError(f"pCR prediction 唯一键重复:\n{duplicate.to_string(index=False)}")
    if source is not None:
        output["source_prediction_file"] = _portable(source)
        output["source_prediction_sha256"] = file_sha256(source)
    return output.sort_values(list(PCR_KEY), kind="mergesort").reset_index(drop=True)


def _prediction_paths(root: Path, kind: str) -> list[Path]:
    subdir = "representation_probes" if kind == "probe" else "pcr_readouts"
    base = root / subdir
    paths = sorted(base.glob("*/fold_*/test_predictions.csv"))
    if not paths and base.is_dir():
        paths = sorted(base.rglob("test_predictions.csv"))
    return [path for path in paths if path.is_file()]


def discover_predictions(
    root: Path, *, allow_partial: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """只发现锁定目录中的 test_predictions.csv，避免误聚合 pilot/smoke。"""

    probe_paths = _prediction_paths(root, "probe")
    pcr_paths = _prediction_paths(root, "pcr")
    if not probe_paths:
        raise AnalysisInputError(f"未发现 representation probe prediction: {root}")
    if not pcr_paths:
        raise AnalysisInputError(f"未发现 pCR prediction: {root}")
    probe_frames: list[pd.DataFrame] = []
    pcr_frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    for kind, paths, destination, normalizer in (
        ("representation_probe", probe_paths, probe_frames, normalise_probe_predictions),
        ("pcr_readout", pcr_paths, pcr_frames, normalise_pcr_predictions),
    ):
        for path in paths:
            frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
            normalized = normalizer(frame, path, allow_partial=allow_partial)
            destination.append(normalized)
            manifest.append(
                {
                    "kind": kind,
                    "path": _portable(path),
                    "sha256": file_sha256(path),
                    "rows": len(normalized),
                    "patients": normalized["patient_id"].nunique(),
                    "models": ",".join(sorted(normalized["model"].unique())),
                    "folds": ",".join(map(str, sorted(normalized["fold"].unique()))),
                }
            )
    probe = normalise_probe_predictions(
        pd.concat(probe_frames, ignore_index=True), allow_partial=allow_partial
    )
    pcr = normalise_pcr_predictions(
        pd.concat(pcr_frames, ignore_index=True), allow_partial=allow_partial
    )
    return probe, pcr, pd.DataFrame(manifest)


def _correlation(target: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    if len(target) < 3 or np.std(target) <= 0 or np.std(prediction) <= 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = spearmanr(target, prediction).statistic if kind == "spearman" else pearsonr(target, prediction).statistic
    return float(value) if math.isfinite(float(value)) else math.nan


def regression_metrics(group: pd.DataFrame) -> dict[str, Any]:
    target = group["y_true"].to_numpy(dtype=float)
    prediction = group["y_pred"].to_numpy(dtype=float)
    n = len(group)
    target_var = float(np.var(target, ddof=1)) if n > 1 else math.nan
    pred_var = float(np.var(prediction, ddof=1)) if n > 1 else math.nan
    rmse = math.sqrt(float(mean_squared_error(target, prediction)))
    b0_values = group["b0_prediction"].to_numpy(dtype=float)
    if np.isfinite(b0_values).all():
        b0_rmse = math.sqrt(float(mean_squared_error(target, b0_values)))
        b0_gain = (b0_rmse - rmse) / max(b0_rmse, 1e-12)
    else:
        b0_rmse = math.nan
        b0_gain = math.nan
    return {
        "n": n,
        "n_patients": group["patient_id"].nunique(),
        "n_folds": group["fold"].nunique(),
        "spearman": _correlation(target, prediction, "spearman"),
        "pearson": _correlation(target, prediction, "pearson"),
        "r2": float(r2_score(target, prediction)) if n > 1 and target_var > 0 else math.nan,
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": rmse,
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": b0_gain,
        "target_variance": target_var,
        "prediction_variance": pred_var,
        "prediction_target_variance_ratio": pred_var / max(target_var, 1e-12) if n > 1 else math.nan,
    }


def classification_metrics(group: pd.DataFrame) -> dict[str, Any]:
    target = group["y_true"].to_numpy(dtype=int)
    probability = group["probability"].to_numpy(dtype=float)
    prediction = group["predicted_label"].to_numpy(dtype=int)
    classes = np.unique(target)
    return {
        "n": len(group),
        "n_patients": group["patient_id"].nunique(),
        "n_folds": group["fold"].nunique(),
        "prevalence": float(np.mean(target)),
        "auroc": float(roc_auc_score(target, probability)) if len(classes) == 2 else math.nan,
        "auprc": float(average_precision_score(target, probability)) if len(classes) == 2 else math.nan,
        "accuracy": float(accuracy_score(target, prediction)),
        "sensitivity": float(recall_score(target, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(target, prediction, pos_label=0, zero_division=0)),
    }


def metric_tables(probe: pd.DataFrame, pcr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    probe_group = ["model", "task", "target", "timepoint", "transition", "representation"]
    pcr_group = ["model", "decision_point"]
    probe_fold: list[dict[str, Any]] = []
    probe_oof: list[dict[str, Any]] = []
    pcr_fold: list[dict[str, Any]] = []
    pcr_oof: list[dict[str, Any]] = []
    for keys, group in probe.groupby([*probe_group, "fold"], dropna=False, sort=False):
        probe_fold.append({**dict(zip([*probe_group, "fold"], keys)), **regression_metrics(group)})
    for keys, group in probe.groupby(probe_group, dropna=False, sort=False):
        probe_oof.append({**dict(zip(probe_group, keys)), **regression_metrics(group)})
    for keys, group in pcr.groupby([*pcr_group, "fold"], dropna=False, sort=False):
        pcr_fold.append({**dict(zip([*pcr_group, "fold"], keys)), **classification_metrics(group)})
    for keys, group in pcr.groupby(pcr_group, dropna=False, sort=False):
        pcr_oof.append({**dict(zip(pcr_group, keys)), **classification_metrics(group)})
    return {
        "probe_fold": pd.DataFrame(probe_fold),
        "probe_oof": pd.DataFrame(probe_oof),
        "pcr_fold": pd.DataFrame(pcr_fold),
        "pcr_oof": pd.DataFrame(pcr_oof),
    }


def _bootstrap_indices(group: pd.DataFrame, replicates: int, seed: int) -> np.ndarray:
    if group.duplicated(["patient_id", "fold"]).any():
        raise AnalysisInputError("bootstrap cell 内 patient/fold 重复")
    rng = np.random.default_rng(seed)
    folds = group["fold"].to_numpy(dtype=int)
    parts = []
    for fold in sorted(np.unique(folds)):
        positions = np.flatnonzero(folds == fold)
        parts.append(rng.choice(positions, size=(replicates, len(positions)), replace=True))
    if not parts:
        raise AnalysisInputError("bootstrap cell 为空")
    return np.concatenate(parts, axis=1)


def _rowwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.square(x).sum(axis=1) * np.square(y).sum(axis=1))
    values = np.full(x.shape[0], np.nan)
    valid = denominator > 0
    values[valid] = (x[valid] * y[valid]).sum(axis=1) / denominator[valid]
    return values


def _regression_bootstrap_arrays(group: pd.DataFrame, indices: np.ndarray) -> dict[str, np.ndarray]:
    y = group["y_true"].to_numpy(dtype=float)[indices]
    p = group["y_pred"].to_numpy(dtype=float)[indices]
    error = p - y
    y_centered = y - y.mean(axis=1, keepdims=True)
    p_centered = p - p.mean(axis=1, keepdims=True)
    target_ss = np.square(y_centered).sum(axis=1)
    pred_ss = np.square(p_centered).sum(axis=1)
    rmse = np.sqrt(np.mean(np.square(error), axis=1))
    result = {
        "spearman": _rowwise_corr(rankdata(y, axis=1), rankdata(p, axis=1)),
        "pearson": _rowwise_corr(y, p),
        "r2": np.divide(
            -np.square(error).sum(axis=1),
            target_ss,
            out=np.full(len(indices), np.nan),
            where=target_ss > 0,
        )
        + 1.0,
        "mae": np.mean(np.abs(error), axis=1),
        "rmse": rmse,
        "prediction_target_variance_ratio": np.divide(
            pred_ss,
            target_ss,
            out=np.full(len(indices), np.nan),
            where=target_ss > 0,
        ),
        "rmse_gain_over_b0": np.full(len(indices), np.nan),
    }
    b0 = group["b0_prediction"].to_numpy(dtype=float)
    if np.isfinite(b0).all():
        b0 = b0[indices]
        b0_rmse = np.sqrt(np.mean(np.square(b0 - y), axis=1))
        result["rmse_gain_over_b0"] = (b0_rmse - rmse) / np.maximum(b0_rmse, 1e-12)
    return result


def _classification_bootstrap_arrays(group: pd.DataFrame, indices: np.ndarray) -> dict[str, np.ndarray]:
    target = group["y_true"].to_numpy(dtype=int)[indices]
    probability = group["probability"].to_numpy(dtype=float)[indices]
    prediction = group["predicted_label"].to_numpy(dtype=int)[indices]
    replicates = len(indices)
    result = {metric: np.full(replicates, np.nan) for metric in CLASSIFICATION_METRICS}
    for index in range(replicates):
        y = target[index]
        p = probability[index]
        label = prediction[index]
        if len(np.unique(y)) == 2:
            result["auroc"][index] = roc_auc_score(y, p)
            result["auprc"][index] = average_precision_score(y, p)
        result["accuracy"][index] = np.mean(y == label)
        positives = y == 1
        negatives = ~positives
        if positives.any():
            result["sensitivity"][index] = np.mean(label[positives] == 1)
        if negatives.any():
            result["specificity"][index] = np.mean(label[negatives] == 0)
    return result


def _ci_rows(
    keys: Mapping[str, Any], point: Mapping[str, Any], arrays: Mapping[str, np.ndarray], replicates: int
) -> list[dict[str, Any]]:
    rows = []
    for metric, values in arrays.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                **keys,
                "metric": metric,
                "estimate": point.get(metric, math.nan),
                "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else math.nan,
                "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else math.nan,
                "bootstrap_replicates": replicates,
                "finite_replicates": len(finite),
                "bootstrap_unit": "patient_within_outer_fold",
            }
        )
    return rows


def bootstrap_tables(
    probe: pd.DataFrame,
    pcr: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probe_group = ["model", "task", "target", "timepoint", "transition", "representation"]
    pcr_group = ["model", "decision_point"]
    probe_rows: list[dict[str, Any]] = []
    pcr_rows: list[dict[str, Any]] = []
    for keys, group in probe.groupby(probe_group, dropna=False, sort=False):
        key_map = dict(zip(probe_group, keys))
        point = regression_metrics(group)
        indices = _bootstrap_indices(group, replicates, _stable_seed(seed, "probe", *keys))
        arrays = _regression_bootstrap_arrays(group, indices)
        probe_rows.extend(_ci_rows(key_map, point, arrays, replicates))
    for keys, group in pcr.groupby(pcr_group, dropna=False, sort=False):
        key_map = dict(zip(pcr_group, keys))
        point = classification_metrics(group)
        indices = _bootstrap_indices(group, replicates, _stable_seed(seed, "pcr", *keys))
        arrays = _classification_bootstrap_arrays(group, indices)
        pcr_rows.extend(_ci_rows(key_map, point, arrays, replicates))
    return pd.DataFrame(probe_rows), pd.DataFrame(pcr_rows)


def _paired_merge(
    frame: pd.DataFrame,
    grounded: str,
    baseline: str,
    key: Sequence[str],
    values: Sequence[str],
) -> pd.DataFrame:
    left = frame.loc[frame["model"].eq(grounded), [*key, *values]].copy()
    right = frame.loc[frame["model"].eq(baseline), [*key, *values]].copy()
    paired = left.merge(right, on=list(key), how="inner", suffixes=("_grounded", "_baseline"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise AnalysisInputError(f"{grounded}/{baseline} 不是 exact-patient paired set")
    truth_left = paired["y_true_grounded"].to_numpy(dtype=float)
    truth_right = paired["y_true_baseline"].to_numpy(dtype=float)
    if not np.allclose(truth_left, truth_right, rtol=0, atol=1e-9):
        raise AnalysisInputError(f"{grounded}/{baseline} paired y_true 不一致")
    paired["y_true"] = truth_left
    paired["comparison"] = f"{grounded}-{baseline}"
    return paired


def _paired_regression_point(group: pd.DataFrame) -> dict[str, float]:
    synthetic_grounded = pd.DataFrame(
        {
            "patient_id": group["patient_id"],
            "fold": group["fold"],
            "y_true": group["y_true"],
            "y_pred": group["y_pred_grounded"],
            "b0_prediction": group.get("b0_prediction_grounded", np.nan),
        }
    )
    synthetic_baseline = synthetic_grounded.copy()
    synthetic_baseline["y_pred"] = group["y_pred_baseline"].to_numpy()
    synthetic_baseline["b0_prediction"] = group.get("b0_prediction_baseline", np.nan)
    g = regression_metrics(synthetic_grounded)
    b = regression_metrics(synthetic_baseline)
    return {metric: g[metric] - b[metric] for metric in REGRESSION_METRICS}


def _paired_regression_arrays(group: pd.DataFrame, indices: np.ndarray) -> dict[str, np.ndarray]:
    base = pd.DataFrame(
        {
            "y_true": group["y_true"],
            "y_pred": group["y_pred_baseline"],
            "b0_prediction": group.get("b0_prediction_baseline", np.nan),
        }
    )
    grounded = base.copy()
    grounded["y_pred"] = group["y_pred_grounded"].to_numpy()
    grounded["b0_prediction"] = group.get("b0_prediction_grounded", np.nan)
    b = _regression_bootstrap_arrays(base, indices)
    g = _regression_bootstrap_arrays(grounded, indices)
    return {metric: g[metric] - b[metric] for metric in REGRESSION_METRICS}


def _classification_from_arrays(target: np.ndarray, probability: np.ndarray, label: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "patient_id": np.arange(len(target)),
            "fold": 0,
            "y_true": target,
            "probability": probability,
            "predicted_label": label,
        }
    )
    return classification_metrics(frame)


def _paired_classification_point(group: pd.DataFrame) -> dict[str, float]:
    y = group["y_true"].to_numpy(dtype=int)
    g = _classification_from_arrays(
        y,
        group["probability_grounded"].to_numpy(dtype=float),
        group["predicted_label_grounded"].to_numpy(dtype=int),
    )
    b = _classification_from_arrays(
        y,
        group["probability_baseline"].to_numpy(dtype=float),
        group["predicted_label_baseline"].to_numpy(dtype=int),
    )
    return {metric: g[metric] - b[metric] for metric in CLASSIFICATION_METRICS}


def _paired_classification_arrays(group: pd.DataFrame, indices: np.ndarray) -> dict[str, np.ndarray]:
    target = group["y_true"].to_numpy(dtype=int)
    dummy = pd.DataFrame({"y_true": target})
    grounded = dummy.assign(
        probability=group["probability_grounded"].to_numpy(dtype=float),
        predicted_label=group["predicted_label_grounded"].to_numpy(dtype=int),
    )
    baseline = dummy.assign(
        probability=group["probability_baseline"].to_numpy(dtype=float),
        predicted_label=group["predicted_label_baseline"].to_numpy(dtype=int),
    )
    g = _classification_bootstrap_arrays(grounded, indices)
    b = _classification_bootstrap_arrays(baseline, indices)
    return {metric: g[metric] - b[metric] for metric in CLASSIFICATION_METRICS}


def paired_difference_tables(
    probe: pd.DataFrame, pcr: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G3−G1 与 G4−G2 的 exact-patient paired bootstrap。"""

    probe_cells = ["task", "target", "timepoint", "transition", "representation"]
    pcr_cells = ["decision_point"]
    fold_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    for grounded, baseline in PRIMARY_COMPARISONS:
        paired_probe = _paired_merge(
            probe,
            grounded,
            baseline,
            ["patient_id", "fold", *probe_cells],
            ["y_true", "y_pred", "b0_prediction"],
        )
        for keys, group in paired_probe.groupby(probe_cells, dropna=False, sort=False):
            cell = dict(zip(probe_cells, keys))
            for fold, fold_group in group.groupby("fold"):
                point = _paired_regression_point(fold_group)
                for metric, estimate in point.items():
                    fold_rows.append(
                        {"comparison": f"{grounded}-{baseline}", "kind": "probe", **cell, "fold": fold, "metric": metric, "estimate": estimate, "n_patients": len(fold_group)}
                    )
            point = _paired_regression_point(group)
            indices = _bootstrap_indices(group, replicates, _stable_seed(seed, grounded, baseline, "probe", *keys))
            arrays = _paired_regression_arrays(group, indices)
            for row in _ci_rows({"comparison": f"{grounded}-{baseline}", "kind": "probe", **cell}, point, arrays, replicates):
                row["n_patients"] = len(group)
                ci_rows.append(row)
        paired_pcr = _paired_merge(
            pcr,
            grounded,
            baseline,
            ["patient_id", "fold", *pcr_cells],
            ["y_true", "probability", "predicted_label"],
        )
        for keys, group in paired_pcr.groupby(pcr_cells, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            cell = dict(zip(pcr_cells, keys))
            for fold, fold_group in group.groupby("fold"):
                point = _paired_classification_point(fold_group)
                for metric, estimate in point.items():
                    fold_rows.append(
                        {"comparison": f"{grounded}-{baseline}", "kind": "pcr", **cell, "fold": fold, "metric": metric, "estimate": estimate, "n_patients": len(fold_group)}
                    )
            point = _paired_classification_point(group)
            indices = _bootstrap_indices(group, replicates, _stable_seed(seed, grounded, baseline, "pcr", *keys))
            arrays = _paired_classification_arrays(group, indices)
            for row in _ci_rows({"comparison": f"{grounded}-{baseline}", "kind": "pcr", **cell}, point, arrays, replicates):
                row["n_patients"] = len(group)
                ci_rows.append(row)
    return pd.DataFrame(fold_rows), pd.DataFrame(ci_rows)


def _macro_regression_bootstrap(
    paired: pd.DataFrame,
    cells: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, np.ndarray]]:
    groups = []
    patient_reference: pd.DataFrame | None = None
    for cell in cells:
        column = "timepoint" if "timepoint" in paired and cell in TIMEPOINTS else "transition"
        group = paired.loc[paired[column].eq(cell)].sort_values(["fold", "patient_id"]).reset_index(drop=True)
        patients = group[["patient_id", "fold"]]
        if patient_reference is None:
            patient_reference = patients
        elif not patients.equals(patient_reference):
            raise AnalysisInputError("macro paired bootstrap 各 cell 患者集合/顺序不一致")
        groups.append(group)
    if patient_reference is None or patient_reference.empty:
        raise AnalysisInputError("macro paired bootstrap 无数据")
    indices = _bootstrap_indices(groups[0], replicates, seed)
    metrics = ("spearman", "r2")
    arrays: dict[str, list[np.ndarray]] = {metric: [] for metric in metrics}
    points: dict[str, list[float]] = {metric: [] for metric in metrics}
    for group in groups:
        sampled = _paired_regression_arrays(group, indices)
        point = _paired_regression_point(group)
        for metric in metrics:
            arrays[metric].append(sampled[metric])
            points[metric].append(point[metric])
    return {
        metric: (float(np.nanmean(points[metric])), np.nanmean(np.vstack(arrays[metric]), axis=0))
        for metric in metrics
    }


def _macro_pcr_bootstrap(
    paired: pd.DataFrame, replicates: int, seed: int
) -> tuple[float, np.ndarray]:
    groups = []
    reference: pd.DataFrame | None = None
    for decision in ("T0-T1", "T0-T2"):
        group = paired.loc[paired["decision_point"].eq(decision)].sort_values(["fold", "patient_id"]).reset_index(drop=True)
        patients = group[["patient_id", "fold"]]
        if reference is None:
            reference = patients
        elif not patients.equals(reference):
            raise AnalysisInputError("pCR macro paired bootstrap 患者集合不一致")
        groups.append(group)
    if reference is None or reference.empty:
        raise AnalysisInputError("pCR macro paired bootstrap 无数据")
    indices = _bootstrap_indices(groups[0], replicates, seed)
    points = []
    arrays = []
    for group in groups:
        points.append(_paired_classification_point(group)["auroc"])
        arrays.append(_paired_classification_arrays(group, indices)["auroc"])
    return float(np.nanmean(points)), np.nanmean(np.vstack(arrays), axis=0)


def macro_paired_table(
    probe: pd.DataFrame, pcr: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grounded, baseline in PRIMARY_COMPARISONS:
        comparison = f"{grounded}-{baseline}"
        paired_probe = _paired_merge(
            probe,
            grounded,
            baseline,
            ["patient_id", "fold", "task", "target", "timepoint", "transition", "representation"],
            ["y_true", "y_pred", "b0_prediction"],
        )
        for scope, task, cells in (
            ("A_static_ftv", "static", TIMEPOINTS),
            ("B_change_ftv", "change", TRANSITIONS),
        ):
            part = paired_probe.loc[paired_probe["task"].eq(task) & paired_probe["target"].eq("ftv")]
            # 正式 primary representation 每 model 只能有一个；若命名重复则拒绝平均。
            representations = sorted(part["representation"].unique())
            if len(representations) != 1:
                raise AnalysisInputError(f"{comparison}/{scope} representation 不唯一: {representations}")
            result = _macro_regression_bootstrap(
                part,
                cells,
                replicates,
                _stable_seed(seed, comparison, scope),
            )
            for metric, (estimate, array) in result.items():
                finite = array[np.isfinite(array)]
                rows.append(
                    {
                        "comparison": comparison,
                        "scope": scope,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": float(np.quantile(finite, 0.025)),
                        "ci_high": float(np.quantile(finite, 0.975)),
                        "cells": len(cells),
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "same_patient_draw_across_cells_within_outer_fold",
                    }
                )
        paired_pcr = _paired_merge(
            pcr,
            grounded,
            baseline,
            ["patient_id", "fold", "decision_point"],
            ["y_true", "probability", "predicted_label"],
        )
        estimate, array = _macro_pcr_bootstrap(
            paired_pcr, replicates, _stable_seed(seed, comparison, "C_pcr")
        )
        finite = array[np.isfinite(array)]
        rows.append(
            {
                "comparison": comparison,
                "scope": "C_longitudinal_pcr",
                "metric": "auroc",
                "estimate": estimate,
                "ci_low": float(np.quantile(finite, 0.025)),
                "ci_high": float(np.quantile(finite, 0.975)),
                "cells": 2,
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "same_patient_draw_across_cells_within_outer_fold",
            }
        )
    return pd.DataFrame(rows)


def coverage_audit(
    probe: pd.DataFrame,
    pcr: pd.DataFrame,
    *,
    expected_probe_patients: int | None,
    expected_pcr_patients: int | None,
    allow_partial: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """核验 5×3×(4+3) probe 与 5×3 pCR OOF 矩阵。"""

    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    probe_universe = set(probe["patient_id"])
    pcr_universe = set(pcr["patient_id"])
    if expected_probe_patients is not None and len(probe_universe) != expected_probe_patients:
        issues.append(
            {
                "severity": "error",
                "scope": "probe patient universe",
                "detail": f"observed={len(probe_universe)}, expected={expected_probe_patients}",
            }
        )
    if expected_pcr_patients is not None and len(pcr_universe) != expected_pcr_patients:
        issues.append(
            {
                "severity": "error",
                "scope": "pCR patient universe",
                "detail": f"observed={len(pcr_universe)}, expected={expected_pcr_patients}",
            }
        )
    if not probe_universe.issubset(pcr_universe):
        issues.append(
            {
                "severity": "error",
                "scope": "cohort nesting",
                "detail": "measurement probe patient set 不是 pCR cohort 的子集",
            }
        )
    probe_cells = []
    for model in MODEL_ORDER:
        for target in TARGETS:
            probe_cells.extend((model, "static", target, item, "") for item in TIMEPOINTS)
            probe_cells.extend((model, "change", target, "", item) for item in TRANSITIONS)
    for model, task, target, timepoint, transition in probe_cells:
        part = probe.loc[
            probe["model"].eq(model)
            & probe["task"].eq(task)
            & probe["target"].eq(target)
            & probe["timepoint"].eq(timepoint)
            & probe["transition"].eq(transition)
        ]
        item = {
            "kind": "probe",
            "model": model,
            "cell": timepoint or transition,
            "target": target,
            "rows": len(part),
            "patients": part["patient_id"].nunique(),
            "fold_count": part["fold"].nunique(),
        }
        rows.append(item)
        expected = expected_probe_patients
        exact_set = expected is None or set(part["patient_id"]) == probe_universe
        if (
            part.empty
            or part["fold"].nunique() != 5
            or (expected is not None and len(part) != expected)
            or not exact_set
        ):
            item["exact_patient_set"] = bool(exact_set)
            issues.append({"severity": "error", "scope": "probe coverage", "detail": json.dumps(item, ensure_ascii=False)})
    for model in MODEL_ORDER:
        for decision in DECISION_POINTS:
            part = pcr.loc[pcr["model"].eq(model) & pcr["decision_point"].eq(decision)]
            item = {
                "kind": "pcr",
                "model": model,
                "cell": decision,
                "target": "pcr",
                "rows": len(part),
                "patients": part["patient_id"].nunique(),
                "fold_count": part["fold"].nunique(),
            }
            rows.append(item)
            expected = expected_pcr_patients
            exact_set = expected is None or set(part["patient_id"]) == pcr_universe
            if (
                part.empty
                or part["fold"].nunique() != 5
                or (expected is not None and len(part) != expected)
                or not exact_set
            ):
                item["exact_patient_set"] = bool(exact_set)
                issues.append({"severity": "error", "scope": "pCR coverage", "detail": json.dumps(item, ensure_ascii=False)})
    patient_fold = pd.concat(
        [probe[["patient_id", "fold"]], pcr[["patient_id", "fold"]]], ignore_index=True
    ).drop_duplicates()
    if patient_fold["patient_id"].duplicated().any():
        offender_count = patient_fold.loc[
            patient_fold["patient_id"].duplicated(False), "patient_id"
        ].nunique()
        issues.append(
            {
                "severity": "error",
                "scope": "OOF patient fold",
                "detail": f"跨 fold patient count={offender_count}（ID 已脱敏）",
            }
        )
    probe_truth = probe.groupby(
        ["patient_id", "fold", "task", "target", "timepoint", "transition"],
        dropna=False,
    )["y_true"].agg(["min", "max"])
    if ((probe_truth["max"] - probe_truth["min"]).abs() > 1e-9).any():
        issues.append(
            {
                "severity": "error",
                "scope": "probe truth consistency",
                "detail": "同一 patient/cell 的 y_true 在模型间不一致（ID 已脱敏）",
            }
        )
    pcr_truth = pcr.groupby(["patient_id", "fold"], dropna=False)["y_true"].nunique()
    if pcr_truth.gt(1).any():
        issues.append(
            {
                "severity": "error",
                "scope": "pCR truth consistency",
                "detail": "同一 patient 的 pCR label 在模型/decision point 间不一致（ID 已脱敏）",
            }
        )
    issue_frame = pd.DataFrame(issues, columns=("severity", "scope", "detail"))
    if not allow_partial and not issue_frame.empty:
        raise AnalysisInputError(f"严格 coverage 失败，共 {len(issue_frame)} 项:\n{issue_frame.head().to_string(index=False)}")
    return pd.DataFrame(rows), issue_frame


def normalise_history(frame: pd.DataFrame, source: Path | None = None) -> pd.DataFrame:
    aliases = {
        "learning_rate": ("lr",),
        "base_loss": ("train_base_loss", "train_state_loss"),
        "ftv_loss": ("train_ftv_loss",),
        "weighted_ftv_loss": ("train_weighted_ftv_loss",),
        "val_base_loss": ("validation_base_loss", "val_state_loss"),
        "val_ftv_metric": ("val_ftv_loss", "validation_ftv_loss"),
        "representation_std": ("visit_feature_std", "val_visit_feature_std", "train_visit_feature_std"),
    }
    output = _rename_aliases(frame, aliases)
    required = {"epoch", "fold", "model", "base_loss", "val_base_loss", "representation_std", "learning_rate"}
    missing = sorted(required.difference(output.columns))
    if missing:
        raise AnalysisInputError(f"training history 缺列 {missing}: {source or '<dataframe>'}")
    for column, default in (
        ("total_loss", np.nan),
        ("ftv_loss", 0.0),
        ("weighted_ftv_loss", 0.0),
        ("val_ftv_metric", np.nan),
        ("encoder_grad_norm", np.nan),
        ("ftv_head_grad_norm", np.nan),
        ("grounded_patients", np.nan),
        ("grounded_visits", np.nan),
        ("ungrounded_patients", np.nan),
    ):
        if column not in output:
            output[column] = default
    output["model"] = output["model"].map(_normalise_model)
    for column in (
        "epoch",
        "fold",
        "base_loss",
        "val_base_loss",
        "representation_std",
        "learning_rate",
        "total_loss",
        "ftv_loss",
        "weighted_ftv_loss",
        "val_ftv_metric",
        "encoder_grad_norm",
        "ftv_head_grad_norm",
        "grounded_patients",
        "grounded_visits",
        "ungrounded_patients",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["epoch"] = output["epoch"].astype(int)
    output["fold"] = output["fold"].astype(int)
    if not output["model"].isin({"G1", "G2", "G3", "G4"}).all():
        raise AnalysisInputError("本轮 training history model 必须为 G1–G4")
    if set(output["fold"]).difference(range(5)):
        raise AnalysisInputError("training history fold 必须在 0–4")
    if output["epoch"].le(0).any():
        raise AnalysisInputError("training history epoch 必须为正整数")
    if source is not None:
        output["source_history_file"] = _portable(source)
        output["source_history_sha256"] = file_sha256(source)
        output["run_name"] = source.parent.name
    return output


def discover_histories(root: Path, allow_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(root.rglob("fold_*.csv")) if root.is_dir() else []
    if not paths:
        if allow_partial:
            return pd.DataFrame(), pd.DataFrame()
        raise AnalysisInputError(f"未发现 training history: {root}")
    frames = [normalise_history(pd.read_csv(path), path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    manifest = (
        combined[["source_history_file", "source_history_sha256", "run_name", "model", "fold"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return combined.sort_values(["model", "fold", "epoch"]).reset_index(drop=True), manifest


def _nested_value(payload: Mapping[str, Any], candidates: Sequence[Sequence[str]]) -> Any:
    for candidate in candidates:
        value: Any = payload
        found = True
        for key in candidate:
            if not isinstance(value, Mapping) or key not in value:
                found = False
                break
            value = value[key]
        if found:
            return value
    return None


def discover_selections(root: Path, allow_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取训练器 selection.json；正式决策绝不从 history 重选 epoch。"""

    paths = sorted(root.rglob("selection.json")) if root.is_dir() else []
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AnalysisInputError(f"无法读取 selection JSON {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise AnalysisInputError(f"selection JSON 顶层必须为 object: {path}")
        resolved_path = path.parent / "resolved_run.json"
        resolved: Mapping[str, Any] = {}
        if resolved_path.is_file():
            loaded = json.loads(resolved_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                resolved = loaded
        model_value = _nested_value(
            payload,
            (("model",), ("model_name",), ("config", "model"), ("resolved_config", "model")),
        )
        if model_value is None:
            model_value = _nested_value(
                resolved,
                (("model",), ("model_name",), ("config", "model"), ("resolved_config", "model")),
            )
        fold_value = _nested_value(payload, (("fold",), ("config", "fold"), ("resolved_config", "fold")))
        if fold_value is None:
            fold_value = _nested_value(resolved, (("fold",), ("config", "fold"), ("resolved_config", "fold")))
        epoch_value = _nested_value(
            payload,
            (
                ("selected_epoch",),
                ("best_epoch",),
                ("selected", "epoch"),
                ("selection", "selected_epoch"),
                ("checkpoint_selection", "selected_epoch"),
            ),
        )
        if model_value is None or fold_value is None or epoch_value is None:
            if allow_partial:
                continue
            raise AnalysisInputError(
                f"selection JSON 缺 model/fold/selected_epoch，正式分析拒绝 history proxy: {path}"
            )
        model = _normalise_model(model_value)
        fold = int(fold_value)
        epoch = int(epoch_value)
        run_name = path.parents[1].name
        test_usage_value = _nested_value(
            payload,
            (
                ("test_used_for_selection",),
                ("test_data_used",),
                ("selection", "test_used"),
                ("data_access_contract", "test_used_for_selection"),
            ),
        )
        rows.append(
            {
                "model": model,
                "fold": fold,
                "selected_epoch": epoch,
                "run_name": run_name,
                "selection_file": _portable(path),
                "selection_sha256": file_sha256(path),
                "test_usage_declared": test_usage_value is not None,
                "test_used_for_selection": bool(test_usage_value or False),
            }
        )
        manifest.append(
            {
                "path": _portable(path),
                "sha256": file_sha256(path),
                "resolved_run_path": _portable(resolved_path) if resolved_path.is_file() else "",
                "resolved_run_sha256": file_sha256(resolved_path) if resolved_path.is_file() else "",
                "model": model,
                "fold": fold,
                "selected_epoch": epoch,
                "run_name": run_name,
                "test_usage_declared": test_usage_value is not None,
                "test_used_for_selection": bool(test_usage_value or False),
            }
        )
    selections = pd.DataFrame(rows)
    if selections.empty:
        if allow_partial:
            return selections, pd.DataFrame(manifest)
        raise AnalysisInputError(f"未发现可解析 selection.json: {root}")
    if selections.duplicated(["model", "fold"]).any():
        duplicate = selections.loc[
            selections.duplicated(["model", "fold"], False),
            ["model", "fold", "run_name", "selection_file"],
        ]
        raise AnalysisInputError(f"同一 model/fold 有多个正式 selection:\n{duplicate.to_string(index=False)}")
    if not allow_partial:
        observed = set(selections[["model", "fold"]].itertuples(index=False, name=None))
        if observed != FORMAL_HISTORY_KEYS:
            raise AnalysisInputError(
                "正式 selection coverage 必须恰好为 G1–G4×5 folds: "
                f"missing={sorted(FORMAL_HISTORY_KEYS-observed)}, "
                f"extra={sorted(observed-FORMAL_HISTORY_KEYS)}"
            )
        if not selections["test_usage_declared"].all():
            raise AnalysisInputError("正式 selection 必须显式声明 test_data_used=false")
        if selections["test_used_for_selection"].any():
            raise AnalysisInputError("正式 checkpoint selection 检出 test data 使用")
    return selections.sort_values(["model", "fold"]).reset_index(drop=True), pd.DataFrame(manifest)


def history_stability(
    history: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    allow_partial: bool,
) -> pd.DataFrame:
    """严格关联 selection.json 指定 epoch；proxy 仅供 allow_partial 自测。"""

    if history.empty:
        if not allow_partial:
            raise AnalysisInputError("正式 stability analysis 缺 training history")
        return pd.DataFrame(columns=("model", "fold", "selected_epoch", "val_base_loss", "representation_std", "finite"))
    rows = []
    expected = set(FORMAL_HISTORY_KEYS)
    selection_lookup = {
        (row.model, int(row.fold)): row for row in selections.itertuples(index=False)
    }
    history_keys = set(history[["model", "fold"]].itertuples(index=False, name=None))
    if not allow_partial:
        selection_keys = set(selection_lookup)
        if selection_keys != expected:
            raise AnalysisInputError(
                f"正式稳定性 selection coverage 错误: missing={sorted(expected-selection_keys)}, "
                f"extra={sorted(selection_keys-expected)}"
            )
        if history_keys != expected:
            raise AnalysisInputError(
                f"正式 stability history coverage 错误: missing={sorted(expected-history_keys)}, "
                f"extra={sorted(history_keys-expected)}"
            )
        if "run_name" not in history:
            raise AnalysisInputError("正式 history 缺 run_name provenance")
        run_coverage = history[["model", "fold", "run_name"]].drop_duplicates()
        if run_coverage.duplicated(["model", "fold"], keep=False).any():
            raise AnalysisInputError("正式 history 同一 model/fold 混入多个 run")
        if history.duplicated(["model", "fold", "epoch"]).any():
            raise AnalysisInputError("正式 history 同一 model/fold/epoch 重复")
    groups = history.groupby(["model", "fold"])
    for (model, fold), group in groups:
        key = (model, int(fold))
        selection = selection_lookup.get(key)
        if selection is not None:
            if not allow_partial:
                run_names = set(group["run_name"].astype(str))
                if run_names != {str(selection.run_name)}:
                    raise AnalysisInputError(
                        f"formal history/selection run_name 不一致: {model}/fold_{fold}"
                    )
            selected_rows = group.loc[group["epoch"].eq(int(selection.selected_epoch))]
            if hasattr(selection, "run_name") and "run_name" in selected_rows:
                matching_run = selected_rows.loc[selected_rows["run_name"].eq(selection.run_name)]
                if not matching_run.empty:
                    selected_rows = matching_run
            if len(selected_rows) != 1:
                if not allow_partial:
                    raise AnalysisInputError(
                        f"selection epoch 无法唯一关联 history: {model}/fold_{fold}/epoch_{selection.selected_epoch}"
                    )
                selected_rows = selected_rows.iloc[:1]
            if not selected_rows.empty:
                selected = selected_rows.iloc[0]
                selection_source = str(selection.selection_file)
            else:
                selection = None
        if selection is None:
            if not allow_partial:
                # G0 可没有本轮 history；G1-G4 在上方 expected 已拦截。
                continue
            candidate = group.loc[
                np.isfinite(group["val_base_loss"]) & np.isfinite(group["representation_std"])
            ].copy()
            if candidate.empty:
                selected = group.iloc[-1]
            elif model in {"G3", "G4"} and candidate["val_ftv_metric"].notna().any():
                selected = candidate.sort_values(["val_ftv_metric", "val_base_loss", "epoch"]).iloc[0]
            else:
                selected = candidate.sort_values(["val_base_loss", "epoch"]).iloc[0]
            selection_source = "allow_partial_history_proxy"
        finite_columns = ["base_loss", "val_base_loss", "representation_std", "learning_rate"]
        if model in {"G3", "G4"}:
            finite_columns.extend(["ftv_loss", "weighted_ftv_loss", "val_ftv_metric"])
        numeric = pd.to_numeric(selected[finite_columns], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "fold": int(fold),
                "selected_epoch": int(selected["epoch"]),
                "val_base_loss": float(selected["val_base_loss"]),
                "representation_std": float(selected["representation_std"]),
                "finite": bool(np.isfinite(numeric).all()),
                "selection_source": selection_source,
            }
        )
    summary = pd.DataFrame(rows)
    if not allow_partial:
        observed = set(summary[["model", "fold"]].itertuples(index=False, name=None))
        if observed != expected or len(summary) != len(expected):
            raise AnalysisInputError("正式 training stability 必须恰好产生 20 个 selected rows")
    for grounded, baseline in PRIMARY_COMPARISONS:
        baseline_lookup = summary.loc[summary["model"].eq(baseline)].set_index("fold")["val_base_loss"]
        mask = summary["model"].eq(grounded)
        summary.loc[mask, "paired_baseline"] = baseline
        summary.loc[mask, "base_degradation_fraction"] = [
            (value - baseline_lookup.loc[int(fold)]) / max(baseline_lookup.loc[int(fold)], 1e-12)
            if int(fold) in baseline_lookup.index
            else math.nan
            for fold, value in zip(summary.loc[mask, "fold"], summary.loc[mask, "val_base_loss"])
        ]
    return summary


def _quantile_ci(array: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def compute_decision(
    macro: pd.DataFrame,
    paired_ci: pd.DataFrame,
    stability: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """按 EXPERIMENT_PLAN §14–15 的预注册阈值机械化决策。"""

    gate_rows: list[dict[str, Any]] = []
    comparison_results: dict[str, Any] = {}
    for grounded, baseline in PRIMARY_COMPARISONS:
        comparison = f"{grounded}-{baseline}"
        subset = macro.loc[macro["comparison"].eq(comparison)]

        def macro_row(scope: str, metric: str) -> pd.Series:
            rows = subset.loc[subset["scope"].eq(scope) & subset["metric"].eq(metric)]
            if len(rows) != 1:
                raise AnalysisInputError(f"decision 缺少唯一 {comparison}/{scope}/{metric}")
            return rows.iloc[0]

        static_s = macro_row("A_static_ftv", "spearman")
        static_r = macro_row("A_static_ftv", "r2")
        pass_s = bool(static_s.estimate >= 0.05 and static_s.ci_low > 0)
        pass_r = bool(static_r.estimate >= 0.05 and static_r.ci_low > 0)
        reverse_s = bool(static_s.estimate <= -0.05 and static_s.ci_high < 0)
        reverse_r = bool(static_r.estimate <= -0.05 and static_r.ci_high < 0)
        gate_a = (pass_s and not reverse_r) or (pass_r and not reverse_s)

        change_s = macro_row("B_change_ftv", "spearman")
        change_r = macro_row("B_change_ftv", "r2")
        macro_pass_s = bool(change_s.estimate >= 0.05 and change_s.ci_low > 0)
        macro_pass_r = bool(change_r.estimate >= 0.05 and change_r.ci_low > 0)
        cell = paired_ci.loc[
            paired_ci["comparison"].eq(comparison)
            & paired_ci["kind"].eq("probe")
            & paired_ci["task"].eq("change")
            & paired_ci["target"].eq("ftv")
            & paired_ci["metric"].isin({"spearman", "r2"})
        ].copy()
        transition_pass = False
        for metric in ("spearman", "r2"):
            metric_rows = cell.loc[cell["metric"].eq(metric)]
            positives = metric_rows["estimate"].ge(0.05) & metric_rows["ci_low"].gt(0)
            stable_reverse = metric_rows["estimate"].le(-0.05) & metric_rows["ci_high"].lt(0)
            if positives.any() and not stable_reverse.any():
                transition_pass = True
        gate_b = macro_pass_s or macro_pass_r or transition_pass

        pcr_macro = macro_row("C_longitudinal_pcr", "auroc")
        pcr_cells = paired_ci.loc[
            paired_ci["comparison"].eq(comparison)
            & paired_ci["kind"].eq("pcr")
            & paired_ci["metric"].eq("auroc")
            & paired_ci["decision_point"].isin({"T0-T1", "T0-T2"})
        ]
        both_positive = len(pcr_cells) == 2 and pcr_cells["estimate"].gt(0).all()
        gate_c = bool(both_positive and pcr_macro.estimate >= 0.02 and pcr_macro.ci_low > 0)
        clear_pcr_decline = bool(((pcr_cells["estimate"] < 0) & (pcr_cells["ci_high"] < 0)).any())

        grounded_stability = stability.loc[stability["model"].eq(grounded)]
        stable = bool(
            len(grounded_stability) == 5
            and grounded_stability["finite"].all()
            and grounded_stability["representation_std"].ge(0.05).all()
            and grounded_stability["base_degradation_fraction"].le(0.05 + 1e-12).all()
        )
        eligible = stable and not clear_pcr_decline
        result = {
            "A_static_grounding": gate_a,
            "B_observed_delta_ftv": gate_b,
            "C_longitudinal_pcr": gate_c,
            "no_clear_pcr_decline": not clear_pcr_decline,
            "stability_and_base_gate": stable,
            "eligible": eligible,
            "go": eligible and gate_a and (gate_b or gate_c),
            "partial_go": eligible and gate_a and not (gate_b or gate_c),
        }
        comparison_results[comparison] = result
        for gate, passed in result.items():
            gate_rows.append({"comparison": comparison, "gate": gate, "passed": bool(passed)})
    if any(item["go"] for item in comparison_results.values()):
        decision = "GO"
        rationale = "至少一个 grounded pairing 满足 A，且满足 B/C 之一，同时通过稳定性、base 与 pCR 非下降门槛。"
    elif any(item["partial_go"] for item in comparison_results.values()):
        decision = "PARTIAL GO"
        rationale = "至少一个 grounded pairing 只满足可靠 static grounding，尚无可靠 dynamic/pCR 改善。"
    else:
        decision = "NO-GO"
        rationale = "没有 grounded pairing 在全部安全门槛下满足预注册 static grounding 标准。"
    payload = {
        "schema_version": 1,
        "formal_analysis": True,
        "decision": decision,
        "rationale_cn": rationale,
        "comparisons": comparison_results,
        "thresholds": {
            "A_macro_gain": 0.05,
            "B_macro_or_transition_gain": 0.05,
            "C_macro_auroc_gain": 0.02,
            "ci_requirement": "95% paired patient bootstrap lower bound > 0",
            "max_validation_base_degradation": 0.05,
            "minimum_representation_std": 0.05,
        },
        "bootstrap_conditional_on_single_training_seed": True,
        "multiple_comparison_adjustment": "none; pre-registered cells, interpret individual CIs conditionally",
    }
    return payload, pd.DataFrame(gate_rows)


def _setup_matplotlib() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    matplotlib.rcParams["font.family"] = next((item for item in candidates if item in available), "DejaVu Sans")
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 130


def _placeholder(title: str, message: str) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return figure


def _require_data(frame: pd.DataFrame, title: str, allow_partial: bool) -> Figure | None:
    if not frame.empty:
        return None
    if allow_partial:
        return _placeholder(title, "开发模式：当前输入缺少该注册图所需数据；未填造数值。")
    raise AnalysisInputError(f"无法生成注册图，缺少数据: {title}")


MODEL_COLORS = dict(zip(MODEL_ORDER, ("#636363", "#3182bd", "#31a354", "#756bb1", "#e6550d")))


def _metric_lines(oof: pd.DataFrame, task: str, target: str, metric: str, title: str) -> Figure:
    column = "timepoint" if task == "static" else "transition"
    order = TIMEPOINTS if task == "static" else TRANSITIONS
    data = oof.loc[oof["task"].eq(task) & oof["target"].eq(target)]
    figure, axis = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        part = data.loc[data["model"].eq(model)].set_index(column).reindex(order)
        axis.plot(order, part[metric], marker="o", label=model, color=MODEL_COLORS[model])
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_ylabel(metric.upper() if metric == "r2" else "Spearman ρ")
    axis.set_title(title)
    axis.legend(ncol=5, frameon=False)
    figure.tight_layout()
    return figure


def _paired_gain_figure(paired_ci: pd.DataFrame, comparison: str, allow_partial: bool) -> Figure:
    probe = paired_ci.loc[
        paired_ci["comparison"].eq(comparison)
        & paired_ci["kind"].eq("probe")
        & paired_ci["target"].eq("ftv")
        & paired_ci["metric"].isin({"spearman", "r2"})
    ].copy()
    pcr = paired_ci.loc[
        paired_ci["comparison"].eq(comparison)
        & paired_ci["kind"].eq("pcr")
        & paired_ci["metric"].eq("auroc")
    ].copy()
    placeholder = _require_data(probe, f"{comparison} paired gain", allow_partial)
    if placeholder is not None:
        return placeholder
    probe["cell"] = np.where(probe["task"].eq("static"), probe["timepoint"], probe["transition"])
    probe["label"] = probe["metric"] + " | " + probe["cell"]
    pcr["label"] = "AUROC | " + pcr["decision_point"]
    data = pd.concat([probe, pcr], ignore_index=True, sort=False)
    figure, axis = plt.subplots(figsize=(11, 8))
    positions = np.arange(len(data))
    estimate = data["estimate"].to_numpy(dtype=float)
    axis.errorbar(
        estimate,
        positions,
        xerr=np.vstack([estimate - data["ci_low"], data["ci_high"] - estimate]),
        fmt="o",
        capsize=3,
        color="#2c7fb8",
    )
    axis.axvline(0, color="#555555", linewidth=0.9)
    axis.set_yticks(positions, data["label"])
    axis.set_xlabel("grounded − paired baseline（95% paired bootstrap CI）")
    axis.set_title(f"{comparison}：预注册 paired improvement")
    figure.tight_layout()
    return figure


def _scatter_figure(probe: pd.DataFrame, task: str, allow_partial: bool) -> Figure:
    data = probe.loc[probe["task"].eq(task) & probe["target"].eq("ftv")]
    title = "Static FTV OOF true vs predicted" if task == "static" else "ΔFTV OOF true vs predicted"
    placeholder = _require_data(data, title, allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    for axis, model in zip(axes, MODEL_ORDER):
        part = data.loc[data["model"].eq(model)]
        axis.scatter(part["y_true"], part["y_pred"], s=8, alpha=0.25, color=MODEL_COLORS[model])
        limits = np.quantile(np.concatenate([part["y_true"], part["y_pred"]]), [0.01, 0.99])
        axis.plot(limits, limits, linestyle="--", color="#555555", linewidth=0.8)
        axis.set_title(model)
        axis.set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _pcr_figure(oof: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DECISION_POINTS))
    width = 0.15
    for index, model in enumerate(MODEL_ORDER):
        part = oof.loc[oof["model"].eq(model)].set_index("decision_point").reindex(DECISION_POINTS)
        axis.bar(x + (index - 2) * width, part["auroc"], width=width, label=model, color=MODEL_COLORS[model])
    axis.axhline(0.5, linestyle="--", color="#555555", linewidth=0.8)
    axis.set_xticks(x, DECISION_POINTS)
    axis.set_ylim(0.35, 0.85)
    axis.set_ylabel("Pooled OOF AUROC")
    axis.set_title("严格 image-state-only pCR readout")
    axis.legend(ncol=5, frameon=False)
    figure.tight_layout()
    return figure


def load_geometry_control(path: Path, *, allow_partial: bool) -> pd.DataFrame:
    """只读复用 OSRA mask-geometry upper/control；不把 geometry 送入新模型。"""

    columns = ("task", "cell", "target", "spearman", "r2", "n_patients", "source_path", "source_sha256")
    if not path.is_file():
        if allow_partial:
            return pd.DataFrame(columns=columns)
        raise AnalysisInputError(f"缺旧 OSRA mask-geometry control: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"task_type", "model", "representation_role", "input_variant", "target_name", "timepoint", "transition", "spearman", "r2", "n_patients"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AnalysisInputError(f"OSRA geometry control 缺列 {missing}: {path}")
    data = frame.loc[
        frame["model"].astype(str).str.lower().eq("m0")
        & frame["representation_role"].astype(str).str.lower().eq("mask_geometry")
        & frame["target_name"].astype(str).str.lower().eq("ftv")
        & (
            (frame["task_type"].astype(str).str.lower().eq("static") & frame["input_variant"].astype(str).str.lower().eq("current"))
            | (frame["task_type"].astype(str).str.lower().eq("change") & frame["input_variant"].astype(str).str.lower().eq("observed_difference"))
        )
    ].copy()
    data["task"] = data["task_type"].astype(str).str.lower()
    data["cell"] = np.where(
        data["task"].eq("static"),
        data["timepoint"].astype(str).str.upper(),
        data["transition"].map(_normalise_transition),
    )
    data["target"] = "ftv"
    data["source_path"] = _portable(path)
    data["source_sha256"] = file_sha256(path)
    output = data.loc[:, columns].drop_duplicates(["task", "cell", "target"])
    expected = {("static", item) for item in TIMEPOINTS} | {("change", item) for item in TRANSITIONS}
    observed = set(output[["task", "cell"]].itertuples(index=False, name=None))
    if observed != expected and not allow_partial:
        raise AnalysisInputError(f"OSRA geometry control coverage 不完整: 缺 {sorted(expected-observed)}")
    return output.reset_index(drop=True)


def _mask_contract_figure(
    probe_oof: pd.DataFrame,
    pcr_oof: pd.DataFrame,
    geometry_control: pd.DataFrame,
) -> Figure:
    rows = []
    for model in MODEL_ORDER:
        static = probe_oof.loc[
            probe_oof["model"].eq(model) & probe_oof["task"].eq("static") & probe_oof["target"].eq("ftv")
        ]
        change = probe_oof.loc[
            probe_oof["model"].eq(model) & probe_oof["task"].eq("change") & probe_oof["target"].eq("ftv")
        ]
        pcr = pcr_oof.loc[
            pcr_oof["model"].eq(model) & pcr_oof["decision_point"].isin({"T0-T1", "T0-T2"})
        ]
        rows.append((model, static["spearman"].mean(), change["spearman"].mean(), pcr["auroc"].mean()))
    if not geometry_control.empty:
        rows.append(
            (
                "Geometry control",
                geometry_control.loc[geometry_control["task"].eq("static"), "spearman"].mean(),
                geometry_control.loc[geometry_control["task"].eq("change"), "spearman"].mean(),
                math.nan,
            )
        )
    data = pd.DataFrame(rows, columns=("model", "static_ftv_spearman", "change_ftv_spearman", "longitudinal_pcr_auroc"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title in zip(
        axes,
        data.columns[1:],
        ("Static FTV", "Observed ΔFTV", "Longitudinal pCR"),
    ):
        axis.bar(
            data["model"],
            data[metric],
            color=[MODEL_COLORS.get(item, "#bdbdbd") for item in data["model"]],
        )
        axis.axhline(0 if "pcr" not in metric else 0.5, color="#777777", linewidth=0.8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel(metric)
    figure.suptitle("Mask contract：channel / no-mask / normalized ROI pooling / grounding")
    figure.tight_layout()
    return figure


def _transfer_figure(probe_oof: pd.DataFrame) -> Figure:
    data = probe_oof.loc[probe_oof["target"].isin({"ld", "sphericity"})]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, target in enumerate(("ld", "sphericity")):
        for column, task in enumerate(("static", "change")):
            axis = axes[row, column]
            part = data.loc[data["target"].eq(target) & data["task"].eq(task)]
            cell_col = "timepoint" if task == "static" else "transition"
            order = TIMEPOINTS if task == "static" else TRANSITIONS
            for model in MODEL_ORDER:
                cell = part.loc[part["model"].eq(model)].set_index(cell_col).reindex(order)
                axis.plot(order, cell["spearman"], marker="o", color=MODEL_COLORS[model], label=model)
            axis.axhline(0, color="#777777", linewidth=0.8)
            axis.set_title(f"{target.upper()} · {task}")
            axis.set_ylabel("Spearman ρ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("LD / sphericity secondary transfer probe", y=0.995)
    figure.legend(
        handles,
        labels,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _history_figure(history: pd.DataFrame, allow_partial: bool) -> Figure:
    placeholder = _require_data(history, "Training curves", allow_partial)
    if placeholder is not None:
        return placeholder
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = (
        ("val_base_loss", "Validation JEPA base loss"),
        ("ftv_loss", "Raw FTV loss"),
        ("weighted_ftv_loss", "Weighted FTV loss"),
        ("representation_std", "Representation std"),
    )
    summary = history.groupby(["model", "epoch"], as_index=False).agg(
        **{column: (column, "mean") for column, _ in specs}
    )
    for axis, (metric, title) in zip(axes.flat, specs):
        for model in MODEL_ORDER:
            part = summary.loc[summary["model"].eq(model)]
            if not part.empty:
                axis.plot(part["epoch"], part[metric], label=model, color=MODEL_COLORS[model])
        if metric == "representation_std":
            axis.axhline(0.05, color="#d7301f", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("JEPA / FTV / representation stability training curves", y=0.995)
    figure.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def save_figures(
    probe: pd.DataFrame,
    pcr: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    paired_ci: pd.DataFrame,
    history: pd.DataFrame,
    geometry_control: pd.DataFrame,
    output_dir: Path,
    *,
    allow_partial: bool,
) -> pd.DataFrame:
    """生成恰好十二类预注册图与 SHA manifest。"""

    _setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, str, str, Callable[[], Figure]]] = [
        ("01_static_ftv_spearman.png", "G0–G4 static FTV Spearman", "probe_oof_metrics", lambda: _metric_lines(tables["probe_oof"], "static", "ftv", "spearman", "G0–G4 static FTV Spearman")),
        ("02_static_ftv_r2.png", "G0–G4 static FTV R²", "probe_oof_metrics", lambda: _metric_lines(tables["probe_oof"], "static", "ftv", "r2", "G0–G4 static FTV R²")),
        ("03_delta_ftv_spearman.png", "G0–G4 observed ΔFTV Spearman", "probe_oof_metrics", lambda: _metric_lines(tables["probe_oof"], "change", "ftv", "spearman", "G0–G4 observed ΔFTV Spearman")),
        ("04_delta_ftv_r2.png", "G0–G4 observed ΔFTV R²", "probe_oof_metrics", lambda: _metric_lines(tables["probe_oof"], "change", "ftv", "r2", "G0–G4 observed ΔFTV R²")),
        ("05_g3_minus_g1_paired.png", "G3−G1 paired improvement", "paired_difference_bootstrap_ci", lambda: _paired_gain_figure(paired_ci, "G3-G1", allow_partial)),
        ("06_g4_minus_g2_paired.png", "G4−G2 paired improvement", "paired_difference_bootstrap_ci", lambda: _paired_gain_figure(paired_ci, "G4-G2", allow_partial)),
        ("07_static_ftv_scatter.png", "Static FTV true vs predicted", "probe predictions", lambda: _scatter_figure(probe, "static", allow_partial)),
        ("08_delta_ftv_scatter.png", "Observed ΔFTV true vs predicted", "probe predictions", lambda: _scatter_figure(probe, "change", allow_partial)),
        ("09_image_only_pcr_auroc.png", "Image-only pCR AUROC", "pcr_oof_metrics", lambda: _pcr_figure(tables["pcr_oof"])),
        ("10_mask_contract_comparison.png", "Mask contract comparison", "probe+pcr oof metrics + prior OSRA mask-geometry control", lambda: _mask_contract_figure(tables["probe_oof"], tables["pcr_oof"], geometry_control)),
        ("11_ld_sphericity_transfer.png", "LD/sphericity transfer", "probe_oof_metrics", lambda: _transfer_figure(tables["probe_oof"])),
        ("12_training_curves.png", "JEPA/FTV/std training curves", "training histories", lambda: _history_figure(history, allow_partial)),
    ]
    rows = []
    for filename, title, source, builder in specs:
        figure = builder()
        path = output_dir / filename
        figure.savefig(path, bbox_inches="tight", metadata={"Title": title, "Author": "DGRS aggregate_results"})
        plt.close(figure)
        decoded = plt.imread(path)
        if decoded.size == 0 or decoded.ndim not in (2, 3) or not np.isfinite(decoded).all():
            raise AnalysisInputError(f"注册图无法可靠解码: {filename}")
        rows.append(
            {
                "figure": filename,
                "title": title,
                "source": source,
                "path": _portable(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "decodable": True,
            }
        )
    return pd.DataFrame(rows)


def _fold_summary(fold: pd.DataFrame, group: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, part in fold.groupby(list(group), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group, keys))
        row["fold_count"] = part["fold"].nunique()
        for metric in metrics:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_sd_ddof1"] = part[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_analysis_acceptance_evidence(
    *,
    config: AnalysisConfig,
    probe: pd.DataFrame,
    pcr: pd.DataFrame,
    coverage_issues: pd.DataFrame,
    prediction_manifest: pd.DataFrame,
    history_manifest: pd.DataFrame,
    selection_manifest: pd.DataFrame,
    stability: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """供独立 19 项 acceptance verifier 消费的聚合层机器证据。"""

    formal = not config.allow_partial
    prediction_kind_counts = (
        prediction_manifest.groupby("kind").size().to_dict()
        if not prediction_manifest.empty and "kind" in prediction_manifest
        else {}
    )
    selection_test_safe = bool(
        len(selection_manifest) == 20
        and selection_manifest["path"].nunique() == 20
        and "test_usage_declared" in selection_manifest
        and selection_manifest["test_usage_declared"].astype(bool).all()
        and not selection_manifest["test_used_for_selection"].astype(bool).any()
    )
    figure_safe = bool(
        len(figure_manifest) == 12
        and figure_manifest["decodable"].astype(bool).all()
        and figure_manifest["sha256"].astype(str).map(
            lambda value: bool(SHA256_VALUE.fullmatch(value))
        ).all()
    )
    checks = {
        "formal_mode": {
            "passed": formal,
            "evidence": "allow_partial=false" if formal else "development-only partial mode",
        },
        "bootstrap_2000": {
            "passed": formal and config.bootstrap_replicates == 2000,
            "evidence": int(config.bootstrap_replicates),
        },
        "strict_prediction_rows": {
            "passed": formal and len(probe) == 39_375 and len(pcr) == 12_120,
            "evidence": {"probe": len(probe), "pcr": len(pcr)},
        },
        "prediction_files_5_models_x_5_folds": {
            "passed": formal
            and prediction_kind_counts.get("representation_probe") == 25
            and prediction_kind_counts.get("pcr_readout") == 25,
            "evidence": prediction_kind_counts,
        },
        "coverage_clean": {
            "passed": formal and coverage_issues.empty,
            "evidence": {"registered_issues": len(coverage_issues)},
        },
        "formal_histories_20": {
            "passed": formal
            and len(history_manifest) == 20
            and history_manifest["source_history_file"].nunique() == 20,
            "evidence": {
                "rows": len(history_manifest),
                "files": (
                    history_manifest["source_history_file"].nunique()
                    if "source_history_file" in history_manifest
                    else 0
                ),
            },
        },
        "formal_selections_20_test_blind": {
            "passed": formal and selection_test_safe,
            "evidence": {
                "selection_assets": len(selection_manifest),
                "test_safe": selection_test_safe,
            },
        },
        "selected_stability_rows_20": {
            "passed": formal and len(stability) == 20,
            "evidence": len(stability),
        },
        "registered_figures_12_decodable": {
            "passed": formal and figure_safe,
            "evidence": {"figures": len(figure_manifest), "all_decodable": figure_safe},
        },
        "registered_decision_only_in_formal_mode": {
            "passed": formal and decision.get("decision") in {"GO", "PARTIAL GO", "NO-GO"},
            "evidence": str(decision.get("decision")),
        },
    }
    eligible = bool(formal and all(bool(item["passed"]) for item in checks.values()))
    return {
        "schema_version": 1,
        "artifact": "DGRS analysis-layer acceptance evidence",
        "formal_analysis": formal,
        "eligible_for_independent_acceptance_verifier": eligible,
        "checks": checks,
        "analysis_source_sha256": file_sha256(Path(__file__).resolve()),
        "aggregate_script_sha256": file_sha256(
            EXPERIMENT_ROOT / "scripts" / "aggregate_results.py"
        ),
        "contains_patient_level_rows": False,
        "path_policy": "repo-relative; external directories replaced by a 12-hex provenance token",
    }


def _commit_staged(
    stage_metric: Path,
    stage_figure: Path,
    metric_dir: Path,
    figure_dir: Path,
    overwrite: bool,
) -> None:
    metric_dir.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (metric_dir, figure_dir) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出已存在，默认拒绝覆盖: " + ", ".join(map(str, existing)))
    token = next(tempfile._get_candidate_names())  # noqa: SLF001
    backups: dict[Path, Path] = {}
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
        for destination in committed:
            if destination.exists():
                shutil.rmtree(destination)
        for destination, backup in backups.items():
            if backup.exists():
                backup.replace(destination)
        raise
    for backup in backups.values():
        shutil.rmtree(backup)


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    if config.bootstrap_replicates < 20:
        raise ValueError("bootstrap_replicates 至少为 20；正式必须为 2000")
    if not config.allow_partial and config.bootstrap_replicates != 2000:
        raise ValueError("正式严格聚合固定 2000 次 patient bootstrap")
    if not config.allow_partial and (
        config.expected_probe_patients != 375 or config.expected_pcr_patients != 808
    ):
        raise ValueError("正式聚合 patient coverage 固定为 probe=375、pCR=808，不允许 CLI 改写")
    if config.allow_partial and not (
        _is_partial_output(config.metric_dir) and _is_partial_output(config.figure_dir)
    ):
        raise ValueError(
            "allow_partial 只能写入 dev_/partial_/selftest_ 前缀目录，禁止污染 final 输出"
        )
    if config.metric_dir.resolve() == config.figure_dir.resolve():
        raise ValueError("metric_dir 与 figure_dir 必须不同")
    existing = [path for path in (config.metric_dir, config.figure_dir) if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError("输出已存在，默认拒绝覆盖: " + ", ".join(map(str, existing)))

    probe, pcr, prediction_manifest = discover_predictions(
        config.prediction_root, allow_partial=config.allow_partial
    )
    coverage, issues = coverage_audit(
        probe,
        pcr,
        expected_probe_patients=config.expected_probe_patients,
        expected_pcr_patients=config.expected_pcr_patients,
        allow_partial=config.allow_partial,
    )
    history, history_manifest = discover_histories(config.history_root, config.allow_partial)
    selections, selection_manifest = discover_selections(
        config.checkpoint_root, config.allow_partial
    )
    if not config.allow_partial:
        history_files = (
            history_manifest["source_history_file"].nunique()
            if "source_history_file" in history_manifest
            else 0
        )
        selection_files = (
            selection_manifest["path"].nunique() if "path" in selection_manifest else 0
        )
        if len(history_manifest) != 20 or history_files != 20:
            raise AnalysisInputError(
                "正式分析必须恰好发现 20 个独立 model×fold history assets，"
                f"rows={len(history_manifest)}, files={history_files}"
            )
        if len(selection_manifest) != 20 or selection_files != 20:
            raise AnalysisInputError(
                "正式分析必须恰好发现 20 个独立 model×fold selection assets，"
                f"rows={len(selection_manifest)}, files={selection_files}"
            )
    geometry_control = load_geometry_control(
        config.geometry_control_path, allow_partial=config.allow_partial
    )
    tables = metric_tables(probe, pcr)
    probe_bootstrap, pcr_bootstrap = bootstrap_tables(
        probe, pcr, tables, config.bootstrap_replicates, config.seed
    )
    paired_fold, paired_ci = paired_difference_tables(
        probe, pcr, config.bootstrap_replicates, config.seed
    )
    macro = macro_paired_table(probe, pcr, config.bootstrap_replicates, config.seed)
    stability = history_stability(
        history, selections, allow_partial=config.allow_partial
    )
    if config.allow_partial:
        decision = {
            "schema_version": 1,
            "decision": "UNAVAILABLE",
            "formal_analysis": False,
            "rationale_cn": "开发/partial 模式永不输出 GO/PARTIAL GO/NO-GO 正式结论。",
        }
        gate_table = pd.DataFrame(columns=("comparison", "gate", "passed"))
    else:
        decision, gate_table = compute_decision(macro, paired_ci, stability)

    common_parent = Path(os.path.commonpath([config.metric_dir.parent.resolve(), config.figure_dir.parent.resolve()]))
    common_parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".dgrs-analysis-", dir=common_parent))
    stage_metric = stage_root / "metrics"
    stage_figure = stage_root / "figures"
    stage_metric.mkdir()
    stage_figure.mkdir()
    try:
        probe_fold_summary = _fold_summary(
            tables["probe_fold"],
            ("model", "task", "target", "timepoint", "transition", "representation"),
            REGRESSION_METRICS,
        )
        pcr_fold_summary = _fold_summary(
            tables["pcr_fold"], ("model", "decision_point"), CLASSIFICATION_METRICS
        )
        output_tables = {
            "prediction_file_manifest.csv": prediction_manifest,
            "history_file_manifest.csv": history_manifest,
            "selection_file_manifest.csv": selection_manifest,
            "geometry_control_reference.csv": geometry_control,
            "coverage.csv": coverage,
            "input_issues.csv": issues,
            "probe_fold_metrics.csv": tables["probe_fold"],
            "probe_fold_summary.csv": probe_fold_summary,
            "probe_oof_metrics.csv": tables["probe_oof"],
            "probe_bootstrap_ci.csv": probe_bootstrap,
            "pcr_fold_metrics.csv": tables["pcr_fold"],
            "pcr_fold_summary.csv": pcr_fold_summary,
            "pcr_oof_metrics.csv": tables["pcr_oof"],
            "pcr_bootstrap_ci.csv": pcr_bootstrap,
            "paired_differences_fold.csv": paired_fold,
            "paired_differences_bootstrap_ci.csv": paired_ci,
            "paired_macro_bootstrap_ci.csv": macro,
            "training_history_combined.csv": history,
            "training_stability.csv": stability,
            "decision_gates.csv": gate_table,
        }
        for name, frame in output_tables.items():
            _write_csv(stage_metric / name, frame)
        figure_manifest = save_figures(
            probe,
            pcr,
            tables,
            paired_ci,
            history,
            geometry_control,
            stage_figure,
            allow_partial=config.allow_partial,
        )
        figure_manifest["path"] = [
            _portable(config.figure_dir / filename) for filename in figure_manifest["figure"]
        ]
        _write_csv(stage_metric / "figure_manifest.csv", figure_manifest)
        decision["input_metric_sha256"] = {
            name: file_sha256(stage_metric / name)
            for name in (
                "probe_oof_metrics.csv",
                "pcr_oof_metrics.csv",
                "paired_differences_bootstrap_ci.csv",
                "paired_macro_bootstrap_ci.csv",
                "training_stability.csv",
            )
        }
        _write_json(stage_metric / "decision.json", decision)
        acceptance_evidence = build_analysis_acceptance_evidence(
            config=config,
            probe=probe,
            pcr=pcr,
            coverage_issues=issues,
            prediction_manifest=prediction_manifest,
            history_manifest=history_manifest,
            selection_manifest=selection_manifest,
            stability=stability,
            figure_manifest=figure_manifest,
            decision=decision,
        )
        if not config.allow_partial and not acceptance_evidence[
            "eligible_for_independent_acceptance_verifier"
        ]:
            failed = [
                name
                for name, item in acceptance_evidence["checks"].items()
                if not item["passed"]
            ]
            raise AnalysisInputError(f"正式 analysis acceptance evidence 失败: {failed}")
        _write_json(stage_metric / "analysis_acceptance_evidence.json", acceptance_evidence)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete"
                if not config.allow_partial
                else "development_partial_do_not_use_for_conclusions"
            ),
            "formal_analysis": not config.allow_partial,
            "eligible_for_independent_acceptance_verifier": acceptance_evidence[
                "eligible_for_independent_acceptance_verifier"
            ],
            "decision": decision["decision"],
            "prediction_root": _portable(config.prediction_root),
            "history_root": _portable(config.history_root),
            "geometry_control_path": _portable(config.geometry_control_path),
            "geometry_control_sha256": (
                file_sha256(config.geometry_control_path)
                if config.geometry_control_path.is_file()
                else ""
            ),
            "metric_dir": _portable(config.metric_dir),
            "figure_dir": _portable(config.figure_dir),
            "bootstrap_replicates": config.bootstrap_replicates,
            "bootstrap_seed": config.seed,
            "bootstrap_unit": "patient within outer fold; same draw across paired models/cells",
            "bootstrap_conditional_on_fitted_single_seed_models": True,
            "probe_rows": len(probe),
            "probe_patients": probe["patient_id"].nunique(),
            "pcr_rows": len(pcr),
            "pcr_patients": pcr["patient_id"].nunique(),
            "models": list(MODEL_ORDER),
            "folds": sorted(map(int, probe["fold"].unique())),
            "figures": len(figure_manifest),
            "registered_issues": len(issues),
            "analysis_source_sha256": file_sha256(Path(__file__).resolve()),
            "multiple_comparison_note_cn": "单 cell CI 未作多重比较校正，且只条件于单训练 seed。",
        }
        _write_json(stage_metric / "aggregation_summary.json", summary)
        _commit_staged(stage_metric, stage_figure, config.metric_dir, config.figure_dir, config.overwrite)
        return summary
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _synthetic_predictions(seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    probe_rows = []
    pcr_rows = []
    history_rows = []
    patients = [(f"SYN-{fold}-{index:03d}", fold, index) for fold in range(5) for index in range(10)]
    quality = {"G0": 0.48, "G1": 0.60, "G2": 0.55, "G3": 0.42, "G4": 0.36}
    for patient_id, fold, index in patients:
        latent = (index - 4.5) / 3 + 0.05 * fold
        label = int(latent + rng.normal(scale=0.7) > 0)
        for model in MODEL_ORDER:
            for target_index, target in enumerate(TARGETS):
                for visit_index, timepoint in enumerate(TIMEPOINTS):
                    truth = (
                        latent
                        - 0.2 * visit_index
                        + 0.1 * target_index
                        + 0.03 * math.sin(index + 3 * target_index + visit_index)
                    )
                    prediction = truth + quality[model] * rng.normal()
                    probe_rows.append(
                        {
                            "patient_id": patient_id,
                            "fold": fold,
                            "split": "test",
                            "model": model,
                            "task": "static",
                            "timepoint": timepoint,
                            "transition": "",
                            "representation": "response_preprojector",
                            "target": target,
                            "y_true": truth,
                            "y_pred": prediction,
                            "y_true_standardized": truth,
                            "y_pred_standardized": prediction,
                            "b0_prediction": -0.2 * visit_index,
                            "b0_prediction_standardized": -0.2 * visit_index,
                            "selected_alpha": 1.0,
                        }
                    )
                for transition_index, transition in enumerate(TRANSITIONS):
                    truth = (
                        -0.3
                        + 0.08 * latent
                        + 0.05 * transition_index
                        + 0.02 * math.cos(index + 2 * target_index + transition_index)
                    )
                    prediction = truth + quality[model] * 0.25 * rng.normal()
                    probe_rows.append(
                        {
                            "patient_id": patient_id,
                            "fold": fold,
                            "split": "test",
                            "model": model,
                            "task": "change",
                            "timepoint": "",
                            "transition": transition,
                            "representation": "response_preprojector",
                            "target": target,
                            "y_true": truth,
                            "y_pred": prediction,
                            "y_true_standardized": truth,
                            "y_pred_standardized": prediction,
                            "b0_prediction": -0.3 + 0.05 * transition_index,
                            "b0_prediction_standardized": -0.3 + 0.05 * transition_index,
                            "selected_alpha": 1.0,
                        }
                    )
            for decision_index, decision in enumerate(DECISION_POINTS):
                score = latent + 0.15 * decision_index + (0.15 if model in {"G3", "G4"} else 0) + rng.normal(scale=quality[model])
                probability = 1 / (1 + np.exp(-score))
                threshold = 0.5
                pcr_rows.append(
                    {
                        "patient_id": patient_id,
                        "fold": fold,
                        "split": "test",
                        "model": model,
                        "decision_point": decision,
                        "y_true": label,
                        "probability": probability,
                        "predicted_label": int(probability >= threshold),
                        "threshold": threshold,
                    }
                )
    for model in ("G1", "G2", "G3", "G4"):
        for fold in range(5):
            for epoch in range(1, 4):
                history_rows.append(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "model": model,
                        "total_loss": 1.0 - 0.1 * epoch,
                        "base_loss": 0.8 - 0.08 * epoch,
                        "ftv_loss": 0.2 - 0.02 * epoch if model in {"G3", "G4"} else 0.0,
                        "weighted_ftv_loss": 0.02 if model in {"G3", "G4"} else 0.0,
                        "val_base_loss": 0.82 - 0.06 * epoch,
                        "val_ftv_metric": 0.25 - 0.03 * epoch if model in {"G3", "G4"} else np.nan,
                        "representation_std": 0.3,
                        "learning_rate": 5e-5,
                    }
                )
    return pd.DataFrame(probe_rows), pd.DataFrame(pcr_rows), pd.DataFrame(history_rows)


def run_self_test() -> dict[str, Any]:
    """在系统临时目录运行合成端到端测试，不写实验假结果。"""

    probe, pcr, history = _synthetic_predictions()
    with tempfile.TemporaryDirectory(prefix="dgrs-analysis-selftest-") as name:
        root = Path(name)
        for model in MODEL_ORDER:
            for fold in range(5):
                probe_path = root / "predictions" / "representation_probes" / model / f"fold_{fold}" / "test_predictions.csv"
                pcr_path = root / "predictions" / "pcr_readouts" / model / f"fold_{fold}" / "test_predictions.csv"
                probe_path.parent.mkdir(parents=True, exist_ok=True)
                pcr_path.parent.mkdir(parents=True, exist_ok=True)
                probe.loc[probe["model"].eq(model) & probe["fold"].eq(fold)].to_csv(probe_path, index=False)
                pcr.loc[pcr["model"].eq(model) & pcr["fold"].eq(fold)].to_csv(pcr_path, index=False)
                if model != "G0":
                    history_path = root / "metrics" / "training" / f"{model.lower()}_final" / f"fold_{fold}.csv"
                    history_path.parent.mkdir(parents=True, exist_ok=True)
                    history.loc[
                        history["model"].eq(model) & history["fold"].eq(fold)
                    ].to_csv(history_path, index=False)
        config = AnalysisConfig(
            prediction_root=root / "predictions",
            history_root=root / "metrics" / "training",
            checkpoint_root=root / "checkpoints",
            metric_dir=root / "metrics" / "selftest_analysis",
            figure_dir=root / "figures" / "selftest_analysis",
            bootstrap_replicates=25,
            seed=123,
            allow_partial=True,
            expected_probe_patients=50,
            expected_pcr_patients=50,
        )
        summary = run_analysis(config)
        if summary["figures"] != 12:
            raise AssertionError("self-test 未生成 12 张图")
        if len(list(config.figure_dir.glob("*.png"))) != 12:
            raise AssertionError("self-test 图文件数错误")
        paired = pd.read_csv(config.metric_dir / "paired_differences_bootstrap_ci.csv")
        if set(paired["comparison"]) != {"G3-G1", "G4-G2"}:
            raise AssertionError("self-test paired comparisons 缺失")
        decision = json.loads((config.metric_dir / "decision.json").read_text(encoding="utf-8"))
        evidence = json.loads(
            (config.metric_dir / "analysis_acceptance_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        if decision.get("decision") != "UNAVAILABLE" or decision.get("formal_analysis") is not False:
            raise AssertionError("partial self-test 错误地产生正式 GO gate")
        if evidence.get("eligible_for_independent_acceptance_verifier") is not False:
            raise AssertionError("partial self-test 错误标记为 acceptance-eligible")
        refused = False
        try:
            run_analysis(config)
        except FileExistsError:
            refused = True
        if not refused:
            raise AssertionError("聚合器未默认拒绝覆盖")
        return {
            "status": "self-test passed",
            "synthetic_probe_rows": len(probe),
            "synthetic_pcr_rows": len(pcr),
            "figures_verified": 12,
            "paired_comparisons_verified": True,
            "partial_never_emits_formal_decision_verified": True,
            "analysis_acceptance_interface_verified": True,
            "default_refuse_overwrite_verified": True,
            "temporary_outputs_removed": True,
        }
