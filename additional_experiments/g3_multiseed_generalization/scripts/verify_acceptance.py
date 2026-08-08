#!/usr/bin/env python3
"""独立复算 G3 multi-seed 正式聚合、R1–R4、资产闭环与公开隐私验收。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta, binomtest, pearsonr, spearmanr, t as student_t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, Ridge
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgrs.analysis import (  # noqa: E402
    CLASSIFICATION_METRICS,
    DECISION_POINTS,
    EXPECTED_ANALYSIS_SEED,
    EXPECTED_CONDITIONAL_REPLICATES,
    EXPECTED_CROSSED_REPLICATES,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_PLAN_SHA256,
    EXPECTED_PCR_ROWS,
    EXPECTED_PROBE_ROWS,
    FOLDS,
    MODELS,
    REGRESSION_METRICS,
    SEEDS,
    TIMEPOINTS,
    TRANSITIONS,
    _validate_freeze_provenance as _analysis_validate_freeze_provenance,
    file_sha256,
    run_self_test,
    variance_decomposition_self_test,
)
from dgrs.config import json_sha256, load_config  # noqa: E402
from dgrs.features import extraction_implementation_sha256  # noqa: E402
from dgrs.pcr import pcr_implementation_sha256  # noqa: E402
from dgrs.probes import probe_implementation_sha256  # noqa: E402


REQUIRED_TABLES = (
    "training_stability_seed_fold.csv",
    "probe_seed_cell_metrics.csv",
    "probe_seed_fold_cell_metrics.csv",
    "seed_fold_effects.csv",
    "seed_level_robustness.csv",
    "fold_level_robustness.csv",
    "seed_uncertainty.csv",
    "conditional_seed_bootstrap_ci.csv",
    "crossed_bootstrap_ci.csv",
    "leave_one_out_sensitivity.csv",
    "variance_decomposition.csv",
    "pcr_secondary_seed_metrics.csv",
    "pcr_seed_model_metrics.csv",
    "decision_gates.csv",
    "input_manifest.csv",
    "history_manifest.csv",
    "selection_manifest.csv",
    "prediction_manifest.csv",
    "figure_manifest.csv",
    "downstream_selection_audit.csv",
    "coverage.csv",
    "issues.csv",
)


FROZEN_PLAN_SHA256 = EXPECTED_PLAN_SHA256
EXPECTED_GRID = {
    (seed, model, fold) for seed in SEEDS for model in MODELS for fold in FOLDS
}
PROBE_FALSE_FLAGS = (
    "test_used_for_target_transform",
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_alpha_selection",
)
PCR_FALSE_FLAGS = (
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_hyperparameter_selection",
    "test_used_for_threshold_selection",
)
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
PCR_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RAW_FEATURE_ORDER = ("ftv", "sphericity", "ld", "bpe")
REQUIRED_FIGURES = (
    "01_static_seed_conditional_ci.png",
    "02_dynamic_seed_conditional_ci.png",
    "03_base_degradation_heatmap.png",
    "04_dynamic_gain_heatmap.png",
    "05_fold3_base_degradation.png",
    "06_static_gain_distribution.png",
    "07_dynamic_gain_distribution.png",
    "08_pcr_secondary_auroc.png",
    "09_variance_decomposition.png",
    "10_fold_level_mean_sd.png",
    "11_selected_epoch_representation_std.png",
)
REPORT_QUESTIONS = (
    "静态 FTV 改善是否跨训练种子可重复",
    "观测 ΔFTV 改善是否跨训练种子可重复",
    "上一轮 fold 3 失败是否重复",
    "不稳定性主要来自哪里",
    "正式类别",
    "是否值得作为 Factorized Grounded Response State 基础",
    "下一步应扩展监督目标，还是先解决优化问题",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ABSOLUTE_PATH_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s\"'=:(]))(?:/(?:home|data|mnt|scratch|Users)/|[A-Za-z]:[\\/])",
    re.MULTILINE,
)
PATIENT_VALUE_RE = re.compile(r"(?i)\b(?:ACRIN-6698-\d+|ISPY[-_ ]?\d*[-_ ]?\d{4,})\b")
SECRET_RE = re.compile(
    r"(?i)(?:-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    r"\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|(?:api[_-]?key|secret|token)\s*[:=]\s*[\"']?[A-Za-z0-9_./+-]{16,})"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层非 object: {path}")
    return payload


def _resolve_public_path(value: str) -> Path:
    text_value = str(value)
    path = Path(text_value)
    if path.is_absolute() or "\\" in text_value or ".." in path.parts:
        raise ValueError(f"公开 manifest 禁止绝对路径: {value}")
    if text_value.startswith("<external:"):
        raise ValueError(f"本实验正式资产不应是 external path: {value}")
    resolved = (REPO / path).resolve()
    if REPO.resolve() not in resolved.parents and resolved != REPO.resolve():
        raise ValueError(f"manifest path 逃逸 repository: {value}")
    if str(resolved.relative_to(REPO.resolve())) != text_value:
        raise ValueError(f"manifest path 非 canonical repo-relative path: {value}")
    return resolved


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "evidence": evidence}


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    if frame.columns.duplicated().any():
        raise ValueError(f"{label} 含重复列名")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} 缺列: {missing}")


def _strict_bool_value(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value) in (0.0, 1.0):
            return bool(int(value))
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1"}:
            return True
        if normalised in {"false", "0"}:
            return False
    raise ValueError(f"{label} 含非法布尔值: {value!r}")


def _strict_bool(series: pd.Series, label: str, *, allow_na: bool = False) -> pd.Series:
    parsed: list[Any] = []
    for index, value in series.items():
        if pd.isna(value):
            if allow_na:
                parsed.append(pd.NA)
                continue
            raise ValueError(f"{label}[{index}] 布尔值缺失")
        parsed.append(_strict_bool_value(value, f"{label}[{index}]"))
    return pd.Series(parsed, index=series.index, dtype="boolean" if allow_na else bool)


def _strict_int(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} 含缺失/非有限整数")
    rounded = np.rint(numeric.to_numpy(dtype=float))
    if not np.array_equal(numeric.to_numpy(dtype=float), rounded):
        raise ValueError(f"{label} 含非整数")
    return pd.Series(rounded.astype(np.int64), index=series.index)


def _finite_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} 含 NaN/Inf 或非数值")


def _require_sha(values: Iterable[Any], label: str) -> None:
    for index, value in enumerate(values):
        if not SHA256_RE.fullmatch(str(value).strip().lower()):
            raise ValueError(f"{label}[{index}] 非 64 位 SHA-256")


def _normalise_transition(value: Any) -> str:
    return (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("->", "→")
        .replace("–", "→")
        .replace("—", "→")
        .replace("-", "→")
    )


def _normalise_decision(value: Any) -> str:
    return (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("→", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _grid_from_frame(frame: pd.DataFrame, label: str) -> set[tuple[int, str, int]]:
    _require_columns(frame, ("seed_base", "model", "fold"), label)
    seed = _strict_int(frame["seed_base"], f"{label}.seed_base")
    fold = _strict_int(frame["fold"], f"{label}.fold")
    model = frame["model"].astype(str).str.upper()
    keys = list(zip(seed, model, fold, strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} seed×model×fold 键重复")
    return set(keys)


def _require_grid(frame: pd.DataFrame, label: str) -> None:
    observed = _grid_from_frame(frame, label)
    if observed != EXPECTED_GRID or len(frame) != len(EXPECTED_GRID):
        raise ValueError(
            f"{label} 不是冻结 5×2×5 唯一网格: 缺={sorted(EXPECTED_GRID-observed)} "
            f"多={sorted(observed-EXPECTED_GRID)} rows={len(frame)}"
        )


def _key_value(value: Any) -> Any:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_scalar(
    actual: Any, expected: Any, label: str, *, tolerance: float = 1e-10
) -> None:
    if isinstance(expected, (bool, np.bool_)):
        if _strict_bool_value(actual, label) is not bool(expected):
            raise ValueError(f"{label} 不一致: {actual!r} != {expected!r}")
        return
    if pd.isna(expected):
        if not pd.isna(actual):
            raise ValueError(f"{label} 应为空: {actual!r}")
        return
    if isinstance(expected, (int, float, np.integer, np.floating)) and not isinstance(
        expected, (bool, np.bool_)
    ):
        try:
            observed = float(actual)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} 非数值: {actual!r}") from exc
        if math.isnan(float(expected)) and math.isnan(observed):
            return
        if not math.isclose(
            observed, float(expected), rel_tol=tolerance, abs_tol=tolerance
        ):
            raise ValueError(f"{label} 不一致: {observed} != {expected}")
        return
    if str(actual) != str(expected):
        raise ValueError(f"{label} 不一致: {actual!r} != {expected!r}")


def _compare_table(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: Sequence[str],
    label: str,
    columns: Sequence[str] | None = None,
    *,
    tolerance: float = 1e-10,
) -> None:
    compared = list(columns or expected.columns)
    _require_columns(actual, set(keys) | set(compared), label)
    _require_columns(expected, set(keys) | set(compared), f"expected {label}")

    def rows_by_key(frame: pd.DataFrame) -> dict[tuple[Any, ...], pd.Series]:
        output: dict[tuple[Any, ...], pd.Series] = {}
        for _, row in frame.iterrows():
            key = tuple(_key_value(row[name]) for name in keys)
            if key in output:
                raise ValueError(f"{label} 键重复: {key}")
            output[key] = row
        return output

    observed_rows, expected_rows = rows_by_key(actual), rows_by_key(expected)
    if set(observed_rows) != set(expected_rows):
        raise ValueError(
            f"{label} 键覆盖不一致: 缺={sorted(set(expected_rows)-set(observed_rows), key=str)} "
            f"多={sorted(set(observed_rows)-set(expected_rows), key=str)}"
        )
    for key, expected_row in expected_rows.items():
        actual_row = observed_rows[key]
        for column in compared:
            _compare_scalar(
                actual_row[column],
                expected_row[column],
                f"{label}{key}.{column}",
                tolerance=tolerance,
            )


def _fixed_asset_paths() -> dict[str, dict[tuple[int, str, int], str]]:
    specs: dict[str, dict[tuple[int, str, int], str]] = {
        name: {}
        for name in (
            "checkpoint",
            "resolved_run",
            "feature",
            "feature_metadata",
            "feature_fragment",
            "training_history",
            "training_selection",
            "probe_selection",
            "probe_selection_summary",
            "pcr_selection",
            "pcr_selection_summary",
            "probe_prediction",
            "pcr_prediction",
        )
    }
    for seed, model, fold in sorted(EXPECTED_GRID):
        key = (seed, model, fold)
        checkpoint_dir = (
            ROOT
            / "checkpoints"
            / "formal"
            / f"seed_{seed}"
            / model.lower()
            / f"fold_{fold}"
        )
        feature_dir = ROOT / "features" / f"seed_{seed}" / model / f"fold_{fold}"
        history = (
            ROOT
            / "metrics"
            / "training"
            / "formal"
            / f"seed_{seed}"
            / model.lower()
            / f"fold_{fold}.csv"
        )
        specs["checkpoint"][key] = _relative(checkpoint_dir / "best.pt")
        specs["resolved_run"][key] = _relative(checkpoint_dir / "resolved_run.json")
        specs["feature"][key] = _relative(feature_dir / "observed_features.npz")
        specs["feature_metadata"][key] = _relative(
            feature_dir / "extraction_metadata.json"
        )
        specs["feature_fragment"][key] = _relative(
            feature_dir / "feature_manifest_fragment.csv"
        )
        specs["training_history"][key] = _relative(history)
        specs["training_selection"][key] = _relative(checkpoint_dir / "selection.json")
        for token, directory in (
            ("probe", "representation_probes"),
            ("pcr", "pcr_readouts"),
        ):
            selection_dir = (
                ROOT / "metrics" / directory / f"seed_{seed}" / model / f"fold_{fold}"
            )
            prediction_dir = (
                ROOT
                / "predictions"
                / directory
                / f"seed_{seed}"
                / model
                / f"fold_{fold}"
            )
            specs[f"{token}_selection"][key] = _relative(
                selection_dir / "selection_records.csv"
            )
            specs[f"{token}_selection_summary"][key] = _relative(
                selection_dir / "summary.json"
            )
            specs[f"{token}_prediction"][key] = _relative(
                prediction_dir / "test_predictions.csv"
            )
    return specs


def _verify_grid_manifest(
    frame: pd.DataFrame,
    label: str,
    expected: Mapping[str, Mapping[tuple[int, str, int], str]],
) -> dict[tuple[str, int, str, int], pd.Series]:
    _require_columns(
        frame,
        ("kind", "path", "sha256", "bytes", "seed_base", "model", "fold"),
        label,
    )
    if frame["path"].astype(str).duplicated().any():
        raise ValueError(f"{label} path 重复")
    observed_kinds = set(frame["kind"].astype(str))
    if observed_kinds != set(expected):
        raise ValueError(
            f"{label} kind 覆盖错误: 缺={sorted(set(expected)-observed_kinds)} "
            f"多={sorted(observed_kinds-set(expected))}"
        )
    indexed: dict[tuple[str, int, str, int], pd.Series] = {}
    for kind, expected_paths in expected.items():
        part = frame.loc[frame["kind"].astype(str).eq(kind)].copy()
        _require_grid(part, f"{label}/{kind}")
        for _, row in part.iterrows():
            seed = int(row["seed_base"])
            model = str(row["model"]).upper()
            fold = int(row["fold"])
            key = (seed, model, fold)
            expected_path = expected_paths[key]
            if str(row["path"]) != expected_path:
                raise ValueError(
                    f"{label}/{kind}/{key} 非固定路径: {row['path']} != {expected_path}"
                )
            _require_sha((row["sha256"],), f"{label}/{kind}/{key}.sha256")
            size = int(_strict_int(pd.Series([row["bytes"]]), "manifest.bytes").iloc[0])
            path = _resolve_public_path(str(row["path"]))
            if not path.is_file():
                raise ValueError(f"{label}/{kind}/{key} 文件不存在")
            if path.stat().st_size != size or file_sha256(path) != str(row["sha256"]):
                raise ValueError(f"{label}/{kind}/{key} live hash/bytes 不闭合")
            indexed[(kind, seed, model, fold)] = row
    if len(indexed) != len(expected) * len(EXPECTED_GRID):
        raise ValueError(f"{label} 行数错误: {len(indexed)}")
    return indexed


def _verify_manifests(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[
    dict[tuple[str, int, str, int], pd.Series],
    dict[tuple[str, int, str, int], pd.Series],
    dict[tuple[str, int, str, int], pd.Series],
]:
    specs = _fixed_asset_paths()
    input_kinds = {
        name: specs[name]
        for name in (
            "checkpoint",
            "resolved_run",
            "feature",
            "feature_metadata",
            "feature_fragment",
            "probe_selection",
            "probe_selection_summary",
            "pcr_selection",
            "pcr_selection_summary",
            "probe_prediction",
            "pcr_prediction",
        )
    }
    input_index = _verify_grid_manifest(
        tables["input_manifest.csv"], "input_manifest", input_kinds
    )
    history_index = _verify_grid_manifest(
        tables["history_manifest.csv"],
        "history_manifest",
        {"training_history": specs["training_history"]},
    )
    _require_columns(tables["history_manifest.csv"], ("rows",), "history_manifest")
    selection_index = _verify_grid_manifest(
        tables["selection_manifest.csv"],
        "selection_manifest",
        {"training_selection": specs["training_selection"]},
    )
    prediction_index = _verify_grid_manifest(
        tables["prediction_manifest.csv"],
        "prediction_manifest",
        {
            "probe_prediction": specs["probe_prediction"],
            "pcr_prediction": specs["pcr_prediction"],
        },
    )
    for kind in ("probe_prediction", "pcr_prediction"):
        for seed, model, fold in EXPECTED_GRID:
            left = input_index[(kind, seed, model, fold)]
            right = prediction_index[(kind, seed, model, fold)]
            for column in ("path", "sha256", "bytes"):
                _compare_scalar(
                    left[column],
                    right[column],
                    f"input/prediction manifest {kind}/{seed}/{model}/{fold}.{column}",
                )
    return input_index, history_index, selection_index


def _verify_freeze_provenance() -> dict[str, Any]:
    analysis_evidence = dict(_analysis_validate_freeze_provenance())
    plan = ROOT / "EXPERIMENT_PLAN.md"
    plan_hash = file_sha256(plan)
    if plan_hash != FROZEN_PLAN_SHA256:
        raise ValueError(f"冻结 EXPERIMENT_PLAN SHA 漂移: {plan_hash}")
    plan_freeze = _read_json(ROOT / "PLAN_FREEZE.json")
    required_plan = {
        "schema_version": 1,
        "status": "frozen",
        "plan_file": _relative(plan),
        "plan_sha256": FROZEN_PLAN_SHA256,
        "frozen_before_formal_training": True,
        "formal_training_started": False,
    }
    for key, expected in required_plan.items():
        _compare_scalar(plan_freeze.get(key), expected, f"PLAN_FREEZE.{key}")
    _require_sha((plan_freeze.get("goal_objective_sha256"),), "PLAN_FREEZE.goal")

    source_freeze = _read_json(ROOT / "SOURCE_FREEZE.json")
    if (
        source_freeze.get("schema_version") != 1
        or source_freeze.get("status") != "frozen before formal training"
        or source_freeze.get("scope") != "formal_training_only"
        or source_freeze.get("formal_training_started_at_freeze") is not False
        or source_freeze.get("plan_sha256") != FROZEN_PLAN_SHA256
    ):
        raise ValueError("SOURCE_FREEZE 顶层 contract 漂移")
    files = source_freeze.get("source_files")
    if not isinstance(files, dict) or int(
        source_freeze.get("source_file_count", -1)
    ) != len(files):
        raise ValueError("SOURCE_FREEZE source_files/schema 错误")
    for relative, expected_hash in files.items():
        if Path(str(relative)).is_absolute() or ".." in Path(str(relative)).parts:
            raise ValueError(f"SOURCE_FREEZE 非 portable path: {relative}")
        path = REPO / str(relative)
        _require_sha((expected_hash,), f"SOURCE_FREEZE.{relative}")
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"SOURCE_FREEZE live hash 漂移: {relative}")
    analysis_evidence["frozen_source_files"] = len(files)
    return analysis_evidence


def _all_tensors_finite(value: Any) -> tuple[bool, int]:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()), 1
    if isinstance(value, Mapping):
        states = [_all_tensors_finite(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        states = [_all_tensors_finite(item) for item in value]
    else:
        return True, 0
    return all(state for state, _ in states), sum(count for _, count in states)


def _validate_state_dict_mirrors(
    checkpoint: Mapping[str, Any], label: str
) -> tuple[Mapping[str, torch.Tensor], tuple[tuple[str, tuple[int, ...], str], ...]]:
    state_dict = checkpoint.get("state_dict")
    model_state = checkpoint.get("model_state")
    if (
        not isinstance(state_dict, Mapping)
        or not isinstance(model_state, Mapping)
        or not state_dict
        or set(state_dict) != set(model_state)
        or any(not torch.is_tensor(value) for value in state_dict.values())
        or any(not torch.is_tensor(value) for value in model_state.values())
    ):
        raise ValueError(f"{label} state_dict/model_state schema/key 错误")
    for tensor_name in state_dict:
        state_tensor = state_dict[tensor_name]
        model_tensor = model_state[tensor_name]
        if (
            tuple(state_tensor.shape) != tuple(model_tensor.shape)
            or state_tensor.dtype != model_tensor.dtype
            or not torch.equal(state_tensor, model_tensor)
        ):
            raise ValueError(f"{label} state_dict/model_state.{tensor_name} 不一致")
    signature = tuple(
        sorted(
            (str(name), tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in state_dict.items()
        )
    )
    return state_dict, signature


def _hash_ordered_rows(*arrays: Sequence[Any]) -> str:
    rows = zip(*(map(str, values) for values in arrays), strict=True)
    return hashlib.sha256(
        "\n".join("\t".join(row) for row in rows).encode("utf-8")
    ).hexdigest()


def _patient_hash(values: Iterable[Any]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    ).hexdigest()


def _ordered_patient_hash(values: Iterable[Any]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _patient_label_hash(patient_ids: Iterable[Any], labels: Iterable[Any]) -> str:
    pairs = zip(patient_ids, labels, strict=True)
    return hashlib.sha256(
        "\n".join(f"{patient_id}\t{int(label)}" for patient_id, label in pairs).encode(
            "utf-8"
        )
    ).hexdigest()


def _json_vector(value: Any, expected_size: int, label: str) -> np.ndarray:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 非合法 JSON vector") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != expected_size
        or any(isinstance(item, (bool, np.bool_)) for item in payload)
    ):
        raise ValueError(f"{label} vector 长度/schema 错误")
    try:
        array = np.asarray(payload, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} vector 非数值") from exc
    if array.shape != (expected_size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} vector 含 NaN/Inf")
    return array


def _compare_vector(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    tolerance: float = 1e-10,
) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or not np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    ):
        difference = (
            math.inf
            if actual.shape != expected.shape
            else float(np.max(np.abs(actual - expected), initial=0.0))
        )
        raise ValueError(f"{label} 独立逐值重算不一致: max_abs={difference}")


def _standard_scaler_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 1 or not np.isfinite(matrix).all():
        raise ValueError("live StandardScaler train matrix 非 finite 2-D")
    mean = matrix.mean(axis=0)
    variance = matrix.var(axis=0)
    scale = np.sqrt(variance)
    epsilon = np.finfo(np.float64).eps
    upper_bound = len(matrix) * epsilon * variance + (len(matrix) * mean * epsilon) ** 2
    scale[variance <= upper_bound] = 1.0
    return mean, scale


def _load_raw_targets_live(path: Path) -> tuple[dict[str, np.ndarray], str]:
    frame = pd.read_csv(path, dtype={"patient_id": str}, low_memory=False)
    required = {"patient_id", "transition", "start_visit", "end_visit"}
    for feature in RAW_FEATURE_ORDER:
        required.update({f"{feature}_start", f"{feature}_end", f"{feature}_valid"})
    _require_columns(frame, required, "live raw radiomics targets")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["transition"] = frame["transition"].map(_normalise_transition)
    if (
        len(frame) != 375 * len(TRANSITIONS)
        or frame["patient_id"].nunique() != 375
        or frame.duplicated(["patient_id", "transition"]).any()
    ):
        raise ValueError("live raw radiomics target patient/transition coverage 错误")
    output: dict[str, np.ndarray] = {}
    for patient_id, group in frame.groupby("patient_id", sort=False):
        if set(group["transition"]) != set(TRANSITIONS) or len(group) != len(
            TRANSITIONS
        ):
            raise ValueError(f"raw target {patient_id} transition 不完整")
        indexed = group.set_index("transition").reindex(TRANSITIONS)
        values = np.full((3, len(RAW_FEATURE_ORDER), 3), np.nan, dtype=np.float64)
        for transition_index, transition in enumerate(TRANSITIONS):
            row = indexed.loc[transition]
            start_visit, end_visit = transition.split("→")
            if (
                str(row["start_visit"]) != start_visit
                or str(row["end_visit"]) != end_visit
            ):
                raise ValueError(f"raw target {patient_id}/{transition} endpoint 错位")
            for feature_index, feature in enumerate(RAW_FEATURE_ORDER):
                valid = _strict_bool_value(
                    row[f"{feature}_valid"],
                    f"raw target {patient_id}/{transition}/{feature}.valid",
                )
                endpoints = np.asarray(
                    [row[f"{feature}_start"], row[f"{feature}_end"]],
                    dtype=np.float64,
                )
                if valid and not np.isfinite(endpoints).all():
                    raise ValueError(
                        f"raw target {patient_id}/{transition}/{feature} valid 但非 finite"
                    )
                values[transition_index, feature_index] = (
                    endpoints[0],
                    endpoints[1],
                    float(valid),
                )
        for transition_index in (0, 1):
            both = values[transition_index, :, 2].astype(bool) & values[
                transition_index + 1, :, 2
            ].astype(bool)
            if not np.allclose(
                values[transition_index, both, 1],
                values[transition_index + 1, both, 0],
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(
                    f"raw target {patient_id} shared visit endpoint 不一致"
                )
        output[str(patient_id)] = values
    digest = hashlib.sha256()
    for patient_id in sorted(output):
        digest.update(patient_id.encode("utf-8"))
        digest.update(np.asarray(output[patient_id], dtype="<f8").tobytes())
    return output, digest.hexdigest()


def _probe_transform_specs(
    static_payload: Mapping[str, Any],
    change_payload: Mapping[str, Any],
    *,
    fold: int,
    train_patient_hash: str,
    raw_targets_hash: str,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    shared_ok = all(
        payload.get("fold") == fold
        and payload.get("train_patient_hash") == train_patient_hash
        and payload.get("raw_targets_sha256") == raw_targets_hash
        and payload.get("feature_order") == list(RAW_FEATURE_ORDER)
        and payload.get("quantiles") == [0.01, 0.99]
        for payload in (static_payload, change_payload)
    )
    if not shared_ok:
        raise ValueError(f"fold={fold} target transform fold/train/raw/schema 漂移")
    if (
        static_payload.get("schema_version") != 1
        or static_payload.get("timepoints") != list(TIMEPOINTS)
        or change_payload.get("spec_version")
        != "adjacent_v2_ftv_ld_logepsilon_sphericity_bpe_absolute_winsor01_99_robust"
    ):
        raise ValueError(f"fold={fold} target transform version/timepoint 漂移")
    static_items = static_payload.get("specs")
    change_items = change_payload.get("features")
    if not isinstance(static_items, list) or not isinstance(change_items, list):
        raise ValueError(f"fold={fold} target transform specs 非 list")
    static_specs = {
        (str(item.get("timepoint")), str(item.get("feature_name"))): item
        for item in static_items
        if isinstance(item, dict)
    }
    change_specs = {
        str(item.get("name")): item for item in change_items if isinstance(item, dict)
    }
    if set(static_specs) != {
        (timepoint, feature)
        for timepoint in TIMEPOINTS
        for feature in RAW_FEATURE_ORDER
    } or set(change_specs) != set(RAW_FEATURE_ORDER):
        raise ValueError(f"fold={fold} target transform cell 覆盖错误")
    spec_rows = [(f"static/{cell}", spec) for cell, spec in static_specs.items()] + [
        (f"change/{name}", spec) for name, spec in change_specs.items()
    ]
    for label, spec in spec_rows:
        numeric = {}
        for name in ("epsilon", "winsor_low", "winsor_high", "center", "scale"):
            try:
                numeric[name] = float(spec[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{label}.{name} 缺失/非数值") from exc
        if (
            not np.isfinite(list(numeric.values())).all()
            or numeric["scale"] <= 0
            or numeric["winsor_low"] > numeric["winsor_high"]
            or str(spec.get("value_transform"))
            not in {"log_epsilon", "log1p", "identity"}
        ):
            raise ValueError(f"{label} transform 参数非法")
    return static_specs, change_specs


def _target_analysis_value(values: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    transform = str(spec["value_transform"])
    if transform == "log_epsilon":
        shifted = values + float(spec["epsilon"])
        if np.any(shifted <= 0):
            raise ValueError("probe live target log_epsilon 非正")
        return np.log(shifted)
    if transform == "log1p":
        if np.any(values <= -1):
            raise ValueError("probe live target log1p 越界")
        return np.log1p(values)
    if transform == "identity":
        return values
    raise ValueError(f"probe live target transform 非法: {transform}")


def _probe_target_live(
    raw_targets: Mapping[str, np.ndarray],
    patient_id: str,
    task: str,
    cell: str,
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, bool, str]:
    raw = raw_targets.get(str(patient_id))
    if raw is None:
        return math.nan, math.nan, False, ""
    feature = "ftv"
    feature_index = RAW_FEATURE_ORDER.index(feature)
    if task == "static":
        timepoint_index = TIMEPOINTS.index(cell)
        values = np.stack(
            (raw[0, :, 0], raw[0, :, 1], raw[1, :, 1], raw[2, :, 1]), axis=0
        )
        valid = np.stack(
            (
                raw[0, :, 2].astype(bool),
                raw[0, :, 2].astype(bool) & raw[1, :, 2].astype(bool),
                raw[1, :, 2].astype(bool) & raw[2, :, 2].astype(bool),
                raw[2, :, 2].astype(bool),
            ),
            axis=0,
        ) & np.isfinite(values)
        if not valid[timepoint_index, feature_index]:
            return math.nan, math.nan, False, ""
        natural = float(values[timepoint_index, feature_index])
        spec = static_specs[(cell, feature)]
        analysis = float(_target_analysis_value(np.asarray([natural]), spec)[0])
        clipped = float(
            np.clip(analysis, float(spec["winsor_low"]), float(spec["winsor_high"]))
        )
        standardised = (clipped - float(spec["center"])) / float(spec["scale"])
        label = f"static:{spec['value_transform']}+winsor01_99+median_iqr:{cell}"
        return float(standardised), natural, True, label
    transition_index = TRANSITIONS.index(cell)
    endpoints = np.asarray(raw[transition_index, feature_index], dtype=np.float64)
    if not bool(endpoints[2]) or not np.isfinite(endpoints[:2]).all():
        return math.nan, math.nan, False, ""
    spec = change_specs[feature]
    transformed = _target_analysis_value(endpoints[:2], spec)
    natural = float(transformed[1] - transformed[0])
    clipped = float(
        np.clip(natural, float(spec["winsor_low"]), float(spec["winsor_high"]))
    )
    standardised = (clipped - float(spec["center"])) / float(spec["scale"])
    label = f"delta:{spec['value_transform']}_difference+winsor01_99+median_iqr"
    return float(standardised), natural, True, label


def _prepare_probe_split_live(
    split: str,
    task: str,
    cell: str,
    patient_ids: np.ndarray,
    splits: np.ndarray,
    response: np.ndarray,
    raw_targets: Mapping[str, np.ndarray],
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output_ids: list[str] = []
    features: list[np.ndarray] = []
    y_standardised: list[float] = []
    y_natural: list[float] = []
    transform_labels: set[str] = set()
    cell_index = TIMEPOINTS.index(cell) if task == "static" else TRANSITIONS.index(cell)
    for index in np.flatnonzero(splits == split):
        patient_id = str(patient_ids[index])
        standardised, natural, valid, label = _probe_target_live(
            raw_targets,
            patient_id,
            task,
            cell,
            static_specs,
            change_specs,
        )
        if not valid:
            continue
        feature = (
            response[index, cell_index]
            if task == "static"
            else response[index, cell_index + 1] - response[index, cell_index]
        )
        if feature.shape != (192,) or not np.isfinite(feature).all():
            raise ValueError(f"probe live feature 非法: {patient_id}/{task}/{cell}")
        output_ids.append(patient_id)
        features.append(feature)
        y_standardised.append(standardised)
        y_natural.append(natural)
        transform_labels.add(label)
    if not features or len(transform_labels) != 1:
        raise ValueError(f"probe live {split}/{task}/{cell} rows/transform 非法")
    return {
        "patient_ids": output_ids,
        "matrix": np.stack(features).astype(np.float64, copy=False),
        "y_standardised": np.asarray(y_standardised, dtype=np.float64),
        "y_natural": np.asarray(y_natural, dtype=np.float64),
        "target_transform": next(iter(transform_labels)),
    }


def _inverse_probe_prediction_live(
    values: np.ndarray,
    task: str,
    cell: str,
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    spec = static_specs[(cell, "ftv")] if task == "static" else change_specs["ftv"]
    analysis = values * float(spec["scale"]) + float(spec["center"])
    if task == "change":
        return analysis
    transform = str(spec["value_transform"])
    if transform == "log_epsilon":
        return np.exp(analysis) - float(spec["epsilon"])
    if transform == "log1p":
        return np.expm1(analysis)
    return analysis


def _response_readout_live(response: np.ndarray, decision_point: str) -> np.ndarray:
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 3 or response.shape[1:] != (4, 192):
        raise ValueError(f"pCR live response_state shape 非法: {response.shape}")
    if decision_point == "T0":
        matrix = response[:, 0]
    elif decision_point == "T0-T1":
        r0, r1 = response[:, 0], response[:, 1]
        matrix = np.concatenate((r0, r1, r1 - r0), axis=1)
    elif decision_point == "T0-T2":
        r0, r1, r2 = response[:, 0], response[:, 1], response[:, 2]
        matrix = np.concatenate((r0, r1, r2, r1 - r0, r2 - r1, r2 - r0), axis=1)
    else:
        raise ValueError(f"pCR decision point 非法: {decision_point}")
    if not np.isfinite(matrix).all():
        raise ValueError("pCR live readout matrix 含 NaN/Inf")
    return matrix


def _sigmoid_live(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _youden_live(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float, float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError("pCR live validation Youden 输入非法")
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    rows: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        sensitivity = float((predicted & (labels == 1)).sum() / positives)
        specificity = float(((~predicted) & (labels == 0)).sum() / negatives)
        rows.append(
            (
                float(threshold),
                sensitivity + specificity - 1.0,
                sensitivity,
                specificity,
            )
        )
    best = max(row[1] for row in rows)
    eligible = [row for row in rows if row[1] >= best - 1e-12]
    return min(eligible, key=lambda row: (abs(row[0] - 0.5), row[0]))


def _audit_assets(
    tables: Mapping[str, pd.DataFrame],
    input_index: Mapping[tuple[str, int, str, int], pd.Series],
    history_index: Mapping[tuple[str, int, str, int], pd.Series],
    selection_index: Mapping[tuple[str, int, str, int], pd.Series],
) -> tuple[pd.DataFrame, dict[tuple[int, str, int], dict[str, Any]]]:
    required_history = {
        "epoch",
        "seed_base",
        "fold",
        "effective_seed",
        "model",
        "lambda_ftv",
        "total_loss",
        "base_loss",
        "ftv_loss",
        "weighted_ftv_loss",
        "val_base_loss",
        "val_ftv_metric",
        "representation_std",
        "noncollapse",
        "base_gate_pass",
        "checkpoint_eligible",
        "validation_selection_metric",
        "is_selected_checkpoint",
        "val_grounded_patients",
    }
    expected_architecture = {
        "backbone_input": "DCE7",
        "image_channels": 7,
        "first_conv_in_channels": 7,
        "roi_mask_backbone_input": False,
        "pooling": "gap",
        "roi_mask_use": "absent",
        "observed_response_state": "online_preprojector_r",
        "response_dim": 192,
        "transition": "M0_direct_next_state_causal_transformer",
        "ftv_is_forward_input": False,
    }
    assets: dict[tuple[int, str, int], dict[str, Any]] = {}
    stability_rows: list[dict[str, Any]] = []
    feature_fingerprints: dict[tuple[int, str, int], str] = {}
    checkpoint_contracts: dict[tuple[int, str, int], dict[str, Any]] = {}
    canonical_folds: dict[int, pd.DataFrame] = {}
    fold_manifest_paths: set[Path] = set()
    current_extractor_sha = extraction_implementation_sha256()

    for seed, model, fold in sorted(EXPECTED_GRID):
        key = (seed, model, fold)
        expected_resolved_config = load_config(ROOT / "configs" / f"seed_{seed}.yaml")
        expected_resolved_config = dict(expected_resolved_config)
        expected_resolved_config["model"] = dict(expected_resolved_config["model"])
        expected_resolved_config["loss"] = dict(expected_resolved_config["loss"])
        expected_resolved_config["train"] = dict(expected_resolved_config["train"])
        expected_resolved_config["model"]["model_name"] = model
        expected_resolved_config["loss"]["lambda_ftv"] = 0.0 if model == "G1" else 0.25
        expected_resolved_config_sha = json_sha256(expected_resolved_config)
        checkpoint_path = _resolve_public_path(
            str(input_index[("checkpoint", seed, model, fold)]["path"])
        )
        resolved_path = _resolve_public_path(
            str(input_index[("resolved_run", seed, model, fold)]["path"])
        )
        feature_path = _resolve_public_path(
            str(input_index[("feature", seed, model, fold)]["path"])
        )
        metadata_path = _resolve_public_path(
            str(input_index[("feature_metadata", seed, model, fold)]["path"])
        )
        fragment_path = _resolve_public_path(
            str(input_index[("feature_fragment", seed, model, fold)]["path"])
        )
        history_path = _resolve_public_path(
            str(history_index[("training_history", seed, model, fold)]["path"])
        )
        training_selection_path = _resolve_public_path(
            str(selection_index[("training_selection", seed, model, fold)]["path"])
        )
        expected_lambda = 0.0 if model == "G1" else 0.25

        history = pd.read_csv(history_path, low_memory=False)
        _require_columns(history, required_history, f"history {key}")
        if int(history_index[("training_history", seed, model, fold)]["rows"]) != len(
            history
        ):
            raise ValueError(f"history manifest rows 未闭合: {key}")
        if not 1 <= len(history) <= 12:
            raise ValueError(f"history {key} epoch 行数不在 1..12")
        epochs = _strict_int(history["epoch"], f"history {key}.epoch")
        if list(epochs) != list(range(1, len(history) + 1)):
            raise ValueError(f"history {key} epoch 不是连续 1..N")
        for column, expected in (
            ("seed_base", seed),
            ("fold", fold),
            ("effective_seed", seed + fold),
        ):
            values = _strict_int(history[column], f"history {key}.{column}")
            if set(values) != {expected}:
                raise ValueError(f"history {key}.{column} contract 漂移")
        if set(history["model"].astype(str).str.upper()) != {model}:
            raise ValueError(f"history {key}.model 漂移")
        lambda_values = pd.to_numeric(history["lambda_ftv"], errors="coerce")
        if not lambda_values.eq(expected_lambda).all():
            raise ValueError(f"history {key}.lambda_FTV 未锁定")
        _finite_columns(
            history,
            (
                "total_loss",
                "base_loss",
                "ftv_loss",
                "weighted_ftv_loss",
                "val_base_loss",
                "val_ftv_metric",
                "representation_std",
                "validation_selection_metric",
                "val_grounded_patients",
            ),
            f"history {key}",
        )
        boolean_history = {
            column: _strict_bool(history[column], f"history {key}.{column}")
            for column in (
                "noncollapse",
                "base_gate_pass",
                "checkpoint_eligible",
                "is_selected_checkpoint",
            )
        }
        if int(boolean_history["is_selected_checkpoint"].sum()) != 1:
            raise ValueError(f"history {key} 必须恰有一个 selected epoch")
        for column in history.columns:
            lowered = column.lower()
            if lowered.startswith("test_") and (
                "used" in lowered or "selection" in lowered
            ):
                if _strict_bool(history[column], f"history {key}.{column}").any():
                    raise ValueError(f"history {key} 声称 test 参与训练/选择")
        selected = history.loc[boolean_history["is_selected_checkpoint"]].iloc[0]

        selection = _read_json(training_selection_path)
        if (
            selection.get("schema_version") != 1
            or selection.get("test_data_used") is not False
            or int(selection.get("seed_base", -1)) != seed
            or int(selection.get("effective_seed", -1)) != seed + fold
            or int(selection.get("fold", -1)) != fold
            or str(selection.get("model_name", "")).upper() != model
            or int(selection.get("selected_epoch", -1)) != int(selected["epoch"])
        ):
            raise ValueError(f"training selection {key} schema/seed/test contract 错误")
        for json_name, history_name in (
            ("selected_validation_base_loss", "val_base_loss"),
            ("selected_validation_ftv_loss", "val_ftv_metric"),
            ("selected_representation_std", "representation_std"),
        ):
            _compare_float(
                float(selection[json_name]),
                float(selected[history_name]),
                f"selection/history {key}.{json_name}",
            )
        selection_epochs = selection.get("epochs")
        if not isinstance(selection_epochs, list) or len(selection_epochs) != len(
            history
        ):
            raise ValueError(f"training selection {key}.epochs schema/rows 错误")
        for index, epoch_payload in enumerate(selection_epochs):
            if not isinstance(epoch_payload, dict):
                raise ValueError(f"training selection {key}.epochs[{index}] 非 object")
            history_row = history.iloc[index]
            for name, column in (
                ("epoch", "epoch"),
                ("seed_base", "seed_base"),
                ("fold", "fold"),
                ("effective_seed", "effective_seed"),
                ("val_base_loss", "val_base_loss"),
                ("val_ftv_loss", "val_ftv_metric"),
                ("val_representation_std", "representation_std"),
            ):
                _compare_scalar(
                    epoch_payload.get(name),
                    history_row[column],
                    f"selection/history {key}.epochs[{index}].{name}",
                )
            for name, column in (
                ("noncollapse", "noncollapse"),
                ("base_gate_pass", "base_gate_pass"),
                ("eligible", "checkpoint_eligible"),
            ):
                _compare_scalar(
                    epoch_payload.get(name),
                    bool(boolean_history[column].iloc[index]),
                    f"selection/history {key}.epochs[{index}].{name}",
                )
        baseline_contract = selection.get("baseline_selection_contract")
        if not isinstance(baseline_contract, dict):
            raise ValueError(f"training selection {key} 缺 baseline contract")
        if baseline_contract.get("base_metric") != (
            "validation_normalized_next_state_loss_without_sigreg"
        ):
            raise ValueError(f"training selection {key} baseline metric 漂移")
        expected_selection_rule = (
            "non-collapse epoch with minimum validation normalized next-state loss"
            if model == "G1"
            else "non-collapse and <=5% paired-baseline normalized next-state loss "
            "degradation; minimum validation FTV loss"
        )
        if (
            selection.get("selection_rule") != expected_selection_rule
            or selection.get("fallback_rule")
            != "minimum normalized-next-state gate violation, then minimum validation "
            "FTV loss among non-collapse finite epochs"
        ):
            raise ValueError(f"training selection {key} rule 文本漂移")
        if model == "G1":
            if any(
                baseline_contract.get(name) is not None
                for name in (
                    "paired_model",
                    "baseline_checkpoint",
                    "baseline_checkpoint_sha256",
                    "baseline_val_base_loss",
                    "maximum_relative_degradation",
                    "allowed_val_base_loss",
                )
            ):
                raise ValueError(f"G1 training selection {key} 含伪 baseline pairing")
        else:
            baseline_asset = assets[(seed, "G1", fold)]
            baseline_path = _resolve_public_path(
                str(input_index[("checkpoint", seed, "G1", fold)]["path"])
            )
            expected_baseline_loss = float(baseline_asset["val_state_loss"])
            if (
                baseline_contract.get("paired_model") != "G1"
                or Path(str(baseline_contract.get("baseline_checkpoint"))).resolve()
                != baseline_path
                or baseline_contract.get("baseline_checkpoint_sha256")
                != baseline_asset["checkpoint_sha256"]
            ):
                raise ValueError(f"G3 training selection {key} baseline asset 未闭合")
            _compare_scalar(
                baseline_contract.get("baseline_val_base_loss"),
                expected_baseline_loss,
                f"training selection {key}.baseline_val_base_loss",
            )
            _compare_scalar(
                baseline_contract.get("maximum_relative_degradation"),
                0.05,
                f"training selection {key}.maximum_relative_degradation",
            )
            _compare_scalar(
                baseline_contract.get("allowed_val_base_loss"),
                expected_baseline_loss * 1.05,
                f"training selection {key}.allowed_val_base_loss",
            )
        representation_std = history["representation_std"].to_numpy(dtype=float)
        val_base_loss = history["val_base_loss"].to_numpy(dtype=float)
        val_ftv_metric = history["val_ftv_metric"].to_numpy(dtype=float)
        val_grounded = history["val_grounded_patients"].to_numpy(dtype=float)
        expected_noncollapse = np.isfinite(representation_std) & (
            representation_std >= 0.05
        )
        expected_base_gate = (
            np.ones(len(history), dtype=bool)
            if model == "G1"
            else val_base_loss <= float(baseline_contract["allowed_val_base_loss"])
        )
        ftv_finite = np.isfinite(val_ftv_metric) & (val_grounded > 0)
        expected_eligible = (
            expected_noncollapse
            & expected_base_gate
            & (ftv_finite if model == "G3" else True)
        )
        expected_metric = val_ftv_metric if model == "G3" else val_base_loss
        for name, expected_values in (
            ("noncollapse", expected_noncollapse),
            ("base_gate_pass", expected_base_gate),
            ("checkpoint_eligible", expected_eligible),
        ):
            if not np.array_equal(
                boolean_history[name].to_numpy(dtype=bool), expected_values
            ):
                raise ValueError(
                    f"history {key}.{name} 未按 live validation 指标机械重建"
                )
        _compare_vector(
            history["validation_selection_metric"].to_numpy(dtype=float),
            expected_metric,
            f"history {key}.validation_selection_metric",
        )
        best_metric = math.inf
        fallback_metric = (math.inf, math.inf)
        simulated_selected: int | None = None
        simulated_fallback: int | None = None
        stale = 0
        for index, row in history.iterrows():
            improved = False
            epoch = int(row["epoch"])
            if expected_eligible[index] and expected_metric[index] < best_metric:
                best_metric = float(expected_metric[index])
                simulated_selected = epoch
                improved = True
            if expected_noncollapse[index] and ftv_finite[index]:
                violation = (
                    max(
                        0.0,
                        val_base_loss[index]
                        - float(baseline_contract["allowed_val_base_loss"]),
                    )
                    if model == "G3"
                    else 0.0
                )
                candidate = (violation, float(expected_metric[index]))
                if candidate < fallback_metric:
                    fallback_metric = candidate
                    simulated_fallback = epoch
                    if simulated_selected is None:
                        improved = True
            stale = 0 if improved else stale + 1
            if stale >= 4 and index != history.index[-1]:
                raise ValueError(f"history {key} early-stop 后仍有 epoch")
        if len(history) < 12 and stale < 4:
            raise ValueError(f"history {key} 在 patience=4 前提前停止")
        expected_mode = "primary"
        expected_epoch = simulated_selected
        if expected_epoch is None:
            expected_mode = "fallback_base_gate_failed"
            expected_epoch = simulated_fallback
            if expected_eligible.any():
                raise ValueError(f"history {key} fallback 但存在 eligible epoch")
        if (
            expected_epoch is None
            or selection.get("selection_mode") != expected_mode
            or int(selected["epoch"]) != expected_epoch
        ):
            raise ValueError(
                f"training selection {key} early-stop/selection 重建不一致"
            )
        eligible = expected_eligible
        metric = expected_metric
        if eligible.any():
            simple_expected_epoch = int(
                history.iloc[
                    int(np.flatnonzero(eligible)[np.argmin(metric[eligible])])
                ]["epoch"]
            )
            if (
                selection.get("selection_mode") != "primary"
                or int(selected["epoch"]) != simple_expected_epoch
            ):
                raise ValueError(
                    f"training selection {key} 未按 validation-only primary rule"
                )
        else:
            if (
                model != "G3"
                or selection.get("selection_mode") != "fallback_base_gate_failed"
            ):
                raise ValueError(f"training selection {key} 非法 fallback")
            allowed = float(baseline_contract.get("allowed_val_base_loss", math.nan))
            candidates: list[tuple[float, float, int]] = []
            for index, row in history.iterrows():
                if (
                    bool(boolean_history["noncollapse"].loc[index])
                    and math.isfinite(float(row["val_ftv_metric"]))
                    and float(row["val_grounded_patients"]) > 0
                ):
                    candidates.append(
                        (
                            max(0.0, float(row["val_base_loss"]) - allowed),
                            float(row["validation_selection_metric"]),
                            int(row["epoch"]),
                        )
                    )
            if not candidates or int(selected["epoch"]) != min(candidates)[2]:
                raise ValueError(
                    f"training selection {key} 未按 validation-only fallback rule"
                )

        resolved = _read_json(resolved_path)
        if (
            int(resolved.get("seed_base", -1)) != seed
            or int(resolved.get("effective_seed", -1)) != seed + fold
            or int(resolved.get("fold", -1)) != fold
            or str(resolved.get("model_name", "")).upper() != model
            or float(resolved.get("lambda_ftv", math.nan)) != expected_lambda
            or resolved.get("experiment_plan_sha256") != FROZEN_PLAN_SHA256
            or resolved.get("smoke_patients") is not None
            or int(resolved.get("epochs_requested", -1)) != 12
        ):
            raise ValueError(f"resolved_run {key} contract 漂移")

        checkpoint_sha = file_sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint {key} 顶层非 mapping")
        tensors_finite, tensor_count = _all_tensors_finite(checkpoint)
        if not tensors_finite or tensor_count <= 0:
            raise ValueError(f"checkpoint {key} tensor nonfinite/empty")
        if (
            checkpoint.get("schema_version") != 2
            or checkpoint.get("finalized") is not True
            or int(checkpoint.get("seed_base", -1)) != seed
            or int(checkpoint.get("effective_seed", -1)) != seed + fold
            or int(checkpoint.get("fold", -1)) != fold
            or str(checkpoint.get("model_name", "")).upper() != model
            or int(checkpoint.get("epoch", -1)) != int(selected["epoch"])
            or checkpoint.get("plan_sha256") != FROZEN_PLAN_SHA256
            or checkpoint.get("history_sha256") != file_sha256(history_path)
            or checkpoint.get("selection_sha256")
            != file_sha256(training_selection_path)
            or checkpoint.get("resolved_config_sha256") != expected_resolved_config_sha
        ):
            raise ValueError(f"checkpoint {key} finalized/schema/hash closure 失败")
        if checkpoint.get("baseline_selection_contract") != baseline_contract:
            raise ValueError(f"checkpoint/selection {key} baseline contract 不一致")
        selected_metrics = checkpoint.get("selected_epoch_metrics")
        if not isinstance(selected_metrics, dict):
            raise ValueError(f"checkpoint {key} 缺 selected_epoch_metrics")
        for checkpoint_name, history_name in (
            ("epoch", "epoch"),
            ("seed_base", "seed_base"),
            ("fold", "fold"),
            ("effective_seed", "effective_seed"),
            ("val_base_loss", "val_base_loss"),
            ("val_ftv_loss", "val_ftv_metric"),
            ("val_representation_std", "representation_std"),
        ):
            _compare_scalar(
                selected_metrics.get(checkpoint_name),
                selected[history_name],
                f"checkpoint/history {key}.{checkpoint_name}",
            )
        selected_index = history.index[boolean_history["is_selected_checkpoint"]][0]
        for checkpoint_name, history_name in (
            ("noncollapse", "noncollapse"),
            ("base_gate_pass", "base_gate_pass"),
            ("eligible", "checkpoint_eligible"),
        ):
            _compare_scalar(
                selected_metrics.get(checkpoint_name),
                bool(boolean_history[history_name].loc[selected_index]),
                f"checkpoint/history {key}.{checkpoint_name}",
            )
        state_dict, state_signature = _validate_state_dict_mirrors(
            checkpoint, f"checkpoint {key}"
        )
        state_tensor_count = len(state_dict)
        if state_tensor_count <= 0 or tensor_count < state_tensor_count:
            raise ValueError(f"checkpoint {key} tensor_count/state tensors 非法")
        first_conv = state_dict.get("encoder.features.0.main.0.weight")
        ftv_keys = {name for name in state_dict if str(name).startswith("ftv_head.")}
        if (
            not torch.is_tensor(first_conv)
            or tuple(first_conv.shape) != (16, 7, 3, 3, 3)
            or (model == "G1" and ftv_keys)
            or (model == "G3" and ftv_keys != {"ftv_head.weight", "ftv_head.bias"})
        ):
            raise ValueError(
                f"checkpoint {key} live DCE7/FTV-head tensor architecture 错误"
            )
        loss_config = dict(checkpoint.get("loss_config", {}))
        if float(loss_config.get("lambda_ftv", math.nan)) != expected_lambda:
            raise ValueError(f"checkpoint {key} lambda_FTV 漂移")
        architecture = dict(checkpoint.get("architecture_contract", {}))
        if any(
            architecture.get(name) != value
            for name, value in expected_architecture.items()
        ):
            raise ValueError(f"checkpoint {key} architecture/transition contract 漂移")
        expected_head = None if model == "G1" else "Linear(response_dim,1)"
        if architecture.get("ftv_head") != expected_head:
            raise ValueError(f"checkpoint {key} FTV head contract 漂移")
        forbidden = set(architecture.get("forbidden_inputs_absent", ()))
        if not {
            "clinical",
            "treatment",
            "radiomics",
            "mask_geometry",
            "voxel_count",
            "explicit_volume",
        }.issubset(forbidden):
            raise ValueError(f"checkpoint {key} forbidden-input contract 不完整")
        data_contract = dict(checkpoint.get("data_contract", {}))
        if data_contract.get("fold_manifest_sha256") != EXPECTED_FOLD_MANIFEST_SHA256:
            raise ValueError(f"checkpoint {key} fold manifest SHA 漂移")
        fold_manifest_path = Path(str(data_contract.get("fold_manifest", ""))).resolve()
        if (
            not fold_manifest_path.is_file()
            or file_sha256(fold_manifest_path) != EXPECTED_FOLD_MANIFEST_SHA256
        ):
            raise ValueError(f"checkpoint {key} canonical fold manifest live SHA 失败")
        fold_manifest_paths.add(fold_manifest_path)
        if fold not in canonical_folds:
            canonical = pd.read_csv(
                fold_manifest_path, dtype={"patient_id": str}, low_memory=False
            )
            _require_columns(
                canonical,
                ("patient_id", "fold", "split", "label_pcr"),
                "fold manifest",
            )
            canonical = canonical.loc[
                _strict_int(canonical["fold"], "fold manifest.fold").eq(fold)
            ].reset_index(drop=True)
            if (
                len(canonical) != 808
                or canonical["patient_id"].duplicated().any()
                or set(canonical["split"].astype(str)) != {"train", "val", "test"}
            ):
                raise ValueError(f"canonical fold={fold} patient closure 错误")
            canonical = pd.concat(
                [
                    canonical.loc[canonical["split"].astype(str).eq(split)]
                    for split in ("train", "val", "test")
                ],
                ignore_index=True,
            )
            canonical_folds[fold] = canonical
        canonical = canonical_folds[fold]
        for split, contract_name in (
            ("train", "train_patient_hash"),
            ("val", "val_patient_hash"),
            ("test", "test_patient_hash"),
        ):
            expected_hash = _patient_hash(
                canonical.loc[
                    canonical["split"].astype(str).eq(split), "patient_id"
                ].astype(str)
            )
            if data_contract.get(contract_name) != expected_hash:
                raise ValueError(
                    f"checkpoint {key} {contract_name} 未闭合 canonical manifest"
                )
        split_hashes = checkpoint.get("split_hashes")
        if not isinstance(split_hashes, dict) or set(split_hashes) != {
            "train",
            "val",
            "test",
            "pretrain_train",
        }:
            raise ValueError(f"checkpoint {key}.split_hashes schema 漂移")
        _require_sha(split_hashes.values(), f"checkpoint {key}.split_hashes")
        for split in ("train", "val", "test"):
            if split_hashes[split] != data_contract.get(f"{split}_patient_hash"):
                raise ValueError(
                    f"checkpoint {key}.split_hashes.{split} 未闭合 data_contract"
                )
        _require_sha(
            (
                data_contract.get("extra_pretrain_patient_hash"),
                data_contract.get("raw_ftv_sha256"),
            ),
            f"checkpoint {key}.data_contract",
        )
        checkpoint_contracts[key] = {
            name: checkpoint.get(name)
            for name in (
                "shared_initialization_sha256",
                "split_hashes",
                "ftv_transform_sha256",
                "data_contract",
                "model_config",
                "train_config",
                "loss_config",
                "baseline_selection_contract",
                "implementation_sha256",
            )
        }
        transform_path = ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
        if (
            not transform_path.is_file()
            or checkpoint.get("ftv_transform_sha256") != file_sha256(transform_path)
            or resolved.get("ftv_transform_sha256") != file_sha256(transform_path)
        ):
            raise ValueError(f"checkpoint/resolved FTV transform hash 未闭合: {key}")
        del checkpoint
        gc.collect()

        feature_sha = file_sha256(feature_path)
        metadata = _read_json(metadata_path)
        coverage = metadata.get("coverage")
        if (
            metadata.get("schema_version") != 2
            or metadata.get("status")
            != "frozen observed response-state extraction complete"
            or int(metadata.get("seed_base", -1)) != seed
            or int(metadata.get("effective_seed", -1)) != seed + fold
            or int(metadata.get("fold", -1)) != fold
            or str(metadata.get("model", "")).upper() != model
            or Path(str(metadata.get("checkpoint"))).resolve() != checkpoint_path
            or metadata.get("feature_file_sha256") != feature_sha
            or Path(str(metadata.get("feature_file"))).resolve() != feature_path
            or Path(str(metadata.get("metadata_file"))).resolve() != metadata_path
            or metadata.get("checkpoint_sha256") != checkpoint_sha
            or metadata.get("checkpoint_schema_version") != 2
            or metadata.get("checkpoint_epoch") != int(selected["epoch"])
            or metadata.get("checkpoint_plan_sha256") != FROZEN_PLAN_SHA256
            or metadata.get("checkpoint_training_implementation_sha256")
            != checkpoint_contracts[key]["implementation_sha256"]
            or metadata.get("checkpoint_resolved_config_sha256")
            != expected_resolved_config_sha
            or metadata.get("checkpoint_split_hashes") != split_hashes
            or metadata.get("shared_initialization_sha256")
            != checkpoint_contracts[key]["shared_initialization_sha256"]
            or metadata.get("extractor_sha256") != current_extractor_sha
            or metadata.get("fold_manifest_sha256") != EXPECTED_FOLD_MANIFEST_SHA256
            or Path(str(metadata.get("fold_manifest"))).resolve() != fold_manifest_path
            or Path(str(metadata.get("ftv_transform_path"))).resolve()
            != transform_path.resolve()
            or metadata.get("ftv_transform_sha256") != file_sha256(transform_path)
            or Path(str(metadata.get("manifest_file"))).resolve() != fragment_path
            or metadata.get("feature_shape") != [808, 4, 192]
            or metadata.get("canonical_manifest_rows_verified") is not True
            or metadata.get("canonical_label_rows_verified") is not True
            or metadata.get("measurement_targets_read_during_extraction") is not False
            or metadata.get("world_model_trained_or_finetuned") is not False
            or metadata.get("checkpoint_cache_contract_match") is not True
            or metadata.get("online_encoder") is not True
            or metadata.get("ftv_head_loaded_but_not_called") is not (model == "G3")
            or metadata.get("inference_inputs") != ["DCE7"]
            or metadata.get(
                "pcr_labels_attached_from_locked_manifest_for_downstream_only"
            )
            is not True
            or not isinstance(coverage, dict)
            or coverage.get("formal_complete") is not True
            or coverage.get("expected_primary_patients") != 808
            or coverage.get("observed_primary_patients") != 808
            or coverage.get("all_four_visits_present") is not True
            or coverage.get("response_rows_finite") is not True
            or coverage.get("patient_ids_unique") is not True
            or metadata.get("max_patients_per_split") is not None
        ):
            raise ValueError(
                f"feature metadata {key} schema/source/formal closure 失败"
            )
        if metadata.get("manifest_file_sha256") != file_sha256(fragment_path):
            raise ValueError(f"feature metadata {key} fragment SHA 未闭合")
        with np.load(feature_path, allow_pickle=False) as archive:
            required_arrays = {
                "patient_ids",
                "splits",
                "response_state",
                "timepoints",
                "model",
                "seed_base",
                "fold",
                "effective_seed",
                "label_pcr",
            }
            if required_arrays.difference(archive.files):
                raise ValueError(f"feature {key} NPZ schema 缺失")
            patient_ids = archive["patient_ids"].astype(str)
            splits = archive["splits"].astype(str)
            response = archive["response_state"]
            labels = archive["label_pcr"]
            if (
                response.shape != (808, 4, 192)
                or response.dtype != np.float32
                or not np.isfinite(response).all()
                or patient_ids.shape != (808,)
                or len(set(patient_ids)) != 808
                or splits.shape != (808,)
                or labels.shape != (808,)
                or labels.dtype.kind not in {"i", "u"}
                or not set(labels.astype(int)).issubset({0, 1})
                or tuple(archive["timepoints"].astype(str)) != TIMEPOINTS
                or str(archive["model"].reshape(()).item()).upper() != model
                or int(archive["seed_base"].reshape(()).item()) != seed
                or int(archive["effective_seed"].reshape(()).item()) != seed + fold
                or int(archive["fold"].reshape(()).item()) != fold
            ):
                raise ValueError(
                    f"feature {key} [808,4,192]/finite/canonical schema 失败"
                )
            expected_split_counts = {
                "train": 525 if fold < 3 else 526,
                "val": 121,
                "test": 162 if fold < 3 else 161,
            }
            observed_split_counts = pd.Series(splits).value_counts().to_dict()
            if observed_split_counts != expected_split_counts:
                raise ValueError(
                    f"feature {key} split closure 错误: {observed_split_counts}"
                )
            canonical = canonical_folds[fold]
            if (
                patient_ids.tolist() != canonical["patient_id"].astype(str).tolist()
                or splits.tolist() != canonical["split"].astype(str).tolist()
                or not np.array_equal(
                    labels.astype(int),
                    _strict_int(canonical["label_pcr"], "canonical label").to_numpy(),
                )
            ):
                raise ValueError(f"feature {key} 未逐行闭合 canonical fold manifest")
            canonical_order_hash = _ordered_patient_hash(patient_ids)
            canonical_label_hash = _patient_label_hash(patient_ids, labels.astype(int))
            if (
                metadata.get("canonical_patient_order_sha256") != canonical_order_hash
                or metadata.get("canonical_patient_label_sha256")
                != canonical_label_hash
            ):
                raise ValueError(f"feature metadata {key} canonical hash 未独立闭合")
            fingerprint = _hash_ordered_rows(patient_ids, splits, labels.astype(int))
            feature_fingerprints[key] = fingerprint

        fragment = pd.read_csv(fragment_path, low_memory=False)
        _require_columns(
            fragment,
            (
                "patient_id",
                "seed_base",
                "fold",
                "effective_seed",
                "split",
                "model",
                "patient_index",
                "label_pcr",
                "visits",
                "representation",
                "feature_dim",
                "feature_file_sha256",
                "source_checkpoint_sha256",
                "fold_manifest_sha256",
            ),
            f"feature fragment {key}",
        )
        if (
            len(fragment) != 808
            or fragment["patient_id"].astype(str).duplicated().any()
        ):
            raise ValueError(f"feature fragment {key} patient coverage 错误")
        if int(input_index[("feature_fragment", seed, model, fold)]["rows"]) != 808:
            raise ValueError(f"feature fragment manifest rows 未闭合: {key}")
        if not np.array_equal(fragment["patient_id"].astype(str), patient_ids):
            raise ValueError(f"feature fragment {key} patient order 未闭合")
        if not np.array_equal(
            fragment["split"].astype(str), splits
        ) or not np.array_equal(
            _strict_int(fragment["label_pcr"], "feature label"), labels.astype(int)
        ):
            raise ValueError(f"feature fragment {key} split/label 未闭合")
        if not np.array_equal(
            _strict_int(fragment["patient_index"], "feature patient_index"),
            np.arange(808),
        ):
            raise ValueError(f"feature fragment {key} patient_index 错误")
        if (
            set(fragment["model"].astype(str).str.upper()) != {model}
            or set(_strict_int(fragment["seed_base"], "fragment seed")) != {seed}
            or set(_strict_int(fragment["effective_seed"], "fragment effective"))
            != {seed + fold}
            or set(_strict_int(fragment["fold"], "fragment fold")) != {fold}
            or set(_strict_int(fragment["visits"], "fragment visits")) != {4}
            or set(_strict_int(fragment["feature_dim"], "fragment dim")) != {192}
            or set(fragment["representation"].astype(str)) != {"response_state"}
            or set(fragment["feature_file_sha256"].astype(str)) != {feature_sha}
            or set(fragment["source_checkpoint_sha256"].astype(str)) != {checkpoint_sha}
            or set(fragment["fold_manifest_sha256"].astype(str))
            != {EXPECTED_FOLD_MANIFEST_SHA256}
        ):
            raise ValueError(f"feature fragment {key} schema/hash closure 错误")

        std = float(selection["selected_representation_std"])
        assets[key] = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "feature_path": feature_path,
            "feature_sha256": feature_sha,
            "extractor_sha256": current_extractor_sha,
            "canonical_patient_order_sha256": canonical_order_hash,
            "canonical_patient_label_sha256": canonical_label_hash,
            "shared_initialization_sha256": str(
                checkpoint_contracts[key]["shared_initialization_sha256"]
            ),
            "val_state_loss": float(selection["selected_validation_base_loss"]),
            "selected_epoch": int(selection["selected_epoch"]),
            "representation_std": std,
            "tensor_count": tensor_count,
            "state_tensor_count": state_tensor_count,
            "state_signature": state_signature,
            "fingerprint": fingerprint,
            "test_patient_ids": tuple(patient_ids[splits == "test"]),
            "test_labels": {
                str(patient_id): int(label)
                for patient_id, label, split in zip(
                    patient_ids, labels.astype(int), splits, strict=True
                )
                if split == "test"
            },
            "train_patient_hash": _patient_hash(patient_ids[splits == "train"]),
        }
        stability_rows.append(
            {
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "model": model,
                "selected_epoch": int(selection["selected_epoch"]),
                "selection_mode": str(selection["selection_mode"]),
                "val_state_loss": float(selection["selected_validation_base_loss"]),
                "val_ftv_loss": float(selection["selected_validation_ftv_loss"]),
                "representation_std": std,
                "lambda_ftv": expected_lambda,
                "selected_scalars_finite": True,
                "checkpoint_tensors_finite": True,
                "checkpoint_tensor_count": tensor_count,
                "feature_finite": True,
                "no_collapse": bool(std >= 0.05),
                "architecture_contract_verified": True,
                "test_data_used": False,
                "checkpoint_sha256": checkpoint_sha,
                "feature_sha256": feature_sha,
                "shared_initialization_sha256": assets[key][
                    "shared_initialization_sha256"
                ],
                "paired_baseline": "G1" if model == "G3" else pd.NA,
                "base_degradation_fraction": pd.NA,
                "base_pass": pd.NA,
            }
        )

    for fold in FOLDS:
        fingerprints = {
            feature_fingerprints[(seed, model, fold)]
            for seed in SEEDS
            for model in MODELS
        }
        if len(fingerprints) != 1:
            raise ValueError(
                f"feature canonical patient/split/label 跨 seed/model 漂移: fold={fold}"
            )
    signatures: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
    for model in MODELS:
        observed = {
            assets[(seed, model, fold)]["state_signature"]
            for seed in SEEDS
            for fold in FOLDS
        }
        if len(observed) != 1:
            raise ValueError(f"{model} state_dict key/shape/dtype 跨 seed/fold 漂移")
        signatures[model] = {
            name: (shape, dtype) for name, shape, dtype in next(iter(observed))
        }
    if (
        set(signatures["G1"]).difference(signatures["G3"])
        or set(signatures["G3"]).difference(signatures["G1"])
        != {"ftv_head.weight", "ftv_head.bias"}
        or any(
            signatures["G1"][name] != signatures["G3"][name]
            for name in signatures["G1"]
        )
    ):
        raise ValueError("G1/G3 state_dict architecture 差异不等于唯一 FTV head")
    for seed in SEEDS:
        for fold in FOLDS:
            g1 = checkpoint_contracts[(seed, "G1", fold)]
            g3 = checkpoint_contracts[(seed, "G3", fold)]
            for name in (
                "shared_initialization_sha256",
                "split_hashes",
                "ftv_transform_sha256",
            ):
                if g1[name] != g3[name]:
                    raise ValueError(f"G1/G3 {name} 不一致: seed={seed}/fold={fold}")
            for name in (
                "fold_manifest_sha256",
                "train_patient_hash",
                "val_patient_hash",
                "test_patient_hash",
                "extra_pretrain_patient_hash",
                "raw_ftv_sha256",
            ):
                if dict(g1["data_contract"]).get(name) != dict(g3["data_contract"]).get(
                    name
                ):
                    raise ValueError(
                        f"G1/G3 data_contract.{name} 不一致: seed={seed}/fold={fold}"
                    )
            for config_name, ignored in (
                ("model_config", {"model_name", "direct_ftv_grounding"}),
                ("loss_config", {"lambda_ftv"}),
                ("train_config", set()),
            ):
                left, right = dict(g1[config_name]), dict(g3[config_name])
                for name in ignored:
                    left.pop(name, None)
                    right.pop(name, None)
                if left != right:
                    raise ValueError(
                        f"G1/G3 common {config_name} 不一致: seed={seed}/fold={fold}"
                    )
            baseline = dict(g3.get("baseline_selection_contract", {}))
            if (
                baseline.get("baseline_checkpoint_sha256")
                != assets[(seed, "G1", fold)]["checkpoint_sha256"]
            ):
                raise ValueError(
                    f"G3 paired baseline checkpoint hash 未闭合: seed={seed}/fold={fold}"
                )
        implementations = {
            checkpoint_contracts[(seed, model, fold)]["implementation_sha256"]
            for model in MODELS
            for fold in FOLDS
        }
        if len(implementations) != 1:
            raise ValueError(f"training implementation SHA 在 seed={seed} 内漂移")
    for fold in FOLDS:
        initialisations = {
            assets[(seed, "G1", fold)]["shared_initialization_sha256"] for seed in SEEDS
        }
        if len(initialisations) <= 1:
            raise ValueError(f"不同 seed initialization 全相同: fold={fold}")
    if len(fold_manifest_paths) != 1:
        raise ValueError("50 checkpoints canonical fold manifest path 不唯一")
    implementation_hashes = {
        checkpoint_contracts[key]["implementation_sha256"] for key in EXPECTED_GRID
    }
    if len(implementation_hashes) != 1:
        raise ValueError("50 checkpoints training implementation SHA 不唯一")

    expected_stability = pd.DataFrame(stability_rows)
    for seed in SEEDS:
        for fold in FOLDS:
            g1_loss = assets[(seed, "G1", fold)]["val_state_loss"]
            g3_loss = assets[(seed, "G3", fold)]["val_state_loss"]
            degradation = (g3_loss - g1_loss) / max(g1_loss, 1e-12)
            mask = (
                expected_stability["seed_base"].eq(seed)
                & expected_stability["fold"].eq(fold)
                & expected_stability["model"].eq("G3")
            )
            expected_stability.loc[mask, "base_degradation_fraction"] = degradation
            expected_stability.loc[mask, "base_pass"] = bool(
                math.isfinite(degradation) and degradation <= 0.05 + 1e-12
            )
    actual_stability = tables["training_stability_seed_fold.csv"]
    _require_grid(actual_stability, "training_stability_seed_fold")
    _compare_table(
        actual_stability,
        expected_stability,
        ("seed_base", "model", "fold"),
        "training_stability_seed_fold",
        tuple(expected_stability.columns),
    )
    return expected_stability, assets


def _audit_downstream(
    tables: Mapping[str, pd.DataFrame],
    input_index: Mapping[tuple[str, int, str, int], pd.Series],
    assets: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, int, str, int], pd.DataFrame],
]:
    expected_rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, int, str, int], pd.DataFrame] = {}
    implementation_hashes = {
        "probe": probe_implementation_sha256(),
        "pcr": pcr_implementation_sha256(),
    }
    for seed, model, fold in sorted(EXPECTED_GRID):
        asset = assets[(seed, model, fold)]
        for token in ("probe", "pcr"):
            selection_path = _resolve_public_path(
                str(input_index[(f"{token}_selection", seed, model, fold)]["path"])
            )
            summary_path = _resolve_public_path(
                str(
                    input_index[(f"{token}_selection_summary", seed, model, fold)][
                        "path"
                    ]
                )
            )
            prediction_path = _resolve_public_path(
                str(input_index[(f"{token}_prediction", seed, model, fold)]["path"])
            )
            expected_feature_path = _resolve_public_path(
                str(input_index[("feature", seed, model, fold)]["path"])
            )
            expected_checkpoint_path = _resolve_public_path(
                str(input_index[("checkpoint", seed, model, fold)]["path"])
            )
            frame = pd.read_csv(selection_path, low_memory=False)
            summary = _read_json(summary_path)
            if int(
                input_index[(f"{token}_selection", seed, model, fold)]["rows"]
            ) != len(frame):
                raise ValueError(
                    f"{token} selection manifest rows 未闭合: {seed}/{model}/{fold}"
                )
            common = {
                "seed_base",
                "fold",
                "effective_seed",
                "model",
                "source_feature_file",
                "source_feature_sha256",
                "feature_extractor_sha256",
                "source_checkpoint",
                "source_checkpoint_sha256",
                "fold_manifest_sha256",
                "canonical_patient_order_sha256",
                "canonical_patient_label_sha256",
            }
            if token == "probe":
                required = common | {
                    "task",
                    "timepoint",
                    "transition",
                    "representation",
                    "input_variant",
                    "target",
                    "feature_dim",
                    "n_train",
                    "n_val",
                    "n_test",
                    "selected_alpha",
                    "val_mse_standardized",
                    "alpha_validation_mse_json",
                    "ridge_solver",
                    "ridge_tol",
                    "ridge_max_iter",
                    "ridge_fit_intercept",
                    "ridge_intercept",
                    "ridge_coef_json",
                    "feature_scaler_mean_json",
                    "feature_scaler_scale_json",
                    "feature_scaler_n_samples_seen",
                    "train_target_mean_standardized",
                    "b0_val_mse_standardized",
                    "static_transform_file",
                    "static_transform_sha256",
                    "change_transform_file",
                    "change_transform_sha256",
                    "raw_target_file",
                    "raw_target_file_sha256",
                    "raw_targets_sha256",
                    "test_prediction_guard_enforced",
                    "test_predict_call_count",
                    *PROBE_FALSE_FLAGS,
                }
                _require_columns(
                    frame, required, f"probe selection {seed}/{model}/{fold}"
                )
                if len(frame) != 7:
                    raise ValueError(
                        f"probe selection {seed}/{model}/{fold} 非 7 cells"
                    )
                task = frame["task"].astype(str).str.lower()
                timepoint = frame["timepoint"].fillna("").astype(str).str.upper()
                transition = frame["transition"].fillna("").map(_normalise_transition)
                cell = np.where(task.eq("static"), timepoint, transition)
                observed = set(zip(task, cell, strict=True))
                expected_cells = {("static", value) for value in TIMEPOINTS} | {
                    ("change", value) for value in TRANSITIONS
                }
                if observed != expected_cells or len(observed) != len(frame):
                    raise ValueError(
                        f"probe selection {seed}/{model}/{fold} cell 覆盖错误"
                    )
                if (
                    set(frame["target"].astype(str).str.lower()) != {"ftv"}
                    or set(frame["representation"].astype(str).str.lower())
                    != {"response_state"}
                    or set(_strict_int(frame["feature_dim"], "probe selection dim"))
                    != {192}
                    or set(frame.loc[task.eq("static"), "input_variant"].astype(str))
                    != {"current"}
                    or set(frame.loc[task.eq("change"), "input_variant"].astype(str))
                    != {"observed_difference"}
                    or set(frame["ridge_solver"].astype(str)) != {"lsqr"}
                    or not pd.to_numeric(frame["ridge_tol"], errors="coerce")
                    .eq(1e-8)
                    .all()
                    or set(_strict_int(frame["ridge_max_iter"], "probe ridge max_iter"))
                    != {10000}
                ):
                    raise ValueError(
                        f"probe selection {seed}/{model}/{fold} protocol 漂移"
                    )
                alpha = pd.to_numeric(frame["selected_alpha"], errors="coerce")
                if alpha.isna().any() or not alpha.isin(ALPHAS).all():
                    raise ValueError(
                        f"probe selection {seed}/{model}/{fold} alpha 非冻结 grid"
                    )
                for payload in frame["alpha_validation_mse_json"]:
                    grid = json.loads(str(payload))
                    if not isinstance(grid, dict) or len(grid) != len(ALPHAS):
                        raise ValueError(
                            f"probe selection {seed}/{model}/{fold} alpha grid 不完整"
                        )
                    if {float(value) for value in grid} != set(
                        ALPHAS
                    ) or not np.isfinite(
                        np.asarray(list(grid.values()), dtype=float)
                    ).all():
                        raise ValueError(
                            f"probe selection {seed}/{model}/{fold} alpha grid 漂移"
                        )
                for _, selection_row in frame.iterrows():
                    grid = {
                        float(alpha): float(score)
                        for alpha, score in json.loads(
                            str(selection_row["alpha_validation_mse_json"])
                        ).items()
                    }
                    best_mse = min(grid.values())
                    selected_alpha = min(
                        alpha
                        for alpha, score in grid.items()
                        if score <= best_mse + 1e-12
                    )
                    if float(
                        selection_row["selected_alpha"]
                    ) != selected_alpha or not math.isclose(
                        float(selection_row["val_mse_standardized"]),
                        grid[selected_alpha],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"probe selection {seed}/{model}/{fold} 未按 validation MSE/tie rule"
                        )
                target_assets: dict[str, tuple[Path, str]] = {}
                for prefix, path_column, hash_column in (
                    (
                        "static_transform",
                        "static_transform_file",
                        "static_transform_sha256",
                    ),
                    (
                        "change_transform",
                        "change_transform_file",
                        "change_transform_sha256",
                    ),
                    ("raw_target", "raw_target_file", "raw_target_file_sha256"),
                ):
                    paths = set(frame[path_column].astype(str))
                    hashes = set(frame[hash_column].astype(str))
                    if len(paths) != 1 or len(hashes) != 1:
                        raise ValueError(
                            f"probe selection {seed}/{model}/{fold} {prefix} 非唯一"
                        )
                    source_path = Path(next(iter(paths))).resolve()
                    source_hash = next(iter(hashes))
                    _require_sha((source_hash,), f"probe {prefix} hash")
                    if (
                        not source_path.is_file()
                        or file_sha256(source_path) != source_hash
                    ):
                        raise ValueError(
                            f"probe selection {seed}/{model}/{fold} {prefix} live hash 失败"
                        )
                    target_assets[prefix] = (source_path, source_hash)
                raw_targets_hashes = set(frame["raw_targets_sha256"].astype(str))
                if len(raw_targets_hashes) != 1:
                    raise ValueError(
                        f"probe selection {seed}/{model}/{fold} raw targets hash 非唯一"
                    )
                raw_targets_hash = next(iter(raw_targets_hashes))
                _require_sha((raw_targets_hash,), "probe raw_targets_sha256")
                for transform_name in ("static_transform", "change_transform"):
                    transform = _read_json(target_assets[transform_name][0])
                    if (
                        int(transform.get("fold", -1)) != fold
                        or transform.get("train_patient_hash")
                        != asset["train_patient_hash"]
                        or transform.get("raw_targets_sha256") != raw_targets_hash
                    ):
                        raise ValueError(
                            f"probe selection {seed}/{model}/{fold} {transform_name} train/raw closure 失败"
                        )
                false_flags, true_flags, call_column, cells = (
                    PROBE_FALSE_FLAGS,
                    ("test_prediction_guard_enforced",),
                    "test_predict_call_count",
                    7,
                )
                forbidden_summary = (
                    *PROBE_FALSE_FLAGS,
                    "radiomics_used_as_input",
                    "ftv_head_output_used_as_input",
                    "transition_prediction_used_as_input",
                    "world_model_trained_or_finetuned",
                )
                expected_status = "frozen response-state Ridge probes complete"
                summary_impl = "probe_implementation_sha256"
            else:
                required = common | {
                    "decision_point",
                    "feature_schema",
                    "feature_schema_sha256",
                    "feature_dim",
                    "n_train",
                    "n_val",
                    "n_test",
                    "train_positive",
                    "val_positive",
                    "test_positive",
                    "selected_penalty",
                    "selected_C",
                    "selected_threshold",
                    "val_auroc",
                    "val_auprc",
                    "val_youden",
                    "val_sensitivity",
                    "val_specificity",
                    "grid_validation_metrics_json",
                    "selection_rule",
                    "threshold_tie_rule",
                    "solver",
                    "class_weight",
                    "max_iter",
                    "tol",
                    "random_state",
                    "logistic_intercept_json",
                    "logistic_coef_json",
                    "feature_scaler_mean_json",
                    "feature_scaler_scale_json",
                    "feature_scaler_n_samples_seen",
                    "test_feature_matrix_constructed_after_selection_lock",
                    "test_prediction_guard_enforced",
                    "test_predict_proba_call_count",
                    *PCR_FALSE_FLAGS,
                }
                _require_columns(
                    frame, required, f"pCR selection {seed}/{model}/{fold}"
                )
                frame = frame.copy()
                frame["decision_point"] = frame["decision_point"].map(
                    _normalise_decision
                )
                if len(frame) != 3 or set(frame["decision_point"]) != set(
                    DECISION_POINTS
                ):
                    raise ValueError(
                        f"pCR selection {seed}/{model}/{fold} decision 覆盖错误"
                    )
                schemas = {
                    "T0": ("r0", 192),
                    "T0-T1": ("concat(r0,r1,r1-r0)", 576),
                    "T0-T2": ("concat(r0,r1,r2,r1-r0,r2-r1,r2-r0)", 1152),
                }
                for index, point in enumerate(DECISION_POINTS):
                    row = frame.loc[frame["decision_point"].eq(point)]
                    schema, dimension = schemas[point]
                    if (
                        len(row) != 1
                        or str(row.iloc[0]["feature_schema"]) != schema
                        or str(row.iloc[0]["feature_schema_sha256"])
                        != hashlib.sha256(schema.encode("utf-8")).hexdigest()
                        or int(row.iloc[0]["feature_dim"]) != dimension
                        or int(row.iloc[0]["random_state"]) != 2026 + fold * 100 + index
                    ):
                        raise ValueError(
                            f"pCR selection {seed}/{model}/{fold}/{point} schema/RNG 错误"
                        )
                if (
                    not set(frame["selected_penalty"].astype(str)).issubset(
                        {"l1", "l2"}
                    )
                    or not pd.to_numeric(frame["selected_C"], errors="coerce")
                    .isin(PCR_C_GRID)
                    .all()
                    or not pd.to_numeric(frame["selected_threshold"], errors="coerce")
                    .between(0, 1)
                    .all()
                    or set(frame["solver"].astype(str)) != {"liblinear"}
                    or set(frame["class_weight"].astype(str)) != {"balanced"}
                    or set(_strict_int(frame["max_iter"], "pCR max_iter")) != {20000}
                    or not pd.to_numeric(frame["tol"], errors="coerce").eq(1e-7).all()
                    or set(frame["selection_rule"].astype(str))
                    != {
                        "max validation AUROC; <=1e-12 tie max AUPRC; then smaller C; "
                        "then penalty order l1,l2"
                    }
                    or set(frame["threshold_tie_rule"].astype(str))
                    != {
                        "max validation Youden J; <=1e-12 tie closest to 0.5; "
                        "then smaller threshold"
                    }
                ):
                    raise ValueError(
                        f"pCR selection {seed}/{model}/{fold} logistic contract 漂移"
                    )
                for payload in frame["grid_validation_metrics_json"]:
                    grid = json.loads(str(payload))
                    if not isinstance(grid, list) or len(grid) != 18:
                        raise ValueError(
                            f"pCR selection {seed}/{model}/{fold} C/penalty grid 不完整"
                        )
                    observed_grid = {
                        (str(item["penalty"]), float(item["C"])) for item in grid
                    }
                    expected_grid = {
                        (penalty, value)
                        for penalty in ("l1", "l2")
                        for value in PCR_C_GRID
                    }
                    if observed_grid != expected_grid:
                        raise ValueError(
                            f"pCR selection {seed}/{model}/{fold} C/penalty grid 漂移"
                        )
                    for item in grid:
                        if not isinstance(item, dict) or not {
                            "penalty",
                            "C",
                            "val_auroc",
                            "val_auprc",
                            "n_iter",
                        }.issubset(item):
                            raise ValueError(
                                f"pCR selection {seed}/{model}/{fold} grid item schema 错误"
                            )
                        metrics = np.asarray(
                            [item["C"], item["val_auroc"], item["val_auprc"]],
                            dtype=float,
                        )
                        try:
                            iteration = int(item["n_iter"])
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"pCR selection {seed}/{model}/{fold} n_iter 非整数"
                            ) from exc
                        if (
                            not np.isfinite(metrics).all()
                            or iteration != float(item["n_iter"])
                            or iteration < 0
                        ):
                            raise ValueError(
                                f"pCR selection {seed}/{model}/{fold} grid metric 非法"
                            )
                penalty_order = {"l1": 0, "l2": 1}
                for _, selection_row in frame.iterrows():
                    grid = json.loads(
                        str(selection_row["grid_validation_metrics_json"])
                    )
                    best_auroc = max(float(item["val_auroc"]) for item in grid)
                    auroc_tied = [
                        item
                        for item in grid
                        if float(item["val_auroc"]) >= best_auroc - 1e-12
                    ]
                    best_auprc = max(float(item["val_auprc"]) for item in auroc_tied)
                    metric_tied = [
                        item
                        for item in auroc_tied
                        if float(item["val_auprc"]) >= best_auprc - 1e-12
                    ]
                    chosen = min(
                        metric_tied,
                        key=lambda item: (
                            float(item["C"]),
                            penalty_order[str(item["penalty"])],
                        ),
                    )
                    if (
                        str(selection_row["selected_penalty"]) != str(chosen["penalty"])
                        or float(selection_row["selected_C"]) != float(chosen["C"])
                        or not math.isclose(
                            float(selection_row["val_auroc"]),
                            float(chosen["val_auroc"]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        or not math.isclose(
                            float(selection_row["val_auprc"]),
                            float(chosen["val_auprc"]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        raise ValueError(
                            f"pCR selection {seed}/{model}/{fold} 未按 validation AUROC/AUPRC tie rule"
                        )
                false_flags, true_flags, call_column, cells = (
                    PCR_FALSE_FLAGS,
                    (
                        "test_feature_matrix_constructed_after_selection_lock",
                        "test_prediction_guard_enforced",
                    ),
                    "test_predict_proba_call_count",
                    3,
                )
                forbidden_summary = (
                    "clinical_used",
                    "treatment_used",
                    "radiomics_used",
                    "mask_geometry_used",
                    "ground_truth_ftv_used",
                    "predicted_ftv_or_ftv_head_used",
                    "test_used_for_checkpoint_selection",
                    "test_used_for_lambda_selection",
                    "test_used_for_any_selection",
                    "world_model_trained_or_finetuned",
                )
                expected_status = "strict image-state-only pCR readouts complete"
                summary_impl = "pcr_implementation_sha256"

            if (
                set(_strict_int(frame["seed_base"], "downstream seed")) != {seed}
                or set(_strict_int(frame["fold"], "downstream fold")) != {fold}
                or set(_strict_int(frame["effective_seed"], "downstream effective"))
                != {seed + fold}
                or set(frame["model"].astype(str).str.upper()) != {model}
                or {Path(value).resolve() for value in frame["source_feature_file"]}
                != {expected_feature_path}
                or set(frame["source_feature_sha256"].astype(str))
                != {asset["feature_sha256"]}
                or set(frame["feature_extractor_sha256"].astype(str))
                != {asset["extractor_sha256"]}
                or {Path(value).resolve() for value in frame["source_checkpoint"]}
                != {expected_checkpoint_path}
                or set(frame["source_checkpoint_sha256"].astype(str))
                != {asset["checkpoint_sha256"]}
                or set(frame["fold_manifest_sha256"].astype(str))
                != {EXPECTED_FOLD_MANIFEST_SHA256}
                or set(frame["canonical_patient_order_sha256"].astype(str))
                != {asset["canonical_patient_order_sha256"]}
                or set(frame["canonical_patient_label_sha256"].astype(str))
                != {asset["canonical_patient_label_sha256"]}
            ):
                raise ValueError(
                    f"{token} selection {seed}/{model}/{fold} source/grid closure 失败"
                )
            for column in false_flags:
                if _strict_bool(frame[column], f"{token} selection.{column}").any():
                    raise ValueError(
                        f"{token} selection {seed}/{model}/{fold} 使用 test"
                    )
            for column in true_flags:
                if not _strict_bool(frame[column], f"{token} selection.{column}").all():
                    raise ValueError(
                        f"{token} selection {seed}/{model}/{fold} guard false"
                    )
            if (
                not _strict_int(frame[call_column], f"{token} selection.{call_column}")
                .eq(1)
                .all()
            ):
                raise ValueError(
                    f"{token} selection {seed}/{model}/{fold} test call count 非1"
                )

            if (
                summary.get("schema_version") != 2
                or summary.get("status") != expected_status
                or int(summary.get("seed_base", -1)) != seed
                or int(summary.get("effective_seed", -1)) != seed + fold
                or int(summary.get("fold", -1)) != fold
                or str(summary.get("model", "")).upper() != model
                or Path(str(summary.get("prediction_file"))).resolve()
                != prediction_path
                or summary.get("prediction_file_sha256") != file_sha256(prediction_path)
                or Path(str(summary.get("selection_file"))).resolve() != selection_path
                or summary.get("selection_file_sha256") != file_sha256(selection_path)
                or Path(str(summary.get("source_feature_file"))).resolve()
                != expected_feature_path
                or summary.get("source_feature_sha256") != asset["feature_sha256"]
                or Path(str(summary.get("source_checkpoint"))).resolve()
                != expected_checkpoint_path
                or summary.get("source_checkpoint_sha256") != asset["checkpoint_sha256"]
                or summary.get("fold_manifest_sha256") != EXPECTED_FOLD_MANIFEST_SHA256
                or summary.get("canonical_patient_order_sha256")
                != asset["canonical_patient_order_sha256"]
                or summary.get("canonical_patient_label_sha256")
                != asset["canonical_patient_label_sha256"]
            ):
                raise ValueError(
                    f"{token} summary {seed}/{model}/{fold} schema/hash closure 失败"
                )
            if summary.get(summary_impl) != implementation_hashes[token]:
                raise ValueError(
                    f"{token} summary {seed}/{model}/{fold} implementation live SHA 漂移"
                )
            guards = summary.get("leakage_guards")
            if not isinstance(guards, dict):
                raise ValueError(
                    f"{token} summary {seed}/{model}/{fold} leakage_guards 缺失"
                )
            for name in forbidden_summary:
                if guards.get(name) is not False:
                    raise ValueError(
                        f"{token} summary {seed}/{model}/{fold}.{name} 非 false"
                    )
            if guards.get("test_prediction_guard_enforced") is not True:
                raise ValueError(
                    f"{token} summary {seed}/{model}/{fold} test guard 非 true"
                )
            if token == "probe":
                if (
                    guards.get("test_constructed_after_alpha_lock") is not True
                    or guards.get("test_predict_calls_per_cell") != 1
                    or summary.get("targets") != ["ftv"]
                    or summary.get("probe_cells") != 7
                ):
                    raise ValueError(
                        f"probe summary {seed}/{model}/{fold} protocol 漂移"
                    )
                for prefix, path_field, hash_field in (
                    (
                        "static_transform",
                        "static_transform_file",
                        "static_transform_sha256",
                    ),
                    (
                        "change_transform",
                        "change_transform_file",
                        "change_transform_sha256",
                    ),
                    ("raw_target", "raw_target_file", "raw_target_file_sha256"),
                ):
                    if (
                        Path(str(summary.get(path_field))).resolve()
                        != target_assets[prefix][0]
                        or summary.get(hash_field) != target_assets[prefix][1]
                    ):
                        raise ValueError(
                            f"probe summary {seed}/{model}/{fold} {prefix} 未闭合 selection/live asset"
                        )
                if summary.get("raw_targets_sha256") != raw_targets_hash:
                    raise ValueError(
                        f"probe summary {seed}/{model}/{fold} raw target semantic hash 未闭合"
                    )
            else:
                if (
                    guards.get("test_feature_matrix_constructed_after_selection_lock")
                    is not True
                    or guards.get("test_predict_proba_calls_per_decision") != 1
                    or summary.get("readout_rng_base_seed") != 2026
                    or summary.get("decision_points") != list(DECISION_POINTS)
                ):
                    raise ValueError(f"pCR summary {seed}/{model}/{fold} protocol 漂移")
            selections[(token, seed, model, fold)] = frame
            expected_rows.append(
                {
                    "kind": f"{token}_selection",
                    "seed_base": seed,
                    "model": model,
                    "fold": fold,
                    "cells": cells,
                    "test_once": True,
                    "source_feature_sha256": asset["feature_sha256"],
                    "source_checkpoint_sha256": asset["checkpoint_sha256"],
                }
            )

    expected = pd.DataFrame(expected_rows)
    actual = tables["downstream_selection_audit.csv"]
    _compare_table(
        actual,
        expected,
        ("kind", "seed_base", "model", "fold"),
        "downstream_selection_audit",
        tuple(expected.columns),
    )
    return expected, selections


def _load_feature_arrays_live(
    asset: Mapping[str, Any], key: tuple[int, str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(asset["feature_path"])
    if not path.is_file() or file_sha256(path) != asset["feature_sha256"]:
        raise ValueError(f"live feature 在 prediction 复算前漂移: {key}")
    with np.load(path, allow_pickle=False) as archive:
        patient_ids = archive["patient_ids"].astype(str)
        splits = archive["splits"].astype(str)
        response = archive["response_state"].astype(np.float64)
        labels = archive["label_pcr"].astype(np.int64)
    if (
        patient_ids.shape != (808,)
        or splits.shape != (808,)
        or response.shape != (808, 4, 192)
        or labels.shape != (808,)
        or not np.isfinite(response).all()
        or _ordered_patient_hash(patient_ids) != asset["canonical_patient_order_sha256"]
        or _patient_label_hash(patient_ids, labels)
        != asset["canonical_patient_label_sha256"]
    ):
        raise ValueError(f"live feature canonical arrays 在 prediction 复算失败: {key}")
    return patient_ids, splits, response, labels


def _audit_probe_prediction_cell_live(
    part: pd.DataFrame,
    selected: pd.Series,
    *,
    key: tuple[int, str, int],
    task: str,
    cell: str,
    patient_ids: np.ndarray,
    splits: np.ndarray,
    response: np.ndarray,
    asset: Mapping[str, Any],
    raw_cache: dict[Path, tuple[dict[str, np.ndarray], str]],
    transform_cache: dict[Path, dict[str, Any]],
) -> None:
    raw_path = Path(str(selected["raw_target_file"])).resolve()
    static_path = Path(str(selected["static_transform_file"])).resolve()
    change_path = Path(str(selected["change_transform_file"])).resolve()
    if raw_path not in raw_cache:
        raw_cache[raw_path] = _load_raw_targets_live(raw_path)
    if static_path not in transform_cache:
        transform_cache[static_path] = _read_json(static_path)
    if change_path not in transform_cache:
        transform_cache[change_path] = _read_json(change_path)
    raw_targets, raw_hash = raw_cache[raw_path]
    if raw_hash != str(selected["raw_targets_sha256"]) or file_sha256(raw_path) != str(
        selected["raw_target_file_sha256"]
    ):
        raise ValueError(f"probe {key}/{task}/{cell} live raw target SHA 未闭合")
    static_specs, change_specs = _probe_transform_specs(
        transform_cache[static_path],
        transform_cache[change_path],
        fold=key[2],
        train_patient_hash=str(asset["train_patient_hash"]),
        raw_targets_hash=raw_hash,
    )
    prepared = {
        split: _prepare_probe_split_live(
            split,
            task,
            cell,
            patient_ids,
            splits,
            response,
            raw_targets,
            static_specs,
            change_specs,
        )
        for split in ("train", "val", "test")
    }
    for column, split in (("n_train", "train"), ("n_val", "val"), ("n_test", "test")):
        if int(selected[column]) != len(prepared[split]["patient_ids"]):
            raise ValueError(f"probe {key}/{task}/{cell}.{column} live count 不一致")
    train = prepared["train"]
    validation = prepared["val"]
    test = prepared["test"]
    expected_mean, expected_scale = _standard_scaler_stats(train["matrix"])
    scaler_mean = _json_vector(
        selected["feature_scaler_mean_json"], 192, "probe scaler mean"
    )
    scaler_scale = _json_vector(
        selected["feature_scaler_scale_json"], 192, "probe scaler scale"
    )
    _compare_vector(
        scaler_mean, expected_mean, f"probe {key}/{task}/{cell} scaler mean"
    )
    _compare_vector(
        scaler_scale, expected_scale, f"probe {key}/{task}/{cell} scaler scale"
    )
    if int(selected["feature_scaler_n_samples_seen"]) != len(
        train["patient_ids"]
    ) or not _strict_bool_value(
        selected["ridge_fit_intercept"], f"probe {key}/{task}/{cell}.fit_intercept"
    ):
        raise ValueError(
            f"probe {key}/{task}/{cell} train-only scaler/Ridge contract 失败"
        )
    coefficient = _json_vector(selected["ridge_coef_json"], 192, "probe Ridge coef")
    intercept = float(selected["ridge_intercept"])
    if not math.isfinite(intercept):
        raise ValueError(f"probe {key}/{task}/{cell} Ridge intercept 非 finite")
    try:
        recorded_grid = {
            float(alpha): float(score)
            for alpha, score in json.loads(
                str(selected["alpha_validation_mse_json"])
            ).items()
        }
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"probe {key}/{task}/{cell} alpha grid 非法") from exc
    train_scaled = (train["matrix"] - expected_mean) / expected_scale
    validation_scaled = (validation["matrix"] - expected_mean) / expected_scale
    refit_rows: list[tuple[float, float, Ridge]] = []
    for alpha in ALPHAS:
        refit_model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-8,
            max_iter=10000,
        )
        refit_model.fit(train_scaled, train["y_standardised"])
        refit_prediction = np.asarray(
            refit_model.predict(validation_scaled), dtype=float
        )
        refit_mse = float(
            np.mean((refit_prediction - validation["y_standardised"]) ** 2)
        )
        _compare_scalar(
            recorded_grid.get(alpha),
            refit_mse,
            f"probe {key}/{task}/{cell} refit alpha={alpha} validation MSE",
        )
        refit_rows.append((alpha, refit_mse, refit_model))
    best_mse = min(row[1] for row in refit_rows)
    refit_alpha, _, selected_refit = min(
        (row for row in refit_rows if row[1] <= best_mse + 1e-12),
        key=lambda row: row[0],
    )
    _compare_scalar(
        selected["selected_alpha"],
        refit_alpha,
        f"probe {key}/{task}/{cell} refit selected alpha",
    )
    _compare_vector(
        coefficient,
        np.asarray(selected_refit.coef_, dtype=float).reshape(-1),
        f"probe {key}/{task}/{cell} refit Ridge coefficient",
    )
    _compare_scalar(
        intercept,
        float(selected_refit.intercept_),
        f"probe {key}/{task}/{cell} refit Ridge intercept",
    )
    validation_prediction = (
        (validation["matrix"] - scaler_mean) / scaler_scale
    ) @ coefficient + intercept
    validation_mse = float(
        np.mean((validation_prediction - validation["y_standardised"]) ** 2)
    )
    _compare_scalar(
        selected["val_mse_standardized"],
        validation_mse,
        f"probe {key}/{task}/{cell} live validation MSE",
    )
    train_mean = float(np.mean(train["y_standardised"]))
    b0_validation_mse = float(np.mean((validation["y_standardised"] - train_mean) ** 2))
    _compare_scalar(
        selected["train_target_mean_standardized"],
        train_mean,
        f"probe {key}/{task}/{cell} live train target mean",
    )
    _compare_scalar(
        selected["b0_val_mse_standardized"],
        b0_validation_mse,
        f"probe {key}/{task}/{cell} live B0 validation MSE",
    )
    test_prediction_standardised = (
        (test["matrix"] - scaler_mean) / scaler_scale
    ) @ coefficient + intercept
    test_prediction_natural = _inverse_probe_prediction_live(
        test_prediction_standardised,
        task,
        cell,
        static_specs,
        change_specs,
    )
    b0_natural = float(
        _inverse_probe_prediction_live(
            np.asarray([train_mean]), task, cell, static_specs, change_specs
        )[0]
    )
    if part["patient_id"].astype(str).tolist() != test["patient_ids"]:
        raise ValueError(
            f"probe {key}/{task}/{cell} test patient/order 未闭合 live target"
        )
    if set(part["target_transform"].astype(str)) != {test["target_transform"]}:
        raise ValueError(f"probe {key}/{task}/{cell} target transform label 漂移")
    _compare_vector(
        part["y_true"].to_numpy(dtype=float),
        test["y_natural"],
        f"probe {key}/{task}/{cell} y_true",
    )
    _compare_vector(
        part["y_true_standardized"].to_numpy(dtype=float),
        test["y_standardised"],
        f"probe {key}/{task}/{cell} y_true_standardized",
    )
    _compare_vector(
        part["y_pred_standardized"].to_numpy(dtype=float),
        test_prediction_standardised,
        f"probe {key}/{task}/{cell} y_pred_standardized",
    )
    _compare_vector(
        part["y_pred"].to_numpy(dtype=float),
        test_prediction_natural,
        f"probe {key}/{task}/{cell} y_pred",
    )
    _compare_vector(
        part["b0_prediction_standardized"].to_numpy(dtype=float),
        np.full(len(part), train_mean),
        f"probe {key}/{task}/{cell} B0 standardized",
    )
    _compare_vector(
        part["b0_prediction"].to_numpy(dtype=float),
        np.full(len(part), b0_natural),
        f"probe {key}/{task}/{cell} B0 natural",
    )


def _audit_pcr_prediction_point_live(
    part: pd.DataFrame,
    selected: pd.Series,
    *,
    key: tuple[int, str, int],
    decision_point: str,
    patient_ids: np.ndarray,
    splits: np.ndarray,
    response: np.ndarray,
    labels: np.ndarray,
) -> None:
    dimensions = {"T0": 192, "T0-T1": 576, "T0-T2": 1152}
    dimension = dimensions[decision_point]
    matrix = _response_readout_live(response, decision_point)
    indices = {
        split: np.flatnonzero(splits == split) for split in ("train", "val", "test")
    }
    for column, split in (("n_train", "train"), ("n_val", "val"), ("n_test", "test")):
        if int(selected[column]) != len(indices[split]):
            raise ValueError(f"pCR {key}/{decision_point}.{column} live count 不一致")
    for column, split in (
        ("train_positive", "train"),
        ("val_positive", "val"),
        ("test_positive", "test"),
    ):
        if int(selected[column]) != int(labels[indices[split]].sum()):
            raise ValueError(f"pCR {key}/{decision_point}.{column} live label 不一致")
    train_matrix = matrix[indices["train"]]
    validation_matrix = matrix[indices["val"]]
    test_matrix = matrix[indices["test"]]
    expected_mean, expected_scale = _standard_scaler_stats(train_matrix)
    scaler_mean = _json_vector(
        selected["feature_scaler_mean_json"], dimension, "pCR scaler mean"
    )
    scaler_scale = _json_vector(
        selected["feature_scaler_scale_json"], dimension, "pCR scaler scale"
    )
    _compare_vector(
        scaler_mean, expected_mean, f"pCR {key}/{decision_point} scaler mean"
    )
    _compare_vector(
        scaler_scale, expected_scale, f"pCR {key}/{decision_point} scaler scale"
    )
    if int(selected["feature_scaler_n_samples_seen"]) != len(indices["train"]):
        raise ValueError(f"pCR {key}/{decision_point} scaler 非 outer-train fit")
    train_scaled = (train_matrix - expected_mean) / expected_scale
    validation_scaled = (validation_matrix - expected_mean) / expected_scale
    train_labels = labels[indices["train"]]
    validation_labels = labels[indices["val"]]
    try:
        recorded_items = json.loads(str(selected["grid_validation_metrics_json"]))
        recorded_grid = {
            (str(item["penalty"]), float(item["C"])): item for item in recorded_items
        }
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"pCR {key}/{decision_point} recorded grid 非法") from exc
    refit_penalty = str(selected["selected_penalty"])
    refit_c = float(selected["selected_C"])
    recorded = recorded_grid.get((refit_penalty, refit_c))
    if recorded is None:
        raise ValueError(f"pCR {key}/{decision_point} selected grid row 缺失")
    refit_model = LogisticRegression(
        l1_ratio=1.0 if refit_penalty == "l1" else 0.0,
        C=refit_c,
        solver="liblinear",
        class_weight="balanced",
        max_iter=20000,
        tol=1e-7,
        random_state=int(selected["random_state"]),
    )
    refit_model.fit(train_scaled, train_labels)
    refit_validation_probability = np.asarray(
        refit_model.predict_proba(validation_scaled)[:, 1], dtype=float
    )
    refit_auroc = float(roc_auc_score(validation_labels, refit_validation_probability))
    refit_auprc = float(
        average_precision_score(validation_labels, refit_validation_probability)
    )
    for name, expected in (
        ("val_auroc", refit_auroc),
        ("val_auprc", refit_auprc),
        ("n_iter", int(np.max(refit_model.n_iter_))),
    ):
        _compare_scalar(
            recorded[name],
            expected,
            f"pCR {key}/{decision_point} selected outer-train refit.{name}",
        )
    coefficient = _json_vector(
        selected["logistic_coef_json"], dimension, "pCR logistic coef"
    )
    intercept = _json_vector(
        selected["logistic_intercept_json"], 1, "pCR logistic intercept"
    )[0]
    _compare_vector(
        coefficient,
        np.asarray(refit_model.coef_, dtype=float).reshape(-1),
        f"pCR {key}/{decision_point} refit logistic coefficient",
    )
    _compare_scalar(
        intercept,
        float(np.asarray(refit_model.intercept_).reshape(-1)[0]),
        f"pCR {key}/{decision_point} refit logistic intercept",
    )
    validation_probability = _sigmoid_live(
        ((validation_matrix - scaler_mean) / scaler_scale) @ coefficient + intercept
    )
    _compare_vector(
        validation_probability,
        refit_validation_probability,
        f"pCR {key}/{decision_point} saved/refit validation probability",
    )
    validation_auroc = float(roc_auc_score(validation_labels, validation_probability))
    validation_auprc = float(
        average_precision_score(validation_labels, validation_probability)
    )
    threshold, youden, sensitivity, specificity = _youden_live(
        validation_labels, validation_probability
    )
    for column, expected in (
        ("val_auroc", validation_auroc),
        ("val_auprc", validation_auprc),
        ("selected_threshold", threshold),
        ("val_youden", youden),
        ("val_sensitivity", sensitivity),
        ("val_specificity", specificity),
    ):
        _compare_scalar(
            selected[column], expected, f"pCR {key}/{decision_point} live {column}"
        )
    test_probability = _sigmoid_live(
        ((test_matrix - scaler_mean) / scaler_scale) @ coefficient + intercept
    )
    test_labels = (test_probability >= threshold).astype(np.int64)
    expected_ids = patient_ids[indices["test"]].astype(str).tolist()
    if part["patient_id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"pCR {key}/{decision_point} test patient/order 未闭合")
    _compare_vector(
        part["y_true"].to_numpy(dtype=float),
        labels[indices["test"]].astype(float),
        f"pCR {key}/{decision_point} y_true",
    )
    _compare_vector(
        part["probability"].to_numpy(dtype=float),
        test_probability,
        f"pCR {key}/{decision_point} probability",
    )
    _compare_vector(
        part["threshold"].to_numpy(dtype=float),
        np.full(len(part), threshold),
        f"pCR {key}/{decision_point} threshold",
    )
    _compare_vector(
        part["predicted_label"].to_numpy(dtype=float),
        test_labels.astype(float),
        f"pCR {key}/{decision_point} predicted_label",
    )


def _load_validated_predictions(
    manifest: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], Mapping[str, Any]],
    selections: Mapping[tuple[str, int, str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_probe = {
        "patient_id",
        "seed_base",
        "fold",
        "effective_seed",
        "split",
        "model",
        "task",
        "timepoint",
        "transition",
        "representation",
        "input_variant",
        "target",
        "feature_dim",
        "y_true",
        "y_pred",
        "y_true_standardized",
        "y_pred_standardized",
        "b0_prediction",
        "b0_prediction_standardized",
        "selected_alpha",
        "target_transform",
        "source_feature_file",
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
        "static_transform_sha256",
        "change_transform_sha256",
        "raw_target_file_sha256",
        "raw_targets_sha256",
        "test_prediction_guard_enforced",
        "test_predict_call_count",
        *PROBE_FALSE_FLAGS,
    }
    required_pcr = {
        "patient_id",
        "seed_base",
        "fold",
        "effective_seed",
        "split",
        "model",
        "decision_point",
        "y_true",
        "probability",
        "predicted_label",
        "threshold",
        "penalty",
        "C",
        "readout",
        "class_weight",
        "feature_schema",
        "feature_schema_sha256",
        "feature_dim",
        "val_auroc",
        "val_auprc",
        "val_youden",
        "source_feature_file",
        "source_feature_sha256",
        "feature_extractor_sha256",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "fold_manifest_sha256",
        "canonical_patient_order_sha256",
        "canonical_patient_label_sha256",
        "test_feature_matrix_constructed_after_selection_lock",
        "test_prediction_guard_enforced",
        "test_predict_proba_call_count",
        *PCR_FALSE_FLAGS,
    }
    _require_columns(
        manifest,
        (
            "kind",
            "path",
            "sha256",
            "bytes",
            "rows",
            "patients",
            "seed_base",
            "model",
            "fold",
        ),
        "prediction_manifest",
    )
    probe_parts: list[pd.DataFrame] = []
    pcr_parts: list[pd.DataFrame] = []
    raw_cache: dict[Path, tuple[dict[str, np.ndarray], str]] = {}
    transform_cache: dict[Path, dict[str, Any]] = {}
    for row in manifest.itertuples(index=False):
        kind = str(row.kind)
        seed, model, fold = int(row.seed_base), str(row.model).upper(), int(row.fold)
        key = (seed, model, fold)
        path = _resolve_public_path(str(row.path))
        frame = pd.read_csv(path, low_memory=False)
        expected_source = assets[key]
        if len(frame) != int(row.rows) or frame["patient_id"].nunique() != int(
            row.patients
        ):
            raise ValueError(f"prediction manifest rows/patients 不闭合: {kind}/{key}")
        if kind == "probe_prediction":
            _require_columns(frame, required_probe, f"probe prediction {key}")
        elif kind == "pcr_prediction":
            _require_columns(frame, required_pcr, f"pCR prediction {key}")
        else:
            raise ValueError(f"prediction kind 非法: {kind}")
        frame = frame.copy()
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["model"] = frame["model"].astype(str).str.upper()
        frame["seed_base"] = _strict_int(frame["seed_base"], f"{kind}.seed_base")
        frame["fold"] = _strict_int(frame["fold"], f"{kind}.fold")
        frame["effective_seed"] = _strict_int(
            frame["effective_seed"], f"{kind}.effective_seed"
        )
        if (
            set(frame["seed_base"]) != {seed}
            or set(frame["fold"]) != {fold}
            or set(frame["effective_seed"]) != {seed + fold}
            or set(frame["model"]) != {model}
            or set(frame["split"].astype(str).str.lower()) != {"test"}
            or set(frame["source_feature_sha256"].astype(str))
            != {expected_source["feature_sha256"]}
            or set(frame["source_checkpoint_sha256"].astype(str))
            != {expected_source["checkpoint_sha256"]}
            or set(frame["fold_manifest_sha256"].astype(str))
            != {EXPECTED_FOLD_MANIFEST_SHA256}
        ):
            raise ValueError(
                f"prediction path/source/split contract 错误: {kind}/{key}"
            )
        feature_patient_ids, feature_splits, feature_response, feature_labels = (
            _load_feature_arrays_live(expected_source, key)
        )

        if kind == "probe_prediction":
            frame["task"] = frame["task"].astype(str).str.lower()
            frame["timepoint"] = frame["timepoint"].fillna("").astype(str).str.upper()
            frame["transition"] = (
                frame["transition"].fillna("").map(_normalise_transition)
            )
            frame["cell"] = np.where(
                frame["task"].eq("static"), frame["timepoint"], frame["transition"]
            )
            cells = set(frame[["task", "cell"]].itertuples(index=False, name=None))
            expected_cells = {("static", value) for value in TIMEPOINTS} | {
                ("change", value) for value in TRANSITIONS
            }
            if cells != expected_cells:
                raise ValueError(f"probe prediction {key} cell coverage 错误")
            patient_count = frame["patient_id"].nunique()
            if (
                len(frame) != patient_count * 7
                or frame.duplicated(
                    [
                        "patient_id",
                        "seed_base",
                        "fold",
                        "model",
                        "task",
                        "cell",
                        "target",
                    ]
                ).any()
            ):
                raise ValueError(f"probe prediction {key} patient/cell key 重复或缺失")
            if not set(frame["patient_id"]).issubset(
                set(expected_source["test_patient_ids"])
            ):
                raise ValueError(f"probe prediction {key} 含非 outer-test patient")
            cell_sets = [
                set(part["patient_id"]) for _, part in frame.groupby(["task", "cell"])
            ]
            if any(values != cell_sets[0] for values in cell_sets[1:]):
                raise ValueError(f"probe prediction {key} 七 cell patient 集不一致")
            if (
                set(frame["target"].astype(str).str.lower()) != {"ftv"}
                or set(frame["representation"].astype(str).str.lower())
                != {"response_state"}
                or set(_strict_int(frame["feature_dim"], "probe feature_dim")) != {192}
                or set(
                    frame.loc[frame["task"].eq("static"), "input_variant"].astype(str)
                )
                != {"current"}
                or set(
                    frame.loc[frame["task"].eq("change"), "input_variant"].astype(str)
                )
                != {"observed_difference"}
            ):
                raise ValueError(
                    f"probe prediction {key} FTV/response-state contract 漂移"
                )
            _finite_columns(
                frame,
                (
                    "y_true",
                    "y_pred",
                    "y_true_standardized",
                    "y_pred_standardized",
                    "b0_prediction",
                    "b0_prediction_standardized",
                    "selected_alpha",
                ),
                f"probe prediction {key}",
            )
            for column in (
                "source_feature_sha256",
                "feature_extractor_sha256",
                "source_checkpoint_sha256",
                "fold_manifest_sha256",
                "canonical_patient_order_sha256",
                "canonical_patient_label_sha256",
                "static_transform_sha256",
                "change_transform_sha256",
                "raw_target_file_sha256",
                "raw_targets_sha256",
            ):
                _require_sha(frame[column], f"probe prediction {key}.{column}")
            for column in PROBE_FALSE_FLAGS:
                if _strict_bool(frame[column], f"probe prediction.{column}").any():
                    raise ValueError(f"probe prediction {key} test 参与 selection")
            if (
                not _strict_bool(
                    frame["test_prediction_guard_enforced"], "probe prediction guard"
                ).all()
                or not _strict_int(
                    frame["test_predict_call_count"], "probe prediction call count"
                )
                .eq(1)
                .all()
            ):
                raise ValueError(f"probe prediction {key} test-once guard 失败")
            selection = selections[("probe", seed, model, fold)].copy()
            selection["task"] = selection["task"].astype(str).str.lower()
            selection["cell"] = np.where(
                selection["task"].eq("static"),
                selection["timepoint"].fillna("").astype(str).str.upper(),
                selection["transition"].fillna("").map(_normalise_transition),
            )
            alpha_by_cell = selection.set_index(["task", "cell"])["selected_alpha"]
            for cell_key, part in frame.groupby(["task", "cell"]):
                selected_row = selection.loc[
                    selection["task"].eq(cell_key[0])
                    & selection["cell"].eq(cell_key[1])
                ].iloc[0]
                expected_alpha = float(alpha_by_cell.loc[cell_key])
                if (
                    not pd.to_numeric(part["selected_alpha"], errors="coerce")
                    .eq(expected_alpha)
                    .all()
                ):
                    raise ValueError(f"probe prediction {key}/{cell_key} alpha 未闭合")
                for column in (
                    "target_transform",
                    "source_feature_file",
                    "source_checkpoint",
                    "feature_extractor_sha256",
                    "canonical_patient_order_sha256",
                    "canonical_patient_label_sha256",
                    "static_transform_sha256",
                    "change_transform_sha256",
                    "raw_target_file_sha256",
                    "raw_targets_sha256",
                ):
                    if set(part[column].astype(str)) != {str(selected_row[column])}:
                        raise ValueError(
                            f"probe prediction {key}/{cell_key}.{column} 未闭合"
                        )
                _audit_probe_prediction_cell_live(
                    part,
                    selected_row,
                    key=key,
                    task=cell_key[0],
                    cell=cell_key[1],
                    patient_ids=feature_patient_ids,
                    splits=feature_splits,
                    response=feature_response,
                    asset=expected_source,
                    raw_cache=raw_cache,
                    transform_cache=transform_cache,
                )
            probe_parts.append(frame)
        else:
            frame["decision_point"] = frame["decision_point"].map(_normalise_decision)
            if set(frame["decision_point"]) != set(DECISION_POINTS):
                raise ValueError(f"pCR prediction {key} decision coverage 错误")
            patient_count = frame["patient_id"].nunique()
            if (
                len(frame) != patient_count * 3
                or frame.duplicated(
                    ["patient_id", "seed_base", "fold", "model", "decision_point"]
                ).any()
            ):
                raise ValueError(
                    f"pCR prediction {key} patient/decision key 重复或缺失"
                )
            if set(frame["patient_id"]) != set(expected_source["test_patient_ids"]):
                raise ValueError(
                    f"pCR prediction {key} 未闭合到 feature outer-test patients"
                )
            point_sets = [
                set(part["patient_id"]) for _, part in frame.groupby("decision_point")
            ]
            if any(values != point_sets[0] for values in point_sets[1:]):
                raise ValueError(f"pCR prediction {key} 三 decision patient 集不一致")
            _finite_columns(
                frame,
                ("y_true", "probability", "predicted_label", "threshold", "C"),
                f"pCR prediction {key}",
            )
            labels = _strict_int(frame["y_true"], "pCR y_true")
            predicted = _strict_int(frame["predicted_label"], "pCR predicted_label")
            probability = pd.to_numeric(frame["probability"], errors="coerce")
            threshold = pd.to_numeric(frame["threshold"], errors="coerce")
            if (
                not set(labels).issubset({0, 1})
                or not set(predicted).issubset({0, 1})
                or not probability.between(0, 1).all()
                or not threshold.between(0, 1).all()
                or not np.array_equal(
                    predicted.to_numpy(), (probability >= threshold).astype(int)
                )
                or set(frame["readout"].astype(str).str.lower())
                != {"class-balanced logisticregression"}
                or set(frame["class_weight"].astype(str).str.lower()) != {"balanced"}
            ):
                raise ValueError(
                    f"pCR prediction {key} label/probability/readout contract 错误"
                )
            expected_labels = frame["patient_id"].map(expected_source["test_labels"])
            if expected_labels.isna().any() or not np.array_equal(
                labels.to_numpy(), expected_labels.to_numpy(dtype=int)
            ):
                raise ValueError(f"pCR prediction {key} y_true 未闭合到 feature labels")
            for column in (
                "source_feature_sha256",
                "feature_extractor_sha256",
                "source_checkpoint_sha256",
                "fold_manifest_sha256",
                "canonical_patient_order_sha256",
                "canonical_patient_label_sha256",
                "feature_schema_sha256",
            ):
                _require_sha(frame[column], f"pCR prediction {key}.{column}")
            schemas = {
                "T0": ("r0", 192),
                "T0-T1": ("concat(r0,r1,r1-r0)", 576),
                "T0-T2": ("concat(r0,r1,r2,r1-r0,r2-r1,r2-r0)", 1152),
            }
            selection = selections[("pcr", seed, model, fold)].set_index(
                "decision_point"
            )
            for point, (schema, dimension) in schemas.items():
                part = frame.loc[frame["decision_point"].eq(point)]
                selected = selection.loc[point]
                if (
                    set(part["feature_schema"].astype(str)) != {schema}
                    or set(part["feature_schema_sha256"].astype(str))
                    != {hashlib.sha256(schema.encode("utf-8")).hexdigest()}
                    or set(_strict_int(part["feature_dim"], "pCR feature_dim"))
                    != {dimension}
                    or set(part["penalty"].astype(str))
                    != {str(selected["selected_penalty"])}
                    or not pd.to_numeric(part["C"], errors="coerce")
                    .eq(float(selected["selected_C"]))
                    .all()
                    or not pd.to_numeric(part["threshold"], errors="coerce")
                    .eq(float(selected["selected_threshold"]))
                    .all()
                    or not pd.to_numeric(part["val_auroc"], errors="coerce")
                    .eq(float(selected["val_auroc"]))
                    .all()
                    or not pd.to_numeric(part["val_auprc"], errors="coerce")
                    .eq(float(selected["val_auprc"]))
                    .all()
                    or not pd.to_numeric(part["val_youden"], errors="coerce")
                    .eq(float(selected["val_youden"]))
                    .all()
                ):
                    raise ValueError(
                        f"pCR prediction {key}/{point} selection/schema 未闭合"
                    )
                for column in (
                    "source_feature_file",
                    "source_checkpoint",
                    "feature_extractor_sha256",
                    "canonical_patient_order_sha256",
                    "canonical_patient_label_sha256",
                ):
                    if set(part[column].astype(str)) != {str(selected[column])}:
                        raise ValueError(
                            f"pCR prediction {key}/{point}.{column} 未闭合"
                        )
                _audit_pcr_prediction_point_live(
                    part,
                    selected,
                    key=key,
                    decision_point=point,
                    patient_ids=feature_patient_ids,
                    splits=feature_splits,
                    response=feature_response,
                    labels=feature_labels,
                )
            for column in PCR_FALSE_FLAGS:
                if _strict_bool(frame[column], f"pCR prediction.{column}").any():
                    raise ValueError(f"pCR prediction {key} test 参与 selection")
            for column in (
                "test_feature_matrix_constructed_after_selection_lock",
                "test_prediction_guard_enforced",
            ):
                if not _strict_bool(frame[column], f"pCR prediction.{column}").all():
                    raise ValueError(f"pCR prediction {key} guard false")
            if (
                not _strict_int(
                    frame["test_predict_proba_call_count"], "pCR predict call count"
                )
                .eq(1)
                .all()
            ):
                raise ValueError(f"pCR prediction {key} test-once guard 失败")
            pcr_parts.append(frame)

    probe = pd.concat(probe_parts, ignore_index=True)
    pcr = pd.concat(pcr_parts, ignore_index=True)
    if len(probe) != EXPECTED_PROBE_ROWS or len(pcr) != EXPECTED_PCR_ROWS:
        raise ValueError(f"prediction 总行数错误: probe={len(probe)} pCR={len(pcr)}")
    for seed in SEEDS:
        for frame, label, expected_n, row_factor, keys in (
            (probe, "probe", 375, 7, ["patient_id", "fold", "task", "cell"]),
            (pcr, "pCR", 808, 3, ["patient_id", "fold", "decision_point"]),
        ):
            seed_part = frame.loc[frame["seed_base"].eq(seed)]
            by_model: dict[str, pd.DataFrame] = {}
            for model in MODELS:
                part = seed_part.loc[seed_part["model"].eq(model)]
                if (
                    part["patient_id"].nunique() != expected_n
                    or len(part) != expected_n * row_factor
                ):
                    raise ValueError(
                        f"{label} seed={seed}/{model} patient closure 错误"
                    )
                assignments = part[["patient_id", "fold"]].drop_duplicates()
                if (
                    len(assignments) != expected_n
                    or assignments["patient_id"].duplicated().any()
                ):
                    raise ValueError(
                        f"{label} seed={seed}/{model} patient 跨 fold 重复"
                    )
                by_model[model] = part.sort_values(keys).reset_index(drop=True)
            if not by_model["G1"][keys].equals(
                by_model["G3"][keys]
            ) or not np.array_equal(
                by_model["G1"]["y_true"].to_numpy(), by_model["G3"]["y_true"].to_numpy()
            ):
                raise ValueError(
                    f"{label} seed={seed} G1/G3 exact-patient/target closure 失败"
                )
            if label == "probe" and not np.allclose(
                by_model["G1"]["b0_prediction"],
                by_model["G3"]["b0_prediction"],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"probe seed={seed} G1/G3 B0 不一致")
    for frame, keys, label in (
        (probe, ["patient_id", "fold", "task", "cell", "y_true"], "probe"),
        (pcr, ["patient_id", "fold", "decision_point", "y_true"], "pCR"),
    ):
        reference = (
            frame.loc[frame["seed_base"].eq(SEEDS[0]) & frame["model"].eq("G1"), keys]
            .sort_values(keys[:-1])
            .reset_index(drop=True)
        )
        for seed in SEEDS[1:]:
            current = (
                frame.loc[frame["seed_base"].eq(seed) & frame["model"].eq("G1"), keys]
                .sort_values(keys[:-1])
                .reset_index(drop=True)
            )
            if not reference.equals(current):
                raise ValueError(f"{label} patient/fold/target 跨 seed 漂移: {seed}")
    return probe, pcr


def _metric_rows(
    frame: pd.DataFrame, keys: Sequence[str], *, classifier: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric = _classification if classifier else _regression
    for values, part in frame.groupby(list(keys), sort=False, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values, strict=True))
        row.update(metric(part))
        rows.append(row)
    return pd.DataFrame(rows)


def _mean(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)) if len(array) else math.nan


def _sd(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.std(array, ddof=1)) if len(array) >= 2 else math.nan


def _recompute_metric_tables(
    probe: pd.DataFrame, pcr: pd.DataFrame, stability: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    probe_seed = _metric_rows(probe, ("seed_base", "model", "task", "cell"))
    probe_fold = _metric_rows(probe, ("seed_base", "fold", "model", "task", "cell"))
    pcr_seed_model = _metric_rows(
        pcr, ("seed_base", "model", "decision_point"), classifier=True
    )
    pcr_secondary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for point in DECISION_POINTS:
            part = pcr_seed_model.loc[
                pcr_seed_model["seed_base"].eq(seed)
                & pcr_seed_model["decision_point"].eq(point)
            ].set_index("model")
            if set(part.index) != set(MODELS):
                raise ValueError(f"pCR metric 缺 model: seed={seed}/{point}")
            row: dict[str, Any] = {"seed_base": seed, "decision_point": point}
            for metric in CLASSIFICATION_METRICS:
                row[f"g1_{metric}"] = float(part.loc["G1", metric])
                row[f"g3_{metric}"] = float(part.loc["G3", metric])
                row[f"delta_{metric}"] = row[f"g3_{metric}"] - row[f"g1_{metric}"]
            row["n_patients"] = int(part.loc["G1", "n_patients"])
            pcr_secondary_rows.append(row)
    pcr_secondary = pd.DataFrame(pcr_secondary_rows)

    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        cells = probe_seed.loc[probe_seed["seed_base"].eq(seed)]
        fold_cells = probe_fold.loc[probe_fold["seed_base"].eq(seed)]
        static = cells.loc[cells["task"].eq("static")]
        dynamic = cells.loc[cells["task"].eq("change")]
        d_s = _mean(static.loc[static["model"].eq("G3"), "spearman"]) - _mean(
            static.loc[static["model"].eq("G1"), "spearman"]
        )
        d_d = _mean(dynamic.loc[dynamic["model"].eq("G3"), "spearman"]) - _mean(
            dynamic.loc[dynamic["model"].eq("G1"), "spearman"]
        )
        d_d_r2 = _mean(dynamic.loc[dynamic["model"].eq("G3"), "r2"]) - _mean(
            dynamic.loc[dynamic["model"].eq("G1"), "r2"]
        )
        longitudinal = pcr_secondary.loc[
            pcr_secondary["seed_base"].eq(seed)
            & pcr_secondary["decision_point"].isin(("T0-T1", "T0-T2"))
        ]
        failures = stability.loc[
            stability["seed_base"].eq(seed)
            & stability["model"].eq("G3")
            & stability["base_pass"].eq(False)
        ]
        row = {
            "seed_base": seed,
            "dS": d_s,
            "static_delta_spearman": d_s,
            "dD": d_d,
            "delta_ftv_delta_spearman": d_d,
            "dD_R2": d_d_r2,
            "delta_ftv_delta_r2": d_d_r2,
            "pcr_longitudinal_delta_auroc": _mean(longitudinal["delta_auroc"]),
            "pcr_longitudinal_delta_auprc": _mean(longitudinal["delta_auprc"]),
            "failed_fold_count": len(failures),
            "static_positive": bool(d_s > 0),
            "dynamic_positive": bool(d_d > 0),
            "pooled_oof_probe_patients": int(
                fold_cells.loc[fold_cells["model"].eq("G1"), "n_patients"].sum() / 7
            ),
        }
        for point in DECISION_POINTS:
            paired = pcr_secondary.loc[
                pcr_secondary["seed_base"].eq(seed)
                & pcr_secondary["decision_point"].eq(point)
            ].iloc[0]
            token = point.replace("-", "_")
            for metric in ("auroc", "auprc"):
                row[f"pcr_{token}_delta_{metric}"] = float(paired[f"delta_{metric}"])
        seed_rows.append(row)
    seed_level = pd.DataFrame(seed_rows)

    seed_fold_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            cells = probe_fold.loc[
                probe_fold["seed_base"].eq(seed) & probe_fold["fold"].eq(fold)
            ]
            static = cells.loc[cells["task"].eq("static")]
            dynamic = cells.loc[cells["task"].eq("change")]
            g1 = stability.loc[
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G1")
            ].iloc[0]
            g3 = stability.loc[
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G3")
            ].iloc[0]
            seed_fold_rows.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "dS_sf": _mean(static.loc[static["model"].eq("G3"), "spearman"])
                    - _mean(static.loc[static["model"].eq("G1"), "spearman"]),
                    "dD_sf": _mean(dynamic.loc[dynamic["model"].eq("G3"), "spearman"])
                    - _mean(dynamic.loc[dynamic["model"].eq("G1"), "spearman"]),
                    "D": float(g3["base_degradation_fraction"]),
                    "base_pass": bool(g3["base_pass"]),
                    "g1_val_state_loss": float(g1["val_state_loss"]),
                    "g3_val_state_loss": float(g3["val_state_loss"]),
                    "g1_selected_epoch": int(g1["selected_epoch"]),
                    "g3_selected_epoch": int(g3["selected_epoch"]),
                    "g1_representation_std": float(g1["representation_std"]),
                    "g3_representation_std": float(g3["representation_std"]),
                }
            )
    seed_fold = pd.DataFrame(seed_fold_rows)
    fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        part = seed_fold.loc[seed_fold["fold"].eq(fold)]
        base = _strict_bool(part["base_pass"], "expected base_pass")
        fold_rows.append(
            {
                "fold": fold,
                "static_delta_spearman_mean": _mean(part["dS_sf"]),
                "static_delta_spearman_sd": _sd(part["dS_sf"]),
                "delta_ftv_delta_spearman_mean": _mean(part["dD_sf"]),
                "delta_ftv_delta_spearman_sd": _sd(part["dD_sf"]),
                "base_failure_count": int((~base).sum()),
                "base_failure_rate": float((~base).mean()),
                "base_degradation_mean": _mean(part["D"]),
                "base_degradation_sd": _sd(part["D"]),
            }
        )
    return {
        "probe_seed_cell_metrics.csv": probe_seed,
        "probe_seed_fold_cell_metrics.csv": probe_fold,
        "pcr_seed_model_metrics.csv": pcr_seed_model,
        "pcr_secondary_seed_metrics.csv": pcr_secondary,
        "seed_level_robustness.csv": seed_level,
        "seed_fold_effects.csv": seed_fold,
        "fold_level_robustness.csv": pd.DataFrame(fold_rows),
    }


def _compare_recomputed_metric_tables(
    tables: Mapping[str, pd.DataFrame], expected: Mapping[str, pd.DataFrame]
) -> None:
    keys = {
        "probe_seed_cell_metrics.csv": ("seed_base", "model", "task", "cell"),
        "probe_seed_fold_cell_metrics.csv": (
            "seed_base",
            "fold",
            "model",
            "task",
            "cell",
        ),
        "pcr_seed_model_metrics.csv": ("seed_base", "model", "decision_point"),
        "pcr_secondary_seed_metrics.csv": ("seed_base", "decision_point"),
        "seed_level_robustness.csv": ("seed_base",),
        "seed_fold_effects.csv": ("seed_base", "fold"),
        "fold_level_robustness.csv": ("fold",),
    }
    for name, expected_frame in expected.items():
        _compare_table(
            tables[name],
            expected_frame,
            keys[name],
            name,
            tuple(expected_frame.columns),
        )


def _recompute_seed_uncertainty(seed_level: pd.DataFrame) -> pd.DataFrame:
    endpoint_columns = {
        "dS": "primary_static_spearman",
        "dD": "primary_dynamic_spearman",
        "dD_R2": "descriptive_dynamic_r2",
        "pcr_longitudinal_delta_auroc": "secondary_pcr_longitudinal_auroc",
        "pcr_longitudinal_delta_auprc": "secondary_pcr_longitudinal_auprc",
    }
    for point in DECISION_POINTS:
        token = point.replace("-", "_")
        for metric in ("auroc", "auprc"):
            endpoint_columns[f"pcr_{token}_delta_{metric}"] = (
                f"secondary_pcr_{token}_{metric}"
            )
    n = len(SEEDS)
    critical = float(student_t.ppf(0.975, n - 1))
    rows: list[dict[str, Any]] = []
    for column, endpoint in endpoint_columns.items():
        values = seed_level[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        positive_n = int(np.count_nonzero(finite > 0))
        if len(values) != n:
            raise ValueError(f"seed uncertainty {column} seed 数错误")
        if len(finite) != n:
            row = {
                "endpoint": endpoint,
                "column": column,
                "n_seeds": n,
                "finite_seeds": len(finite),
                "verifiable": False,
                "mean": math.nan,
                "sample_sd": math.nan,
                "minimum": math.nan,
                "median": math.nan,
                "maximum": math.nan,
                "positive_n": positive_n,
                "positive_rate": math.nan,
                "t_critical": critical,
                "t_ci_low": math.nan,
                "t_ci_high": math.nan,
                "exact_sign_test_p_two_sided": math.nan,
                "clopper_pearson_positive_rate_low": math.nan,
                "clopper_pearson_positive_rate_high": math.nan,
                "used_for_R1": column == "dS",
                "used_for_R2": column == "dD",
            }
        else:
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            margin = critical * sd / math.sqrt(n)
            cp_low = (
                0.0
                if positive_n == 0
                else float(beta.ppf(0.025, positive_n, n - positive_n + 1))
            )
            cp_high = (
                1.0
                if positive_n == n
                else float(beta.ppf(0.975, positive_n + 1, n - positive_n))
            )
            row = {
                "endpoint": endpoint,
                "column": column,
                "n_seeds": n,
                "finite_seeds": n,
                "verifiable": True,
                "mean": mean,
                "sample_sd": sd,
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
                "positive_n": positive_n,
                "positive_rate": positive_n / n,
                "t_critical": critical,
                "t_ci_low": mean - margin,
                "t_ci_high": mean + margin,
                "exact_sign_test_p_two_sided": float(
                    binomtest(positive_n, n, 0.5).pvalue
                ),
                "clopper_pearson_positive_rate_low": cp_low,
                "clopper_pearson_positive_rate_high": cp_high,
                "used_for_R1": column == "dS",
                "used_for_R2": column == "dD",
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _effect_from_probe(part: pd.DataFrame) -> dict[str, float]:
    metrics = _metric_rows(part, ("model", "task", "cell"))
    static = metrics.loc[metrics["task"].eq("static")]
    dynamic = metrics.loc[metrics["task"].eq("change")]
    return {
        "dS": _mean(static.loc[static["model"].eq("G3"), "spearman"])
        - _mean(static.loc[static["model"].eq("G1"), "spearman"]),
        "dD": _mean(dynamic.loc[dynamic["model"].eq("G3"), "spearman"])
        - _mean(dynamic.loc[dynamic["model"].eq("G1"), "spearman"]),
        "dD_R2": _mean(dynamic.loc[dynamic["model"].eq("G3"), "r2"])
        - _mean(dynamic.loc[dynamic["model"].eq("G1"), "r2"]),
    }


def _recompute_leave_one_out(
    probe: pd.DataFrame, seed_level: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for omitted_seed in SEEDS:
        kept = seed_level.loc[~seed_level["seed_base"].eq(omitted_seed)]
        rows.append(
            {
                "scope": "leave_one_seed_out",
                "omitted_seed": omitted_seed,
                "omitted_fold": pd.NA,
                "seed_base": pd.NA,
                "static_mean": _mean(kept["dS"]),
                "dynamic_mean": _mean(kept["dD"]),
                "dynamic_r2_mean": _mean(kept["dD_R2"]),
                "spearman_recomputed_from_patient_rows": False,
                "n_training_seeds": 4,
            }
        )
    for omitted_fold in FOLDS:
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            effect = _effect_from_probe(
                probe.loc[probe["seed_base"].eq(seed) & ~probe["fold"].eq(omitted_fold)]
            )
            per_seed.append(effect)
            rows.append(
                {
                    "scope": "leave_one_fold_out_seed",
                    "omitted_seed": pd.NA,
                    "omitted_fold": omitted_fold,
                    "seed_base": seed,
                    "static_mean": effect["dS"],
                    "dynamic_mean": effect["dD"],
                    "dynamic_r2_mean": effect["dD_R2"],
                    "spearman_recomputed_from_patient_rows": True,
                    "n_training_seeds": 1,
                }
            )
        per_seed_frame = pd.DataFrame(per_seed)
        rows.append(
            {
                "scope": "leave_one_fold_out_across_seed",
                "omitted_seed": pd.NA,
                "omitted_fold": omitted_fold,
                "seed_base": pd.NA,
                "static_mean": _mean(per_seed_frame["dS"]),
                "dynamic_mean": _mean(per_seed_frame["dD"]),
                "dynamic_r2_mean": _mean(per_seed_frame["dD_R2"]),
                "spearman_recomputed_from_patient_rows": True,
                "n_training_seeds": 5,
            }
        )
    return pd.DataFrame(rows)


def _variance_row(
    seed_fold: pd.DataFrame, value_column: str, endpoint: str, primary: bool
) -> dict[str, Any]:
    matrix = (
        seed_fold.pivot(index="seed_base", columns="fold", values=value_column)
        .reindex(index=SEEDS, columns=FOLDS)
        .to_numpy(dtype=float)
    )
    if matrix.shape != (5, 5) or not np.isfinite(matrix).all():
        raise ValueError(f"variance {endpoint} 不是 finite balanced 5×5")
    grand = float(matrix.mean())
    row_mean, fold_mean = matrix.mean(axis=1), matrix.mean(axis=0)
    ss_seed = float(5 * np.sum((row_mean - grand) ** 2))
    ss_fold = float(5 * np.sum((fold_mean - grand) ** 2))
    residual = matrix - row_mean[:, None] - fold_mean[None, :] + grand
    ss_residual = float(np.sum(residual**2))
    ms_seed, ms_fold, ms_residual = ss_seed / 4, ss_fold / 4, ss_residual / 16
    raw = np.asarray(
        [(ms_seed - ms_residual) / 5, (ms_fold - ms_residual) / 5, ms_residual]
    )
    clipped = np.maximum(raw, 0.0)
    shares = np.zeros(3) if clipped.sum() == 0 else clipped / clipped.sum()
    if clipped.sum() == 0:
        dominance = "no_variation"
    elif shares.max() > 0.5:
        dominance = (
            "seed",
            "fold",
            "interaction_and_metric_sampling_error",
        )[int(np.argmax(shares))]
    else:
        dominance = "mixed"
    return {
        "endpoint": endpoint,
        "value_column": value_column,
        "primary": primary,
        "verifiable": True,
        "grand_mean": grand,
        "ss_seed": ss_seed,
        "ss_fold": ss_fold,
        "ss_residual": ss_residual,
        "df_seed": 4,
        "df_fold": 4,
        "df_residual": 16,
        "ms_seed": ms_seed,
        "ms_fold": ms_fold,
        "ms_residual": ms_residual,
        "raw_seed_component": float(raw[0]),
        "raw_fold_component": float(raw[1]),
        "raw_interaction_sampling_component": float(raw[2]),
        "clipped_seed_component": float(clipped[0]),
        "clipped_fold_component": float(clipped[1]),
        "clipped_interaction_sampling_component": float(clipped[2]),
        "sqrt_seed_component": float(math.sqrt(clipped[0])),
        "sqrt_fold_component": float(math.sqrt(clipped[1])),
        "sqrt_interaction_sampling_component": float(math.sqrt(clipped[2])),
        "seed_share": float(shares[0]),
        "fold_share": float(shares[1]),
        "interaction_sampling_share": float(shares[2]),
        "dominance": dominance,
        "residual_label": "seed×fold interaction + metric sampling error",
        "no_cell_replication": True,
    }


def _recompute_variance_table(seed_fold: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _variance_row(seed_fold, "dD_sf", "dynamic_spearman", True),
            _variance_row(seed_fold, "dS_sf", "static_spearman", False),
            _variance_row(seed_fold, "D", "base_degradation", False),
        ]
    )


def _validate_bootstrap_tables(
    conditional: pd.DataFrame,
    crossed: pd.DataFrame,
    seed_level: pd.DataFrame,
    pcr_secondary: pd.DataFrame,
) -> None:
    required = {
        "seed_base",
        "cohort",
        "endpoint",
        "estimate",
        "ci_low",
        "ci_high",
        "bootstrap_replicates",
        "finite_replicates",
        "rng_seed",
        "bootstrap_unit",
    }
    _require_columns(conditional, required, "conditional bootstrap")
    expected_keys = {
        (seed, "FTV", endpoint) for seed in SEEDS for endpoint in ("dS", "dD", "dD_R2")
    } | {
        (seed, "pCR", endpoint)
        for seed in SEEDS
        for endpoint in (
            "pcr_T0_delta_auroc",
            "pcr_T0_delta_auprc",
            "pcr_T0_T1_delta_auroc",
            "pcr_T0_T1_delta_auprc",
            "pcr_T0_T2_delta_auroc",
            "pcr_T0_T2_delta_auprc",
            "pcr_longitudinal_delta_auroc",
            "pcr_longitudinal_delta_auprc",
        )
    }
    keys = list(
        zip(
            _strict_int(conditional["seed_base"], "conditional.seed"),
            conditional["cohort"].astype(str),
            conditional["endpoint"].astype(str),
            strict=True,
        )
    )
    if len(keys) != 55 or len(set(keys)) != 55 or set(keys) != expected_keys:
        raise ValueError(
            "conditional bootstrap 55-row seed/cohort/endpoint coverage 错误"
        )
    if set(
        _strict_int(conditional["bootstrap_replicates"], "conditional.replicates")
    ) != {EXPECTED_CONDITIONAL_REPLICATES} or set(
        _strict_int(conditional["rng_seed"], "conditional.rng")
    ) != {
        EXPECTED_ANALYSIS_SEED
    }:
        raise ValueError("conditional bootstrap replicate/RNG lock 漂移")
    finite = _strict_int(conditional["finite_replicates"], "conditional.finite")
    _finite_columns(conditional, ("estimate", "ci_low", "ci_high"), "conditional")
    if (
        not finite.between(1, EXPECTED_CONDITIONAL_REPLICATES).all()
        or not (
            pd.to_numeric(conditional["ci_low"])
            <= pd.to_numeric(conditional["ci_high"])
        ).all()
    ):
        raise ValueError("conditional bootstrap CI/finite replicates 非法")
    units = {
        "FTV": "patient_within_outer_fold_same_draw_across_G1_G3_and_cells",
        "pCR": "patient_within_outer_fold_same_draw_across_G1_G3_and_decision_points",
    }
    for cohort, unit in units.items():
        if set(
            conditional.loc[conditional["cohort"].eq(cohort), "bootstrap_unit"].astype(
                str
            )
        ) != {unit}:
            raise ValueError(f"conditional {cohort} 未声明冻结同步 draw contract")
    seed_index = seed_level.set_index("seed_base")
    pcr_index = pcr_secondary.set_index(["seed_base", "decision_point"])
    for row in conditional.itertuples(index=False):
        if row.cohort == "FTV" or row.endpoint.startswith("pcr_longitudinal"):
            expected = float(seed_index.loc[int(row.seed_base), row.endpoint])
        else:
            metric = "auroc" if row.endpoint.endswith("auroc") else "auprc"
            token = row.endpoint.removeprefix("pcr_").removesuffix(f"_delta_{metric}")
            point = token.replace("_", "-")
            expected = float(
                pcr_index.loc[(int(row.seed_base), point), f"delta_{metric}"]
            )
        _compare_float(
            float(row.estimate),
            expected,
            f"conditional estimate {row.seed_base}/{row.endpoint}",
        )

    _require_columns(
        crossed,
        (
            "endpoint",
            "estimate",
            "ci_low",
            "ci_high",
            "bootstrap_replicates",
            "finite_replicates",
            "rng_seed",
            "bootstrap_unit",
            "used_for_decision",
        ),
        "crossed bootstrap",
    )
    if len(crossed) != 3 or set(crossed["endpoint"].astype(str)) != {
        "dS",
        "dD",
        "dD_R2",
    }:
        raise ValueError("crossed bootstrap 必须恰有三个 FTV endpoints")
    if (
        set(_strict_int(crossed["bootstrap_replicates"], "crossed.replicates"))
        != {EXPECTED_CROSSED_REPLICATES}
        or set(_strict_int(crossed["rng_seed"], "crossed.rng"))
        != {EXPECTED_ANALYSIS_SEED}
        or _strict_bool(crossed["used_for_decision"], "crossed.used_for_decision").any()
        or set(crossed["bootstrap_unit"].astype(str))
        != {"resampled_training_seeds_and_synchronized_patients_within_outer_fold"}
    ):
        raise ValueError("crossed bootstrap replicate/sync/non-gate contract 漂移")
    _finite_columns(crossed, ("estimate", "ci_low", "ci_high"), "crossed")
    for row in crossed.itertuples(index=False):
        _compare_float(
            float(row.estimate),
            _mean(seed_level[str(row.endpoint)]),
            f"crossed estimate {row.endpoint}",
        )


def _compute_decision_expected(
    seed_level: pd.DataFrame,
    seed_fold: pd.DataFrame,
    uncertainty: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    stability: pd.DataFrame,
    variance: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    static = uncertainty.loc[uncertainty["column"].eq("dS")]
    dynamic = uncertainty.loc[uncertainty["column"].eq("dD")]
    loso = leave_one_out.loc[leave_one_out["scope"].eq("leave_one_seed_out")]
    lofo = leave_one_out.loc[
        leave_one_out["scope"].eq("leave_one_fold_out_across_seed")
    ]
    if len(static) != 1 or len(dynamic) != 1 or len(loso) != 5 or len(lofo) != 5:
        raise ValueError("decision uncertainty/LOSO/LOFO coverage 非冻结值")
    r1_checks = {
        "all_five_dS_positive": bool(seed_level["dS"].gt(0).all()),
        "mean_dS_at_least_0_05": bool(float(static.iloc[0]["mean"]) >= 0.05),
        "seed_t_ci_lower_positive": bool(float(static.iloc[0]["t_ci_low"]) > 0),
        "all_leave_one_seed_out_means_positive": bool(loso["static_mean"].gt(0).all()),
        "all_leave_one_fold_out_recomputed_means_positive": bool(
            lofo["static_mean"].gt(0).all()
        ),
    }
    r2_checks = {
        "at_least_four_of_five_dD_positive": bool(seed_level["dD"].gt(0).sum() >= 4),
        "mean_dD_at_least_0_05": bool(float(dynamic.iloc[0]["mean"]) >= 0.05),
        "seed_t_ci_lower_positive": bool(float(dynamic.iloc[0]["t_ci_low"]) > 0),
        "all_leave_one_seed_out_means_positive": bool(loso["dynamic_mean"].gt(0).all()),
        "all_leave_one_fold_out_recomputed_means_positive": bool(
            lofo["dynamic_mean"].gt(0).all()
        ),
    }
    base = _strict_bool(seed_fold["base_pass"], "seed_fold.base_pass")
    fold_failures = (
        seed_fold.assign(failed=~base).groupby("fold")["failed"].sum().reindex(FOLDS)
    )
    r3_checks = {
        "at_least_23_of_25_base_pass": bool(base.sum() >= 23),
        "each_fold_at_most_two_failures": bool(fold_failures.le(2).all()),
    }
    r4_checks = {
        "all_50_selected_scalars_finite": bool(
            len(stability) == 50
            and _strict_bool(
                stability["selected_scalars_finite"],
                "stability.selected_scalars_finite",
            ).all()
        ),
        "all_50_checkpoint_tensors_finite": bool(
            len(stability) == 50
            and _strict_bool(
                stability["checkpoint_tensors_finite"],
                "stability.checkpoint_tensors_finite",
            ).all()
        ),
        "all_50_representation_std_at_least_0_05": bool(
            len(stability) == 50
            and pd.to_numeric(stability["representation_std"], errors="coerce")
            .ge(0.05)
            .all()
        ),
        "all_50_feature_arrays_finite": bool(
            len(stability) == 50
            and _strict_bool(
                stability["feature_finite"], "stability.feature_finite"
            ).all()
        ),
    }
    gates = {
        "R1_static_reproducibility": all(r1_checks.values()),
        "R2_dynamic_reproducibility": all(r2_checks.values()),
        "R3_optimization_safety": all(r3_checks.values()),
        "R4_no_collapse": all(r4_checks.values()),
    }
    no_single = bool(
        loso[["static_mean", "dynamic_mean"]].gt(0).all().all()
        and lofo[["static_mean", "dynamic_mean"]].gt(0).all().all()
    )
    majority_failure = bool(fold_failures.ge(3).any())
    if all(gates.values()):
        conclusion = "ROBUST"
    elif (
        gates["R1_static_reproducibility"]
        and gates["R2_dynamic_reproducibility"]
        and no_single
        and not majority_failure
    ):
        conclusion = "PROMISING BUT UNSTABLE"
    else:
        conclusion = "NOT ROBUST"
    fold3 = int(fold_failures.loc[3])
    interpretation = (
        "不复现"
        if fold3 == 0
        else (
            "孤立/seed-dependent"
            if fold3 == 1
            else (
                "少数重复、提示冲突但不足以称系统"
                if fold3 == 2
                else "fold 3 systematic conflict"
            )
        )
    )
    dynamic_variance = variance.loc[variance["endpoint"].eq("dynamic_spearman")]
    if len(dynamic_variance) != 1:
        raise ValueError("dynamic variance row 非唯一")
    decision = {
        "schema_version": 1,
        "conclusion": conclusion,
        "gates": gates,
        "gate_details": {
            "R1": r1_checks,
            "R2": r2_checks,
            "R3": r3_checks,
            "R4": r4_checks,
        },
        "no_single_seed_or_fold_drive": no_single,
        "fold_majority_base_failure": majority_failure,
        "base_pass_count": int(base.sum()),
        "base_failure_count": int((~base).sum()),
        "base_failures_by_fold": {
            str(index): int(value) for index, value in fold_failures.items()
        },
        "fold3_failures": fold3,
        "fold3_interpretation_cn": interpretation,
        "old_fold3_reference_degradation": 0.095934,
        "general_seed_fold_optimization_instability": bool(
            int((fold_failures > 0).sum()) >= 2 and not majority_failure
        ),
        "dynamic_variance_dominance": str(dynamic_variance.iloc[0]["dominance"]),
        "pcr_used_in_decision": False,
        "seed_t_ci_is_formal_uncertainty_gate": True,
        "conditional_patient_bootstrap_used_for_decision": False,
        "crossed_bootstrap_used_for_decision": False,
        "thresholds": {
            "R1_mean_dS": 0.05,
            "R2_mean_dD": 0.05,
            "R3_minimum_base_passes": 23,
            "maximum_failures_per_fold": 2,
            "maximum_base_degradation": 0.05,
            "minimum_representation_std": 0.05,
        },
    }
    gate_rows: list[dict[str, Any]] = []
    names = {
        "R1": "R1_static_reproducibility",
        "R2": "R2_dynamic_reproducibility",
        "R3": "R3_optimization_safety",
        "R4": "R4_no_collapse",
    }
    for gate, checks in (
        ("R1", r1_checks),
        ("R2", r2_checks),
        ("R3", r3_checks),
        ("R4", r4_checks),
    ):
        gate_rows.extend(
            {"gate": gate, "criterion": criterion, "passed": passed}
            for criterion, passed in checks.items()
        )
        gate_rows.append(
            {"gate": gate, "criterion": "overall", "passed": gates[names[gate]]}
        )
    return decision, pd.DataFrame(gate_rows)


def _compare_decision_payload(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for name, value in expected.items():
        if name not in actual:
            raise ValueError(f"decision.json 缺字段: {name}")
        observed = actual[name]
        if isinstance(value, dict):
            if observed != value:
                raise ValueError(f"decision.json.{name} 与独立机械重算不一致")
        else:
            _compare_scalar(observed, value, f"decision.json.{name}")
    if actual.get("formal_analysis") is not True:
        raise ValueError("decision.json formal_analysis 非 true")
    if actual.get("bootstrap_rng_seed") != EXPECTED_ANALYSIS_SEED:
        raise ValueError("decision.json bootstrap RNG 漂移")
    if (
        actual.get("conditional_bootstrap_replicates")
        != EXPECTED_CONDITIONAL_REPLICATES
    ):
        raise ValueError("decision.json conditional replicate 漂移")
    if actual.get("crossed_bootstrap_replicates") != EXPECTED_CROSSED_REPLICATES:
        raise ValueError("decision.json crossed replicate 漂移")


def _validate_coverage(coverage: pd.DataFrame) -> None:
    expected = {
        "training_seeds": 5,
        "models": 2,
        "seed_model_fold_cells": 50,
        "checkpoints": 50,
        "histories": 50,
        "training_selections": 50,
        "features": 50,
        "probe_prediction_files": 50,
        "pcr_prediction_files": 50,
        "probe_prediction_rows": EXPECTED_PROBE_ROWS,
        "pcr_prediction_rows": EXPECTED_PCR_ROWS,
        "probe_downstream_selections": 50,
        "pcr_downstream_selections": 50,
        "registered_png": 11,
    }
    _require_columns(coverage, ("asset", "expected", "observed", "passed"), "coverage")
    if len(coverage) != len(expected) or set(coverage["asset"].astype(str)) != set(
        expected
    ):
        raise ValueError("coverage asset schema/rows 错误")
    if coverage["asset"].astype(str).duplicated().any():
        raise ValueError("coverage asset 重复")
    for row in coverage.itertuples(index=False):
        value = expected[str(row.asset)]
        if (
            int(row.expected) != value
            or int(row.observed) != value
            or not _strict_bool_value(row.passed, f"coverage.{row.asset}.passed")
        ):
            raise ValueError(f"coverage.{row.asset} 与独立固定计数不一致")


def _validate_required_figure_names(frame: pd.DataFrame) -> None:
    _require_columns(
        frame,
        ("figure", "path", "sha256", "bytes", "width", "height", "decodable"),
        "figure_manifest",
    )
    figures = frame["figure"].astype(str)
    paths = frame["path"].astype(str)
    if (
        len(frame) != 11
        or figures.duplicated().any()
        or paths.duplicated().any()
        or set(figures) != set(REQUIRED_FIGURES)
    ):
        raise ValueError("figure manifest 必须恰为 11 张 distinct required figures")


def _validate_figures(frame: pd.DataFrame) -> None:
    _validate_required_figure_names(frame)
    for row in frame.itertuples(index=False):
        filename = str(row.figure)
        expected_path = _relative(ROOT / "figures" / "final" / filename)
        if str(row.path) != expected_path:
            raise ValueError(f"figure {filename} 不在 figures/final 固定路径")
        _require_sha((row.sha256,), f"figure {filename}.sha256")
        if not _strict_bool_value(row.decodable, f"figure {filename}.decodable"):
            raise ValueError(f"figure {filename} manifest decodable 非 true")
        path = _resolve_public_path(str(row.path))
        if (
            not path.is_file()
            or path.suffix.lower() != ".png"
            or path.stat().st_size != int(row.bytes)
            or file_sha256(path) != str(row.sha256)
        ):
            raise ValueError(f"figure {filename} live path/hash/bytes 失败")
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError(f"figure {filename} 实际格式非 PNG")
                image.verify()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                metadata = json.dumps(image.info, ensure_ascii=False, default=str)
        except Exception as exc:
            raise ValueError(f"figure {filename} 无法完整解码") from exc
        if (
            width <= 0
            or height <= 0
            or width != int(row.width)
            or height != int(row.height)
        ):
            raise ValueError(f"figure {filename} dimensions 未闭合")
        _scan_public_text(metadata, f"PNG metadata {filename}")


def _validate_report_text(text: str, conclusion: str) -> None:
    if not text:
        raise ValueError("final_report.md 缺失/为空")
    cjk_count = len(CJK_RE.findall(text))
    alphabetic = sum(character.isalpha() for character in text)
    if cjk_count < 250 or cjk_count / max(alphabetic, 1) < 0.25:
        raise ValueError(
            f"final_report.md 中文内容不足: CJK={cjk_count}, ratio={cjk_count/max(alphabetic,1):.3f}"
        )
    if "## 七个冻结问题的回答" not in text or not all(
        question in text for question in REPORT_QUESTIONS
    ):
        raise ValueError("final_report.md 未逐项回答七个冻结问题")
    evidence_links = (
        "../metrics/final/seed_level_robustness.csv",
        "../metrics/final/seed_uncertainty.csv",
        "../metrics/final/leave_one_out_sensitivity.csv",
        "../metrics/final/decision.json",
    )
    if not all(link in text for link in evidence_links):
        raise ValueError("final_report.md 缺机器证据链接")
    match = re.search(r"预注册机械结论为\s*\*\*([^*]+)\*\*", text)
    if match is None or match.group(1).strip() != conclusion:
        raise ValueError("final_report.md 结论与 decision.json 不一致")
    if (
        "pcr_used_in_decision=false" not in text.lower()
        or "没有改变正式结论" not in text
    ):
        raise ValueError("final_report.md 未明确声明 pCR secondary/non-gate")


def _scan_public_text(text: str, label: str) -> None:
    if ABSOLUTE_PATH_RE.search(text):
        raise ValueError(f"公开文件含真实绝对路径: {label}")
    if PATIENT_VALUE_RE.search(text):
        raise ValueError(f"公开文件含患者标识值: {label}")
    if SECRET_RE.search(text):
        raise ValueError(f"公开文件疑似含 secret: {label}")


def _audit_public_privacy(final: Path, report: Path) -> dict[str, Any]:
    public_files = sorted(
        path
        for path in final.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}
    )
    if report.is_file():
        public_files.append(report)
    acceptance = ROOT / "metrics" / "acceptance_check.json"
    if acceptance.is_file():
        public_files.append(acceptance)
    forbidden_columns = {"patientid", "subjectid", "mrn", "caseid", "patientidentifier"}
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        _scan_public_text(text, _relative(path))
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, low_memory=False)
            normalised = {
                re.sub(r"[^a-z0-9]", "", str(column).lower())
                for column in frame.columns
            }
            if normalised.intersection(forbidden_columns):
                raise ValueError(f"公开 CSV 含患者标识列: {_relative(path)}")

    command = subprocess.run(
        ["git", "ls-files", "-z", "--", _relative(ROOT)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    tracked = [value for value in command.stdout.decode("utf-8").split("\0") if value]
    prefix = _relative(ROOT) + "/"
    forbidden_tracked: list[str] = []
    for relative in tracked:
        local = relative.removeprefix(prefix)
        if local.endswith("/.gitkeep") or local in {
            "checkpoints/.gitkeep",
            "features/.gitkeep",
            "predictions/.gitkeep",
            "logs/.gitkeep",
            "metrics/.gitkeep",
            "figures/.gitkeep",
        }:
            continue
        if local.startswith(("checkpoints/", "features/", "predictions/", "logs/")):
            forbidden_tracked.append(relative)
        elif local.startswith("metrics/") and not (
            local.startswith("metrics/final/")
            or local == "metrics/acceptance_check.json"
        ):
            forbidden_tracked.append(relative)
        elif local.startswith("figures/") and not local.startswith("figures/final/"):
            forbidden_tracked.append(relative)
    if forbidden_tracked:
        raise ValueError(f"Git 已跟踪禁止资产: {forbidden_tracked[:10]}")
    return {
        "public_text_files_scanned": len(set(public_files)),
        "git_tracked_files": len(tracked),
    }


def _expect_reject(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (ValueError, KeyError, TypeError, AssertionError):
        return True
    return False


def _run_verifier_self_test() -> dict[str, bool]:
    full_grid = pd.DataFrame(
        [(seed, model, fold) for seed, model, fold in sorted(EXPECTED_GRID)],
        columns=("seed_base", "model", "fold"),
    )
    duplicate_grid = pd.concat(
        [full_grid.iloc[:-1], full_grid.iloc[[0]]], ignore_index=True
    )
    figure_rows = pd.DataFrame(
        {
            "figure": REQUIRED_FIGURES,
            "path": [f"figures/final/{name}" for name in REQUIRED_FIGURES],
            "sha256": ["0" * 64] * 11,
            "bytes": [1] * 11,
            "width": [1] * 11,
            "height": [1] * 11,
            "decodable": [True] * 11,
        }
    )
    duplicate_figures = figure_rows.copy()
    duplicate_figures.loc[10, "figure"] = duplicate_figures.loc[0, "figure"]
    expected_table = pd.DataFrame(
        [{"seed_base": seed, "dS": float(index)} for index, seed in enumerate(SEEDS)]
    )
    missing_table = expected_table.iloc[:-1].copy()
    synthetic_matrix = np.asarray(
        [[index, (index % 5) - 2] for index in range(-12, 13)], dtype=float
    )
    synthetic_mean, synthetic_scale = _standard_scaler_stats(synthetic_matrix)
    synthetic_scaled = (synthetic_matrix - synthetic_mean) / synthetic_scale
    ridge_reference = Ridge(alpha=1.0, solver="lsqr", tol=1e-8, max_iter=10000).fit(
        synthetic_scaled, synthetic_matrix[:, 0] + 0.2 * synthetic_matrix[:, 1]
    )
    ridge_test_fitted = Ridge(alpha=1.0, solver="lsqr", tol=1e-8, max_iter=10000).fit(
        synthetic_scaled, -synthetic_matrix[:, 0] + 0.2 * synthetic_matrix[:, 1]
    )
    binary = (synthetic_matrix[:, 0] >= 0).astype(int)
    logistic_reference = LogisticRegression(
        l1_ratio=0.0,
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=20000,
        tol=1e-7,
        random_state=7,
    ).fit(synthetic_scaled, binary)
    logistic_test_fitted = LogisticRegression(
        l1_ratio=0.0,
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=20000,
        tol=1e-7,
        random_state=7,
    ).fit(synthetic_scaled, 1 - binary)
    checks = {
        "strict_false_string_preserved": _strict_bool_value("False", "test") is False,
        "illegal_boolean_rejected": _expect_reject(
            lambda: _strict_bool(pd.Series(["FAIL"]), "tampered bool")
        ),
        "nan_boolean_rejected": _expect_reject(
            lambda: _strict_bool(pd.Series([np.nan]), "tampered bool")
        ),
        "duplicate_grid_rejected": _expect_reject(
            lambda: _require_grid(duplicate_grid, "tampered grid")
        ),
        "missing_grid_rejected": _expect_reject(
            lambda: _require_grid(full_grid.iloc[:-1], "tampered grid")
        ),
        "missing_metric_row_rejected": _expect_reject(
            lambda: _compare_table(
                missing_table,
                expected_table,
                ("seed_base",),
                "tampered metrics",
            )
        ),
        "tampered_live_prediction_rejected": _expect_reject(
            lambda: _compare_vector(
                np.asarray([0.1, 0.2]),
                np.asarray([0.1, 0.2001]),
                "tampered saved-parameter prediction",
            )
        ),
        "malformed_saved_coefficient_rejected": _expect_reject(
            lambda: _json_vector("[1.0, false]", 2, "tampered coefficient")
        ),
        "test_fitted_ridge_rejected_by_train_refit": _expect_reject(
            lambda: _compare_vector(
                ridge_test_fitted.coef_,
                ridge_reference.coef_,
                "test-fitted Ridge",
            )
        ),
        "test_fitted_logistic_rejected_by_train_refit": _expect_reject(
            lambda: _compare_vector(
                logistic_test_fitted.coef_.reshape(-1),
                logistic_reference.coef_.reshape(-1),
                "test-fitted logistic",
            )
        ),
        "state_dict_model_state_mismatch_rejected": _expect_reject(
            lambda: _validate_state_dict_mirrors(
                {
                    "state_dict": {"weight": torch.tensor([1.0])},
                    "model_state": {"weight": torch.tensor([2.0])},
                },
                "tampered checkpoint",
            )
        ),
        "duplicate_required_figure_rejected": _expect_reject(
            lambda: _validate_required_figure_names(duplicate_figures)
        ),
        "english_stub_report_rejected": _expect_reject(
            lambda: _validate_report_text(
                "七个冻结问题 Static FTV Observed ΔFTV fold 3 variance "
                "Factorized Grounded Response State optimization",
                "ROBUST",
            )
        ),
        "wrong_report_conclusion_rejected": _expect_reject(
            lambda: _validate_report_text(
                "中" * 300
                + "\n## 七个冻结问题的回答\n"
                + "\n".join(REPORT_QUESTIONS)
                + "\n../metrics/final/seed_level_robustness.csv"
                + "\n../metrics/final/seed_uncertainty.csv"
                + "\n../metrics/final/leave_one_out_sensitivity.csv"
                + "\n../metrics/final/decision.json"
                + "\n预注册机械结论为 **NOT ROBUST**"
                + "\npcr_used_in_decision=false，没有改变正式结论",
                "ROBUST",
            )
        ),
        "absolute_path_rejected": _expect_reject(
            lambda: _scan_public_text("source=/data/private/file.csv", "tampered")
        ),
        "patient_identifier_rejected": _expect_reject(
            lambda: _scan_public_text("ACRIN-6698-123456", "tampered")
        ),
        "secret_rejected": _expect_reject(
            lambda: _scan_public_text(
                "api_key=abcdefghijklmnopqrstuvwxyz123456", "tampered"
            )
        ),
        "decision_signature_excludes_pcr": "pcr"
        not in _compute_decision_expected.__code__.co_varnames[
            : _compute_decision_expected.__code__.co_argcount
        ],
    }
    if not all(checks.values()):
        raise AssertionError(f"verifier negative self-tests failed: {checks}")
    return checks


def _corr(target: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    result = (
        spearmanr(target, prediction)
        if kind == "spearman"
        else pearsonr(target, prediction)
    )
    return float(result.statistic)


def _regression(part: pd.DataFrame) -> dict[str, float]:
    target = part["y_true"].to_numpy(dtype=float)
    prediction = part["y_pred"].to_numpy(dtype=float)
    baseline = part["b0_prediction"].to_numpy(dtype=float)
    rmse = math.sqrt(mean_squared_error(target, prediction))
    b0 = math.sqrt(mean_squared_error(target, baseline))
    target_var = float(np.var(target))
    prediction_var = float(np.var(prediction))
    return {
        "n": int(len(part)),
        "n_patients": int(part["patient_id"].nunique()),
        "n_folds": int(part["fold"].nunique()),
        "spearman": _corr(target, prediction, "spearman"),
        "pearson": _corr(target, prediction, "pearson"),
        "r2": float(r2_score(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(rmse),
        "b0_rmse": float(b0),
        "rmse_gain_over_b0": float((b0 - rmse) / b0),
        "target_variance": target_var,
        "prediction_variance": prediction_var,
        "prediction_target_variance_ratio": prediction_var / target_var,
    }


def _classification(part: pd.DataFrame) -> dict[str, float]:
    target = part["y_true"].to_numpy(dtype=int)
    probability = part["probability"].to_numpy(dtype=float)
    label = part["predicted_label"].to_numpy(dtype=int)
    if set(target) != {0, 1}:
        raise ValueError("pCR metric group 缺一个 class")
    return {
        "n": int(len(part)),
        "n_patients": int(part["patient_id"].nunique()),
        "n_folds": int(part["fold"].nunique()),
        "positive": int(target.sum()),
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "accuracy": float(accuracy_score(target, label)),
        "sensitivity": float(recall_score(target, label, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(target, label, pos_label=0, zero_division=0)),
    }


def _compare_float(
    actual: float, expected: float, label: str, tolerance: float = 1e-10
) -> None:
    if math.isnan(float(actual)) and math.isnan(float(expected)):
        return
    if not math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(f"{label} 重算不一致: {actual} != {expected}")


def verify(output_name: str = "final") -> tuple[dict[str, Any], dict[str, Any]]:
    """从原始正式资产独立复算并验收；任何 schema/闭环问题立即拒绝。"""

    if output_name != "final":
        raise ValueError("正式验收只允许固定 output-name=final")
    final = ROOT / "metrics" / "final"
    report = ROOT / "reports" / "final_report.md"
    for name in REQUIRED_TABLES:
        if not (final / name).is_file():
            raise FileNotFoundError(final / name)
    for name in (
        "decision.json",
        "aggregation_summary.json",
        "analysis_acceptance_evidence.json",
    ):
        if not (final / name).is_file():
            raise FileNotFoundError(final / name)
    if not report.is_file():
        raise FileNotFoundError(report)

    tables = {
        name: pd.read_csv(final / name, low_memory=False) for name in REQUIRED_TABLES
    }
    decision = _read_json(final / "decision.json")
    summary = _read_json(final / "aggregation_summary.json")
    evidence = _read_json(final / "analysis_acceptance_evidence.json")
    checks: list[dict[str, Any]] = []

    verifier_tests = _run_verifier_self_test()
    variance_tests = variance_decomposition_self_test()
    checks.append(
        _check(
            "1_verifier_negative_and_variance_selftests",
            all(verifier_tests.values()) and all(variance_tests.values()),
            {"negative": verifier_tests, "variance": variance_tests},
        )
    )

    freeze = _verify_freeze_provenance()
    checks.append(_check("2_frozen_plan_and_source_provenance", True, freeze))

    input_index, history_index, selection_index = _verify_manifests(tables)
    _validate_coverage(tables["coverage.csv"])
    checks.append(
        _check(
            "3_exact_fixed_manifest_grid_and_live_hashes",
            True,
            {
                "checkpoints": 50,
                "histories": 50,
                "training_selections": 50,
                "features": 50,
                "probe_files": 50,
                "pcr_files": 50,
            },
        )
    )

    stability, assets = _audit_assets(
        tables, input_index, history_index, selection_index
    )
    checks.append(
        _check(
            "4_checkpoint_history_selection_feature_schema_hash_closure",
            True,
            {
                "asset_cells": len(assets),
                "feature_shape": [808, 4, 192],
                "checkpoint_tensor_audit": "recursive_live",
                "canonical_fold_fingerprints": 5,
            },
        )
    )

    downstream, selections = _audit_downstream(tables, input_index, assets)
    checks.append(
        _check(
            "5_downstream_validation_only_selection_and_test_once",
            len(downstream) == 100,
            {"probe_selections": 50, "pcr_selections": 50},
        )
    )

    probe, pcr = _load_validated_predictions(
        tables["prediction_manifest.csv"], assets, selections
    )
    checks.append(
        _check(
            "6_live_target_feature_saved_parameter_prediction_recomputation",
            len(probe) == EXPECTED_PROBE_ROWS and len(pcr) == EXPECTED_PCR_ROWS,
            {
                "probe_rows": len(probe),
                "pcr_rows": len(pcr),
                "probe_patients_per_seed": 375,
                "pcr_patients_per_seed": 808,
                "probe_recomputed_fields": [
                    "y_true",
                    "y_true_standardized",
                    "b0_prediction",
                    "y_pred",
                ],
                "pcr_recomputed_fields": ["probability", "predicted_label"],
            },
        )
    )

    recomputed = _recompute_metric_tables(probe, pcr, stability)
    _compare_recomputed_metric_tables(tables, recomputed)
    checks.append(
        _check(
            "7_all_seed_fold_probe_and_pcr_metrics_recomputed",
            True,
            {
                "probe_seed_cells": len(recomputed["probe_seed_cell_metrics.csv"]),
                "probe_seed_fold_cells": len(
                    recomputed["probe_seed_fold_cell_metrics.csv"]
                ),
                "pcr_seed_model_cells": len(recomputed["pcr_seed_model_metrics.csv"]),
                "seed_fold_effects": len(recomputed["seed_fold_effects.csv"]),
            },
        )
    )

    seed_level = recomputed["seed_level_robustness.csv"]
    uncertainty = _recompute_seed_uncertainty(seed_level)
    _compare_table(
        tables["seed_uncertainty.csv"],
        uncertainty,
        ("column",),
        "seed_uncertainty",
        tuple(uncertainty.columns),
        tolerance=1e-12,
    )
    leave_one_out = _recompute_leave_one_out(probe, seed_level)
    _compare_table(
        tables["leave_one_out_sensitivity.csv"],
        leave_one_out,
        ("scope", "omitted_seed", "omitted_fold", "seed_base"),
        "leave_one_out_sensitivity",
        tuple(leave_one_out.columns),
    )
    variance = _recompute_variance_table(recomputed["seed_fold_effects.csv"])
    _compare_table(
        tables["variance_decomposition.csv"],
        variance,
        ("endpoint",),
        "variance_decomposition",
        tuple(variance.columns),
        tolerance=1e-12,
    )
    checks.append(
        _check(
            "8_seed_uncertainty_loso_lofo_and_variance_recomputed",
            True,
            {"uncertainty_rows": len(uncertainty), "loo_rows": len(leave_one_out)},
        )
    )

    _validate_bootstrap_tables(
        tables["conditional_seed_bootstrap_ci.csv"],
        tables["crossed_bootstrap_ci.csv"],
        seed_level,
        recomputed["pcr_secondary_seed_metrics.csv"],
    )
    checks.append(
        _check(
            "9_bootstrap_row_rng_and_synchronized_draw_contract",
            True,
            {
                "conditional_rows": 55,
                "conditional_replicates": EXPECTED_CONDITIONAL_REPLICATES,
                "crossed_rows": 3,
                "crossed_replicates": EXPECTED_CROSSED_REPLICATES,
                "rng_seed": EXPECTED_ANALYSIS_SEED,
            },
        )
    )

    expected_decision, expected_gates = _compute_decision_expected(
        seed_level,
        recomputed["seed_fold_effects.csv"],
        uncertainty,
        leave_one_out,
        stability,
        variance,
    )
    _compare_decision_payload(decision, expected_decision)
    _compare_table(
        tables["decision_gates.csv"],
        expected_gates,
        ("gate", "criterion"),
        "decision_gates",
        tuple(expected_gates.columns),
    )
    if decision.get("pcr_used_in_decision") is not False:
        raise ValueError("decision pCR 必须明确不入 gate")
    checks.append(
        _check(
            "10_R1_R4_decision_gates_mechanically_recomputed_without_pcr",
            True,
            {
                "conclusion": expected_decision["conclusion"],
                "gates": expected_decision["gates"],
            },
        )
    )

    analysis_path = ROOT / "src" / "dgrs" / "analysis.py"
    analysis_hash = file_sha256(analysis_path)
    required_summary = {
        "schema_version": 1,
        "status": "complete",
        "formal_analysis": True,
        "conclusion": expected_decision["conclusion"],
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "folds": list(FOLDS),
        "probe_rows_consumed_in_memory": EXPECTED_PROBE_ROWS,
        "pcr_rows_consumed_in_memory": EXPECTED_PCR_ROWS,
        "public_patient_rows": 0,
        "conditional_bootstrap_replicates": EXPECTED_CONDITIONAL_REPLICATES,
        "crossed_bootstrap_replicates": EXPECTED_CROSSED_REPLICATES,
        "bootstrap_rng_seed": EXPECTED_ANALYSIS_SEED,
        "seed_t_ci_is_formal_gate": True,
        "pcr_used_in_decision": False,
        "figures": 11,
        "registered_issues": 0,
        "analysis_source_sha256": analysis_hash,
    }
    for name, expected in required_summary.items():
        observed = summary.get(name)
        if isinstance(expected, list):
            if observed != expected:
                raise ValueError(f"aggregation_summary.{name} 漂移")
        else:
            _compare_scalar(observed, expected, f"aggregation_summary.{name}")
    required_evidence = {
        "schema_version": 1,
        "formal_analysis": True,
        "public_tables_contain_patient_rows": False,
        "decision_recomputed_from_unrounded_tables": True,
        "pcr_used_in_decision": False,
        "registered_issues": 0,
        "coverage_all_passed": True,
        "figure_count": 11,
        "analysis_source_sha256": analysis_hash,
        "plan_sha256": FROZEN_PLAN_SHA256,
    }
    for name, expected in required_evidence.items():
        _compare_scalar(
            evidence.get(name), expected, f"analysis_acceptance_evidence.{name}"
        )
    if (
        evidence.get("prediction_rows_recomputed_in_memory")
        != {
            "probe": EXPECTED_PROBE_ROWS,
            "pcr": EXPECTED_PCR_ROWS,
        }
        or evidence.get("variance_synthetic_tests") != variance_tests
    ):
        raise ValueError("analysis_acceptance_evidence rows/variance tests 漂移")
    if decision.get("analysis_source_sha256") != analysis_hash:
        raise ValueError("decision analysis source SHA 未闭合")
    issues = tables["issues.csv"]
    _require_columns(issues, ("severity", "asset", "issue"), "issues")
    if not issues.empty:
        raise ValueError("issues.csv 存在 registered issues")
    checks.append(
        _check(
            "11_summary_evidence_source_and_zero_issue_closure",
            True,
            {"analysis_source_sha256": analysis_hash, "issues": 0},
        )
    )

    _validate_figures(tables["figure_manifest.csv"])
    checks.append(
        _check(
            "12_eleven_required_distinct_png_hash_decode",
            True,
            {"figures": list(REQUIRED_FIGURES)},
        )
    )

    report_text = report.read_text(encoding="utf-8")
    _validate_report_text(report_text, expected_decision["conclusion"])
    checks.append(
        _check(
            "13_chinese_report_seven_answers_evidence_and_decision_match",
            True,
            {"cjk_characters": len(CJK_RE.findall(report_text)), "questions": 7},
        )
    )

    privacy = _audit_public_privacy(final, report)
    checks.append(_check("14_public_privacy_and_git_tracking_gate", True, privacy))

    passed = all(item["passed"] for item in checks)
    output = {
        "schema_version": 2,
        "status": "accepted" if passed else "rejected",
        "passed": passed,
        "conclusion": expected_decision["conclusion"],
        "criteria": checks,
        "public_patient_rows": 0,
        "pcr_used_in_decision": False,
        "formal_output_name": "final",
        "frozen_plan_sha256": FROZEN_PLAN_SHA256,
    }
    _scan_public_text(
        json.dumps(output, ensure_ascii=False, default=str), "acceptance_check payload"
    )
    compact = {
        "passed": passed,
        "criteria_passed": sum(item["passed"] for item in checks),
        "criteria_total": len(checks),
        "conclusion": expected_decision["conclusion"],
    }
    return output, compact


def _atomic_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有 acceptance: {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=lambda value: (
                    value.item() if isinstance(value, np.generic) else str(value)
                ),
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="final")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "metrics" / "acceptance_check.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        analysis = run_self_test()
        variance = variance_decomposition_self_test()
        verifier = _run_verifier_self_test()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "analysis": analysis,
                    "variance": variance,
                    "verifier_negative_tests": verifier,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    fixed_output = (ROOT / "metrics" / "acceptance_check.json").resolve()
    if args.output.resolve() != fixed_output:
        raise SystemExit("正式验收输出固定为 metrics/acceptance_check.json")
    try:
        output, compact = verify(args.output_name)
        if not args.no_write:
            _atomic_json(args.output.resolve(), output, args.overwrite)
    except Exception as exc:
        print(f"验收失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if not compact["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
