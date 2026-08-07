"""五折正式结果的严格聚合、统计区间与中文图表。

该模块只消费 ``evaluate_fold.py`` 与 ``run_controls.py`` 已保存的结果，
不会重新拟合 image readout，也不会用 test 数据选择任何超参数。默认要求
M0/M1/M2 各有且仅有五个无歧义 fold；``allow_partial`` 仅用于 smoke/故障
诊断，缺失输入会写入清单并在汇总状态中明确标记，绝不会补造数值。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .data import patient_hash, read_raw_radiomics
from .transforms import TRANSFORM_SPEC_VERSION, raw_targets_hash


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = EXPERIMENT_ROOT / "metrics" / "evaluation"
CONTROL_METRIC_ROOT = EXPERIMENT_ROOT / "metrics" / "controls"
CONTROL_PREDICTION_ROOT = EXPERIMENT_ROOT / "predictions" / "controls"
MODEL_ORDER = ("m0", "m1", "m2")
DECISION_ORDER = ("T0", "T0-T1", "T0-T2")
TRANSITION_ORDER = ("T0→T1", "T1→T2", "T2→T3")
FEATURE_ORDER = ("ftv", "sphericity", "ld", "bpe")
PERTURBATION_ORDER = ("repeated_t0", "temporal_shuffle_t1_t2")
CLASSIFICATION_METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
)
METRIC_CN = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "accuracy": "准确率",
    "sensitivity": "敏感度",
    "specificity": "特异度",
}
MODEL_DISPLAY = {"m0": "M0", "m1": "M1", "m2": "M2"}
FEATURE_DISPLAY = {
    "ftv": "FTV",
    "sphericity": "Sphericity",
    "ld": "LD",
    "bpe": "BPE",
}
PERTURBATION_DISPLAY = {
    "repeated_t0": "Repeated-T0",
    "temporal_shuffle_t1_t2": "Temporal shuffle",
}
ROI_ASSISTED_SHORT_CN = "ROI辅助 image-only（DCE7+ROI mask）"
ROI_ASSISTED_FULL_CN = (
    "ROI辅助 image-only（DCE7+ROI mask；不含独立 geometry/clinical/"
    "treatment/radiomics 输入）"
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class AggregationInputError(RuntimeError):
    """输入缺失、歧义或违反锁定结果契约。"""


@dataclass(frozen=True)
class AggregationConfig:
    """聚合参数；正式默认值对应预注册的三个 run。"""

    run_names: Mapping[str, str]
    output_tag: str
    allow_partial: bool = False
    bootstrap_replicates: int = 2000
    seed: int = 20260806
    controls_name: str | None = None
    fold_manifest: Path = Path(
        "/data/data/Preprocessed/I-SPY2/"
        "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
        "matched_patient_cv_splits_seed2026.csv"
    )
    radiomics_raw_targets: Path = (
        EXPERIMENT_ROOT / "data_audit" / "radiomics_transition_targets_raw.csv"
    )
    radiomics_transform_dir: Path = EXPERIMENT_ROOT / "configs"

    def validate(self) -> None:
        if set(self.run_names) != set(MODEL_ORDER):
            raise ValueError("run_names 必须恰好包含 m0、m1、m2")
        for mode, run_name in self.run_names.items():
            if not _SAFE_NAME.fullmatch(str(run_name)):
                raise ValueError(f"{mode} run name 不安全: {run_name!r}")
        if not _SAFE_NAME.fullmatch(self.output_tag):
            raise ValueError("output_tag 只能包含字母、数字、点、下划线和短横线")
        if self.controls_name is not None and not _SAFE_NAME.fullmatch(
            self.controls_name
        ):
            raise ValueError("controls_name 格式非法")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates 至少为 100")


@dataclass(frozen=True)
class EvaluationSource:
    mode: str
    run_name: str
    fold: int
    checkpoint_sha256: str
    summary_path: Path
    metric_dir: Path
    prediction_dir: Path
    summary: Mapping[str, Any]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            _json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False
        )
        stream.write("\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - 具体异常会在消息中保留
        raise AggregationInputError(f"无法读取 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AggregationInputError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise AggregationInputError(f"无法读取 {label}: {path}: {exc}") from exc
    missing = set(required).difference(frame.columns)
    if missing:
        raise AggregationInputError(f"{label} 缺少字段 {sorted(missing)}: {path}")
    if frame.empty:
        raise AggregationInputError(f"{label} 为空: {path}")
    return frame


def _record_issue(
    issues: list[dict[str, str]], category: str, message: str, path: Path | None = None
) -> None:
    issues.append(
        {
            "类别_category": category,
            "说明_message": message,
            "路径_path": "" if path is None else str(path),
        }
    )


def _missing_or_raise(
    config: AggregationConfig,
    issues: list[dict[str, str]],
    message: str,
    path: Path | None = None,
) -> None:
    if config.allow_partial:
        _record_issue(issues, "缺失输入_missing_input", message, path)
        return
    suffix = "" if path is None else f": {path}"
    raise AggregationInputError(message + suffix)


def discover_evaluations(
    config: AggregationConfig, issues: list[dict[str, str]]
) -> list[EvaluationSource]:
    """递归发现并验证每个 run/fold 的唯一 evaluation namespace。"""

    sources: list[EvaluationSource] = []
    for mode in MODEL_ORDER:
        run_name = str(config.run_names[mode])
        run_root = EVALUATION_ROOT / run_name
        if not run_root.is_dir():
            _missing_or_raise(
                config,
                issues,
                f"{mode.upper()} 找不到 evaluation run {run_name}",
                run_root,
            )
            continue
        by_fold: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
        for path in sorted(run_root.rglob("evaluation_summary.json")):
            payload = _read_json(path)
            if payload.get("run_name") != run_name:
                raise AggregationInputError(
                    f"evaluation_summary run_name 与目录不一致: {path}"
                )
            if payload.get("model_name") != mode:
                raise AggregationInputError(
                    f"{run_name} 预期 model_name={mode}，实际 "
                    f"{payload.get('model_name')!r}: {path}"
                )
            fold = payload.get("fold")
            if (
                isinstance(fold, bool)
                or not isinstance(fold, int)
                or fold not in range(5)
            ):
                raise AggregationInputError(f"fold 非法: {path}: {fold!r}")
            sha = str(payload.get("checkpoint_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise AggregationInputError(f"checkpoint SHA256 非法: {path}")
            by_fold.setdefault(fold, []).append((path, payload))

        if not by_fold:
            _missing_or_raise(
                config, issues, f"{run_name} 没有 evaluation_summary.json", run_root
            )
            continue
        ambiguous = {fold: items for fold, items in by_fold.items() if len(items) != 1}
        if ambiguous:
            details = "; ".join(
                f"fold {fold}: {[str(path.parent.name) for path, _ in items]}"
                for fold, items in sorted(ambiguous.items())
            )
            raise AggregationInputError(
                f"{run_name} 同一 fold 存在多个 evaluation，拒绝静默选择：{details}"
            )
        found_folds = set(by_fold)
        missing_folds = set(range(5)).difference(found_folds)
        if missing_folds:
            _missing_or_raise(
                config,
                issues,
                f"{run_name} 缺少 fold {sorted(missing_folds)}；正式聚合要求 0–4",
                run_root,
            )
        for fold, ((summary_path, payload),) in sorted(by_fold.items()):
            metric_dir = summary_path.parent
            prediction_dir = (
                EXPERIMENT_ROOT
                / "predictions"
                / run_name
                / f"fold_{fold}"
                / metric_dir.name
            )
            declared = payload.get("outputs", {}).get("prediction_dir")
            if declared is not None:
                declared_path = Path(str(declared)).expanduser().resolve()
                if declared_path != prediction_dir.resolve():
                    raise AggregationInputError(
                        f"summary 声明的 prediction_dir 与锁定目录不一致: {summary_path}"
                    )
            sources.append(
                EvaluationSource(
                    mode=mode,
                    run_name=run_name,
                    fold=fold,
                    checkpoint_sha256=str(payload["checkpoint_sha256"]),
                    summary_path=summary_path,
                    metric_dir=metric_dir,
                    prediction_dir=prediction_dir,
                    summary=payload,
                )
            )
    if not sources:
        raise AggregationInputError("没有任何可聚合的 evaluation 输入")
    return sources


def _validate_method_input_contract(
    sources: Sequence[EvaluationSource],
    config: AggregationConfig,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    """核验所有 evaluation 对影像输入的冻结声明并生成准确方法标签。"""

    contracts: list[tuple[int, bool, tuple[str, ...], tuple[str, ...]]] = []
    required_forbidden = {"clinical", "treatment", "geometry_descriptor", "radiomics"}
    for source in sources:
        payload = source.summary.get("architecture_contract")
        if not isinstance(payload, Mapping):
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} summary 缺 architecture_contract",
                source.summary_path,
            )
            continue
        channels = payload.get("image_channels")
        roi_assisted = payload.get("roi_assisted")
        inputs = payload.get("inputs")
        forbidden = payload.get("forbidden_inputs_absent")
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
            or not isinstance(roi_assisted, bool)
            or not isinstance(inputs, list)
            or not all(isinstance(value, str) for value in inputs)
            or not isinstance(forbidden, list)
            or not all(isinstance(value, str) for value in forbidden)
        ):
            raise AggregationInputError(
                f"architecture_contract 字段类型非法: {source.summary_path}"
            )
        if not required_forbidden.issubset(set(forbidden)):
            raise AggregationInputError(
                f"architecture_contract 未声明排除全部禁用输入: {source.summary_path}"
            )
        contracts.append(
            (channels, roi_assisted, tuple(sorted(inputs)), tuple(sorted(forbidden)))
        )
    if not contracts:
        return {
            "verification_complete": False,
            "image_channels": None,
            "roi_assisted": None,
            "method_label_cn": "影像 readout（输入契约缺失）",
            "input_definition_cn": "输入契约缺失；仅允许 partial 诊断",
        }
    if len(set(contracts)) != 1:
        raise AggregationInputError(
            "M0/M1/M2 或各 fold 的 architecture_contract 不一致"
        )
    channels, roi_assisted, inputs, forbidden = contracts[0]
    if roi_assisted and channels != 8:
        raise AggregationInputError(
            f"ROI-assisted DCE 契约预期 DCE7+ROI mask 共 8 通道，实际 {channels}"
        )
    if roi_assisted:
        method_label = ROI_ASSISTED_FULL_CN
        input_definition = "DCE7 强度通道 + 第8通道二值 ROI mask"
    else:
        method_label = (
            f"image-only（{channels} 个影像通道；不含独立 geometry/clinical/"
            "treatment/radiomics 输入）"
        )
        input_definition = f"{channels} 个影像通道；未声明 ROI mask 辅助"
    return {
        "verification_complete": len(contracts) == len(sources),
        "image_channels": channels,
        "roi_assisted": roi_assisted,
        "declared_inputs": list(inputs),
        "forbidden_inputs_absent": list(forbidden),
        "method_label_cn": method_label,
        "input_definition_cn": input_definition,
    }


def _validate_probability_frame(frame: pd.DataFrame, label: str) -> None:
    probability = pd.to_numeric(frame["predicted_probability"], errors="coerce")
    threshold = pd.to_numeric(frame["threshold"], errors="coerce")
    labels = pd.to_numeric(frame["y_true"], errors="coerce")
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise AggregationInputError(f"{label} predicted_probability 非法")
    if not np.isfinite(threshold).all() or ((threshold < 0) | (threshold > 1)).any():
        raise AggregationInputError(f"{label} threshold 非法")
    if not labels.isin([0, 1]).all():
        raise AggregationInputError(f"{label} y_true 必须为 0/1")
    predicted = pd.to_numeric(frame["predicted_label"], errors="coerce")
    expected = (probability.to_numpy() >= threshold.to_numpy()).astype(int)
    if not np.array_equal(predicted.to_numpy(), expected):
        raise AggregationInputError(
            f"{label} predicted_label 与 validation threshold 不一致"
        )


def load_native_predictions(
    sources: Sequence[EvaluationSource],
    config: AggregationConfig,
    issues: list[dict[str, str]],
    consumed: list[Path],
) -> pd.DataFrame:
    required = {
        "patient_id",
        "fold",
        "split",
        "model_name",
        "run_name",
        "decision_point",
        "y_true",
        "predicted_probability",
        "predicted_label",
        "threshold",
        "checkpoint_sha256",
        "has_radiomics",
        "available_visits",
    }
    manifest: pd.DataFrame | None = None
    manifest_sha256: str | None = None
    if not config.fold_manifest.is_file():
        _missing_or_raise(
            config,
            issues,
            "找不到锁定 fold manifest，无法核验 OOF patient/label/fold",
            config.fold_manifest,
        )
    else:
        manifest = _read_csv(
            config.fold_manifest,
            {"patient_id", "fold", "split", "label_pcr"},
            "fold manifest",
        )
        manifest["patient_id"] = manifest["patient_id"].astype(str)
        if (
            not manifest["fold"].isin(range(5)).all()
            or not manifest["split"].isin(["train", "val", "test"]).all()
        ):
            raise AggregationInputError("fold manifest 含非法 fold/split")
        if not pd.to_numeric(manifest["label_pcr"], errors="coerce").isin([0, 1]).all():
            raise AggregationInputError("fold manifest label_pcr 必须为 0/1")
        if manifest.duplicated(["patient_id", "fold"]).any():
            raise AggregationInputError("fold manifest 同一 patient/fold 出现重复记录")
        manifest_sha256 = _sha256(config.fold_manifest)
        consumed.append(config.fold_manifest)
    frames: list[pd.DataFrame] = []
    for source in sources:
        path = source.prediction_dir / "test_predictions.csv"
        if not path.is_file():
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} 缺 test_predictions.csv",
                path,
            )
            continue
        frame = _read_csv(path, required, "test prediction")
        frame = frame.loc[frame["split"].eq("test")].copy()
        if frame.empty:
            raise AggregationInputError(f"test_predictions.csv 没有 test 行: {path}")
        if not frame["fold"].eq(source.fold).all():
            raise AggregationInputError(f"prediction fold 与 summary 不一致: {path}")
        if not frame["run_name"].eq(source.run_name).all():
            raise AggregationInputError(
                f"prediction run_name 与 summary 不一致: {path}"
            )
        if not frame["model_name"].eq(source.mode).all():
            raise AggregationInputError(
                f"prediction model_name 与 summary 不一致: {path}"
            )
        if not frame["checkpoint_sha256"].eq(source.checkpoint_sha256).all():
            raise AggregationInputError(f"prediction checkpoint SHA256 不一致: {path}")
        declared_manifest_hash = source.summary.get("fold_manifest_sha256")
        if declared_manifest_hash is None:
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} summary 缺 fold_manifest_sha256",
                source.summary_path,
            )
        elif manifest_sha256 is not None and declared_manifest_hash != manifest_sha256:
            raise AggregationInputError(
                f"{source.run_name}/fold_{source.fold} summary 的 fold manifest hash "
                f"与传入 manifest 不一致: {source.summary_path}"
            )
        if "fold_manifest_sha256" in frame.columns:
            if (
                manifest_sha256 is not None
                and not frame["fold_manifest_sha256"].eq(manifest_sha256).all()
            ):
                raise AggregationInputError(
                    f"prediction fold_manifest_sha256 不一致: {path}"
                )
        elif not config.allow_partial:
            raise AggregationInputError(
                f"strict 模式 prediction 缺 fold_manifest_sha256: {path}"
            )
        if not frame["decision_point"].isin(DECISION_ORDER).all():
            raise AggregationInputError(f"prediction 含未知 decision_point: {path}")
        if frame.duplicated(["patient_id", "decision_point"]).any():
            raise AggregationInputError(
                f"prediction 存在重复 patient/decision point: {path}"
            )
        _validate_probability_frame(frame, str(path))
        point_patient_sets: dict[str, set[str]] = {}
        point_label_maps: dict[str, dict[str, int]] = {}
        point_manifest_complete: dict[str, bool] = {}
        point_manifest_expected_n: dict[str, int | None] = {}
        declared_thresholds = source.summary.get("validation_thresholds")
        if not isinstance(declared_thresholds, dict):
            raise AggregationInputError(
                f"summary 缺 validation_thresholds: {source.summary_path}"
            )
        expected_test: pd.DataFrame | None = None
        if manifest is not None:
            expected_test = manifest.loc[
                manifest["fold"].eq(source.fold) & manifest["split"].eq("test"),
                ["patient_id", "label_pcr"],
            ].copy()
            if expected_test.empty or expected_test["patient_id"].duplicated().any():
                raise AggregationInputError(
                    f"manifest fold {source.fold} test patient 为空或重复"
                )
            expected_test["label_pcr"] = expected_test["label_pcr"].astype(int)
            expected_label_map = dict(
                zip(expected_test["patient_id"], expected_test["label_pcr"])
            )
        expected_n = int(source.summary.get("split_counts", {}).get("test", -1))
        for point, group in frame.groupby("decision_point"):
            invalid_n = expected_n >= 0 and (
                len(group) > expected_n
                or (not config.allow_partial and len(group) != expected_n)
            )
            if invalid_n:
                raise AggregationInputError(
                    f"{path} 的 {point} 行数 {len(group)} 与 summary test={expected_n} 不一致"
                )
            thresholds = pd.to_numeric(group["threshold"], errors="coerce").unique()
            if len(thresholds) != 1:
                raise AggregationInputError(
                    f"{source.run_name}/fold_{source.fold}/{point} 必须只有一个 "
                    "fold-validation threshold"
                )
            threshold_payload = declared_thresholds.get(point)
            if (
                not isinstance(threshold_payload, dict)
                or "threshold" not in threshold_payload
            ):
                raise AggregationInputError(
                    f"summary 缺 {point} validation threshold: {source.summary_path}"
                )
            if not math.isclose(
                float(thresholds[0]),
                float(threshold_payload["threshold"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise AggregationInputError(
                    f"{source.run_name}/fold_{source.fold}/{point} prediction threshold "
                    "与 summary validation threshold 不一致"
                )
            patient_ids = set(group["patient_id"].astype(str))
            label_map = dict(
                zip(group["patient_id"].astype(str), group["y_true"].astype(int))
            )
            point_patient_sets[point] = patient_ids
            point_label_maps[point] = label_map
            if expected_test is not None:
                expected_ids = set(expected_test["patient_id"])
                point_manifest_complete[point] = patient_ids == expected_ids
                point_manifest_expected_n[point] = len(expected_ids)
                if config.allow_partial:
                    if not patient_ids.issubset(expected_ids):
                        raise AggregationInputError(
                            f"{source.run_name}/fold_{source.fold}/{point} 含非 manifest-test patient"
                        )
                elif patient_ids != expected_ids:
                    missing_ids = sorted(expected_ids.difference(patient_ids))[:5]
                    extra_ids = sorted(patient_ids.difference(expected_ids))[:5]
                    raise AggregationInputError(
                        f"{source.run_name}/fold_{source.fold}/{point} patient set 与 "
                        f"manifest-test 不一致；missing={missing_ids}, extra={extra_ids}"
                    )
                for patient_id, label in label_map.items():
                    if expected_label_map.get(patient_id) != label:
                        raise AggregationInputError(
                            f"{source.run_name}/fold_{source.fold}/{point}/{patient_id} "
                            "y_true 与 manifest label_pcr 不一致"
                        )
            else:
                point_manifest_complete[point] = False
                point_manifest_expected_n[point] = None
        if point_patient_sets:
            reference_point = next(iter(point_patient_sets))
            for point in point_patient_sets:
                if (
                    point_patient_sets[point] != point_patient_sets[reference_point]
                    or point_label_maps[point] != point_label_maps[reference_point]
                ):
                    raise AggregationInputError(
                        f"{source.run_name}/fold_{source.fold} patient set/y_true 在 "
                        f"decision points 间不一致"
                    )
        missing_points = set(DECISION_ORDER).difference(
            frame["decision_point"].unique()
        )
        if missing_points:
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} 缺 decision points {sorted(missing_points)}",
                path,
            )
        frame.insert(0, "aggregation_model", source.mode)
        frame["manifest_test_complete"] = frame["decision_point"].map(
            point_manifest_complete
        )
        frame["manifest_test_expected_n"] = frame["decision_point"].map(
            point_manifest_expected_n
        )
        frame["source_file"] = str(path)
        frames.append(frame)
        consumed.append(path)
    if not frames:
        raise AggregationInputError("没有有效 test prediction")
    output = pd.concat(frames, ignore_index=True, sort=False)
    if output.duplicated(["aggregation_model", "patient_id", "decision_point"]).any():
        raise AggregationInputError(
            "OOF prediction 中同一模型/患者/decision point 重复"
        )
    if not config.allow_partial:
        for mode in MODEL_ORDER:
            candidate = output.loc[output["aggregation_model"].eq(mode)]
            expected_total = max(
                int(source.summary.get("primary_patient_count", 808))
                for source in sources
                if source.mode == mode
            )
            for point in DECISION_ORDER:
                n = candidate.loc[
                    candidate["decision_point"].eq(point), "patient_id"
                ].nunique()
                if n != expected_total:
                    raise AggregationInputError(
                        f"{mode}/{point} OOF 患者数 {n}，预期 {expected_total}"
                    )
        reference = {
            point: set(
                output.loc[
                    output["aggregation_model"].eq("m0")
                    & output["decision_point"].eq(point),
                    "patient_id",
                ]
            )
            for point in DECISION_ORDER
        }
        for mode in ("m1", "m2"):
            for point in DECISION_ORDER:
                candidate = set(
                    output.loc[
                        output["aggregation_model"].eq(mode)
                        & output["decision_point"].eq(point),
                        "patient_id",
                    ]
                )
                if candidate != reference[point]:
                    raise AggregationInputError(
                        f"{mode}/{point} OOF 患者集合与 M0 不一致"
                    )
    return output


def _classification_values(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["y_true"].to_numpy(dtype=int)
    probability = frame["predicted_probability"].to_numpy(dtype=float)
    prediction = frame["predicted_label"].to_numpy(dtype=int)
    positive = y == 1
    negative = y == 0
    return {
        "n": int(len(y)),
        "n_positive": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "auroc": (
            float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else math.nan
        ),
        "auprc": (
            float(average_precision_score(y, probability))
            if positive.any()
            else math.nan
        ),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity": (
            float(recall_score(y, prediction, pos_label=1, zero_division=0))
            if positive.any()
            else math.nan
        ),
        "specificity": (
            float(recall_score(y, prediction, pos_label=0, zero_division=0))
            if negative.any()
            else math.nan
        ),
    }


def classification_tables(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    for (mode, run_name, fold, point), group in predictions.groupby(
        ["aggregation_model", "run_name", "fold", "decision_point"], sort=False
    ):
        fold_rows.append(
            {
                "model_name": mode,
                "model_label_cn": MODEL_DISPLAY.get(mode, mode.upper()),
                "run_name": run_name,
                "fold": int(fold),
                "decision_point": point,
                "cohort_scope_cn": (
                    "完整 fold-test"
                    if "manifest_test_complete" in group.columns
                    and group["manifest_test_complete"].fillna(False).astype(bool).all()
                    else "显式部分 fold-test"
                ),
                **_classification_values(group),
            }
        )
    folds = pd.DataFrame(fold_rows)

    oof_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for (mode, run_name, point), group in predictions.groupby(
        ["aggregation_model", "run_name", "decision_point"], sort=False
    ):
        values = _classification_values(group)
        oof_rows.append(
            {
                "model_name": mode,
                "model_label_cn": MODEL_DISPLAY.get(mode, mode.upper()),
                "run_name": run_name,
                "decision_point": point,
                "aggregation_level": "pooled_oof_test",
                "cohort_scope_cn": (
                    "完整五折 OOF test"
                    if group["fold"].nunique() == 5
                    and "manifest_test_complete" in group.columns
                    and group["manifest_test_complete"].fillna(False).astype(bool).all()
                    else "显式部分 OOF test"
                ),
                "n_folds": int(group["fold"].nunique()),
                **values,
            }
        )
        rng = np.random.default_rng(_stable_seed(seed, mode, point, "classification"))
        sampled_metrics = {name: [] for name in CLASSIFICATION_METRICS}
        n = len(group)
        for _ in range(replicates):
            indices = rng.integers(0, n, size=n)
            sample = group.iloc[indices]
            result = _classification_values(sample)
            for name in CLASSIFICATION_METRICS:
                value = float(result[name])
                if math.isfinite(value):
                    sampled_metrics[name].append(value)
        for name in CLASSIFICATION_METRICS:
            samples = np.asarray(sampled_metrics[name], dtype=float)
            bootstrap_rows.append(
                {
                    "model_name": mode,
                    "run_name": run_name,
                    "decision_point": point,
                    "metric": name,
                    "metric_cn": METRIC_CN[name],
                    "estimate": float(values[name]),
                    "ci_lower_95": (
                        float(np.quantile(samples, 0.025)) if samples.size else math.nan
                    ),
                    "ci_upper_95": (
                        float(np.quantile(samples, 0.975)) if samples.size else math.nan
                    ),
                    "n_bootstrap_requested": replicates,
                    "n_bootstrap_valid": int(samples.size),
                    "resampling_unit": "patient",
                    "n_patients": n,
                    "seed": _stable_seed(seed, mode, point, "classification"),
                }
            )
    oof = pd.DataFrame(oof_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)

    summary_rows: list[dict[str, Any]] = []
    for (mode, run_name, point), group in folds.groupby(
        ["model_name", "run_name", "decision_point"], sort=False
    ):
        oof_n = int(
            oof.loc[
                oof["model_name"].eq(mode) & oof["decision_point"].eq(point), "n"
            ].iloc[0]
        )
        for name in CLASSIFICATION_METRICS:
            values = group[name].dropna().to_numpy(dtype=float)
            mean = float(values.mean()) if values.size else math.nan
            std = float(values.std(ddof=1)) if values.size > 1 else math.nan
            summary_rows.append(
                {
                    "model_name": mode,
                    "run_name": run_name,
                    "decision_point": point,
                    "metric": name,
                    "metric_cn": METRIC_CN[name],
                    "fold_mean": mean,
                    "fold_std": std,
                    "mean_plus_minus_std": (
                        f"{mean:.4f} ± {std:.4f}"
                        if math.isfinite(mean) and math.isfinite(std)
                        else f"{mean:.4f} ± NA"
                    ),
                    "n_folds": int(group["fold"].nunique()),
                    "n_oof_patients": oof_n,
                    "errorbar_definition_cn": "fold 间样本标准差（ddof=1）",
                }
            )
    return folds, oof, pd.DataFrame(summary_rows), bootstrap


def paired_classification_bootstrap_differences(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    """以同一患者重采样估计预注册模型差值的 paired bootstrap CI。"""

    rows: list[dict[str, Any]] = []
    comparisons = (("m0", "m1"), ("m1", "m2"))
    for point in DECISION_ORDER:
        point_frame = predictions.loc[predictions["decision_point"].eq(point)].copy()
        by_model = {
            mode: group.set_index("patient_id").sort_index()
            for mode, group in point_frame.groupby("aggregation_model", sort=False)
        }
        for reference_mode, candidate_mode in comparisons:
            if reference_mode not in by_model or candidate_mode not in by_model:
                continue
            reference = by_model[reference_mode]
            candidate = by_model[candidate_mode]
            patient_ids = reference.index.intersection(candidate.index).sort_values()
            if patient_ids.empty:
                continue
            reference = reference.loc[patient_ids]
            candidate = candidate.loc[patient_ids]
            if not np.array_equal(
                reference["y_true"].to_numpy(dtype=int),
                candidate["y_true"].to_numpy(dtype=int),
            ):
                raise AggregationInputError(
                    f"{point} {candidate_mode}-{reference_mode} paired y_true 不一致"
                )
            y = reference["y_true"].to_numpy(dtype=int)
            reference_probability = reference["predicted_probability"].to_numpy(
                dtype=float
            )
            candidate_probability = candidate["predicted_probability"].to_numpy(
                dtype=float
            )
            rng_seed = _stable_seed(
                seed,
                point,
                reference_mode,
                candidate_mode,
                "paired_classification_difference",
            )
            rng = np.random.default_rng(rng_seed)
            for metric in ("auroc", "auprc"):
                if metric == "auroc":
                    scorer = roc_auc_score
                else:
                    scorer = average_precision_score
                reference_estimate = float(scorer(y, reference_probability))
                candidate_estimate = float(scorer(y, candidate_probability))
                differences: list[float] = []
                for _ in range(replicates):
                    indices = rng.integers(0, len(y), size=len(y))
                    sampled_y = y[indices]
                    if metric == "auroc" and np.unique(sampled_y).size < 2:
                        continue
                    differences.append(
                        float(scorer(sampled_y, candidate_probability[indices]))
                        - float(scorer(sampled_y, reference_probability[indices]))
                    )
                samples = np.asarray(differences, dtype=float)
                rows.append(
                    {
                        "decision_point": point,
                        "metric": metric,
                        "metric_cn": METRIC_CN[metric],
                        "reference_model": reference_mode,
                        "candidate_model": candidate_mode,
                        "contrast": f"{candidate_mode}-{reference_mode}",
                        "reference_estimate": reference_estimate,
                        "candidate_estimate": candidate_estimate,
                        "difference_candidate_minus_reference": (
                            candidate_estimate - reference_estimate
                        ),
                        "ci_lower_95": (
                            float(np.quantile(samples, 0.025))
                            if samples.size
                            else math.nan
                        ),
                        "ci_upper_95": (
                            float(np.quantile(samples, 0.975))
                            if samples.size
                            else math.nan
                        ),
                        "n_bootstrap_requested": replicates,
                        "n_bootstrap_valid": int(samples.size),
                        "resampling_unit": "paired patient",
                        "n_paired_patients": len(y),
                        "n_positive": int(y.sum()),
                        "seed": rng_seed,
                    }
                )
    return pd.DataFrame(rows)


def load_perturbations(
    sources: Sequence[EvaluationSource],
    native: pd.DataFrame,
    config: AggregationConfig,
    issues: list[dict[str, str]],
    consumed: list[Path],
) -> pd.DataFrame:
    required = {
        "patient_id",
        "fold",
        "split",
        "model_name",
        "run_name",
        "decision_point",
        "perturbation",
        "y_true",
        "predicted_probability",
        "predicted_label",
        "threshold",
        "native_predicted_probability",
        "native_predicted_label",
        "probability_change",
        "absolute_probability_change",
        "same_native_readout",
        "checkpoint_sha256",
    }
    frames: list[pd.DataFrame] = []
    for source in sources:
        path = source.prediction_dir / "test_perturbation_predictions.csv"
        if not path.is_file():
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} 缺 shortcut perturbation prediction",
                path,
            )
            continue
        frame = _read_csv(path, required, "shortcut perturbation prediction")
        if (
            not frame["fold"].eq(source.fold).all()
            or not frame["split"].eq("test").all()
        ):
            raise AggregationInputError(f"shortcut prediction fold/split 非法: {path}")
        if (
            not frame["run_name"].eq(source.run_name).all()
            or not frame["model_name"].eq(source.mode).all()
        ):
            raise AggregationInputError(f"shortcut prediction run/model 不一致: {path}")
        if not frame["checkpoint_sha256"].eq(source.checkpoint_sha256).all():
            raise AggregationInputError(
                f"shortcut prediction checkpoint 不一致: {path}"
            )
        if not frame["perturbation"].isin(PERTURBATION_ORDER).all():
            raise AggregationInputError(
                f"shortcut prediction 含未知 perturbation: {path}"
            )
        if frame.duplicated(["patient_id", "decision_point", "perturbation"]).any():
            raise AggregationInputError(f"shortcut prediction 重复: {path}")
        _validate_probability_frame(frame, str(path))
        same_readout = (
            frame["same_native_readout"].astype(str).str.lower().isin(["true", "1"])
        )
        if not same_readout.all():
            raise AggregationInputError(
                f"shortcut perturbation 未复用 native readout: {path}"
            )
        delta = frame["predicted_probability"].to_numpy(dtype=float) - frame[
            "native_predicted_probability"
        ].to_numpy(dtype=float)
        if not np.allclose(delta, frame["probability_change"], atol=1e-9, rtol=1e-7):
            raise AggregationInputError(
                f"shortcut probability_change 字段不一致: {path}"
            )
        absolute_delta = pd.to_numeric(
            frame["absolute_probability_change"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(absolute_delta).all()
            or (absolute_delta < 0).any()
            or not np.allclose(absolute_delta, np.abs(delta), atol=1e-9, rtol=1e-7)
        ):
            raise AggregationInputError(
                f"shortcut absolute_probability_change 字段不一致: {path}"
            )
        expected_cells: set[tuple[str, str, str]] = set()
        for perturbation, point in (
            ("repeated_t0", "T0-T1"),
            ("repeated_t0", "T0-T2"),
            ("temporal_shuffle_t1_t2", "T0-T2"),
        ):
            patient_ids = native.loc[
                native["aggregation_model"].eq(source.mode)
                & native["fold"].eq(source.fold)
                & native["decision_point"].eq(point),
                "patient_id",
            ].astype(str)
            expected_cells.update(
                (patient_id, point, perturbation) for patient_id in patient_ids
            )
        actual_cells = set(
            zip(
                frame["patient_id"].astype(str),
                frame["decision_point"].astype(str),
                frame["perturbation"].astype(str),
            )
        )
        extra_cells = actual_cells.difference(expected_cells)
        missing_cells = expected_cells.difference(actual_cells)
        if extra_cells:
            raise AggregationInputError(
                f"shortcut 含非预注册 patient/point/perturbation cell: {sorted(extra_cells)[:5]}"
            )
        if missing_cells:
            message = (
                f"{source.run_name}/fold_{source.fold} shortcut 缺 "
                f"{len(missing_cells)} 个预注册 patient-level cell"
            )
            if config.allow_partial:
                _record_issue(issues, "缺失输入_missing_input", message, path)
            else:
                raise AggregationInputError(message)
        declared_rows = source.summary.get("test_perturbation_rows")
        if not isinstance(declared_rows, int):
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} summary 缺 test_perturbation_rows",
                source.summary_path,
            )
        elif len(frame) > declared_rows or (
            not config.allow_partial and len(frame) != declared_rows
        ):
            raise AggregationInputError(
                f"shortcut rows={len(frame)} 与 summary "
                f"test_perturbation_rows={declared_rows} 不一致"
            )
        elif len(frame) < declared_rows:
            _record_issue(
                issues,
                "缺失输入_missing_input",
                f"shortcut rows={len(frame)} 少于 summary 声明 {declared_rows}",
                path,
            )
        frame.insert(0, "aggregation_model", source.mode)
        frame["source_file"] = str(path)
        frames.append(frame)
        consumed.append(path)
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False)
    if output.duplicated(
        ["aggregation_model", "patient_id", "decision_point", "perturbation"]
    ).any():
        raise AggregationInputError("OOF shortcut prediction 重复")
    native_lookup = native.set_index(
        ["aggregation_model", "fold", "patient_id", "decision_point"]
    )
    for row in output.itertuples(index=False):
        key = (row.aggregation_model, row.fold, row.patient_id, row.decision_point)
        if key not in native_lookup.index:
            raise AggregationInputError(f"shortcut 行找不到 native prediction: {key}")
        reference = native_lookup.loc[key]
        if not math.isclose(
            float(row.native_predicted_probability),
            float(reference["predicted_probability"]),
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise AggregationInputError(
                f"shortcut native 概率与正式 test prediction 不一致: {key}"
            )
        if (
            int(row.y_true) != int(reference["y_true"])
            or int(row.native_predicted_label) != int(reference["predicted_label"])
            or not math.isclose(
                float(row.threshold),
                float(reference["threshold"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise AggregationInputError(
                f"shortcut y_true/native label/threshold 与正式 test prediction 不一致: {key}"
            )
    return output


def _shortcut_group_metrics(group: pd.DataFrame) -> dict[str, Any]:
    native_frame = group.copy()
    native_frame["predicted_probability"] = group["native_predicted_probability"]
    native_frame["predicted_label"] = group["native_predicted_label"]
    native_values = _classification_values(native_frame)
    perturbed_values = _classification_values(group)
    return {
        "n": int(len(group)),
        **{
            f"native_{key}": value
            for key, value in native_values.items()
            if key in CLASSIFICATION_METRICS
        },
        **{
            f"perturbed_{key}": value
            for key, value in perturbed_values.items()
            if key in CLASSIFICATION_METRICS
        },
        **{
            f"{key}_change_perturbed_minus_native": float(perturbed_values[key])
            - float(native_values[key])
            for key in CLASSIFICATION_METRICS
        },
        "mean_probability_change": float(group["probability_change"].mean()),
        "mean_absolute_probability_change": float(
            group["absolute_probability_change"].mean()
        ),
        "median_absolute_probability_change": float(
            group["absolute_probability_change"].median()
        ),
    }


def shortcut_tables(
    perturbations: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if perturbations.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    grouping = ["aggregation_model", "run_name", "decision_point", "perturbation"]
    for level, keys in (
        ("fold_test", grouping + ["fold"]),
        ("pooled_oof_test", grouping),
    ):
        for key, group in perturbations.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            values = dict(zip(keys, key))
            rows.append(
                {
                    "aggregation_level": level,
                    "model_name": values["aggregation_model"],
                    "run_name": values["run_name"],
                    "fold": values.get("fold", pd.NA),
                    "decision_point": values["decision_point"],
                    "perturbation": values["perturbation"],
                    "perturbation_cn": PERTURBATION_DISPLAY[values["perturbation"]],
                    **_shortcut_group_metrics(group),
                }
            )
            abs_change = group["absolute_probability_change"].to_numpy(dtype=float)
            signed_change = group["probability_change"].to_numpy(dtype=float)
            change_rows.append(
                {
                    "aggregation_level": level,
                    "model_name": values["aggregation_model"],
                    "run_name": values["run_name"],
                    "fold": values.get("fold", pd.NA),
                    "decision_point": values["decision_point"],
                    "perturbation": values["perturbation"],
                    "n_patients": len(group),
                    "mean_signed_change": float(signed_change.mean()),
                    "std_signed_change": (
                        float(signed_change.std(ddof=1)) if len(group) > 1 else math.nan
                    ),
                    "mean_absolute_change": float(abs_change.mean()),
                    "median_absolute_change": float(np.median(abs_change)),
                    "q25_absolute_change": float(np.quantile(abs_change, 0.25)),
                    "q75_absolute_change": float(np.quantile(abs_change, 0.75)),
                    "q95_absolute_change": float(np.quantile(abs_change, 0.95)),
                }
            )
            if level != "pooled_oof_test":
                continue
            rng_seed = _stable_seed(
                seed,
                values["aggregation_model"],
                values["decision_point"],
                values["perturbation"],
                "shortcut",
            )
            rng = np.random.default_rng(rng_seed)
            sampled = {"auroc": [], "auprc": []}
            n = len(group)
            for _ in range(replicates):
                sample = group.iloc[rng.integers(0, n, size=n)]
                metrics = _shortcut_group_metrics(sample)
                for metric in sampled:
                    delta = float(metrics[f"{metric}_change_perturbed_minus_native"])
                    if math.isfinite(delta):
                        sampled[metric].append(delta)
            point_metrics = _shortcut_group_metrics(group)
            for metric, samples_list in sampled.items():
                samples = np.asarray(samples_list, dtype=float)
                bootstrap_rows.append(
                    {
                        "model_name": values["aggregation_model"],
                        "run_name": values["run_name"],
                        "decision_point": values["decision_point"],
                        "perturbation": values["perturbation"],
                        "metric_change": f"{metric}_perturbed_minus_native",
                        "estimate": point_metrics[
                            f"{metric}_change_perturbed_minus_native"
                        ],
                        "ci_lower_95": (
                            float(np.quantile(samples, 0.025))
                            if samples.size
                            else math.nan
                        ),
                        "ci_upper_95": (
                            float(np.quantile(samples, 0.975))
                            if samples.size
                            else math.nan
                        ),
                        "n_bootstrap_requested": replicates,
                        "n_bootstrap_valid": int(samples.size),
                        "resampling_unit": "patient",
                        "n_patients": n,
                        "seed": rng_seed,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(change_rows), pd.DataFrame(bootstrap_rows)


def load_transition_predictions(
    sources: Sequence[EvaluationSource],
    native: pd.DataFrame,
    config: AggregationConfig,
    issues: list[dict[str, str]],
    consumed: list[Path],
) -> pd.DataFrame:
    required = {
        "patient_id",
        "fold",
        "split",
        "model_name",
        "run_name",
        "transition",
        "y_true",
        "learned_error",
        "copy_error",
        "gain",
        "learned_raw_mse",
        "copy_raw_mse",
        "predicted_delta_norm",
        "target_delta_norm",
        "delta_norm_ratio",
        "delta_cosine_similarity",
        "checkpoint_sha256",
    }
    frames: list[pd.DataFrame] = []
    numeric = [
        "learned_error",
        "copy_error",
        "gain",
        "learned_raw_mse",
        "copy_raw_mse",
        "predicted_delta_norm",
        "target_delta_norm",
        "delta_norm_ratio",
        "delta_cosine_similarity",
    ]
    for source in sources:
        path = source.metric_dir / "test_transition_metrics.csv"
        if not path.is_file():
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} 缺 transition metrics",
                path,
            )
            continue
        frame = _read_csv(path, required, "transition metric")
        if (
            not frame["fold"].eq(source.fold).all()
            or not frame["split"].eq("test").all()
        ):
            raise AggregationInputError(f"transition fold/split 非法: {path}")
        if (
            not frame["run_name"].eq(source.run_name).all()
            or not frame["model_name"].eq(source.mode).all()
        ):
            raise AggregationInputError(f"transition run/model 不一致: {path}")
        if not frame["checkpoint_sha256"].eq(source.checkpoint_sha256).all():
            raise AggregationInputError(f"transition checkpoint 不一致: {path}")
        if not frame["transition"].isin(TRANSITION_ORDER).all():
            raise AggregationInputError(f"transition 名称非法: {path}")
        if frame.duplicated(["patient_id", "transition"]).any():
            raise AggregationInputError(f"transition patient/step 重复: {path}")
        values = frame[numeric].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise AggregationInputError(f"transition metric 含 NaN/Inf: {path}")
        if (
            (
                values[
                    ["learned_error", "copy_error", "learned_raw_mse", "copy_raw_mse"]
                ]
                < 0
            )
            .any()
            .any()
        ):
            raise AggregationInputError(f"transition error/MSE 不得为负: {path}")
        native_source = native.loc[
            native["aggregation_model"].eq(source.mode)
            & native["fold"].eq(source.fold)
            & native["decision_point"].eq("T0")
        ].copy()
        native_ids = set(native_source["patient_id"].astype(str))
        native_labels = dict(
            zip(
                native_source["patient_id"].astype(str),
                native_source["y_true"].astype(int),
            )
        )
        expected_cells = {
            (patient_id, transition)
            for patient_id in native_ids
            for transition in TRANSITION_ORDER
        }
        actual_cells = set(
            zip(frame["patient_id"].astype(str), frame["transition"].astype(str))
        )
        extra_cells = actual_cells.difference(expected_cells)
        missing_cells = expected_cells.difference(actual_cells)
        if extra_cells:
            raise AggregationInputError(
                "transition metric 含非 native-test patient/step: "
                f"{sorted(extra_cells)[:5]}"
            )
        if missing_cells:
            message = (
                f"{source.run_name}/fold_{source.fold} transition metric 缺 "
                f"{len(missing_cells)} 个 patient×step cell"
            )
            if config.allow_partial:
                _record_issue(issues, "缺失输入_missing_input", message, path)
            else:
                raise AggregationInputError(message)
        for row in frame[["patient_id", "y_true"]].itertuples(index=False):
            if native_labels.get(str(row.patient_id)) != int(row.y_true):
                raise AggregationInputError(
                    f"transition {row.patient_id} y_true 与 native prediction 不一致: {path}"
                )
        declared_rows = source.summary.get("test_transition_rows")
        if not isinstance(declared_rows, int):
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} summary 缺 test_transition_rows",
                source.summary_path,
            )
        elif len(frame) > declared_rows or (
            not config.allow_partial and len(frame) != declared_rows
        ):
            raise AggregationInputError(
                f"transition rows={len(frame)} 与 summary "
                f"test_transition_rows={declared_rows} 不一致"
            )
        elif len(frame) < declared_rows:
            _record_issue(
                issues,
                "缺失输入_missing_input",
                f"transition rows={len(frame)} 少于 summary 声明 {declared_rows}",
                path,
            )
        frame.insert(0, "aggregation_model", source.mode)
        frame["source_file"] = str(path)
        frames.append(frame)
        consumed.append(path)
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False)
    if output.duplicated(["aggregation_model", "patient_id", "transition"]).any():
        raise AggregationInputError("OOF transition metric 重复")
    return output


def _transition_aggregate(group: pd.DataFrame) -> dict[str, Any]:
    learned = group["learned_error"].to_numpy(dtype=float)
    copy = group["copy_error"].to_numpy(dtype=float)
    learned_raw = group["learned_raw_mse"].to_numpy(dtype=float)
    copy_raw = group["copy_raw_mse"].to_numpy(dtype=float)
    copy_sum = float(copy.sum())
    copy_raw_sum = float(copy_raw.sum())
    return {
        "n_patient_transitions": int(len(group)),
        "n_patients": int(group["patient_id"].nunique()),
        "normalized_learned_error_sum": float(learned.sum()),
        "normalized_copy_error_sum": copy_sum,
        "normalized_aggregate_gain": (copy_sum - float(learned.sum()))
        / max(copy_sum, 1e-8),
        "raw_learned_mse_sum": float(learned_raw.sum()),
        "raw_copy_mse_sum": copy_raw_sum,
        "raw_aggregate_gain": (copy_raw_sum - float(learned_raw.sum()))
        / max(copy_raw_sum, 1e-8),
        "mean_patient_step_normalized_gain": float(group["gain"].mean()),
        "median_patient_step_normalized_gain": float(group["gain"].median()),
        "positive_patient_step_gain_fraction": float((group["gain"] > 0).mean()),
        "mean_normalized_learned_error": float(learned.mean()),
        "mean_normalized_copy_error": float(copy.mean()),
        "mean_raw_learned_mse": float(learned_raw.mean()),
        "mean_raw_copy_mse": float(copy_raw.mean()),
        "predicted_delta_norm_mean": float(group["predicted_delta_norm"].mean()),
        "predicted_delta_norm_std": float(group["predicted_delta_norm"].std(ddof=1)),
        "target_delta_norm_mean": float(group["target_delta_norm"].mean()),
        "target_delta_norm_std": float(group["target_delta_norm"].std(ddof=1)),
        "delta_norm_ratio_mean": float(group["delta_norm_ratio"].mean()),
        "delta_norm_ratio_median": float(group["delta_norm_ratio"].median()),
        "delta_cosine_mean": float(group["delta_cosine_similarity"].mean()),
        "delta_cosine_std": float(group["delta_cosine_similarity"].std(ddof=1)),
        "delta_cosine_median": float(group["delta_cosine_similarity"].median()),
        "gain_definition": "(sum(copy_error)-sum(learned_error))/max(sum(copy_error),1e-8)",
    }


def transition_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    base_keys = ["aggregation_model", "run_name"]
    for level, extra in (("fold_test", ["fold"]), ("pooled_oof_test", [])):
        for key, model_group in frame.groupby(base_keys + extra, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            values = dict(zip(base_keys + extra, key))
            for transition in (*TRANSITION_ORDER, "全部"):
                group = (
                    model_group
                    if transition == "全部"
                    else model_group.loc[model_group["transition"].eq(transition)]
                )
                if group.empty:
                    continue
                rows.append(
                    {
                        "aggregation_level": level,
                        "model_name": values["aggregation_model"],
                        "run_name": values["run_name"],
                        "fold": values.get("fold", pd.NA),
                        "transition": transition,
                        **_transition_aggregate(group),
                    }
                )
    return pd.DataFrame(rows)


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = (
            spearmanr(x, y).statistic
            if kind == "spearman"
            else pearsonr(x, y).statistic
        )
    return float(value) if math.isfinite(float(value)) else math.nan


def _regression_values(
    target: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray | None = None,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape or target.ndim != 1 or target.size == 0:
        raise AggregationInputError("radiomics regression target/prediction shape 非法")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise AggregationInputError("radiomics regression target/prediction 含 NaN/Inf")
    target_var = float(np.var(target, ddof=1)) if target.size > 1 else math.nan
    prediction_var = float(np.var(prediction, ddof=1)) if target.size > 1 else math.nan
    rmse = math.sqrt(float(mean_squared_error(target, prediction)))
    result: dict[str, Any] = {
        "n": int(target.size),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": rmse,
        "spearman": _correlation(target, prediction, "spearman"),
        "pearson": _correlation(target, prediction, "pearson"),
        "r2": float(r2_score(target, prediction)) if target.size >= 2 else math.nan,
        "direction_accuracy": float((np.sign(target) == np.sign(prediction)).mean()),
        "target_mean": float(target.mean()),
        "predicted_mean": float(prediction.mean()),
        "target_variance_ddof1": target_var,
        "predicted_variance_ddof1": prediction_var,
        "predicted_to_target_variance_ratio": (
            prediction_var / max(target_var, 1e-12)
            if math.isfinite(target_var) and math.isfinite(prediction_var)
            else math.nan
        ),
        "near_constant_prediction": (
            bool(
                math.isfinite(prediction_var)
                and prediction_var <= max(1e-10, 0.01 * target_var)
            )
            if math.isfinite(target_var)
            else False
        ),
    }
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        if baseline.shape != target.shape or not np.isfinite(baseline).all():
            raise AggregationInputError("train-mean baseline shape/数值非法")
        baseline_rmse = math.sqrt(float(mean_squared_error(target, baseline)))
        result.update(
            {
                "train_mean_baseline_mae": float(mean_absolute_error(target, baseline)),
                "train_mean_baseline_rmse": baseline_rmse,
                "rmse_gain_over_train_mean_baseline": (baseline_rmse - rmse)
                / max(baseline_rmse, 1e-12),
                "train_mean_baseline_direction_accuracy": float(
                    (np.sign(target) == np.sign(baseline)).mean()
                ),
            }
        )
    return result


def _radiomics_train_mean_baselines(
    config: AggregationConfig,
    m2_sources: Sequence[EvaluationSource],
    issues: list[dict[str, str]],
    consumed: list[Path],
) -> tuple[
    dict[tuple[int, str, str], tuple[float, float, int]],
    dict[tuple[int, str], dict[str, Any]],
    dict[str, np.ndarray],
]:
    """重建 fold×transition×feature 的 train-only 均值常数基线。

    M2 head 有三个相邻 transition 输出，常数基线必须分别估计；跨 transition
    混均值会掩盖不同治疗阶段的系统性变化，并且不是同一输出通道的公平基线。
    返回值依次为标准化均值、变换后 change 均值和有效 fold-train 样本数。
    """

    paths = [config.fold_manifest, config.radiomics_raw_targets]
    if any(not path.is_file() for path in paths):
        missing = [path for path in paths if not path.is_file()]
        message = f"无法构造 radiomics train-mean baseline，缺文件: {missing}"
        if config.allow_partial:
            _record_issue(issues, "缺失基线_missing_baseline", message)
            return {}, {}, {}
        raise AggregationInputError(message)
    manifest = _read_csv(
        config.fold_manifest,
        {"patient_id", "fold", "split"},
        "fold manifest",
    )
    raw = _read_csv(
        config.radiomics_raw_targets,
        {"patient_id", "transition"},
        "raw radiomics transition targets",
    )
    raw_mapping = read_raw_radiomics(config.radiomics_raw_targets)
    computed_raw_hash = raw_targets_hash(raw_mapping)
    sources_by_fold = {source.fold: source for source in m2_sources}
    for source in m2_sources:
        declared_raw_hash = source.summary.get("radiomics_raw_targets_sha256")
        if declared_raw_hash is None:
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} summary 缺 radiomics raw-target hash",
                source.summary_path,
            )
        elif declared_raw_hash != computed_raw_hash:
            raise AggregationInputError(
                f"{source.run_name}/fold_{source.fold} radiomics raw-target hash "
                "与当前审计表不一致"
            )
    consumed.extend(paths)
    baselines: dict[tuple[int, str, str], tuple[float, float, int]] = {}
    transform_specs: dict[tuple[int, str], dict[str, Any]] = {}
    for fold in sorted(sources_by_fold):
        transform_path = (
            config.radiomics_transform_dir / f"radiomics_transform_fold_{fold}.json"
        )
        if not transform_path.is_file():
            if config.allow_partial:
                _record_issue(
                    issues,
                    "缺失基线_missing_baseline",
                    f"fold {fold} 缺 radiomics transform，不能计算 train-mean baseline",
                    transform_path,
                )
                continue
            raise AggregationInputError(f"缺 radiomics transform: {transform_path}")
        transform = _read_json(transform_path)
        if int(transform.get("fold", -1)) != fold:
            raise AggregationInputError(
                f"radiomics transform fold 不一致: {transform_path}"
            )
        if transform.get("spec_version") != TRANSFORM_SPEC_VERSION:
            raise AggregationInputError(
                f"radiomics transform spec_version 不一致: {transform_path}"
            )
        if transform.get("raw_targets_sha256") != computed_raw_hash:
            raise AggregationInputError(
                f"radiomics transform raw_targets_sha256 不一致: {transform_path}"
            )
        features = transform.get("features")
        if not isinstance(features, list) or [
            item.get("name") for item in features
        ] != list(FEATURE_ORDER):
            raise AggregationInputError(
                f"radiomics transform feature schema 非法: {transform_path}"
            )
        train_ids = set(
            manifest.loc[
                manifest["fold"].eq(fold) & manifest["split"].eq("train"),
                "patient_id",
            ].astype(str)
        )
        if not train_ids:
            raise AggregationInputError(f"fold {fold} manifest 没有 train patients")
        if int(transform.get("train_patient_count", -1)) != len(train_ids):
            raise AggregationInputError(
                f"fold {fold} transform train_patient_count 与 manifest 不一致"
            )
        if transform.get("train_patient_hash") != patient_hash(train_ids):
            raise AggregationInputError(
                f"fold {fold} transform train_patient_hash 与 manifest 不一致"
            )
        paired_train_ids = train_ids.intersection(raw_mapping)
        if int(transform.get("paired_train_patient_count", -1)) != len(
            paired_train_ids
        ):
            raise AggregationInputError(
                f"fold {fold} transform paired_train_patient_count 不一致"
            )
        train_raw = raw.loc[raw["patient_id"].astype(str).isin(train_ids)].copy()
        transform_sha256 = _sha256(transform_path)
        for feature_index, item in enumerate(features):
            numeric_fields = (
                "epsilon",
                "winsor_low",
                "winsor_high",
                "center",
                "scale",
                "n_train_values",
            )
            if any(name not in item for name in numeric_fields):
                raise AggregationInputError(
                    f"fold {fold}/{item.get('name')} transform 缺数值字段"
                )
            numeric_values = np.asarray(
                [float(item[name]) for name in numeric_fields[:-1]], dtype=float
            )
            if not np.isfinite(numeric_values).all() or float(item["scale"]) <= 0:
                raise AggregationInputError(
                    f"fold {fold}/{item['name']} transform 数值非法"
                )
            if float(item["winsor_low"]) > float(item["winsor_high"]):
                raise AggregationInputError(
                    f"fold {fold}/{item['name']} winsor 区间反向"
                )
            valid_count = sum(
                int(
                    np.asarray(raw_mapping[patient_id])[:, feature_index, 2]
                    .astype(bool)
                    .sum()
                )
                for patient_id in paired_train_ids
            )
            if int(item["n_train_values"]) != valid_count:
                raise AggregationInputError(
                    f"fold {fold}/{item['name']} n_train_values={item['n_train_values']} "
                    f"与 raw target 有效数 {valid_count} 不一致"
                )
            transform_specs[(fold, str(item["name"]))] = {
                **item,
                "transform_sha256": transform_sha256,
                "spec_version": str(transform["spec_version"]),
                "raw_targets_sha256": computed_raw_hash,
            }
        for transition in TRANSITION_ORDER:
            transition_raw = train_raw.loc[
                train_raw["transition"].eq(transition)
            ].copy()
            if transition_raw.empty:
                raise AggregationInputError(
                    f"fold {fold}/{transition} 无 train radiomics target"
                )
            for item in features:
                feature = str(item["name"])
                start_col, end_col, valid_col = (
                    f"{feature}_start",
                    f"{feature}_end",
                    f"{feature}_valid",
                )
                missing = {start_col, end_col, valid_col}.difference(
                    transition_raw.columns
                )
                if missing:
                    raise AggregationInputError(
                        f"raw radiomics 缺 {feature} baseline 字段: {sorted(missing)}"
                    )
                valid = (
                    transition_raw[valid_col]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1"])
                )
                start = pd.to_numeric(
                    transition_raw.loc[valid, start_col], errors="coerce"
                ).to_numpy()
                end = pd.to_numeric(
                    transition_raw.loc[valid, end_col], errors="coerce"
                ).to_numpy()
                finite = np.isfinite(start) & np.isfinite(end)
                start, end = start[finite], end[finite]
                if not len(start):
                    raise AggregationInputError(
                        f"fold {fold}/{transition}/{feature} 无 train radiomics target"
                    )
                epsilon = float(item["epsilon"])
                transform_name = str(item["value_transform"])
                if transform_name == "log_epsilon":
                    change = np.log(end + epsilon) - np.log(start + epsilon)
                elif transform_name == "identity":
                    change = end - start
                elif transform_name == "log1p":
                    change = np.log1p(end) - np.log1p(start)
                else:
                    raise AggregationInputError(
                        "未知 radiomics value_transform="
                        f"{transform_name}: {transform_path}"
                    )
                clipped = np.clip(
                    change, float(item["winsor_low"]), float(item["winsor_high"])
                )
                standardized = (clipped - float(item["center"])) / float(item["scale"])
                mean_standardized = float(standardized.mean())
                mean_change = mean_standardized * float(item["scale"]) + float(
                    item["center"]
                )
                baselines[(fold, transition, feature)] = (
                    mean_standardized,
                    mean_change,
                    int(len(standardized)),
                )
        consumed.append(transform_path)
    return baselines, transform_specs, raw_mapping


def load_radiomics_predictions(
    sources: Sequence[EvaluationSource],
    native: pd.DataFrame,
    config: AggregationConfig,
    issues: list[dict[str, str]],
    consumed: list[Path],
) -> pd.DataFrame:
    required = {
        "patient_id",
        "fold",
        "split",
        "model_name",
        "run_name",
        "transition",
        "feature_name",
        "target_standardized",
        "predicted_standardized",
        "target_change",
        "predicted_change",
        "valid_mask",
        "transformed_change_unit",
        "checkpoint_sha256",
    }
    frames: list[pd.DataFrame] = []
    m2_sources = [source for source in sources if source.mode == "m2"]
    for source in m2_sources:
        path = source.prediction_dir / "test_paired_radiomics_predictions.csv"
        if not path.is_file():
            _missing_or_raise(
                config,
                issues,
                f"M2 {source.run_name}/fold_{source.fold} 缺原生 radiomics-head prediction",
                path,
            )
            continue
        frame = _read_csv(path, required, "M2 native radiomics-head prediction")
        if (
            not frame["fold"].eq(source.fold).all()
            or not frame["split"].eq("test").all()
        ):
            raise AggregationInputError(
                f"M2 radiomics prediction fold/split 非法: {path}"
            )
        if (
            not frame["run_name"].eq(source.run_name).all()
            or not frame["model_name"].eq("m2").all()
        ):
            raise AggregationInputError(
                f"M2 radiomics prediction run/model 不一致: {path}"
            )
        if not frame["checkpoint_sha256"].eq(source.checkpoint_sha256).all():
            raise AggregationInputError(
                f"M2 radiomics prediction checkpoint 不一致: {path}"
            )
        valid = frame["valid_mask"].astype(str).str.lower().isin(["true", "1"])
        if not valid.all():
            raise AggregationInputError(
                f"保存的 paired radiomics prediction 含 invalid mask: {path}"
            )
        if (
            not frame["transition"].isin(TRANSITION_ORDER).all()
            or not frame["feature_name"].isin(FEATURE_ORDER).all()
        ):
            raise AggregationInputError(
                f"M2 radiomics transition/feature schema 非法: {path}"
            )
        if frame.duplicated(["patient_id", "transition", "feature_name"]).any():
            raise AggregationInputError(f"M2 radiomics prediction 重复: {path}")
        numbers = frame[
            [
                "target_standardized",
                "predicted_standardized",
                "target_change",
                "predicted_change",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numbers.to_numpy()).all():
            raise AggregationInputError(f"M2 radiomics prediction 含 NaN/Inf: {path}")
        native_test = native.loc[
            native["aggregation_model"].eq("m2")
            & native["fold"].eq(source.fold)
            & native["decision_point"].eq("T0")
        ].copy()
        has_radiomics = (
            native_test["has_radiomics"].astype(str).str.lower().isin(["true", "1"])
        )
        paired_ids = set(native_test.loc[has_radiomics, "patient_id"].astype(str))
        expected_cells = {
            (patient_id, transition, feature)
            for patient_id in paired_ids
            for transition in TRANSITION_ORDER
            for feature in FEATURE_ORDER
        }
        actual_cells = set(
            zip(
                frame["patient_id"].astype(str),
                frame["transition"].astype(str),
                frame["feature_name"].astype(str),
            )
        )
        extra_cells = actual_cells.difference(expected_cells)
        missing_cells = expected_cells.difference(actual_cells)
        if extra_cells:
            raise AggregationInputError(
                "M2 radiomics prediction 含非 paired native-test cell: "
                f"{sorted(extra_cells)[:5]}"
            )
        if missing_cells:
            message = (
                f"M2 {source.run_name}/fold_{source.fold} radiomics head 缺 "
                f"{len(missing_cells)} 个 patient×transition×feature cell"
            )
            if config.allow_partial:
                _record_issue(issues, "缺失输入_missing_input", message, path)
            else:
                raise AggregationInputError(message)
        declared_rows = source.summary.get("test_paired_radiomics_rows")
        if not isinstance(declared_rows, int):
            _missing_or_raise(
                config,
                issues,
                f"{source.run_name}/fold_{source.fold} summary 缺 test_paired_radiomics_rows",
                source.summary_path,
            )
        elif len(frame) > declared_rows or (
            not config.allow_partial and len(frame) != declared_rows
        ):
            raise AggregationInputError(
                f"M2 radiomics rows={len(frame)} 与 summary "
                f"test_paired_radiomics_rows={declared_rows} 不一致"
            )
        elif len(frame) < declared_rows:
            _record_issue(
                issues,
                "缺失输入_missing_input",
                f"M2 radiomics rows={len(frame)} 少于 summary 声明 {declared_rows}",
                path,
            )
        frame["native_head"] = True
        frame["head_input_cn"] = "仅 predicted image delta；radiomics 非推理输入"
        frame["source_file"] = str(path)
        frames.append(frame)
        consumed.append(path)
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False)
    if output.duplicated(["patient_id", "transition", "feature_name"]).any():
        raise AggregationInputError("M2 OOF radiomics-head prediction 重复")
    return output


def radiomics_tables(
    frame: pd.DataFrame,
    baselines: Mapping[tuple[int, str, str], tuple[float, float, int]],
    transform_specs: Mapping[tuple[int, str], Mapping[str, Any]],
    raw_mapping: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    annotated = frame.copy()
    annotated["radiomics_transform_sha256"] = [
        transform_specs.get((int(fold), str(feature)), {}).get(
            "transform_sha256", pd.NA
        )
        for fold, feature in zip(annotated["fold"], annotated["feature_name"])
    ]
    annotated["radiomics_transform_spec_version"] = [
        transform_specs.get((int(fold), str(feature)), {}).get("spec_version", pd.NA)
        for fold, feature in zip(annotated["fold"], annotated["feature_name"])
    ]
    for (fold, feature), group in annotated.groupby(
        ["fold", "feature_name"], sort=False
    ):
        spec = transform_specs.get((int(fold), str(feature)))
        if spec is None:
            continue
        scale = float(spec["scale"])
        center = float(spec["center"])
        expected_target_change = (
            group["target_standardized"].to_numpy(dtype=float) * scale + center
        )
        expected_predicted_change = (
            group["predicted_standardized"].to_numpy(dtype=float) * scale + center
        )
        if not np.allclose(
            expected_target_change,
            group["target_change"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-7,
        ) or not np.allclose(
            expected_predicted_change,
            group["predicted_change"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise AggregationInputError(
                f"fold {fold}/{feature} standardized↔change 与锁定 transform 不一致"
            )
        if not group["transformed_change_unit"].eq(spec["value_transform"]).all():
            raise AggregationInputError(
                f"fold {fold}/{feature} transformed_change_unit 与 transform 不一致"
            )
        if "epsilon" in group.columns and not np.allclose(
            group["epsilon"].to_numpy(dtype=float),
            float(spec["epsilon"]),
            rtol=1e-8,
            atol=1e-10,
        ):
            raise AggregationInputError(
                f"fold {fold}/{feature} epsilon 与锁定 transform 不一致"
            )
        feature_index = FEATURE_ORDER.index(str(feature))
        expected_standardized: list[float] = []
        expected_change: list[float] = []
        for row in group.itertuples(index=False):
            patient_id = str(row.patient_id)
            raw_patient = raw_mapping.get(patient_id)
            if raw_patient is None:
                raise AggregationInputError(
                    f"fold {fold}/{patient_id} radiomics prediction 找不到 raw target"
                )
            raw_patient = np.asarray(raw_patient, dtype=float)
            if raw_patient.shape != (len(TRANSITION_ORDER), len(FEATURE_ORDER), 3):
                raise AggregationInputError(
                    f"{patient_id} raw radiomics shape 非法: {raw_patient.shape}"
                )
            transition_index = TRANSITION_ORDER.index(str(row.transition))
            start, end, raw_valid = raw_patient[transition_index, feature_index]
            if not bool(raw_valid) or not np.isfinite([start, end]).all():
                raise AggregationInputError(
                    f"{patient_id}/{row.transition}/{feature} prediction 与 raw valid mask 不一致"
                )
            transform_name = str(spec["value_transform"])
            epsilon = float(spec["epsilon"])
            if transform_name == "log_epsilon":
                raw_change = math.log(float(end) + epsilon) - math.log(
                    float(start) + epsilon
                )
            elif transform_name == "log1p":
                raw_change = math.log1p(float(end)) - math.log1p(float(start))
            elif transform_name == "identity":
                raw_change = float(end) - float(start)
            else:
                raise AggregationInputError(
                    f"fold {fold}/{feature} value_transform 非法: {transform_name}"
                )
            clipped_change = float(
                np.clip(
                    raw_change, float(spec["winsor_low"]), float(spec["winsor_high"])
                )
            )
            expected_change.append(clipped_change)
            expected_standardized.append(
                (clipped_change - float(spec["center"])) / float(spec["scale"])
            )
        if not np.allclose(
            np.asarray(expected_standardized),
            group["target_standardized"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
        ) or not np.allclose(
            np.asarray(expected_change),
            group["target_change"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise AggregationInputError(
                f"fold {fold}/{feature} prediction target 与 raw patient/transition target 不一致"
            )
    annotated["train_mean_baseline_standardized"] = [
        baselines.get(
            (int(fold), str(transition), str(feature)),
            (math.nan, math.nan, 0),
        )[0]
        for fold, transition, feature in zip(
            annotated["fold"],
            annotated["transition"],
            annotated["feature_name"],
        )
    ]
    annotated["train_mean_baseline_change"] = [
        baselines.get(
            (int(fold), str(transition), str(feature)),
            (math.nan, math.nan, 0),
        )[1]
        for fold, transition, feature in zip(
            annotated["fold"],
            annotated["transition"],
            annotated["feature_name"],
        )
    ]
    annotated["train_mean_baseline_n"] = [
        baselines.get(
            (int(fold), str(transition), str(feature)),
            (math.nan, math.nan, 0),
        )[2]
        for fold, transition, feature in zip(
            annotated["fold"],
            annotated["transition"],
            annotated["feature_name"],
        )
    ]
    rows: list[dict[str, Any]] = []
    groupings: list[tuple[str, list[str]]] = [
        ("fold_transition_feature", ["fold", "transition", "feature_name"]),
        ("oof_transition_feature", ["transition", "feature_name"]),
        ("fold_feature_all_transitions", ["fold", "feature_name"]),
        ("oof_feature_all_transitions", ["feature_name"]),
    ]
    for level, keys in groupings:
        for key, group in annotated.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            values = dict(zip(keys, key))
            group_has_baseline = bool(
                np.isfinite(group["train_mean_baseline_change"]).all()
                and np.isfinite(group["train_mean_baseline_standardized"]).all()
            )
            raw_baseline = (
                group["train_mean_baseline_change"].to_numpy(dtype=float)
                if group_has_baseline
                else None
            )
            standardized_baseline = (
                group["train_mean_baseline_standardized"].to_numpy(dtype=float)
                if group_has_baseline
                else None
            )
            raw_metrics = _regression_values(
                group["target_change"].to_numpy(),
                group["predicted_change"].to_numpy(),
                raw_baseline,
            )
            standardized_metrics = _regression_values(
                group["target_standardized"].to_numpy(),
                group["predicted_standardized"].to_numpy(),
                standardized_baseline,
            )
            baseline_changes = group["train_mean_baseline_change"].dropna().unique()
            baseline_standardized = (
                group["train_mean_baseline_standardized"].dropna().unique()
            )
            baseline_ns = group["train_mean_baseline_n"].dropna().to_numpy(dtype=int)
            variation_sources = []
            if group["fold"].nunique() > 1:
                variation_sources.append("fold")
            if group["transition"].nunique() > 1:
                variation_sources.append("transition")
            rows.append(
                {
                    "aggregation_level": level,
                    "model_name": "m2",
                    "native_head": True,
                    "fold": values.get("fold", pd.NA),
                    "transition": values.get("transition", "全部"),
                    "feature_name": values["feature_name"],
                    "feature_label": FEATURE_DISPLAY[values["feature_name"]],
                    "n": raw_metrics.pop("n"),
                    **raw_metrics,
                    **{
                        f"standardized_{name}": value
                        for name, value in standardized_metrics.items()
                        if name != "n"
                    },
                    "train_mean_baseline_change": (
                        float(baseline_changes[0])
                        if len(baseline_changes) == 1
                        else math.nan
                    ),
                    "train_mean_baseline_standardized": (
                        float(baseline_standardized[0])
                        if len(baseline_standardized) == 1
                        else math.nan
                    ),
                    "train_mean_baseline_n_min": (
                        int(baseline_ns.min()) if baseline_ns.size else 0
                    ),
                    "train_mean_baseline_n_max": (
                        int(baseline_ns.max()) if baseline_ns.size else 0
                    ),
                    "train_mean_baseline_unique_count": int(len(baseline_changes)),
                    "train_mean_baseline_varies_within_group": bool(
                        len(baseline_changes) > 1
                    ),
                    "train_mean_baseline_variation_sources": (
                        "+".join(variation_sources) if variation_sources else "无_none"
                    ),
                    "mean_baseline_scope_cn": (
                        "每个 fold×transition×feature 仅由该 fold train 计算"
                        if group_has_baseline
                        else "缺失；未计算且未用 test 均值替代"
                    ),
                    "direction_rule": "sign(target_change) == sign(predicted_change)，零单独成类",
                    "variance_definition": "样本方差，ddof=1",
                }
            )
    distribution_rows: list[dict[str, Any]] = []
    for (transition, feature), group in annotated.groupby(
        ["transition", "feature_name"], sort=False
    ):
        target = group["target_change"].to_numpy(dtype=float)
        prediction = group["predicted_change"].to_numpy(dtype=float)
        distribution_rows.append(
            {
                "aggregation_level": "pooled_oof_transition_feature",
                "transition": transition,
                "feature_name": feature,
                "n": len(group),
                "target_mean": float(target.mean()),
                "target_std": float(target.std(ddof=1)),
                "target_median": float(np.median(target)),
                "target_q05": float(np.quantile(target, 0.05)),
                "target_q95": float(np.quantile(target, 0.95)),
                "predicted_mean": float(prediction.mean()),
                "predicted_std": float(prediction.std(ddof=1)),
                "predicted_median": float(np.median(prediction)),
                "predicted_q05": float(np.quantile(prediction, 0.05)),
                "predicted_q95": float(np.quantile(prediction, 0.95)),
            }
        )
    return annotated, pd.DataFrame(rows), pd.DataFrame(distribution_rows)


def _discover_controls(
    sources: Sequence[EvaluationSource],
    native: pd.DataFrame,
    config: AggregationConfig,
    issues: list[dict[str, str]],
    consumed: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取与已选 checkpoint 精确匹配的 C0/C1/C2 和 Ridge probe。"""

    if not CONTROL_METRIC_ROOT.is_dir():
        _missing_or_raise(
            config,
            issues,
            "未发现正式 controls 输出；strict 模式要求 C0/C1/C2 五折齐全",
            CONTROL_METRIC_ROOT,
        )
        return pd.DataFrame(), pd.DataFrame()
    selected = {(source.mode, source.fold): source for source in sources}
    expected_folds = set.intersection(
        *(
            {source.fold for source in sources if source.mode == mode}
            for mode in MODEL_ORDER
        )
    )
    candidates: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(CONTROL_METRIC_ROOT.rglob("selection.json")):
        payload = _read_json(path)
        if (
            config.controls_name is not None
            and payload.get("output_name") != config.controls_name
        ):
            continue
        fold = payload.get("fold")
        checkpoints = payload.get("checkpoints")
        if not isinstance(fold, int) or not isinstance(checkpoints, dict):
            raise AggregationInputError(f"controls selection schema 非法: {path}")
        match = True
        for mode in MODEL_ORDER:
            source = selected.get((mode, fold))
            record = checkpoints.get(mode)
            if source is None or not isinstance(record, dict):
                match = False
                break
            if (
                record.get("run_name") != source.run_name
                or record.get("sha256") != source.checkpoint_sha256
            ):
                match = False
                break
        if match:
            candidates.setdefault(fold, []).append((path, payload))
    ambiguous = {fold: items for fold, items in candidates.items() if len(items) > 1}
    if ambiguous:
        raise AggregationInputError(
            "同一 fold 有多个与所选 checkpoint 匹配的 controls；请用 --controls-name 指定："
            + "; ".join(
                f"fold {fold}: {[str(path) for path, _ in items]}"
                for fold, items in ambiguous.items()
            )
        )
    if not candidates:
        _missing_or_raise(
            config,
            issues,
            "未找到与已选 M0/M1/M2 checkpoint SHA256 精确匹配的 controls；"
            "strict 模式要求 C0/C1/C2 五折齐全",
            CONTROL_METRIC_ROOT,
        )
        return pd.DataFrame(), pd.DataFrame()
    missing_control_folds = expected_folds.difference(candidates)
    if missing_control_folds:
        _missing_or_raise(
            config,
            issues,
            f"正式 controls 缺 fold {sorted(missing_control_folds)}",
            CONTROL_METRIC_ROOT,
        )
    control_frames: list[pd.DataFrame] = []
    probe_frames: list[pd.DataFrame] = []
    control_implementation_hashes: set[str] = set()
    matched_output_names: set[str] = set()
    for fold, ((selection_path, payload),) in sorted(candidates.items()):
        output_name = str(payload["output_name"])
        controls_hash = str(payload.get("controls_implementation_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", controls_hash):
            raise AggregationInputError(
                f"controls implementation SHA256 非法: {selection_path}"
            )
        control_implementation_hashes.add(controls_hash)
        matched_output_names.add(output_name)
        selected_m2 = selected[("m2", fold)]
        declared_raw_hash = payload.get("raw_radiomics_sha256")
        expected_raw_hash = selected_m2.summary.get("radiomics_raw_targets_sha256")
        if expected_raw_hash is not None and declared_raw_hash != expected_raw_hash:
            raise AggregationInputError(
                f"controls raw radiomics hash 与 M2 evaluation 不一致: {selection_path}"
            )
        namespace = selection_path.parent.name
        prediction_dir = (
            CONTROL_PREDICTION_ROOT / output_name / f"fold_{fold}" / namespace
        )
        control_path = prediction_dir / "paired_control_predictions.csv"
        probe_path = prediction_dir / "posthoc_ridge_test_predictions.csv"
        if not control_path.is_file() or not probe_path.is_file():
            message = f"controls selection 存在但 prediction 文件缺失: {prediction_dir}"
            if config.allow_partial:
                _record_issue(issues, "缺失输入_missing_input", message, prediction_dir)
                continue
            raise AggregationInputError(message)
        control = _read_csv(
            control_path,
            {
                "patient_id",
                "fold",
                "split",
                "control_name",
                "decision_point",
                "y_true",
                "predicted_probability",
                "predicted_label",
                "threshold",
                "paired_subset",
                "radiomics_used_as_input",
                "source_checkpoint_sha256",
            },
            "paired control prediction",
        )
        control = control.loc[control["split"].eq("test")].copy()
        if not control["fold"].eq(fold).all():
            raise AggregationInputError(
                f"control prediction fold 不一致: {control_path}"
            )
        _validate_probability_frame(control, str(control_path))
        if control.duplicated(["patient_id", "control_name", "decision_point"]).any():
            raise AggregationInputError(f"control prediction 重复: {control_path}")
        expected_controls = {
            "C0_radiomics_only",
            "C1_m2_image_plus_radiomics",
            "C2_m2_image_only",
        }
        if set(control["control_name"]) != expected_controls:
            raise AggregationInputError(
                f"control_name 必须恰好为 C0/C1/C2: {control_path}；"
                f"实际={sorted(set(control['control_name']))}"
            )
        if set(control["decision_point"]) != set(DECISION_ORDER):
            raise AggregationInputError(
                f"controls 必须覆盖三个 decision point: {control_path}"
            )
        paired_flag = (
            control["paired_subset"].astype(str).str.lower().isin(["true", "1"])
        )
        if not paired_flag.all():
            raise AggregationInputError(
                f"controls 含非 paired subset 行: {control_path}"
            )
        radiomics_flag = (
            control["radiomics_used_as_input"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        c2_mask = control["control_name"].eq("C2_m2_image_only")
        if not radiomics_flag.loc[~c2_mask].all() or radiomics_flag.loc[c2_mask].any():
            raise AggregationInputError(
                f"C0/C1/C2 radiomics input 标记不符合控制定义: {control_path}"
            )
        m2_source = selected[("m2", fold)]
        if (
            not control["source_checkpoint_sha256"]
            .eq(m2_source.checkpoint_sha256)
            .all()
        ):
            raise AggregationInputError(
                f"controls 不是来自已选 M2 checkpoint: {control_path}"
            )
        paired_ids_by_point: dict[str, set[str]] = {}
        paired_labels_by_point: dict[str, dict[str, int]] = {}
        for point in DECISION_ORDER:
            point_groups = {
                name: group.copy()
                for name, group in control.loc[
                    control["decision_point"].eq(point)
                ].groupby("control_name", sort=False)
            }
            if set(point_groups) != expected_controls:
                raise AggregationInputError(
                    f"fold {fold}/{point} 未同时包含 C0/C1/C2: {control_path}"
                )
            reference = point_groups["C0_radiomics_only"]
            reference_ids = set(reference["patient_id"].astype(str))
            reference_labels = dict(
                zip(
                    reference["patient_id"].astype(str),
                    reference["y_true"].astype(int),
                )
            )
            if not reference_ids:
                raise AggregationInputError(
                    f"fold {fold}/{point} paired control subset 为空"
                )
            for name, group in point_groups.items():
                ids = set(group["patient_id"].astype(str))
                labels = dict(
                    zip(group["patient_id"].astype(str), group["y_true"].astype(int))
                )
                if ids != reference_ids or labels != reference_labels:
                    raise AggregationInputError(
                        f"fold {fold}/{point} 的 {name} patient set/y_true 与 C0 不一致"
                    )
            declared_n = int(
                payload.get("paired_counts", {}).get(point, {}).get("test", -1)
            )
            if declared_n >= 0 and declared_n != len(reference_ids):
                raise AggregationInputError(
                    f"fold {fold}/{point} controls n={len(reference_ids)} 与 selection n={declared_n} 不一致"
                )
            paired_ids_by_point[point] = reference_ids
            paired_labels_by_point[point] = reference_labels

            # C2 必须是正式 M2 image-only test prediction 的逐行切片；不允许重拟合。
            c2 = point_groups["C2_m2_image_only"].set_index("patient_id")
            native_m2 = native.loc[
                native["aggregation_model"].eq("m2")
                & native["fold"].eq(fold)
                & native["decision_point"].eq(point)
            ].set_index("patient_id")
            if not reference_ids.issubset(set(native_m2.index.astype(str))):
                raise AggregationInputError(
                    f"fold {fold}/{point} C2 patient 不是正式 M2 test subset"
                )
            native_slice = native_m2.loc[list(c2.index)]
            if not np.array_equal(
                c2["y_true"].to_numpy(dtype=int),
                native_slice["y_true"].to_numpy(dtype=int),
            ):
                raise AggregationInputError(
                    f"fold {fold}/{point} C2 y_true 与正式 M2 prediction 不一致"
                )
            for column in ("predicted_probability", "threshold"):
                if not np.allclose(
                    c2[column].to_numpy(dtype=float),
                    native_slice[column].to_numpy(dtype=float),
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise AggregationInputError(
                        f"fold {fold}/{point} C2 {column} 与正式 M2 prediction 不一致"
                    )
        control["controls_output_name"] = output_name
        control["source_file"] = str(control_path)
        control_frames.append(control)

        probe = _read_csv(
            probe_path,
            {
                "patient_id",
                "fold",
                "split",
                "model_name",
                "run_name",
                "decision_point",
                "transition",
                "feature_name",
                "posthoc_ridge_target_standardized",
                "posthoc_ridge_predicted_standardized",
                "posthoc_ridge_target_change",
                "posthoc_ridge_predicted_change",
                "is_native_m2_head",
                "source_checkpoint_sha256",
            },
            "posthoc Ridge probe prediction",
        )
        if not probe["fold"].eq(fold).all() or not probe["split"].eq("test").all():
            raise AggregationInputError(f"Ridge probe fold/split 不一致: {probe_path}")
        native_flag = (
            probe["is_native_m2_head"].astype(str).str.lower().isin(["true", "1"])
        )
        if native_flag.any():
            raise AggregationInputError(
                f"后验 Ridge probe 被错误标为 native head: {probe_path}"
            )
        for mode in MODEL_ORDER:
            part = probe.loc[probe["model_name"].eq(mode)]
            source = selected[(mode, fold)]
            if (
                part.empty
                or not part["run_name"].eq(source.run_name).all()
                or not part["source_checkpoint_sha256"]
                .eq(source.checkpoint_sha256)
                .all()
            ):
                raise AggregationInputError(
                    f"Ridge probe {mode} 来源不一致: {probe_path}"
                )
        if probe.duplicated(
            ["patient_id", "model_name", "decision_point", "feature_name"]
        ).any():
            raise AggregationInputError(f"Ridge probe prediction 重复: {probe_path}")
        point_to_transition = dict(zip(DECISION_ORDER, TRANSITION_ORDER))
        expected_probe_keys = {
            (mode, point, point_to_transition[point], feature)
            for mode in MODEL_ORDER
            for point in DECISION_ORDER
            for feature in FEATURE_ORDER
        }
        actual_probe_keys = set(
            zip(
                probe["model_name"],
                probe["decision_point"],
                probe["transition"],
                probe["feature_name"],
            )
        )
        if actual_probe_keys != expected_probe_keys:
            raise AggregationInputError(
                f"Ridge probe 未完整覆盖 3 model×3 transition×4 feature: {probe_path}"
            )
        for mode in MODEL_ORDER:
            for point in DECISION_ORDER:
                for feature in FEATURE_ORDER:
                    ids = set(
                        probe.loc[
                            probe["model_name"].eq(mode)
                            & probe["decision_point"].eq(point)
                            & probe["feature_name"].eq(feature),
                            "patient_id",
                        ].astype(str)
                    )
                    if ids != paired_ids_by_point[point]:
                        raise AggregationInputError(
                            f"fold {fold}/{mode}/{point}/{feature} Ridge probe patient set "
                            "与 controls paired subset 不一致"
                        )
        probe["controls_output_name"] = output_name
        probe["source_file"] = str(probe_path)
        probe_frames.append(probe)
        consumed.extend([selection_path, control_path, probe_path])
    controls = (
        pd.concat(control_frames, ignore_index=True, sort=False)
        if control_frames
        else pd.DataFrame()
    )
    probes = (
        pd.concat(probe_frames, ignore_index=True, sort=False)
        if probe_frames
        else pd.DataFrame()
    )
    if len(control_implementation_hashes) != 1:
        raise AggregationInputError(
            "五折 controls implementation SHA256 不一致: "
            f"{sorted(control_implementation_hashes)}"
        )
    if len(matched_output_names) != 1:
        raise AggregationInputError(
            f"五折 controls output_name 不一致: {sorted(matched_output_names)}"
        )
    if (
        not controls.empty
        and controls.duplicated(["patient_id", "control_name", "decision_point"]).any()
    ):
        raise AggregationInputError("controls OOF 中患者跨 fold 重复")
    if (
        not probes.empty
        and probes.duplicated(
            ["patient_id", "model_name", "decision_point", "feature_name"]
        ).any()
    ):
        raise AggregationInputError("Ridge probe OOF 中患者跨 fold 重复")
    return controls, probes


def control_tables(
    controls: pd.DataFrame, method_input_contract: Mapping[str, Any]
) -> pd.DataFrame:
    if controls.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["control_name", "decision_point"]
    for level, group_keys in (
        ("fold_test", keys + ["fold"]),
        ("pooled_oof_test", keys),
    ):
        for key, group in controls.groupby(group_keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            values = dict(zip(group_keys, key))
            rows.append(
                {
                    "aggregation_level": level,
                    "control_name": values["control_name"],
                    "control_label_cn": {
                        "C0_radiomics_only": "C0 Radiomics-only（推理依赖 radiomics）",
                        "C1_m2_image_plus_radiomics": (
                            "C1 M2 ROI辅助影像 + radiomics（推理依赖 radiomics）"
                        ),
                        "C2_m2_image_only": (
                            "C2 M2 "
                            + str(
                                method_input_contract.get(
                                    "method_label_cn", ROI_ASSISTED_FULL_CN
                                )
                            )
                        ),
                    }.get(values["control_name"], str(values["control_name"])),
                    "decision_point": values["decision_point"],
                    "fold": values.get("fold", pd.NA),
                    "cohort_scope_cn": "radiomics 可用 paired test subset",
                    **_classification_values(group),
                }
            )
    return pd.DataFrame(rows)


def probe_table(probes: pd.DataFrame) -> pd.DataFrame:
    if probes.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    base = ["model_name", "run_name", "decision_point", "transition", "feature_name"]
    for level, keys in (("fold_test", base + ["fold"]), ("pooled_oof_test", base)):
        for key, group in probes.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            values = dict(zip(keys, key))
            raw = _regression_values(
                group["posthoc_ridge_target_change"].to_numpy(),
                group["posthoc_ridge_predicted_change"].to_numpy(),
            )
            standardized = _regression_values(
                group["posthoc_ridge_target_standardized"].to_numpy(),
                group["posthoc_ridge_predicted_standardized"].to_numpy(),
            )
            rows.append(
                {
                    "aggregation_level": level,
                    "model_name": values["model_name"],
                    "run_name": values["run_name"],
                    "decision_point": values["decision_point"],
                    "transition": values["transition"],
                    "feature_name": values["feature_name"],
                    "fold": values.get("fold", pd.NA),
                    "n": raw.pop("n"),
                    **raw,
                    **{
                        f"standardized_{name}": value
                        for name, value in standardized.items()
                        if name != "n"
                    },
                    "probe_type": "posthoc_train_only_multioutput_ridge",
                    "is_native_m2_head": False,
                }
            )
    return pd.DataFrame(rows)


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # 当前节点的 Noto CJK TTC 由 matplotlib 注册为 JP family；该字库
            # 同时覆盖简体中文与 Latin。不要退到 DroidSansFallbackFull，它在
            # 本机缺少 Latin glyph，会把 AUROC/T0/n 等渲染为空框。
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _finish_figure(
    fig: plt.Figure,
    path: Path,
    manifest: list[dict[str, Any]],
    *,
    title: str,
    source: str,
    cohort: str,
    sample_n: str,
    folds: str,
    errorbar: str,
    decision_points: str,
    notes: str,
) -> None:
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.text(
        0.01,
        0.008,
        f"范围：{cohort}；n={sample_n}；fold={folds}；误差条：{errorbar}",
        fontsize=8,
        color="#444444",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "figure_file": path.name,
            "title_cn": title,
            "source_metric": source,
            "cohort_scope_cn": cohort,
            "sample_n": sample_n,
            "folds": folds,
            "errorbar_definition_cn": errorbar,
            "decision_points": decision_points,
            "notes_cn": notes,
        }
    )


def _classification_figure(
    folds: pd.DataFrame,
    oof: pd.DataFrame,
    metric: str,
    path: Path,
    manifest: list[dict[str, Any]],
    method_input_contract: Mapping[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    modes = [mode for mode in MODEL_ORDER if mode in set(folds["model_name"])]
    x = np.arange(len(DECISION_ORDER), dtype=float)
    width = 0.8 / max(len(modes), 1)
    colors = {"m0": "#4C78A8", "m1": "#F58518", "m2": "#54A24B"}
    for index, mode in enumerate(modes):
        means, errors, ns = [], [], []
        positions = x - 0.4 + width / 2 + index * width
        for point in DECISION_ORDER:
            values = folds.loc[
                folds["model_name"].eq(mode) & folds["decision_point"].eq(point), metric
            ].dropna()
            means.append(float(values.mean()) if len(values) else math.nan)
            errors.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            row = oof.loc[oof["model_name"].eq(mode) & oof["decision_point"].eq(point)]
            ns.append(int(row["n"].iloc[0]) if not row.empty else 0)
        bars = ax.bar(
            positions,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=MODEL_DISPLAY[mode],
            color=colors[mode],
            alpha=0.88,
        )
        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    method_label = str(
        method_input_contract.get("method_label_cn", "影像输入契约未确认")
    )
    title = f"M0/M1/M2 冻结 {method_label} readout：{METRIC_CN[metric]}"
    ax.set_title(title)
    ax.set_ylabel(METRIC_CN[metric])
    ax.set_xticks(x, DECISION_ORDER)
    ax.set_ylim(0, 1.08)
    if metric == "auroc":
        ax.axhline(0.5, color="#999999", lw=1, ls="--", label="随机基线 0.5")
        baseline_note = "虚线为随机 AUROC=0.5"
    else:
        prevalences = oof["prevalence"].dropna().to_numpy(dtype=float)
        unique_prevalences = np.unique(np.round(prevalences, 12))
        if unique_prevalences.size == 1:
            prevalence = float(unique_prevalences[0])
            ax.axhline(
                prevalence,
                color="#999999",
                lw=1,
                ls="--",
                label=f"no-skill prevalence={prevalence:.3f}",
            )
            baseline_note = f"虚线为 no-skill prevalence={prevalence:.3f}"
        else:
            baseline_note = "各部分 cohort prevalence 不同，未画单一 no-skill 线"
    ax.legend(title="模型")
    fold_count = sorted(folds["fold"].unique().tolist())
    sample_n = ", ".join(
        f"{MODEL_DISPLAY[m]}:{int(oof.loc[oof['model_name'].eq(m), 'n'].max())}"
        for m in modes
    )
    _finish_figure(
        fig,
        path,
        manifest,
        title=title,
        source="classification_fold_metrics.csv/classification_oof_metrics.csv",
        cohort=f"完整或显式部分 OOF test；{method_label}",
        sample_n=sample_n,
        folds=",".join(map(str, fold_count)),
        errorbar="fold 间样本标准差（单 fold smoke 为 0，仅用于管线自测）",
        decision_points=",".join(DECISION_ORDER),
        notes=(
            "柱高为 fold 均值，柱上 n 为对应 pooled OOF 患者数；"
            f"{baseline_note}；{method_input_contract.get('input_definition_cn', '')}"
        ),
    )


def _patient_level_transition(frame: pd.DataFrame) -> pd.DataFrame:
    patient = frame.groupby(
        ["aggregation_model", "run_name", "fold", "patient_id"], as_index=False
    ).agg(
        n_transitions=("transition", "size"),
        normalized_learned_error_sum=("learned_error", "sum"),
        normalized_copy_error_sum=("copy_error", "sum"),
        raw_learned_mse_sum=("learned_raw_mse", "sum"),
        raw_copy_mse_sum=("copy_raw_mse", "sum"),
        mean_step_gain_diagnostic=("gain", "mean"),
    )
    patient["learned_error"] = (
        patient["normalized_learned_error_sum"] / patient["n_transitions"]
    )
    patient["copy_error"] = (
        patient["normalized_copy_error_sum"] / patient["n_transitions"]
    )
    patient["learned_raw_mse"] = (
        patient["raw_learned_mse_sum"] / patient["n_transitions"]
    )
    patient["copy_raw_mse"] = patient["raw_copy_mse_sum"] / patient["n_transitions"]
    patient["gain"] = (
        patient["normalized_copy_error_sum"] - patient["normalized_learned_error_sum"]
    ) / patient["normalized_copy_error_sum"].clip(lower=1e-8)
    patient["raw_gain"] = (
        patient["raw_copy_mse_sum"] - patient["raw_learned_mse_sum"]
    ) / patient["raw_copy_mse_sum"].clip(lower=1e-8)
    patient["gain_definition"] = (
        "patient (sum(copy_error)-sum(learned_error))/max(sum(copy_error),1e-8)"
    )
    return patient


def _bootstrap_mean_ci(
    values: np.ndarray, seed: int, replicates: int = 1000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for index in range(replicates):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _transition_error_figure(
    frame: pd.DataFrame,
    path: Path,
    manifest: list[dict[str, Any]],
    seed: int,
) -> None:
    patient = _patient_level_transition(frame)
    modes = [mode for mode in MODEL_ORDER if mode in set(patient["aggregation_model"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    specs = [
        ("learned_error", "copy_error", "独立 LayerNorm 后误差"),
        ("learned_raw_mse", "copy_raw_mse", "原始 latent MSE"),
    ]
    x = np.arange(len(modes))
    width = 0.34
    for axis, (learned_col, copy_col, label) in zip(axes, specs):
        for offset, column, legend, color in (
            (-width / 2, learned_col, "Learned next-state", "#4C78A8"),
            (width / 2, copy_col, "Copy-current", "#E45756"),
        ):
            means, low, high = [], [], []
            for mode in modes:
                values = patient.loc[
                    patient["aggregation_model"].eq(mode), column
                ].to_numpy()
                mean = float(values.mean())
                lo, hi = _bootstrap_mean_ci(values, _stable_seed(seed, mode, column))
                means.append(mean)
                low.append(mean - lo)
                high.append(hi - mean)
            axis.bar(
                x + offset,
                means,
                width=width,
                yerr=np.asarray([low, high]),
                capsize=3,
                label=legend,
                color=color,
                alpha=0.86,
            )
        axis.set_title(label)
        axis.set_xticks(x, [MODEL_DISPLAY[mode] for mode in modes])
        axis.set_ylabel("误差（越低越好）")
        axis.legend(fontsize=8)
    title = "Learned next-state 与 copy-current 基线误差"
    fig.suptitle(title)
    n = ", ".join(
        f"{MODEL_DISPLAY[m]}:{patient.loc[patient['aggregation_model'].eq(m), 'patient_id'].nunique()}"
        for m in modes
    )
    _finish_figure(
        fig,
        path,
        manifest,
        title=title,
        source="transition_predictions_aggregated.csv",
        cohort="OOF test；先对每患者三个 transition 取均值",
        sample_n=n,
        folds=",".join(map(str, sorted(frame["fold"].unique()))),
        errorbar="患者 bootstrap 95% percentile CI（1000 次）",
        decision_points=",".join(TRANSITION_ORDER),
        notes="左图为独立 feature-wise LayerNorm MSE，右图为原始 latent MSE",
    )


def _gain_figure(
    frame: pd.DataFrame, path: Path, manifest: list[dict[str, Any]]
) -> None:
    patient = _patient_level_transition(frame)
    modes = [mode for mode in MODEL_ORDER if mode in set(patient["aggregation_model"])]
    data = [
        patient.loc[patient["aggregation_model"].eq(mode), "gain"].to_numpy()
        for mode in modes
    ]
    zoom_modes = [mode for mode in modes if mode != "m0"]
    fig, axes = plt.subplots(
        1,
        2 if zoom_modes else 1,
        figsize=(11 if zoom_modes else 8, 5.5),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.25, 1]} if zoom_modes else None,
    )
    ax = axes[0, 0]
    ax.boxplot(
        data, tick_labels=[MODEL_DISPLAY[mode] for mode in modes], showfliers=False
    )
    ax.axhline(0, color="#E45756", ls="--", lw=1)
    title = "Normalized transition gain 的患者分布（全尺度与局部）"
    ax.set_title("全尺度")
    ax.set_ylabel("(copy error − learned error) / copy error")
    for index, values in enumerate(data, start=1):
        median = float(np.median(values))
        ax.text(index, median, f"  中位数={median:.3f}", va="center", fontsize=8)
    if zoom_modes:
        zoom_ax = axes[0, 1]
        zoom_data = [
            patient.loc[patient["aggregation_model"].eq(mode), "gain"].to_numpy()
            for mode in zoom_modes
        ]
        zoom_ax.boxplot(
            zoom_data,
            tick_labels=[MODEL_DISPLAY[mode] for mode in zoom_modes],
            showfliers=False,
        )
        zoom_ax.axhline(0, color="#E45756", ls="--", lw=1)
        finite = np.concatenate(zoom_data)
        low, high = np.quantile(finite, [0.05, 0.95])
        margin = max(float(high - low) * 0.2, 0.05)
        zoom_ax.set_ylim(float(low - margin), float(high + margin))
        zoom_ax.set_title("M1/M2 局部尺度（5%–95%）")
        for index, values in enumerate(zoom_data, start=1):
            median = float(np.median(values))
            zoom_ax.text(
                index, median, f"  中位数={median:.3f}", va="center", fontsize=8
            )
    fig.suptitle(title)
    n = ", ".join(f"{MODEL_DISPLAY[m]}:{len(values)}" for m, values in zip(modes, data))
    _finish_figure(
        fig,
        path,
        manifest,
        title=title,
        source="transition_predictions_aggregated.csv",
        cohort="OOF test；每患者先跨三个 transition 求误差和，再计算稳定 ratio-of-sums",
        sample_n=n,
        folds=",".join(map(str, sorted(frame["fold"].unique()))),
        errorbar=(
            "无；箱体=IQR，中线=中位数，须=1.5×IQR，离群点不显示；"
            "右图纵轴按 M1/M2 患者 gain 的5%–95%局部缩放"
        ),
        decision_points=",".join(TRANSITION_ORDER),
        notes=(
            "gain>0 表示 learned prediction 优于 copy-current；"
            "逐 step ratio 的算术平均仅保留为 CSV 诊断，不用于本图；"
            "左图保留全尺度，右图用于观察 M1/M2 的零附近差异"
        ),
    )


def _radiomics_scatter_figure(
    frame: pd.DataFrame, path: Path, manifest: list[dict[str, Any]]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    total_n = 0
    for axis, feature in zip(axes.ravel(), FEATURE_ORDER):
        group = frame.loc[frame["feature_name"].eq(feature)]
        target = group["target_change"].to_numpy(dtype=float)
        prediction = group["predicted_change"].to_numpy(dtype=float)
        total_n += len(group)
        axis.scatter(
            target, prediction, s=10, alpha=0.35, color="#4C78A8", edgecolors="none"
        )
        low = min(float(target.min()), float(prediction.min()))
        high = max(float(target.max()), float(prediction.max()))
        axis.plot([low, high], [low, high], color="#E45756", ls="--", lw=1)
        rho = _correlation(target, prediction, "spearman")
        axis.set_title(
            f"{FEATURE_DISPLAY[feature]}（n={len(group)}, Spearman={rho:.3f}）"
        )
        axis.set_xlabel("真实变化（fold transform 后单位）")
        axis.set_ylabel("预测变化")
    title = "M2 原生 radiomics head：真实变化与预测变化"
    fig.suptitle(title)
    _finish_figure(
        fig,
        path,
        manifest,
        title=title,
        source="radiomics_head_predictions_aggregated.csv",
        cohort="radiomics 可用 OOF test subset；三个相邻 transition 合并",
        sample_n=f"总特征行 {total_n}；患者 {frame['patient_id'].nunique()}",
        folds=",".join(map(str, sorted(frame["fold"].unique()))),
        errorbar="无；虚线为 y=x",
        decision_points=",".join(TRANSITION_ORDER),
        notes="FTV/LD 为 log-epsilon change，Sphericity/BPE 为 absolute change；均经 fold-train winsorization",
    )


def _radiomics_spearman_figure(
    metrics: pd.DataFrame, path: Path, manifest: list[dict[str, Any]]
) -> None:
    pooled = metrics.loc[metrics["aggregation_level"].eq("oof_transition_feature")]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(FEATURE_ORDER))
    width = 0.24
    for index, transition in enumerate(TRANSITION_ORDER):
        values, ns = [], []
        for feature in FEATURE_ORDER:
            row = pooled.loc[
                pooled["transition"].eq(transition) & pooled["feature_name"].eq(feature)
            ]
            values.append(float(row["spearman"].iloc[0]) if not row.empty else math.nan)
            ns.append(int(row["n"].iloc[0]) if not row.empty else 0)
        positions = x + (index - 1) * width
        bars = ax.bar(positions, values, width=width, label=transition)
        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={n}",
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=7,
                rotation=90,
            )
    title = "M2 原生 radiomics head 的 Spearman 相关"
    ax.set_title(title)
    ax.set_ylabel("Spearman ρ")
    ax.set_xticks(x, [FEATURE_DISPLAY[feature] for feature in FEATURE_ORDER])
    ax.axhline(0, color="#777777", lw=1)
    ax.set_ylim(-1, 1)
    ax.legend(title="Transition")
    _finish_figure(
        fig,
        path,
        manifest,
        title=title,
        source="radiomics_head_metrics.csv",
        cohort="radiomics 可用 pooled OOF test subset",
        sample_n=f"患者 {int(pooled['n'].max()) if not pooled.empty else 0}/transition",
        folds=",".join(map(str, sorted(metrics["fold"].dropna().unique().astype(int)))),
        errorbar="无；显示 pooled OOF 点估计，柱上为有效 patient-feature 行数",
        decision_points=",".join(TRANSITION_ORDER),
        notes="相关性在各 feature 的 fold-specific transformed change 单位上池化计算",
    )


def _shortcut_figure(
    metrics: pd.DataFrame,
    perturbation: str,
    path: Path,
    manifest: list[dict[str, Any]],
) -> None:
    pooled = metrics.loc[
        metrics["aggregation_level"].eq("pooled_oof_test")
        & metrics["perturbation"].eq(perturbation)
    ].copy()
    if pooled.empty:
        raise AggregationInputError(f"没有 {perturbation} pooled shortcut metric")
    pooled["sort_model"] = pooled["model_name"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    pooled["sort_point"] = pooled["decision_point"].map(
        {p: i for i, p in enumerate(DECISION_ORDER)}
    )
    pooled = pooled.sort_values(["sort_point", "sort_model"])
    labels = [
        f"{MODEL_DISPLAY[m]}\n{p}"
        for m, p in zip(pooled["model_name"], pooled["decision_point"])
    ]
    x = np.arange(len(pooled))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(pooled)), 5.5))
    ax.plot(x, pooled["native_auroc"], "o-", label="Native", color="#4C78A8")
    ax.plot(
        x,
        pooled["perturbed_auroc"],
        "o-",
        label=PERTURBATION_DISPLAY[perturbation],
        color="#E45756",
    )
    for pos, row in zip(x, pooled.itertuples(index=False)):
        ax.text(
            pos,
            max(row.native_auroc, row.perturbed_auroc) + 0.025,
            f"n={row.n}",
            ha="center",
            fontsize=7,
        )
    title = f"{PERTURBATION_DISPLAY[perturbation]} 前后 pooled OOF AUROC"
    ax.set_title(title)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="#999999", ls="--", lw=1)
    ax.legend()
    _finish_figure(
        fig,
        path,
        manifest,
        title=title,
        source="shortcut_performance_metrics.csv",
        cohort="完整或显式部分 OOF test；扰动复用同一冻结 native readout",
        sample_n=",".join(str(int(n)) for n in pooled["n"]),
        folds=",".join(
            map(
                str,
                sorted(
                    metrics.loc[metrics["aggregation_level"].eq("fold_test"), "fold"]
                    .dropna()
                    .unique()
                    .astype(int)
                ),
            )
        ),
        errorbar="无；pooled OOF 点估计；差值的患者 bootstrap CI 另见 shortcut_bootstrap_ci.csv",
        decision_points=",".join(pooled["decision_point"].unique()),
        notes="同时保存 AUROC/AUPRC 及逐患者概率变化；图中仅展示 AUROC",
    )


def _control_figure(
    metrics: pd.DataFrame,
    path: Path,
    manifest: list[dict[str, Any]],
    method_input_contract: Mapping[str, Any],
) -> None:
    pooled = metrics.loc[metrics["aggregation_level"].eq("pooled_oof_test")].copy()
    if pooled.empty:
        return
    controls = [
        name
        for name in (
            "C0_radiomics_only",
            "C1_m2_image_plus_radiomics",
            "C2_m2_image_only",
        )
        if name in set(pooled["control_name"])
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(DECISION_ORDER))
    width = 0.8 / max(len(controls), 1)
    for index, name in enumerate(controls):
        values, ns = [], []
        for point in DECISION_ORDER:
            row = pooled.loc[
                pooled["control_name"].eq(name) & pooled["decision_point"].eq(point)
            ]
            values.append(float(row["auroc"].iloc[0]) if not row.empty else math.nan)
            ns.append(int(row["n"].iloc[0]) if not row.empty else 0)
        positions = x - 0.4 + width / 2 + index * width
        display_name = {
            "C0_radiomics_only": "C0 radiomics-only",
            "C1_m2_image_plus_radiomics": "C1 ROI辅助影像 + radiomics",
            "C2_m2_image_only": f"C2 {ROI_ASSISTED_SHORT_CN}",
        }.get(name, name)
        bars = ax.bar(positions, values, width=width, label=display_name)
        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"n={n}",
                ha="center",
                fontsize=7,
                rotation=90,
            )
    title = "C0/C1/C2 在相同 radiomics-paired test subset 上的 AUROC"
    ax.set_title(title)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x, DECISION_ORDER)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="#999999", ls="--", lw=1)
    ax.legend(fontsize=8)
    _finish_figure(
        fig,
        path,
        manifest,
        title=title,
        source="control_metrics_aggregated.csv",
        cohort="radiomics 可用 paired OOF test subset",
        sample_n=",".join(str(int(value)) for value in pooled["n"].unique()),
        folds=",".join(
            map(
                str,
                sorted(
                    metrics.loc[metrics["aggregation_level"].eq("fold_test"), "fold"]
                    .dropna()
                    .unique()
                    .astype(int)
                ),
            )
        ),
        errorbar="无；pooled OOF 点估计",
        decision_points=",".join(DECISION_ORDER),
        notes=(
            "C0/C1 推理依赖 radiomics；仅 C2 是主要 "
            f"{method_input_contract.get('method_label_cn', ROI_ASSISTED_FULL_CN)} 方法"
        ),
    )


def generate_figures(
    figure_dir: Path,
    classification_folds: pd.DataFrame,
    classification_oof: pd.DataFrame,
    transitions: pd.DataFrame,
    radiomics_predictions: pd.DataFrame,
    radiomics_metrics: pd.DataFrame,
    shortcut_metrics: pd.DataFrame,
    control_metrics: pd.DataFrame,
    issues: list[dict[str, str]],
    seed: int,
    method_input_contract: Mapping[str, Any],
) -> pd.DataFrame:
    _setup_matplotlib()
    manifest: list[dict[str, Any]] = []
    _classification_figure(
        classification_folds,
        classification_oof,
        "auroc",
        figure_dir / "01_image_only_auroc.png",
        manifest,
        method_input_contract,
    )
    _classification_figure(
        classification_folds,
        classification_oof,
        "auprc",
        figure_dir / "02_image_only_auprc.png",
        manifest,
        method_input_contract,
    )
    if transitions.empty:
        _record_issue(
            issues, "缺失图表_missing_figure", "缺 transition 数据，未生成图 03/04"
        )
    else:
        _transition_error_figure(
            transitions, figure_dir / "03_learned_vs_copy_error.png", manifest, seed
        )
        _gain_figure(
            transitions, figure_dir / "04_transition_gain_distribution.png", manifest
        )
    if radiomics_predictions.empty or radiomics_metrics.empty:
        _record_issue(
            issues,
            "缺失图表_missing_figure",
            "缺 M2 radiomics-head 数据，未生成图 05/06",
        )
    else:
        _radiomics_scatter_figure(
            radiomics_predictions,
            figure_dir / "05_radiomics_true_vs_predicted.png",
            manifest,
        )
        _radiomics_spearman_figure(
            radiomics_metrics,
            figure_dir / "06_radiomics_spearman.png",
            manifest,
        )
    if shortcut_metrics.empty:
        _record_issue(
            issues, "缺失图表_missing_figure", "缺 shortcut 数据，未生成图 07/08"
        )
    else:
        for perturbation, filename in (
            ("repeated_t0", "07_repeated_t0_auroc.png"),
            ("temporal_shuffle_t1_t2", "08_temporal_shuffle_auroc.png"),
        ):
            if perturbation not in set(shortcut_metrics["perturbation"]):
                _record_issue(
                    issues,
                    "缺失图表_missing_figure",
                    f"缺 {perturbation} 数据，未生成 {filename}",
                )
            else:
                _shortcut_figure(
                    shortcut_metrics,
                    perturbation,
                    figure_dir / filename,
                    manifest,
                )
    if not control_metrics.empty:
        _control_figure(
            control_metrics,
            figure_dir / "09_c0_c1_c2_auroc.png",
            manifest,
            method_input_contract,
        )
    return pd.DataFrame(manifest)


def aggregate_results(config: AggregationConfig) -> dict[str, Any]:
    """执行严格聚合，并以新目录一次性发布 CSV/JSON/PNG。"""

    config.validate()
    metric_target = EXPERIMENT_ROOT / "metrics" / "final" / config.output_tag
    figure_target = EXPERIMENT_ROOT / "figures" / "final" / config.output_tag
    if metric_target.exists() or figure_target.exists():
        raise FileExistsError(
            "聚合输出已存在，默认拒绝覆盖；请使用新的 --output-tag。"
            f" metrics={metric_target}, figures={figure_target}"
        )
    metric_parent = metric_target.parent
    figure_parent = figure_target.parent
    metric_parent.mkdir(parents=True, exist_ok=True)
    figure_parent.mkdir(parents=True, exist_ok=True)
    metric_stage = Path(
        tempfile.mkdtemp(prefix=f".{config.output_tag}.", dir=metric_parent)
    )
    figure_stage = Path(
        tempfile.mkdtemp(prefix=f".{config.output_tag}.", dir=figure_parent)
    )
    issues: list[dict[str, str]] = []
    consumed: list[Path] = []
    try:
        sources = discover_evaluations(config, issues)
        consumed.extend(source.summary_path for source in sources)
        method_input_contract = _validate_method_input_contract(sources, config, issues)
        native = load_native_predictions(sources, config, issues, consumed)
        class_fold, class_oof, class_summary, class_bootstrap = classification_tables(
            native, config.bootstrap_replicates, config.seed
        )
        class_paired_difference = paired_classification_bootstrap_differences(
            native, config.bootstrap_replicates, config.seed
        )
        perturbations = load_perturbations(sources, native, config, issues, consumed)
        shortcut_metric, shortcut_change, shortcut_bootstrap = shortcut_tables(
            perturbations, config.bootstrap_replicates, config.seed
        )
        transition_predictions = load_transition_predictions(
            sources, native, config, issues, consumed
        )
        transition_metrics = transition_table(transition_predictions)
        transition_patient_metrics = (
            _patient_level_transition(transition_predictions)
            if not transition_predictions.empty
            else pd.DataFrame()
        )
        radiomics_predictions = load_radiomics_predictions(
            sources, native, config, issues, consumed
        )
        m2_sources = [source for source in sources if source.mode == "m2"]
        baselines, transform_specs, raw_mapping = (
            _radiomics_train_mean_baselines(config, m2_sources, issues, consumed)
            if not radiomics_predictions.empty
            else ({}, {}, {})
        )
        (
            radiomics_predictions,
            radiomics_metrics,
            radiomics_distribution,
        ) = radiomics_tables(
            radiomics_predictions, baselines, transform_specs, raw_mapping
        )
        controls, probes = _discover_controls(sources, native, config, issues, consumed)
        control_metrics = control_tables(controls, method_input_contract)
        probe_metrics = probe_table(probes)

        tables: dict[str, pd.DataFrame] = {
            "oof_test_predictions.csv": native,
            "classification_fold_metrics.csv": class_fold,
            "classification_oof_metrics.csv": class_oof,
            "classification_fold_mean_std.csv": class_summary,
            "classification_patient_bootstrap_ci.csv": class_bootstrap,
            "classification_paired_model_difference_bootstrap_ci.csv": (
                class_paired_difference
            ),
            "shortcut_predictions_aggregated.csv": perturbations,
            "shortcut_performance_metrics.csv": shortcut_metric,
            "shortcut_probability_change_summary.csv": shortcut_change,
            "shortcut_bootstrap_ci.csv": shortcut_bootstrap,
            "transition_predictions_aggregated.csv": transition_predictions,
            "transition_metrics_aggregated.csv": transition_metrics,
            "transition_patient_metrics.csv": transition_patient_metrics,
            "radiomics_head_predictions_aggregated.csv": radiomics_predictions,
            "radiomics_head_metrics.csv": radiomics_metrics,
            "radiomics_head_distribution_summary.csv": radiomics_distribution,
            "control_predictions_aggregated.csv": controls,
            "control_metrics_aggregated.csv": control_metrics,
            "posthoc_probe_predictions_aggregated.csv": probes,
            "posthoc_probe_metrics_aggregated.csv": probe_metrics,
        }
        written_tables: list[dict[str, Any]] = []
        for filename, frame in tables.items():
            if frame.empty:
                _record_issue(
                    issues,
                    "空输出_empty_output",
                    f"没有真实输入，未写 {filename}",
                )
                continue
            _write_csv(metric_stage / filename, frame)
            written_tables.append(
                {"file": filename, "rows": len(frame), "columns": len(frame.columns)}
            )

        figure_manifest = generate_figures(
            figure_stage,
            class_fold,
            class_oof,
            transition_predictions,
            radiomics_predictions,
            radiomics_metrics,
            shortcut_metric,
            control_metrics,
            issues,
            config.seed,
            method_input_contract,
        )
        if not figure_manifest.empty:
            _write_csv(metric_stage / "figure_manifest.csv", figure_manifest)
            written_tables.append(
                {
                    "file": "figure_manifest.csv",
                    "rows": len(figure_manifest),
                    "columns": len(figure_manifest.columns),
                }
            )
        issue_frame = pd.DataFrame(
            issues,
            columns=["类别_category", "说明_message", "路径_path"],
        )
        _write_csv(metric_stage / "input_issues.csv", issue_frame)
        source_rows = []
        for path in sorted(set(path.resolve() for path in consumed)):
            if not path.is_file():
                raise AggregationInputError(f"聚合期间来源文件消失: {path}")
            source_rows.append(
                {
                    "source_path": str(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        _write_csv(metric_stage / "source_file_manifest.csv", pd.DataFrame(source_rows))

        if not config.allow_partial and len(figure_manifest) < 9:
            raise AggregationInputError(
                f"正式聚合只生成 {len(figure_manifest)} 张图，少于严格要求的 9 张；"
                "其中第 9 张必须为 C0/C1/C2 paired-subset 对比；"
                "缺失原因见 input_issues.csv"
            )
        required_issue_prefixes = ("缺失输入_", "缺失图表_", "缺失基线_")
        has_required_issue = any(
            str(item["类别_category"]).startswith(required_issue_prefixes)
            for item in issues
        )
        status = (
            "部分聚合_partial"
            if config.allow_partial or has_required_issue
            else "正式聚合完成_complete"
        )
        summary = {
            "schema_version": 1,
            "status": status,
            "generated_at_utc": _now_utc(),
            "aggregation_implementation_sha256": _sha256(Path(__file__)),
            "output_tag": config.output_tag,
            "allow_partial": config.allow_partial,
            "run_names": dict(config.run_names),
            "method_input_contract": method_input_contract,
            "folds_by_model": {
                mode: sorted(source.fold for source in sources if source.mode == mode)
                for mode in MODEL_ORDER
            },
            "bootstrap": {
                "replicates": config.bootstrap_replicates,
                "seed": config.seed,
                "resampling_unit": "patient",
                "interval": "95% percentile",
            },
            "counts": {
                "native_prediction_rows": len(native),
                "native_oof_patients_by_model": {
                    mode: int(
                        native.loc[
                            native["aggregation_model"].eq(mode), "patient_id"
                        ].nunique()
                    )
                    for mode in sorted(native["aggregation_model"].unique())
                },
                "shortcut_prediction_rows": len(perturbations),
                "transition_prediction_rows": len(transition_predictions),
                "native_m2_radiomics_prediction_rows": len(radiomics_predictions),
                "control_prediction_rows": len(controls),
                "posthoc_probe_prediction_rows": len(probes),
                "figures": len(figure_manifest),
                "issues": len(issues),
            },
            "output_tables": written_tables,
            "metric_dir": str(metric_target),
            "figure_dir": str(figure_target),
            "guards": {
                "missing_values_fabricated": False,
                "test_statistics_used_for_model_selection": False,
                "classification_threshold_source": "原 fold validation threshold；聚合阶段不重选",
                "architecture_input_contract_verified": bool(
                    method_input_contract.get("verification_complete", False)
                ),
                "fold_manifest_patient_label_hash_verified": bool(
                    not config.allow_partial and config.fold_manifest.is_file()
                ),
                "registered_transition_shortcut_radiomics_cells_verified": bool(
                    not config.allow_partial
                    and not any(
                        str(item["类别_category"]).startswith("缺失输入_")
                        for item in issues
                    )
                ),
                "radiomics_transform_and_raw_target_provenance_verified": bool(
                    not config.allow_partial
                    and bool(raw_mapping)
                    and bool(transform_specs)
                    and not any(
                        str(item["类别_category"]).startswith("缺失基线_")
                        for item in issues
                    )
                ),
                "radiomics_mean_baseline_source": (
                    "每个 fold×transition×feature train-only；"
                    "缺失时不以 test 均值替代"
                ),
                "native_m2_head_distinguished_from_posthoc_probe": True,
                "strict_requires_fivefold_c0_c1_c2_and_probe": True,
                "overwrite_refused_by_default": True,
            },
        }
        _write_json(metric_stage / "aggregation_summary.json", summary)
        figure_stage.rename(figure_target)
        try:
            metric_stage.rename(metric_target)
        except Exception:
            # 两个 final 目录跨父目录无法做单次原子 rename；若第二步失败，
            # 立即把第一步退回 staging，避免留下可被误认作完整结果的半发布。
            if figure_target.exists() and not figure_stage.exists():
                figure_target.rename(figure_stage)
            raise
        return summary
    except Exception:
        shutil.rmtree(metric_stage, ignore_errors=True)
        shutil.rmtree(figure_stage, ignore_errors=True)
        raise
