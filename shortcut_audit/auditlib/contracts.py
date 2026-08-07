"""Prediction-level CSV 的固定 schema 与对齐校验。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
PREDICTION_COLUMNS = (
    "patient_id",
    "fold",
    "decision_point",
    "audit_condition",
    "y_true",
    "predicted_probability",
    "predicted_label",
    "threshold",
    "checkpoint",
    "donor_patient_id",
    "repetition_id",
    "matching_distance",
)


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    require_donor: bool = False,
    expected_folds: Iterable[int] = range(5),
) -> pd.DataFrame:
    """返回规范化副本；任何 schema、范围或标签错误都会显式失败。"""

    missing = sorted(set(PREDICTION_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"prediction CSV 缺少列：{missing}")
    output = frame.loc[:, PREDICTION_COLUMNS].copy()
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    output["decision_point"] = output["decision_point"].astype(str)
    output["audit_condition"] = output["audit_condition"].astype(str).str.strip()
    output["checkpoint"] = output["checkpoint"].astype(str).str.strip()
    if (output["patient_id"] == "").any() or (output["audit_condition"] == "").any():
        raise ValueError("patient_id 与 audit_condition 不得为空")
    if (output["checkpoint"] == "").any():
        raise ValueError("每行必须记录 checkpoint provenance")
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    if not output["fold"].isin(tuple(expected_folds)).all():
        raise ValueError("prediction CSV 含未知 fold")
    if not output["decision_point"].isin(DECISION_POINTS).all():
        raise ValueError("prediction CSV 含未知 decision_point")
    output["y_true"] = pd.to_numeric(output["y_true"], errors="raise").astype(int)
    output["predicted_label"] = pd.to_numeric(output["predicted_label"], errors="raise").astype(int)
    if not output["y_true"].isin((0, 1)).all() or not output["predicted_label"].isin((0, 1)).all():
        raise ValueError("y_true 与 predicted_label 必须为 0/1")
    for column in ("predicted_probability", "threshold"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
        if not np.isfinite(output[column]).all() or not output[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} 必须为 [0,1] 内有限值")
    expected_label = (output["predicted_probability"] >= output["threshold"]).astype(int)
    if not expected_label.equals(output["predicted_label"]):
        raise ValueError("predicted_label 与 probability/threshold 不一致")

    if require_donor:
        donor = output["donor_patient_id"].astype("string")
        if donor.isna().any() or donor.str.strip().eq("").any():
            raise ValueError("donor audit 每行必须记录 donor_patient_id")
        if donor.astype(str).eq(output["patient_id"]).any():
            raise ValueError("donor_patient_id 不得等于 recipient patient_id")
        if output["repetition_id"].isna().any() or output["matching_distance"].isna().any():
            raise ValueError("donor audit 每行必须记录 repetition_id 与 matching_distance")

    duplicate_key = [
        "patient_id",
        "fold",
        "decision_point",
        "audit_condition",
        "donor_patient_id",
        "repetition_id",
    ]
    duplicated = output.duplicated(duplicate_key, keep=False)
    if duplicated.any():
        raise ValueError(f"prediction 主键重复，共 {int(duplicated.sum())} 行")
    return output


def validate_label_alignment(frames: Iterable[pd.DataFrame]) -> None:
    """验证所有 audit condition 的 patient/fold/decision label 一致。"""

    normalized = [validate_prediction_frame(frame) for frame in frames]
    if not normalized:
        raise ValueError("至少需要一个 prediction frame")
    combined = pd.concat(normalized, ignore_index=True)
    label_count = combined.groupby(["patient_id", "fold", "decision_point"])["y_true"].nunique()
    if not label_count.eq(1).all():
        bad = label_count[~label_count.eq(1)].head(5).to_dict()
        raise ValueError(f"audit condition 间 label 不一致：{bad}")


def write_prediction_csv(frame: pd.DataFrame, path: str | Path, *, require_donor: bool = False) -> Path:
    """校验后原子写入 CSV，避免中断时留下半文件。"""

    output = validate_prediction_frame(frame, require_donor=require_donor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        output.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
