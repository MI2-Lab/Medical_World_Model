#!/usr/bin/env python3
"""Batch the header-only DICOM audit for Stage A singular-sform visits.

The patient/visit table is a private, git-ignored sidecar.  Public outputs are
aggregate only.  This script never reads DICOM PixelData and therefore never
marks the input contract model-ready: pixel reconstruction and pixel-order
verification remain separate required work.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[3]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from observable_crop.dicom_geometry import (  # noqa: E402
    DicomAuditTolerances,
    audit_dicom_geometry,
)
from observable_crop.nifti import read_nifti_geometry  # noqa: E402


EXPECTED_INVALID_VISITS = 72
EXPECTED_DECISIONS = {
    "TRUST_DCE_QFORM": 35,
    "MASK_SFORM_GEOMETRY_CANDIDATE": 37,
}
VISITS = ("T0", "T1", "T2", "T3")
MINIMUM_PUBLIC_CELL = 5
PRIVATE_DETAIL_RELATIVE = Path("metrics/patient_level_dicom_geometry.csv")
PUBLIC_SUMMARY_RELATIVE = Path("metrics/dicom_geometry_repair_summary.csv")
PUBLIC_GATE_RELATIVE = Path("metrics/dicom_geometry_repair_gate.json")
PUBLIC_REPORT_RELATIVE = Path("reports/dicom_geometry_repair_audit.md")


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_finite_or_none(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _finite_or_none(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    converted = normalized.map(mapping)
    if converted.isna().any():
        raise ValueError(f"{name} contains a non-boolean value")
    return converted.astype(bool)


def _load_invalid_visits(path: Path, expected_count: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "patient_id",
        "visit",
        "shape_x",
        "shape_y",
        "shape_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "dce_sform_valid",
        "dce_sform_failure_reason",
        "affine_decision",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"private geometry table is missing {len(missing)} columns")
    frame = frame.copy()
    frame["dce_sform_valid"] = _as_bool(
        frame["dce_sform_valid"], "dce_sform_valid"
    )
    invalid = frame.loc[~frame["dce_sform_valid"]].copy()
    invalid["patient_id"] = invalid["patient_id"].astype(str)
    invalid["visit"] = invalid["visit"].astype(str)
    if len(invalid) != expected_count:
        raise ValueError(
            f"singular-sform visit count mismatch: expected {expected_count}, "
            f"observed {len(invalid)}"
        )
    if invalid[["patient_id", "visit"]].duplicated().any():
        raise ValueError("private geometry table has duplicate patient/visit rows")
    if not set(invalid["visit"]).issubset(VISITS):
        raise ValueError("private geometry table has an unexpected visit label")
    if set(invalid["dce_sform_failure_reason"].dropna()) != {"SFORM_SINGULAR"}:
        raise ValueError("invalid DCE rows are not exclusively SFORM_SINGULAR")
    expected_stage_decisions = {
        "TRUST_DCE_QFORM_SFORM_SINGULAR",
        "MASK_SFORM_GEOMETRY_CANDIDATE_REBUILD_DICOM_PIXELS",
    }
    if set(invalid["affine_decision"].dropna()) != expected_stage_decisions:
        raise ValueError("singular DCE rows have an unexpected Stage A decision set")
    numeric = [
        "shape_x",
        "shape_y",
        "shape_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
    ]
    for column in numeric:
        invalid[column] = pd.to_numeric(invalid[column], errors="raise")
    return invalid.sort_values(["patient_id", "visit"]).reset_index(drop=True)


def _job_from_row(row: pd.Series, preprocessed_root: Path) -> dict[str, Any]:
    return {
        "patient_id": str(row["patient_id"]),
        "visit": str(row["visit"]),
        "shape_xyz": (
            int(row["shape_x"]),
            int(row["shape_y"]),
            int(row["shape_z"]),
        ),
        "spacing_xyz_mm": (
            float(row["spacing_x_mm"]),
            float(row["spacing_y_mm"]),
            float(row["spacing_z_mm"]),
        ),
        "stage_a_affine_decision": str(row["affine_decision"]),
        "preprocessed_root": str(preprocessed_root),
    }


def _audit_one(job: dict[str, Any]) -> dict[str, Any]:
    """Worker returning identifiers only for the private sidecar."""

    patient_id = job["patient_id"]
    visit_label = job["visit"]
    manifest_path = Path(job["preprocessed_root"]) / patient_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visits = {
        str(item["visit"]): item
        for item in manifest.get("visits", [])
        if item.get("visit") is not None
    }
    if visit_label not in visits:
        raise ValueError("manifest is missing an expected visit")
    visit = visits[visit_label]
    dce = read_nifti_geometry(visit["dce_nifti"])
    mask = read_nifti_geometry(visit["ftv_mask_nifti"])
    if tuple(int(value) for value in dce.shape[:3]) != tuple(job["shape_xyz"]):
        raise ValueError("DCE NIfTI shape differs from private Stage A geometry")
    if len(dce.shape) != 4:
        raise ValueError("DCE NIfTI is not four-dimensional")
    if dce.sform_failure_reason != "SFORM_SINGULAR":
        raise ValueError("selected DCE NIfTI no longer has a singular sform")

    audit = audit_dicom_geometry(
        visit["raw_dce_series"],
        expected_shape_xyz_t=tuple(int(value) for value in dce.shape),
        expected_spacing_xyz_mm=tuple(float(value) for value in dce.spacing_xyz_mm),
        mask_affine_ras=mask.sform,
        mask_shape_xyz=tuple(int(value) for value in mask.shape[:3]),
        dce_sform_ras=dce.sform,
        dce_qform_ras=dce.qform,
    )
    tolerances = DicomAuditTolerances()
    expected_module_decision = (
        "TRUST_DCE_QFORM"
        if job["stage_a_affine_decision"]
        == "TRUST_DCE_QFORM_SFORM_SINGULAR"
        else "MASK_SFORM_GEOMETRY_CANDIDATE"
    )
    return {
        "patient_id": patient_id,
        "visit": visit_label,
        "stage_a_affine_decision": job["stage_a_affine_decision"],
        "dicom_decision": audit.decision,
        "decision_matches_stage_a": audit.decision == expected_module_decision,
        "file_count": audit.file_count,
        "readable_header_count": audit.readable_header_count,
        "required_header_complete_count": audit.required_header_complete_count,
        "expected_file_count": audit.expected_file_count,
        "rows": audit.rows,
        "columns": audit.columns,
        "slice_count": audit.slice_count,
        "timepoint_count": audit.temporal_position.group_count,
        "unique_sop_instance_uid_count": audit.unique_sop_instance_uid_count,
        "unique_series_instance_uid_count": audit.unique_series_instance_uid_count,
        "all_headers_readable": audit.file_count == audit.readable_header_count,
        "required_headers_complete": (
            audit.file_count == audit.required_header_complete_count
        ),
        "file_count_matches_expected": audit.file_count == audit.expected_file_count,
        "rows_columns_match_expected": audit.dicom_shape_xyz
        == tuple(job["shape_xyz"]),
        "iop_consistent_orthonormal": bool(
            audit.iop_orthonormal
            and audit.iop_max_abs_delta is not None
            and audit.iop_max_abs_delta <= tolerances.iop_atol
        ),
        "pixel_spacing_consistent": bool(
            audit.pixel_spacing_max_abs_delta_mm is not None
            and audit.pixel_spacing_max_abs_delta_mm
            <= tolerances.pixel_spacing_atol_mm
        ),
        "slice_grid_complete_regular": bool(
            audit.slice_count == dce.shape[2]
            and audit.slice_spacing_max_deviation_mm is not None
            and audit.slice_spacing_max_deviation_mm
            <= tolerances.slice_spacing_atol_mm
            and audit.in_plane_position_max_deviation_mm is not None
            and audit.in_plane_position_max_deviation_mm
            <= tolerances.in_plane_position_atol_mm
        ),
        "temporal_position_complete": audit.temporal_position.complete,
        "acquisition_time_complete": audit.acquisition_time.complete,
        "temporal_groupings_agree": audit.temporal_groupings_agree,
        "series_geometry_valid": audit.series_geometry_valid,
        "mask_geometry_consistent": audit.mask_geometry_consistent,
        "audit_pass": audit.audit_pass,
        "header_only_safe": audit.header_only_safe,
        "geometry_auto_repairable": audit.geometry_auto_repairable,
        "pixel_data_read": audit.pixel_data_read,
        "pixel_rebuild_required": True,
        "pixel_rebuild_executed": False,
        "pixel_order_verified": False,
        "model_ready": False,
        "recommended_action": audit.recommended_action,
        "iop_max_abs_delta": audit.iop_max_abs_delta,
        "pixel_spacing_max_abs_delta_mm": audit.pixel_spacing_max_abs_delta_mm,
        "slice_spacing_mm": audit.slice_spacing_mm,
        "slice_spacing_max_deviation_mm": (
            audit.slice_spacing_max_deviation_mm
        ),
        "in_plane_position_max_deviation_mm": (
            audit.in_plane_position_max_deviation_mm
        ),
        "mask_center_corner_hausdorff_mm": (
            audit.mask_center_corner_hausdorff_mm
        ),
        "mask_footprint_corner_hausdorff_mm": (
            audit.mask_footprint_corner_hausdorff_mm
        ),
        "dce_qform_mask_index_corner_max_mm": (
            audit.dce_qform_mask_index_corner_max_mm
        ),
        "dce_qform_dicom_corner_hausdorff_mm": (
            audit.dce_qform_dicom_corner_hausdorff_mm
        ),
        "quarantine_reason_count": len(audit.quarantine_reason_codes),
        "warning_count": len(audit.warning_codes),
    }


def _quantile(values: Iterable[Any], probability: float) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    return float(np.quantile(numeric, probability)) if len(numeric) else math.nan


def _summary_row(scope: str, value: str, frame: pd.DataFrame) -> dict[str, Any]:
    count = len(frame)
    if count < MINIMUM_PUBLIC_CELL:
        raise ValueError("a requested public aggregate is below the minimum cell size")

    def count_true(column: str) -> int:
        return int(_as_bool(frame[column], column).sum())

    mask_error = pd.to_numeric(
        frame["mask_center_corner_hausdorff_mm"], errors="coerce"
    )
    qform_error = pd.to_numeric(
        frame["dce_qform_dicom_corner_hausdorff_mm"], errors="coerce"
    )
    return {
        "aggregation": scope,
        "group": value,
        "series_count": count,
        "patient_count": int(frame["patient_id"].nunique()),
        "dicom_file_count": int(frame["file_count"].sum()),
        "audit_pass_count": count_true("audit_pass"),
        "audit_pass_rate": count_true("audit_pass") / count,
        "required_headers_complete_rate": (
            count_true("required_headers_complete") / count
        ),
        "rows_columns_match_rate": count_true("rows_columns_match_expected")
        / count,
        "iop_consistent_orthonormal_rate": (
            count_true("iop_consistent_orthonormal") / count
        ),
        "pixel_spacing_consistent_rate": (
            count_true("pixel_spacing_consistent") / count
        ),
        "slice_grid_complete_regular_rate": (
            count_true("slice_grid_complete_regular") / count
        ),
        "temporal_position_complete_rate": (
            count_true("temporal_position_complete") / count
        ),
        "acquisition_time_complete_rate": (
            count_true("acquisition_time_complete") / count
        ),
        "mask_geometry_consistent_rate": (
            count_true("mask_geometry_consistent") / count
        ),
        "trust_dce_qform_count": int(
            (frame["dicom_decision"] == "TRUST_DCE_QFORM").sum()
        ),
        "mask_sform_geometry_candidate_count": int(
            (
                frame["dicom_decision"]
                == "MASK_SFORM_GEOMETRY_CANDIDATE"
            ).sum()
        ),
        "quarantine_count": int((frame["dicom_decision"] == "QUARANTINE").sum()),
        "pixel_rebuild_executed_count": count_true("pixel_rebuild_executed"),
        "pixel_order_verified_count": count_true("pixel_order_verified"),
        "model_ready_count": count_true("model_ready"),
        "mask_corner_hausdorff_median_mm": float(mask_error.median()),
        "mask_corner_hausdorff_max_mm": float(mask_error.max()),
        "qform_dicom_corner_hausdorff_median_mm": float(qform_error.median()),
        "qform_dicom_corner_hausdorff_q95_mm": _quantile(qform_error, 0.95),
        "qform_dicom_corner_hausdorff_max_mm": float(qform_error.max()),
    }


def _build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_row("overall", "ALL", detail)]
    for visit in VISITS:
        subset = detail.loc[detail["visit"] == visit]
        rows.append(_summary_row("visit", visit, subset))
    for decision in sorted(EXPECTED_DECISIONS):
        subset = detail.loc[detail["dicom_decision"] == decision]
        rows.append(_summary_row("decision", decision, subset))
    return pd.DataFrame(rows)


def _build_gate(
    detail: pd.DataFrame,
    input_sha256: str,
    expected_count: int,
) -> dict[str, Any]:
    decision_counts = {
        str(key): int(value)
        for key, value in detail["dicom_decision"].value_counts().sort_index().items()
    }

    def all_true(column: str) -> bool:
        return bool(_as_bool(detail[column], column).all())

    def any_true(column: str) -> bool:
        return bool(_as_bool(detail[column], column).any())

    header_checks = {
        "expected_singular_visit_count": len(detail) == expected_count,
        "unique_patient_visit_rows": not detail[
            ["patient_id", "visit"]
        ].duplicated().any(),
        "all_headers_readable": all_true("all_headers_readable"),
        "required_headers_complete": all_true("required_headers_complete"),
        "file_counts_match_xyz_t": all_true("file_count_matches_expected"),
        "rows_columns_match_nifti": all_true("rows_columns_match_expected"),
        "iop_consistent_and_orthonormal": all_true(
            "iop_consistent_orthonormal"
        ),
        "pixel_spacing_consistent": all_true("pixel_spacing_consistent"),
        "ipp_slice_grid_complete_and_regular": all_true(
            "slice_grid_complete_regular"
        ),
        "temporal_position_groups_complete": all_true(
            "temporal_position_complete"
        ),
        "acquisition_time_groups_complete": all_true(
            "acquisition_time_complete"
        ),
        "temporal_groupings_agree": all_true("temporal_groupings_agree"),
        "mask_sform_matches_dicom_volume": all_true(
            "mask_geometry_consistent"
        ),
        "no_quarantined_series": not (
            detail["dicom_decision"] == "QUARANTINE"
        ).any(),
        "decisions_reproduce_stage_a": all_true("decision_matches_stage_a"),
        "expected_35_37_decision_partition": decision_counts
        == EXPECTED_DECISIONS,
        "pixel_data_was_not_read": not any_true("pixel_data_read"),
    }
    header_audit_pass = bool(all(header_checks.values()))
    model_ready_checks = {
        "header_geometry_audit_pass": header_audit_pass,
        "pixel_rebuild_executed_for_all_72": all_true(
            "pixel_rebuild_executed"
        ),
        "pixel_order_verified_for_all_72": all_true("pixel_order_verified"),
        "model_ready_for_all_72": all_true("model_ready"),
    }
    model_ready = bool(all(model_ready_checks.values()))
    candidate = detail.loc[
        detail["dicom_decision"] == "MASK_SFORM_GEOMETRY_CANDIDATE"
    ]
    trusted = detail.loc[detail["dicom_decision"] == "TRUST_DCE_QFORM"]
    return {
        "schema_version": 1,
        "stage": "A_DICOM_HEADER_GEOMETRY_ONLY",
        "outcome_free": True,
        "patient_level_geometry_sha256": input_sha256,
        "expected_singular_visit_count": expected_count,
        "audited_singular_visit_count": int(len(detail)),
        "audited_patient_count": int(detail["patient_id"].nunique()),
        "dicom_header_count": int(detail["file_count"].sum()),
        "decision_counts": decision_counts,
        "thresholds": {
            "affine_corner_atol_mm": DicomAuditTolerances().affine_corner_atol_mm,
            "slice_spacing_atol_mm": (
                DicomAuditTolerances().slice_spacing_atol_mm
            ),
            "in_plane_position_atol_mm": (
                DicomAuditTolerances().in_plane_position_atol_mm
            ),
            "minimum_public_cell": MINIMUM_PUBLIC_CELL,
        },
        "header_checks": header_checks,
        "header_geometry_audit_pass": header_audit_pass,
        "mask_dicom_corner_hausdorff_max_mm": float(
            detail["mask_center_corner_hausdorff_mm"].max()
        ),
        "trusted_qform_dicom_corner_hausdorff_max_mm": float(
            trusted["dce_qform_dicom_corner_hausdorff_mm"].max()
        ),
        "mask_candidate_qform_dicom_corner_hausdorff": {
            "min_mm": float(
                candidate["dce_qform_dicom_corner_hausdorff_mm"].min()
            ),
            "median_mm": float(
                candidate["dce_qform_dicom_corner_hausdorff_mm"].median()
            ),
            "q95_mm": _quantile(
                candidate["dce_qform_dicom_corner_hausdorff_mm"], 0.95
            ),
            "max_mm": float(
                candidate["dce_qform_dicom_corner_hausdorff_mm"].max()
            ),
        },
        "pixel_data_read": False,
        "pixel_rebuild_required": True,
        "pixel_rebuild_executed": False,
        "pixel_order_verified": False,
        "model_ready_checks": model_ready_checks,
        "failed_model_ready_checks": [
            name for name, passed in model_ready_checks.items() if not passed
        ],
        "model_ready": model_ready,
        "stage_b_authorized": False,
        "decision": (
            "HEADER_GEOMETRY_PASS_PIXEL_REBUILD_PENDING"
            if header_audit_pass and not model_ready
            else "DICOM_GEOMETRY_AUDIT_FAILED"
        ),
        "required_next_action": (
            "rebuild_all_72_singular_visits_from_raw_DICOM_pixels_then_verify_"
            "voxel_order_and_affine_before_model_use"
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _build_report(detail: pd.DataFrame, gate: dict[str, Any]) -> str:
    counts = gate["decision_counts"]
    candidate = gate["mask_candidate_qform_dicom_corner_hausdorff"]
    trusted_count = counts.get("TRUST_DCE_QFORM", 0)
    candidate_count = counts.get("MASK_SFORM_GEOMETRY_CANDIDATE", 0)
    quarantine_count = counts.get("QUARANTINE", 0)
    visits = detail.groupby("visit", sort=True).size().to_dict()
    visit_text = "、".join(f"{key}={value}" for key, value in visits.items())
    return f"""# DICOM 几何修复审计

## 范围与方法

本审计从私有 `patient_level_geometry.csv` 中筛出全部
`SFORM_SINGULAR` DCE visit，共 {len(detail)} 个、
{detail['patient_id'].nunique()} 名患者（{visit_text}）。逐 series 仅读取 DICOM header：
`Rows/Columns`、`ImageOrientationPatient`、`PixelSpacing`、
`ImagePositionPatient`、`TemporalPositionIdentifier` 与 `AcquisitionTime`。
代码使用 `stop_before_pixels=True`，没有请求、解码或写出 PixelData。

空间 affine 由 DICOM LPS 坐标转换为 RAS+；IPP 沿 IOP 法向聚类、排序，
并以无序八角点 Hausdorff 距离比较 DICOM physical volume 与 mask sform。
中心角点与完整 voxel-footprint 角点均以
{gate['thresholds']['affine_corner_atol_mm']} mm 为通过阈值。

## Header 审计结果

- {len(detail)}/{len(detail)} 个 series 的文件数均等于 `X×Y×Z×T` 所隐含的
  `Z×T` cell 数；累计读取 {gate['dicom_header_count']} 个 header。
- {len(detail)}/{len(detail)} 的 Rows/Columns、IOP、PixelSpacing、IPP 层网格均完整一致；
  TPI 和 AcquisitionTime 两种分组均完整且一一对应。
- mask sform 与 raw DICOM physical volume 的最大中心角点 Hausdorff 误差为
  {_fmt(gate['mask_dicom_corner_hausdorff_max_mm'])} mm，低于 0.1 mm gate。
- 未出现 quarantine；header geometry gate 为
  **{'PASS' if gate['header_geometry_audit_pass'] else 'FAIL'}**。

## Repair decision

| 决策 | visit 数 | 物理几何解释 | 当前可否进入模型 |
|---|---:|---|---|
| `TRUST_DCE_QFORM` | {trusted_count} | qform、mask sform、DICOM 一致 | 否 |
| `MASK_SFORM_GEOMETRY_CANDIDATE` | {candidate_count} | mask/DICOM 一致；DCE 不可靠 | 否 |
| `QUARANTINE` | {quarantine_count} | header grid 或 affine gate 失败 | 否 |

37 个 mask-sform candidate 的 DCE qform–DICOM 角点误差为：
min={_fmt(candidate['min_mm'])} mm、median={_fmt(candidate['median_mm'])} mm、
Q95={_fmt(candidate['q95_mm'])} mm、max={_fmt(candidate['max_mm'])} mm。
这批数据不能用 DCE qform 或奇异 sform 作为物理真值。

35 个 `TRUST_DCE_QFORM` 仅表示 header-level physical geometry 可确认，
不表示 NIfTI 的 z/t pixel order 已经逐像素验收。为了与 Stage A 的 fail-closed
输入契约一致，72 个 singular-sform visit 均维持 `pixel_rebuild_required=true`。

## Model-ready gate

**Model-ready = false。** 本轮没有执行 DICOM pixel rebuild，也没有验证重建后
体素与 raw DICOM `(time, slice)` cell 的逐一对应；因此不能生成或使用
model-ready cache，Stage B 仍未授权。

解除 blocker 必须完成：从 raw DICOM PixelData 按时间组及 IPP 顺序重建全部
72 个 visit，应用像素缩放，写入经验证的 RAS affine，并验证每个 cell 恰好一次、
像素顺序正确、与 mask sform 的 physical corner 误差不超过 0.1 mm。

公开 CSV/JSON 仅含聚合计数、比例与误差；患者级明细保存在被 `.gitignore`
排除的本地 sidecar 中。
"""


def _assert_private_output_ignored(path: Path) -> None:
    relative = path.resolve().relative_to(REPO_ROOT.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(relative)],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise ValueError("private DICOM audit output is not covered by .gitignore")


def _validate_public_outputs(
    summary_path: Path,
    gate_path: Path,
    report_path: Path,
    private_identifiers: Iterable[str],
) -> None:
    summary = pd.read_csv(summary_path)
    forbidden_columns = {"patient_id", "visit_id", "subject_id"}
    if forbidden_columns & set(summary.columns):
        raise ValueError("public DICOM summary contains a direct identifier column")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("model_ready") is not False:
        raise ValueError("public DICOM gate must remain model_ready=false")
    if gate.get("pixel_rebuild_executed") is not False:
        raise ValueError("public DICOM gate incorrectly claims pixel rebuild")
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (summary_path, gate_path, report_path)
    )
    if any(
        identifier and identifier in public_text
        for identifier in private_identifiers
    ):
        raise ValueError("a private identifier leaked into a public DICOM artifact")
    if "/" + "data/" in public_text or "/" + "home/" in public_text:
        raise ValueError("an absolute host path leaked into a public DICOM artifact")


def _parse_args() -> argparse.Namespace:
    configured_root = os.environ.get("ISPY2_PREPROCESSED_ROOT")
    parser = argparse.ArgumentParser(
        description="Aggregate header-only DICOM geometry repair decisions"
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path(configured_root) if configured_root else None,
        required=configured_root is None,
        help=(
            "preprocessed patient-root directory; alternatively set "
            "ISPY2_PREPROCESSED_ROOT (never written to public output)"
        ),
    )
    parser.add_argument(
        "--patient-level-geometry",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/patient_level_geometry.csv",
        help="private Stage A patient-level geometry table",
    )
    parser.add_argument(
        "--expected-invalid-visits",
        type=int,
        default=EXPECTED_INVALID_VISITS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.expected_invalid_visits <= 0:
        raise ValueError("--expected-invalid-visits must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    preprocessed_root = args.preprocessed_root.expanduser().resolve(strict=True)
    private_input = args.patient_level_geometry.expanduser().resolve(strict=True)
    invalid = _load_invalid_visits(private_input, args.expected_invalid_visits)
    jobs = [
        _job_from_row(row, preprocessed_root)
        for _, row in invalid.iterrows()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(_audit_one, jobs))
    detail = pd.DataFrame(records).sort_values(
        ["patient_id", "visit"]
    ).reset_index(drop=True)
    if len(detail) != args.expected_invalid_visits:
        raise ValueError("DICOM worker result count differs from selected visit count")

    summary = _build_summary(detail)
    gate = _build_gate(
        detail,
        input_sha256=_sha256(private_input),
        expected_count=args.expected_invalid_visits,
    )
    if not gate["header_geometry_audit_pass"]:
        raise ValueError("DICOM header geometry gate failed; refusing public outputs")
    if gate["model_ready"]:
        raise ValueError("header-only audit must not set model_ready=true")
    report = _build_report(detail, gate)

    private_output = EXPERIMENT_ROOT / PRIVATE_DETAIL_RELATIVE
    summary_path = EXPERIMENT_ROOT / PUBLIC_SUMMARY_RELATIVE
    gate_path = EXPERIMENT_ROOT / PUBLIC_GATE_RELATIVE
    report_path = EXPERIMENT_ROOT / PUBLIC_REPORT_RELATIVE
    _assert_private_output_ignored(private_output)
    _atomic_csv(private_output, detail)
    _atomic_csv(summary_path, summary)
    _atomic_json(gate_path, gate)
    _atomic_text(report_path, report)
    _validate_public_outputs(
        summary_path,
        gate_path,
        report_path,
        private_identifiers=detail["patient_id"].astype(str).unique(),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "audited_singular_visits": len(detail),
                "decision_counts": gate["decision_counts"],
                "header_geometry_audit_pass": gate[
                    "header_geometry_audit_pass"
                ],
                "pixel_rebuild_executed": False,
                "pixel_order_verified": False,
                "model_ready": False,
                "public_outputs_written": 3,
                "private_ignored_sidecars_written": 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
