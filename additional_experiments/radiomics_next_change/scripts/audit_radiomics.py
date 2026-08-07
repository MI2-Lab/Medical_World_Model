#!/usr/bin/env python3
"""只读审计 I-SPY2 纵向 MRI measurement，并写入独立实验目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RAW_RADIOMICS = Path("/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx")
DEFAULT_RAW_CLINICAL = Path("/data/data/Breast_Cancer/I-SPY2/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx")
DEFAULT_PROCESSED_ROOT = Path("/data/data/Preprocessed/I-SPY2")
DEFAULT_FOLD_MANIFEST = DEFAULT_PROCESSED_ROOT / (
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
DEFAULT_CACHE_ROOT = DEFAULT_PROCESSED_ROOT / (
    "_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_"
    "autoroi_t0fallback_minfrac05_z32_y96_x96"
)

ID_PATTERN = re.compile(r"^(?:ISPY2-|ACRIN-6698-)(\d{6})$")
VISITS = ("T0", "T1", "T2", "T3")
TRANSITIONS = (("T0", "T1"), ("T1", "T2"), ("T2", "T3"))
FEATURES = {
    "ftv": {
        "columns": ("VOLUME_TUM_BLU_V10", "VOLUME_TUM_BLU_V20", "VOLUME_TUM_BLU_V30", "VOLUME_TUM_BLU_V40"),
        "unit": "cc（随附 FTV DICOM 文档；工作簿未单列单位）",
        "planned_change": "log(x+epsilon) 的相邻差；epsilon 仅由各 fold 训练患者拟合",
    },
    "sphericity": {
        "columns": ("SPHERICITY_T0", "SPHERICITY_T1", "SPHERICITY_T2", "SPHERICITY_T3"),
        "unit": "无量纲",
        "planned_change": "相邻绝对差",
    },
    "ld": {
        "columns": ("LD_T0", "LD_T1", "LD_T2", "LD_T3"),
        "unit": "源工作簿/字典未明示；不依赖单位假设",
        "planned_change": "log(x+epsilon) 的相邻差；epsilon 仅由各 fold 训练患者拟合",
    },
    "bpe": {
        "columns": ("BPE_5slice_mean_T0", "BPE_5slice_mean_T1", "BPE_5slice_mean_T2", "BPE_5slice_mean_T3"),
        "unit": "源工作簿/字典未明示；按原始数值处理",
        "planned_change": "相邻绝对差",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-radiomics", type=Path, default=DEFAULT_RAW_RADIOMICS)
    parser.add_argument("--raw-clinical", type=Path, default=DEFAULT_RAW_CLINICAL)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_audit",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_trial_id(patient_id: str) -> str:
    match = ID_PATTERN.fullmatch(str(patient_id).strip())
    if match is None:
        raise ValueError(f"患者 ID 不满足显式 I-SPY2/ACRIN 规则: {patient_id!r}")
    return match.group(1)


def treatment_family(arm: str) -> str:
    value = str(arm).lower()
    if "pembrolizumab" in value:
        return "io"
    if "carboplatin" in value or "abt 888" in value:
        return "platinum_parp"
    if any(token in value for token in ("trastuzumab", "pertuzumab", "t-dm1", "neratinib")):
        return "her2_targeted"
    if value.strip() == "paclitaxel":
        return "taxane"
    return "targeted_other"


def feature_lookup() -> dict[str, tuple[str, str, str]]:
    lookup: dict[str, tuple[str, str, str]] = {}
    for feature, spec in FEATURES.items():
        for visit, column in zip(VISITS, spec["columns"]):
            lookup[column] = (feature, visit, str(spec["unit"]))
    return lookup


def workbook_schema(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook = pd.ExcelFile(path)
    sheets: list[dict[str, Any]] = []
    schema: list[dict[str, Any]] = []
    selected: pd.DataFrame | None = None
    lookup = feature_lookup()
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        sheets.append(
            {
                "file": str(path),
                "sheet": sheet,
                "rows": len(frame),
                "columns": len(frame.columns),
            }
        )
        if sheet == "datawith4visits":
            selected = frame.copy()
        for column in frame.columns:
            series = frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            feature, visit, unit = lookup.get(str(column), ("", "", ""))
            role = "patient_id" if column == "CLINICAL-TRIAL-SUBJECT-ID" else "feature"
            if "_pch_" in str(column).lower():
                role = "baseline_relative_percent_change"
                unit = "%"
            row: dict[str, Any] = {
                "file": str(path),
                "sheet": sheet,
                "column": str(column),
                "dtype": str(series.dtype),
                "role": role,
                "feature_name": feature,
                "timepoint": visit,
                "unit": unit or "不适用/未明示",
                "n_rows": len(series),
                "n_non_null": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "n_duplicates": int(series.duplicated(keep=False).sum()) if role == "patient_id" else "",
                "min": float(finite.min()) if len(finite) else "",
                "q1": float(finite.quantile(0.25)) if len(finite) else "",
                "median": float(finite.median()) if len(finite) else "",
                "q3": float(finite.quantile(0.75)) if len(finite) else "",
                "max": float(finite.max()) if len(finite) else "",
                "planned_change": FEATURES.get(feature, {}).get("planned_change", "源表保留，仅用于一致性核验"),
            }
            if len(finite):
                iqr = float(finite.quantile(0.75) - finite.quantile(0.25))
                lower = float(finite.quantile(0.25) - 1.5 * iqr)
                upper = float(finite.quantile(0.75) + 1.5 * iqr)
                row["iqr_outlier_count"] = int(((finite < lower) | (finite > upper)).sum())
            else:
                row["iqr_outlier_count"] = ""
            schema.append(row)
    if selected is None:
        raise ValueError(f"缺少 datawith4visits sheet: {path}")
    return selected, pd.DataFrame(sheets), pd.DataFrame(schema)


def validate_raw_radiomics(frame: pd.DataFrame) -> None:
    expected = {"CLINICAL-TRIAL-SUBJECT-ID"}
    for spec in FEATURES.values():
        expected.update(spec["columns"])
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"radiomics 工作簿缺列: {sorted(missing)}")
    identifiers = frame["CLINICAL-TRIAL-SUBJECT-ID"].astype(str).str.strip()
    if not identifiers.str.fullmatch(r"\d{6}").all():
        raise ValueError("radiomics trial ID 并非全部为六位数字")
    if identifiers.duplicated().any():
        raise ValueError("radiomics trial ID 存在重复")
    if frame[list(expected)].isna().any().any():
        raise ValueError("发布 measurement 表的核心字段存在缺失")


def raw_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        trial_id = str(int(row["CLINICAL-TRIAL-SUBJECT-ID"])).zfill(6)
        for visit_index, visit in enumerate(VISITS):
            item: dict[str, Any] = {"trial_id": trial_id, "visit": visit}
            for feature, spec in FEATURES.items():
                item[feature] = float(row[spec["columns"][visit_index]])
            rows.append(item)
    return pd.DataFrame(rows)


def build_overlap(
    labels: pd.DataFrame,
    raw_long: pd.DataFrame,
    folds: pd.DataFrame,
    cache_root: Path,
) -> pd.DataFrame:
    output = labels.copy()
    output["trial_id_from_patient_id"] = output["patient_id"].map(exact_trial_id)
    output["trial_id_from_clinical"] = output["clinical_patient_id"].astype(int).astype(str).str.zfill(6)
    if not (output["trial_id_from_patient_id"] == output["trial_id_from_clinical"]).all():
        raise ValueError("patient_id 后缀与 clinical_patient_id 不一致")
    if output["trial_id_from_clinical"].duplicated().any():
        raise ValueError("808 cohort 内 trial ID 不唯一")
    radiomics_ids = set(raw_long["trial_id"])
    output["has_radiomics"] = output["trial_id_from_clinical"].isin(radiomics_ids)
    output["radiomics_visit_count"] = output["trial_id_from_clinical"].map(
        raw_long.groupby("trial_id")["visit"].nunique()
    ).fillna(0).astype(int)
    output["match_rule"] = np.where(
        output["has_radiomics"],
        "显式六位 ClinicalTrialSubjectID 等值匹配",
        "无匹配；未做模糊匹配",
    )
    output["treatment_family"] = output["arm"].map(treatment_family)
    cache_index: dict[str, str] = {}
    if cache_root.exists():
        for path in cache_root.glob("*.npz"):
            patient_id = path.name.split("_dce", 1)[0]
            cache_index[patient_id] = str(path)
    output["legacy_dce8_cache"] = output["patient_id"].map(cache_index)
    output["has_legacy_dce8_cache"] = output["legacy_dce8_cache"].notna()
    for fold in range(5):
        mapping = folds.loc[folds["fold"] == fold].set_index("patient_id")["split"]
        output[f"fold_{fold}_split"] = output["patient_id"].map(mapping)
    keep = [
        "patient_id",
        "clinical_patient_id",
        "trial_id_from_patient_id",
        "trial_id_from_clinical",
        "match_rule",
        "has_radiomics",
        "radiomics_visit_count",
        "has_legacy_dce8_cache",
        "legacy_dce8_cache",
        "label_pcr",
        "label_hr",
        "label_her2",
        "label_mp",
        "hr_her2_subtype",
        "age_at_screening",
        "arm",
        "treatment_family",
        "n_visits",
        "complete_4visits",
        *[f"fold_{fold}_split" for fold in range(5)],
    ]
    return output[keep].sort_values("patient_id").reset_index(drop=True)


def build_transition_targets(overlap: pd.DataFrame, raw_long: pd.DataFrame) -> pd.DataFrame:
    wide = raw_long.pivot(index="trial_id", columns="visit", values=list(FEATURES))
    patient_by_trial = overlap.set_index("trial_id_from_clinical")["patient_id"]
    rows: list[dict[str, Any]] = []
    for trial_id in sorted(set(wide.index).intersection(patient_by_trial.index)):
        patient_id = patient_by_trial.loc[trial_id]
        for start, end in TRANSITIONS:
            item: dict[str, Any] = {
                "patient_id": patient_id,
                "trial_id": trial_id,
                "transition": f"{start}→{end}",
                "start_visit": start,
                "end_visit": end,
            }
            for feature in FEATURES:
                start_value = float(wide.loc[trial_id, (feature, start)])
                end_value = float(wide.loc[trial_id, (feature, end)])
                item[f"{feature}_start"] = start_value
                item[f"{feature}_end"] = end_value
                item[f"{feature}_absolute_change"] = end_value - start_value
                item[f"{feature}_valid"] = bool(np.isfinite(start_value) and np.isfinite(end_value))
            rows.append(item)
    return pd.DataFrame(rows)


def build_missingness(
    raw_frame: pd.DataFrame,
    overlap: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paired = int(overlap["has_radiomics"].sum())
    cohort = len(overlap)
    for feature, spec in FEATURES.items():
        for visit, column in zip(VISITS, spec["columns"]):
            rows.append(
                {
                    "scope": "原始 measurement 工作簿",
                    "feature_name": feature,
                    "timepoint_or_transition": visit,
                    "source_column": column,
                    "n_total": len(raw_frame),
                    "n_valid": int(raw_frame[column].notna().sum()),
                    "n_missing": int(raw_frame[column].isna().sum()),
                    "missing_ratio": float(raw_frame[column].isna().mean()),
                }
            )
            rows.append(
                {
                    "scope": "808 人完整四访 MRI cohort",
                    "feature_name": feature,
                    "timepoint_or_transition": visit,
                    "source_column": column,
                    "n_total": cohort,
                    "n_valid": paired,
                    "n_missing": cohort - paired,
                    "missing_ratio": float((cohort - paired) / cohort),
                }
            )
        for start, end in TRANSITIONS:
            transition = f"{start}→{end}"
            subset = targets[targets["transition"] == transition]
            valid = int(subset[f"{feature}_valid"].sum())
            rows.append(
                {
                    "scope": "MRI-radiomics paired subset",
                    "feature_name": feature,
                    "timepoint_or_transition": transition,
                    "source_column": "由相邻两访视值计算",
                    "n_total": paired,
                    "n_valid": valid,
                    "n_missing": paired - valid,
                    "missing_ratio": float((paired - valid) / max(paired, 1)),
                }
            )
    return pd.DataFrame(rows)


def build_transition_counts(
    folds: pd.DataFrame,
    overlap: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    paired_ids = set(overlap.loc[overlap["has_radiomics"], "patient_id"])
    valid_by_transition = {
        transition: set(group.loc[group[[f"{feature}_valid" for feature in FEATURES]].all(axis=1), "patient_id"])
        for transition, group in targets.groupby("transition")
    }
    rows: list[dict[str, Any]] = []

    def append_row(fold: str | int, split: str, ids: set[str], transition: str) -> None:
        valid_ids = valid_by_transition.get(transition, set())
        row: dict[str, Any] = {
            "fold": fold,
            "split": split,
            "transition": transition,
            "n_mri_patients": len(ids),
            "n_radiomics_patients": len(ids & paired_ids),
            "n_all_features_valid": len(ids & valid_ids),
        }
        group = targets[(targets["transition"] == transition) & targets["patient_id"].isin(ids)]
        for feature in FEATURES:
            row[f"n_valid_{feature}"] = int(group[f"{feature}_valid"].sum())
        rows.append(row)

    all_ids = set(overlap["patient_id"])
    for transition in ("T0→T1", "T1→T2", "T2→T3"):
        append_row("all", "all", all_ids, transition)
    for fold in range(5):
        fold_frame = folds[folds["fold"] == fold]
        for split in ("train", "val", "test"):
            ids = set(fold_frame.loc[fold_frame["split"] == split, "patient_id"])
            for transition in ("T0→T1", "T1→T2", "T2→T3"):
                append_row(fold, split, ids, transition)
    return pd.DataFrame(rows)


def build_complete_case_bias(overlap: pd.DataFrame) -> pd.DataFrame:
    frame = overlap.copy()
    frame["group"] = np.where(frame["has_radiomics"], "radiomics可用", "radiomics不可用")
    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby("group", sort=False):
        rows.extend(
            [
                {"group": group_name, "variable": "n", "level": "", "value": len(group), "denominator": len(group)},
                {"group": group_name, "variable": "pCR比例", "level": "1", "value": float(group["label_pcr"].mean()), "denominator": len(group)},
                {"group": group_name, "variable": "年龄均值", "level": "years", "value": float(group["age_at_screening"].mean()), "denominator": int(group["age_at_screening"].notna().sum())},
                {"group": group_name, "variable": "年龄标准差", "level": "years", "value": float(group["age_at_screening"].std()), "denominator": int(group["age_at_screening"].notna().sum())},
                {"group": group_name, "variable": "MRI访视数均值", "level": "visits", "value": float(group["n_visits"].mean()), "denominator": len(group)},
                {"group": group_name, "variable": "radiomics访视数均值", "level": "visits", "value": float(group["radiomics_visit_count"].mean()), "denominator": len(group)},
            ]
        )
        for variable in ("hr_her2_subtype", "label_mp", "treatment_family"):
            counts = group[variable].astype(str).value_counts(dropna=False)
            for level, count in counts.items():
                rows.append(
                    {
                        "group": group_name,
                        "variable": variable,
                        "level": level,
                        "value": float(count / len(group)),
                        "denominator": len(group),
                    }
                )
        for fold in range(5):
            counts = group[f"fold_{fold}_split"].value_counts()
            for split, count in counts.items():
                rows.append(
                    {
                        "group": group_name,
                        "variable": f"fold_{fold}_split",
                        "level": split,
                        "value": float(count / len(group)),
                        "denominator": len(group),
                    }
                )
    rows.append(
        {
            "group": "两组比较限制",
            "variable": "baseline lesion volume",
            "level": "radiomics不可用组无同源 FTV 字段，不能进行同定义比较",
            "value": np.nan,
            "denominator": 0,
        }
    )
    return pd.DataFrame(rows)


def validate_folds(folds: pd.DataFrame, overlap: pd.DataFrame) -> dict[str, Any]:
    required = {"patient_id", "fold", "split", "label_pcr"}
    if missing := required.difference(folds.columns):
        raise ValueError(f"fold manifest 缺列: {sorted(missing)}")
    if set(folds["fold"]) != set(range(5)):
        raise ValueError("fold manifest 不是 0–4 五折")
    cohort_ids = set(overlap["patient_id"])
    for fold in range(5):
        subset = folds[folds["fold"] == fold]
        if set(subset["patient_id"]) != cohort_ids or subset["patient_id"].duplicated().any():
            raise ValueError(f"fold {fold} 患者集合/唯一性不符合 808 cohort")
    test_counts = folds.assign(is_test=folds["split"].eq("test")).groupby("patient_id")["is_test"].sum()
    if not test_counts.eq(1).all():
        raise ValueError("并非每位患者恰好一次进入 test")
    return {
        "n_rows": len(folds),
        "n_patients": folds["patient_id"].nunique(),
        "n_folds": folds["fold"].nunique(),
        "test_once_per_patient": bool(test_counts.eq(1).all()),
        "split_counts": {
            str(fold): {str(k): int(v) for k, v in group["split"].value_counts().items()}
            for fold, group in folds.groupby("fold")
        },
    }


def build_file_inventory(args: argparse.Namespace) -> pd.DataFrame:
    paths = [
        args.raw_radiomics,
        args.raw_clinical,
        args.processed_root / "clinical_labels.csv",
        args.processed_root / "clinical_labels_complete4visits.csv",
        args.processed_root / "mri_nact_features_long.csv",
        args.processed_root / "mri_nact_features_wide.csv",
        args.processed_root / "mri_nact_features_complete4visits_wide.csv",
        args.processed_root / "mri_nact_features_with_clinical_labels.csv",
        args.processed_root / "mri_nact_feature_dictionary.json",
        args.processed_root / "clinical_label_dictionary.json",
        args.processed_root / "_manifest_audit.csv",
        args.fold_manifest,
        args.cache_root,
        args.processed_root / "_corejepa_clean_dce8",
        args.processed_root / "corejepa_response_features.npz",
    ]
    purposes = [
        "原始纵向 MRI measurement/radiomics 工作簿",
        "原始 I-SPY2 clinical/pCR 工作簿",
        "985 人临床与预处理状态索引",
        "808 人完整四访临床索引",
        "384 人×4 访视纵向 measurement 长表",
        "384 人 measurement 宽表",
        "375 人同时拥有完整 MRI 的 measurement 宽表",
        "384 人 measurement+clinical 合并表（仅审计/控制组）",
        "measurement 字段映射与摘要",
        "clinical 字段字典与摘要",
        "MRI manifest 完整性审计",
        "seed-2026 候选五折 patient manifest",
        "964 人 legacy DCE8 可复用只读 cache",
        "clean 分支期望 tensor cache（当前缺失）",
        "clean 分支期望 response cache（当前缺失）",
    ]
    rows = []
    for path, purpose in zip(paths, purposes):
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "type": "directory" if path.is_dir() else path.suffix.lower().lstrip(".") or "missing",
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                "purpose": purpose,
                "read_only_source": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.raw_radiomics, args.raw_clinical, args.processed_root, args.fold_manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    raw_radiomics, sheets, schema = workbook_schema(args.raw_radiomics)
    validate_raw_radiomics(raw_radiomics)
    raw_long = raw_to_long(raw_radiomics)
    labels = pd.read_csv(args.processed_root / "clinical_labels_complete4visits.csv")
    folds = pd.read_csv(args.fold_manifest)
    overlap = build_overlap(labels, raw_long, folds, args.cache_root)
    fold_validation = validate_folds(folds, overlap)
    targets = build_transition_targets(overlap, raw_long)
    missingness = build_missingness(raw_radiomics, overlap, targets)
    transition_counts = build_transition_counts(folds, overlap, targets)
    bias = build_complete_case_bias(overlap)
    inventory = build_file_inventory(args)

    raw_ids = set(raw_long["trial_id"])
    cohort_ids = set(overlap["trial_id_from_clinical"])
    unmatched_radiomics = sorted(raw_ids - cohort_ids)
    matched = int(overlap["has_radiomics"].sum())
    summary = {
        "raw_radiomics": {
            "path": str(args.raw_radiomics),
            "sha256": sha256(args.raw_radiomics),
            "sheet_names": sheets["sheet"].tolist(),
            "n_patients": int(raw_radiomics["CLINICAL-TRIAL-SUBJECT-ID"].nunique()),
            "n_rows": len(raw_radiomics),
            "n_columns": len(raw_radiomics.columns),
            "duplicate_patient_ids": int(raw_radiomics["CLINICAL-TRIAL-SUBJECT-ID"].duplicated().sum()),
            "core_missing_cells": int(raw_radiomics.isna().sum().sum()),
        },
        "cohort_overlap": {
            "mri_complete_4visit_patients": len(overlap),
            "matched_radiomics_patients": matched,
            "unmatched_mri_patients": len(overlap) - matched,
            "overlap_ratio": matched / len(overlap),
            "radiomics_patients_outside_808": len(unmatched_radiomics),
            "radiomics_trial_ids_outside_808": unmatched_radiomics,
            "id_rule": "仅接受 ^(?:ISPY2-|ACRIN-6698-)(\\d{6})$，并与 clinical_patient_id/工作簿六位 ID 等值比较",
            "fuzzy_matching_used": False,
        },
        "transitions": {
            transition: int((targets["transition"] == transition).sum())
            for transition in ("T0→T1", "T1→T2", "T2→T3")
        },
        "fold_manifest": {
            "path": str(args.fold_manifest),
            "sha256": sha256(args.fold_manifest),
            "provenance_status": "valid_candidate_copy；无配套 clean checkpoint，不能宣称 native reproduction",
            **fold_validation,
        },
        "cache": {
            "legacy_cache_root": str(args.cache_root),
            "cohort_patients_with_cache": int(overlap["has_legacy_dce8_cache"].sum()),
            "input_contract": "legacy NPZ key x，shape [4,8,32,96,96]；第 8 通道为 ROI mask",
        },
        "units_note": "FTV 的 cc 来自随附 DICOM 数据说明；LD/BPE 单位未在工作簿/字典明确给出，分析不会据此作单位换算。",
    }

    sheets.to_csv(args.output_dir / "radiomics_workbook_sheets.csv", index=False)
    schema.to_csv(args.output_dir / "radiomics_schema.csv", index=False)
    overlap.to_csv(args.output_dir / "radiomics_patient_overlap.csv", index=False)
    missingness.to_csv(args.output_dir / "radiomics_missingness.csv", index=False)
    transition_counts.to_csv(args.output_dir / "radiomics_transition_counts.csv", index=False)
    targets.to_csv(args.output_dir / "radiomics_transition_targets_raw.csv", index=False)
    bias.to_csv(args.output_dir / "radiomics_complete_case_bias.csv", index=False)
    inventory.to_csv(args.output_dir / "data_file_inventory.csv", index=False)
    (args.output_dir / "radiomics_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
