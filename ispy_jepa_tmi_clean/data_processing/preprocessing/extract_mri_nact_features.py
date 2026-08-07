#!/usr/bin/env python3
"""Extract multi-feature MRI NACT tabular features and align them to I-SPY2 outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from path_defaults import ispy2_preprocessed_root, ispy2_raw_root


DEFAULT_FEATURE_XLSX = ispy2_raw_root() / "Multi-feature-MRI-NACT-Data.xlsx"
DEFAULT_OUTPUT_ROOT = ispy2_preprocessed_root()
DEFAULT_SHEET = "datawith4visits"

RENAME_MAP = {
    "CLINICAL-TRIAL-SUBJECT-ID": "clinical_patient_id",
    "VOLUME_TUM_BLU_V10": "tumor_volume_blu_t0",
    "VOLUME_TUM_BLU_V20": "tumor_volume_blu_t1",
    "VOLUME_TUM_BLU_V30": "tumor_volume_blu_t2",
    "VOLUME_TUM_BLU_V40": "tumor_volume_blu_t3",
    "SPHERICITY_T0": "sphericity_t0",
    "SPHERICITY_T1": "sphericity_t1",
    "SPHERICITY_T2": "sphericity_t2",
    "SPHERICITY_T3": "sphericity_t3",
    "LD_T0": "ld_t0",
    "LD_T1": "ld_t1",
    "LD_T2": "ld_t2",
    "LD_T3": "ld_t3",
    "BPE_5slice_mean_T0": "bpe_5slice_mean_t0",
    "BPE_5slice_mean_T1": "bpe_5slice_mean_t1",
    "BPE_5slice_mean_T2": "bpe_5slice_mean_t2",
    "BPE_5slice_mean_T3": "bpe_5slice_mean_t3",
    "FTV_pch_T0_T1": "ftv_pch_t0_t1",
    "FTV_pch_T0_T2": "ftv_pch_t0_t2",
    "FTV_pch_T0_T3": "ftv_pch_t0_t3",
    "Sphericity_pch_T0_T1": "sphericity_pch_t0_t1",
    "Sphericity_pch_T0_T2": "sphericity_pch_t0_t2",
    "Sphericity_pch_T0_T3": "sphericity_pch_t0_t3",
    "LD_pch_T0_T1": "ld_pch_t0_t1",
    "LD_pch_T0_T2": "ld_pch_t0_t2",
    "LD_pch_T0_T3": "ld_pch_t0_t3",
    "BPE_pch_T0_T1": "bpe_pch_t0_t1",
    "BPE_pch_T0_T2": "bpe_pch_t0_t2",
    "BPE_pch_T0_T3": "bpe_pch_t0_t3",
}

VISITS = ("T0", "T1", "T2", "T3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-xlsx", type=Path, default=DEFAULT_FEATURE_XLSX)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wide-csv", type=Path, default=None)
    parser.add_argument("--long-csv", type=Path, default=None)
    parser.add_argument("--complete4-wide-csv", type=Path, default=None)
    parser.add_argument("--with-clinical-csv", type=Path, default=None)
    parser.add_argument("--dictionary-json", type=Path, default=None)
    return parser.parse_args()


def clean_id(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def discover_patient_map(output_root: Path) -> dict[str, str]:
    suffix_to_patient: dict[str, list[str]] = {}
    for path in sorted(output_root.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if not (path / "manifest.json").exists():
            continue
        suffix = path.name.split("-")[-1]
        suffix_to_patient.setdefault(suffix, []).append(path.name)

    return {suffix: values[0] for suffix, values in suffix_to_patient.items() if len(values) == 1}


def load_audit(output_root: Path) -> pd.DataFrame:
    audit_path = output_root / "_manifest_audit.csv"
    if not audit_path.exists():
        return pd.DataFrame()
    audit = pd.read_csv(audit_path, dtype=object)
    keep = [
        "patient_id",
        "audit_status",
        "n_visits",
        "complete_4visits",
        "missing",
        "failed_visits",
        "aligned_dce_visits",
    ]
    return audit[[col for col in keep if col in audit.columns]]


def load_clinical_labels(output_root: Path) -> pd.DataFrame:
    labels_path = output_root / "clinical_labels.csv"
    if not labels_path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path, dtype=object)
    keep = [
        "patient_id",
        "label_pcr",
        "label_hr",
        "label_her2",
        "label_mp",
        "age_at_screening",
        "arm",
        "hr_her2_subtype",
        "race_simple",
        "menopausal_status_simple",
        "ethnicity",
    ]
    return labels[[col for col in keep if col in labels.columns]]


def build_wide(feature_xlsx: Path, sheet: str, output_root: Path) -> pd.DataFrame:
    raw = pd.read_excel(feature_xlsx, sheet_name=sheet, dtype=object)
    wide = raw.rename(columns=RENAME_MAP).copy()
    wide["clinical_patient_id"] = wide["clinical_patient_id"].map(clean_id)

    patient_map = discover_patient_map(output_root)
    wide.insert(1, "patient_id", wide["clinical_patient_id"].map(patient_map))
    wide.insert(
        2,
        "preprocessed_dir",
        wide["patient_id"].map(lambda pid: str(output_root / pid) if isinstance(pid, str) else None),
    )
    wide.insert(
        3,
        "collection",
        wide["patient_id"].map(
            lambda pid: "acrin_6698" if isinstance(pid, str) and pid.startswith("ACRIN-") else "ispy2"
        ),
    )

    for col in wide.columns:
        if col not in {"clinical_patient_id", "patient_id", "preprocessed_dir", "collection"}:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    audit = load_audit(output_root)
    if not audit.empty:
        wide = wide.merge(audit, on="patient_id", how="left")

    return wide


def build_long(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_cols = [
        "clinical_patient_id",
        "patient_id",
        "preprocessed_dir",
        "collection",
        "audit_status",
        "n_visits",
        "complete_4visits",
        "missing",
        "failed_visits",
        "aligned_dce_visits",
    ]
    for record in wide.to_dict(orient="records"):
        base = {col: record.get(col) for col in base_cols if col in record}
        for visit in VISITS:
            suffix = visit.lower()
            row = {
                **base,
                "visit": visit,
                "tumor_volume_blu": record.get(f"tumor_volume_blu_{suffix}"),
                "sphericity": record.get(f"sphericity_{suffix}"),
                "ld": record.get(f"ld_{suffix}"),
                "bpe_5slice_mean": record.get(f"bpe_5slice_mean_{suffix}"),
                "ftv_pch_from_t0": None,
                "sphericity_pch_from_t0": None,
                "ld_pch_from_t0": None,
                "bpe_pch_from_t0": None,
            }
            if visit != "T0":
                pair = f"t0_{suffix}"
                row["ftv_pch_from_t0"] = record.get(f"ftv_pch_{pair}")
                row["sphericity_pch_from_t0"] = record.get(f"sphericity_pch_{pair}")
                row["ld_pch_from_t0"] = record.get(f"ld_pch_{pair}")
                row["bpe_pch_from_t0"] = record.get(f"bpe_pch_{pair}")
            rows.append(row)
    return pd.DataFrame(rows)


def with_clinical_labels(wide: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    clinical = load_clinical_labels(output_root)
    if clinical.empty:
        return wide
    return wide.merge(clinical, on="patient_id", how="left")


def stats_for_json(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    stats: dict[str, dict[str, float | int | None]] = {}
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        stats[col] = {
            "missing": int(series.isna().sum()),
            "min": None if series.dropna().empty else float(series.min()),
            "max": None if series.dropna().empty else float(series.max()),
            "mean": None if series.dropna().empty else float(series.mean()),
        }
    return stats


def write_dictionary(
    wide: pd.DataFrame,
    long: pd.DataFrame,
    feature_xlsx: Path,
    output_path: Path,
) -> None:
    feature_cols = [
        col
        for col in wide.columns
        if col
        not in {
            "clinical_patient_id",
            "patient_id",
            "preprocessed_dir",
            "collection",
            "audit_status",
            "n_visits",
            "complete_4visits",
            "missing",
            "failed_visits",
            "aligned_dce_visits",
        }
    ]
    dictionary = {
        "source": str(feature_xlsx),
        "n_patients": int(len(wide)),
        "n_long_rows": int(len(long)),
        "original_to_extracted_column_map": RENAME_MAP,
        "notes": {
            "wide": "One row per patient; V10/V20/V30/V40 are represented as T0/T1/T2/T3.",
            "long": "One row per patient per visit with per-visit features and percent change from T0 for T1-T3.",
            "pch": "Percent-change columns are copied from the source spreadsheet.",
        },
        "missing_counts": {col: int(wide[col].isna().sum()) for col in wide.columns},
        "audit_status_counts": wide.get("audit_status", pd.Series(dtype=object))
        .fillna("<NA>")
        .value_counts()
        .astype(int)
        .to_dict(),
        "n_visits_counts": wide.get("n_visits", pd.Series(dtype=object))
        .fillna("<NA>")
        .value_counts()
        .astype(int)
        .to_dict(),
        "feature_summary": stats_for_json(wide, feature_cols),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dictionary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    wide_csv = args.wide_csv or output_root / "mri_nact_features_wide.csv"
    long_csv = args.long_csv or output_root / "mri_nact_features_long.csv"
    complete4_wide_csv = args.complete4_wide_csv or output_root / "mri_nact_features_complete4visits_wide.csv"
    with_clinical_csv = args.with_clinical_csv or output_root / "mri_nact_features_with_clinical_labels.csv"
    dictionary_json = args.dictionary_json or output_root / "mri_nact_feature_dictionary.json"

    wide = build_wide(args.feature_xlsx, args.sheet, output_root)
    long = build_long(wide)
    wide_with_clinical = with_clinical_labels(wide, output_root)

    wide_csv.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)
    wide[wide.get("complete_4visits").astype(str).str.lower() == "true"].to_csv(
        complete4_wide_csv, index=False
    )
    wide_with_clinical.to_csv(with_clinical_csv, index=False)
    write_dictionary(wide, long, args.feature_xlsx, dictionary_json)

    print(f"Wrote {wide_csv} ({len(wide)} rows)")
    print(f"Wrote {long_csv} ({len(long)} rows)")
    print(
        f"Wrote {complete4_wide_csv} "
        f"({int((wide.get('complete_4visits').astype(str).str.lower() == 'true').sum())} rows)"
    )
    print(f"Wrote {with_clinical_csv} ({len(wide_with_clinical)} rows)")
    print(f"Wrote {dictionary_json}")


if __name__ == "__main__":
    main()
