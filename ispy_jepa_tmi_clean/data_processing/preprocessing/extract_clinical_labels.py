#!/usr/bin/env python3
"""Extract I-SPY2 clinical labels and align them to preprocessed patient folders."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from path_defaults import ispy2_preprocessed_root, ispy2_raw_root


DEFAULT_CLINICAL_XLSX = ispy2_raw_root() / "ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx"
DEFAULT_OUTPUT_ROOT = ispy2_preprocessed_root()
DEFAULT_SHEET = "ISPY2_n985_TCIA_clinical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-xlsx", type=Path, default=DEFAULT_CLINICAL_XLSX)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--complete4-csv", type=Path, default=None)
    parser.add_argument("--dictionary-json", type=Path, default=None)
    return parser.parse_args()


def clean_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def clinical_id(value: Any) -> str:
    text = clean_string(value)
    if text is None:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def nullable_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_race(value: Any) -> str | None:
    text = clean_string(value)
    if text is None:
        return None
    if ";" in text or "," in text:
        return "Multiple"
    if text == "Native Hawaiian or Other Pacific Islande":
        return "Native Hawaiian or Pacific Islander"
    return text


def normalize_menopausal_status(value: Any) -> str | None:
    text = clean_string(value)
    if text is None:
        return None
    lower = text.lower()
    if "premenopausal" in lower:
        return "Premenopausal"
    if "postmenopausal" in lower:
        return "Postmenopausal"
    if "perimenopausal" in lower:
        return "Perimenopausal"
    if "age > 50" in lower:
        return "Other_age_gt_50"
    if "age < 50" in lower:
        return "Other_age_lt_50"
    return text


def hr_her2_subtype(hr: int | None, her2: int | None) -> str | None:
    if hr is None or her2 is None:
        return None
    return f"HR{'+' if hr == 1 else '-'}/HER2{'+' if her2 == 1 else '-'}"


def discover_patient_map(output_root: Path) -> dict[str, str]:
    suffix_to_patient: dict[str, list[str]] = {}
    for path in sorted(output_root.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if not (path / "manifest.json").exists():
            continue
        suffix = path.name.split("-")[-1]
        suffix_to_patient.setdefault(suffix, []).append(path.name)

    patient_map: dict[str, str] = {}
    for suffix, patients in suffix_to_patient.items():
        if len(patients) == 1:
            patient_map[suffix] = patients[0]
    return patient_map


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


def value_counts_for_json(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("<NA>").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def build_labels(clinical_xlsx: Path, sheet: str, output_root: Path) -> pd.DataFrame:
    raw = pd.read_excel(clinical_xlsx, sheet_name=sheet, dtype=object)
    patient_map = discover_patient_map(output_root)

    labels = pd.DataFrame()
    labels["clinical_patient_id"] = raw["Patient_ID"].map(clinical_id)
    labels["patient_id"] = labels["clinical_patient_id"].map(patient_map)
    labels["preprocessed_dir"] = labels["patient_id"].map(
        lambda pid: str(output_root / pid) if isinstance(pid, str) else None
    )

    labels["label_pcr"] = raw["pCR"].map(nullable_int)
    labels["label_hr"] = raw["HR"].map(nullable_int)
    labels["label_her2"] = raw["HER2"].map(nullable_int)
    labels["label_mp"] = raw["MP"].map(nullable_int)
    labels["age_at_screening"] = raw["Age_at_Screening"].map(nullable_int)
    labels["arm"] = raw["Arm"].map(clean_string)
    labels["hr_her2_subtype"] = [
        hr_her2_subtype(hr, her2) for hr, her2 in zip(labels["label_hr"], labels["label_her2"])
    ]
    labels["race_raw"] = raw["Race"].map(clean_string)
    labels["race_simple"] = raw["Race"].map(normalize_race)
    labels["menopausal_status_raw"] = raw["menopausal_status"].map(clean_string)
    labels["menopausal_status_simple"] = raw["menopausal_status"].map(
        normalize_menopausal_status
    )
    labels["ethnicity"] = raw["ethnicity"].map(clean_string)

    labels["raw_Patient_ID"] = raw["Patient_ID"].map(clean_string)
    labels["raw_HR"] = raw["HR"].map(clean_string)
    labels["raw_HER2"] = raw["HER2"].map(clean_string)
    labels["raw_MP"] = raw["MP"].map(clean_string)
    labels["raw_pCR"] = raw["pCR"].map(clean_string)

    audit = load_audit(output_root)
    if not audit.empty:
        labels = labels.merge(audit, on="patient_id", how="left")

    return labels


def write_dictionary(labels: pd.DataFrame, clinical_xlsx: Path, output_path: Path) -> None:
    dictionary = {
        "source": str(clinical_xlsx),
        "n_rows": int(len(labels)),
        "label_columns": {
            "label_pcr": "Pathologic complete response, from pCR; binary 0/1.",
            "label_hr": "Hormone receptor status, from HR; binary 0/1.",
            "label_her2": "HER2 status, from HER2; binary 0/1.",
            "label_mp": "MammaPrint/MP status as provided; binary 0/1.",
            "age_at_screening": "Age at screening, integer years.",
            "arm": "Treatment arm, categorical.",
            "hr_her2_subtype": "Derived from HR and HER2 as HR+/HER2-, etc.",
            "race_simple": "Race with multi-race values collapsed to Multiple.",
            "menopausal_status_simple": "Whitespace-normalized menopausal category.",
            "ethnicity": "Ethnicity category as provided.",
        },
        "missing_counts": {col: int(labels[col].isna().sum()) for col in labels.columns},
        "value_counts": {
            col: value_counts_for_json(labels[col])
            for col in [
                "label_pcr",
                "label_hr",
                "label_her2",
                "label_mp",
                "arm",
                "hr_her2_subtype",
                "race_simple",
                "menopausal_status_simple",
                "ethnicity",
                "audit_status",
                "n_visits",
            ]
            if col in labels.columns
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dictionary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    labels_csv = args.labels_csv or output_root / "clinical_labels.csv"
    complete4_csv = args.complete4_csv or output_root / "clinical_labels_complete4visits.csv"
    dictionary_json = args.dictionary_json or output_root / "clinical_label_dictionary.json"

    labels = build_labels(args.clinical_xlsx, args.sheet, output_root)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_csv, index=False)

    complete4 = labels[labels.get("complete_4visits").astype(str).str.lower() == "true"]
    complete4.to_csv(complete4_csv, index=False)

    write_dictionary(labels, args.clinical_xlsx, dictionary_json)

    print(f"Wrote {labels_csv} ({len(labels)} rows)")
    print(f"Wrote {complete4_csv} ({len(complete4)} rows)")
    print(f"Wrote {dictionary_json}")


if __name__ == "__main__":
    main()
