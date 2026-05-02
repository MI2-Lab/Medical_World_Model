#!/usr/bin/env python3
"""
Preprocess the HCC-TACE-Seg dataset for longitudinal CT latent prediction.

This script intentionally uses only the Python standard library so it can run
in a minimal environment. It does not decode pixel data. Instead, it:

1. Indexes DICOM series and reads lightweight metadata from representative files.
2. Selects the best CT series per study.
3. Sorts studies by StudyDate for each patient.
4. Builds longitudinal CT pairs.
5. Attaches SEG series paths when segmentations are available in a study.
6. Writes study- and pair-level manifests for downstream training code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LONG_VR = {b"OB", b"OD", b"OF", b"OL", b"OW", b"SQ", b"UC", b"UR", b"UT", b"UN"}

TAGS = {
    (0x0008, 0x0016): "SOPClassUID",
    (0x0008, 0x0008): "ImageType",
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0030): "StudyTime",
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x103E): "SeriesDescription",
    (0x0010, 0x0020): "PatientID",
    (0x0018, 0x1030): "ProtocolName",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Preprocess HCC-TACE-Seg for longitudinal CT latent prediction."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "datasets" / "HCC-TACE-Seg_v1_202201" / "hcc_tace_seg",
        help="Root directory containing patient/study/series DICOM folders.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=repo_root / "datasets" / "HCC-TACE-Seg_v1_202201" / "metadata" / "metadata.csv",
        help="IDC metadata CSV for the dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "datasets" / "processed" / "hcc_tace_longitudinal_ct",
        help="Output directory for generated manifests.",
    )
    return parser.parse_args()


def read_dicom_header(path: Path, wanted: Sequence[Tuple[int, int]]) -> Dict[str, str]:
    with path.open("rb") as f:
        data = f.read(262144)

    if len(data) < 132 or data[128:132] != b"DICM":
        return {}

    out: Dict[str, str] = {}
    wanted_set = set(wanted)
    i = 132

    while i + 8 <= len(data):
        group, elem = struct.unpack("<HH", data[i : i + 4])
        vr = data[i + 4 : i + 6]

        if all(65 <= b <= 90 for b in vr):
            if vr in LONG_VR:
                if i + 12 > len(data):
                    break
                length = struct.unpack("<I", data[i + 8 : i + 12])[0]
                value_start = i + 12
            else:
                if i + 8 > len(data):
                    break
                length = struct.unpack("<H", data[i + 6 : i + 8])[0]
                value_start = i + 8
        else:
            length = struct.unpack("<I", data[i + 4 : i + 8])[0]
            value_start = i + 8

        value_end = value_start + length
        if value_end > len(data):
            break

        tag = (group, elem)
        if tag in wanted_set:
            out[TAGS[tag]] = data[value_start:value_end].decode("utf-8", "ignore").strip(" \0")

        i = value_end

    return out


def load_metadata(metadata_csv: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with metadata_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["SeriesInstanceUID"]] = row
    return rows


def relpath_str(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def discover_series(dataset_root: Path, metadata_by_series: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    series_records: List[Dict[str, str]] = []

    for patient_dir in sorted(dataset_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue

            for series_dir in sorted(study_dir.iterdir()):
                if not series_dir.is_dir():
                    continue

                dicom_files = sorted(series_dir.glob("*.dcm"))
                if not dicom_files:
                    continue

                series_uid = series_dir.name
                header = read_dicom_header(dicom_files[0], list(TAGS))
                metadata_row = metadata_by_series.get(series_uid, {})

                record = {
                    "patient_id": patient_dir.name,
                    "study_uid": study_dir.name,
                    "series_uid": series_uid,
                    "series_path": str(series_dir.resolve()),
                    "representative_dicom": str(dicom_files[0].resolve()),
                    "dicom_count": str(len(dicom_files)),
                    "study_date": header.get("StudyDate", ""),
                    "study_time": header.get("StudyTime", ""),
                    "modality": header.get("Modality", ""),
                    "sop_class_uid": header.get("SOPClassUID", ""),
                    "series_description": header.get("SeriesDescription", ""),
                    "protocol_name": header.get("ProtocolName", ""),
                    "image_type": header.get("ImageType", ""),
                    "metadata_file_size": metadata_row.get("FileSize", ""),
                    "metadata_image_count": metadata_row.get("ImageCount", ""),
                }
                series_records.append(record)

    return series_records


def compute_ct_score(record: Dict[str, str]) -> int:
    if record["modality"] != "CT":
        return -10_000

    score = 0
    desc = record["series_description"].lower()
    image_type = record["image_type"].upper()

    score += 1000

    if "ORIGINAL" in image_type:
        score += 100
    if "PRIMARY" in image_type:
        score += 50
    if "AXIAL" in image_type:
        score += 20
    if "DERIVED" in image_type:
        score -= 120
    if "SECONDARY" in image_type:
        score -= 60
    if "PROCESSED" in image_type:
        score -= 40

    keywords = [
        "liver",
        "pre liver",
        "phase",
        "abd",
        "c-a-p",
        "cap",
        "standard",
        "soft",
    ]
    for keyword in keywords:
        if keyword in desc:
            score += 15

    if desc.startswith("recon 2:"):
        score += 12
    if desc.startswith("recon 3:"):
        score += 10

    try:
        score += min(int(record["dicom_count"]), 400) // 10
    except ValueError:
        pass

    try:
        score += min(int(record["metadata_file_size"]) // 1_000_000, 50)
    except ValueError:
        pass

    return score


def patient_split(patient_id: str) -> str:
    bucket = int(hashlib.md5(patient_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root
    metadata_csv = args.metadata_csv
    output_root = args.output_root

    ensure_dir(output_root)

    metadata_by_series = load_metadata(metadata_csv)
    series_records = discover_series(dataset_root, metadata_by_series)

    by_patient_study: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for record in series_records:
        by_patient_study[(record["patient_id"], record["study_uid"])].append(record)

    study_rows: List[Dict[str, str]] = []
    patient_studies: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "metadata_csv": str(metadata_csv.resolve()),
        "patients_total": 0,
        "studies_total": 0,
        "series_total": len(series_records),
        "ct_series_total": 0,
        "seg_series_total": 0,
        "usable_ct_studies": 0,
        "patients_with_longitudinal_ct": 0,
        "patients_with_seg": 0,
        "adjacent_pairs_total": 0,
        "first_last_pairs_total": 0,
    }

    series_modality_counter = Counter(record["modality"] for record in series_records)
    summary["ct_series_total"] = series_modality_counter.get("CT", 0)
    summary["seg_series_total"] = series_modality_counter.get("SEG", 0)

    patients_with_seg = set()

    for (patient_id, study_uid), records in sorted(by_patient_study.items()):
        ct_candidates = [r for r in records if r["modality"] == "CT"]
        seg_candidates = [r for r in records if r["modality"] == "SEG"]

        if seg_candidates:
            patients_with_seg.add(patient_id)

        if not ct_candidates:
            continue

        best_ct = max(ct_candidates, key=compute_ct_score)
        best_ct = dict(best_ct)
        best_ct["ct_score"] = str(compute_ct_score(best_ct))

        seg_series = seg_candidates[0] if seg_candidates else None

        study_row = {
            "patient_id": patient_id,
            "study_uid": study_uid,
            "study_date": best_ct["study_date"],
            "study_time": best_ct["study_time"],
            "selected_ct_series_uid": best_ct["series_uid"],
            "selected_ct_series_path": relpath_str(Path(best_ct["series_path"]), repo_root),
            "selected_ct_representative_dicom": relpath_str(
                Path(best_ct["representative_dicom"]), repo_root
            ),
            "selected_ct_dicom_count": best_ct["dicom_count"],
            "selected_ct_series_description": best_ct["series_description"],
            "selected_ct_protocol_name": best_ct["protocol_name"],
            "selected_ct_image_type": best_ct["image_type"],
            "selected_ct_score": best_ct["ct_score"],
            "seg_series_uid": seg_series["series_uid"] if seg_series else "",
            "seg_series_path": relpath_str(Path(seg_series["series_path"]), repo_root) if seg_series else "",
            "seg_representative_dicom": (
                relpath_str(Path(seg_series["representative_dicom"]), repo_root) if seg_series else ""
            ),
            "seg_dicom_count": seg_series["dicom_count"] if seg_series else "",
            "seg_series_description": seg_series["series_description"] if seg_series else "",
        }

        study_rows.append(study_row)
        patient_studies[patient_id].append(study_row)

    summary["patients_total"] = len(patient_studies)
    summary["studies_total"] = len(study_rows)
    summary["usable_ct_studies"] = len(study_rows)
    summary["patients_with_seg"] = len(patients_with_seg)

    for rows in patient_studies.values():
        rows.sort(key=lambda row: (row["study_date"], row["study_uid"]))

    adjacent_pairs: List[Dict[str, str]] = []
    first_last_pairs: List[Dict[str, str]] = []

    for patient_id, rows in sorted(patient_studies.items()):
        if len(rows) < 2:
            continue

        summary["patients_with_longitudinal_ct"] += 1

        for idx in range(len(rows) - 1):
            source = rows[idx]
            target = rows[idx + 1]
            adjacent_pairs.append(
                {
                    "pair_id": f"{patient_id}_adj_{idx:02d}",
                    "pair_type": "adjacent",
                    "split": patient_split(patient_id),
                    "patient_id": patient_id,
                    "source_study_uid": source["study_uid"],
                    "source_study_date": source["study_date"],
                    "source_ct_series_uid": source["selected_ct_series_uid"],
                    "source_ct_series_path": source["selected_ct_series_path"],
                    "source_seg_series_uid": source["seg_series_uid"],
                    "source_seg_series_path": source["seg_series_path"],
                    "target_study_uid": target["study_uid"],
                    "target_study_date": target["study_date"],
                    "target_ct_series_uid": target["selected_ct_series_uid"],
                    "target_ct_series_path": target["selected_ct_series_path"],
                    "target_seg_series_uid": target["seg_series_uid"],
                    "target_seg_series_path": target["seg_series_path"],
                    "days_delta_proxy": "",  # real date math is omitted because dates are de-identified strings
                }
            )

        first = rows[0]
        last = rows[-1]
        first_last_pairs.append(
            {
                "pair_id": f"{patient_id}_first_last",
                "pair_type": "first_last",
                "split": patient_split(patient_id),
                "patient_id": patient_id,
                "source_study_uid": first["study_uid"],
                "source_study_date": first["study_date"],
                "source_ct_series_uid": first["selected_ct_series_uid"],
                "source_ct_series_path": first["selected_ct_series_path"],
                "source_seg_series_uid": first["seg_series_uid"],
                "source_seg_series_path": first["seg_series_path"],
                "target_study_uid": last["study_uid"],
                "target_study_date": last["study_date"],
                "target_ct_series_uid": last["selected_ct_series_uid"],
                "target_ct_series_path": last["selected_ct_series_path"],
                "target_seg_series_uid": last["seg_series_uid"],
                "target_seg_series_path": last["seg_series_path"],
                "days_delta_proxy": "",
            }
        )

    summary["adjacent_pairs_total"] = len(adjacent_pairs)
    summary["first_last_pairs_total"] = len(first_last_pairs)

    study_fieldnames = [
        "patient_id",
        "study_uid",
        "study_date",
        "study_time",
        "selected_ct_series_uid",
        "selected_ct_series_path",
        "selected_ct_representative_dicom",
        "selected_ct_dicom_count",
        "selected_ct_series_description",
        "selected_ct_protocol_name",
        "selected_ct_image_type",
        "selected_ct_score",
        "seg_series_uid",
        "seg_series_path",
        "seg_representative_dicom",
        "seg_dicom_count",
        "seg_series_description",
    ]

    pair_fieldnames = [
        "pair_id",
        "pair_type",
        "split",
        "patient_id",
        "source_study_uid",
        "source_study_date",
        "source_ct_series_uid",
        "source_ct_series_path",
        "source_seg_series_uid",
        "source_seg_series_path",
        "target_study_uid",
        "target_study_date",
        "target_ct_series_uid",
        "target_ct_series_path",
        "target_seg_series_uid",
        "target_seg_series_path",
        "days_delta_proxy",
    ]

    write_csv(output_root / "study_manifest.csv", study_rows, study_fieldnames)
    write_csv(output_root / "longitudinal_pairs_adjacent.csv", adjacent_pairs, pair_fieldnames)
    write_csv(output_root / "longitudinal_pairs_first_last.csv", first_last_pairs, pair_fieldnames)

    split_counts = Counter(row["split"] for row in adjacent_pairs)
    summary["adjacent_pair_split_counts"] = dict(split_counts)

    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with (output_root / "README.md").open("w") as f:
        f.write(
            "# HCC-TACE Longitudinal CT Preprocessing\n\n"
            "This folder contains manifests generated for longitudinal CT latent prediction.\n\n"
            "## Files\n\n"
            "- `study_manifest.csv`: one selected CT series per study, plus SEG path when available\n"
            "- `longitudinal_pairs_adjacent.csv`: adjacent study pairs per patient\n"
            "- `longitudinal_pairs_first_last.csv`: first-to-last study pair per patient\n"
            "- `summary.json`: preprocessing summary and counts\n\n"
            "## Selection Logic\n\n"
            "- Keeps CT series only for latent prediction inputs/targets\n"
            "- Selects one best CT series per study using DICOM metadata and a heuristic score\n"
            "- Sorts studies by `StudyDate`\n"
            "- Attaches SEG series from the same study when present\n"
            "- Splits data deterministically by patient into train/val/test\n"
        )

    print(f"Wrote preprocessing outputs to: {output_root}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
