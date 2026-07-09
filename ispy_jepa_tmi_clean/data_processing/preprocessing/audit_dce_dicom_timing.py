#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from path_defaults import breastdcedl_metadata_csv, ispy2_preprocessed_root


VISITS = ("T0", "T1", "T2", "T3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit raw DICOM temporal phase timing for I-SPY2 original DCE series."
    )
    parser.add_argument("--preprocessed-root", type=Path, default=ispy2_preprocessed_root())
    parser.add_argument(
        "--breastdcedl-metadata-csv",
        type=Path,
        default=breastdcedl_metadata_csv(),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ispy2_preprocessed_root() / "_audits" / "dce_dicom_timing_audit.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ispy2_preprocessed_root() / "_audits" / "dce_dicom_timing_audit_summary.json",
    )
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-dicoms-per-series", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def safe_int(value: Any, default: int) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def clip_idx(index: int, n_times: int) -> int:
    return int(np.clip(index, 0, max(n_times - 1, 0)))


def parse_time_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        text = text.rsplit("T", 1)[-1]
    text = text.replace(":", "")
    try:
        parts = text.split(".", 1)
        whole = parts[0].rjust(6, "0")
        frac = float("0." + parts[1]) if len(parts) == 2 and parts[1] else 0.0
        hour = int(whole[:2])
        minute = int(whole[2:4])
        second = int(whole[4:6])
        return float(hour * 3600 + minute * 60 + second) + frac
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(str(value).strip())
        if not np.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def elapsed_seconds(times: list[float | None]) -> list[float | None]:
    finite = [time for time in times if time is not None and np.isfinite(time)]
    if not finite:
        return [None for _ in times]
    base = min(finite)
    out: list[float | None] = []
    for time in times:
        if time is None or not np.isfinite(time):
            out.append(None)
            continue
        delta = float(time - base)
        if delta < -12 * 3600:
            delta += 24 * 3600
        out.append(delta)
    return out


def has_nonzero_timing(values: list[float | None]) -> bool:
    finite = [round(float(value), 3) for value in values if value is not None and np.isfinite(value)]
    return len(set(finite)) > 1 and max(finite) > 0


def cumulative_temporal_resolution(values: list[float | None]) -> list[float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value) and float(value) > 0]
    if not finite:
        return [None for _ in values]
    default_interval = float(np.median(finite))
    elapsed = [0.0]
    for index in range(1, len(values)):
        interval = values[index]
        if interval is None or not np.isfinite(interval) or float(interval) <= 0:
            interval = default_interval
        elapsed.append(float(elapsed[-1]) + float(interval))
    return elapsed


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "pid" not in frame.columns:
        return {}
    return frame.set_index("pid").to_dict(orient="index")


def iter_patient_manifests(root: Path, max_patients: int | None) -> list[Path]:
    manifests = sorted(root.glob("*/manifest.json"))
    if max_patients is not None:
        manifests = manifests[:max_patients]
    return manifests


def representative_time(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    value, _ = counter.most_common(1)[0]
    return None if value == "None" else value


def compact_list(values: list[Any]) -> str:
    return json.dumps(values, separators=(",", ":"))


def audit_series(
    series_dir: str,
    expected_n_times: int,
    max_dicoms: int,
) -> dict[str, Any]:
    path = Path(series_dir)
    files = sorted(path.glob("*.dcm"))
    by_temporal: dict[int, Counter[str]] = defaultdict(Counter)
    temporal_resolution_by_temporal: dict[int, Counter[str]] = defaultdict(Counter)
    acq_time_counter: Counter[str] = Counter()
    temporal_counter: Counter[str] = Counter()
    number_of_temporal_positions: Counter[str] = Counter()
    acquisition_number_counter: Counter[str] = Counter()
    instance_counter: Counter[str] = Counter()
    sampled = 0
    dicom_errors = 0

    for file_path in files:
        if sampled >= max_dicoms:
            break
        sampled += 1
        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        except Exception:
            dicom_errors += 1
            continue
        acq_time = str(getattr(ds, "AcquisitionTime", None))
        temporal = str(getattr(ds, "TemporalPositionIdentifier", None))
        number_of_temporal_positions[str(getattr(ds, "NumberOfTemporalPositions", None))] += 1
        acquisition_number_counter[str(getattr(ds, "AcquisitionNumber", None))] += 1
        instance_counter[str(getattr(ds, "InstanceNumber", None))] += 1
        acq_time_counter[acq_time] += 1
        temporal_counter[temporal] += 1
        try:
            temporal_int = int(temporal)
        except ValueError:
            temporal_int = -1
        if temporal_int >= 0:
            by_temporal[temporal_int][acq_time] += 1
            temporal_resolution = ds.get((0x0020, 0x0110))
            if temporal_resolution is not None:
                temporal_resolution_by_temporal[temporal_int][str(temporal_resolution.value)] += 1
        if len(by_temporal) >= expected_n_times and len(acq_time_counter) >= expected_n_times:
            break

    temporal_ids = sorted(by_temporal)
    frame_acq_times: list[str | None]
    order_source: str
    if temporal_ids:
        frame_acq_times = [representative_time(by_temporal[idx]) for idx in temporal_ids]
        order_source = "TemporalPositionIdentifier"
    else:
        frame_acq_times = sorted([key for key in acq_time_counter if key != "None"], key=lambda x: parse_time_seconds(x) or 0)
        order_source = "AcquisitionTime"

    acq_seconds = [parse_time_seconds(value) for value in frame_acq_times]
    acq_elapsed = elapsed_seconds(acq_seconds)
    frame_temporal_resolution = [
        parse_float(representative_time(temporal_resolution_by_temporal[idx])) for idx in temporal_ids
    ]
    if has_nonzero_timing(acq_elapsed):
        frame_elapsed = acq_elapsed
        frame_elapsed_source = "AcquisitionTime"
    elif temporal_ids and any(value is not None for value in frame_temporal_resolution):
        frame_elapsed = cumulative_temporal_resolution(frame_temporal_resolution)
        frame_elapsed_source = "TemporalResolution"
    else:
        frame_elapsed = acq_elapsed
        frame_elapsed_source = "AcquisitionTime_constant_or_missing"
    return {
        "raw_dce_series": str(path),
        "dicom_count": len(files),
        "sampled_dicoms": sampled,
        "dicom_errors": dicom_errors,
        "expected_n_times": int(expected_n_times),
        "unique_temporal_positions_sampled": int(len([key for key in temporal_counter if key != "None"])),
        "unique_acq_times_sampled": int(len([key for key in acq_time_counter if key != "None"])),
        "number_of_temporal_positions_tag": representative_time(number_of_temporal_positions),
        "unique_acquisition_numbers_sampled": int(len([key for key in acquisition_number_counter if key != "None"])),
        "unique_instance_numbers_sampled": int(len([key for key in instance_counter if key != "None"])),
        "order_source": order_source,
        "frame_elapsed_source": frame_elapsed_source,
        "temporal_ids": temporal_ids,
        "frame_acq_times": frame_acq_times,
        "frame_temporal_resolution_sec": frame_temporal_resolution,
        "frame_elapsed_sec": frame_elapsed,
    }


def row_for_visit(
    manifest_path: Path,
    visit: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    max_dicoms: int,
) -> dict[str, Any]:
    patient_id = visit.get("patient_id") or json.loads(manifest_path.read_text()).get("patient_id", manifest_path.parent.name)
    visit_name = str(visit["visit"])
    n_times = int(visit.get("n_times") or visit.get("dce_shape", [0, 0, 0, 0])[3])
    series = audit_series(str(visit["raw_dce_series"]), n_times, max_dicoms)
    frame_elapsed = series["frame_elapsed_sec"]
    meta = metadata.get(str(patient_id), {})
    meta_pre = clip_idx(safe_int(meta.get("pre"), 0), n_times)
    meta_early = clip_idx(safe_int(meta.get("post_early"), min(2, n_times - 1)), n_times)
    meta_late = clip_idx(safe_int(meta.get("post_late"), min(5, n_times - 1)), n_times)
    current_pre = 0
    current_early = clip_idx(1, n_times)
    current_late = clip_idx(n_times - 1, n_times)
    fixed_early = clip_idx(2, n_times)
    fixed_late = clip_idx(5, n_times)

    def time_at(index: int) -> float | None:
        if 0 <= index < len(frame_elapsed):
            value = frame_elapsed[index]
            return None if value is None else round(float(value), 3)
        return None

    row = {
        "patient_id": patient_id,
        "visit": visit_name,
        "n_times": n_times,
        "n_z": int(visit.get("n_z", 0)),
        "dce_dicoms_manifest": int(visit.get("dce_dicoms", 0)),
        "raw_dce_series": visit.get("raw_dce_series"),
        "dce_series_description": visit.get("dce_series_description"),
        "meta_pre_idx": meta_pre,
        "meta_early_idx": meta_early,
        "meta_late_idx": meta_late,
        "current_pre_idx": current_pre,
        "current_early_idx": current_early,
        "current_late_idx": current_late,
        "fixed_early2_idx": fixed_early,
        "fixed_late5_idx": fixed_late,
        "current_early_matches_meta": int(current_early == meta_early),
        "current_late_matches_meta": int(current_late == meta_late),
        "current_early_sec": time_at(current_early),
        "current_late_sec": time_at(current_late),
        "meta_early_sec": time_at(meta_early),
        "meta_late_sec": time_at(meta_late),
        "fixed_early2_sec": time_at(fixed_early),
        "fixed_late5_sec": time_at(fixed_late),
        "frame_elapsed_sec": compact_list([None if value is None else round(float(value), 3) for value in frame_elapsed]),
        "frame_acq_times": compact_list(series["frame_acq_times"]),
        "frame_temporal_resolution_sec": compact_list(
            [None if value is None else round(float(value), 3) for value in series["frame_temporal_resolution_sec"]]
        ),
        "temporal_ids": compact_list(series["temporal_ids"]),
    }
    for key, value in series.items():
        if key not in {
            "raw_dce_series",
            "frame_elapsed_sec",
            "frame_acq_times",
            "frame_temporal_resolution_sec",
            "temporal_ids",
        }:
            row[key] = value
    return row


def patient_rows(task: tuple[str, dict[str, dict[str, Any]], int]) -> list[dict[str, Any]]:
    manifest_path = Path(task[0])
    metadata = task[1]
    max_dicoms = task[2]
    manifest = json.loads(manifest_path.read_text())
    visits = [visit for visit in manifest.get("visits", []) if visit.get("visit") in VISITS]
    if len(visits) != 4:
        return []
    return [row_for_visit(manifest_path, visit, metadata, max_dicoms) for visit in visits]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    priority = [
        "patient_id",
        "visit",
        "n_times",
        "n_z",
        "dce_dicoms_manifest",
        "dicom_count",
        "sampled_dicoms",
        "expected_n_times",
        "unique_temporal_positions_sampled",
        "unique_acq_times_sampled",
        "number_of_temporal_positions_tag",
        "current_early_idx",
        "current_early_sec",
        "meta_early_idx",
        "meta_early_sec",
        "current_late_idx",
        "current_late_sec",
        "meta_late_idx",
        "meta_late_sec",
        "fixed_early2_idx",
        "fixed_early2_sec",
        "fixed_late5_idx",
        "fixed_late5_sec",
        "current_early_matches_meta",
        "current_late_matches_meta",
        "frame_elapsed_sec",
        "frame_elapsed_source",
        "frame_acq_times",
        "frame_temporal_resolution_sec",
        "temporal_ids",
        "raw_dce_series",
    ]
    ordered = [name for name in priority if name in fieldnames] + [name for name in fieldnames if name not in priority]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_patients": int(frame["patient_id"].nunique()) if len(frame) else 0,
    }
    if len(frame) == 0:
        return summary
    for column in [
        "n_times",
        "unique_temporal_positions_sampled",
        "unique_acq_times_sampled",
        "number_of_temporal_positions_tag",
        "frame_elapsed_source",
    ]:
        summary[f"{column}_counts"] = frame[column].astype(str).value_counts(dropna=False).to_dict()
    for column in [
        "current_early_sec",
        "meta_early_sec",
        "current_late_sec",
        "meta_late_sec",
        "fixed_early2_sec",
        "fixed_late5_sec",
    ]:
        values = pd.to_numeric(frame[column], errors="coerce")
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(values.median()),
            "p10": float(values.quantile(0.10)),
            "p90": float(values.quantile(0.90)),
            "missing": int(values.isna().sum()),
        }
    summary["current_early_mismatch_rate"] = float(1.0 - frame["current_early_matches_meta"].mean())
    summary["current_late_mismatch_rate"] = float(1.0 - frame["current_late_matches_meta"].mean())
    by_visit = {}
    for visit, group in frame.groupby("visit"):
        by_visit[str(visit)] = {
            "n": int(len(group)),
            "current_early_mismatch_rate": float(1.0 - group["current_early_matches_meta"].mean()),
            "current_late_mismatch_rate": float(1.0 - group["current_late_matches_meta"].mean()),
            "current_early_sec_median": float(pd.to_numeric(group["current_early_sec"], errors="coerce").median()),
            "meta_early_sec_median": float(pd.to_numeric(group["meta_early_sec"], errors="coerce").median()),
            "current_late_sec_median": float(pd.to_numeric(group["current_late_sec"], errors="coerce").median()),
            "meta_late_sec_median": float(pd.to_numeric(group["meta_late_sec"], errors="coerce").median()),
        }
    summary["by_visit"] = by_visit
    return summary


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.breastdcedl_metadata_csv)
    manifests = iter_patient_manifests(args.preprocessed_root, args.max_patients)
    tasks = [(str(path), metadata, args.max_dicoms_per_series) for path in manifests]
    rows: list[dict[str, Any]] = []
    if args.num_workers <= 1:
        for task in tqdm(tasks, desc="audit DCE DICOM timing"):
            rows.extend(patient_rows(task))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(patient_rows, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="audit DCE DICOM timing"):
                rows.extend(future.result())
    rows.sort(key=lambda row: (str(row["patient_id"]), str(row["visit"])))
    write_csv(args.output_csv, rows)
    summary = summarize(rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
