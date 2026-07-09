#!/usr/bin/env python3
"""Batch preprocess I-SPY2 patients.

The batch runner delegates each patient to preprocess_one_patient.py so that
each patient has an isolated log and failures do not stop the whole run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from path_defaults import ispy2_preprocessed_root, ispy2_raw_root


DEFAULT_RAW_ROOT = ispy2_raw_root()
DEFAULT_OUTPUT_ROOT = ispy2_preprocessed_root()
THIS_DIR = Path(__file__).resolve().parent
ONE_PATIENT_SCRIPT = THIS_DIR / "preprocess_one_patient.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--patients-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def discover_patients(raw_root: Path) -> list[str]:
    patients = []
    for collection in ["ispy2", "acrin_6698"]:
        collection_dir = raw_root / collection
        if not collection_dir.is_dir():
            continue
        patients.extend(p.name for p in collection_dir.iterdir() if p.is_dir())
    return sorted(set(patients))


def load_patients(raw_root: Path, patients_file: Path | None) -> list[str]:
    if patients_file is None:
        return discover_patients(raw_root)
    return [
        line.strip()
        for line in patients_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def is_complete(patient_id: str, output_root: Path) -> bool:
    manifest_path = output_root / patient_id / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("missing"):
        return False
    if manifest.get("failed_visits"):
        return False
    visits = manifest.get("visits", [])
    if len(visits) != 4:
        return False
    for visit in visits:
        for key in ["dce_nifti", "mask_nifti", "ftv_mask_nifti"]:
            path = Path(visit.get(key, ""))
            if not path.exists():
                return False
    return True


def run_patient(
    patient_id: str,
    raw_root: Path,
    output_root: Path,
    logs_dir: Path,
    overwrite: bool,
    resume: bool,
) -> dict[str, object]:
    started = time.time()
    if resume and not overwrite and is_complete(patient_id, output_root):
        return {
            "patient_id": patient_id,
            "status": "skipped_complete",
            "seconds": 0.0,
            "returncode": 0,
            "log": "",
        }

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{patient_id}.log"
    cmd = [
        sys.executable,
        str(ONE_PATIENT_SCRIPT),
        "--patient-id",
        patient_id,
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(output_root),
    ]
    if overwrite:
        cmd.append("--overwrite")

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, text=True, stdout=log, stderr=subprocess.STDOUT)

    status = "ok" if proc.returncode == 0 else "failed"
    manifest_path = output_root / patient_id / "manifest.json"
    if proc.returncode == 0 and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("missing") or manifest.get("failed_visits"):
            status = "partial"

    return {
        "patient_id": patient_id,
        "status": status,
        "seconds": round(time.time() - started, 3),
        "returncode": proc.returncode,
        "log": str(log_path),
    }


def write_batch_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "status", "seconds", "returncode", "log"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    patients = load_patients(args.raw_root, args.patients_file)
    if args.limit is not None:
        patients = patients[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_root / "_logs"
    summary_path = args.output_root / "_batch_summary.csv"

    print(f"Patients: {len(patients)}")
    print(f"Output: {args.output_root}")
    print(f"Workers: {args.workers}")

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_patient,
                patient_id,
                args.raw_root,
                args.output_root,
                logs_dir,
                args.overwrite,
                args.resume,
            ): patient_id
            for patient_id in patients
        }
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            completed += 1
            print(
                f"[{completed}/{len(patients)}] {row['patient_id']} "
                f"{row['status']} ({row['seconds']}s)"
            )
            write_batch_summary(sorted(rows, key=lambda r: str(r["patient_id"])), summary_path)

    failed = [r for r in rows if r["status"] == "failed"]
    partial = [r for r in rows if r["status"] == "partial"]
    print(f"Wrote {summary_path}")
    print(f"Failed: {len(failed)}")
    print(f"Partial: {len(partial)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
