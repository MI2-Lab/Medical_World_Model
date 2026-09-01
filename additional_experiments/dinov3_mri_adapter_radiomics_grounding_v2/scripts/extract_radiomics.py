#!/usr/bin/env python3
"""Extract seven-channel appearance/kinetic radiomics in the locked side env."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json, canonical_sha256, file_sha256, private_patient_token  # noqa: E402
from dinov3_rg.cache_io import load_c1b_manifest  # noqa: E402
from dinov3_rg.radiomics import extract_patient_radiomics, ftv_wide, make_pyradiomics_extractor  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiomics-python", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--roi-dir", type=Path, default=ROOT / "cache/radiomics_rois")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cache/radiomics_raw")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


_WORKER_EXTRACTOR = None


def _worker_task(arguments):
    global _WORKER_EXTRACTOR
    patient_id, entry, roi_dir, output_dir, overwrite = arguments
    if _WORKER_EXTRACTOR is None:
        _WORKER_EXTRACTOR = make_pyradiomics_extractor()
        try:
            import logging
            import radiomics
            radiomics.setVerbosity(logging.ERROR)
            import SimpleITK as sitk
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        except Exception:
            pass
    roi = Path(roi_dir) / f"{private_patient_token(patient_id)}.private.npz"
    result, names = extract_patient_radiomics(
        entry,
        roi,
        output_dir,
        _WORKER_EXTRACTOR,
        expected_feature_names=None,
        overwrite=overwrite,
    )
    return patient_id, result, names


def main() -> None:
    args = parse_args()
    if args.radiomics_python is not None and not args.worker:
        command = [str(args.radiomics_python), str(Path(__file__).resolve()), "--worker", "--roi-dir", str(args.roi_dir), "--output-dir", str(args.output_dir)]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        command.extend(("--workers", str(args.workers)))
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
        return
    RepresentationReadSentinel().install()
    roi_gate_path = ROOT / "metrics/roi_feasibility.json"
    if not roi_gate_path.is_file() or json.loads(roi_gate_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit("formal radiomics extraction requires a passing V2 ROI feasibility gate")
    lock_path = ROOT / "environment/radiomics_environment_lock.json"
    if not lock_path.is_file():
        raise SystemExit("formal radiomics extraction requires radiomics_environment_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("python_major_minor") != "3.9" or lock.get("pyradiomics") != "3.1.0":
        raise SystemExit("radiomics environment lock is invalid")
    entries = load_c1b_manifest()
    patient_ids = tuple(sorted(ftv_wide()["patient_id"].astype(str)))
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]
    observed: dict[str, tuple[dict, tuple[str, ...] | None]] = {}
    tasks = [
        (patient_id, entries[patient_id], args.roi_dir, args.output_dir, args.overwrite)
        for patient_id in patient_ids
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(_worker_task, task): task[0] for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            patient_id, result, names = future.result()
            observed[patient_id] = (result, names)
            print({"patient": index, "total": len(patient_ids), "status": result["status"]}, flush=True)
    name_sets = {
        names for _, names in observed.values() if names is not None
    }
    if len(name_sets) != 1:
        raise RuntimeError("parallel extraction produced inconsistent feature-name contracts")
    names = next(iter(name_sets))
    hashes = [observed[patient_id][0]["sha256"] for patient_id in patient_ids]
    if args.limit is None:
        if names is None or len(hashes) != 375:
            raise RuntimeError("formal radiomics extraction is incomplete")
        payload = {
            "schema_version": 1, "status": "COMPLETE", "patients": 375,
            "features": len(names), "feature_list_sha256": canonical_sha256(names),
            "ordered_private_hashes_sha256": canonical_sha256(hashes),
            "environment_lock_sha256": file_sha256(lock_path),
            "outcome_fields_read": [], "clinical_fields_read": [],
        }
        atomic_json(ROOT / "manifests/radiomics_raw_complete.json", payload)
        print(payload)


if __name__ == "__main__":
    main()
