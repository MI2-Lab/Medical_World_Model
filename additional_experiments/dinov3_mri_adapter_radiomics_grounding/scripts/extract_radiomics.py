#!/usr/bin/env python3
"""Extract seven-channel appearance/kinetic radiomics in the locked side env."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.radiomics_python is not None and not args.worker:
        command = [str(args.radiomics_python), str(Path(__file__).resolve()), "--worker", "--roi-dir", str(args.roi_dir), "--output-dir", str(args.output_dir)]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
        return
    RepresentationReadSentinel().install()
    lock_path = ROOT / "environment/radiomics_environment_lock.json"
    if not lock_path.is_file():
        raise SystemExit("formal radiomics extraction requires radiomics_environment_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("python_major_minor") != "3.9" or lock.get("pyradiomics") != "3.1.0":
        raise SystemExit("radiomics environment lock is invalid")
    extractor = make_pyradiomics_extractor()
    entries = load_c1b_manifest()
    patient_ids = tuple(sorted(ftv_wide()["patient_id"].astype(str)))
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]
    names = None
    hashes: list[str] = []
    for index, patient_id in enumerate(patient_ids, start=1):
        roi = args.roi_dir / f"{private_patient_token(patient_id)}.private.npz"
        result, observed_names = extract_patient_radiomics(
            entries[patient_id], roi, args.output_dir, extractor,
            expected_feature_names=names, overwrite=args.overwrite,
        )
        if observed_names is not None:
            names = observed_names
        if result["sha256"]:
            hashes.append(result["sha256"])
        print({"patient": index, "total": len(patient_ids), "status": result["status"]}, flush=True)
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
