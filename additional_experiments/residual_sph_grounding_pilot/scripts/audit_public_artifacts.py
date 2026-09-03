#!/usr/bin/env python3
"""Fail closed if the pilot's public delivery crosses its privacy boundary."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys


os.umask(0o077)
sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
OUTPUT = EXPERIMENT_ROOT / "metrics" / "public_artifact_privacy_gate.json"
PRIVATE_TARGET = (
    REPO_ROOT
    / "additional_experiments/radiomics_next_change/data_audit/"
    "radiomics_transition_targets_raw.csv"
)
PRIVATE_TARGET_SHA256 = "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"

PRIVATE_DIRECTORIES = {"checkpoints", "features", "predictions", "logs", "__pycache__"}
PRIVATE_SUFFIXES = {
    ".ckpt",
    ".dcm",
    ".nii",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".xls",
    ".xlsx",
}
PRIVATE_PATH_RE = re.compile(r"(?:^|[\s\"'=:(])(?:/(?:data|home|mnt)/|[A-Za-z]:[\\/]Users[\\/])")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_artifacts() -> list[Path]:
    """Return files eligible for the public commit, excluding runtime-private trees."""

    paths: list[Path] = []
    for path in EXPERIMENT_ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(EXPERIMENT_ROOT)
        if any(part in PRIVATE_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or ".private." in path.name:
            continue
        paths.append(path)
    return sorted(paths)


def _private_patient_ids() -> set[str]:
    if not PRIVATE_TARGET.is_file() or file_sha256(PRIVATE_TARGET) != PRIVATE_TARGET_SHA256:
        raise ValueError("private target table is absent or hash-mismatched")
    with PRIVATE_TARGET.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "patient_id" not in reader.fieldnames:
            raise ValueError("private target table lacks patient_id")
        values = {str(row["patient_id"]).strip() for row in reader}
    if len(values) != 375 or "" in values:
        raise ValueError("private target patient denylist drifted")
    return values


def scan_paths(
    paths: list[Path], patient_ids: set[str]
) -> dict[str, list[dict[str, str]]]:
    findings: dict[str, list[dict[str, str]]] = {
        "absolute_private_path_findings": [],
        "identifier_findings": [],
        "private_filename_findings": [],
        "restricted_extension_findings": [],
        "unsupported_binary_findings": [],
    }
    for path in paths:
        token = (
            path.relative_to(REPO_ROOT).as_posix()
            if path.is_relative_to(REPO_ROOT)
            else path.name
        )
        lower_name = path.name.lower()
        suffixes = "".join(path.suffixes[-2:]).lower()
        if ".private." in lower_name:
            findings["private_filename_findings"].append({"artifact": token})
        if path.suffix.lower() in PRIVATE_SUFFIXES or suffixes == ".nii.gz":
            findings["restricted_extension_findings"].append({"artifact": token})
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings["unsupported_binary_findings"].append({"artifact": token})
            continue
        if PRIVATE_PATH_RE.search(text):
            findings["absolute_private_path_findings"].append({"artifact": token})
        matched = sorted(patient_id for patient_id in patient_ids if patient_id in text)
        if matched:
            findings["identifier_findings"].append(
                {"artifact": token, "matched_identifier_count": str(len(matched))}
            )
    return findings


def main() -> None:
    artifacts = public_artifacts()
    findings = scan_paths(artifacts, _private_patient_ids())
    finding_count = sum(len(rows) for rows in findings.values())
    payload = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "status": "PASS" if finding_count == 0 else "FAIL",
        "scanned_public_artifact_count": len(artifacts),
        "finding_count": finding_count,
        **findings,
        "artifact_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): file_sha256(path) for path in artifacts
        },
        "private_patient_denylist_count": 375,
        "private_patient_identifiers_persisted": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUTPUT.chmod(0o644)
    print(json.dumps({"status": payload["status"], "finding_count": finding_count}))
    if finding_count:
        raise SystemExit("public artifact privacy audit failed")


if __name__ == "__main__":
    main()
