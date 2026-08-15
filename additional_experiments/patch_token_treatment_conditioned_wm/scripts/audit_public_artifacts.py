#!/usr/bin/env python3
"""Fail closed if the proposed Git delivery contains private experiment data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
EXPERIMENT_RELATIVE = EXPERIMENT_ROOT.relative_to(REPO_ROOT).as_posix()
PRIVATE_DIRECTORIES = {"checkpoints", "features", "predictions", "logs"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".nii", ".nii.gz", ".npz", ".npy"}
PATIENT_COLUMNS = {
    "patient_id",
    "clinical_patient_id",
    "raw_patient_id",
    "predicted_probability",
    "y_true",
    "y_pred",
}
PERSONAL_PATH_RE = re.compile(r"/(?:data/mi2-interns|home)/[^/\s\"']+")
PATIENT_VALUE_RE = re.compile(r"(?:ISPY[12][-_]\d|ACRIN-6698-\d)", re.IGNORECASE)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}


def _git_paths() -> list[str]:
    commands = (
        ["git", "diff", "--name-only", "--cached"],
        ["git", "diff", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    output: set[str] = set()
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        output.update(
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        )
    return sorted(path for path in output if path.startswith(f"{EXPERIMENT_RELATIVE}/"))


def _suffix(path: Path) -> str:
    return ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix.lower()


def audit(paths: Iterable[str] | None = None) -> dict[str, object]:
    candidates = _git_paths() if paths is None else sorted(set(paths))
    violations: list[str] = []
    inspected = 0
    for relative in candidates:
        path = (REPO_ROOT / relative).resolve()
        try:
            local = path.relative_to(EXPERIMENT_ROOT.resolve())
        except ValueError:
            violations.append(f"path escapes experiment root: {relative}")
            continue
        if not path.is_file():
            continue
        inspected += 1
        parts = set(local.parts)
        if parts & PRIVATE_DIRECTORIES and path.name != ".gitkeep":
            violations.append(
                f"private directory artifact proposed for Git: {relative}"
            )
        if "private" in path.name.lower():
            violations.append(f"private-named artifact proposed for Git: {relative}")
        if _suffix(path) in FORBIDDEN_SUFFIXES:
            violations.append(
                f"forbidden binary/data type proposed for Git: {relative}"
            )
        if path.suffix.lower() == ".csv":
            header = (
                path.open("r", encoding="utf-8").readline().strip().lower().split(",")
            )
            overlap = sorted(PATIENT_COLUMNS.intersection(header))
            if overlap:
                violations.append(
                    f"patient/prediction columns in public CSV {relative}: {overlap}"
                )
        if path.suffix.lower() in TEXT_SUFFIXES:
            content = path.read_text(encoding="utf-8")
            if PERSONAL_PATH_RE.search(content):
                violations.append(f"personal absolute path in public file: {relative}")
            if PATIENT_VALUE_RE.search(content):
                violations.append(
                    f"probable patient identifier in public file: {relative}"
                )
    outside = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    outside_paths = [
        row[3:]
        for row in outside
        if len(row) > 3 and not row[3:].startswith(f"{EXPERIMENT_RELATIVE}/")
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if not violations else "FAIL",
        "experiment_paths_inspected": inspected,
        "violations": violations,
        "unrelated_worktree_changes_present": bool(outside_paths),
        "unrelated_paths_will_not_be_staged": True,
        "raw_mri_committed": False,
        "patient_identifiers_committed": False,
        "patient_predictions_committed": False,
        "checkpoints_committed": False,
    }
    if violations:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
