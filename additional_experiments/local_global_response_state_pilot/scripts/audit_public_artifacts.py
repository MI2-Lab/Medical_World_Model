#!/usr/bin/env python3
"""Fail closed if public pilot artifacts expose private identifiers or paths."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

sys.dont_write_bytecode = True

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
scripts_value = str(SCRIPTS_ROOT.resolve())
while scripts_value in sys.path:
    sys.path.remove(scripts_value)
sys.path.insert(0, scripts_value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402

OUTPUT = EXPERIMENT_ROOT / "metrics" / "public_artifact_privacy_gate.json"
IDENTIFIER_SOURCES = (
    REPO_ROOT
    / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/technical_eligibility_patients.private.csv",
    REPO_ROOT
    / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/stage_b_c1b_cache.private.csv",
    REPO_ROOT
    / "additional_experiments/c1b_model_ready_ftv_sanity/manifests/ispy1_base_eligibility_patients.private.csv",
)
IDENTIFIER_COLUMNS = ("patient_id", "patient_token")
TEXT_SUFFIXES = {
    ".md",
    ".json",
    ".csv",
    ".txt",
    ".py",
    ".yaml",
    ".yml",
    ".svg",
}
BINARY_SUFFIXES = {".png"}
PUBLIC_ROOTS = (
    "configs",
    "scripts",
    "src",
    "tests",
    "metrics",
    "figures",
    "reports",
)
PRIVATE_ROOTS = ("checkpoints", "features", "predictions", "logs")
PRIVATE_NAMED_PUBLIC_ROOTS = ("metrics", "reports")
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:file://|(?<![A-Za-z0-9_.-])/(?:data|home|tmp|mnt|scratch|workspace)/|"
    r"[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/])"
)
ABSOLUTE_PATH_BYTES_PATTERN = re.compile(
    rb"(?:file://|(?<![A-Za-z0-9_.-])/(?:data|home|tmp|mnt|scratch|workspace)/|"
    rb"[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/])"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifiers() -> set[str]:
    values: set[str] = set()
    for path in IDENTIFIER_SOURCES:
        frame = pd.read_csv(path, dtype="string")
        columns = [column for column in IDENTIFIER_COLUMNS if column in frame]
        if not columns:
            raise ValueError(f"private identifier source has no denylist column: {path}")
        for column in columns:
            values.update(str(value) for value in frame[column].dropna())
    return {value for value in values if len(value) >= 5}


def _root_placeholder(root: Path) -> Path:
    return root / ".gitkeep"


def _private_named(path: Path) -> bool:
    return "private" in path.name.lower()


def _derived_bytecode(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix.lower() in {
        ".pyc",
        ".pyo",
        ".pyd",
    }


def public_artifacts() -> list[Path]:
    paths = [
        path.resolve()
        for path in (
            EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md",
            EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json",
        )
        if path.is_file()
    ]
    for name in PUBLIC_ROOTS:
        root = EXPERIMENT_ROOT / name
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.resolve() != OUTPUT.resolve()
            and path != _root_placeholder(root)
            and not _derived_bytecode(path)
            and not (name in PRIVATE_NAMED_PUBLIC_ROOTS and _private_named(path))
        )
    return sorted(set(path.resolve() for path in paths))


def scan_public_artifacts(
    paths: list[Path], private_ids: set[str]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    identifier_findings: list[dict[str, object]] = []
    absolute_path_findings: list[dict[str, object]] = []
    unsupported_findings: list[dict[str, object]] = []
    encoded_ids = {value: value.encode("utf-8") for value in private_ids}
    for path in paths:
        artifact = str(path.relative_to(REPO_ROOT))
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES | BINARY_SUFFIXES:
            unsupported_findings.append(
                {"artifact": artifact, "reason": f"unsupported suffix {suffix!r}"}
            )
            continue
        content = path.read_bytes()
        matches = sorted(
            value for value, encoded in encoded_ids.items() if encoded in content
        )
        if matches:
            identifier_findings.append(
                {"artifact": artifact, "identifier_count": len(matches)}
            )
        if suffix in TEXT_SUFFIXES:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                unsupported_findings.append(
                    {"artifact": artifact, "reason": "invalid UTF-8 text"}
                )
                continue
            # Source code necessarily contains the deny-pattern literals used
            # by this validator.  Public data/report artifacts may not.
            contains_absolute_path = (
                suffix != ".py" and ABSOLUTE_PATH_PATTERN.search(text) is not None
            )
        else:
            # PNG metadata chunks remain byte-searchable.  Only the explicitly
            # supported preregistered image format reaches this branch.
            contains_absolute_path = (
                ABSOLUTE_PATH_BYTES_PATTERN.search(content) is not None
            )
        if contains_absolute_path:
            absolute_path_findings.append({"artifact": artifact})
    return identifier_findings, absolute_path_findings, unsupported_findings


def private_permission_findings() -> list[str]:
    findings: list[str] = []
    for name in PRIVATE_ROOTS:
        root = EXPERIMENT_ROOT / name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path == _root_placeholder(root):
                continue
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                findings.append(str(path.relative_to(REPO_ROOT)))
    for name in PRIVATE_NAMED_PUBLIC_ROOTS:
        root = EXPERIMENT_ROOT / name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _private_named(path):
                continue
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                findings.append(str(path.relative_to(REPO_ROOT)))
    return sorted(findings)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary, 0o644)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    preregistration = verify_preregistration()
    private_ids = identifiers()
    scanned = public_artifacts()
    (
        identifier_findings,
        absolute_path_findings,
        unsupported_findings,
    ) = scan_public_artifacts(scanned, private_ids)
    permission_findings = private_permission_findings()
    passed = not any(
        (
            identifier_findings,
            absolute_path_findings,
            unsupported_findings,
            permission_findings,
        )
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "scanned_public_artifacts": len(scanned),
        "private_identifier_values_checked": len(private_ids),
        "identifier_findings": identifier_findings,
        "absolute_path_findings": absolute_path_findings,
        "unsupported_public_artifact_findings": unsupported_findings,
        "private_permission_findings": permission_findings,
        "scanner_sha256": file_sha256(Path(__file__)),
    }
    atomic_json(OUTPUT, payload)
    if not passed:
        raise RuntimeError("public-artifact privacy gate failed; see metrics JSON")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
