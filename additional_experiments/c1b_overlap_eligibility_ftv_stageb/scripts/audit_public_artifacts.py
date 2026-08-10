#!/usr/bin/env python3
"""Fail closed when a public experiment artifact leaks private identifiers.

Files whose path contains a ``.private`` component are deliberately excluded:
they are the preregistered identifier-bearing evidence boundary.  Everything
else in the public metric/manifest/report/config roots, the plan, and Stage-A
sentinels is scanned.  Findings contain only file names and counts, never the
matched text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = ("configs", "manifests", "metrics", "reports")
ROOT_PUBLIC_FILES = ("EXPERIMENT_PLAN.md", "STAGE_A_GO.json", "STAGE_A_NO_GO.json")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}

# The patterns intentionally prefer false positives to an identifier leak.
# Aggregate cohort phrases such as "I-SPY2 patients" do not match because an
# identifier must end in at least four digits after a delimiter.
PATTERNS = {
    "absolute_workspace_path": re.compile(
        r"(?:"
        r"/(?:data|home|mnt|opt|srv|scratch|gpfs|tmp|var|workspace|project)/"
        r"[^\s`\"']+"
        r"|[A-Za-z]:\\[^\s`\"']+"
        r"|\\\\[^\\\s`\"']+\\[^\s`\"']+"
        r")"
    ),
    "ispy_patient_identifier": re.compile(
        r"\b(?:ACRIN[-_ ]?6698[-_]|I[-_]?SPY[12]?[-_])\d{4,}\b",
        re.IGNORECASE,
    ),
    "dicom_uid": re.compile(r"\b\d{1,3}(?:\.\d+){5,}\b"),
    "json_identifier_value": re.compile(
        r'"(?:patient_id|patient_token|subject_id|subject_token|pid|mrn)"'
        r'\s*:\s*(?:"[^\"]+"|\d+)',
        re.IGNORECASE,
    ),
    "csv_identifier_column": re.compile(
        r"(?:^|,)(?:patient_id|patient_token|subject_id|subject_token|pid|mrn)"
        r"(?:,|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "csv_sensitive_path_column": re.compile(
        r"(?:^|,)(?:cache_path|source_path|dce_nifti|ftv_mask_nifti|"
        r"resolved_dce_nifti|raw_dce_series_json)(?:,|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _is_private(relative: Path) -> bool:
    return any("private" in component.lower() for component in relative.parts)


def public_paths(*, root: Path = ROOT) -> list[Path]:
    """Return every public text artifact in deterministic order."""

    paths: set[Path] = set()
    for name in PUBLIC_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root)
            if _is_private(relative) or path.name == "public_artifact_privacy_gate.json":
                continue
            paths.add(path)
    for name in ROOT_PUBLIC_FILES:
        path = root / name
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=lambda value: value.relative_to(root).as_posix())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_identifiers(root: Path) -> set[str]:
    """Collect identifier values from private CSVs without exposing them."""

    values: set[str] = set()
    for name in PUBLIC_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*.csv"):
            if not _is_private(path.relative_to(root)):
                continue
            try:
                columns = pd.read_csv(path, nrows=0).columns
                identifier_columns = [
                    column
                    for column in columns
                    if str(column).lower()
                    in {
                        "patient_id",
                        "patient_token",
                        "subject_id",
                        "subject_token",
                        "pid",
                        "mrn",
                    }
                ]
                if not identifier_columns:
                    continue
                frame = pd.read_csv(path, usecols=identifier_columns, dtype=str)
            except (OSError, UnicodeError, ValueError):
                # Unreadable private evidence is handled by its producing gate;
                # it cannot make the public privacy scan pass by leaking text.
                continue
            for column in identifier_columns:
                values.update(
                    value
                    for value in frame[column].dropna().astype(str)
                    if len(value.strip()) >= 4
                )
    return values


def _private_permission_findings(root: Path) -> list[dict[str, object]]:
    """Require private CSVs and tokenized cache names to be owner-only."""

    candidates: set[Path] = set()
    for name in PUBLIC_ROOTS:
        base = root / name
        if base.is_dir():
            candidates.update(base.rglob("*.private.csv"))
    cache_root = root / "cache"
    cache_cohort = cache_root / "c1b_h"
    candidates.update(path for path in (cache_root, cache_cohort) if path.exists())
    if cache_cohort.is_dir():
        candidates.update(cache_cohort.glob("*.npz"))
    findings: list[dict[str, object]] = []
    for path in sorted(candidates):
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            findings.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "finding": "group_or_world_private_asset_permission",
                    "mode": f"{mode:04o}",
                }
            )
    return findings


def _scan_text(
    relative: str,
    payload: str,
    *,
    private_identifiers: set[str],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for name, pattern in PATTERNS.items():
        count = sum(1 for _match in pattern.finditer(payload))
        if count:
            findings.append(
                {"file": relative, "finding": name, "match_count": count}
            )
    if PATTERNS["ispy_patient_identifier"].search(Path(relative).name):
        findings.append(
            {"file": relative, "finding": "identifier_in_filename", "match_count": 1}
        )
    private_matches = sum(
        1 for identifier in private_identifiers if identifier and identifier in payload
    )
    if private_matches:
        findings.append(
            {
                "file": relative,
                "finding": "private_manifest_identifier",
                "match_count": private_matches,
            }
        )
    return findings


def scan_public_artifacts(
    *,
    root: Path = ROOT,
    virtual_text: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    """Return a current side-effect-free scan.

    ``virtual_text`` lets the Stage-A finalizer include not-yet-written output
    in the same fail-closed decision.  A virtual path replaces a live path of
    the same relative name and is held only in memory; ``None`` represents a
    file that will be removed during the same atomic closure.
    """

    virtual = dict(virtual_text or {})
    private_identifiers = _private_identifiers(root)
    inventory: dict[str, tuple[str, str]] = {}
    for path in public_paths(root=root):
        relative = path.relative_to(root).as_posix()
        if relative in virtual:
            continue
        payload = path.read_text(encoding="utf-8", errors="strict")
        inventory[relative] = (payload, sha256(path))
    for relative, payload in virtual.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("virtual public-artifact paths must be relative")
        if _is_private(candidate):
            raise ValueError("virtual public artifact cannot use a private path")
        if payload is None:
            inventory.pop(candidate.as_posix(), None)
            continue
        inventory[candidate.as_posix()] = (
            str(payload),
            sha256_bytes(str(payload).encode("utf-8")),
        )

    findings: list[dict[str, object]] = []
    for relative in sorted(inventory):
        findings.extend(
            _scan_text(
                relative,
                inventory[relative][0],
                private_identifiers=private_identifiers,
            )
        )
    stale_debug = sorted(
        relative
        for relative in inventory
        if "_smoke_" in Path(relative).name.lower()
        or "_limited_" in Path(relative).name.lower()
    )
    identifier_findings = [
        item for item in findings if item["finding"] != "absolute_workspace_path"
    ]
    permission_findings = _private_permission_findings(root)
    status = (
        "PASS"
        if not findings and not stale_debug and not permission_findings
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "scanned_public_text_artifacts": len(inventory),
        "scanned_files_sha256": {
            relative: inventory[relative][1] for relative in sorted(inventory)
        },
        "identifier_or_path_findings": findings,
        "stale_smoke_or_limited_public_artifacts": stale_debug,
        "contains_patient_identifiers": bool(identifier_findings),
        "contains_sensitive_identifiers_or_paths": bool(findings),
        "private_identifier_values_checked": len(private_identifiers),
        "private_permission_findings": permission_findings,
    }


def atomic_text(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    payload = scan_public_artifacts()
    atomic_text(
        ROOT / "metrics/public_artifact_privacy_gate.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["status"] != "PASS":
        raise SystemExit("public artifact privacy gate failed")


if __name__ == "__main__":
    main()
