#!/usr/bin/env python3
"""Fail closed if a public result leaks identifiers, UIDs, or absolute paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = ("metrics", "manifests", "reports")
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".yaml", ".yml"}
PATTERNS = {
    "absolute_workspace_path": re.compile(
        r"(?:"
        # Common single-component roots plus any multi-component POSIX path.
        # The negative lookbehind prevents treating the path part of an HTTP
        # URL as a local absolute path.
        r"/(?:data|home|mnt|opt|srv|scratch|gpfs|tmp|var|workspace|project)/[^\s`\"']+"
        r"|(?<![A-Za-z0-9/:])/(?:[^/\s`\"']+/)+[^/\s`\"']+"
        r"|[A-Za-z]:\\[^\s`\"']+"
        r"|\\\\[^\\\s`\"']+\\[^\s`\"']+"
        r")"
    ),
    "ispy_patient_identifier": re.compile(
        # I-SPY1 identifiers in this cohort end in four digits, whereas I-SPY2
        # and ACRIN identifiers are longer.  Requiring the delimiter before the
        # terminal number avoids mistaking aggregate prose such as "I-SPY1 has
        # 1,500 visits" for an identifier.
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
        r'(?:^|,)(?:patient_id|patient_token|subject_id|subject_token|pid|mrn)(?:,|$)',
        re.IGNORECASE | re.MULTILINE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_text(path: Path, content: str, overwrite: bool) -> None:
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


def public_paths() -> list[Path]:
    paths: list[Path] = []
    for name in PUBLIC_ROOTS:
        for path in (ROOT / name).rglob("*"):
            relative = path.relative_to(ROOT).as_posix().lower()
            if (
                path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and "private" not in relative
                and path.name != "public_artifact_privacy_gate.json"
            ):
                paths.append(path)
    return sorted(paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_public_artifacts() -> dict[str, object]:
    """Return a current, side-effect-free privacy scan.

    The file inventory hashes let a downstream hard gate distinguish a current
    scan from an earlier PASS produced before later reports were written.
    """

    findings: list[dict[str, object]] = []
    paths = public_paths()
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="strict")
        relative = path.relative_to(ROOT).as_posix()
        for name, pattern in PATTERNS.items():
            matches = list(pattern.finditer(text))
            if matches:
                findings.append(
                    {
                        "file": relative,
                        "finding": name,
                        "match_count": len(matches),
                    }
                )
        # An identifier in a filename is a leak even when the file body is an
        # otherwise anonymous aggregate.
        identifier_pattern = PATTERNS["ispy_patient_identifier"]
        if identifier_pattern.search(path.name):
            findings.append(
                {
                    "file": relative,
                    "finding": "identifier_in_filename",
                    "match_count": 1,
                }
            )
    stale_debug = [
        path.relative_to(ROOT).as_posix()
        for path in paths
        if "_smoke_" in path.name or "_limited_" in path.name
    ]
    status = "PASS" if not findings and not stale_debug else "FAIL"
    identifier_findings = [
        item
        for item in findings
        if item["finding"] != "absolute_workspace_path"
    ]
    return {
        "schema_version": 2,
        "status": status,
        "scanned_public_text_artifacts": len(paths),
        "scanned_files_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in paths
        },
        "identifier_or_path_findings": findings,
        "stale_smoke_or_limited_public_artifacts": stale_debug,
        "contains_patient_identifiers": bool(identifier_findings),
        "contains_sensitive_identifiers_or_paths": bool(findings),
    }


def main() -> None:
    args = parse_args()
    payload = scan_public_artifacts()
    output = ROOT / "metrics/public_artifact_privacy_gate.json"
    atomic_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n", args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("public artifact privacy gate failed")


if __name__ == "__main__":
    main()
