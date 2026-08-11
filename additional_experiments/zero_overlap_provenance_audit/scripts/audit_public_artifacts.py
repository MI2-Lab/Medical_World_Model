#!/usr/bin/env python3
"""Fail-closed privacy audit for public zero-overlap audit artifacts.

The scanner never includes matched text in its result. Explicitly private
paths are not opened. Public Markdown, CSV, JSON, and PNG files are checked for
identifiers, DICOM UIDs, absolute paths, private filenames, raw coordinates,
and aliases other than ``CASE_ZERO_OVERLAP_001``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Iterable
import zlib


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRECTORIES = ("metrics", "manifests", "reports", "figures")
TOP_LEVEL_PUBLIC_FILES = (
    "EXPERIMENT_PLAN.md",
    "AUDIT_REPAIRABLE.json",
    "AUDIT_NOT_REPAIRABLE.json",
    "AUDIT_AMBIGUOUS.json",
)
PUBLIC_SUFFIXES = {".md", ".csv", ".json", ".png"}
IGNORED_PUBLIC_FILENAMES = {".gitkeep"}
SELF_OUTPUT_RELATIVE = "metrics/public_artifact_privacy_gate.json"
ALLOWED_CASE_ALIASES = frozenset({"CASE_ZERO_OVERLAP_001"})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_METADATA_BYTES = 1_048_576

URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
ABSOLUTE_PATH_RE = re.compile(
    r"(?:"
    r"file://[^\s`\"'<>]+"
    r"|(?<![A-Za-z0-9/:])/(?:[^/\s`\"'<>]+/)+[^/\s`\"'<>]+"
    r"|(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\s`\"'<>]+"
    r")",
    re.IGNORECASE,
)
DICOM_UID_RE = re.compile(
    r"(?<![\d.])(?:"
    r"2\.25\.(?:0|[1-9]\d{15,})"
    r"|(?:0|1|2)(?:\.(?:0|[1-9]\d*)){4,}"
    r")(?![\d.])"
)
COHORT_PATIENT_ID_RE = re.compile(
    r"\b(?:ACRIN(?:[-_ ]?6698)?|I[-_ ]?SPY[12]?|ISPY[12]?)"
    r"[-_ ]?\d{3,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
GENERIC_PATIENT_ID_RE = re.compile(
    r"\b(?:MRN|PATIENT|SUBJECT|PARTICIPANT)\s*"
    r"(?:[-_ ]?(?:ID|IDENTIFIER|TOKEN))?[-_ :=#]*[A-Z]*\d{3,}"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
IDENTIFIER_FIELD_RE = re.compile(
    r"[\"'](?:patient(?:_id|_identifier|_token)?|subject(?:_id|_identifier|_token)?|"
    r"participant(?:_id|_identifier|_token)?|pid|mrn)[\"']\s*:",
    re.IGNORECASE,
)
PRIVATE_FILENAME_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]*private[A-Za-z0-9_.-]*"
    r"\.(?:csv|json|md|txt|ya?ml|png|dcm|ima|nii(?:\.gz)?|npy|npz)"
    r"|(?<![A-Za-z0-9_.-])private[\\/][^\s`\"'<>]+"
    r")",
    re.IGNORECASE,
)
CASE_ALIAS_RE = re.compile(r"\bCASE(?:[_-][A-Z0-9]+){1,}\b", re.IGNORECASE)

NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?"
COORDINATE_LABEL = (
    r"(?:ImagePositionPatient|IPP|SliceLocation|source_center_(?:ras|lps)|"
    r"volume_center_(?:ras|lps)|physical_center_(?:ras|lps)|"
    r"origin_(?:ras|lps)|bbox_(?:min|max)(?:_(?:ras|lps))?|"
    r"source_bounding_box|affine(?:_(?:ras|lps))?)"
)
RAW_COORDINATE_VECTOR_RE = re.compile(
    rf"\b{COORDINATE_LABEL}\b\s*[\"']?\s*[:=]\s*"
    rf"[\[(]?\s*{NUMBER}\s*[,;]\s*{NUMBER}\s*[,;]\s*{NUMBER}",
    re.IGNORECASE,
)
RAW_XYZ_RE = re.compile(
    rf"\bx\s*[:=]\s*{NUMBER}\s*[,;]\s*y\s*[:=]\s*{NUMBER}"
    rf"\s*[,;]\s*z\s*[:=]\s*{NUMBER}",
    re.IGNORECASE,
)

IDENTIFIER_COLUMNS = frozenset(
    {
        "patient",
        "patientid",
        "patientidentifier",
        "patienttoken",
        "subject",
        "subjectid",
        "subjectidentifier",
        "subjecttoken",
        "participant",
        "participantid",
        "participantidentifier",
        "participanttoken",
        "pid",
        "mrn",
    }
)
RAW_COORDINATE_COLUMNS = frozenset(
    {
        "imagepositionpatient",
        "ipp",
        "slicelocation",
        "sourcecenterras",
        "sourcecenterlps",
        "volumecenterras",
        "volumecenterlps",
        "physicalcenterras",
        "physicalcenterlps",
        "originras",
        "originlps",
        "bboxmin",
        "bboxmax",
        "bboxminras",
        "bboxmaxras",
        "bboxminlps",
        "bboxmaxlps",
        "sourceboundingbox",
        "affine",
        "affineras",
        "affinelps",
    }
)
ALLOWED_RELATIVE_COORDINATE_COLUMNS = frozenset(
    {"sourcecenterrast0relativemm"}
)


def _normalise_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _has_private_component(relative: Path) -> bool:
    """Classify explicit private paths without opening their contents."""

    for part in relative.parts:
        tokens = [token for token in re.split(r"[._-]+", part.casefold()) if token]
        if "private" in tokens:
            return True
    return False


def public_paths(root: Path | str = ROOT) -> list[Path]:
    """Return public artifact candidates; explicitly private paths are omitted."""

    root = Path(root)
    paths: set[Path] = set()
    for name in TOP_LEVEL_PUBLIC_FILES:
        path = root / name
        if path.exists() or path.is_symlink():
            paths.add(path)
    for directory_name in PUBLIC_DIRECTORIES:
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not (path.is_file() or path.is_symlink()):
                continue
            relative = path.relative_to(root)
            if relative.as_posix() == SELF_OUTPUT_RELATIVE:
                continue
            if path.name in IGNORED_PUBLIC_FILENAMES:
                continue
            if _has_private_component(relative):
                continue
            paths.add(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _mask_public_urls(text: str) -> str:
    return URL_RE.sub(lambda match: " " * len(match.group(0)), text)


def _text_pattern_findings(text: str) -> Counter[str]:
    """Scan text while exempting only URL syntax from absolute-path parsing."""

    path_visible = _mask_public_urls(text)
    findings: Counter[str] = Counter()
    patterns = {
        "absolute_path": (ABSOLUTE_PATH_RE, path_visible),
        "dicom_uid": (DICOM_UID_RE, text),
        "patient_identifier": (COHORT_PATIENT_ID_RE, text),
        "generic_patient_identifier": (GENERIC_PATIENT_ID_RE, text),
        "identifier_field": (IDENTIFIER_FIELD_RE, text),
        "private_filename": (PRIVATE_FILENAME_RE, text),
        "raw_coordinate_vector": (RAW_COORDINATE_VECTOR_RE, text),
        "raw_xyz_coordinates": (RAW_XYZ_RE, text),
    }
    for name, (pattern, candidate) in patterns.items():
        findings[name] += sum(1 for _ in pattern.finditer(candidate))

    for match in CASE_ALIAS_RE.finditer(text):
        alias = match.group(0)
        # ``case_alias``/``case_aliases`` are public schema labels, not values.
        if alias.casefold() in {"case_alias", "case_aliases"}:
            continue
        if alias not in ALLOWED_CASE_ALIASES:
            findings["unexpected_case_alias"] += 1
    return +findings


def _json_semantic_findings(value: Any) -> Counter[str]:
    findings: Counter[str] = Counter()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            normalised_keys = {_normalise_field(str(key)) for key in item}
            if {"x", "y", "z"}.issubset(normalised_keys):
                findings["raw_coordinate_fields"] += 1
            for key, child in item.items():
                normalised = _normalise_field(str(key))
                if normalised in IDENTIFIER_COLUMNS:
                    findings["identifier_field"] += 1
                if (
                    normalised in RAW_COORDINATE_COLUMNS
                    and normalised not in ALLOWED_RELATIVE_COORDINATE_COLUMNS
                ):
                    findings["raw_coordinate_field"] += 1
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return +findings


def _csv_semantic_findings(text: str) -> Counter[str]:
    findings: Counter[str] = Counter()
    reader = csv.reader(io.StringIO(text), strict=True)
    rows = list(reader)
    if not rows:
        return findings
    header = [_normalise_field(value) for value in rows[0]]
    findings["identifier_field"] += sum(
        value in IDENTIFIER_COLUMNS for value in header
    )
    findings["raw_coordinate_field"] += sum(
        value in RAW_COORDINATE_COLUMNS
        and value not in ALLOWED_RELATIVE_COORDINATE_COLUMNS
        for value in header
    )
    if {"x", "y", "z"}.issubset(set(header)):
        findings["raw_coordinate_fields"] += 1
    return +findings


def _bounded_zlib_decompress(data: bytes) -> bytes:
    decoder = zlib.decompressobj()
    output = decoder.decompress(data, MAX_PNG_METADATA_BYTES + 1)
    if len(output) > MAX_PNG_METADATA_BYTES or decoder.unconsumed_tail:
        raise ValueError("PNG metadata exceeds the audit limit")
    output += decoder.flush(MAX_PNG_METADATA_BYTES + 1 - len(output))
    if len(output) > MAX_PNG_METADATA_BYTES or not decoder.eof:
        raise ValueError("invalid or oversized compressed PNG metadata")
    return output


def _printable_byte_strings(data: bytes) -> str:
    strings = re.findall(rb"[\x20-\x7e]{4,}", data)
    return "\n".join(item.decode("ascii") for item in strings)


def _png_text_and_structure_findings(raw: bytes) -> tuple[str, Counter[str]]:
    """Extract visible/hidden PNG strings and validate the chunk envelope."""

    findings: Counter[str] = Counter()
    # Do not regex-scan the compressed IDAT byte stream as if it were text.
    # Arbitrary DEFLATE bytes frequently contain short printable runs such as
    # ``/x/y`` by chance, producing false path findings.  Text-bearing PNG
    # chunks, unknown ancillary chunks, EXIF, and trailing bytes are all
    # extracted explicitly below; those are the locations where hidden clear
    # text can exist without being pixel-compression noise.
    extracted: list[str] = []
    if not raw.startswith(PNG_SIGNATURE):
        findings["malformed_png"] += 1
        return "\n".join(extracted), findings

    offset = len(PNG_SIGNATURE)
    chunk_index = 0
    saw_ihdr = False
    saw_iend = False
    while offset < len(raw):
        if len(raw) - offset < 12:
            findings["malformed_png"] += 1
            break
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(raw):
            findings["malformed_png"] += 1
            break
        chunk_data = raw[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", raw[offset + 8 + length : end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            findings["png_crc_error"] += 1
        if not re.fullmatch(rb"[A-Za-z]{4}", chunk_type):
            findings["malformed_png"] += 1

        if chunk_index == 0:
            if chunk_type != b"IHDR" or length != 13:
                findings["malformed_png"] += 1
            else:
                saw_ihdr = True

        try:
            if chunk_type == b"tEXt":
                keyword, separator, payload = chunk_data.partition(b"\x00")
                if not separator:
                    raise ValueError("invalid tEXt")
                extracted.append((keyword + b" " + payload).decode("latin-1"))
            elif chunk_type == b"zTXt":
                keyword, separator, remainder = chunk_data.partition(b"\x00")
                if not separator or len(remainder) < 2 or remainder[0] != 0:
                    raise ValueError("invalid zTXt")
                payload = _bounded_zlib_decompress(remainder[1:])
                extracted.append((keyword + b" " + payload).decode("latin-1"))
            elif chunk_type == b"iTXt":
                keyword, separator, remainder = chunk_data.partition(b"\x00")
                if not separator or len(remainder) < 2:
                    raise ValueError("invalid iTXt")
                compressed, method = remainder[0], remainder[1]
                remainder = remainder[2:]
                language, separator, remainder = remainder.partition(b"\x00")
                if not separator:
                    raise ValueError("invalid iTXt language")
                translated, separator, payload = remainder.partition(b"\x00")
                if not separator or compressed not in (0, 1) or method != 0:
                    raise ValueError("invalid iTXt payload")
                if compressed:
                    payload = _bounded_zlib_decompress(payload)
                extracted.append(keyword.decode("latin-1"))
                extracted.append(language.decode("ascii"))
                extracted.append(translated.decode("utf-8"))
                extracted.append(payload.decode("utf-8"))
            elif chunk_type == b"eXIf":
                extracted.append(_printable_byte_strings(chunk_data))
            elif chunk_type not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}:
                # Unknown ancillary chunks can carry clear-text application data.
                extracted.append(_printable_byte_strings(chunk_data))
        except (UnicodeDecodeError, ValueError, zlib.error):
            findings["malformed_png_metadata"] += 1

        offset = end
        chunk_index += 1
        if chunk_type == b"IEND":
            if length != 0:
                findings["malformed_png"] += 1
            saw_iend = True
            if offset != len(raw):
                findings["png_trailing_bytes"] += 1
                extracted.append(_printable_byte_strings(raw[offset:]))
            break

    if not saw_ihdr or not saw_iend:
        findings["malformed_png"] += 1
    return "\n".join(extracted), +findings


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_file_label(relative: str) -> str:
    """Avoid repeating an identifier-bearing filename in stdout/gate JSON."""

    if _text_pattern_findings(relative):
        token = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        return f"REDACTED_PUBLIC_PATH_{token}"
    return relative


def _merge(target: Counter[str], source: Counter[str]) -> None:
    for name, count in source.items():
        target[name] += count


def _scan_artifact(path: Path, relative: str) -> tuple[Counter[str], str | None]:
    findings = _text_pattern_findings(relative)
    if path.is_symlink():
        findings["public_symlink"] += 1
        return findings, None
    if path.suffix.casefold() not in PUBLIC_SUFFIXES:
        findings["unsupported_public_artifact_type"] += 1
        return findings, None

    try:
        raw = path.read_bytes()
    except OSError:
        findings["unreadable_public_artifact"] += 1
        return findings, None
    digest = _sha256_bytes(raw)

    if path.suffix.casefold() == ".png":
        text, png_findings = _png_text_and_structure_findings(raw)
        _merge(findings, png_findings)
        _merge(findings, _text_pattern_findings(text))
        return +findings, digest

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        findings["invalid_utf8"] += 1
        return +findings, digest
    _merge(findings, _text_pattern_findings(text))

    suffix = path.suffix.casefold()
    if suffix == ".json":
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            findings["malformed_json"] += 1
        else:
            _merge(findings, _json_semantic_findings(parsed))
    elif suffix == ".csv":
        try:
            _merge(findings, _csv_semantic_findings(text))
        except (csv.Error, UnicodeError):
            findings["malformed_csv"] += 1
    return +findings, digest


def scan_public_artifacts(root: Path | str = ROOT) -> dict[str, object]:
    """Return a side-effect-free, privacy-safe scan of current public files."""

    root = Path(root)
    paths = public_paths(root)
    findings: list[dict[str, object]] = []
    hashes: dict[str, str | None] = {}
    scanned_by_type: Counter[str] = Counter()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        safe_label = _safe_file_label(relative)
        suffix = path.suffix.casefold().lstrip(".") or "no_suffix"
        scanned_by_type[suffix] += 1
        file_findings, digest = _scan_artifact(path, relative)
        hashes[safe_label] = digest
        for name, count in sorted(file_findings.items()):
            if count:
                findings.append(
                    {"file": safe_label, "finding": name, "match_count": int(count)}
                )

    status = "PASS" if not findings else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "allowed_case_aliases": sorted(ALLOWED_CASE_ALIASES),
        "private_artifacts_ignored": True,
        "scanned_public_artifacts": len(paths),
        "scanned_by_type": dict(sorted(scanned_by_type.items())),
        "scanned_files_sha256": hashes,
        "privacy_findings": findings,
        # Compatibility name used by the preceding C1B hard gate.
        "identifier_or_path_findings": findings,
        "contains_sensitive_identifiers_or_paths": bool(findings),
    }


def _atomic_write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("privacy gate already exists; pass --overwrite")
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.root.is_dir():
        raise SystemExit("audit root is not a directory")
    payload = scan_public_artifacts(args.root)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not args.check_only:
        _atomic_write(
            args.root / "metrics" / "public_artifact_privacy_gate.json",
            serialized,
            args.overwrite,
        )
    print(serialized, end="")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
