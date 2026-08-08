#!/usr/bin/env python3
"""Fail-closed audit of the public Stage-A artifact bundle.

The public surface is the set of tracked or non-ignored files under this
experiment.  Patient-level diagnostics may exist locally only when excluded by
the experiment's ignore policy.  The verifier writes nothing; it emits one
JSON result to stdout and returns non-zero on any violation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
MAX_PUBLIC_FILE_BYTES = 25 * 1024 * 1024
MIN_REPORT_BYTES = 100

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_REPORTS = (
    "reports/legacy_contract_review.md",
    "reports/physical_geometry_audit.md",
    "reports/dicom_geometry_repair_audit.md",
    "reports/containment_audit.md",
    "reports/image_quality_context_analysis.md",
    "reports/final_report.md",
)

REQUIRED_METRICS = (
    "metrics/physical_geometry_by_visit.csv",
    "metrics/physical_geometry_summary.csv",
    "metrics/grid_selection_basis.csv",
    "metrics/tensor_footprint_estimate.csv",
    "metrics/containment_by_contract_visit.csv",
    "metrics/containment_summary.csv",
    "metrics/large_ld_subgroups.csv",
    "metrics/ld_rank_sanity.csv",
    "metrics/ftv_retention_summary.csv",
    "metrics/morphology_readiness.csv",
    "metrics/context_summary.csv",
    "metrics/resampling_summary.csv",
    "metrics/temporal_consistency.csv",
    "metrics/image_quality_preview.csv",
    "metrics/dicom_geometry_repair_summary.csv",
    "metrics/dicom_geometry_repair_gate.json",
    "metrics/stage_a_gate.json",
    "metrics/input_recommendation.json",
    "metrics/stage_execution_status.csv",
)

REQUIRED_SUPPORT_FILES = (
    ".gitignore",
    "EXPERIMENT_PLAN.md",
    "configs/stage_a.json",
    "manifests/stage_a_provenance.json",
    "scripts/run_stage_a.py",
    "scripts/run_dicom_geometry_audit.py",
    "scripts/make_previews.py",
    "scripts/smoke_geometry.py",
    "scripts/smoke_dicom_geometry.py",
    "scripts/validate_contract.py",
    "scripts/verify_public_artifacts.py",
    "src/observable_crop/__init__.py",
    "src/observable_crop/dicom_geometry.py",
    "src/observable_crop/geometry.py",
    "src/observable_crop/nifti.py",
)

REQUIRED_EXACT_FILES = frozenset(
    (*REQUIRED_REPORTS, *REQUIRED_METRICS, *REQUIRED_SUPPORT_FILES)
)

RAW_OR_MODEL_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".dcm",
    ".dicom",
    ".npz",
    ".npy",
    ".xlsx",
    ".xls",
    ".h5",
    ".hdf5",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
)

TEXT_SUFFIXES = {
    "",
    ".csv",
    ".gitignore",
    ".gitkeep",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

LOCAL_ONLY_DIRS = {
    "cache",
    "cache_preview",
    "checkpoints",
    "features",
    "logs",
    "predictions",
}

IDENTIFIER_HEADERS = {
    "clinical_patient_id",
    "clinical_trial_subject_id",
    "clinicaltrialsubjectid",
    "medical_record_number",
    "mrn",
    "patient_id",
    "patientid",
    "subject_id",
    "subjectid",
}

AGGREGATE_COUNT_HEADERS = {
    "n",
    "n_cases",
    "n_patients",
    "patient_count",
    "patients",
    "n_subjects",
    "subject_count",
}

PATIENT_FILE_RE = re.compile(
    r"(?:patient|subject)[_-]?(?:detail|level|record|row|visit|prediction)",
    flags=re.IGNORECASE,
)
PREFIXED_PATIENT_ID_RE = re.compile(
    r"\b(?:ACRIN[-_]?6698[-_]?|ISPY[12][-_]?)[0-9]{5,8}\b",
    flags=re.IGNORECASE,
)
CONTEXTUAL_PATIENT_ID_RE = re.compile(
    r"\b(?:patients?|subjects?|participants?|cases?|clinical[-_ ]?trial[-_ ]?subject[-_ ]?id|i[-_ ]?spy[12])"
    r"(?:[-_ ]?id)?\s*[:=#/\\-]\s*[0-9]{5,8}\b",
    flags=re.IGNORECASE,
)


def _absolute_host_path_re() -> re.Pattern[str]:
    # Build host-specific prefixes so this source passes its own text scan.
    prefixes = (
        "/" + "data",
        "/" + "home",
        "/" + "mnt",
        "/" + "scratch",
        "/" + "workspace",
        "/" + "Users",
    )
    alternatives = "|".join(re.escape(value) for value in prefixes)
    return re.compile(
        r"(?<![A-Za-z0-9_:/.-])(?:" + alternatives + r")(?:/|$)",
        flags=re.MULTILINE,
    )


ABSOLUTE_HOST_PATH_RE = _absolute_host_path_re()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _add_issue(
    issues: list[dict[str, str]],
    code: str,
    path: str,
    detail: str,
) -> None:
    issues.append({"code": code, "path": path, "detail": detail})


def _find_repo_root(root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    candidate = Path(result.stdout.strip()).resolve()
    try:
        root.relative_to(candidate)
    except ValueError:
        return None
    return candidate


def _git_public_files(root: Path, repo_root: Path) -> list[Path]:
    experiment_rel = root.relative_to(repo_root).as_posix()
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            experiment_rel,
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    files: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        candidate = (repo_root / os.fsdecode(raw)).absolute()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        files.append(candidate)
    return sorted(set(files), key=lambda path: _relative(path, root))


def _fallback_public_files(root: Path) -> list[Path]:
    """Best-effort non-Git fallback, mainly for isolated bundle verification."""

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(root)
        lowered = tuple(part.lower() for part in relative.parts)
        if ".git" in lowered or "__pycache__" in lowered:
            continue
        if path.name != ".gitkeep" and any(
            part in LOCAL_ONLY_DIRS for part in lowered[:-1]
        ):
            continue
        if path.suffix.lower() == ".pyc":
            continue
        if relative.match("metrics/patient_visit_contracts.csv"):
            continue
        if relative.match("metrics/patient_level_*.csv"):
            continue
        if relative.match("manifests/patient_*.jsonl"):
            continue
        files.append(path.absolute())
    return sorted(set(files), key=lambda value: _relative(value, root))


def discover_public_files(root: Path) -> tuple[list[Path], str]:
    repo_root = _find_repo_root(root)
    if repo_root is None:
        return _fallback_public_files(root), "filesystem_fallback"
    try:
        return _git_public_files(root, repo_root), "git_tracked_or_nonignored"
    except (OSError, subprocess.CalledProcessError):
        return _fallback_public_files(root), "filesystem_fallback"


def _has_forbidden_suffix(relative: str) -> bool:
    lowered = relative.lower()
    return lowered.endswith(RAW_OR_MODEL_SUFFIXES)


def _is_local_only_path(relative: str) -> bool:
    path = Path(relative)
    lowered = tuple(part.lower() for part in path.parts)
    if path.name == ".gitkeep":
        return False
    if any(part in LOCAL_ONLY_DIRS for part in lowered[:-1]):
        return True
    return bool(path.suffix.lower() == ".csv" and PATIENT_FILE_RE.search(path.name))


def _sensitive_text_codes(text: str) -> set[str]:
    codes: set[str] = set()
    if ABSOLUTE_HOST_PATH_RE.search(text):
        codes.add("ABSOLUTE_HOST_PATH")
    if PREFIXED_PATIENT_ID_RE.search(text) or CONTEXTUAL_PATIENT_ID_RE.search(text):
        codes.add("PATIENT_IDENTIFIER")
    return codes


def _png_metadata(path: Path) -> tuple[int, int, list[str]]:
    payload = path.read_bytes()
    if len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")

    width = 0
    height = 0
    metadata: list[str] = []
    saw_ihdr = False
    saw_iend = False
    offset = 8
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(payload):
            raise ValueError("truncated PNG chunk")
        chunk = payload[start:end]
        stored_crc = struct.unpack(">I", payload[end : end + 4])[0]
        actual_crc = zlib.crc32(chunk_type + chunk) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError("invalid PNG checksum")

        if chunk_type == b"IHDR":
            if saw_ihdr or length != 13:
                raise ValueError("invalid PNG header")
            width, height = struct.unpack(">II", chunk[:8])
            saw_ihdr = True
        elif chunk_type == b"tEXt":
            metadata.append(chunk.decode("latin-1"))
        elif chunk_type == b"zTXt":
            try:
                keyword, remainder = chunk.split(b"\0", 1)
                if not remainder or remainder[0] != 0:
                    raise ValueError("unsupported PNG text compression")
                value = zlib.decompress(remainder[1:]).decode("latin-1")
                metadata.append(keyword.decode("latin-1") + value)
            except (ValueError, UnicodeDecodeError, zlib.error) as error:
                raise ValueError("invalid compressed PNG metadata") from error
        elif chunk_type == b"iTXt":
            try:
                keyword, remainder = chunk.split(b"\0", 1)
                if len(remainder) < 2:
                    raise ValueError("truncated international PNG metadata")
                compression_flag = remainder[0]
                compression_method = remainder[1]
                remainder = remainder[2:]
                language, remainder = remainder.split(b"\0", 1)
                translated, value = remainder.split(b"\0", 1)
                if compression_flag == 1:
                    if compression_method != 0:
                        raise ValueError("unsupported PNG text compression")
                    value = zlib.decompress(value)
                elif compression_flag != 0:
                    raise ValueError("invalid PNG compression flag")
                metadata.append(
                    b" ".join((keyword, language, translated, value)).decode(
                        "utf-8", errors="strict"
                    )
                )
            except (ValueError, UnicodeDecodeError, zlib.error) as error:
                raise ValueError("invalid international PNG metadata") from error

        offset = end + 4
        if chunk_type == b"IEND":
            saw_iend = True
            break

    if not saw_ihdr or not saw_iend or width <= 0 or height <= 0:
        raise ValueError("incomplete PNG structure")
    return width, height, metadata


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _validate_csv(
    path: Path,
    relative: str,
    issues: list[dict[str, str]],
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header or not any(value.strip() for value in header):
                raise ValueError("CSV header is empty")
            normalized = {_normalized_header(value) for value in header}
            if normalized & IDENTIFIER_HEADERS:
                _add_issue(
                    issues,
                    "PATIENT_LEVEL_CSV",
                    relative,
                    "CSV contains a direct subject identifier column",
                )
            normalized_header = [_normalized_header(value) for value in header]
            aggregate_count_indices = {
                index
                for index, value in enumerate(normalized_header)
                if value in AGGREGATE_COUNT_HEADERS
            }
            small_public_cell = False
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) != len(header):
                    raise ValueError("CSV row width differs from header")
                for count_index in aggregate_count_indices:
                    try:
                        count = int(float(row[count_index]))
                    except (TypeError, ValueError):
                        raise ValueError("aggregate count is not numeric")
                    small_public_cell |= 0 < count < 5
            if small_public_cell:
                _add_issue(
                    issues,
                    "SMALL_PUBLIC_CELL",
                    relative,
                    "CSV exposes an aggregate based on fewer than five cases",
                )
            if relative == "metrics/image_quality_preview.csv":
                required_preview_columns = {
                    "n_cases",
                    "normalization_sensitivity",
                    "legacy_norm_median_mean",
                    "legacy_norm_scale_mean",
                    "legacy_norm_source_std_mean",
                    "nonconstant_fraction",
                }
                missing = sorted(required_preview_columns - normalized)
                if missing or row_count == 0:
                    raise ValueError(
                        "image preview CSV lacks 5-case normalization audit fields"
                    )
    except (OSError, UnicodeError, csv.Error, ValueError):
        _add_issue(issues, "INVALID_CSV", relative, "CSV is not well formed")


def _json_small_count_paths(
    value: Any,
    path: str = "$",
) -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_header(str(key))
            child = f"{path}.{key}"
            if (
                normalized in AGGREGATE_COUNT_HEADERS - {"n"}
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
                and float(item).is_integer()
                and 0 < int(item) < 5
            ):
                findings.append(child)
            findings.extend(_json_small_count_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_json_small_count_paths(item, f"{path}[{index}]"))
    return findings


def _validate_json(
    path: Path,
    relative: str,
    issues: list[dict[str, str]],
) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("root must be an object")
        small_counts = _json_small_count_paths(payload)
        if small_counts:
            _add_issue(
                issues,
                "SMALL_PUBLIC_CELL",
                relative,
                "JSON exposes a patient/case aggregate based on fewer than five cases",
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        _add_issue(issues, "INVALID_JSON", relative, "JSON root must be an object")


def _check_required_files(
    public_relative: set[str],
    root: Path,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    missing = sorted(REQUIRED_EXACT_FILES - public_relative)
    for relative in missing:
        _add_issue(
            issues,
            "MISSING_REQUIRED",
            relative,
            "required artifact is absent from the public bundle",
        )

    short_reports: list[str] = []
    for relative in REQUIRED_REPORTS:
        path = root / relative
        if relative in public_relative and path.is_file():
            try:
                if path.stat().st_size < MIN_REPORT_BYTES:
                    short_reports.append(relative)
                    _add_issue(
                        issues,
                        "EMPTY_REPORT",
                        relative,
                        "report is empty or only a placeholder",
                    )
            except OSError:
                short_reports.append(relative)

    figure_slots: dict[str, list[str]] = {}
    for number in range(1, 13):
        prefix = f"{number:02d}"
        pattern = re.compile(rf"^{prefix}(?:[_-].+)?\.png$", re.IGNORECASE)
        matches = sorted(
            relative
            for relative in public_relative
            if Path(relative).parent.as_posix() == "figures"
            and pattern.fullmatch(Path(relative).name)
        )
        figure_slots[prefix] = matches
        if not matches:
            _add_issue(
                issues,
                "MISSING_FIGURE_SLOT",
                f"figures/{prefix}_*.png",
                "numbered public figure is absent",
            )

    return {
        "required_exact_count": len(REQUIRED_EXACT_FILES),
        "required_exact_present": len(REQUIRED_EXACT_FILES) - len(missing),
        "numbered_figure_slots_present": sum(bool(value) for value in figure_slots.values()),
        "numbered_figure_slots_required": 12,
        "short_report_count": len(short_reports),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_provenance(root: Path, issues: list[dict[str, str]]) -> None:
    relative = "manifests/stage_a_provenance.json"
    path = root / relative
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hashes = payload.get("experiment_source_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("source hash map is absent")
        for source_relative, expected in hashes.items():
            source = (root / str(source_relative)).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("hashed source is absent or outside root")
            if not isinstance(expected, str) or _sha256(source) != expected:
                raise ValueError("source hash mismatch")
        config = root / "configs" / "stage_a.json"
        if payload.get("config_sha256") != _sha256(config):
            raise ValueError("config hash mismatch")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        _add_issue(
            issues,
            "INVALID_PROVENANCE",
            relative,
            "provenance source/config hashes are absent or inconsistent",
        )


def _check_public_files(
    root: Path,
    public_files: Sequence[Path],
    issues: list[dict[str, str]],
) -> None:
    seen: set[str] = set()
    for path in public_files:
        try:
            relative = _relative(path, root)
        except ValueError:
            _add_issue(
                issues,
                "OUTSIDE_ROOT",
                "<outside-root>",
                "public file resolves outside the experiment root",
            )
            continue

        if relative in seen:
            continue
        seen.add(relative)

        for code in _sensitive_text_codes(relative):
            _add_issue(
                issues,
                code,
                relative,
                "public filename contains sensitive material",
            )

        if path.is_symlink():
            _add_issue(
                issues,
                "SYMLINK",
                relative,
                "symlinks are not allowed in the public bundle",
            )
            continue
        if not path.is_file():
            _add_issue(
                issues,
                "UNSAFE_FILE_KIND",
                relative,
                "public path is not a regular file",
            )
            continue

        try:
            size = path.stat().st_size
        except OSError:
            _add_issue(issues, "UNREADABLE", relative, "file cannot be inspected")
            continue
        if size > MAX_PUBLIC_FILE_BYTES:
            _add_issue(
                issues,
                "FILE_TOO_LARGE",
                relative,
                "public file exceeds the size limit",
            )
            continue

        if _has_forbidden_suffix(relative):
            _add_issue(
                issues,
                "FORBIDDEN_BINARY",
                relative,
                "raw image, spreadsheet, array, feature, or model artifact is public",
            )
            continue
        if _is_local_only_path(relative):
            code = "PATIENT_LEVEL_CSV" if path.suffix.lower() == ".csv" else "LOCAL_ONLY_ARTIFACT"
            _add_issue(
                issues,
                code,
                relative,
                "patient-level or local diagnostic artifact is public",
            )

        suffix = path.suffix.lower()
        if suffix == ".png":
            try:
                _, _, metadata = _png_metadata(path)
            except (OSError, ValueError):
                _add_issue(issues, "INVALID_PNG", relative, "PNG structure is invalid")
                continue
            metadata_codes: set[str] = set()
            for value in metadata:
                metadata_codes.update(_sensitive_text_codes(value))
            for code in sorted(metadata_codes):
                _add_issue(
                    issues,
                    code,
                    relative,
                    "PNG metadata contains sensitive material",
                )
            continue

        if suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitkeep"}:
            _add_issue(
                issues,
                "UNKNOWN_FILE_TYPE",
                relative,
                "file type is not approved for the public bundle",
            )
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            _add_issue(
                issues,
                "NON_UTF8_TEXT",
                relative,
                "approved text artifact is not valid UTF-8",
            )
            continue
        for code in sorted(_sensitive_text_codes(text)):
            _add_issue(
                issues,
                code,
                relative,
                "public text contains sensitive material",
            )

        if suffix == ".csv":
            _validate_csv(path, relative, issues)
        elif suffix == ".json":
            _validate_json(path, relative, issues)


def audit_public_tree(
    root: Path,
    *,
    public_files: Iterable[Path] | None = None,
    discovery_mode: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "discovery_mode": "unavailable",
            "public_file_count": 0,
            "summary": {},
            "issues": [
                {
                    "code": "INVALID_ROOT",
                    "path": ".",
                    "detail": "experiment root is not a directory",
                }
            ],
        }

    if public_files is None:
        discovered, mode = discover_public_files(root)
    else:
        discovered = sorted(
            (Path(path).absolute() for path in public_files),
            key=lambda path: path.as_posix(),
        )
        mode = discovery_mode or "caller_supplied"

    issues: list[dict[str, str]] = []
    public_relative: set[str] = set()
    for path in discovered:
        try:
            public_relative.add(_relative(path, root))
        except ValueError:
            pass

    summary = _check_required_files(public_relative, root, issues)
    _check_public_files(root, discovered, issues)
    _check_provenance(root, issues)
    issues.sort(key=lambda item: (item["path"], item["code"], item["detail"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not issues else "FAIL",
        "discovery_mode": mode,
        "public_file_count": len(public_relative),
        "summary": summary,
        "issues": issues,
    }


def _png_chunk(kind: bytes, value: bytes) -> bytes:
    return (
        struct.pack(">I", len(value))
        + kind
        + value
        + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
    )


def _minimal_png() -> bytes:
    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixels = zlib.compress(b"\x00\x00\x00\x00\xff")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _all_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())


def _self_test_fixture(root: Path) -> None:
    for relative in REQUIRED_EXACT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".json"):
            path.write_text('{"schema_version": 1}\n', encoding="utf-8")
        elif relative.endswith(".csv"):
            if relative == "metrics/image_quality_preview.csv":
                path.write_text(
                    "n_cases,normalization_sensitivity,legacy_norm_median_mean,"
                    "legacy_norm_scale_mean,legacy_norm_source_std_mean,"
                    "nonconstant_fraction\n"
                    "5,2D_LEGACY,0,1,1,1\n",
                    encoding="utf-8",
                )
            else:
                path.write_text("scope,n\nALL,5\n", encoding="utf-8")
        elif relative.endswith(".md"):
            path.write_text(
                "# Aggregate report\n\n"
                + "Outcome-free aggregate observability evidence. " * 4
                + "Aggregate retained voxel count: 150000.\n",
                encoding="utf-8",
            )
        elif relative.endswith(".py"):
            path.write_text('"""Public source fixture."""\n', encoding="utf-8")
        else:
            path.write_text("public fixture\n", encoding="utf-8")

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for number in range(1, 13):
        (figures / f"{number:02d}_aggregate_fixture.png").write_bytes(_minimal_png())
    config = root / "configs" / "stage_a.json"
    provenance = root / "manifests" / "stage_a_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": _sha256(config),
                "experiment_source_sha256": {
                    "configs/stage_a.json": _sha256(config)
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def run_self_test() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="observable-crop-verifier-") as temporary:
        root = Path(temporary)
        _self_test_fixture(root)

        baseline = audit_public_tree(root, public_files=_all_files(root))
        cases.append({"name": "valid_bundle", "pass": baseline["status"] == "PASS"})

        report = root / "reports" / "final_report.md"
        original_report = report.read_text(encoding="utf-8")
        synthetic_identifier = "ISPY2-" + "123" + "456"
        report.write_text(original_report + synthetic_identifier, encoding="utf-8")
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "patient_identifier_rejected",
                "pass": any(issue["code"] == "PATIENT_IDENTIFIER" for issue in result["issues"]),
            }
        )
        report.write_text(original_report, encoding="utf-8")

        host_root = "/" + "mnt"
        report.write_text(original_report + host_root + "/restricted", encoding="utf-8")
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "mounted_host_path_rejected",
                "pass": any(issue["code"] == "ABSOLUTE_HOST_PATH" for issue in result["issues"]),
            }
        )
        report.write_text(original_report, encoding="utf-8")

        aggregate_csv = root / "metrics" / "containment_summary.csv"
        aggregate_content = aggregate_csv.read_text(encoding="utf-8")
        aggregate_csv.write_text("scope,n_patients\nALL,3\n", encoding="utf-8")
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "csv_small_patient_cell_rejected",
                "pass": any(issue["code"] == "SMALL_PUBLIC_CELL" for issue in result["issues"]),
            }
        )
        aggregate_csv.write_text(aggregate_content, encoding="utf-8")

        aggregate_json = root / "metrics" / "input_recommendation.json"
        aggregate_json_content = aggregate_json.read_text(encoding="utf-8")
        aggregate_json.write_text('{"patient_count": 3}\n', encoding="utf-8")
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "json_small_patient_cell_rejected",
                "pass": any(issue["code"] == "SMALL_PUBLIC_CELL" for issue in result["issues"]),
            }
        )
        aggregate_json.write_text(aggregate_json_content, encoding="utf-8")

        host_root = "/" + "data"
        report.write_text(original_report + host_root + "/restricted", encoding="utf-8")
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "absolute_host_path_rejected",
                "pass": any(issue["code"] == "ABSOLUTE_HOST_PATH" for issue in result["issues"]),
            }
        )
        report.write_text(original_report, encoding="utf-8")

        patient_csv = root / "metrics" / "patient_level_rows.csv"
        patient_csv.write_text("patient_id,value\nopaque,1\n", encoding="utf-8")
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "patient_level_csv_rejected",
                "pass": any(issue["code"] == "PATIENT_LEVEL_CSV" for issue in result["issues"]),
            }
        )
        patient_csv.unlink()

        leaked_array = root / "figures" / "local_preview.npz"
        leaked_array.write_bytes(b"not-a-public-array")
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "forbidden_binary_rejected",
                "pass": any(issue["code"] == "FORBIDDEN_BINARY" for issue in result["issues"]),
            }
        )
        leaked_array.unlink()

        required_report = root / REQUIRED_REPORTS[0]
        required_content = required_report.read_text(encoding="utf-8")
        required_report.unlink()
        result = audit_public_tree(root, public_files=_all_files(root))
        cases.append(
            {
                "name": "missing_required_rejected",
                "pass": any(issue["code"] == "MISSING_REQUIRED" for issue in result["issues"]),
            }
        )
        required_report.write_text(required_content, encoding="utf-8")

    passed = all(bool(case["pass"]) for case in cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "self_test": True,
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit required aggregate artifacts and reject private-data leakage."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="experiment root (defaults to the directory above this script)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run isolated positive and negative verifier fixtures",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_self_test() if args.self_test else audit_public_tree(args.root)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=args.indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
