#!/usr/bin/env python3
"""Fail closed if public confirmation artifacts expose identifiers or paths."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

import pandas as pd
from PIL import Image, UnidentifiedImageError


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
scripts_value = str(SCRIPTS_ROOT.resolve())
while scripts_value in sys.path:
    sys.path.remove(scripts_value)
sys.path.insert(0, scripts_value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402

OUTPUT = EXPERIMENT_ROOT / "metrics" / "public_artifact_privacy_gate.json"
PRIVATE_INPUT_ROOT_ENV = "MWM_PRIVATE_INPUT_REPO_ROOT"
IDENTIFIER_RELATIVE_SOURCES = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/technical_eligibility_patients.private.csv",
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/stage_b_c1b_cache.private.csv",
    "additional_experiments/c1b_model_ready_ftv_sanity/manifests/ispy1_base_eligibility_patients.private.csv",
)
DATA_CONTRACT_RELATIVE = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
    "manifests/stage_b_data_contract.private.json"
)
STAGE_A_SENTINEL_RELATIVE = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/STAGE_A_GO.json"
)
IDENTIFIER_SOURCE_CONTRACT_FIELDS: Mapping[str, tuple[str, str]] = {
    IDENTIFIER_RELATIVE_SOURCES[0]: (
        "technical_eligibility_manifest",
        "technical_eligibility_manifest_sha256",
    ),
    IDENTIFIER_RELATIVE_SOURCES[1]: (
        "c1b_cache_manifest",
        "c1b_cache_manifest_sha256",
    ),
    IDENTIFIER_RELATIVE_SOURCES[2]: (
        "train_only_candidate_manifest",
        "train_only_candidate_manifest_sha256",
    ),
}
# Test-only override hook; formal execution leaves this empty and resolves the
# canonical relative sources through the authorized private input root.
IDENTIFIER_SOURCES: tuple[Path, ...] = ()
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
REQUIRED_PUBLIC_RESULT_FILES = (
    "metrics/aggregation_summary.json",
    "metrics/decision_summary.json",
    "metrics/natural_pooled_metrics.csv",
    "metrics/paired_fold_effects.csv",
    "metrics/report_context.json",
    "metrics/seed_level_summary.csv",
    "metrics/table1_architecture_contract.csv",
    "metrics/table2_static_ftv.csv",
    "metrics/table3_observed_delta_ftv.csv",
    "metrics/table4_paired_architecture_effects.csv",
    "metrics/table5_grounding_effects.csv",
    "metrics/table6_optimization_safety.csv",
    "metrics/table7_prediction_variance_calibration.csv",
    "metrics/training_trajectories.csv",
    "metrics/transformed_fold_summaries.csv",
    "figures/01_gap_local_architecture_schematic.png",
    "figures/02_static_ftv_spearman_comparison.png",
    "figures/03_static_ftv_natural_r2_comparison.png",
    "figures/04_delta_ftv_spearman_comparison.png",
    "figures/05_delta_ftv_natural_r2_comparison.png",
    "figures/06_prediction_target_variance_ratio.png",
    "figures/07_descriptive_calibration_slope.png",
    "figures/08_paired_fold_effects.png",
    "figures/09_optimization_safety_heatmap.png",
    "figures/10_representative_training_curves.png",
    "reports/final_report.md",
)
SENSITIVE_IDENTIFIER_KEYS = {
    "accession_number",
    "cache_path",
    "mrn",
    "patient_id",
    "patient_ids",
    "patient_token",
    "patient_tokens",
    "pid",
    "series_uid",
    "sop_instance_uid",
    "subject_id",
    "subject_ids",
    "subject_token",
    "subject_tokens",
    "study_uid",
}
HTTP_URL_PATTERN = re.compile(r"https?://[^\s<>\]\[(){}\"']+", re.IGNORECASE)
HTTP_URL_BYTES_PATTERN = re.compile(rb"https?://[^\s<>\]\[(){}\"']+", re.IGNORECASE)
DICOM_UID_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d+){5,}\b")
CANONICAL_PYTHON_SHEBANGS = {
    "#!" + "/" + "usr/bin/env python",
    "#!" + "/" + "usr/bin/env python3",
}
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:file:" r"//|(?<![A-Za-z0-9/:])/(?!/)(?:[A-Za-z0-9._~-]+/)+"
    r"(?:[A-Za-z0-9._~-]+)?|(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
    r"|(?<![\\])\\\\[A-Za-z0-9._~-]+\\[A-Za-z0-9._~-]+)"
)
ABSOLUTE_PATH_BYTES_PATTERN = re.compile(
    rb"(?:file:" rb"//|(?<![A-Za-z0-9/:])/(?!/)(?:[A-Za-z0-9._~-]+/)+"
    rb"(?:[A-Za-z0-9._~-]+)?|(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
    rb"|(?<![\\])\\\\[A-Za-z0-9._~-]+\\[A-Za-z0-9._~-]+)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_source(relative: str) -> Path:
    raw_root = os.environ.get(PRIVATE_INPUT_ROOT_ENV, "").strip()
    root = (
        Path(raw_root).expanduser().resolve()
        if raw_root
        else REPO_ROOT.resolve()
    )
    candidate = (root / relative).resolve()
    if not raw_root and not candidate.is_file():
        raise FileNotFoundError(
            f"{candidate}; set {PRIVATE_INPUT_ROOT_ENV} to the authorized source repository"
        )
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("private identifier source escaped its authorized root") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def verified_identifier_sources(
    preregistration: Mapping[str, Any],
) -> tuple[tuple[Path, ...], dict[str, str], str]:
    """Resolve the exact locked denylist sources and re-hash every one."""

    upstream = preregistration.get("upstream_sha256")
    if not isinstance(upstream, Mapping):
        raise ValueError("preregistration upstream inventory is invalid")
    locked_contract_sha256 = str(upstream.get(DATA_CONTRACT_RELATIVE, ""))
    contract_path = _private_source(DATA_CONTRACT_RELATIVE)
    if file_sha256(contract_path) != locked_contract_sha256:
        raise RuntimeError("private data contract differs from preregistration lock")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("private data contract is unreadable or invalid") from error
    if not isinstance(contract, Mapping):
        raise ValueError("private data contract must be a JSON object")

    sources: list[Path] = []
    source_sha256: dict[str, str] = {}
    for relative in IDENTIFIER_RELATIVE_SOURCES:
        path_key, sha_key = IDENTIFIER_SOURCE_CONTRACT_FIELDS[relative]
        source = _private_source(relative)
        declared_path = contract.get(path_key)
        declared_sha256 = str(contract.get(sha_key, ""))
        if not isinstance(declared_path, str) or Path(declared_path).resolve() != source:
            raise ValueError(f"private identifier source path drifted: {path_key}")
        observed_sha256 = file_sha256(source)
        if (
            re.fullmatch(r"[0-9a-f]{64}", declared_sha256) is None
            or observed_sha256 != declared_sha256
        ):
            raise RuntimeError(f"private identifier source hash drifted: {path_key}")
        sources.append(source)
        source_sha256[relative] = observed_sha256
    return tuple(sources), source_sha256, locked_contract_sha256


def identifiers(sources: Sequence[Path] | None = None) -> set[str]:
    values: set[str] = set()
    resolved_sources = tuple(sources) if sources is not None else (
        IDENTIFIER_SOURCES
        or tuple(_private_source(relative) for relative in IDENTIFIER_RELATIVE_SOURCES)
    )
    for path in resolved_sources:
        frame = pd.read_csv(path, dtype="string")
        columns = [column for column in IDENTIFIER_COLUMNS if column in frame]
        if not columns:
            raise ValueError(f"private identifier source has no denylist column: {path}")
        for column in columns:
            values.update(str(value) for value in frame[column].dropna())
        if "patient_id" in frame:
            values.update(
                hashlib.sha256(str(value).encode("utf-8")).hexdigest()
                for value in frame["patient_id"].dropna()
            )
        if "cache_path" in frame:
            for row_index, value in frame["cache_path"].dropna().items():
                token = Path(str(value)).stem
                if re.fullmatch(r"[0-9a-f]{64}", token) is None:
                    raise ValueError(f"cache manifest has an invalid token basename: {path}")
                if "patient_id" in frame and pd.notna(frame.at[row_index, "patient_id"]):
                    expected = hashlib.sha256(
                        str(frame.at[row_index, "patient_id"]).encode("utf-8")
                    ).hexdigest()
                    if token != expected:
                        raise ValueError("cache token does not match its patient identifier")
                values.add(token)
    if any(not value for value in values):
        raise ValueError("private identifier source contains an empty identifier")
    return values


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


def _artifact_token(path: Path) -> str:
    relative = path.resolve().relative_to(EXPERIMENT_ROOT.resolve()).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def _sensitive_json_key_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(str(key).strip().casefold() in SENSITIVE_IDENTIFIER_KEYS)
            + _sensitive_json_key_count(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(_sensitive_json_key_count(child) for child in value)
    return 0


def _sensitive_csv_header_count(text: str, suffix: str) -> int:
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        header = next(csv.reader(text.splitlines(), delimiter=delimiter), [])
    except csv.Error:
        return 1
    return sum(
        str(column).strip().casefold() in SENSITIVE_IDENTIFIER_KEYS
        for column in header
    )


def _text_for_path_scan(text: str, suffix: str) -> str:
    if suffix == ".py":
        lines = text.splitlines(keepends=True)
        if lines and lines[0].rstrip("\r\n") in CANONICAL_PYTHON_SHEBANGS:
            text = "".join(lines[1:])
    return HTTP_URL_PATTERN.sub("", text)


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
        artifact_token = _artifact_token(path)
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES | BINARY_SUFFIXES:
            unsupported_findings.append(
                {
                    "artifact_token": artifact_token,
                    "reason": f"unsupported suffix {suffix!r}",
                }
            )
            continue
        content = path.read_bytes()
        relative_bytes = path.resolve().relative_to(EXPERIMENT_ROOT.resolve()).as_posix().encode(
            "utf-8"
        )
        matches = sorted(
            value
            for value, encoded in encoded_ids.items()
            if encoded in content or encoded in relative_bytes
        )
        structural_count = 0
        dicom_uid_found = False
        if suffix in TEXT_SUFFIXES:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                unsupported_findings.append(
                    {"artifact_token": artifact_token, "reason": "invalid UTF-8 text"}
                )
                continue
            if suffix in {".csv", ".tsv"}:
                structural_count += _sensitive_csv_header_count(text, suffix)
            elif suffix == ".json":
                try:
                    structural_count += _sensitive_json_key_count(json.loads(text))
                except json.JSONDecodeError:
                    unsupported_findings.append(
                        {"artifact_token": artifact_token, "reason": "invalid JSON"}
                    )
            dicom_uid_found = bool(
                DICOM_UID_PATTERN.search(text)
                or DICOM_UID_PATTERN.search(relative_bytes.decode("utf-8"))
            )
            contains_absolute_path = (
                ABSOLUTE_PATH_PATTERN.search(_text_for_path_scan(text, suffix))
                is not None
            )
        else:
            # PNG metadata chunks remain byte-searchable.  Only the explicitly
            # supported preregistered image format reaches this branch.
            contains_absolute_path = (
                ABSOLUTE_PATH_BYTES_PATTERN.search(
                    HTTP_URL_BYTES_PATTERN.sub(b"", content)
                )
                is not None
            )
        if matches or structural_count or dicom_uid_found:
            identifier_findings.append(
                {
                    "artifact_token": artifact_token,
                    "exact_identifier_count": len(matches),
                    "sensitive_identifier_key_count": structural_count,
                    "dicom_uid_pattern_found": dicom_uid_found,
                }
            )
        if contains_absolute_path:
            absolute_path_findings.append({"artifact_token": artifact_token})
    return identifier_findings, absolute_path_findings, unsupported_findings


def private_permission_findings() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for name in PRIVATE_ROOTS:
        root = EXPERIMENT_ROOT / name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path == _root_placeholder(root):
                continue
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                findings.append(
                    {"artifact_token": _artifact_token(path), "reason": "mode_not_0600"}
                )
    for name in PRIVATE_NAMED_PUBLIC_ROOTS:
        root = EXPERIMENT_ROOT / name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _private_named(path):
                continue
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                findings.append(
                    {"artifact_token": _artifact_token(path), "reason": "mode_not_0600"}
                )
    return sorted(findings, key=lambda row: row["artifact_token"])


def private_layout_findings() -> list[dict[str, str]]:
    """Private-named public-root artifacts may only be direct root children."""

    findings: list[dict[str, str]] = []
    for name in PRIVATE_NAMED_PUBLIC_ROOTS:
        root = EXPERIMENT_ROOT / name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and _private_named(path)
                and path.parent.resolve() != root.resolve()
            ):
                findings.append(
                    {
                        "artifact_token": _artifact_token(path),
                        "reason": "nested_private_artifact_forbidden",
                    }
                )
    return sorted(findings, key=lambda row: row["artifact_token"])


def private_git_hygiene_findings() -> list[dict[str, str]]:
    """Require every private artifact to be covered by repository ignore rules."""

    candidates: set[Path] = set()
    for name in PRIVATE_ROOTS:
        root = EXPERIMENT_ROOT / name
        if root.exists():
            candidates.update(
                path for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep"
            )
    for name in PRIVATE_NAMED_PUBLIC_ROOTS:
        root = EXPERIMENT_ROOT / name
        if root.exists():
            candidates.update(
                path for path in root.rglob("*") if path.is_file() and _private_named(path)
            )
    findings: list[dict[str, str]] = []
    for path in sorted(candidates):
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        tracked = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", "--", relative],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode == 0:
            findings.append(
                {
                    "artifact_token": _artifact_token(path),
                    "reason": "private_artifact_is_tracked",
                }
            )
            continue
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "--no-index", "-q", "--", relative],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            findings.append(
                {"artifact_token": _artifact_token(path), "reason": "not_git_ignored"}
            )
    return findings


def public_result_completeness_findings() -> list[dict[str, str]]:
    expected = set(REQUIRED_PUBLIC_RESULT_FILES)
    actual: set[str] = set()
    for name in PRIVATE_NAMED_PUBLIC_ROOTS + ("figures",):
        root = EXPERIMENT_ROOT / name
        if not root.exists():
            continue
        actual.update(
            path.resolve().relative_to(EXPERIMENT_ROOT.resolve()).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != ".gitkeep"
            and path.resolve() != OUTPUT.resolve()
            and not _private_named(path)
        )
    findings: list[dict[str, str]] = [
        {"required_artifact": relative, "reason": "missing"}
        for relative in sorted(expected - actual)
    ]
    findings.extend(
        {
            "artifact_token": hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16],
            "reason": "unexpected_public_result",
        }
        for relative in sorted(actual - expected)
    )
    return findings


def public_result_content_findings(
    preregistration: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Reject correctly named but empty, fake, stale, or incomplete outputs."""

    if not REQUIRED_PUBLIC_RESULT_FILES:
        return []
    findings: list[dict[str, str]] = []
    expected = set(REQUIRED_PUBLIC_RESULT_FILES)
    for relative in sorted(expected):
        path = EXPERIMENT_ROOT / relative
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        try:
            if suffix == ".csv":
                frame = pd.read_csv(path)
                if frame.empty or not len(frame.columns):
                    raise ValueError("empty public CSV")
            elif suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, Mapping) or not value:
                    raise ValueError("empty public JSON")
            elif suffix == ".png":
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()
                if width < 100 or height < 100:
                    raise ValueError("implausibly small public figure")
            elif relative == "reports/final_report.md":
                report = path.read_text(encoding="utf-8")
                if (
                    len(report) < 2000
                    or any(
                        re.search(rf"(?m)^##\s+{number}\.\s", report) is None
                        for number in range(1, 11)
                    )
                    or not all(
                        marker in report
                        for marker in (
                            "FTV",
                            "HR/HER2",
                            "LOCAL_MULTISEED_",
                            "PREREGISTRATION_LOCK.json",
                        )
                    )
                    or any(marker in report for marker in ("TODO", "TBD", "待补"))
                ):
                    raise ValueError("final report is incomplete")
        except (
            OSError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
            pd.errors.ParserError,
            UnidentifiedImageError,
        ):
            findings.append(
                {"required_artifact": relative, "reason": "invalid_or_empty_content"}
            )

    summary_path = EXPERIMENT_ROOT / "metrics" / "aggregation_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            summary = None
        expected_hashed = {
            Path(relative).name
            for relative in expected
            if relative not in {
                "metrics/aggregation_summary.json",
                "reports/final_report.md",
            }
        }
        hashes = summary.get("artifact_sha256") if isinstance(summary, Mapping) else None
        if preregistration is None:
            provenance_valid = True
        else:
            upstream = preregistration.get("upstream_sha256")
            provenance_valid = bool(
                isinstance(summary, Mapping)
                and isinstance(upstream, Mapping)
                and summary.get("preregistration_lock") == "PREREGISTRATION_LOCK.json"
                and summary.get("preregistration_lock_sha256")
                == preregistration.get("lock_sha256")
                and summary.get("config_sha256")
                == preregistration.get("config_sha256")
                and summary.get("stage_a_sentinel_sha256")
                == upstream.get(STAGE_A_SENTINEL_RELATIVE)
                and summary.get("data_contract_sha256")
                == upstream.get(DATA_CONTRACT_RELATIVE)
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(summary.get("data_provenance_sha256", ""))
                )
                is not None
            )
        valid_summary = bool(
            isinstance(summary, Mapping)
            and summary.get("status") == "COMPLETE"
            and summary.get("formal_cells") == 100
            and summary.get("seeds") == [2026, 3026, 4026, 5026, 6026]
            and summary.get("folds_per_seed") == 5
            and summary.get("patient_level_outputs_private") is True
            and summary.get("public_outputs_deidentified") is True
            and isinstance(hashes, Mapping)
            and set(hashes) == expected_hashed
            and provenance_valid
        )
        if valid_summary:
            valid_summary = all(
                hashes.get(Path(relative).name)
                == file_sha256(EXPERIMENT_ROOT / relative)
                for relative in expected
                if Path(relative).name in expected_hashed
                and (EXPERIMENT_ROOT / relative).is_file()
            ) and all(
                (EXPERIMENT_ROOT / relative).is_file()
                for relative in expected
                if Path(relative).name in expected_hashed
            )
        if not valid_summary:
            findings.append(
                {
                    "required_artifact": "metrics/aggregation_summary.json",
                    "reason": "invalid_or_stale_aggregation_chain",
                }
            )
    return findings


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
    # A pre-lock invocation must not create a result that would itself prevent
    # freezing.  Once a gate exists, however, even a lock-verification failure
    # invalidates a stale PASS before propagating the exception.
    try:
        preregistration = verify_preregistration()
    except BaseException:
        if OUTPUT.exists():
            atomic_json(OUTPUT, {"schema_version": 1, "status": "IN_PROGRESS"})
        raise
    atomic_json(OUTPUT, {"schema_version": 1, "status": "IN_PROGRESS"})
    if IDENTIFIER_SOURCES:
        sources = tuple(IDENTIFIER_SOURCES)
        identifier_source_sha256 = {
            f"test_override_{index}": file_sha256(path)
            for index, path in enumerate(sources)
        }
        data_contract_sha256 = "test_override"
    else:
        (
            sources,
            identifier_source_sha256,
            data_contract_sha256,
        ) = verified_identifier_sources(preregistration)
    private_ids = identifiers(sources)
    scanned = public_artifacts()
    (
        identifier_findings,
        absolute_path_findings,
        unsupported_findings,
    ) = scan_public_artifacts(scanned, private_ids)
    permission_findings = private_permission_findings()
    layout_findings = private_layout_findings()
    git_hygiene_findings = private_git_hygiene_findings()
    completeness_findings = public_result_completeness_findings()
    content_findings = public_result_content_findings(preregistration)
    passed = not any(
        (
            identifier_findings,
            absolute_path_findings,
            unsupported_findings,
            permission_findings,
            layout_findings,
            git_hygiene_findings,
            completeness_findings,
            content_findings,
        )
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "scanned_public_artifacts": len(scanned),
        "public_artifact_sha256": {
            _artifact_token(path): file_sha256(path) for path in scanned
        },
        "private_identifier_values_checked": len(private_ids),
        "private_identifier_source_sha256": identifier_source_sha256,
        "private_data_contract_sha256": data_contract_sha256,
        "identifier_findings": identifier_findings,
        "absolute_path_findings": absolute_path_findings,
        "unsupported_public_artifact_findings": unsupported_findings,
        "private_permission_findings": permission_findings,
        "private_layout_findings": layout_findings,
        "private_git_hygiene_findings": git_hygiene_findings,
        "public_result_completeness_findings": completeness_findings,
        "public_result_content_findings": content_findings,
        "required_public_result_artifacts": len(REQUIRED_PUBLIC_RESULT_FILES),
        "scanner_sha256": file_sha256(Path(__file__)),
    }
    atomic_json(OUTPUT, payload)
    if not passed:
        raise RuntimeError("public-artifact privacy gate failed; see metrics JSON")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
