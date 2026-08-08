#!/usr/bin/env python3
"""Fail-closed verification for the public Stage-A artifact bundle.

The verifier deliberately separates public, Git-eligible artifacts from local
patient-level diagnostics.  On success it writes a deterministic verification
record and a SHA-256 manifest.  The manifest never hashes itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_PATIENTS = 375
EXPECTED_VISITS = ("T0", "T1", "T2", "T3")
EXPECTED_PATIENT_VISITS = 1_500
MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024
MIN_PUBLIC_SUBGROUP_N = 5
MIN_FIGURE_BYTES = 10 * 1024
MIN_FIGURE_WIDTH = 640
MIN_FIGURE_HEIGHT = 480
MAX_FIGURE_DIMENSION = 6_000

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = EXPERIMENT_ROOT / "metrics"
MANIFEST_PATH = METRICS_ROOT / "artifact_manifest.csv"
VERIFICATION_PATH = METRICS_ROOT / "public_artifact_verification.json"
PATIENT_DETAIL_PATH = METRICS_ROOT / "crop_containment_patient_visit.csv"

GENERATED_OUTPUTS = {
    "metrics/artifact_manifest.csv",
    "metrics/public_artifact_verification.json",
}

EXPECTED_FIGURES = tuple(
    f"figures/{name}"
    for name in (
        "01_boundary_touch_rate_by_visit.png",
        "02_margin_distribution_by_visit.png",
        "03_ld_vs_margin_hexbin.png",
        "04_contained_vs_truncated_ld_distribution.png",
        "05_large_ld_subgroup_truncation.png",
        "06_privacy_safe_containment_schematic.png",
    )
)

REQUIRED_PUBLIC_FILES = {
    ".gitignore",
    "EXPERIMENT_PLAN.md",
    "configs/stage_a.json",
    "configs/stage_b.json",
    "metrics/crop_containment_by_ld_quantile.csv",
    "metrics/crop_containment_by_timepoint.csv",
    "metrics/crop_containment_gate.json",
    "metrics/crop_containment_summary.csv",
    "metrics/final_decision.json",
    "metrics/ld_containment_distribution.csv",
    "metrics/stage_a_input_provenance.json",
    "metrics/stage_execution_status.csv",
    "reports/crop_containment_report.md",
    "reports/final_report.md",
    "scripts/finalize_no_go.py",
    "scripts/run_stage_a.py",
    "scripts/smoke_geometry.py",
    "scripts/smoke_stage_a_real.py",
    "scripts/verify_public_artifacts.py",
    "src/ftv_ld_pilot/__init__.py",
    "src/ftv_ld_pilot/geometry.py",
    *EXPECTED_FIGURES,
}

RAW_OR_MODEL_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".dcm",
    ".dicom",
    ".xlsx",
    ".xls",
    ".npz",
    ".npy",
    ".h5",
    ".hdf5",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
)
LOCAL_ONLY_DIRS = {"checkpoints", "features", "predictions", "logs"}
TEXT_SUFFIXES = {
    "",
    ".csv",
    ".gitignore",
    ".gitkeep",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _repo_root() -> Path:
    for candidate in (EXPERIMENT_ROOT, *EXPERIMENT_ROOT.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("repository root unavailable")


def _git_public_files(repo_root: Path) -> list[Path]:
    experiment_rel = EXPERIMENT_ROOT.relative_to(repo_root).as_posix()
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        experiment_rel,
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    paths: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        candidate = repo_root / os.fsdecode(raw)
        relative = candidate.relative_to(EXPERIMENT_ROOT).as_posix()
        if relative not in GENERATED_OUTPUTS:
            paths.append(candidate)
    return sorted(
        set(paths), key=lambda path: path.relative_to(EXPERIMENT_ROOT).as_posix()
    )


def _git_ignored(repo_root: Path, path: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--",
            path.relative_to(repo_root).as_posix(),
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV header unavailable")
        return list(reader)


def _number(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in {key}")
    return value


def _integer(row: Mapping[str, str], key: str) -> int:
    value = _number(row, key)
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"non-integral value in {key}")
    return int(rounded)


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _single(rows: Sequence[Mapping[str, str]], **selectors: str) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in selectors.items())
    ]
    if len(matches) != 1:
        raise ValueError("aggregate row cardinality mismatch")
    return matches[0]


def _png_metadata(path: Path) -> tuple[int, int, list[str]]:
    payload = path.read_bytes()
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
    width, height = struct.unpack(">II", payload[16:24])
    metadata: list[str] = []
    offset = 8
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(payload):
            raise ValueError("truncated PNG chunk")
        chunk = payload[start:end]
        try:
            if chunk_type == b"tEXt":
                metadata.append(chunk.decode("latin-1"))
            elif chunk_type == b"zTXt" and b"\0" in chunk:
                keyword, compressed = chunk.split(b"\0", 1)
                if compressed[:1] == b"\0":
                    metadata.append(
                        keyword.decode("latin-1")
                        + zlib.decompress(compressed[1:]).decode("latin-1")
                    )
            elif chunk_type == b"iTXt":
                metadata.append(chunk.decode("utf-8", errors="ignore"))
        except (UnicodeDecodeError, zlib.error):
            raise ValueError("invalid PNG metadata") from None
        offset = end + 4
        if chunk_type == b"IEND":
            break
    return width, height, metadata


def _sensitive_text(text: str) -> bool:
    # Construct host-path tokens so the verifier source does not trip its own scan.
    host_data_prefix = "/" + "data" + "/"
    host_home_prefix = "/" + "home" + "/"
    if host_data_prefix in text or host_home_prefix in text:
        return True
    if re.search(r"ACRIN-6698-[0-9]{6}", text, flags=re.IGNORECASE):
        return True
    # A standalone six-digit integer is the clinical subject-ID shape.  Exclude
    # decimal fragments and hashes by treating letters, digits, and dots as part
    # of a surrounding token.
    return re.search(r"(?<![A-Za-z0-9.])[0-9]{6}(?![A-Za-z0-9.])", text) is not None


def _privacy_class(relative: str) -> str:
    path = Path(relative)
    if relative == "metrics/public_artifact_verification.json":
        return "verification"
    if path.name == ".gitkeep":
        return "empty_placeholder"
    if relative == ".gitignore":
        return "privacy_policy"
    if path.parts[0] == "metrics":
        return "aggregate_metric"
    if path.parts[0] == "figures":
        return "privacy_safe_aggregate_figure"
    if path.parts[0] == "configs":
        return "frozen_configuration"
    if path.parts[0] in {"scripts", "src"}:
        return "source_code"
    if path.suffix == ".md":
        return "documentation"
    return "public_support_file"


def _looks_local_only(relative: str) -> bool:
    path = Path(relative)
    lowered_parts = tuple(part.lower() for part in path.parts)
    if (
        lowered_parts
        and lowered_parts[0] in LOCAL_ONLY_DIRS
        and path.name != ".gitkeep"
    ):
        return True
    if any(
        part in {"cache", "caches"} or part.endswith("_cache") for part in lowered_parts
    ):
        return True
    name = path.name.lower()
    return re.search(r"patient[-_](level|visit|detail|prediction)", name) is not None


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        passed: bool,
        *,
        observed: int | float | str | None = None,
        expected: int | float | str | None = None,
    ) -> None:
        item: dict[str, Any] = {"name": name, "pass": bool(passed)}
        if observed is not None:
            item["observed"] = observed
        if expected is not None:
            item["expected"] = expected
        self.items.append(item)

    @property
    def passed(self) -> bool:
        return all(bool(item["pass"]) for item in self.items)


def _validate_public_file_policy(
    checks: Checks,
    repo_root: Path,
    public_files: Sequence[Path],
) -> None:
    relative_paths = [
        path.relative_to(EXPERIMENT_ROOT).as_posix() for path in public_files
    ]
    relative_set = set(relative_paths)
    checks.add(
        "required_public_files_present",
        REQUIRED_PUBLIC_FILES.issubset(relative_set),
        observed=len(REQUIRED_PUBLIC_FILES & relative_set),
        expected=len(REQUIRED_PUBLIC_FILES),
    )
    checks.add(
        "patient_detail_is_git_ignored",
        PATIENT_DETAIL_PATH.exists()
        and _git_ignored(repo_root, PATIENT_DETAIL_PATH)
        and "metrics/crop_containment_patient_visit.csv" not in relative_set,
    )

    unsafe_kind_count = 0
    oversize_count = 0
    symlink_count = 0
    privacy_violation_count = 0
    unknown_binary_count = 0
    for path, relative in zip(public_files, relative_paths):
        if path.is_symlink():
            symlink_count += 1
            continue
        if not path.is_file():
            unsafe_kind_count += 1
            continue
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            oversize_count += 1
        lower = relative.lower()
        if lower.endswith(RAW_OR_MODEL_SUFFIXES):
            unsafe_kind_count += 1
        if _looks_local_only(relative):
            unsafe_kind_count += 1
        if _sensitive_text(relative):
            privacy_violation_count += 1

        if path.suffix.lower() == ".png":
            try:
                _, _, metadata = _png_metadata(path)
                if any(_sensitive_text(value) for value in metadata):
                    privacy_violation_count += 1
            except ValueError:
                unsafe_kind_count += 1
        elif path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            ".gitignore",
            ".gitkeep",
        }:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                unknown_binary_count += 1
            else:
                if _sensitive_text(text):
                    privacy_violation_count += 1
        else:
            unknown_binary_count += 1

    checks.add("no_symlinks", symlink_count == 0, observed=symlink_count, expected=0)
    checks.add(
        "no_raw_mri_excel_or_model_artifacts",
        unsafe_kind_count == 0,
        observed=unsafe_kind_count,
        expected=0,
    )
    checks.add(
        "public_file_size_limit",
        oversize_count == 0,
        observed=oversize_count,
        expected=0,
    )
    checks.add(
        "known_public_file_types_only",
        unknown_binary_count == 0,
        observed=unknown_binary_count,
        expected=0,
    )
    checks.add(
        "no_patient_ids_or_host_absolute_paths",
        privacy_violation_count == 0,
        observed=privacy_violation_count,
        expected=0,
    )


def _validate_count_rate_rows(rows: Sequence[Mapping[str, str]]) -> bool:
    pairs = (
        ("diagnostic_support_n", "diagnostic_support_fraction"),
        ("origin_exact_n", "origin_exact_fraction"),
        ("origin_unique_n", "origin_unique_fraction"),
        ("origin_ambiguous_n", "origin_ambiguous_fraction"),
        ("spacing_reliable_n", "spacing_reliable_fraction"),
        ("complete_miss_n", "complete_miss_rate"),
        ("boundary_touch_n", "boundary_touch_rate"),
        ("suspected_truncation_n", "suspected_truncation_rate"),
        ("severe_truncation_n", "severe_truncation_rate"),
        ("sufficient_containment_n", "sufficient_containment_rate"),
        ("exact_full_support_containment_n", "exact_full_support_containment_rate"),
        ("bbox_fully_contained_n", "bbox_fully_contained_rate"),
        ("ld_zero_n", "ld_zero_fraction"),
    )
    for row in rows:
        denominator = _integer(row, "n")
        if denominator <= 0:
            return False
        for count_key, rate_key in pairs:
            if count_key not in row or rate_key not in row:
                continue
            count = _integer(row, count_key)
            rate = _number(row, rate_key)
            if (
                count < 0
                or count > denominator
                or not _close(rate, count / denominator)
            ):
                return False
    return True


def _validate_stage_a_metrics(checks: Checks) -> None:
    try:
        summary = _read_csv(METRICS_ROOT / "crop_containment_summary.csv")
        timepoint = _read_csv(METRICS_ROOT / "crop_containment_by_timepoint.csv")
        quantile = _read_csv(METRICS_ROOT / "crop_containment_by_ld_quantile.csv")
        distribution = _read_csv(METRICS_ROOT / "ld_containment_distribution.csv")
        provenance = _read_json(METRICS_ROOT / "stage_a_input_provenance.json")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, csv.Error):
        checks.add("aggregate_files_parse", False)
        return
    checks.add("aggregate_files_parse", True)

    try:
        all_row = _single(summary, scope="ALL_VISITS")
        early_row = _single(summary, scope="T0_T1")
        timepoint_rows = {
            visit: _single(timepoint, scope=visit) for visit in EXPECTED_VISITS
        }
        cohort_ok = (
            _integer(all_row, "n") == EXPECTED_PATIENT_VISITS
            and _integer(all_row, "n_patients") == EXPECTED_PATIENTS
            and _integer(early_row, "n") == 2 * EXPECTED_PATIENTS
            and set(row["scope"] for row in timepoint) == set(EXPECTED_VISITS)
            and sum(_integer(row, "n") for row in timepoint_rows.values())
            == EXPECTED_PATIENT_VISITS
            and all(
                _integer(row, "n") == EXPECTED_PATIENTS
                and _integer(row, "n_patients") == EXPECTED_PATIENTS
                for row in timepoint_rows.values()
            )
            and int(provenance["patients"]) == EXPECTED_PATIENTS
            and int(provenance["patient_visits"]) == EXPECTED_PATIENT_VISITS
            and provenance.get("pcr_read_for_stage_a") is False
        )
        checks.add("cohort_375_patients_1500_visits", cohort_ok)

        rate_ok = _validate_count_rate_rows(summary) and _validate_count_rate_rows(
            timepoint
        )
        quantile_pairs_ok = True
        for row in quantile:
            denominator = _integer(row, "n")
            for count_key, rate_key in (
                ("boundary_touch_n", "boundary_touch_rate"),
                ("suspected_truncation_n", "suspected_truncation_rate"),
                ("severe_truncation_n", "severe_truncation_rate"),
                (
                    "exact_full_support_containment_n",
                    "exact_full_support_containment_rate",
                ),
                ("bbox_fully_contained_n", "bbox_fully_contained_rate"),
            ):
                count = _integer(row, count_key)
                if (
                    count < 0
                    or count > denominator
                    or not _close(_number(row, rate_key), count / denominator)
                ):
                    quantile_pairs_ok = False
        checks.add("aggregate_count_rate_arithmetic", rate_ok and quantile_pairs_ok)

        additive_count_columns = (
            "n",
            "diagnostic_support_n",
            "origin_exact_n",
            "origin_unique_n",
            "origin_ambiguous_n",
            "spacing_reliable_n",
            "complete_miss_n",
            "boundary_touch_n",
            "suspected_truncation_n",
            "severe_truncation_n",
            "sufficient_containment_n",
            "exact_full_support_containment_n",
            "bbox_fully_contained_n",
            "ld_zero_n",
        )
        scope_visits = {
            "ALL_VISITS": EXPECTED_VISITS,
            "T0_T1": ("T0", "T1"),
            "T0_T1_T2": ("T0", "T1", "T2"),
            "T3": ("T3",),
        }
        additive_ok = True
        for scope, visits in scope_visits.items():
            aggregate = _single(summary, scope=scope)
            for column in additive_count_columns:
                expected = sum(
                    _integer(timepoint_rows[visit], column) for visit in visits
                )
                if _integer(aggregate, column) != expected:
                    additive_ok = False
        checks.add("summary_reconciles_with_timepoints", additive_ok)

        distribution_ok = True
        distribution_expected = {
            "T0": EXPECTED_PATIENTS,
            "T1": EXPECTED_PATIENTS,
            "T2": EXPECTED_PATIENTS,
            "T3": EXPECTED_PATIENTS,
            "ALL_VISITS": EXPECTED_PATIENT_VISITS,
        }
        for scope, expected_n in distribution_expected.items():
            rows = [row for row in distribution if row.get("scope") == scope]
            groups = {row.get("containment_group") for row in rows}
            if groups != {"SUFFICIENT_CONTAINMENT", "SUSPECTED_TRUNCATION"}:
                distribution_ok = False
            if sum(_integer(row, "n") for row in rows) != expected_n:
                distribution_ok = False
        checks.add("containment_distribution_reconciles", distribution_ok)

        subgroup_denominators: list[int] = []
        for rows in (summary, timepoint, quantile, distribution):
            for row in rows:
                for key in (
                    "n",
                    "n_patients",
                    "ld_margin_spearman_n",
                    "ld_containment_ratio_spearman_n",
                    "ld_approx_extent_spearman_n",
                ):
                    if key in row and row[key] != "":
                        subgroup_denominators.append(_integer(row, key))
        minimum_subgroup = min(subgroup_denominators)
        checks.add(
            "no_small_published_subgroups",
            minimum_subgroup >= MIN_PUBLIC_SUBGROUP_N,
            observed=minimum_subgroup,
            expected=f">={MIN_PUBLIC_SUBGROUP_N}",
        )

        forbidden_columns = {
            "patient_id",
            "clinical_patient_id",
            "subject_id",
            "mrn",
            "pcr",
            "rcb",
            "treatment_arm",
        }
        headers = set().union(
            *(
                set(row.keys())
                for rows in (summary, timepoint, quantile, distribution)
                for row in rows
            )
        )
        checks.add(
            "aggregate_tables_are_outcome_and_identifier_free",
            not bool(headers & forbidden_columns),
        )

        retention_ok = all(
            0.0 <= _number(row, key) <= 1.0
            for rows in (summary, timepoint, quantile)
            for row in rows
            for key in (
                "whole_union_extent_retention_median",
                "whole_union_extent_retention_q05",
            )
            if key in row and row[key] != ""
        )
        checks.add("public_extent_retention_within_unit_interval", retention_ok)
    except (KeyError, TypeError, ValueError):
        checks.add("stage_a_metric_semantics", False)
    else:
        checks.add("stage_a_metric_semantics", True)


def _criterion_pass(value: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<":
        return value < threshold
    raise ValueError("unsupported gate operator")


def _validate_gate(checks: Checks) -> None:
    try:
        config = _read_json(EXPERIMENT_ROOT / "configs" / "stage_a.json")
        gate = _read_json(METRICS_ROOT / "crop_containment_gate.json")
        summary = _read_csv(METRICS_ROOT / "crop_containment_summary.csv")
        quantile = _read_csv(METRICS_ROOT / "crop_containment_by_ld_quantile.csv")
        all_row = _single(summary, scope="ALL_VISITS")
        early_row = _single(summary, scope="T0_T1")
        top_quartile = _single(quantile, scope="T0_T1", ld_group="TOP_25_PERCENT")
        frozen = config["gate"]
        recomputed = {
            "t0_t1_suspected_truncation": (
                _number(early_row, "suspected_truncation_rate"),
                "<=",
                float(frozen["primary_suspected_truncation_rate_max"]),
            ),
            "t0_t1_top_quartile_suspected_truncation": (
                _number(top_quartile, "suspected_truncation_rate"),
                "<=",
                float(frozen["primary_top_quartile_suspected_truncation_rate_max"]),
            ),
            "all_visit_sufficient_containment": (
                _number(all_row, "sufficient_containment_rate"),
                ">=",
                float(frozen["combined_sufficient_containment_rate_min"]),
            ),
            "t0_t1_ld_margin_systematic_association": (
                _number(early_row, "ld_margin_spearman"),
                ">",
                float(frozen["strong_systematic_ld_margin_spearman_threshold"]),
            ),
            "exact_origin_recovery": (
                _number(all_row, "origin_exact_fraction"),
                ">=",
                float(frozen["minimum_exact_origin_recovery_fraction"]),
            ),
        }
        stored_criteria = gate["criteria"]
        gate_values_ok = set(stored_criteria) == set(recomputed)
        failed: list[str] = []
        for name, (observed, operator, threshold) in recomputed.items():
            passed = _criterion_pass(observed, operator, threshold)
            if not passed:
                failed.append(name)
            stored = stored_criteria[name]
            gate_values_ok = gate_values_ok and (
                _close(float(stored["observed"]), observed)
                and stored["operator"] == operator
                and _close(float(stored["threshold"]), threshold)
                and bool(stored["pass"]) == passed
            )
        checks.add("gate_values_recomputed_from_aggregates", gate_values_ok)

        all_pass = not failed
        if all_pass:
            decision_ok = (
                gate.get("decision") in {"GO", "GO_WITH_CAVEAT"}
                and gate.get("stage_b_authorized") is True
            )
        else:
            decision_ok = (
                gate.get("decision") == "NO_GO"
                and gate.get("stage_b_authorized") is False
                and gate.get("stop_code") == "LD_NOT_OBSERVABLE_UNDER_CURRENT_CROP"
            )
        decision_ok = decision_ok and set(gate.get("failed_criteria", [])) == set(
            failed
        )
        checks.add("gate_decision_matches_recomputed_criteria", decision_ok)
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        csv.Error,
    ):
        checks.add("gate_values_recomputed_from_aggregates", False)
        checks.add("gate_decision_matches_recomputed_criteria", False)


def _validate_final_no_go(checks: Checks) -> None:
    try:
        decision = _read_json(METRICS_ROOT / "final_decision.json")
        execution = _read_csv(METRICS_ROOT / "stage_execution_status.csv")
        answer_values = set(decision["answers"].values())
        decision_ok = (
            decision.get("overall_decision") == "NO_GO"
            and decision.get("stop_code") == "LD_NOT_OBSERVABLE_UNDER_CURRENT_CROP"
            and decision["stage_b"].get("authorized") is False
            and decision["stage_b"].get("executed") is False
            and decision["stage_b"].get("lambda_ld_selected") is None
            and decision["stage_b"].get("pcr_secondary_evaluation")
            == "SKIPPED_BY_STAGE_A_GATE_AND_PCR_NOT_READ"
            and len(decision["answers"]) == 10
            and "NOT_EVALUATED_PCR_NOT_READ" in answer_values
        )
        checks.add("final_decision_respects_no_go", decision_ok)

        components = {row["component"]: row for row in execution}
        expected = {
            "stage_a_crop_containment": "COMPLETED_NO_GO",
            "stage_b_dual_grounding_smoke": "SKIPPED_BY_STAGE_A_GATE",
            "stage_b_training": "SKIPPED_BY_STAGE_A_GATE",
            "representation_probes": "SKIPPED_BY_STAGE_A_GATE",
            "pcr_secondary_evaluation": "SKIPPED_BY_STAGE_A_GATE",
        }
        execution_ok = set(components) == set(expected) and all(
            components[name]["status"] == status
            and components[name]["outcome_or_target_data_used"].lower() == "false"
            for name, status in expected.items()
        )
        checks.add("skipped_stages_are_explicit", execution_ok)
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        csv.Error,
    ):
        checks.add("final_decision_respects_no_go", False)
        checks.add("skipped_stages_are_explicit", False)


def _validate_figures(checks: Checks) -> None:
    valid = 0
    for relative in EXPECTED_FIGURES:
        path = EXPERIMENT_ROOT / relative
        try:
            width, height, _ = _png_metadata(path)
            size = path.stat().st_size
            if (
                MIN_FIGURE_BYTES <= size <= MAX_PUBLIC_FILE_BYTES
                and MIN_FIGURE_WIDTH <= width <= MAX_FIGURE_DIMENSION
                and MIN_FIGURE_HEIGHT <= height <= MAX_FIGURE_DIMENSION
            ):
                valid += 1
        except (OSError, ValueError, struct.error):
            pass
    checks.add(
        "six_figures_present_and_sized",
        valid == len(EXPECTED_FIGURES),
        observed=valid,
        expected=len(EXPECTED_FIGURES),
    )


def _verification_document(checks: Checks, public_file_count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if checks.passed else "FAIL",
        "scope": "git_eligible_public_experiment_artifacts",
        "policy": {
            "manifest_self_hash": False,
            "maximum_public_file_bytes": MAX_PUBLIC_FILE_BYTES,
            "minimum_public_subgroup_n": MIN_PUBLIC_SUBGROUP_N,
            "patient_level_detail_public": False,
            "raw_mri_excel_or_model_artifacts_public": False,
        },
        "public_artifact_count_before_generated_outputs": public_file_count,
        "checks": checks.items,
    }


def _write_manifest(public_files: Iterable[Path]) -> None:
    rows: list[dict[str, str | int]] = []
    for path in sorted(
        set(public_files), key=lambda item: item.relative_to(EXPERIMENT_ROOT).as_posix()
    ):
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        if relative == "metrics/artifact_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "privacy_class": _privacy_class(relative),
            }
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=MANIFEST_PATH.parent,
        prefix=f".{MANIFEST_PATH.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(
            stream,
            fieldnames=("relative_path", "size_bytes", "sha256", "privacy_class"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, MANIFEST_PATH)


def main() -> int:
    try:
        repo_root = _repo_root()
        public_files = _git_public_files(repo_root)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        print("FAIL: public artifact inventory unavailable", file=sys.stderr)
        return 2

    checks = Checks()
    _validate_public_file_policy(checks, repo_root, public_files)
    _validate_stage_a_metrics(checks)
    _validate_gate(checks)
    _validate_final_no_go(checks)
    _validate_figures(checks)

    document = _verification_document(checks, len(public_files))
    serialized = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    # This should be impossible because the document contains only policy labels
    # and aggregate counts, but retain the same fail-closed rule for generated output.
    if _sensitive_text(serialized):
        print(
            "FAIL: generated verification record violated privacy policy",
            file=sys.stderr,
        )
        return 2
    _atomic_text(VERIFICATION_PATH, serialized)

    if not checks.passed:
        failed_names = [item["name"] for item in checks.items if not item["pass"]]
        print("FAIL: " + ", ".join(failed_names), file=sys.stderr)
        return 1

    _write_manifest([*public_files, VERIFICATION_PATH])
    print("PASS: metrics/public_artifact_verification.json")
    print("PASS: metrics/artifact_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
