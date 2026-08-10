"""Fail-closed, non-scientific closure checks for the frozen C1B audit.

This module deliberately does not calculate, compare, or reinterpret scientific
metrics.  It validates publication privacy, immutable-input hashes, matrix
inventories, filesystem permissions, report links, and the no-new-training
contract.  Public findings never include a patient identifier or a private
asset's path verbatim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import stat
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
UPSTREAM_RELATIVE = Path(
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb"
)

PUBLIC_ROOTS = ("configs", "manifests", "metrics", "reports")
ROOT_PUBLIC_FILES = ("EXPERIMENT_PLAN.md", "PREREGISTRATION_LOCK.json")
PUBLIC_TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".md",
    ".tex",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
CONTROL_OUTPUTS = {
    "metrics/privacy_gate.json",
    "metrics/final_validation.json",
}
PRIVATE_TREE_ROOTS = {"features", "probes", "logs"}

UPSTREAM_COMPLETION_PATHS = frozenset(
    {
        "checkpoints/formal_4x8_restart1/matrix_complete.json",
        "features/formal_4x8_restart1/feature_export_complete.json",
        "manifests/stage_b_data_contract.private.json",
        "metrics/stage_b_aggregation_summary.json",
        "predictions/formal_4x8_restart1/postprocessing_complete.json",
    }
)
UPSTREAM_SOURCE_PATHS = frozenset(
    {
        "additional_experiments/c1b_model_ready_ftv_sanity/src/c1b_sanity/geometry.py",
        "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/probes.py",
        "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/targets.py",
        "additional_experiments/g3_multiseed_generalization/src/dgrs/model.py",
    }
)

FIGURE_FILES = (
    "01_feature_map_pooling_schematic.png",
    "02_pooling_weight_illustration.png",
    "03_static_macro_spearman.png",
    "04_static_natural_r2.png",
    "05_delta_macro_spearman.png",
    "06_legacy_deficit_recovery.png",
    "07_oracle_vs_local_recovery.png",
    "08_padding_valid_source_decodability.png",
    "09_ftv_vs_nuisance_information.png",
    "10_occupancy_n1_l1_degradation.png",
    "11_selected_epoch_training_budget.png",
    "12_representative_activation_montage.png",
)

TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "table1_feature_map_contract.csv": (
        "stage", "analysis_role", "input_contract", "input_shape_zyx",
        "feature_channels", "feature_shape_zyx", "jump_input_voxels",
        "center_offset_input_voxels", "theoretical_receptive_field_input_voxels",
        "local_window_mm_xyz", "jump_x_mm_median", "jump_x_mm_q25",
        "jump_x_mm_q75", "jump_y_mm_median", "jump_y_mm_q25",
        "jump_y_mm_q75", "jump_z_mm_median", "jump_z_mm_q25",
        "jump_z_mm_q75", "spacing_basis",
    ),
    "table2_static_ftv.csv": (
        "stage", "analysis_role", "seed_base", "arm", "pooling", "endpoint",
        "analysis_scope", "availability", "status_reason", "feature_dim",
        "aggregation", "n_test", "spearman", "pearson", "natural_r2", "rmse",
        "mae", "b0_rmse", "rmse_gain_over_b0",
        "prediction_target_variance_ratio", "calibration_slope",
        "calibration_intercept", "calibration_mean_bias", "transformed_scale",
        "transformed_fold_count", "transformed_spearman_fold_mean",
        "transformed_spearman_fold_sd", "transformed_r2_fold_mean",
        "transformed_r2_fold_sd", "transformed_rmse_fold_mean",
        "transformed_mae_fold_mean",
    ),
    "table3_delta_ftv.csv": (),
    "table4_legacy_deficit_recovery.csv": (
        "stage", "analysis_role", "seed_base", "new_arm", "matched_legacy_arm",
        "pooling", "legacy_p0_spearman", "new_p0_spearman", "legacy_deficit",
        "pooling_spearman", "absolute_gain_vs_new_p0", "recovery_ratio",
        "recovery_defined", "status_reason",
    ),
    "table5_nuisance_decodability.csv": (
        "stage", "seed_base", "arm", "pooling", "target_name", "endpoint",
        "availability", "status_reason", "feature_dim", "aggregation", "n_test",
        "spearman", "pearson", "natural_r2", "rmse", "mae",
        "standardized_scale", "standardized_fold_count",
        "standardized_spearman_fold_mean", "standardized_r2_fold_mean",
        "standardized_r2_fold_sd",
    ),
    "table6_occupancy_downsampling.csv": (
        "analysis", "seed_base", "endpoint", "stratum", "n", "l1_spearman",
        "n1_spearman", "n1_minus_l1_spearman", "l1_mae", "n1_mae",
        "n1_minus_l1_mae", "mean_paired_abs_error_difference",
        "stratifier_error_difference_spearman", "status_reason",
    ),
    "table7_training_budget.csv": (
        "seed", "arm", "fold", "selected_epoch", "observed_max_epoch",
        "configured_max_epoch", "hit_configured_max_epoch",
        "selected_in_last_two_observed_epochs", "selected_validation_state_loss",
        "final_validation_state_loss", "final_minus_selected_state_loss",
        "last_three_normalized_validation_state_slope", "early_stopping_reason",
        "selection_mode", "optimization_safety_pass", "history_sha256",
        "selection_sha256",
    ),
}
# Tables 2 and 3 intentionally have the same exact public schema.
TABLE_SCHEMAS = {
    **TABLE_SCHEMAS,
    "table3_delta_ftv.csv": TABLE_SCHEMAS["table2_static_ftv.csv"],
}

NUISANCE_TARGETS = (
    "padding_fraction", "valid_source_fraction", "native_spacing_x_mm",
    "native_spacing_y_mm", "native_spacing_z_mm", "acquisition_fov_x_mm",
    "acquisition_fov_y_mm", "acquisition_fov_z_mm", "max_resample_factor",
    "resize_anisotropy",
)

IDENTIFIER_COLUMNS = {
    "patient_id",
    "patient_ids",
    "patient_token",
    "patient_tokens",
    "subject_id",
    "subject_ids",
    "subject_token",
    "subject_tokens",
    "pid",
    "mrn",
}
SENSITIVE_CSV_COLUMNS = IDENTIFIER_COLUMNS | {
    "accession_number",
    "cache_path",
    "checkpoint_path",
    "dce_nifti",
    "feature_metadata_path",
    "feature_path",
    "ftv_mask_nifti",
    "raw_dce_series_json",
    "resolved_dce_nifti",
    "series_uid",
    "sop_instance_uid",
    "source_path",
    "state",
    "state_valid",
    "study_uid",
    "support_mask",
    "valid_mask",
    "y_pred",
    "y_pred_analysis",
    "y_true",
    "y_true_analysis",
}

PATTERNS: Mapping[str, re.Pattern[str]] = {
    "absolute_workspace_path": re.compile(
        r"(?:"
        r"(?<![A-Za-z0-9:.])/(?:[A-Za-z0-9._-]+/)+[^\s`\"'<>),;]+"
        r"|(?<![A-Za-z0-9:.])/(?:root|Users)(?:/[^\s`\"'<>),;]*)?"
        r"|[A-Za-z]:\\[^\s`\"'<>]+"
        r"|\\\\[^\\\s`\"'<>]+\\[^\s`\"'<>]+"
        r")"
    ),
    "home_or_file_uri_path": re.compile(
        r"(?:file://[^\s`\"'<>]+|(?<![A-Za-z0-9])~/(?:[^\s`\"'<>]+)|"
        r"\$\{HOME\}(?:/[^\s`\"'<>]+)?)",
        re.IGNORECASE,
    ),
    "ispy_patient_identifier": re.compile(
        r"\b(?:ACRIN[-_ ]?6698[-_]|I[-_]?SPY[12]?[-_])\d{4,}\b",
        re.IGNORECASE,
    ),
    "dicom_uid": re.compile(r"\b\d{1,3}(?:\.\d+){5,}\b"),
    "json_identifier_value": re.compile(
        r'"(?:patient_id|patient_token|subject_id|subject_token|pid|mrn)"'
        r'\s*:\s*(?:"[^"\r\n]+"|\d+)',
        re.IGNORECASE,
    ),
}

_HTTP_URL = re.compile(r"https?://[^\s<>\]\[(){}\"']+", re.IGNORECASE)
_RAW_PRIVATE_REPORT_PATH = re.compile(
    r"(?:^|(?<=[\s`\"'(<]))(?:features|probes|logs)/[^\s`\"'<>),;]+"
    r"|(?:^|(?<=[\s`\"'(<]))[^\s`\"'<>),;]*\.private(?:\.[^\s`\"'<>),;]+)?",
    re.IGNORECASE | re.MULTILINE,
)

SEEDS = (2026, 3026)
ARMS = ("L1", "L3", "N1", "N3")
FOLDS = tuple(range(5))
FEATURE_POOLINGS = {
    "L1": ("P0", "PLOCAL", "PLOCAL+GLOBAL"),
    "L3": ("P0", "PLOCAL", "PLOCAL+GLOBAL"),
    "N1": (
        "P0",
        "PVALID",
        "PLOCAL",
        "PLOCAL+GLOBAL",
        "PORACLE",
        "PLOCAL+PVALID_SECONDARY",
    ),
    "N3": (
        "P0",
        "PVALID",
        "PLOCAL",
        "PLOCAL+GLOBAL",
        "PORACLE",
        "PLOCAL+PVALID_SECONDARY",
    ),
}
PRIMARY_PROBE_POOLINGS = ("P0", "PVALID", "PLOCAL", "PLOCAL+GLOBAL", "PORACLE")
FORMAL_PROBE_POOLINGS = (*PRIMARY_PROBE_POOLINGS, "PLOCAL+PVALID_SECONDARY")
POOLING_SLUGS = {
    "P0": "p0",
    "PVALID": "pvalid",
    "PLOCAL": "plocal",
    "PLOCAL+GLOBAL": "plocal_global",
    "PORACLE": "poracle",
    "PLOCAL+PVALID_SECONDARY": "plocal_pvalid_secondary",
}
TABLE_FILES = (
    "table1_feature_map_contract.csv",
    "table2_static_ftv.csv",
    "table3_delta_ftv.csv",
    "table4_legacy_deficit_recovery.csv",
    "table5_nuisance_decodability.csv",
    "table6_occupancy_downsampling.csv",
    "table7_training_budget.csv",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a public JSON file with exact mode ``0644``."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"refusing to overwrite {target}; pass --overwrite"
                )
            os.replace(temporary, target)
        else:
            # A hard-link publication is atomic and cannot silently clobber a
            # file that appeared after the first existence check.
            os.link(temporary, target)
            temporary.unlink()
        os.chmod(target, 0o644)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _has_private_component(relative: Path) -> bool:
    return any("private" in component.casefold() for component in relative.parts)


def public_text_paths(root: str | Path = EXPERIMENT_ROOT) -> list[Path]:
    """Return the exact public text scan surface in deterministic order."""

    experiment = Path(root).resolve()
    paths: set[Path] = set()
    for name in PUBLIC_ROOTS:
        base = experiment / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.casefold() not in PUBLIC_TEXT_SUFFIXES:
                continue
            relative = path.relative_to(experiment)
            if _has_private_component(relative) or relative.as_posix() in CONTROL_OUTPUTS:
                continue
            paths.add(path)
    for name in ROOT_PUBLIC_FILES:
        path = experiment / name
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(experiment).as_posix())


def _private_identifier_sources(root: Path) -> list[Path]:
    sources: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in {".csv", ".npz", ".tsv"}:
            continue
        if _has_private_component(path.relative_to(root)):
            sources.append(path)
    return sorted(sources, key=lambda path: path.relative_to(root).as_posix())


def _identifier_values_from_csv(path: Path) -> set[str]:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    values: set[str] = set()
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return values
        indices = [
            index
            for index, column in enumerate(header)
            if str(column).strip().casefold() in IDENTIFIER_COLUMNS
        ]
        if not indices:
            return values
        for row in reader:
            if any(index >= len(row) for index in indices):
                raise ValueError("private identifier CSV has a short row")
            for index in indices:
                value = str(row[index]).strip()
                if value:
                    values.add(value)
    return values


def _identifier_values_from_npz(path: Path) -> set[str]:
    values: set[str] = set()
    with np.load(path, allow_pickle=False) as archive:
        keys = [
            key for key in archive.files if str(key).strip().casefold() in IDENTIFIER_COLUMNS
        ]
        for key in keys:
            array = np.asarray(archive[key])
            if array.dtype.kind == "O":
                raise ValueError("object-valued patient identifier array is forbidden")
            for raw in array.reshape(-1):
                value = str(raw).strip()
                if value:
                    values.add(value)
    return values


def collect_private_patient_ids(root: str | Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Collect exact private identifiers without returning their values publicly."""

    experiment = Path(root).resolve()
    identifiers: set[str] = set()
    error_count = 0
    sources = _private_identifier_sources(experiment)
    identifier_sources = 0
    for path in sources:
        try:
            values = (
                _identifier_values_from_npz(path)
                if path.suffix.casefold() == ".npz"
                else _identifier_values_from_csv(path)
            )
        except (OSError, UnicodeError, ValueError, KeyError, csv.Error):
            error_count += 1
            continue
        if values:
            identifier_sources += 1
            identifiers.update(values)
    return {
        "values": identifiers,
        "private_csv_npz_sources_scanned": len(sources),
        "private_identifier_sources": identifier_sources,
        "private_identifier_source_errors": error_count,
    }


def _exact_identifier_count(payload: str, identifiers: Iterable[str]) -> int:
    count = 0
    for identifier in identifiers:
        # Literal comparison with alphanumeric token boundaries avoids matching
        # an identifier merely embedded in a digest or a longer token.
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(identifier)}(?![A-Za-z0-9])"
        )
        if pattern.search(payload):
            count += 1
    return count


def _safe_public_label(relative: str, identifiers: Iterable[str]) -> dict[str, str]:
    """Name an ordinary public file, but tokenize any identifier-bearing name."""

    unsafe = bool(
        PATTERNS["ispy_patient_identifier"].search(relative)
        or PATTERNS["dicom_uid"].search(relative)
        or _exact_identifier_count(relative, identifiers)
    )
    if unsafe:
        token = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        return {"file_token": token}
    return {"file": relative}


def _sensitive_csv_header_count(path: Path, payload: str) -> int:
    if path.suffix.casefold() not in {".csv", ".tsv"}:
        return 0
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    try:
        header = next(csv.reader(payload.splitlines(), delimiter=delimiter), [])
    except csv.Error:
        return 1
    return sum(
        str(column).strip().casefold() in SENSITIVE_CSV_COLUMNS for column in header
    )


def _path_token(value: str | Path) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _finding_for_file(relative: str, finding: str, **details: Any) -> dict[str, Any]:
    """Build a finding without ever serializing a path or filename."""

    return {"file_token": _path_token(relative), "finding": finding, **details}


def _strip_http_urls(payload: str) -> str:
    return _HTTP_URL.sub("", payload)


def _canonical_relative(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return None
    if raw.startswith(("/", "~", "${")) or re.match(r"^[A-Za-z]:", raw):
        return None
    candidate = Path(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    canonical = candidate.as_posix()
    if canonical != raw or posixpath.normpath(raw) != raw:
        return None
    return canonical


def _git_root_for(path: Path, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        repo = Path(explicit).resolve()
        return repo if (repo / ".git").exists() else None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _git_ignored(repo: Path, path: Path) -> bool | None:
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None
    result = _run_git(repo, ["check-ignore", "--no-index", "-q", "--", relative])
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _lock_path_contract_findings(lock: Any) -> list[str]:
    """Validate the only path disclosures allowed in the frozen public lock."""

    if not isinstance(lock, Mapping):
        return ["preregistration_lock_path_schema_invalid"]
    findings: list[str] = []
    cells = _expected_cell_names()
    selected = lock.get("selected_checkpoints")
    references = lock.get("formal_p0_references")
    if not isinstance(selected, Mapping) or set(selected) != cells:
        findings.append("preregistration_lock_checkpoint_path_inventory_drift")
        selected = {}
    if not isinstance(references, Mapping) or set(references) != cells:
        findings.append("preregistration_lock_reference_path_inventory_drift")
        references = {}
    for cell in sorted(cells):
        checkpoint = selected.get(cell)
        expected_checkpoint = (
            UPSTREAM_RELATIVE / "checkpoints/formal_4x8_restart1" / cell / "selected.pt"
        ).as_posix()
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            "path", "sha256", "size_bytes", "mtime_ns"
        } or _canonical_relative(checkpoint.get("path")) != expected_checkpoint:
            findings.append("preregistration_lock_checkpoint_path_drift")
        reference = references.get(cell)
        base = UPSTREAM_RELATIVE / "features/formal_4x8_restart1" / cell
        expected_feature = (base / "response_state.private.npz").as_posix()
        expected_metadata = (base / "response_state.private.metadata.json").as_posix()
        if not isinstance(reference, Mapping) or set(reference) != {
            "feature_path", "feature_sha256", "feature_metadata_path",
            "feature_metadata_sha256", "patient_order_sha256", "probe_outputs_sha256"
        }:
            findings.append("preregistration_lock_reference_path_schema_drift")
        elif (
            _canonical_relative(reference.get("feature_path")) != expected_feature
            or _canonical_relative(reference.get("feature_metadata_path"))
            != expected_metadata
        ):
            findings.append("preregistration_lock_reference_path_drift")
    completion = lock.get("upstream_completion_sha256")
    if not isinstance(completion, Mapping) or set(completion) != UPSTREAM_COMPLETION_PATHS:
        findings.append("preregistration_lock_completion_path_allowlist_drift")
    elif any(_canonical_relative(key) != key for key in completion):
        findings.append("preregistration_lock_completion_path_noncanonical")
    sources = lock.get("upstream_source_sha256")
    if not isinstance(sources, Mapping) or set(sources) != UPSTREAM_SOURCE_PATHS:
        findings.append("preregistration_lock_source_path_allowlist_drift")
    elif any(_canonical_relative(key) != key for key in sources):
        findings.append("preregistration_lock_source_path_noncanonical")
    return sorted(set(findings))


_PATH_KEY_EXCLUSIONS = frozenset(
    {
        "feature_dimension", "feature_dim", "feature_channels", "feature_shape_zyx",
        "valid_source_fraction", "padding_fraction", "source_commit",
        "status_reason", "target_semantics", "analysis_source", "legacy_pvalid",
        "legacy_poracle",
    }
)


def _looks_path_key(key: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = key.casefold()
    if lowered in _PATH_KEY_EXCLUSIONS or lowered.endswith("_sha256"):
        return False
    if lowered.endswith(("_path", "_root", "_dir", "_directory", "_file")):
        return True
    always_path_tokens = (
        "checkpoint", "cache", "feature", "metadata", "reference", "sidecar",
        "output_dir", "dicom", "nifti",
    )
    if any(token in lowered for token in always_path_tokens):
        return True
    return any(token in lowered for token in ("source", "support", "valid", "uid")) and (
        "/" in value or "\\" in value or value.startswith(("~", "${", "file:"))
    )


def _structured_json_path_findings(
    *,
    relative: str,
    payload: str,
    experiment: Path,
    repo: Path | None,
) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return [_finding_for_file(relative, "malformed_public_json")]
    if relative == "PREREGISTRATION_LOCK.json":
        return [
            _finding_for_file(relative, finding)
            for finding in _lock_path_contract_findings(value)
        ]
    findings: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key)
                if _looks_path_key(key, child):
                    if (
                        relative == "metrics/s3_trigger_authorization.json"
                        and key == "final_probe_root"
                        and child == "probes/final"
                    ):
                        pass
                    else:
                        canonical = _canonical_relative(child)
                        candidate = (
                            experiment / canonical
                            if canonical is not None
                            else None
                        )
                        is_private = bool(
                            candidate is not None
                            and _private_asset(Path(canonical), is_directory=candidate.is_dir())
                        )
                        ignored = (
                            _git_ignored(repo, candidate)
                            if repo is not None and candidate is not None
                            else None
                        )
                        expects_directory = key.casefold().endswith(
                            ("_root", "_dir", "_directory")
                        )
                        if (
                            canonical is None
                            or candidate is None
                            or not candidate.resolve().is_relative_to(experiment)
                            or not candidate.exists()
                            or candidate.is_symlink()
                            or (expects_directory and not candidate.is_dir())
                            or (not expects_directory and not candidate.is_file())
                            or is_private
                            or ignored is not False
                        ):
                            findings.append(
                                _finding_for_file(relative, "unsafe_public_json_path_value")
                            )
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return findings


def scan_public_artifacts(
    root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Scan all public text, returning only aggregate/private-safe findings."""

    experiment = Path(root).resolve()
    repo = _git_root_for(experiment, repo_root)
    private = collect_private_patient_ids(experiment)
    identifiers = private.pop("values")
    inventory: dict[str, tuple[Path, str, str]] = {}
    read_errors: list[dict[str, Any]] = []
    for path in public_text_paths(experiment):
        relative = path.relative_to(experiment).as_posix()
        try:
            payload = path.read_text(encoding="utf-8", errors="strict")
            inventory[relative] = (path, payload, file_sha256(path))
        except (OSError, UnicodeError):
            read_errors.append(
                _finding_for_file(
                    relative, "unreadable_public_text", match_count=1
                )
            )

    findings = list(read_errors)
    for relative in sorted(inventory):
        path, payload, _digest = inventory[relative]
        scan_payload = _strip_http_urls(payload)
        for name, pattern in PATTERNS.items():
            matches = sum(1 for _match in pattern.finditer(scan_payload))
            if matches:
                findings.append(
                    _finding_for_file(relative, name, match_count=matches)
                )
        exact = _exact_identifier_count(payload, identifiers)
        if exact:
            findings.append(
                _finding_for_file(
                    relative, "exact_private_patient_identifier", match_count=exact
                )
            )
        sensitive_columns = _sensitive_csv_header_count(path, payload)
        if sensitive_columns:
            findings.append(
                _finding_for_file(
                    relative, "sensitive_csv_column", match_count=sensitive_columns
                )
            )
        filename_exact = _exact_identifier_count(path.name, identifiers)
        if PATTERNS["ispy_patient_identifier"].search(path.name) or filename_exact:
            findings.append(
                _finding_for_file(
                    relative, "identifier_in_public_filename", match_count=1
                )
            )
        if path.suffix.casefold() == ".json":
            findings.extend(
                _structured_json_path_findings(
                    relative=relative,
                    payload=payload,
                    experiment=experiment,
                    repo=repo,
                )
            )
        if relative == "reports/final_report.md":
            private_path_matches = sum(
                1 for _match in _RAW_PRIVATE_REPORT_PATH.finditer(scan_payload)
            )
            if private_path_matches:
                findings.append(
                    _finding_for_file(
                        relative,
                        "raw_private_path_in_final_report",
                        match_count=private_path_matches,
                    )
                )

    stale_paths = sorted(
        relative
        for relative in inventory
        if "_smoke_" in Path(relative).name.casefold()
        or "_limited_" in Path(relative).name.casefold()
    )
    stale = [{"file_token": _path_token(relative)} for relative in stale_paths]
    hygiene = _audit_private_git_hygiene(experiment, repo)
    if hygiene["status"] != "PASS":
        findings.extend(hygiene["findings"])
    status = (
        "PASS"
        if not findings
        and not stale_paths
        and int(private["private_identifier_source_errors"]) == 0
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "scanned_public_text_artifacts": len(inventory),
        "scanned_files_sha256": {
            _path_token(relative): inventory[relative][2]
            for relative in sorted(inventory)
        },
        "identifier_path_or_column_findings": findings,
        "stale_smoke_or_limited_public_artifacts": stale,
        "contains_sensitive_identifiers_paths_or_columns": bool(findings),
        "private_identifier_values_checked": len(identifiers),
        "private_git_hygiene_checked": hygiene["status"] == "PASS",
        **private,
    }


def _run_git(repo: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _safe_repo_path(repo: Path, raw: Any) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("locked path is not repository-relative")
    resolved = (repo / relative).resolve()
    if not resolved.is_relative_to(repo.resolve()):
        raise ValueError("locked path escaped repository")
    return resolved


def _expected_cell_names() -> set[str]:
    return {
        f"seed_{seed}/{arm}/fold_{fold}"
        for seed in SEEDS
        for arm in ARMS
        for fold in FOLDS
    }


P0_EQUIVALENCE_FIELDS = frozenset(
    {
        "schema_version", "status", "formal_cells", "patients_per_cell",
        "visits_per_patient", "feature_dimension", "compared_elements",
        "allclose_required_fraction", "allclose_observed_fraction",
        "bitwise_equal_fraction", "maximum_absolute_error", "mean_absolute_error",
        "rtol", "atol", "preregistration_lock_sha256",
        "probe_execution_authorized",
    }
)
P0_EQUIVALENCE_COLUMNS = (
    "arm", "seed_base", "fold", "pooling", "patients", "visits", "feature_dim",
    "elements", "allclose_fraction", "bitwise_equal_fraction", "max_absolute_error",
    "mean_absolute_error", "rmse", "finite_fraction", "identity_exact",
    "split_exact", "state_valid_fraction", "rtol", "atol", "status",
    "candidate_sha256", "reference_sha256",
)
P0_REPLICATION_FIELDS = frozenset(
    {
        "schema_version", "status", "formal_cells", "selection_cells",
        "outer_test_prediction_rows", "selection_contract_exact_fraction",
        "prediction_contract_exact_fraction", "prediction_allclose_fraction",
        "maximum_prediction_absolute_difference", "pooled_natural_metric_rows",
        "maximum_pooled_metric_absolute_difference", "prediction_rtol",
        "prediction_atol", "pooled_metric_atol",
        "alternate_pooling_interpretation_authorized",
    }
)
P0_REPLICATION_CELL_COLUMNS = (
    "seed_base", "arm", "fold", "selection_rows", "prediction_rows",
    "selection_contract_exact", "prediction_keys_exact",
    "prediction_contract_exact", "prediction_allclose",
    "maximum_prediction_absolute_difference", "y_true_max_abs_difference",
    "y_pred_max_abs_difference", "y_true_analysis_max_abs_difference",
    "y_pred_analysis_max_abs_difference", "b0_prediction_max_abs_difference",
    "b0_prediction_analysis_max_abs_difference", "status",
)
P0_REPLICATION_METRIC_COLUMNS = (
    "seed_base", "arm", "task", "target_name", "endpoint", "analysis_scope",
    "target_semantics", "scale", "maximum_metric_absolute_difference", "status",
)


def _read_csv_exact(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise ValueError("CSV schema drift")
        rows = list(reader)
    if any(None in row or set(row) != set(columns) for row in rows):
        raise ValueError("CSV row width drift")
    return rows


def _csv_true(value: Any) -> bool:
    return str(value) == "True"


def audit_p0_gates(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate both STOP gates, their exact CSV evidence, and live bindings."""

    experiment = Path(experiment_root).resolve()
    repo = Path(repo_root).resolve()
    findings: list[dict[str, Any]] = []
    lock_path = experiment / "PREREGISTRATION_LOCK.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_sha = file_sha256(lock_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "invalid_preregistration_lock"}],
        }

    equivalence_path = experiment / "metrics/p0_equivalence_gate.json"
    equivalence_csv = experiment / "metrics/p0_equivalence_by_cell.csv"
    equivalence_rows: list[dict[str, str]] = []
    try:
        equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
        if not isinstance(equivalence, Mapping) or set(equivalence) != P0_EQUIVALENCE_FIELDS:
            raise ValueError
        expected = {
            "schema_version": 1,
            "status": "PASS",
            "formal_cells": 40,
            "patients_per_cell": 808,
            "visits_per_patient": 4,
            "feature_dimension": 192,
            "compared_elements": 24_821_760,
            "allclose_required_fraction": 1.0,
            "allclose_observed_fraction": 1.0,
            "rtol": 1e-5,
            "atol": 1e-6,
            "preregistration_lock_sha256": lock_sha,
            "probe_execution_authorized": True,
        }
        if any(equivalence.get(key) != value for key, value in expected.items()):
            raise ValueError
        for field in ("bitwise_equal_fraction", "maximum_absolute_error", "mean_absolute_error"):
            if not isinstance(equivalence.get(field), (int, float)):
                raise ValueError
        equivalence_rows = _read_csv_exact(equivalence_csv, P0_EQUIVALENCE_COLUMNS)
        identities: set[str] = set()
        elements = 0
        weighted_mean = 0.0
        bitwise_elements = 0.0
        maximum_error = 0.0
        for row in equivalence_rows:
            cell = f"seed_{int(row['seed_base'])}/{row['arm']}/fold_{int(row['fold'])}"
            if (
                cell not in _expected_cell_names()
                or row["pooling"] != "P0"
                or row["status"] != "PASS"
                or int(row["patients"]) != 808
                or int(row["visits"]) != 3232
                or int(row["feature_dim"]) != 192
                or int(row["elements"]) != 620_544
                or float(row["allclose_fraction"]) != 1.0
                or float(row["finite_fraction"]) != 1.0
                or float(row["state_valid_fraction"]) != 1.0
                or not _csv_true(row["identity_exact"])
                or not _csv_true(row["split_exact"])
                or float(row["rtol"]) != 1e-5
                or float(row["atol"]) != 1e-6
            ):
                raise ValueError
            candidate = (
                experiment / "features/final" / cell / "p0.private.npz"
            )
            reference = lock["formal_p0_references"][cell]
            if (
                row["candidate_sha256"] != file_sha256(candidate)
                or row["reference_sha256"] != reference["feature_sha256"]
                or row["reference_sha256"]
                != file_sha256(_safe_repo_path(repo, reference["feature_path"]))
            ):
                raise ValueError
            identities.add(cell)
            row_elements = int(row["elements"])
            elements += row_elements
            weighted_mean += float(row["mean_absolute_error"]) * row_elements
            bitwise_elements += float(row["bitwise_equal_fraction"]) * row_elements
            maximum_error = max(maximum_error, float(row["max_absolute_error"]))
        if len(equivalence_rows) != 40 or identities != _expected_cell_names():
            raise ValueError
        if (
            elements != equivalence["compared_elements"]
            or abs(weighted_mean / elements - float(equivalence["mean_absolute_error"])) > 1e-15
            or abs(bitwise_elements / elements - float(equivalence["bitwise_equal_fraction"]))
            > 1e-15
            or maximum_error != float(equivalence["maximum_absolute_error"])
        ):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        findings.append({"finding": "p0_equivalence_gate_or_csv_invalid"})

    replication_path = experiment / "metrics/p0_probe_replication_gate.json"
    replication_cells = experiment / "metrics/p0_probe_replication_by_cell.csv"
    replication_metrics = experiment / "metrics/p0_probe_replication_pooled_metrics.csv"
    replication_cell_rows: list[dict[str, str]] = []
    replication_metric_rows: list[dict[str, str]] = []
    try:
        replication = json.loads(replication_path.read_text(encoding="utf-8"))
        if not isinstance(replication, Mapping) or set(replication) != P0_REPLICATION_FIELDS:
            raise ValueError
        expected = {
            "schema_version": 1,
            "status": "PASS",
            "formal_cells": 40,
            "selection_cells": 560,
            "outer_test_prediction_rows": 41_704,
            "selection_contract_exact_fraction": 1.0,
            "prediction_contract_exact_fraction": 1.0,
            "prediction_allclose_fraction": 1.0,
            "pooled_natural_metric_rows": 144,
            "prediction_rtol": 1e-5,
            "prediction_atol": 1e-6,
            "pooled_metric_atol": 1e-6,
            "alternate_pooling_interpretation_authorized": True,
        }
        if any(replication.get(key) != value for key, value in expected.items()):
            raise ValueError
        replication_cell_rows = _read_csv_exact(
            replication_cells, P0_REPLICATION_CELL_COLUMNS
        )
        identities: set[str] = set()
        for row in replication_cell_rows:
            cell = f"seed_{int(row['seed_base'])}/{row['arm']}/fold_{int(row['fold'])}"
            if (
                cell not in _expected_cell_names()
                or row["status"] != "PASS"
                or int(row["selection_rows"]) != 14
                or any(
                    not _csv_true(row[field])
                    for field in (
                        "selection_contract_exact", "prediction_keys_exact",
                        "prediction_contract_exact", "prediction_allclose",
                    )
                )
            ):
                raise ValueError
            identities.add(cell)
        if (
            len(replication_cell_rows) != 40
            or identities != _expected_cell_names()
            or sum(int(row["selection_rows"]) for row in replication_cell_rows) != 560
            or sum(int(row["prediction_rows"]) for row in replication_cell_rows) != 41_704
            or max(
                float(row["maximum_prediction_absolute_difference"])
                for row in replication_cell_rows
            ) != float(replication["maximum_prediction_absolute_difference"])
        ):
            raise ValueError
        replication_metric_rows = _read_csv_exact(
            replication_metrics, P0_REPLICATION_METRIC_COLUMNS
        )
        if (
            len(replication_metric_rows) != 144
            or any(row["status"] != "PASS" for row in replication_metric_rows)
            or max(
                float(row["maximum_metric_absolute_difference"])
                for row in replication_metric_rows
            ) > 1e-6
            or max(
                float(row["maximum_metric_absolute_difference"])
                for row in replication_metric_rows
            ) != float(replication["maximum_pooled_metric_absolute_difference"])
        ):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        findings.append({"finding": "p0_probe_replication_gate_or_csv_invalid"})

    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "equivalence_cells_checked": len(equivalence_rows),
        "replication_cells_checked": len(replication_cell_rows),
        "replication_pooled_rows_checked": len(replication_metric_rows),
        "p0_equivalence_gate_sha256": (
            file_sha256(equivalence_path) if equivalence_path.is_file() else None
        ),
        "p0_probe_replication_gate_sha256": (
            file_sha256(replication_path) if replication_path.is_file() else None
        ),
    }


def audit_locked_inputs(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the old tree and every file hash frozen in preregistration."""

    experiment = Path(experiment_root).resolve()
    repo = Path(repo_root).resolve()
    findings: list[dict[str, Any]] = []
    checked = {
        "checkpoint_files": 0,
        "reference_feature_files": 0,
        "reference_probe_files": 0,
        "upstream_completion_files": 0,
        "upstream_source_files": 0,
    }
    lock_path = experiment / "PREREGISTRATION_LOCK.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "invalid_preregistration_lock"}],
            **checked,
        }
    if not isinstance(lock, dict):
        findings.append({"finding": "invalid_preregistration_lock_schema"})
        lock = {}
    findings.extend({"finding": value} for value in _lock_path_contract_findings(lock))
    mtime_mismatches = 0

    for filename, field in (
        ("EXPERIMENT_PLAN.md", "plan_sha256"),
        ("configs/audit.json", "config_sha256"),
    ):
        path = experiment / filename
        try:
            if file_sha256(path) != lock.get(field):
                findings.append({"finding": f"{field}_mismatch"})
        except OSError:
            findings.append({"finding": f"{field}_missing"})

    expected_cells = _expected_cell_names()
    selected = lock.get("selected_checkpoints")
    references = lock.get("formal_p0_references")
    if (
        lock.get("formal_cell_count") != 40
        or not isinstance(selected, dict)
        or set(selected) != expected_cells
    ):
        findings.append({"finding": "checkpoint_inventory_not_exact_40"})
        selected = selected if isinstance(selected, dict) else {}
    if not isinstance(references, dict) or set(references) != expected_cells:
        findings.append({"finding": "reference_inventory_not_exact_40"})
        references = references if isinstance(references, dict) else {}

    def check_file(raw_path: Any, expected_digest: Any, category: str, label: str) -> None:
        try:
            path = _safe_repo_path(repo, raw_path)
            digest = str(expected_digest)
            if len(digest) != 64 or file_sha256(path) != digest:
                findings.append({"finding": f"{category}_hash_mismatch", "cell": label})
                return
            checked[category] += 1
        except (OSError, ValueError, TypeError):
            findings.append({"finding": f"{category}_missing_or_unsafe", "cell": label})

    for cell in sorted(expected_cells):
        checkpoint = selected.get(cell)
        if isinstance(checkpoint, dict):
            before = checked["checkpoint_files"]
            check_file(
                checkpoint.get("path"),
                checkpoint.get("sha256"),
                "checkpoint_files",
                cell,
            )
            if checked["checkpoint_files"] > before:
                path = _safe_repo_path(repo, checkpoint["path"])
                if "size_bytes" in checkpoint and path.stat().st_size != int(
                    checkpoint["size_bytes"]
                ):
                    findings.append({"finding": "checkpoint_size_mismatch", "cell": cell})
                if "mtime_ns" in checkpoint and path.stat().st_mtime_ns != int(
                    checkpoint["mtime_ns"]
                ):
                    # Content, size, and Git tree are authoritative.  mtime is
                    # recorded only as non-binding operational provenance.
                    mtime_mismatches += 1
        else:
            findings.append({"finding": "checkpoint_record_missing", "cell": cell})

        reference = references.get(cell)
        if not isinstance(reference, dict):
            findings.append({"finding": "reference_record_missing", "cell": cell})
            continue
        check_file(
            reference.get("feature_path"),
            reference.get("feature_sha256"),
            "reference_feature_files",
            cell,
        )
        check_file(
            reference.get("feature_metadata_path"),
            reference.get("feature_metadata_sha256"),
            "reference_feature_files",
            cell,
        )
        probe_hashes = reference.get("probe_outputs_sha256")
        if not isinstance(probe_hashes, dict) or set(probe_hashes) != {
            "probe_metadata.json",
            "probe_metrics.csv",
            "ridge_predictions.private.csv",
            "ridge_selection.csv",
        }:
            findings.append({"finding": "reference_probe_hash_map_drift", "cell": cell})
            continue
        probe_dir = UPSTREAM_RELATIVE / "predictions/formal_4x8_restart1" / cell
        for filename, digest in sorted(probe_hashes.items()):
            check_file(
                probe_dir / filename,
                digest,
                "reference_probe_files",
                cell,
            )

    completion = lock.get("upstream_completion_sha256")
    if not isinstance(completion, dict) or set(completion) != UPSTREAM_COMPLETION_PATHS:
        findings.append({"finding": "upstream_completion_hash_map_drift"})
        completion = completion if isinstance(completion, dict) else {}
    for relative, digest in sorted(completion.items()):
        check_file(
            UPSTREAM_RELATIVE / str(relative),
            digest,
            "upstream_completion_files",
            "upstream_completion",
        )

    sources = lock.get("upstream_source_sha256")
    if not isinstance(sources, dict) or set(sources) != UPSTREAM_SOURCE_PATHS:
        findings.append({"finding": "upstream_source_hash_map_drift"})
        sources = sources if isinstance(sources, dict) else {}
    for relative, digest in sorted(sources.items()):
        check_file(
            relative,
            digest,
            "upstream_source_files",
            "upstream_source",
        )

    locked_tree = str(lock.get("upstream_tracked_tree", ""))
    source_commit = str(lock.get("source_commit", ""))
    tree_checks = {
        "source_commit_ancestor_of_head": False,
        "source_commit_tree_matches": False,
        "head_tree_matches": False,
        "live_tracked_tree_clean": False,
        "index_tracked_tree_clean": False,
        "no_nonignored_untracked_upstream_paths": False,
    }
    if len(locked_tree) != 40 or len(source_commit) != 40:
        findings.append({"finding": "git_tree_or_source_commit_malformed"})
    else:
        ancestor = _run_git(repo, ["merge-base", "--is-ancestor", source_commit, "HEAD"])
        tree_checks["source_commit_ancestor_of_head"] = ancestor.returncode == 0
        source_tree = _run_git(
            repo, ["rev-parse", f"{source_commit}:{UPSTREAM_RELATIVE.as_posix()}"]
        )
        head_tree = _run_git(repo, ["rev-parse", f"HEAD:{UPSTREAM_RELATIVE.as_posix()}"])
        tree_checks["source_commit_tree_matches"] = (
            source_tree.returncode == 0 and source_tree.stdout.strip() == locked_tree
        )
        tree_checks["head_tree_matches"] = (
            head_tree.returncode == 0 and head_tree.stdout.strip() == locked_tree
        )
        live = _run_git(
            repo,
            ["diff", "--quiet", source_commit, "--", UPSTREAM_RELATIVE.as_posix()],
        )
        index = _run_git(
            repo,
            [
                "diff",
                "--cached",
                "--quiet",
                source_commit,
                "--",
                UPSTREAM_RELATIVE.as_posix(),
            ],
        )
        tree_checks["live_tracked_tree_clean"] = live.returncode == 0
        tree_checks["index_tracked_tree_clean"] = index.returncode == 0
        untracked = _run_git(
            repo,
            [
                "ls-files", "--others", "--exclude-standard", "--",
                UPSTREAM_RELATIVE.as_posix(),
            ],
        )
        tree_checks["no_nonignored_untracked_upstream_paths"] = (
            untracked.returncode == 0 and not untracked.stdout.strip()
        )
        for name, passed in tree_checks.items():
            if not passed:
                findings.append({"finding": name})

    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        **checked,
        "expected_formal_cells": 40,
        "checkpoint_mtime_mismatches_informational": mtime_mismatches,
        "git_tree_checks": tree_checks,
        "preregistration_lock_sha256": file_sha256(lock_path),
    }


def _feature_records(experiment: Path, repo: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                cell = f"seed_{seed}/{arm}/fold_{fold}"
                for pooling in FEATURE_POOLINGS[arm]:
                    slug = POOLING_SLUGS[pooling]
                    asset = (
                        experiment
                        / "features/final"
                        / f"seed_{seed}"
                        / arm
                        / f"fold_{fold}"
                        / f"{slug}.private.npz"
                    )
                    metadata = asset.with_suffix(".metadata.json")
                    key = metadata.resolve().relative_to(repo.resolve()).as_posix()
                    records[key] = {
                        "cell": cell,
                        "seed_base": seed,
                        "arm": arm,
                        "fold": fold,
                        "pooling": pooling,
                        "asset": asset,
                        "metadata": metadata,
                    }
    if len(records) != 180:
        raise AssertionError("frozen feature inventory must be exactly 180")
    return records


def _probe_records(experiment: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                cell = f"seed_{seed}/{arm}/fold_{fold}"
                for pooling in FORMAL_PROBE_POOLINGS:
                    if pooling in {
                        "PVALID",
                        "PORACLE",
                        "PLOCAL+PVALID_SECONDARY",
                    } and arm.startswith("L"):
                        continue
                    slug = POOLING_SLUGS[pooling]
                    output = (
                        experiment
                        / "probes/final"
                        / f"seed_{seed}"
                        / arm
                        / f"fold_{fold}"
                        / slug
                    )
                    key = f"{cell}/{slug}"
                    records[key] = {
                        "cell": cell,
                        "seed_base": seed,
                        "arm": arm,
                        "fold": fold,
                        "pooling": pooling,
                        "slug": slug,
                        "output": output,
                    }
    if len(records) != 180:
        raise AssertionError("frozen formal probe inventory must be exactly 180")
    return records


def _path_value_matches(value: Any, expected: Path, repo: Path, context: Path) -> bool:
    candidate = Path(str(value)).expanduser()
    possibilities = (
        {candidate.resolve()}
        if candidate.is_absolute()
        else {(repo / candidate).resolve(), (context / candidate).resolve()}
    )
    return expected.resolve() in possibilities


def _audit_final_inventories_legacy(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate the exact final 40-cell, 180-feature, and 180-probe inventories."""

    experiment = Path(experiment_root).resolve()
    repo = Path(repo_root).resolve()
    findings: list[dict[str, Any]] = []
    try:
        lock_path = experiment / "PREREGISTRATION_LOCK.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_sha = file_sha256(lock_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "invalid_preregistration_lock"}],
            "expected_checkpoint_cells": 40,
            "expected_feature_assets": 180,
            "expected_formal_probe_cells": 180,
        }

    expected_cells = _expected_cell_names()
    if set(lock.get("selected_checkpoints", {})) != expected_cells:
        findings.append({"finding": "formal_checkpoint_inventory_not_exact_40"})

    feature_records = _feature_records(experiment, repo)
    feature_completion_path = experiment / "features/feature_export_complete.private.json"
    feature_valid = 0
    try:
        completion = json.loads(feature_completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        completion = {}
        findings.append({"finding": "feature_completion_missing_or_invalid"})
    inventory = completion.get("feature_metadata_sha256")
    if (
        completion.get("status") != "COMPLETE"
        or completion.get("stage") != "final"
        or completion.get("run_count") != 40
        or completion.get("cell_count") != 40
        or completion.get("expected_asset_count") != 180
        or completion.get("preregistration_lock_sha256") != lock_sha
        or not isinstance(inventory, dict)
        or set(inventory) != set(feature_records)
    ):
        findings.append({"finding": "feature_completion_contract_drift"})
        inventory = inventory if isinstance(inventory, dict) else {}

    live_metadata = {
        path.resolve().relative_to(repo).as_posix()
        for path in (experiment / "features/final").rglob("*.private.metadata.json")
    } if (experiment / "features/final").is_dir() else set()
    live_assets = set((experiment / "features/final").rglob("*.private.npz")) \
        if (experiment / "features/final").is_dir() else set()
    if live_metadata != set(feature_records) or len(live_assets) != 180:
        findings.append({"finding": "live_feature_inventory_not_exact_180"})

    selected = lock.get("selected_checkpoints", {})
    references = lock.get("formal_p0_references", {})
    for relative, expected in feature_records.items():
        metadata_path = expected["metadata"]
        asset_path = expected["asset"]
        label = f'{expected["cell"]}/{POOLING_SLUGS[expected["pooling"]]}'
        try:
            if file_sha256(metadata_path) != inventory.get(relative):
                raise ValueError("metadata digest")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            identity = (
                metadata.get("status") == "COMPLETE"
                and metadata.get("stage") == "final"
                and metadata.get("seed_base") == expected["seed_base"]
                and metadata.get("arm") == expected["arm"]
                and metadata.get("fold") == expected["fold"]
                and metadata.get("pooling") == expected["pooling"]
            )
            if not identity or not _path_value_matches(
                metadata.get("feature_path"), asset_path, repo, metadata_path.parent
            ):
                raise ValueError("metadata identity")
            if file_sha256(asset_path) != metadata.get("feature_sha256"):
                raise ValueError("asset digest")
            cell = expected["cell"]
            if metadata.get("checkpoint_sha256") != selected[cell]["sha256"]:
                raise ValueError("checkpoint binding")
            if (
                metadata.get("reference_feature_sha256")
                != references[cell]["feature_sha256"]
                or metadata.get("reference_feature_metadata_sha256")
                != references[cell]["feature_metadata_sha256"]
                or metadata.get("preregistration_lock_sha256") != lock_sha
                or metadata.get("plan_sha256") != lock.get("plan_sha256")
                or metadata.get("config_sha256") != lock.get("config_sha256")
            ):
                raise ValueError("frozen binding")
            forbidden = (
                "training_performed",
                "projector_called",
                "transition_called",
                "target_encoder_called",
                "ftv_head_called",
                "test_labels_used",
            )
            if any(metadata.get(field) is not False for field in forbidden):
                raise ValueError("forbidden execution")
            feature_valid += 1
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            findings.append({"finding": "feature_asset_or_binding_invalid", "cell": label})

    probe_records = _probe_records(experiment)
    probe_feature_records = {
        f'{record["cell"]}/{POOLING_SLUGS[record["pooling"]]}': record
        for record in feature_records.values()
        if record["pooling"] in FORMAL_PROBE_POOLINGS
    }
    completion_paths = sorted(
        (experiment / "probes/final").glob("probe_matrix_*_complete.private.json")
    ) if (experiment / "probes/final").is_dir() else []
    union: dict[str, Mapping[str, Any]] = {}
    probe_valid = 0
    for completion_path in completion_paths:
        try:
            probe_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            findings.append({"finding": "probe_completion_invalid"})
            continue
        cells = probe_completion.get("cells")
        requested_poolings = probe_completion.get("requested_poolings")
        completion_feature_hashes = probe_completion.get("feature_metadata_sha256")
        if (
            probe_completion.get("status") != "COMPLETE"
            or probe_completion.get("stage") != "final"
            or probe_completion.get("new_training_performed") is not False
            or not isinstance(cells, dict)
            or not isinstance(requested_poolings, list)
            or len(requested_poolings) != len(set(requested_poolings))
            or not set(requested_poolings).issubset(FORMAL_PROBE_POOLINGS)
            or not isinstance(completion_feature_hashes, dict)
            or set(completion_feature_hashes) != set(cells)
            or probe_completion.get("executed_cell_count") != len(cells)
            or probe_completion.get("expected_cell_count") != len(cells)
            or probe_completion.get("preregistration_lock_sha256") != lock_sha
        ):
            findings.append({"finding": "probe_completion_contract_drift"})
            continue
        expected_completion_cells = {
            key
            for key, record in probe_records.items()
            if record["pooling"] in requested_poolings
        }
        if set(cells) != expected_completion_cells:
            findings.append({"finding": "probe_completion_requested_pooling_drift"})
            continue
        completion_binding_failed = False
        for key, digest in completion_feature_hashes.items():
            feature_record = probe_feature_records.get(key)
            try:
                if feature_record is None or file_sha256(feature_record["metadata"]) != digest:
                    raise ValueError
            except (OSError, ValueError):
                completion_binding_failed = True
                break
        if completion_binding_failed:
            findings.append({"finding": "probe_completion_feature_binding_drift"})
            continue
        overlap = set(union).intersection(cells)
        if overlap:
            findings.append({"finding": "duplicate_probe_cell_across_completions"})
        for key, value in cells.items():
            if key not in union and isinstance(value, Mapping):
                union[key] = value

    if set(union) != set(probe_records):
        findings.append({"finding": "probe_inventory_not_exact_180"})

    live_probe_metadata = set(
        (experiment / "probes/final").rglob("probe_metadata.json")
    ) if (experiment / "probes/final").is_dir() else set()
    expected_probe_metadata = {
        record["output"] / "probe_metadata.json" for record in probe_records.values()
    }
    if live_probe_metadata != expected_probe_metadata:
        findings.append({"finding": "live_probe_metadata_inventory_not_exact_180"})

    for key, expected in probe_records.items():
        row = union.get(key)
        if not isinstance(row, Mapping):
            continue
        output = expected["output"]
        try:
            feature_record = probe_feature_records[key]
            if (
                row.get("seed_base") != expected["seed_base"]
                or row.get("arm") != expected["arm"]
                or row.get("fold") != expected["fold"]
                or row.get("pooling") != expected["pooling"]
                or not _path_value_matches(row.get("output_dir"), output, repo, output.parent)
                or not _path_value_matches(
                    row.get("feature_path"),
                    feature_record["asset"],
                    repo,
                    output,
                )
                or not _path_value_matches(
                    row.get("feature_metadata_path"),
                    feature_record["metadata"],
                    repo,
                    output,
                )
                or row.get("feature_sha256") != file_sha256(feature_record["asset"])
                or row.get("feature_metadata_sha256")
                != file_sha256(feature_record["metadata"])
                or row.get("nuisance_included")
                is not (expected["pooling"] in {"P0", "PVALID", "PLOCAL"})
            ):
                raise ValueError("probe identity")
            metadata_path = output / "probe_metadata.json"
            if file_sha256(metadata_path) != row.get("probe_metadata_sha256"):
                raise ValueError("probe metadata digest")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            output_hashes = row.get("output_sha256")
            if output_hashes != metadata.get("output_sha256") or not isinstance(
                output_hashes, dict
            ):
                raise ValueError("probe output map")
            if set(output_hashes) != {
                "probe_metrics.csv",
                "ridge_predictions.private.csv",
                "ridge_selection.csv",
            }:
                raise ValueError("probe output names")
            for filename, digest in output_hashes.items():
                if file_sha256(output / filename) != digest:
                    raise ValueError("probe output digest")
            if (
                metadata.get("seed_base") != expected["seed_base"]
                or metadata.get("arm") != expected["arm"]
                or metadata.get("fold") != expected["fold"]
                or metadata.get("pooling") != expected["pooling"]
                or metadata.get("patient_identifiers_private") is not True
                or metadata.get("test_used_for_scaler_or_selection") is not False
                or metadata.get("feature_sha256") != row.get("feature_sha256")
                or metadata.get("feature_metadata_sha256")
                != row.get("feature_metadata_sha256")
            ):
                raise ValueError("probe metadata contract")
            probe_valid += 1
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            findings.append({"finding": "probe_asset_or_binding_invalid", "cell": key})

    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "expected_checkpoint_cells": 40,
        "expected_feature_assets": 180,
        "validated_feature_assets": feature_valid,
        "expected_formal_probe_cells": 180,
        "validated_formal_probe_cells": probe_valid,
        "primary_probe_cells": 160,
        "secondary_local_valid_probe_cells": 20,
        "probe_completion_inventories": len(completion_paths),
        "secondary_local_valid_feature_assets": 20,
        "secondary_local_valid_probe_status": "REQUIRED_DESCRIPTIVE_SECONDARY",
    }


FINAL_FEATURE_COMPLETION_FIELDS = frozenset(
    {
        "schema_version", "status", "stage", "run_count", "expected_asset_count",
        "cell_count", "feature_metadata_sha256", "preflight_sha256", "sidecar_sha256",
        "preregistration_lock_sha256",
    }
)
FINAL_FEATURE_CLAIM_FIELDS = frozenset(
    {
        "schema_version", "status", "stage", "nonresumable", "cell_count",
        "expected_asset_count", "preregistration_lock_sha256", "sidecar_sha256",
        "matrix_driver_sha256",
    }
)
FINAL_FEATURE_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version", "status", "stage", "cell_count", "expected_asset_count",
        "preregistration_lock_sha256", "sidecar_path", "sidecar_sha256",
        "stage_a_sentinel_sha256", "feature_root", "scheduler", "cell_inventory",
        "code_sha256", "python_executable", "execution_requested", "claim_sha256",
    }
)
FINAL_FEATURE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "stage", "status", "arm", "seed_base", "fold", "pooling",
        "pooling_slug", "feature_path", "feature_sha256", "state_shape", "state_dtype",
        "state_valid_shape", "state_valid_count", "patient_count", "patient_order_sha256",
        "split_order_sha256", "checkpoint_path", "checkpoint_sha256",
        "checkpoint_lock_key", "reference_feature_path", "reference_feature_sha256",
        "reference_feature_metadata_path", "reference_feature_metadata_sha256",
        "preregistration_lock_sha256", "plan_sha256", "config_sha256", "sidecar_path",
        "sidecar_sha256", "sidecar_keys_used", "data_contract_provenance_sha256",
        "checkpoint_data_provenance_sha256", "stage_a_sentinel_sha256",
        "implementation_sha256", "device", "batch_size", "workers", "feature_tensor",
        "response_projection", "training_performed", "projector_called",
        "transition_called", "target_encoder_called", "ftv_head_called",
        "test_labels_used",
    }
)
PROBE_COMPLETION_FIELDS = frozenset(
    {
        "schema_version", "status", "stage", "representation_contract",
        "requested_poolings", "expected_cell_count", "nuisance_cell_count",
        "nuisance_targets", "feature_root", "probe_root", "completion_path",
        "preregistration_lock_sha256", "nuisance_sha256", "gate_sha256",
        "exporter_completion_sha256", "legacy_pvalid", "legacy_poracle",
        "fabricated_unavailable_rows", "executed_cell_count", "workers",
        "probe_runner_sha256", "feature_metadata_sha256", "cells",
        "patient_identifiers_private", "new_training_performed",
    }
)
PROBE_COMPLETION_CELL_FIELDS = frozenset(
    {
        "seed_base", "arm", "fold", "pooling", "pooling_slug", "feature_path",
        "feature_sha256", "feature_metadata_path", "feature_metadata_sha256",
        "output_dir", "probe_metadata_sha256", "output_sha256", "selection_rows",
        "prediction_rows", "metric_rows", "nuisance_included",
    }
)
PROBE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "seed_base", "arm", "fold", "pooling", "feature_dim",
        "feature_path", "feature_sha256", "feature_metadata_path",
        "feature_metadata_sha256", "patient_identifiers_private",
        "prediction_asset_private", "state_valid_enforced",
        "test_used_for_scaler_or_selection", "outer_test_predict_calls_per_cell",
        "alpha_grid", "ridge_selection_implementation", "static_target_implementation",
        "formal_probe_source_sha256", "formal_target_source_sha256",
        "probe_adapter_sha256", "provenance", "provenance_sha256", "tasks",
        "target_names", "selection_rows", "prediction_rows", "metric_rows",
        "output_sha256",
    }
)
FINAL_ALT_POOLINGS = (
    "PVALID", "PLOCAL", "PLOCAL+GLOBAL", "PORACLE", "PLOCAL+PVALID_SECONDARY"
)
FINAL_PROBE_COMPLETIONS = {
    "probe_matrix_p0_complete.private.json": ("P0",),
    (
        "probe_matrix_pvalid_plocal_plocal_global_poracle_"
        "plocal_pvalid_secondary_complete.private.json"
    ): FINAL_ALT_POOLINGS,
}


def _is_sha256(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value))


def _feature_sidecar_keys(arm: str, pooling: str) -> list[str]:
    if pooling == "P0":
        return []
    if pooling in {"PLOCAL", "PLOCAL+GLOBAL"}:
        return ["legacy_local_weight_final" if arm.startswith("L") else "c1b_local_weight_final"]
    if pooling == "PVALID":
        return ["c1b_valid_weight_final"]
    if pooling == "PORACLE":
        return ["c1b_oracle_weight_final", "c1b_oracle_valid"]
    if pooling == "PLOCAL+PVALID_SECONDARY":
        return ["c1b_local_weight_final", "c1b_valid_weight_final"]
    raise ValueError("unknown pooling")


def _expected_feature_code(experiment: Path) -> dict[str, Path]:
    return {
        "matrix_driver": experiment / "scripts/run_feature_matrix.py",
        "feature_cli": experiment / "scripts/export_frozen_features.py",
        "exporter": experiment / "src/c1b_spatial_audit/exporter.py",
        "pooling": experiment / "src/c1b_spatial_audit/pooling.py",
        "runtime": experiment / "src/c1b_spatial_audit/runtime.py",
        "contracts": experiment / "src/c1b_spatial_audit/contracts.py",
    }


def _validate_feature_controls(
    experiment: Path,
    repo: Path,
    lock: Mapping[str, Any],
    lock_sha: str,
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    root = experiment / "features"
    claim_path = root / "feature_export_claim.private.json"
    preflight_path = root / "feature_export_preflight.private.json"
    completion_path = root / "feature_export_complete.private.json"
    completion: dict[str, Any] = {}
    try:
        controls = {
            path.name for path in root.glob("feature_export_*.private.json")
        }
        if controls != {
            claim_path.name, preflight_path.name, completion_path.name
        }:
            raise ValueError
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if set(claim) != FINAL_FEATURE_CLAIM_FIELDS:
            raise ValueError
        if set(preflight) != FINAL_FEATURE_PREFLIGHT_FIELDS:
            raise ValueError
        if set(completion) != FINAL_FEATURE_COMPLETION_FIELDS:
            raise ValueError
        sidecar = experiment / "manifests/audit_sidecars.private.npz"
        sentinel = repo / UPSTREAM_RELATIVE / "STAGE_A_GO.json"
        code = _expected_feature_code(experiment)
        live_code = {name: file_sha256(path) for name, path in code.items()}
        if claim != {
            "schema_version": 1,
            "status": "CLAIMED",
            "stage": "final",
            "nonresumable": True,
            "cell_count": 40,
            "expected_asset_count": 180,
            "preregistration_lock_sha256": lock_sha,
            "sidecar_sha256": file_sha256(sidecar),
            "matrix_driver_sha256": live_code["matrix_driver"],
        }:
            raise ValueError
        scheduler = preflight.get("scheduler")
        devices = scheduler.get("devices") if isinstance(scheduler, Mapping) else None
        if (
            preflight.get("schema_version") != 1
            or preflight.get("status") != "PREFLIGHT_PASS"
            or preflight.get("stage") != "final"
            or preflight.get("cell_count") != 40
            or preflight.get("expected_asset_count") != 180
            or preflight.get("preregistration_lock_sha256") != lock_sha
            or not _path_value_matches(preflight.get("sidecar_path"), sidecar, repo, root)
            or preflight.get("sidecar_sha256") != file_sha256(sidecar)
            or preflight.get("stage_a_sentinel_sha256") != file_sha256(sentinel)
            or not _path_value_matches(preflight.get("feature_root"), root, repo, root)
            or preflight.get("code_sha256") != live_code
            or preflight.get("execution_requested") is not True
            or preflight.get("claim_sha256") != file_sha256(claim_path)
            or not isinstance(preflight.get("python_executable"), str)
            or not preflight["python_executable"]
            or not isinstance(scheduler, Mapping)
            or set(scheduler) != {
                "devices", "parallel_processes", "one_sequential_stream_per_device",
                "batch_size", "workers", "fail_fast_process_group_termination",
            }
            or not isinstance(devices, list)
            or not devices
            or len(devices) != len(set(devices))
            or any(not isinstance(device, str) or not re.fullmatch(r"cuda:\d+", device) for device in devices)
            or scheduler.get("parallel_processes") != len(devices)
            or scheduler.get("one_sequential_stream_per_device") is not True
            or scheduler.get("batch_size") != 4
            or scheduler.get("workers") != 2
            or scheduler.get("fail_fast_process_group_termination") is not True
        ):
            raise ValueError
        cells = preflight.get("cell_inventory")
        if not isinstance(cells, list) or len(cells) != 40:
            raise ValueError
        observed: set[str] = set()
        for index, row in enumerate(cells):
            if not isinstance(row, Mapping) or set(row) != {
                "index", "seed_base", "arm", "fold", "device"
            } or row.get("index") != index or row.get("device") != devices[index % len(devices)]:
                raise ValueError
            observed.add(f"seed_{row['seed_base']}/{row['arm']}/fold_{row['fold']}")
        if observed != _expected_cell_names():
            raise ValueError
        inventory = completion.get("feature_metadata_sha256")
        if (
            completion.get("schema_version") != 1
            or completion.get("status") != "COMPLETE"
            or completion.get("stage") != "final"
            or completion.get("run_count") != 40
            or completion.get("cell_count") != 40
            or completion.get("expected_asset_count") != 180
            or completion.get("preregistration_lock_sha256") != lock_sha
            or completion.get("preflight_sha256") != file_sha256(preflight_path)
            or completion.get("sidecar_sha256") != file_sha256(sidecar)
            or not isinstance(inventory, Mapping)
            or set(inventory) != set(records)
        ):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        findings.append({"finding": "feature_control_schema_or_live_binding_invalid"})
    return completion, findings


def _validate_final_feature_assets(
    experiment: Path,
    repo: Path,
    lock: Mapping[str, Any],
    lock_sha: str,
    records: Mapping[str, Mapping[str, Any]],
    completion: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    inventory = completion.get("feature_metadata_sha256")
    inventory = inventory if isinstance(inventory, Mapping) else {}
    expected_assets = {record["asset"].resolve() for record in records.values()}
    expected_metadata = {record["metadata"].resolve() for record in records.values()}
    stage_root = experiment / "features/final"
    live_assets = {path.resolve() for path in stage_root.rglob("*.private.npz")} if stage_root.is_dir() else set()
    live_metadata = {path.resolve() for path in stage_root.rglob("*.private.metadata.json")} if stage_root.is_dir() else set()
    live_files = {path.resolve() for path in stage_root.rglob("*") if path.is_file() or path.is_symlink()} if stage_root.is_dir() else set()
    if (
        live_assets != expected_assets
        or live_metadata != expected_metadata
        or live_files != expected_assets | expected_metadata
    ):
        findings.append({"finding": "live_feature_inventory_not_exact_180"})
    implementation_paths = {
        "exporter.py": experiment / "src/c1b_spatial_audit/exporter.py",
        "pooling.py": experiment / "src/c1b_spatial_audit/pooling.py",
        "runtime.py": experiment / "src/c1b_spatial_audit/runtime.py",
        "contracts.py": experiment / "src/c1b_spatial_audit/contracts.py",
    }
    sidecar = experiment / "manifests/audit_sidecars.private.npz"
    sentinel = repo / UPSTREAM_RELATIVE / "STAGE_A_GO.json"
    try:
        implementation = {
            name: file_sha256(path) for name, path in implementation_paths.items()
        }
        file_sha256(sidecar)
        file_sha256(sentinel)
    except OSError:
        return 0, [{"finding": "feature_live_implementation_or_sidecar_missing"}]
    valid = 0
    for relative, expected in records.items():
        label = f'{expected["cell"]}/{POOLING_SLUGS[expected["pooling"]]}'
        try:
            metadata_path = expected["metadata"]
            asset_path = expected["asset"]
            if inventory.get(relative) != file_sha256(metadata_path):
                raise ValueError
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if set(metadata) != FINAL_FEATURE_METADATA_FIELDS:
                raise ValueError
            pooling = expected["pooling"]
            cell = expected["cell"]
            shape = [808, 4, 384 if pooling in {"PLOCAL+GLOBAL", "PLOCAL+PVALID_SECONDARY"} else 192]
            valid_count = 1500 if pooling == "PORACLE" else 3232
            selected = lock["selected_checkpoints"][cell]
            reference = lock["formal_p0_references"][cell]
            if (
                metadata.get("schema_version") != 1
                or metadata.get("status") != "COMPLETE"
                or metadata.get("stage") != "final"
                or metadata.get("seed_base") != expected["seed_base"]
                or metadata.get("arm") != expected["arm"]
                or metadata.get("fold") != expected["fold"]
                or metadata.get("pooling") != pooling
                or metadata.get("pooling_slug") != POOLING_SLUGS[pooling]
                or not _path_value_matches(metadata.get("feature_path"), asset_path, repo, metadata_path.parent)
                or metadata.get("feature_sha256") != file_sha256(asset_path)
                or metadata.get("state_shape") != shape
                or metadata.get("state_dtype") != "float32"
                or metadata.get("state_valid_shape") != [808, 4]
                or metadata.get("state_valid_count") != valid_count
                or metadata.get("patient_count") != 808
                or not _is_sha256(metadata.get("patient_order_sha256"))
                or not _is_sha256(metadata.get("split_order_sha256"))
                or not _path_value_matches(metadata.get("checkpoint_path"), _safe_repo_path(repo, selected["path"]), repo, metadata_path.parent)
                or metadata.get("checkpoint_sha256") != selected["sha256"]
                or metadata.get("checkpoint_lock_key") != cell
                or not _path_value_matches(metadata.get("reference_feature_path"), _safe_repo_path(repo, reference["feature_path"]), repo, metadata_path.parent)
                or not _path_value_matches(metadata.get("reference_feature_metadata_path"), _safe_repo_path(repo, reference["feature_metadata_path"]), repo, metadata_path.parent)
                or metadata.get("reference_feature_sha256") != reference["feature_sha256"]
                or metadata.get("reference_feature_metadata_sha256") != reference["feature_metadata_sha256"]
                or metadata.get("preregistration_lock_sha256") != lock_sha
                or metadata.get("plan_sha256") != lock.get("plan_sha256")
                or metadata.get("config_sha256") != lock.get("config_sha256")
                or not _path_value_matches(metadata.get("sidecar_path"), sidecar, repo, metadata_path.parent)
                or metadata.get("sidecar_sha256") != file_sha256(sidecar)
                or metadata.get("sidecar_keys_used") != _feature_sidecar_keys(expected["arm"], pooling)
                or metadata.get("stage_a_sentinel_sha256") != file_sha256(sentinel)
                or metadata.get("implementation_sha256") != implementation
                or not isinstance(metadata.get("device"), str)
                or not metadata["device"].startswith("cuda:")
                or metadata.get("batch_size") != 4
                or metadata.get("workers") != 2
                or metadata.get("feature_tensor") != "full_model.encoder_output_before_gap"
                or metadata.get("response_projection") != "frozen_online_Linear128x192_plus_LayerNorm"
                or any(
                    metadata.get(field) is not False
                    for field in (
                        "training_performed", "projector_called", "transition_called",
                        "target_encoder_called", "ftv_head_called", "test_labels_used",
                    )
                )
                or any(
                    not _is_sha256(metadata.get(field))
                    for field in (
                        "data_contract_provenance_sha256",
                        "checkpoint_data_provenance_sha256",
                    )
                )
            ):
                raise ValueError
            valid += 1
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            findings.append({"finding": "feature_asset_or_binding_invalid", "cell": label})
    return valid, findings


def _completion_expected_keys(poolings: Sequence[str]) -> set[str]:
    return {
        key
        for key, record in _probe_records(EXPERIMENT_ROOT).items()
        if record["pooling"] in poolings
    }


def _validate_probe_completion(
    *,
    completion_path: Path,
    poolings: tuple[str, ...],
    experiment: Path,
    repo: Path,
    lock: Mapping[str, Any],
    lock_sha: str,
    feature_records: Mapping[str, Mapping[str, Any]],
    probe_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], int, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    accepted: dict[str, Mapping[str, Any]] = {}
    valid = 0
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not isinstance(completion, Mapping) or set(completion) != PROBE_COMPLETION_FIELDS:
            raise ValueError
        is_p0 = poolings == ("P0",)
        expected_count = 40 if is_p0 else 140
        expected_nuisance = 40 if is_p0 else 60
        expected_gate = {
            "p0_equivalence_gate_sha256": file_sha256(
                experiment / "metrics/p0_equivalence_gate.json"
            )
        }
        if not is_p0:
            expected_gate["p0_probe_replication_gate_sha256"] = file_sha256(
                experiment / "metrics/p0_probe_replication_gate.json"
            )
        nuisance_path = experiment / "manifests/nuisance_targets.private.csv"
        feature_completion = experiment / "features/feature_export_complete.private.json"
        runner = experiment / "src/c1b_spatial_audit/probe_runner.py"
        cells = completion.get("cells")
        expected_keys = {
            key for key, row in probe_records.items() if row["pooling"] in poolings
        }
        feature_hashes = completion.get("feature_metadata_sha256")
        if (
            completion.get("schema_version") != 1
            or completion.get("status") != "COMPLETE"
            or completion.get("stage") != "final"
            or completion.get("representation_contract")
            != "final_pooled_then_frozen_response_projection"
            or completion.get("requested_poolings") != list(poolings)
            or completion.get("expected_cell_count") != expected_count
            or completion.get("executed_cell_count") != expected_count
            or completion.get("nuisance_cell_count") != expected_nuisance
            or completion.get("nuisance_targets") != list(NUISANCE_TARGETS)
            or not _path_value_matches(completion.get("feature_root"), experiment / "features", repo, completion_path.parent)
            or not _path_value_matches(completion.get("probe_root"), experiment / "probes", repo, completion_path.parent)
            or not _path_value_matches(completion.get("completion_path"), completion_path, repo, completion_path.parent)
            or completion.get("preregistration_lock_sha256") != lock_sha
            or completion.get("nuisance_sha256") != file_sha256(nuisance_path)
            or completion.get("gate_sha256") != expected_gate
            or completion.get("exporter_completion_sha256") != file_sha256(feature_completion)
            or completion.get("legacy_pvalid") != "NA_no_source_authoritative_mask"
            or completion.get("legacy_poracle")
            != "NA_incomplete_source_authoritative_support_1488_of_1500"
            or completion.get("fabricated_unavailable_rows") != 0
            or not isinstance(completion.get("workers"), int)
            or completion["workers"] < 1
            or completion.get("probe_runner_sha256") != file_sha256(runner)
            or completion.get("patient_identifiers_private") is not True
            or completion.get("new_training_performed") is not False
            or not isinstance(cells, Mapping)
            or set(cells) != expected_keys
            or not isinstance(feature_hashes, Mapping)
            or set(feature_hashes) != expected_keys
        ):
            raise ValueError
        feature_by_key = {
            f'{row["cell"]}/{POOLING_SLUGS[row["pooling"]]}': row
            for row in feature_records.values()
        }
        adapter = experiment / "src/c1b_spatial_audit/probes.py"
        for key in sorted(expected_keys):
            row = cells[key]
            expected = probe_records[key]
            feature = feature_by_key[key]
            if not isinstance(row, Mapping) or set(row) != PROBE_COMPLETION_CELL_FIELDS:
                raise ValueError
            output = expected["output"]
            nuisance = expected["pooling"] in {"P0", "PVALID", "PLOCAL"}
            if (
                row.get("seed_base") != expected["seed_base"]
                or row.get("arm") != expected["arm"]
                or row.get("fold") != expected["fold"]
                or row.get("pooling") != expected["pooling"]
                or row.get("pooling_slug") != expected["slug"]
                or row.get("nuisance_included") is not nuisance
                or not _path_value_matches(row.get("feature_path"), feature["asset"], repo, output)
                or not _path_value_matches(row.get("feature_metadata_path"), feature["metadata"], repo, output)
                or not _path_value_matches(row.get("output_dir"), output, repo, output.parent)
                or row.get("feature_sha256") != file_sha256(feature["asset"])
                or row.get("feature_metadata_sha256") != file_sha256(feature["metadata"])
                or feature_hashes.get(key) != file_sha256(feature["metadata"])
                or not all(isinstance(row.get(field), int) and row[field] > 0 for field in ("selection_rows", "prediction_rows", "metric_rows"))
            ):
                raise ValueError
            metadata_path = output / "probe_metadata.json"
            if row.get("probe_metadata_sha256") != file_sha256(metadata_path):
                raise ValueError
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if set(metadata) != PROBE_METADATA_FIELDS:
                raise ValueError
            output_hashes = row.get("output_sha256")
            if not isinstance(output_hashes, Mapping) or set(output_hashes) != {
                "probe_metrics.csv", "ridge_predictions.private.csv", "ridge_selection.csv"
            } or metadata.get("output_sha256") != output_hashes:
                raise ValueError
            for filename, digest in output_hashes.items():
                if not _is_sha256(digest) or file_sha256(output / filename) != digest:
                    raise ValueError
            provenance = metadata.get("provenance")
            expected_tasks = ["delta", "nuisance", "static"] if nuisance else ["delta", "static"]
            expected_targets = sorted(("FTV", *NUISANCE_TARGETS)) if nuisance else ["FTV"]
            if (
                metadata.get("schema_version") != 1
                or metadata.get("seed_base") != expected["seed_base"]
                or metadata.get("arm") != expected["arm"]
                or metadata.get("fold") != expected["fold"]
                or metadata.get("pooling") != expected["pooling"]
                or metadata.get("feature_dim") != (384 if expected["pooling"] in {"PLOCAL+GLOBAL", "PLOCAL+PVALID_SECONDARY"} else 192)
                or not _path_value_matches(
                    metadata.get("feature_path"), feature["asset"], repo, output
                )
                or not _path_value_matches(
                    metadata.get("feature_metadata_path"), feature["metadata"], repo, output
                )
                or metadata.get("feature_sha256") != row["feature_sha256"]
                or metadata.get("feature_metadata_sha256") != row["feature_metadata_sha256"]
                or metadata.get("patient_identifiers_private") is not True
                or metadata.get("prediction_asset_private") is not True
                or metadata.get("state_valid_enforced") is not True
                or metadata.get("test_used_for_scaler_or_selection") is not False
                or metadata.get("outer_test_predict_calls_per_cell") != 1
                or metadata.get("alpha_grid") != [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
                or metadata.get("ridge_selection_implementation") != "immutable_stage_b_select_ridge"
                or metadata.get("static_target_implementation") != "immutable_stage_b_target_adapter"
                or metadata.get("formal_probe_source_sha256")
                != lock["upstream_source_sha256"]["additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/probes.py"]
                or metadata.get("formal_target_source_sha256")
                != lock["upstream_source_sha256"]["additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/targets.py"]
                or metadata.get("probe_adapter_sha256") != file_sha256(adapter)
                or not isinstance(provenance, Mapping)
                or metadata.get("provenance_sha256") != canonical_sha256(provenance)
                or metadata.get("tasks") != expected_tasks
                or metadata.get("target_names") != expected_targets
                or any(metadata.get(field) != row[field] for field in ("selection_rows", "prediction_rows", "metric_rows"))
            ):
                raise ValueError
            for field, digest in expected_gate.items():
                if provenance.get(field) != digest:
                    raise ValueError
            if (
                provenance.get("stage") != "final"
                or provenance.get("checkpoint_lock_key") != expected["cell"]
                or provenance.get("preregistration_lock_sha256") != lock_sha
                or provenance.get("plan_sha256") != lock.get("plan_sha256")
                or provenance.get("config_sha256") != lock.get("config_sha256")
                or provenance.get("exporter_completion_sha256") != file_sha256(feature_completion)
                or provenance.get("exporter_feature_sha256") != row["feature_sha256"]
                or provenance.get("exporter_metadata_sha256") != row["feature_metadata_sha256"]
            ):
                raise ValueError
            if nuisance and (
                provenance.get("nuisance_target_names") != list(NUISANCE_TARGETS)
                or provenance.get("nuisance_targets_sha256") != file_sha256(nuisance_path)
            ):
                raise ValueError
            accepted[key] = row
            valid += 1
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        findings.append({"finding": "probe_completion_schema_or_live_binding_invalid"})
    return accepted, valid, findings


def audit_final_inventories(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate exact final controls, 180 feature assets, and two probe completions."""

    experiment = Path(experiment_root).resolve()
    repo = Path(repo_root).resolve()
    findings: list[dict[str, Any]] = []
    try:
        lock_path = experiment / "PREREGISTRATION_LOCK.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_sha = file_sha256(lock_path)
        if set(lock.get("selected_checkpoints", {})) != _expected_cell_names():
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "invalid_preregistration_lock"}],
            "expected_feature_assets": 180,
            "expected_formal_probe_cells": 180,
        }
    feature_records = _feature_records(experiment, repo)
    completion, control_findings = _validate_feature_controls(
        experiment, repo, lock, lock_sha, feature_records
    )
    findings.extend(control_findings)
    feature_valid, feature_findings = _validate_final_feature_assets(
        experiment, repo, lock, lock_sha, feature_records, completion
    )
    findings.extend(feature_findings)

    probe_records = _probe_records(experiment)
    final_root = experiment / "probes/final"
    live_completions = {
        path.name: path for path in final_root.glob("*complete.private.json")
    } if final_root.is_dir() else {}
    if set(live_completions) != set(FINAL_PROBE_COMPLETIONS):
        findings.append({"finding": "probe_completion_inventory_not_exact_two"})
    union: dict[str, Mapping[str, Any]] = {}
    probe_valid = 0
    for filename, poolings in FINAL_PROBE_COMPLETIONS.items():
        path = final_root / filename
        accepted, count, completion_findings = _validate_probe_completion(
            completion_path=path,
            poolings=poolings,
            experiment=experiment,
            repo=repo,
            lock=lock,
            lock_sha=lock_sha,
            feature_records=feature_records,
            probe_records=probe_records,
        )
        if set(union).intersection(accepted):
            findings.append({"finding": "duplicate_probe_cell_across_completions"})
        union.update(accepted)
        probe_valid += count
        findings.extend(completion_findings)
    if set(union) != set(probe_records):
        findings.append({"finding": "probe_inventory_not_exact_180"})
    expected_metadata = {
        record["output"] / "probe_metadata.json" for record in probe_records.values()
    }
    live_metadata = set(final_root.rglob("probe_metadata.json")) if final_root.is_dir() else set()
    if live_metadata != expected_metadata:
        findings.append({"finding": "live_probe_metadata_inventory_not_exact_180"})
    expected_probe_files = {
        record["output"] / filename
        for record in probe_records.values()
        for filename in (
            "probe_metadata.json", "probe_metrics.csv",
            "ridge_predictions.private.csv", "ridge_selection.csv",
        )
    } | {final_root / filename for filename in FINAL_PROBE_COMPLETIONS}
    live_probe_files = {
        path for path in final_root.rglob("*") if path.is_file() or path.is_symlink()
    } if final_root.is_dir() else set()
    if live_probe_files != expected_probe_files:
        findings.append({"finding": "live_probe_file_inventory_contains_missing_or_extra_files"})
    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "expected_checkpoint_cells": 40,
        "expected_feature_assets": 180,
        "validated_feature_assets": feature_valid,
        "expected_formal_probe_cells": 180,
        "validated_formal_probe_cells": probe_valid,
        "p0_probe_cells": 40,
        "alternate_probe_cells": 140,
        "primary_probe_cells": 160,
        "secondary_local_valid_probe_cells": 20,
        "probe_completion_inventories": len(live_completions),
    }


PROSPECTIVE_FIELDS = frozenset(
    {
        "schema_version", "status", "natural_metrics", "transformed_metrics",
        "new_training_performed", "probe_refit_during_aggregation", "final_stage",
        "conditional_s3", "training_budget", "classification",
    }
)
S3_TRIGGER_FIELDS = frozenset(
    {
        "schema_version", "status", "s3_execution_authorized", "decision_contract",
        "final_stage_strong_oracle_recovery", "final_probe_root",
        "final_probe_cell_count", "final_probe_metadata_inventory_sha256",
        "preregistration_lock_sha256", "plan_sha256", "config_sha256",
        "p0_equivalence_gate_sha256", "p0_probe_replication_gate_sha256",
        "trigger_implementation_sha256", "analysis_implementation_sha256",
        "probe_adapter_sha256", "new_training_performed", "probe_refit_performed",
        "patient_identifiers_present",
    }
)
S3_FEATURE_COMPLETION_FIELDS = frozenset(
    {
        "schema_version", "status", "stage", "representation_contract", "run_count",
        "cell_count", "expected_asset_count", "feature_metadata_sha256",
        "preflight_sha256", "trigger_gate_sha256", "sidecar_sha256",
        "sidecar_metadata_sha256", "preregistration_lock_sha256",
    }
)
S3_FEATURE_METADATA_FIELDS = frozenset(
    {
        "schema_version", "stage", "status", "arm", "seed_base", "fold", "pooling",
        "pooling_slug", "feature_path", "feature_sha256", "state_shape", "state_dtype",
        "state_valid_shape", "state_valid_count", "patient_count", "patient_order_sha256",
        "split_order_sha256", "checkpoint_path", "checkpoint_sha256",
        "checkpoint_lock_key", "reference_feature_path", "reference_feature_sha256",
        "reference_feature_metadata_path", "reference_feature_metadata_sha256",
        "preregistration_lock_sha256", "plan_sha256", "config_sha256", "sidecar_path",
        "sidecar_sha256", "sidecar_metadata_path", "sidecar_metadata_sha256",
        "sidecar_keys_used", "trigger_gate_path", "trigger_gate_sha256", "trigger_status",
        "data_contract_provenance_sha256", "checkpoint_data_provenance_sha256",
        "stage_a_sentinel_sha256", "implementation_sha256", "device", "batch_size",
        "workers", "feature_tensor", "stage_module", "feature_channels",
        "response_projection", "representation_contract", "training_performed",
        "response_projection_called", "projector_called", "transition_called",
        "target_encoder_called", "ftv_head_called", "test_labels_used",
    }
)
S3_REPRESENTATION = "raw_encoder_features2_pooled_64d_no_projection"


def _expected_classification(gates: Mapping[str, Any]) -> dict[str, str] | None:
    try:
        final = gates["final_stage"]
        conditional = gates["conditional_s3"]
        final_oracle = bool(final["strong_oracle_recovery"]["supported"])
        local = bool(final["deployable_local_recovery"]["supported"])
        padding = bool(final["padding_geometry_evidence"]["supported"])
        s3_evidence = conditional["strong_oracle_recovery"]
        s3_oracle = None if s3_evidence is None else bool(s3_evidence["supported"])
    except (KeyError, TypeError):
        return None
    if not final_oracle and s3_oracle is None:
        return None
    if not final_oracle and not bool(s3_oracle):
        return {
            "code": "C", "classification": "C ENCODER BOTTLENECK",
            "next": "Stronger Pretrained 3-D Encoder Pilot",
        }
    if padding:
        return {
            "code": "B", "classification": "B PADDING / GEOMETRY DILUTION",
            "next": "Valid-source-aware + Localized Response Pooling Pilot",
        }
    if final_oracle and local:
        return {
            "code": "A", "classification": "A POOLING BOTTLENECK",
            "next": "Local–Global Response State Pilot",
        }
    if final_oracle and not local:
        next_step = "Learned Spatial Response Aggregation Pilot"
    elif bool(s3_oracle) and not final_oracle:
        next_step = "Preserve Higher-Resolution Spatial Features Pilot"
    else:
        next_step = "Local–Global Response State Minimal Pilot"
    return {"code": "D", "classification": "D MIXED BOTTLENECK", "next": next_step}


def _load_prospective(experiment: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = experiment / "metrics/prospective_gates.json"
    try:
        gates = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(gates, Mapping) or set(gates) != PROSPECTIVE_FIELDS:
            raise ValueError
        final = gates.get("final_stage")
        conditional = gates.get("conditional_s3")
        classification = gates.get("classification")
        if (
            gates.get("schema_version") != 1
            or gates.get("status") != "COMPLETE"
            or gates.get("natural_metrics") != "pooled_five_outer_test_folds_before_metric"
            or gates.get("transformed_metrics") != "outer_fold_summaries_only"
            or gates.get("new_training_performed") is not False
            or gates.get("probe_refit_during_aggregation") is not False
            or not isinstance(final, Mapping)
            or set(final) != {
                "strong_oracle_recovery", "deployable_local_recovery",
                "padding_geometry_evidence",
            }
            or not isinstance(conditional, Mapping)
            or set(conditional) != {
                "trigger_status", "strong_oracle_recovery", "deployable_local_recovery"
            }
            or conditional.get("trigger_status") not in {
                "NOT_TRIGGERED_FINAL_ORACLE_STRONG",
                "TRIGGERED_FINAL_ORACLE_WEAK_COMPLETED",
            }
            or not isinstance(classification, Mapping)
            or set(classification) != {"code", "classification", "next"}
            or dict(classification) != _expected_classification(gates)
        ):
            raise ValueError
        return dict(gates), []
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return {}, [{"finding": "prospective_gate_schema_or_classification_invalid"}]


def _s3_records(experiment: Path, repo: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                cell = f"seed_{seed}/{arm}/fold_{fold}"
                poolings = ("P0", "PLOCAL") if arm.startswith("L") else (
                    "P0", "PLOCAL", "PORACLE"
                )
                for pooling in poolings:
                    slug = POOLING_SLUGS[pooling]
                    asset = (
                        experiment / "features/s3" / f"seed_{seed}" / arm
                        / f"fold_{fold}" / f"{slug}.private.npz"
                    )
                    metadata = asset.with_suffix(".metadata.json")
                    key = f"{cell}/{slug}"
                    result[key] = {
                        "cell": cell, "seed_base": seed, "arm": arm, "fold": fold,
                        "pooling": pooling, "slug": slug, "asset": asset,
                        "metadata": metadata,
                        "inventory_key": metadata.resolve().relative_to(repo).as_posix(),
                        "output": (
                            experiment / "probes/s3" / f"seed_{seed}" / arm
                            / f"fold_{fold}" / slug
                        ),
                    }
    if len(result) != 100:
        raise AssertionError("S3 inventory must be exactly 100")
    return result


def _validate_s3_assets(
    experiment: Path,
    repo: Path,
    lock: Mapping[str, Any],
    lock_sha: str,
    trigger_path: Path,
) -> tuple[int, int, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    feature_valid = probe_valid = 0
    records = _s3_records(experiment, repo)
    stage_root = experiment / "features/s3"
    completion_path = stage_root / "feature_export_complete.private.json"
    preflight_path = stage_root / "feature_export_preflight.private.json"
    claim_path = stage_root / "feature_export_claim.private.json"
    sidecar = experiment / "manifests/audit_sidecars_s3.private.npz"
    sidecar_metadata = experiment / "manifests/audit_sidecars_s3.private.metadata.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        if set(completion) != S3_FEATURE_COMPLETION_FIELDS:
            raise ValueError
        if set(claim) != {
            "schema_version", "status", "stage", "nonresumable", "cell_count",
            "expected_asset_count", "trigger_gate_sha256", "sidecar_sha256",
            "matrix_driver_sha256",
        }:
            raise ValueError
        preflight_fields = {
            "schema_version", "status", "stage", "representation_contract", "cell_count",
            "expected_asset_count", "preregistration_lock_sha256", "trigger_gate_sha256",
            "trigger_status", "sidecar_sha256", "sidecar_metadata_sha256",
            "stage_a_sentinel_sha256", "feature_root", "scheduler", "cell_inventory",
            "code_sha256", "execution_requested", "legacy_poracle", "claim_sha256",
        }
        if set(preflight) != preflight_fields:
            raise ValueError
        code_paths = {
            "matrix_driver": experiment / "scripts/run_s3_feature_matrix.py",
            "feature_cli": experiment / "scripts/export_s3_frozen_features.py",
            "s3_exporter": experiment / "src/c1b_spatial_audit/s3_exporter.py",
            "s3_sidecars": experiment / "src/c1b_spatial_audit/s3_sidecars.py",
            "s3_trigger": experiment / "src/c1b_spatial_audit/s3_trigger.py",
            "pooling": experiment / "src/c1b_spatial_audit/pooling.py",
            "runtime": experiment / "src/c1b_spatial_audit/runtime.py",
            "contracts": experiment / "src/c1b_spatial_audit/contracts.py",
        }
        live_code = {name: file_sha256(path) for name, path in code_paths.items()}
        trigger_sha = file_sha256(trigger_path)
        inventory = completion.get("feature_metadata_sha256")
        if (
            claim.get("schema_version") != 1 or claim.get("status") != "CLAIMED"
            or claim.get("stage") != "s3" or claim.get("nonresumable") is not True
            or claim.get("cell_count") != 40 or claim.get("expected_asset_count") != 100
            or claim.get("trigger_gate_sha256") != trigger_sha
            or claim.get("sidecar_sha256") != file_sha256(sidecar)
            or claim.get("matrix_driver_sha256") != live_code["matrix_driver"]
            or preflight.get("schema_version") != 1
            or preflight.get("status") != "PREFLIGHT_PASS"
            or preflight.get("stage") != "s3"
            or preflight.get("representation_contract") != S3_REPRESENTATION
            or preflight.get("cell_count") != 40
            or preflight.get("expected_asset_count") != 100
            or preflight.get("preregistration_lock_sha256") != lock_sha
            or preflight.get("trigger_gate_sha256") != trigger_sha
            or preflight.get("trigger_status") != "TRIGGERED_FINAL_ORACLE_WEAK"
            or preflight.get("sidecar_sha256") != file_sha256(sidecar)
            or preflight.get("sidecar_metadata_sha256") != file_sha256(sidecar_metadata)
            or preflight.get("code_sha256") != live_code
            or preflight.get("execution_requested") is not True
            or preflight.get("claim_sha256") != file_sha256(claim_path)
            or completion.get("schema_version") != 1
            or completion.get("status") != "COMPLETE"
            or completion.get("stage") != "s3"
            or completion.get("representation_contract") != S3_REPRESENTATION
            or completion.get("run_count") != 40 or completion.get("cell_count") != 40
            or completion.get("expected_asset_count") != 100
            or completion.get("preflight_sha256") != file_sha256(preflight_path)
            or completion.get("trigger_gate_sha256") != trigger_sha
            or completion.get("sidecar_sha256") != file_sha256(sidecar)
            or completion.get("sidecar_metadata_sha256") != file_sha256(sidecar_metadata)
            or completion.get("preregistration_lock_sha256") != lock_sha
            or not isinstance(inventory, Mapping)
            or set(inventory) != {row["inventory_key"] for row in records.values()}
        ):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        completion = {}
        inventory = {}
        findings.append({"finding": "s3_feature_controls_invalid"})
    expected_assets = {row["asset"].resolve() for row in records.values()}
    expected_metadata = {row["metadata"].resolve() for row in records.values()}
    live_assets = {path.resolve() for path in stage_root.rglob("*.private.npz")} if stage_root.is_dir() else set()
    live_metadata = {path.resolve() for path in stage_root.rglob("*.private.metadata.json")} if stage_root.is_dir() else set()
    if live_assets != expected_assets or live_metadata != expected_metadata:
        findings.append({"finding": "s3_feature_inventory_not_exact_100"})
    implementation_paths = {
        "s3_exporter.py": experiment / "src/c1b_spatial_audit/s3_exporter.py",
        "s3_sidecars.py": experiment / "src/c1b_spatial_audit/s3_sidecars.py",
        "s3_trigger.py": experiment / "src/c1b_spatial_audit/s3_trigger.py",
        "pooling.py": experiment / "src/c1b_spatial_audit/pooling.py",
        "runtime.py": experiment / "src/c1b_spatial_audit/runtime.py",
        "contracts.py": experiment / "src/c1b_spatial_audit/contracts.py",
    }
    implementation = {name: file_sha256(path) for name, path in implementation_paths.items()}
    for key, expected in records.items():
        try:
            metadata = json.loads(expected["metadata"].read_text(encoding="utf-8"))
            if set(metadata) != S3_FEATURE_METADATA_FIELDS:
                raise ValueError
            valid_count = 1500 if expected["pooling"] == "PORACLE" else 3232
            if (
                inventory.get(expected["inventory_key"]) != file_sha256(expected["metadata"])
                or metadata.get("schema_version") != 1 or metadata.get("status") != "COMPLETE"
                or metadata.get("stage") != "s3" or metadata.get("seed_base") != expected["seed_base"]
                or metadata.get("arm") != expected["arm"] or metadata.get("fold") != expected["fold"]
                or metadata.get("pooling") != expected["pooling"]
                or metadata.get("pooling_slug") != expected["slug"]
                or metadata.get("state_shape") != [808, 4, 64]
                or metadata.get("state_dtype") != "float32"
                or metadata.get("state_valid_shape") != [808, 4]
                or metadata.get("state_valid_count") != valid_count
                or metadata.get("patient_count") != 808
                or metadata.get("feature_sha256") != file_sha256(expected["asset"])
                or metadata.get("preregistration_lock_sha256") != lock_sha
                or metadata.get("trigger_gate_sha256") != file_sha256(trigger_path)
                or metadata.get("trigger_status") != "TRIGGERED_FINAL_ORACLE_WEAK"
                or metadata.get("sidecar_sha256") != file_sha256(sidecar)
                or metadata.get("sidecar_metadata_sha256") != file_sha256(sidecar_metadata)
                or metadata.get("implementation_sha256") != implementation
                or metadata.get("feature_tensor") != "model.encoder.features[2]_full_residual_output"
                or metadata.get("stage_module") != "encoder.features[2]"
                or metadata.get("feature_channels") != 64
                or metadata.get("response_projection") != "not_called_raw_64d"
                or metadata.get("representation_contract") != S3_REPRESENTATION
                or any(metadata.get(field) is not False for field in (
                    "training_performed", "response_projection_called", "projector_called",
                    "transition_called", "target_encoder_called", "ftv_head_called",
                    "test_labels_used",
                ))
            ):
                raise ValueError
            feature_valid += 1
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            findings.append({"finding": "s3_feature_asset_or_binding_invalid", "cell": key})

    probe_root = experiment / "probes/s3"
    probe_completion_path = probe_root / "probe_matrix_p0_plocal_poracle_complete.private.json"
    live_completions = list(probe_root.glob("*complete.private.json")) if probe_root.is_dir() else []
    try:
        probe = json.loads(probe_completion_path.read_text(encoding="utf-8"))
        cells = probe.get("cells")
        feature_hashes = probe.get("feature_metadata_sha256")
        expected_gates = {
            "p0_equivalence_gate_sha256": file_sha256(experiment / "metrics/p0_equivalence_gate.json"),
            "p0_probe_replication_gate_sha256": file_sha256(experiment / "metrics/p0_probe_replication_gate.json"),
            "s3_trigger_authorization_sha256": file_sha256(trigger_path),
            "s3_probe_runner_sha256": file_sha256(experiment / "src/c1b_spatial_audit/s3_probe_runner.py"),
        }
        if (
            len(live_completions) != 1 or live_completions[0] != probe_completion_path
            or set(probe) != PROBE_COMPLETION_FIELDS
            or probe.get("schema_version") != 1 or probe.get("status") != "COMPLETE"
            or probe.get("stage") != "s3" or probe.get("representation_contract") != S3_REPRESENTATION
            or probe.get("requested_poolings") != ["P0", "PLOCAL", "PORACLE"]
            or probe.get("expected_cell_count") != 100 or probe.get("executed_cell_count") != 100
            or probe.get("nuisance_cell_count") != 0
            or probe.get("nuisance_targets") != list(NUISANCE_TARGETS)
            or probe.get("nuisance_sha256") is not None
            or probe.get("preregistration_lock_sha256") != lock_sha
            or probe.get("gate_sha256") != expected_gates
            or probe.get("exporter_completion_sha256") != file_sha256(completion_path)
            or probe.get("fabricated_unavailable_rows") != 0
            or probe.get("patient_identifiers_private") is not True
            or probe.get("new_training_performed") is not False
            or probe.get("probe_runner_sha256") != file_sha256(experiment / "src/c1b_spatial_audit/probe_runner.py")
            or not isinstance(cells, Mapping) or set(cells) != set(records)
            or not isinstance(feature_hashes, Mapping) or set(feature_hashes) != set(records)
        ):
            raise ValueError
        for key, expected in records.items():
            row = cells[key]
            output = expected["output"]
            if (
                not isinstance(row, Mapping) or set(row) != PROBE_COMPLETION_CELL_FIELDS
                or row.get("seed_base") != expected["seed_base"]
                or row.get("arm") != expected["arm"] or row.get("fold") != expected["fold"]
                or row.get("pooling") != expected["pooling"] or row.get("pooling_slug") != expected["slug"]
                or row.get("nuisance_included") is not False
                or row.get("feature_sha256") != file_sha256(expected["asset"])
                or row.get("feature_metadata_sha256") != file_sha256(expected["metadata"])
                or feature_hashes.get(key) != file_sha256(expected["metadata"])
                or row.get("probe_metadata_sha256") != file_sha256(output / "probe_metadata.json")
            ):
                raise ValueError
            metadata = json.loads((output / "probe_metadata.json").read_text(encoding="utf-8"))
            if (
                set(metadata) != PROBE_METADATA_FIELDS
                or metadata.get("feature_dim") != 64
                or metadata.get("patient_identifiers_private") is not True
                or metadata.get("tasks") != ["delta", "static"]
                or metadata.get("target_names") != ["FTV"]
                or metadata.get("provenance", {}).get("s3_trigger_authorization_sha256")
                != file_sha256(trigger_path)
                or metadata.get("provenance", {}).get("s3_probe_runner_sha256")
                != file_sha256(experiment / "src/c1b_spatial_audit/s3_probe_runner.py")
            ):
                raise ValueError
            output_hashes = row.get("output_sha256")
            if not isinstance(output_hashes, Mapping) or set(output_hashes) != {
                "probe_metrics.csv", "ridge_predictions.private.csv", "ridge_selection.csv"
            } or metadata.get("output_sha256") != output_hashes:
                raise ValueError
            for filename, digest in output_hashes.items():
                if file_sha256(output / filename) != digest:
                    raise ValueError
            probe_valid += 1
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        findings.append({"finding": "s3_probe_completion_or_asset_invalid"})
    live_probe_metadata = set(probe_root.rglob("probe_metadata.json")) if probe_root.is_dir() else set()
    if live_probe_metadata != {row["output"] / "probe_metadata.json" for row in records.values()}:
        findings.append({"finding": "s3_probe_inventory_not_exact_100"})
    return feature_valid, probe_valid, findings


def audit_conditional_s3(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate the trigger-bound exact-zero or exact-100 conditional S3 branch."""

    experiment = Path(experiment_root).resolve()
    repo = Path(repo_root).resolve()
    findings: list[dict[str, Any]] = []
    prospective, prospective_findings = _load_prospective(experiment)
    findings.extend(prospective_findings)
    trigger_path = experiment / "metrics/s3_trigger_authorization.json"
    feature_valid = probe_valid = 0
    try:
        lock_path = experiment / "PREREGISTRATION_LOCK.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_sha = file_sha256(lock_path)
        trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
        if not isinstance(trigger, Mapping) or set(trigger) != S3_TRIGGER_FIELDS:
            raise ValueError
        final_probe_root = experiment / "probes/final"
        final_metadata = sorted(final_probe_root.rglob("probe_metadata.json"))
        inventory = {
            path.relative_to(final_probe_root).as_posix(): file_sha256(path)
            for path in final_metadata
        }
        oracle = trigger.get("final_stage_strong_oracle_recovery")
        supported = oracle.get("supported") if isinstance(oracle, Mapping) else None
        status = (
            "NOT_TRIGGERED_FINAL_ORACLE_STRONG"
            if supported is True else "TRIGGERED_FINAL_ORACLE_WEAK"
        )
        if (
            not isinstance(supported, bool)
            or oracle.get("status")
            != ("SUPPORTED_IN_PILOT" if supported else "NOT_SUPPORTED_IN_PILOT")
            or trigger.get("schema_version") != 1
            or trigger.get("status") != status
            or trigger.get("s3_execution_authorized") is not (not supported)
            or trigger.get("decision_contract") != "final_stage_strong_oracle_recovery_false"
            or trigger.get("final_probe_root") != "probes/final"
            or trigger.get("final_probe_cell_count") != 180
            or len(final_metadata) != 180
            or trigger.get("final_probe_metadata_inventory_sha256") != canonical_sha256(inventory)
            or trigger.get("preregistration_lock_sha256") != lock_sha
            or trigger.get("plan_sha256") != lock.get("plan_sha256")
            or trigger.get("config_sha256") != lock.get("config_sha256")
            or trigger.get("p0_equivalence_gate_sha256") != file_sha256(experiment / "metrics/p0_equivalence_gate.json")
            or trigger.get("p0_probe_replication_gate_sha256") != file_sha256(experiment / "metrics/p0_probe_replication_gate.json")
            or trigger.get("trigger_implementation_sha256") != file_sha256(experiment / "src/c1b_spatial_audit/s3_trigger.py")
            or trigger.get("analysis_implementation_sha256") != file_sha256(experiment / "src/c1b_spatial_audit/analysis.py")
            or trigger.get("probe_adapter_sha256") != file_sha256(experiment / "src/c1b_spatial_audit/probes.py")
            or trigger.get("new_training_performed") is not False
            or trigger.get("probe_refit_performed") is not False
            or trigger.get("patient_identifiers_present") is not False
            or prospective.get("final_stage", {}).get("strong_oracle_recovery") != oracle
        ):
            raise ValueError
        conditional_status = prospective.get("conditional_s3", {}).get("trigger_status")
        if supported:
            if (
                conditional_status != "NOT_TRIGGERED_FINAL_ORACLE_STRONG"
                or prospective.get("conditional_s3", {}).get("strong_oracle_recovery")
                is not None
                or prospective.get("conditional_s3", {}).get("deployable_local_recovery")
                is not None
            ):
                raise ValueError
            forbidden = [
                experiment / "manifests/audit_sidecars_s3.private.npz",
                experiment / "manifests/audit_sidecars_s3.private.metadata.json",
            ]
            for root in (experiment / "features/s3", experiment / "probes/s3"):
                if root.exists() and any(root.rglob("*")):
                    raise ValueError
            if any(path.exists() for path in forbidden):
                raise ValueError
        else:
            if (
                conditional_status != "TRIGGERED_FINAL_ORACLE_WEAK_COMPLETED"
                or not isinstance(
                    prospective.get("conditional_s3", {}).get("strong_oracle_recovery"),
                    Mapping,
                )
                or not isinstance(
                    prospective.get("conditional_s3", {}).get("deployable_local_recovery"),
                    Mapping,
                )
            ):
                raise ValueError
            feature_valid, probe_valid, s3_findings = _validate_s3_assets(
                experiment, repo, lock, lock_sha, trigger_path
            )
            findings.extend(s3_findings)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        findings.append({"finding": "s3_trigger_or_branch_binding_invalid"})
    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "validated_s3_feature_assets": feature_valid,
        "validated_s3_probe_cells": probe_valid,
    }


def _public_trackable_files(experiment: Path, repo: Path) -> tuple[list[Path], bool]:
    try:
        relative_root = experiment.relative_to(repo).as_posix()
    except ValueError:
        return [], False
    result = _run_git(
        repo,
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", relative_root],
    )
    if result.returncode != 0:
        return [], False
    paths: list[Path] = []
    for relative in result.stdout.split("\0"):
        if not relative:
            continue
        path = repo / relative
        if path.exists() or path.is_symlink():
            paths.append(path)
    return sorted(set(paths)), True


def _private_asset(relative: Path, *, is_directory: bool) -> bool:
    if _has_private_component(relative):
        return True
    if relative.parts and relative.parts[0] in PRIVATE_TREE_ROOTS:
        if not is_directory and relative.name == ".gitkeep":
            return False
        return True
    return False


def _private_path_token(relative: Path) -> str:
    return hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]


def _audit_private_git_hygiene(
    experiment: Path, repo: Path | None
) -> dict[str, Any]:
    """Require every private path to be ignored and absent from Git inventories."""

    if repo is None:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "git_repository_unavailable_for_private_hygiene"}],
            "private_paths_checked": 0,
        }
    try:
        experiment_relative = experiment.relative_to(repo).as_posix()
    except ValueError:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "experiment_outside_git_repository"}],
            "private_paths_checked": 0,
        }
    cached_result = _run_git(
        repo, ["ls-files", "-z", "--cached", "--", experiment_relative]
    )
    others_result = _run_git(
        repo,
        ["ls-files", "-z", "--others", "--exclude-standard", "--", experiment_relative],
    )
    if cached_result.returncode != 0 or others_result.returncode != 0:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "git_inventory_failed_for_private_hygiene"}],
            "private_paths_checked": 0,
        }
    cached = {value for value in cached_result.stdout.split("\0") if value}
    nonignored = {value for value in others_result.stdout.split("\0") if value}
    findings: list[dict[str, Any]] = []
    private_paths: list[Path] = []
    for path in experiment.rglob("*"):
        relative = path.relative_to(experiment)
        if len(relative.parts) == 1 and relative.parts[0] in PRIVATE_TREE_ROOTS:
            # The roots themselves contain a tracked public .gitkeep; their
            # descendants are the private Git boundary.
            continue
        if _private_asset(relative, is_directory=path.is_dir() and not path.is_symlink()):
            private_paths.append(path)
    for path in sorted(private_paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(experiment)
        repo_relative = path.relative_to(repo).as_posix()
        token = _private_path_token(relative)
        if repo_relative in cached:
            findings.append({"finding": "private_path_cached_by_git", "path_token": token})
        if repo_relative in nonignored:
            findings.append(
                {"finding": "private_path_nonignored_untracked", "path_token": token}
            )
        if _git_ignored(repo, path) is not True:
            findings.append(
                {"finding": "private_path_not_git_ignored", "path_token": token}
            )
    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "private_paths_checked": len(private_paths),
    }


def _required_public_deliverables(experiment: Path) -> tuple[Path, ...]:
    relative = [
        "EXPERIMENT_PLAN.md",
        "PREREGISTRATION_LOCK.json",
        "reports/final_report.md",
        "metrics/p0_equivalence_gate.json",
        "metrics/p0_equivalence_by_cell.csv",
        "metrics/p0_probe_replication_gate.json",
        "metrics/p0_probe_replication_by_cell.csv",
        "metrics/p0_probe_replication_pooled_metrics.csv",
        "metrics/prospective_gates.json",
        "metrics/s3_trigger_authorization.json",
        "metrics/privacy_gate.json",
        *(f"metrics/{name}" for name in TABLE_FILES),
        *(f"figures/{name}" for name in FIGURE_FILES),
    ]
    return tuple(experiment / value for value in relative)


def audit_permissions(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
    *,
    public_files: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Require Git hygiene plus private 0700/0600 and public 0755/0644."""

    experiment = Path(experiment_root).resolve()
    repo = Path(repo_root).resolve()
    findings: list[dict[str, Any]] = []
    hygiene = _audit_private_git_hygiene(experiment, _git_root_for(experiment, repo))
    findings.extend(hygiene["findings"])
    if public_files is None:
        trackable, git_ok = _public_trackable_files(experiment, repo)
    else:
        trackable = [
            (Path(value) if Path(value).is_absolute() else experiment / Path(value)).resolve()
            for value in public_files
        ]
        git_ok = True
    if not git_ok:
        findings.append({"finding": "cannot_inventory_public_trackable_files"})

    trackable_resolved = {path.resolve() for path in trackable}
    if public_files is None:
        for required in _required_public_deliverables(experiment):
            relative = required.relative_to(experiment)
            if (
                required.resolve() not in trackable_resolved
                or required.is_symlink()
                or not required.is_file()
                or _git_ignored(repo, required) is not False
            ):
                findings.append(
                    {
                        "finding": "required_public_deliverable_not_trackable",
                        "file_token": _private_path_token(relative),
                    }
                )

    public_file_count = 0
    public_directories: set[Path] = {experiment}
    for path in trackable:
        try:
            relative = path.relative_to(experiment)
        except ValueError:
            findings.append({"finding": "public_trackable_path_escaped_experiment"})
            continue
        is_directory = path.is_dir() and not path.is_symlink()
        if _private_asset(relative, is_directory=is_directory):
            continue
        if path.is_symlink() or not path.is_file():
            findings.append(
                {
                    "finding": "public_deliverable_not_regular_file",
                    "file_token": _private_path_token(relative),
                }
            )
            continue
        public_file_count += 1
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode != 0o644:
            findings.append(
                {
                    "finding": "public_file_mode_not_0644",
                    "file_token": _private_path_token(relative),
                    "mode": f"{mode:04o}",
                }
            )
        parent = path.parent
        while parent != experiment.parent:
            parent_relative = parent.relative_to(experiment)
            if not _private_asset(parent_relative, is_directory=True):
                public_directories.add(parent)
            if parent == experiment:
                break
            parent = parent.parent

    for directory in sorted(public_directories):
        relative = Path(".") if directory == experiment else directory.relative_to(experiment)
        if directory.is_symlink() or not directory.is_dir():
            findings.append(
                {
                    "finding": "public_directory_missing_or_symlink",
                    "file_token": _private_path_token(relative),
                }
            )
            continue
        mode = stat.S_IMODE(directory.lstat().st_mode)
        if mode != 0o755:
            findings.append(
                {
                    "finding": "public_directory_mode_not_0755",
                    "file_token": _private_path_token(relative),
                    "mode": f"{mode:04o}",
                }
            )

    private_count = 0
    for path in experiment.rglob("*"):
        relative = path.relative_to(experiment)
        mode_bits = path.lstat().st_mode
        is_directory = stat.S_ISDIR(mode_bits)
        if not _private_asset(relative, is_directory=is_directory):
            continue
        private_count += 1
        token = _private_path_token(relative)
        if stat.S_ISLNK(mode_bits):
            findings.append({"finding": "private_asset_symlink", "path_token": token})
            continue
        expected_mode = 0o700 if is_directory else 0o600
        mode = stat.S_IMODE(mode_bits)
        if mode != expected_mode:
            findings.append(
                {
                    "finding": "private_asset_mode_mismatch",
                    "path_token": token,
                    "asset_type": "directory" if is_directory else "file",
                    "mode": f"{mode:04o}",
                    "expected_mode": f"{expected_mode:04o}",
                }
            )

    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "public_trackable_files_checked": public_file_count,
        "public_directories_checked": len(public_directories),
        "private_assets_checked": private_count,
        "private_git_paths_checked": hygiene["private_paths_checked"],
        "private_git_hygiene_passed": hygiene["status"] == "PASS",
        "private_paths_disclosed": False,
    }


def _strip_markdown_code(text: str) -> str:
    without_fences = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    return re.sub(r"`[^`\r\n]*`", "", without_fences)


def _inline_markdown_destinations(text: str) -> list[str]:
    destinations: list[str] = []
    index = 0
    while True:
        start = text.find("]( ", index)
        compact = text.find("](", index)
        if compact < 0:
            break
        # A space between ] and ( is not standard inline-link syntax.
        if start >= 0 and start < compact:
            index = start + 3
            continue
        cursor = compact + 2
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        if text[cursor] == "<":
            end = text.find(">", cursor + 1)
            if end >= 0:
                destinations.append(text[cursor + 1 : end])
                index = end + 1
                continue
        depth = 0
        escaped = False
        end = cursor
        while end < len(text):
            character = text[end]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "(":
                depth += 1
            elif character == ")":
                if depth == 0:
                    break
                depth -= 1
            elif character.isspace() and depth == 0:
                break
            end += 1
        if end > cursor:
            destinations.append(text[cursor:end])
        index = max(end + 1, compact + 2)
    return destinations


def markdown_destinations(text: str) -> list[str]:
    clean = _strip_markdown_code(text)
    destinations = _inline_markdown_destinations(clean)
    for match in re.finditer(r"^\s*\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))", clean, re.MULTILINE):
        destinations.append(match.group(1) or match.group(2))
    return destinations


def audit_report_links(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    report_relative: str | Path = "reports/final_report.md",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Allow only safe web/mail/anchor links or regular public local files."""

    experiment = Path(experiment_root).resolve()
    report = (experiment / report_relative).resolve()
    repo = _git_root_for(experiment, repo_root)
    findings: list[dict[str, Any]] = []
    try:
        if not report.is_relative_to(experiment) or not report.is_file():
            raise FileNotFoundError
        payload = report.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [{"finding": "final_report_missing_or_unreadable"}],
            "local_links_checked": 0,
            "external_or_anchor_links_skipped": 0,
        }

    local_count = 0
    skipped = 0
    for index, raw in enumerate(markdown_destinations(payload), start=1):
        destination = raw.strip()
        if not destination or destination.startswith("#"):
            skipped += 1
            continue
        parsed = urlsplit(destination)
        scheme = parsed.scheme.casefold()
        if scheme in {"http", "https", "mailto"} and not destination.startswith("//"):
            skipped += 1
            continue
        if scheme or parsed.netloc or destination.startswith("//"):
            findings.append({"finding": "forbidden_markdown_link_scheme", "link_index": index})
            continue
        local_count += 1
        decoded = unquote(parsed.path)
        if "\\" in decoded:
            findings.append({"finding": "non_relative_local_markdown_link", "link_index": index})
            continue
        candidate_raw = Path(decoded)
        if candidate_raw.is_absolute() or not decoded:
            findings.append({"finding": "non_relative_local_markdown_link", "link_index": index})
            continue
        lexical = report.parent / candidate_raw
        symlink_component = False
        cursor = report.parent
        for component in candidate_raw.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                symlink_component = True
                break
        candidate = lexical.resolve()
        if not candidate.is_relative_to(experiment):
            findings.append({"finding": "markdown_link_escaped_experiment", "link_index": index})
            continue
        relative = candidate.relative_to(experiment)
        if _private_asset(relative, is_directory=candidate.is_dir()):
            findings.append({"finding": "markdown_link_target_private", "link_index": index})
        elif symlink_component or candidate.is_symlink():
            findings.append({"finding": "markdown_link_target_symlink", "link_index": index})
        elif not candidate.exists():
            findings.append({"finding": "markdown_link_target_missing", "link_index": index})
        elif not candidate.is_file():
            findings.append({"finding": "markdown_link_target_not_regular", "link_index": index})
        elif repo is None or _git_ignored(repo, candidate) is not False:
            findings.append({"finding": "markdown_link_target_ignored", "link_index": index})

    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "local_links_checked": local_count,
        "external_or_anchor_links_skipped": skipped,
    }


def _walk_json_flags(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_json_flags(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_flags(child)


def audit_no_new_training(root: str | Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Check every JSON training declaration and frozen prohibition."""

    experiment = Path(root).resolve()
    findings: list[dict[str, Any]] = []
    declarations = 0
    json_files = 0
    for path in sorted(experiment.rglob("*.json")):
        if path.relative_to(experiment).as_posix() in CONTROL_OUTPUTS:
            continue
        json_files += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            findings.append({"finding": "json_unreadable_during_training_audit"})
            continue
        for key, value in _walk_json_flags(payload):
            if key.casefold() in {"training_performed", "new_training_performed"}:
                declarations += 1
                if value is not False:
                    findings.append({"finding": "training_declaration_not_false"})
    try:
        config = json.loads((experiment / "configs/audit.json").read_text(encoding="utf-8"))
        lock = json.loads(
            (experiment / "PREREGISTRATION_LOCK.json").read_text(encoding="utf-8")
        )
        if config.get("training_forbidden") is not True:
            findings.append({"finding": "training_not_forbidden_by_frozen_config"})
        if lock.get("new_training_performed") is not False:
            findings.append({"finding": "preregistration_training_flag_drift"})
    except (OSError, UnicodeError, json.JSONDecodeError):
        findings.append({"finding": "training_contract_files_invalid"})
    if declarations == 0:
        findings.append({"finding": "no_training_declarations_found"})
    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "json_files_checked": json_files,
        "false_training_declarations_checked": declarations,
        "new_training_performed": False if not findings else None,
    }


def audit_public_deliverables(root: str | Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    experiment = Path(root).resolve()
    findings: list[dict[str, Any]] = []
    table_rows: dict[str, int] = {}
    for name in TABLE_FILES:
        path = experiment / "metrics" / name
        try:
            rows = _read_csv_exact(path, TABLE_SCHEMAS[name])
            if not rows or any(not any(str(value).strip() for value in row.values()) for row in rows):
                raise ValueError
            if name == "table1_feature_map_contract.csv" and len(rows) != 4:
                raise ValueError
            if name == "table7_training_budget.csv":
                identities = {
                    f"seed_{int(row['seed'])}/{row['arm']}/fold_{int(row['fold'])}"
                    for row in rows
                }
                if len(rows) != 40 or identities != _expected_cell_names():
                    raise ValueError
            table_rows[name] = len(rows)
        except (OSError, UnicodeError, KeyError, TypeError, ValueError, csv.Error):
            findings.append(
                {
                    "finding": "public_table_schema_or_rows_invalid",
                    "file_token": _path_token(name),
                }
            )
    figure_root = experiment / "figures"
    live_png = {
        path.relative_to(figure_root).as_posix(): path
        for path in figure_root.rglob("*")
        if path.suffix.casefold() == ".png"
    } if figure_root.is_dir() else {}
    if set(live_png) != set(FIGURE_FILES):
        findings.append({"finding": "public_png_inventory_not_exact_12"})
    figure_hashes: dict[str, str] = {}
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore[assignment]
        findings.append({"finding": "pillow_unavailable_for_png_validation"})
    for name in FIGURE_FILES:
        path = figure_root / name
        token = _path_token(name)
        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or not stat.S_ISREG(path.lstat().st_mode)
                or stat.S_IMODE(path.lstat().st_mode) != 0o644
                or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
                or Image is None
            ):
                raise ValueError
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.width < 512 or image.height < 256:
                    raise ValueError
                extrema = image.getextrema()
                extrema_rows = [extrema] if extrema and isinstance(extrema[0], int) else list(extrema)
                if not extrema_rows or not any(low != high for low, high in extrema_rows):
                    raise ValueError
                dpi = image.info.get("dpi")
                if (
                    image.info.get("Software") != "c1b_spatial_audit"
                    or not isinstance(image.info.get("Title"), str)
                    or not image.info["Title"].strip()
                    or not isinstance(dpi, tuple)
                    or len(dpi) != 2
                    or any(abs(float(value) - 200.0) > 1.0 for value in dpi)
                ):
                    raise ValueError
            figure_hashes[name] = file_sha256(path)
        except (OSError, TypeError, ValueError):
            findings.append(
                {"finding": "registered_png_decode_or_contract_invalid", "file_token": token}
            )

    prospective, prospective_findings = _load_prospective(experiment)
    findings.extend(prospective_findings)
    report_path = experiment / "reports/final_report.md"
    answer_count = 0
    try:
        report = report_path.read_text(encoding="utf-8", errors="strict")
        classification_matches = re.findall(
            r"^\s*FINAL_CLASSIFICATION\s*:\s*(.*?)\s*$", report, re.MULTILINE
        )
        next_matches = re.findall(r"^\s*NEXT\s*:\s*(.*?)\s*$", report, re.MULTILINE)
        answers = [
            int(value)
            for value in re.findall(r"^\s*#{1,6}\s+([1-9]|1[0-4])\.\s+", report, re.MULTILINE)
        ]
        answer_count = len(answers)
        classification = prospective.get("classification", {})
        if (
            len(classification_matches) != 1
            or len(next_matches) != 1
            or classification_matches[0] != classification.get("classification")
            or next_matches[0] != classification.get("next")
            or answers != list(range(1, 15))
        ):
            raise ValueError
    except (OSError, UnicodeError, TypeError, ValueError):
        findings.append({"finding": "final_report_markers_or_14_answers_invalid"})
    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "required_public_tables": 7,
        "validated_public_tables": len(table_rows),
        "table_row_counts": table_rows,
        "required_registered_png_figures": 12,
        "validated_registered_png_figures": len(figure_hashes),
        "registered_png_sha256": figure_hashes,
        "numbered_report_answers": answer_count,
    }


def _guard(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return function()
    except BaseException as error:  # fail closed, without serializing sensitive text
        return {
            "schema_version": 1,
            "status": "FAIL",
            "findings": [
                {
                    "finding": "unexpected_validation_exception",
                    "check": name,
                    "exception_type": type(error).__name__,
                }
            ],
        }


def run_final_validation(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run every closure check and bind to a current PASS privacy gate."""

    experiment = Path(experiment_root).resolve()
    repo = Path(repo_root).resolve()

    current_privacy = _guard(
        "privacy_recheck", lambda: scan_public_artifacts(experiment, repo)
    )
    gate_path = experiment / "metrics/privacy_gate.json"
    privacy_binding_findings: list[dict[str, Any]] = []
    try:
        frozen_privacy = json.loads(gate_path.read_text(encoding="utf-8"))
        if frozen_privacy != current_privacy or frozen_privacy.get("status") != "PASS":
            privacy_binding_findings.append({"finding": "privacy_gate_stale_or_failed"})
    except (OSError, UnicodeError, json.JSONDecodeError):
        privacy_binding_findings.append({"finding": "privacy_gate_missing_or_invalid"})
    privacy_binding = {
        "schema_version": 1,
        "status": "PASS" if not privacy_binding_findings else "FAIL",
        "findings": privacy_binding_findings,
        "privacy_gate_sha256": file_sha256(gate_path) if gate_path.is_file() else None,
        "privacy_gate_current": not privacy_binding_findings,
    }

    checks = {
        "privacy_gate_binding": privacy_binding,
        "locked_input_integrity": _guard(
            "locked_input_integrity", lambda: audit_locked_inputs(experiment, repo)
        ),
        "p0_stop_gates": _guard(
            "p0_stop_gates", lambda: audit_p0_gates(experiment, repo)
        ),
        "final_inventories": _guard(
            "final_inventories", lambda: audit_final_inventories(experiment, repo)
        ),
        "conditional_s3": _guard(
            "conditional_s3", lambda: audit_conditional_s3(experiment, repo)
        ),
        "no_new_training": _guard(
            "no_new_training", lambda: audit_no_new_training(experiment)
        ),
        "permissions": _guard(
            "permissions", lambda: audit_permissions(experiment, repo)
        ),
        "report_links": _guard(
            "report_links", lambda: audit_report_links(experiment, repo_root=repo)
        ),
        "public_deliverables": _guard(
            "public_deliverables", lambda: audit_public_deliverables(experiment)
        ),
    }
    status = "PASS" if all(check.get("status") == "PASS" for check in checks.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "all_checks_passed": status == "PASS",
        "scientific_metrics_recomputed": False,
        "new_training_performed": False if status == "PASS" else None,
    }


__all__ = [
    "CONTROL_OUTPUTS",
    "EXPERIMENT_ROOT",
    "REPO_ROOT",
    "atomic_json",
    "audit_conditional_s3",
    "audit_final_inventories",
    "audit_locked_inputs",
    "audit_no_new_training",
    "audit_p0_gates",
    "audit_permissions",
    "audit_public_deliverables",
    "audit_report_links",
    "canonical_sha256",
    "collect_private_patient_ids",
    "file_sha256",
    "markdown_destinations",
    "public_text_paths",
    "run_final_validation",
    "scan_public_artifacts",
]
