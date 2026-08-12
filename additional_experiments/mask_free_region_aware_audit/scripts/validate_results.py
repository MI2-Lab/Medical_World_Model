#!/usr/bin/env python3
"""Fail-closed delivery validation for the mask-free regional audit.

Validation is presentation-only: it reads public aggregate outputs and Git
metadata.  It never recomputes science from private predictions or labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_figures import (  # noqa: E402
    FIGURES,
    FIGURE_MANIFEST_COLUMNS,
    PUBLIC_TABLES,
    file_sha256,
    load_public_tables,
    numeric_column,
    resolve_column,
)
from generate_report import (  # noqa: E402
    GATES_PATH,
    REPORT_MANIFEST_PATH,
    REPORT_PATH,
    REQUIRED_REPORT_MARKERS,
    RUN_SUMMARY_PATH,
    _bool_value,
    _validate_run_summary,
    extract_gate_results,
)


MAX_TRACKED_BYTES = 20 * 1024 * 1024
FORBIDDEN_TRACKED_SUFFIXES = {
    ".nii",
    ".nii.gz",
    ".dcm",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".safetensors",
}
PRIVATE_PATIENT_DIRECTORIES = {"features", "predictions", "logs", "checkpoints", "data"}
IDENTIFIER_KEY_RE = re.compile(
    r"^(?:patient|subject|participant|clinical_patient|raw_patient|mrn)(?:_?id)?s?$",
    flags=re.IGNORECASE,
)
PATIENT_TOKEN_RE = re.compile(r"\bACRIN[-_ ]?\d+\b", flags=re.IGNORECASE)
ABSOLUTE_ENV_PATH_RE = re.compile(r"(?:^|[\s\"'`(])/(?:data|home|mnt|scratch)/")

REQUIRED_PUBLIC_OUTPUTS: tuple[Path, ...] = (
    *(Path("metrics") / filename for filename in PUBLIC_TABLES.values()),
    GATES_PATH,
    RUN_SUMMARY_PATH,
    Path("metrics/figure_manifest.csv"),
    REPORT_PATH,
    REPORT_MANIFEST_PATH,
    *(Path("figures") / filename for filename in FIGURES),
)
FEATURE_COMPLETION_PATH = Path("features/feature_matrix_complete.private.json")
FEATURE_COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "experiment",
        "status",
        "cell_count",
        "config_sha256",
        "preregistration_lock_sha256",
        "geometry_contract_sha256",
        "cells",
    }
)
FEATURE_COMPLETION_CELL_KEYS = frozenset(
    {
        "cell",
        "seed_base",
        "arm",
        "fold",
        "feature_path",
        "feature_sha256",
        "metadata_path",
        "metadata_sha256",
        "patient_order_sha256",
    }
)

CLASSIFICATION_METRIC_COLUMNS = (
    "seed", "arm", "analysis", "context", "view", "timing_label", "target",
    "variant", "model", "population", "clinical_contract", "n", "n_positive",
    "n_negative", "n_classes", "auroc", "auprc", "balanced_accuracy", "brier",
)
PUBLIC_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "mri_only_pcr": (
        *CLASSIFICATION_METRIC_COLUMNS,
        "timing", "r0_auroc", "delta_auroc_vs_r0", "delta_auroc",
    ),
    "clinical_pcr": (
        *CLASSIFICATION_METRIC_COLUMNS,
        "timing", "c_auroc", "c_auprc", "c_brier", "delta_auroc_vs_C",
        "delta_auprc_vs_C", "brier_improvement_vs_C",
    ),
    "clinical_ftv_incremental": (
        *CLASSIFICATION_METRIC_COLUMNS,
        "delta_auroc_vs_C+F+R0", "delta_auprc_vs_C+F+R0",
        "brier_improvement_vs_C+F+R0", "delta_auroc_vs_C+F",
        "delta_auprc_vs_C+F", "brier_improvement_vs_C+F", "timing",
        "delta_auroc_vs_cf_r0", "delta_auroc",
    ),
    "phenotype": (
        *CLASSIFICATION_METRIC_COLUMNS,
        "visit", "r0_auroc", "delta_auroc_vs_r0",
    ),
    "ftv": (
        "seed", "arm", "variant", "feature_dim", "task", "endpoint", "view",
        "target", "analysis_scope", "target_semantics", "aggregation", "n_test",
        "spearman", "pearson", "r2", "rmse", "mae", "b0_rmse",
        "rmse_gain_over_b0", "prediction_target_variance_ratio",
        "calibration_slope", "calibration_intercept", "calibration_mean_bias",
        "timing",
    ),
    "oracle_recovery": (
        "row_type", "seed", "arm", "view", "target", "population", "source",
        "variant", "candidate", "reference", "n", "n_positive", "n_negative",
        "auroc", "auprc", "balanced_accuracy", "brier", "r0_auroc",
        "candidate_auroc", "numerator_auroc_uplift", "published_fixed_p3_auroc",
        "published_peri20_auroc", "published_oracle_uplift", "recovery_ratio",
        "recovery_defined", "representation_note", "matched_patient_sha256",
    ),
    "bootstrap": (
        "seed", "arm", "context", "view", "timing", "target", "population",
        "candidate", "reference_model", "comparison_model", "metric", "reference",
        "comparison", "estimate", "improvement", "delta_auroc", "ci_lower",
        "ci_upper", "confidence_level", "n_patients", "n_folds", "n_bootstrap",
        "n_valid_bootstrap", "bootstrap_unit", "ci_method", "orientation",
        "bootstrap_seed",
    ),
    "seed_consistency": (
        "context", "arm", "view", "timing", "target", "population", "candidate",
        "reference", "seed_2026_delta_auroc", "seed_3026_delta_auroc",
        "mean_delta_auroc", "both_seeds_strictly_positive",
    ),
    "timing_sensitivity": (
        "seed", "arm", "context", "view", "timing", "timing_label", "target",
        "population", "variant", "reference", "auroc", "reference_auroc",
        "delta_auroc_vs_r0",
    ),
}
RUN_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "experiment",
        "status",
        "branch",
        "commit_sha",
        "push_status",
        "push_error",
        "elapsed_seconds",
        "feature_cells_validated_before_labels",
        "formal_bootstrap_replicates",
        "outer_test_predicted_once_per_model",
        "new_encoder_or_jepa_training_performed",
        "architecture_intervention_started",
        "r0_goal5_p1_prediction_parity",
        "public_outputs_contain_patient_level_data",
        "private_patient_outputs_mode",
        "scientific_classification",
        "gate_results",
        "public_outputs",
        "private_outputs",
    }
)
RUN_PUBLIC_OUTPUTS = frozenset(
    f"metrics/{filename}" for filename in PUBLIC_TABLES.values()
)
RUN_PRIVATE_OUTPUTS = frozenset(
    {
        "predictions/mri_only_pcr_oof.private.csv",
        "predictions/clinical_pcr_oof.private.csv",
        "predictions/clinical_ftv_pcr_oof.private.csv",
        "predictions/phenotype_oof.private.csv",
        "predictions/ftv_oof.private.csv",
        "predictions/hyperparameters.private.csv",
        "predictions/bootstrap_draws.private.csv",
    }
)


class ValidationError(ValueError):
    """A user-readable closure failure."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--old-tree-manifest",
        type=Path,
        default=None,
        help="Optional pre-run file/tree hash manifest proving old directories unchanged.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/final_validation.json"),
        help="Output path relative to --root (default: metrics/final_validation.json).",
    )
    parser.add_argument(
        "--allow-pending-git",
        action="store_true",
        help=(
            "Permit the exact pre-delivery Git state PENDING/PENDING/null. "
            "Final validation rejects pending provenance by default."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = file_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if file_sha256(path) != before:
        raise RuntimeError(f"JSON changed while being validated: {path}")
    if not isinstance(payload, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return payload


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_required_outputs(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    empty: list[str] = []
    symlinks: list[str] = []
    for relative in REQUIRED_PUBLIC_OUTPUTS:
        path = root / relative
        if not path.exists():
            missing.append(relative.as_posix())
        elif path.is_symlink():
            symlinks.append(relative.as_posix())
        elif not path.is_file() or path.stat().st_size == 0:
            empty.append(relative.as_posix())
    if missing or empty or symlinks:
        raise ValidationError(
            f"required-output inventory failed; missing={missing}, empty={empty}, symlinks={symlinks}"
        )
    return {
        "status": "PASS",
        "required_output_count": len(REQUIRED_PUBLIC_OUTPUTS),
        "missing": [],
        "empty": [],
        "symlinks": [],
    }


def _walk_json(value: Any, *, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, child
            yield from _walk_json(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield child_path, child
            yield from _walk_json(child, path=child_path)


def _check_public_text(path: Path, text: str) -> None:
    if PATIENT_TOKEN_RE.search(text):
        raise ValidationError(f"public artifact contains a patient-like token: {path}")
    if ABSOLUTE_ENV_PATH_RE.search(text):
        raise ValidationError(f"public artifact contains an absolute environment path: {path}")
    if "BEGIN PRIVATE KEY" in text or re.search(
        r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b", text
    ):
        raise ValidationError(f"public artifact contains a credential-like token: {path}")


def scan_public_artifacts(root: Path) -> dict[str, Any]:
    """Scan every public metrics/report artifact, including extra outputs."""

    paths: list[Path] = []
    for directory in (root / "metrics", root / "reports"):
        if not directory.exists():
            continue
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        )
    scanned_bytes = 0
    for path in sorted(paths):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValidationError(f"public artifact may not be a symlink: {relative}")
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            raise ValidationError(f"public artifact exceeds size ceiling: {relative}")
        scanned_bytes += size
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".json", ".md", ".txt"}:
            raise ValidationError(f"unsupported public result format: {relative}")
        text = path.read_text(encoding="utf-8", errors="strict")
        _check_public_text(relative, text)
        if suffix == ".csv":
            frame = pd.read_csv(path, nrows=5)
            bad = [
                str(column)
                for column in frame.columns
                if IDENTIFIER_KEY_RE.fullmatch(
                    re.sub(r"[^A-Za-z0-9_]+", "_", str(column).strip())
                )
            ]
            if bad:
                raise ValidationError(
                    f"public CSV exposes identifier columns {bad}: {relative}"
                )
        elif suffix == ".json":
            payload = json.loads(text)
            for json_path, value in _walk_json(payload):
                key = json_path.rsplit(".", 1)[-1]
                if "[" not in key and IDENTIFIER_KEY_RE.fullmatch(
                    re.sub(r"[^A-Za-z0-9_]+", "_", key)
                ):
                    raise ValidationError(
                        f"public JSON exposes identifier field {json_path}: {relative}"
                    )
                if isinstance(value, str):
                    _check_public_text(relative, value)
    return {
        "status": "PASS",
        "files_scanned": len(paths),
        "bytes_scanned": scanned_bytes,
        "patient_identifier_findings": 0,
        "absolute_path_findings": 0,
        "credential_findings": 0,
    }


def _validate_probability_metrics(frame: pd.DataFrame, name: str) -> None:
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        is_delta = any(token in normalized for token in ("delta", "gain", "uplift", "improvement"))
        is_probability_metric = (
            normalized in {"auroc", "auprc", "brier", "balanced_accuracy", "accuracy"}
            or normalized.startswith("test_auroc")
            or normalized.startswith("oof_auroc")
        )
        if is_probability_metric and not is_delta:
            finite = values.dropna().to_numpy(float)
            if not np.isfinite(finite).all() or ((finite < 0) | (finite > 1)).any():
                raise ValidationError(f"{name} has invalid [0,1] metric column {column}")
        if normalized in {"n", "n_patients", "patient_count", "n_positive", "n_negative"}:
            finite = values.dropna().to_numpy(float)
            if not np.isfinite(finite).all() or (finite < 0).any() or not np.equal(
                finite, np.floor(finite)
            ).all():
                raise ValidationError(f"{name} has invalid aggregate count column {column}")


def validate_public_tables(root: Path) -> dict[str, Any]:
    tables = load_public_tables(root)
    rows: dict[str, int] = {}
    for logical_name, frame in tables.items():
        if logical_name == "occupancy":
            if tuple(frame.columns[:3].astype(str)) != ("geometry", "region", "variant"):
                raise ValidationError(
                    "region_occupancy.csv must begin with geometry,region,variant"
                )
            if len(frame.columns) < 4:
                raise ValidationError("region_occupancy.csv has no numeric occupancy columns")
            numeric_tail = frame.iloc[:, 3:].apply(pd.to_numeric, errors="coerce")
            if numeric_tail.notna().sum().sum() == 0:
                raise ValidationError("region_occupancy.csv has no numeric occupancy values")
        else:
            expected = PUBLIC_TABLE_COLUMNS[logical_name]
            if tuple(frame.columns.astype(str)) != expected:
                raise ValidationError(
                    f"{PUBLIC_TABLES[logical_name]} schema drifted; "
                    f"observed={tuple(frame.columns)}, expected={expected}"
                )
        _validate_probability_metrics(frame, PUBLIC_TABLES[logical_name])
        rows[logical_name] = len(frame)

    oracle = tables["oracle_recovery"]
    ratio = pd.to_numeric(oracle["recovery_ratio"], errors="coerce")
    denominator = pd.to_numeric(
        oracle["published_oracle_uplift"], errors="coerce"
    )
    recovery_defined = oracle["recovery_defined"].map(_strict_optional_bool)
    malformed_defined = oracle["recovery_defined"].notna() & recovery_defined.isna()
    if malformed_defined.any():
        raise ValidationError("Oracle recovery_defined contains a non-boolean value")
    ratio_present = oracle["recovery_ratio"].notna()
    if ratio_present.any() and not np.isfinite(ratio.loc[ratio_present].to_numpy(float)).all():
        raise ValidationError("Oracle recovery_ratio contains a non-finite value")
    declared_defined = recovery_defined.fillna(False).astype(bool)
    if (declared_defined & ~ratio_present).any():
        raise ValidationError("Oracle recovery_defined is true without a recovery_ratio")
    requires_denominator = declared_defined | ratio_present
    required_values = denominator.loc[requires_denominator].to_numpy(float)
    if (
        not np.isfinite(required_values).all()
        or (required_values <= 0).any()
    ):
        raise ValidationError(
            "Oracle recovery requires a positive finite published_oracle_uplift"
        )
    return {"status": "PASS", "row_counts": rows}


def _strict_optional_bool(value: Any) -> bool | float:
    """Parse public CSV booleans without treating arbitrary text as truthy."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return np.nan


def validate_gates_and_summary(
    root: Path,
    *,
    allow_pending_git: bool = False,
) -> dict[str, Any]:
    gates_payload = _read_json(root / GATES_PATH)
    expected_gate_top = {
        "schema_version",
        "experiment",
        "status",
        "gates",
        "any_primary_candidate_two_seed_positive",
        "scientific_classification",
        "classification_precedence",
        "contains_patient_level_data",
    }
    if set(gates_payload) != expected_gate_top:
        raise ValidationError("gates.json top-level schema drifted")
    if (
        gates_payload.get("schema_version") != 1
        or gates_payload.get("experiment") != "mask_free_region_aware_audit"
        or gates_payload.get("status") != "COMPLETE"
        or gates_payload.get("contains_patient_level_data") is not False
    ):
        raise ValidationError("gates.json identity/status/privacy contract failed")
    gate_objects = gates_payload.get("gates")
    if not isinstance(gate_objects, Mapping) or set(gate_objects) != set("ABCD"):
        raise ValidationError("gates.json does not contain exactly Gates A-D")
    for letter, gate in gate_objects.items():
        if not isinstance(gate, Mapping) or set(gate) != {
            "name",
            "passed",
            "evaluated_comparisons",
            "supporting_comparisons",
        }:
            raise ValidationError(f"Gate {letter} schema drifted")
    gates = extract_gate_results(gates_payload)
    summary = _read_json(root / RUN_SUMMARY_PATH)
    if set(summary) != RUN_SUMMARY_KEYS:
        raise ValidationError("run_summary.json schema drifted")
    _validate_run_summary(summary)
    for key in (
        "public_outputs_contain_patient_level_data",
        "contains_patient_level_data",
        "public_contains_patient_level_data",
    ):
        if key in summary and _bool_value(summary[key]) is not False:
            raise ValidationError(f"run_summary privacy attestation is not false: {key}")
    if (
        summary.get("schema_version") != 1
        or summary.get("experiment") != "mask_free_region_aware_audit"
        or summary.get("feature_cells_validated_before_labels") != 20
        or summary.get("formal_bootstrap_replicates") != 2000
        or summary.get("outer_test_predicted_once_per_model") is not True
        or summary.get("new_encoder_or_jepa_training_performed") is not False
        or summary.get("architecture_intervention_started") is not False
        or summary.get("private_patient_outputs_mode") != "0600"
    ):
        raise ValidationError("run_summary execution/safety contract drifted")
    config = _read_json(root / "configs" / "audit.json")
    if summary.get("branch") != config.get("branch"):
        raise ValidationError("run_summary branch differs from audit config")
    elapsed = summary.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
        raise ValidationError("run_summary elapsed_seconds is invalid")
    recorded_gates = summary.get("gate_results")
    if recorded_gates != {letter: gates[letter] for letter in "ABCD"}:
        raise ValidationError("run_summary gate results differ from gates.json")
    if summary.get("scientific_classification") != gates_payload.get(
        "scientific_classification"
    ):
        raise ValidationError("run_summary classification differs from gates.json")

    git_provenance = validate_git_delivery_provenance(
        root,
        summary,
        allow_pending_git=allow_pending_git,
    )

    for key, expected_paths, private in (
        ("public_outputs", RUN_PUBLIC_OUTPUTS, False),
        ("private_outputs", RUN_PRIVATE_OUTPUTS, True),
    ):
        artifacts = summary.get(key)
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_paths):
            raise ValidationError(f"run_summary {key} inventory drifted")
        for relative, expected_hash in artifacts.items():
            if Path(str(relative)).is_absolute() or ".." in Path(str(relative)).parts:
                raise ValidationError(f"run_summary has unsafe artifact path: {relative}")
            artifact = root / str(relative)
            if not artifact.is_file() or file_sha256(artifact) != str(expected_hash):
                raise ValidationError(f"run_summary artifact hash drifted: {relative}")
            if private and artifact.stat().st_mode & 0o077:
                raise ValidationError(f"run_summary private artifact is not owner-only: {relative}")
    return {
        "status": "PASS",
        "gates": {letter: bool(gates[letter]) for letter in "ABCD"},
        "run_status": str(summary.get("status", summary.get("run_status", ""))),
        "git_provenance": git_provenance,
    }


def validate_feature_completion(root: Path) -> dict[str, Any]:
    """Authenticate completion metadata without opening patient feature arrays."""

    completion_path = root / FEATURE_COMPLETION_PATH
    if not completion_path.is_file():
        raise ValidationError(f"feature completion manifest is missing: {completion_path}")
    if completion_path.stat().st_mode & 0o077:
        raise ValidationError("private feature completion manifest is not owner-only")
    payload = _read_json(completion_path)
    if set(payload) != FEATURE_COMPLETION_KEYS:
        raise ValidationError("feature completion manifest schema drifted")
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") != "mask_free_region_aware_audit"
        or payload.get("status") != "COMPLETE"
        or payload.get("cell_count") != 20
    ):
        raise ValidationError("feature matrix is not exactly 20/20 COMPLETE")
    hash_bindings = (
        ("config_sha256", root / "configs" / "audit.json"),
        ("preregistration_lock_sha256", root / "PREREGISTRATION_LOCK.json"),
        ("geometry_contract_sha256", root / "metrics" / "region_occupancy_contract.json"),
    )
    for key, artifact in hash_bindings:
        digest = str(payload.get(key, ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError(f"feature completion has invalid {key}")
        if not artifact.is_file() or file_sha256(artifact) != digest:
            raise ValidationError(f"feature completion {key} binding drifted")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 20:
        raise ValidationError("feature completion cells are not a 20-entry list")
    expected = {
        (seed, arm, fold)
        for seed in (2026, 3026)
        for arm in ("LOCAL0", "LOCAL3")
        for fold in range(5)
    }
    observed: set[tuple[int, str, int]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping) or set(cell) != FEATURE_COMPLETION_CELL_KEYS:
            raise ValidationError("feature completion cell schema drifted")
        identity = (int(cell["seed_base"]), str(cell["arm"]), int(cell["fold"]))
        expected_name = f"seed_{identity[0]}/{identity[1]}/fold_{identity[2]}"
        if str(cell["cell"]) != expected_name or identity in observed:
            raise ValidationError("feature completion cell identity drifted/repeated")
        observed.add(identity)
        for key in ("feature_sha256", "metadata_sha256", "patient_order_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(cell[key])):
                raise ValidationError(f"feature completion has invalid {key}")
        # Paths are private and may be absolute in this owner-only manifest.
        # Resolve and hash them, but never load their patient-level contents.
        for path_key, hash_key in (
            ("feature_path", "feature_sha256"),
            ("metadata_path", "metadata_sha256"),
        ):
            artifact = Path(str(cell[path_key])).resolve()
            try:
                artifact.relative_to((root / "features").resolve())
            except ValueError as error:
                raise ValidationError(
                    f"private feature artifact escaped feature root: {cell['cell']}"
                ) from error
            if not artifact.is_file() or artifact.stat().st_mode & 0o077:
                raise ValidationError(f"private feature artifact missing/not owner-only: {cell['cell']}")
            if file_sha256(artifact) != str(cell[hash_key]):
                raise ValidationError(f"private feature artifact hash drifted: {cell['cell']}")
    if observed != expected:
        raise ValidationError("feature completion 2x2x5 grid drifted")
    return {
        "status": "PASS",
        "cell_count": 20,
        "matrix": "2 seeds x 2 arms x 5 folds",
        "completion_sha256": file_sha256(completion_path),
        "patient_feature_arrays_opened": False,
    }


def validate_figure_manifest(root: Path) -> dict[str, Any]:
    path = root / "metrics" / "figure_manifest.csv"
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(manifest.columns.astype(str)) != FIGURE_MANIFEST_COLUMNS:
        raise ValidationError("figure manifest schema drifted")
    if manifest["figure"].duplicated().any() or set(manifest["figure"]) != set(FIGURES):
        raise ValidationError("figure manifest inventory drifted")
    for row in manifest.to_dict("records"):
        destination = root / "figures" / row["figure"]
        if int(row["size_bytes"]) != destination.stat().st_size:
            raise ValidationError(f"figure size drifted: {row['figure']}")
        if row["sha256"] != file_sha256(destination):
            raise ValidationError(f"figure hash drifted: {row['figure']}")
        source_files = json.loads(row["source_files"])
        source_hashes = json.loads(row["source_sha256"])
        if not isinstance(source_files, list) or not isinstance(source_hashes, dict):
            raise ValidationError(f"figure source declaration is invalid: {row['figure']}")
        if set(source_files) != set(source_hashes):
            raise ValidationError(f"figure source hash inventory drifted: {row['figure']}")
        for source in source_files:
            source_path = root / source if source.startswith("configs/") else root / "metrics" / source
            if not source_path.is_file() or source_hashes[source] != file_sha256(source_path):
                raise ValidationError(f"figure source drifted: {row['figure']} <- {source}")
    return {"status": "PASS", "figures_authenticated": len(manifest)}


def validate_report(root: Path) -> dict[str, Any]:
    report_path = root / REPORT_PATH
    report = report_path.read_text(encoding="utf-8", errors="strict")
    missing = [marker for marker in REQUIRED_REPORT_MARKERS if marker not in report]
    if missing:
        raise ValidationError(f"final report lacks required answers/boundaries: {missing}")
    for filename in FIGURES:
        if f"../figures/{filename}" not in report:
            raise ValidationError(f"final report does not link required figure: {filename}")
    for logical_name in (
        "occupancy",
        "mri_only_pcr",
        "phenotype",
        "clinical_ftv_incremental",
        "ftv",
        "oracle_recovery",
        "bootstrap",
        "seed_consistency",
        "timing_sensitivity",
    ):
        filename = PUBLIC_TABLES[logical_name]
        if f"../metrics/{filename}" not in report:
            raise ValidationError(f"final report does not link required table: {filename}")

    manifest = _read_json(root / REPORT_MANIFEST_PATH)
    if manifest.get("report") != REPORT_PATH.as_posix():
        raise ValidationError("report manifest path drifted")
    if manifest.get("report_sha256") != file_sha256(report_path):
        raise ValidationError("report hash does not match report manifest")
    if manifest.get("public_outputs_contain_patient_level_data") is not False:
        raise ValidationError("report manifest privacy attestation is not false")
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        raise ValidationError("report manifest lacks source artifacts")
    seen: set[str] = set()
    for item in sources:
        if not isinstance(item, Mapping):
            raise ValidationError("report source item is not an object")
        relative = str(item.get("path", ""))
        if not relative or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValidationError("report manifest has duplicate/unsafe source path")
        seen.add(relative)
        source = root / relative
        if not source.is_file():
            raise ValidationError(f"report source is missing: {relative}")
        if item.get("sha256") != file_sha256(source) or int(item.get("size_bytes", -1)) != source.stat().st_size:
            raise ValidationError(f"report source drifted: {relative}")
        if item.get("patient_level_private") is not False:
            raise ValidationError(f"report declares a private source: {relative}")
    return {
        "status": "PASS",
        "questions_answered": 12,
        "required_boundary_markers": 4,
        "sources_authenticated": len(seen),
    }


def _git_repository(root: Path) -> Path:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ValidationError("audit root is not inside a Git worktree") from error
    repository = Path(output).resolve()
    try:
        root.resolve().relative_to(repository)
    except ValueError as error:
        raise ValidationError("audit root escaped its Git worktree") from error
    return repository


def _git_ok(repository: Path, *arguments: str) -> bool:
    return (
        subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).returncode
        == 0
    )


def _real_push_error(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        return None
    text = value.strip()
    placeholder = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if placeholder in {
        "pending",
        "unknown",
        "unknown_error",
        "none",
        "null",
        "na",
        "n_a",
        "not_recorded",
        "not_yet_recorded",
        "placeholder",
        "todo",
        "tbd",
    }:
        return None
    return text


def validate_git_delivery_provenance(
    root: Path,
    summary: Mapping[str, Any],
    *,
    allow_pending_git: bool = False,
) -> dict[str, Any]:
    """Authenticate the non-self-referential experiment delivery commit."""

    commit = summary.get("commit_sha")
    push_status = summary.get("push_status")
    push_error = summary.get("push_error")
    pending = commit == "PENDING" and push_status == "PENDING" and push_error is None
    any_pending = commit == "PENDING" or push_status == "PENDING"
    if pending:
        if not allow_pending_git:
            raise ValidationError(
                "final validation requires completed Git provenance; "
                "PENDING is allowed only with allow_pending_git=True"
            )
        return {
            "status": "PENDING",
            "commit_sha": "PENDING",
            "push_status": "PENDING",
        }
    if any_pending:
        raise ValidationError(
            "pre-delivery Git provenance must be exactly PENDING/PENDING/null"
        )
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValidationError("final run_summary commit_sha must be a full 40-hex SHA")
    if push_status == "PUSHED":
        if push_error is not None:
            raise ValidationError("PUSHED requires push_error=null")
    elif push_status == "GITHUB_PUSH_FAILED":
        if _real_push_error(push_error) is None:
            raise ValidationError(
                "GITHUB_PUSH_FAILED requires a nonempty real push_error"
            )
    else:
        raise ValidationError(
            "final push_status must be PUSHED or GITHUB_PUSH_FAILED"
        )

    repository = _git_repository(root)
    if not _git_ok(repository, "cat-file", "-e", f"{commit}^{{commit}}"):
        raise ValidationError("reported experiment commit does not exist locally")
    subject = subprocess.check_output(
        ["git", "show", "-s", "--format=%s", commit],
        cwd=repository,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()
    if subject != "Add mask-free region-aware representation audit":
        raise ValidationError("experiment commit subject differs from the required message")
    if not _git_ok(repository, "merge-base", "--is-ancestor", commit, "HEAD"):
        raise ValidationError("experiment commit is not an ancestor of current HEAD")

    parent_row = subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", commit],
        cwd=repository,
        text=True,
        stderr=subprocess.PIPE,
    ).strip().split()
    if len(parent_row) != 2:
        raise ValidationError("experiment commit must have exactly one parent")
    changed_paths = subprocess.check_output(
        ["git", "diff", "--no-renames", "--name-only", parent_row[1], commit, "--"],
        cwd=repository,
        text=True,
        stderr=subprocess.PIPE,
    ).splitlines()
    audit_prefix = root.resolve().relative_to(repository).as_posix() + "/"
    if not changed_paths:
        raise ValidationError("experiment commit has an empty diff")
    outside_audit = sorted(
        path for path in changed_paths if not path.startswith(audit_prefix)
    )
    if outside_audit:
        raise ValidationError(
            f"experiment commit changes paths outside the audit: {outside_audit[:5]}"
        )

    remote_tip: str | None = None
    if push_status == "PUSHED":
        branch = str(summary.get("branch", ""))
        if not _git_ok(repository, "check-ref-format", "--branch", branch):
            raise ValidationError("run_summary branch is not a valid Git branch name")
        reference = f"refs/heads/{branch}"
        try:
            output = subprocess.check_output(
                ["git", "ls-remote", "--heads", "origin", reference],
                cwd=repository,
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise ValidationError("git ls-remote failed for origin experiment branch") from error
        rows = [line.split() for line in output.splitlines() if line.strip()]
        if (
            len(rows) != 1
            or len(rows[0]) != 2
            or rows[0][1] != reference
            or re.fullmatch(r"[0-9a-f]{40}", rows[0][0]) is None
        ):
            raise ValidationError("origin does not expose the configured experiment branch")
        remote_tip = rows[0][0]
        if not _git_ok(repository, "cat-file", "-e", f"{remote_tip}^{{commit}}"):
            fetched = subprocess.run(
                [
                    "git",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "--no-write-fetch-head",
                    "origin",
                    reference,
                ],
                cwd=repository,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if fetched.returncode != 0 or not _git_ok(
                repository, "cat-file", "-e", f"{remote_tip}^{{commit}}"
            ):
                raise ValidationError("origin branch tip could not be authenticated locally")
        if not _git_ok(
            repository, "merge-base", "--is-ancestor", commit, remote_tip
        ):
            raise ValidationError(
                "reported experiment commit is not contained in origin branch"
            )

    result = {
        "status": "PASS",
        "commit_sha": commit,
        "push_status": str(push_status),
        "commit_subject": subject,
        "audit_only_changed_path_count": len(changed_paths),
    }
    if remote_tip is not None:
        result["remote_branch_tip"] = remote_tip
    return result


def tracked_delivery_paths(root: Path) -> tuple[Path, list[Path]]:
    repository = _git_repository(root)
    relative_root = root.resolve().relative_to(repository).as_posix()
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        relative_root,
    ]
    names = subprocess.check_output(
        command, cwd=repository, text=True, stderr=subprocess.STDOUT
    ).splitlines()
    return repository, [repository / name for name in sorted(set(filter(None, names)))]


def validate_tracked_artifacts(root: Path) -> dict[str, Any]:
    repository, paths = tracked_delivery_paths(root)
    raw_or_checkpoint: list[str] = []
    private_patient_files: list[str] = []
    oversized: list[str] = []
    symlinks: list[str] = []
    for path in paths:
        relative = path.relative_to(root)
        if path.is_symlink():
            symlinks.append(relative.as_posix())
            continue
        if not path.is_file():
            continue
        suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix.lower()
        components = {part.lower() for part in relative.parts}
        if suffix in FORBIDDEN_TRACKED_SUFFIXES or "checkpoints" in components:
            raw_or_checkpoint.append(relative.as_posix())
        if relative.parts and relative.parts[0].lower() in PRIVATE_PATIENT_DIRECTORIES and path.name != ".gitkeep":
            private_patient_files.append(relative.as_posix())
        if path.stat().st_size > MAX_TRACKED_BYTES:
            oversized.append(relative.as_posix())
    if raw_or_checkpoint or private_patient_files or oversized or symlinks:
        raise ValidationError(
            "tracked-artifact safety failed; "
            f"raw_or_checkpoint={raw_or_checkpoint}, private={private_patient_files}, "
            f"oversized={oversized}, symlinks={symlinks}"
        )
    return {
        "status": "PASS",
        "repository": repository.name,
        "tracked_or_candidate_files_scanned": len(paths),
        "raw_mri_or_checkpoint_files": 0,
        "private_patient_level_files": 0,
        "maximum_allowed_bytes": MAX_TRACKED_BYTES,
    }


def directory_sha256(path: Path) -> str:
    """Hash a directory as sorted relative-name/file-hash pairs."""

    if not path.is_dir():
        raise ValidationError(f"old-tree entry is not a directory: {path}")
    rows: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValidationError(f"old-tree snapshot contains a symlink: {child}")
        if child.is_file():
            rows.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "sha256": file_sha256(child),
                    "size_bytes": child.stat().st_size,
                }
            )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("files", "entries", "artifacts", "protected_entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            return [
                {"path": path, "sha256": digest}
                for path, digest in value.items()
            ]
    hashes = payload.get("hashes")
    if isinstance(hashes, Mapping):
        return [{"path": path, "sha256": digest} for path, digest in hashes.items()]
    return []


def verify_old_tree_manifest(
    root: Path,
    manifest_path: Path | None,
) -> dict[str, Any]:
    """Verify optional pre-run evidence for untouched existing directories.

    Supported evidence is either (a) file/directory entries with ``path`` and
    ``sha256``/``tree_sha256`` or (b) ``baseline_commit`` plus a
    ``protected_paths`` list.  Paths must be inside the repository and outside
    this new experiment directory.
    """

    if manifest_path is None:
        candidates = (
            root / "manifests" / "old_tree_immutability.private.json",
            root / "manifests" / "old_directory_hashes.private.json",
        )
        manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        return {"status": "NOT_SUPPLIED", "entries_verified": 0}
    manifest_path = manifest_path.resolve()
    payload = _read_json(manifest_path)
    repository = _git_repository(root)

    protected = payload.get("protected_paths")
    baseline = payload.get("baseline_commit", payload.get("parent_commit_sha"))
    if isinstance(protected, list) and protected and isinstance(baseline, str):
        safe_paths: list[str] = []
        for value in protected:
            candidate = (repository / str(value)).resolve()
            try:
                relative_repo = candidate.relative_to(repository)
            except ValueError as error:
                raise ValidationError("protected old path escaped repository") from error
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                pass
            else:
                raise ValidationError("old-tree evidence may not target the new experiment")
            safe_paths.append(relative_repo.as_posix())
        subprocess.run(
            ["git", "cat-file", "-e", f"{baseline}^{{commit}}"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        changed = subprocess.run(
            ["git", "diff", "--quiet", str(baseline), "--", *safe_paths],
            cwd=repository,
            check=False,
        ).returncode
        if changed != 0:
            raise ValidationError("a protected old directory differs from baseline_commit")
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *safe_paths],
            cwd=repository,
            text=True,
        ).splitlines()
        if untracked:
            raise ValidationError(f"protected old directories contain new files: {untracked[:5]}")
        return {
            "status": "PASS",
            "mode": "git_baseline",
            "entries_verified": len(safe_paths),
            "baseline_commit": baseline,
            "manifest_sha256": file_sha256(manifest_path),
        }

    entries = _manifest_entries(payload)
    if not entries:
        raise ValidationError("old-tree manifest has no supported entries")
    verified = 0
    for entry in entries:
        raw_path = entry.get("path", entry.get("relative_path"))
        if not isinstance(raw_path, str) or not raw_path:
            raise ValidationError("old-tree manifest entry lacks path")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = repository / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(repository)
        except ValueError as error:
            raise ValidationError("old-tree manifest path escaped repository") from error
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            pass
        else:
            raise ValidationError("old-tree manifest may not cover the new experiment")
        expected = entry.get("sha256", entry.get("tree_sha256", entry.get("digest")))
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValidationError(f"old-tree manifest has invalid SHA-256 for {raw_path}")
        if candidate.is_file():
            observed = file_sha256(candidate)
        elif candidate.is_dir():
            observed = directory_sha256(candidate)
        else:
            raise ValidationError(f"old-tree manifest path is missing: {raw_path}")
        if observed.lower() != expected.lower():
            raise ValidationError(f"old-tree non-modification hash failed: {raw_path}")
        verified += 1
    return {
        "status": "PASS",
        "mode": "sha256_entries",
        "entries_verified": verified,
        "manifest_sha256": file_sha256(manifest_path),
    }


def validate_all(
    root: Path = ROOT,
    *,
    old_tree_manifest: Path | None = None,
    allow_pending_git: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    checks = {
        "required_outputs": validate_required_outputs(root),
        "feature_completion": validate_feature_completion(root),
        "public_tables": validate_public_tables(root),
        "gates_and_run_summary": validate_gates_and_summary(
            root,
            allow_pending_git=allow_pending_git,
        ),
        "figure_manifest": validate_figure_manifest(root),
        "final_report": validate_report(root),
        "public_privacy": scan_public_artifacts(root),
        "tracked_artifact_safety": validate_tracked_artifacts(root),
        "old_tree_non_modification": verify_old_tree_manifest(root, old_tree_manifest),
    }
    return {
        "schema_version": 1,
        "experiment": "mask_free_region_aware_audit",
        "status": "PASS",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "public_outputs_contain_patient_level_data": False,
        "raw_mri_or_checkpoints_tracked": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    os.umask(0o022)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        output.relative_to(root)
    except ValueError as error:
        raise SystemExit("--output must remain inside --root") from error
    if output.exists() and not args.overwrite:
        raise SystemExit("validation output exists; pass --overwrite to replace it")
    try:
        result = validate_all(
            root,
            old_tree_manifest=args.old_tree_manifest,
            allow_pending_git=args.allow_pending_git,
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "experiment": "mask_free_region_aware_audit",
            "status": "FAIL",
            "validated_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(failure, output)
        raise SystemExit(f"validation failed: {error}") from error
    _atomic_json(result, output)
    print("final validation PASS")


if __name__ == "__main__":
    main()
