"""Shared fail-closed utilities for the spatial phenotype audit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "audit.json"
AMENDMENT_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_AMENDMENT.json"
LOCK_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
AMENDMENT_STATUS = "AMENDED_BEFORE_LABEL_DEPENDENT_ANALYSIS"
AMENDED_LOCK_STATUS = (
    "AMENDED_AND_REFROZEN_BEFORE_REEXTRACTION_OR_LABEL_DEPENDENT_PROBING"
)
AMENDMENT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "amendment_number",
        "original_preregistration_commit",
        "original_preregistration_lock_sha256",
        "reason_code",
        "geometry_qc",
        "pre_amendment_execution",
        "discarded_artifact_sha256",
        "contains_patient_identifiers",
    }
)
AMENDMENT_GEOMETRY_QC = {
    "source_authorized_by_visit": [808, 375, 375, 375],
    "post_local_core_valid_by_visit": [808, 375, 374, 374],
    "all_four_source_authorized_patient_count": 375,
    "all_four_upstream_core_parity_patient_count": 375,
    "all_four_post_local_core_valid_patient_count": 373,
    "post_local_empty_authorized_core_visit_count": 2,
    "affected_patient_count": 2,
    "representative_candidate_count_before": 375,
    "representative_candidate_count_after": 373,
    "amendment_scope": "deidentified_figure8_representative_selection_only",
    "probe_population_or_gate_contract_changed": False,
}
AMENDMENT_PRE_EXECUTION = {
    "cache_integrity_completed": True,
    "oracle_sidecar_completed": True,
    "completed_feature_cells": [
        "seed_2026/LOCAL0/fold_0",
        "seed_2026/LOCAL0/fold_1",
        "seed_2026/LOCAL0/fold_2",
    ],
    "interrupted_feature_cells": [
        "seed_2026/LOCAL0/fold_3",
        "seed_2026/LOCAL0/fold_4",
    ],
    "failed_feature_cell": "seed_2026/LOCAL3/fold_0",
    "representative_asset_created": False,
    "clinical_label_table_parsed": False,
    "stage_a_probe_fit": False,
    "stage_a_result_artifacts_created": False,
    "stage_b_started": False,
    "discard_before_refreeze_required": True,
    "reuse_forbidden": True,
}
AMENDMENT_DISCARDED_PATHS = frozenset(
    {
        "features/seed_2026/LOCAL0/fold_0/spatial_statistics.private.metadata.json",
        "features/seed_2026/LOCAL0/fold_0/spatial_statistics.private.npz",
        "features/seed_2026/LOCAL0/fold_1/spatial_statistics.private.metadata.json",
        "features/seed_2026/LOCAL0/fold_1/spatial_statistics.private.npz",
        "features/seed_2026/LOCAL0/fold_2/spatial_statistics.private.metadata.json",
        "features/seed_2026/LOCAL0/fold_2/spatial_statistics.private.npz",
        "logs/export_seed2026_LOCAL0_fold3.private.log",
        "logs/export_seed2026_LOCAL0_fold4.private.log",
        "logs/export_seed2026_LOCAL3_fold0.private.log",
        "manifests/cache_integrity.private.json",
        "manifests/oracle_regions.private.npz",
        "metrics/cache_integrity_contract.json",
        "metrics/oracle_region_contract.json",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def runtime_environment() -> dict[str, str]:
    """Return exact analysis-runtime versions recorded in the formal lock."""

    packages = (
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "Pillow",
    )
    versions = {name: importlib.metadata.version(name) for name in packages}
    import torch

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "torch_cuda_build": str(torch.version.cuda or "NONE"),
        "torch_cudnn_version": str(torch.backends.cudnn.version() or "NONE"),
        **versions,
    }


def file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"not a regular file: {source}")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
        if getattr(before, field) != getattr(after, field):
            raise RuntimeError(f"file changed while hashing: {source}")
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_sha256(values: Any) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def load_preregistration_amendment(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the exact public, patient-free revision-1 amendment record."""

    source = Path(AMENDMENT_PATH if path is None else path).resolve(strict=True)
    try:
        amendment = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("preregistration amendment is unreadable") from error
    if not isinstance(amendment, dict) or set(amendment) != set(AMENDMENT_KEYS):
        raise ValueError("preregistration amendment schema drifted")
    expected_scalars = {
        "schema_version": 1,
        "status": AMENDMENT_STATUS,
        "amendment_number": 1,
        "reason_code": "REPRESENTATIVE_POST_LOCAL_CORE_VALIDITY_COUNT_CORRECTION",
        "contains_patient_identifiers": False,
    }
    if any(amendment.get(name) != value for name, value in expected_scalars.items()):
        raise ValueError("preregistration amendment scalar contract drifted")
    if (
        type(amendment["schema_version"]) is not int
        or type(amendment["amendment_number"]) is not int
    ):
        raise ValueError("preregistration amendment revision fields must be integers")
    original_commit = amendment.get("original_preregistration_commit")
    original_lock = amendment.get("original_preregistration_lock_sha256")
    if (
        not isinstance(original_commit, str)
        or COMMIT_PATTERN.fullmatch(original_commit) is None
    ):
        raise ValueError("amendment original commit must be a full lowercase SHA")
    if (
        not isinstance(original_lock, str)
        or SHA256_PATTERN.fullmatch(original_lock) is None
    ):
        raise ValueError("amendment original lock must be a lowercase SHA-256")
    if amendment.get("geometry_qc") != AMENDMENT_GEOMETRY_QC:
        raise ValueError("preregistration amendment geometry QC drifted")
    if amendment.get("pre_amendment_execution") != AMENDMENT_PRE_EXECUTION:
        raise ValueError("preregistration amendment execution ledger drifted")
    discarded = amendment.get("discarded_artifact_sha256")
    if not isinstance(discarded, dict) or set(discarded) != set(
        AMENDMENT_DISCARDED_PATHS
    ):
        raise ValueError("preregistration amendment discard ledger drifted")
    for relative, digest in discarded.items():
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("preregistration amendment discard record is invalid")
    return amendment


def _experiment_relative_path(filename: str) -> str:
    return (EXPERIMENT_ROOT.relative_to(REPO_ROOT) / filename).as_posix()


def historical_file_sha256(commit: str, filename: str) -> str:
    """Hash an experiment file exactly as committed at a historical anchor."""

    if COMMIT_PATTERN.fullmatch(str(commit)) is None:
        raise ValueError("historical Git anchor must be a full lowercase SHA")
    relative_name = Path(filename)
    if relative_name.is_absolute() or ".." in relative_name.parts:
        raise ValueError("historical experiment path is unsafe")
    try:
        payload = subprocess.check_output(
            [
                "git",
                "show",
                f"{commit}:{_experiment_relative_path(relative_name.as_posix())}",
            ],
            cwd=REPO_ROOT,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"historical Git artifact is unavailable: {filename}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def authenticate_original_preregistration(
    amendment: Mapping[str, Any],
) -> tuple[str, str]:
    """Authenticate the immutable original lock named by the amendment."""

    original_commit = str(amendment["original_preregistration_commit"])
    original_lock = str(amendment["original_preregistration_lock_sha256"])
    if (
        historical_file_sha256(original_commit, "PREREGISTRATION_LOCK.json")
        != original_lock
    ):
        raise ValueError("amendment original lock differs from historical Git bytes")
    return original_commit, original_lock


def preregistration_anchor_commits(
    amendment: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return and authenticate ``(original, active-amended)`` Git anchors."""

    value = load_preregistration_amendment() if amendment is None else dict(amendment)
    original_commit, _original_lock = authenticate_original_preregistration(value)
    lock_relative = _experiment_relative_path("PREREGISTRATION_LOCK.json")
    active_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", lock_relative],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if COMMIT_PATTERN.fullmatch(active_commit) is None:
        raise ValueError("amended preregistration lock has no committed Git anchor")
    if active_commit == original_commit:
        raise ValueError("active preregistration anchor is still the superseded commit")
    for ancestor, descendant, message in (
        (
            original_commit,
            active_commit,
            "original preregistration is not a strict ancestor of the amendment",
        ),
        (
            active_commit,
            "HEAD",
            "active amended preregistration is not an ancestor of HEAD",
        ),
    ):
        if (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor, descendant],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            raise ValueError(message)
    if historical_file_sha256(
        active_commit, "PREREGISTRATION_LOCK.json"
    ) != file_sha256(LOCK_PATH):
        raise ValueError("committed amended lock differs from the current lock")
    if historical_file_sha256(
        active_commit, "PREREGISTRATION_AMENDMENT.json"
    ) != file_sha256(AMENDMENT_PATH):
        raise ValueError("committed amendment differs from the current amendment")
    return original_commit, active_commit


def preregistration_chain(
    lock: Mapping[str, Any],
    amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact authenticated amendment chain for runtime artifacts."""

    value = load_preregistration_amendment() if amendment is None else dict(amendment)
    original_commit, active_commit = preregistration_anchor_commits(value)
    original_lock = str(value["original_preregistration_lock_sha256"])
    amendment_sha256 = file_sha256(AMENDMENT_PATH)
    if (
        lock.get("schema_version") != 3
        or lock.get("preregistration_revision") != 2
        or lock.get("status") != AMENDED_LOCK_STATUS
        or lock.get("amendment_sha256") != amendment_sha256
        or lock.get("superseded_preregistration_commit") != original_commit
        or lock.get("superseded_preregistration_lock_sha256") != original_lock
    ):
        raise ValueError("runtime preregistration amendment chain drifted")
    return {
        "preregistration_revision": 2,
        "active_preregistration_lock_sha256": file_sha256(LOCK_PATH),
        "preregistration_amendment_sha256": amendment_sha256,
        "original_preregistration_lock_sha256": original_lock,
        "original_preregistration_commit": original_commit,
        "active_preregistration_commit": active_commit,
    }


def load_config(
    path: str | Path = DEFAULT_CONFIG, *, verify_inputs: bool = True
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit config must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("audit config schema_version must be 1")
    if payload.get("experiment") != "spatial_heterogeneity_phenotype_audit":
        raise ValueError("audit config names a different experiment")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("audit config paths must be an object")
    hash_pairs = (
        ("local_preregistration_lock", "local_preregistration_lock_sha256"),
        ("spatial_sidecar", "spatial_sidecar_sha256"),
        ("stage_b_data_contract", "stage_b_data_contract_sha256"),
        (
            "stage_b_upstream_authorization",
            "stage_b_upstream_authorization_sha256",
        ),
        ("c1b_cache_manifest", "c1b_cache_manifest_sha256"),
        ("support_inventory", "support_inventory_sha256"),
        ("clinical_labels", "clinical_labels_sha256"),
        ("fold_manifest", "fold_manifest_sha256"),
        ("ftv_table", "ftv_table_sha256"),
    )
    resolved = dict(paths)
    for path_key, hash_key in hash_pairs:
        source_path = Path(str(paths[path_key])).expanduser().resolve()
        resolved[path_key] = source_path
        expected = str(paths[hash_key])
        if verify_inputs and file_sha256(source_path) != expected:
            raise ValueError(f"hash mismatch for {path_key}: {source_path}")
    for directory_key in ("source_repo", "checkpoint_root", "local_feature_root"):
        directory = Path(str(paths[directory_key])).expanduser().resolve()
        if verify_inputs and not directory.is_dir():
            raise FileNotFoundError(directory)
        resolved[directory_key] = directory
    output = dict(payload)
    output["paths"] = resolved
    return output


def require_preregistration_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = LOCK_PATH
    if not lock_path.is_file():
        raise FileNotFoundError("formal execution requires PREREGISTRATION_LOCK.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 3:
        raise ValueError("preregistration lock schema version is invalid")
    if lock.get("preregistration_revision") != 2:
        raise ValueError("preregistration lock revision is invalid")
    if lock.get("status") != AMENDED_LOCK_STATUS:
        raise ValueError("preregistration lock status is invalid")
    if lock.get("branch") != config.get("branch"):
        raise ValueError("preregistration lock names another branch")
    if lock.get("analysis_outputs_present_before_freeze") is not False:
        raise ValueError("preregistration was not frozen before analysis outputs")
    checks = {
        "plan_sha256": EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md",
        "config_sha256": EXPERIMENT_ROOT / "configs" / "audit.json",
        "privacy_policy_sha256": EXPERIMENT_ROOT / ".gitignore",
        "amendment_sha256": AMENDMENT_PATH,
    }
    for field, path in checks.items():
        if lock.get(field) != file_sha256(path):
            raise ValueError(f"preregistered {field} drifted")
    if lock.get("config_canonical_sha256") != canonical_sha256(config):
        raise ValueError("loaded config differs from the frozen canonical config")
    amendment = load_preregistration_amendment()
    original_commit, original_lock = authenticate_original_preregistration(amendment)
    if (
        lock.get("superseded_preregistration_commit") != original_commit
        or lock.get("superseded_preregistration_lock_sha256") != original_lock
    ):
        raise ValueError("amended lock does not mirror its superseded Git anchor")
    implementations = lock.get("implementation_sha256")
    if not isinstance(implementations, dict) or not implementations:
        raise ValueError("preregistration lock has no implementation inventory")
    for relative, expected in implementations.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid preregistered implementation path: {relative}")
        path = EXPERIMENT_ROOT / relative_path
        if file_sha256(path) != expected:
            raise ValueError(f"preregistered implementation drifted: {relative}")
    upstream = lock.get("upstream_code_sha256")
    if not isinstance(upstream, dict) or not upstream:
        raise ValueError("preregistration lock has no upstream-code inventory")
    for source, expected in upstream.items():
        if file_sha256(Path(source)) != expected:
            raise ValueError(f"preregistered upstream code drifted: {source}")
    if lock.get("runtime_environment") != runtime_environment():
        raise ValueError("formal runtime environment differs from preregistration")
    chain = preregistration_chain(lock, amendment)
    if chain["original_preregistration_commit"] != original_commit:
        raise ValueError("amended preregistration anchor chain drifted")
    return lock


def private_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    return output


def atomic_json(
    payload: Mapping[str, Any], path: str | Path, *, private: bool = False
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(0o700 if private else 0o755)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600 if private else 0o644)
        temporary.replace(destination)
        destination.chmod(0o600 if private else 0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_csv(frame: pd.DataFrame, path: str | Path, *, private: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(0o700 if private else 0o755)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600 if private else 0o644)
        temporary.replace(destination)
        destination.chmod(0o600 if private else 0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "AMENDED_LOCK_STATUS",
    "AMENDMENT_PATH",
    "AMENDMENT_STATUS",
    "DEFAULT_CONFIG",
    "EXPERIMENT_ROOT",
    "LOCK_PATH",
    "REPO_ROOT",
    "atomic_csv",
    "atomic_json",
    "authenticate_original_preregistration",
    "canonical_sha256",
    "file_sha256",
    "historical_file_sha256",
    "load_config",
    "load_preregistration_amendment",
    "ordered_sha256",
    "preregistration_anchor_commits",
    "preregistration_chain",
    "private_directory",
    "require_preregistration_lock",
    "runtime_environment",
]
