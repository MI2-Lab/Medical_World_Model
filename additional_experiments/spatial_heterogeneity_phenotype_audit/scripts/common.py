"""Shared fail-closed utilities for the spatial phenotype audit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "audit.json"


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
    lock_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
    if not lock_path.is_file():
        raise FileNotFoundError("formal execution requires PREREGISTRATION_LOCK.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 2:
        raise ValueError("preregistration lock schema version is invalid")
    if lock.get("status") != "FROZEN_BEFORE_FEATURE_EXTRACTION_OR_PROBING":
        raise ValueError("preregistration lock status is invalid")
    if lock.get("branch") != config.get("branch"):
        raise ValueError("preregistration lock names another branch")
    if lock.get("analysis_outputs_present_before_freeze") is not False:
        raise ValueError("preregistration was not frozen before analysis outputs")
    checks = {
        "plan_sha256": EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md",
        "config_sha256": EXPERIMENT_ROOT / "configs" / "audit.json",
        "privacy_policy_sha256": EXPERIMENT_ROOT / ".gitignore",
    }
    for field, path in checks.items():
        if lock.get(field) != file_sha256(path):
            raise ValueError(f"preregistered {field} drifted")
    if lock.get("config_canonical_sha256") != canonical_sha256(config):
        raise ValueError("loaded config differs from the frozen canonical config")
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
    "DEFAULT_CONFIG",
    "EXPERIMENT_ROOT",
    "REPO_ROOT",
    "atomic_csv",
    "atomic_json",
    "canonical_sha256",
    "file_sha256",
    "load_config",
    "ordered_sha256",
    "private_directory",
    "require_preregistration_lock",
    "runtime_environment",
]
