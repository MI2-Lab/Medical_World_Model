"""Fail-closed scientific and implementation preregistration verification."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import canonical_sha256, file_sha256


SHA256 = re.compile(r"[0-9a-f]{64}")


def implementation_files(experiment_root: str | Path) -> tuple[Path, ...]:
    """Return the exact executable/test source set governed by the lock."""

    root = Path(experiment_root).resolve()
    paths: list[Path] = []
    for directory in ("src", "scripts", "tests"):
        paths.extend(
            path
            for path in (root / directory).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    if not paths:
        raise ValueError("implementation source inventory is empty")
    return tuple(sorted({path.resolve() for path in paths}))


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_inventory(
    inventory: Mapping[str, Any], repo_root: Path, *, label: str
) -> dict[str, str]:
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError(f"{label} hash inventory must be nonempty")
    observed: dict[str, str] = {}
    for relative, expected_value in inventory.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{label} contains an unsafe path: {relative}")
        expected = str(expected_value).lower()
        if SHA256.fullmatch(expected) is None:
            raise ValueError(f"{label} contains an invalid SHA-256: {relative}")
        path = (repo_root / relative_path).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"{label} path escaped repository root") from error
        if not path.is_file():
            raise FileNotFoundError(f"{label} file is missing: {relative}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label} hash drift for {relative}: expected {expected}, observed {actual}"
            )
        observed[relative_path.as_posix()] = actual
    return observed


def verify_preregistration(
    experiment_root: str | Path,
    *,
    require_implementation: bool = True,
) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    repo_root = root.parents[1]
    lock_path = root / "PREREGISTRATION_LOCK.json"
    lock = _read_object(lock_path, "preregistration lock")
    if lock.get("experiment") != "residual_sph_grounding_pilot":
        raise ValueError("preregistration experiment identity drifted")
    if lock.get("status") != "SCIENTIFIC_PROTOCOL_FROZEN_BEFORE_FORMAL_RESULTS":
        raise ValueError("scientific preregistration is not frozen")
    scientific = _verify_inventory(
        lock.get("scientific_protocol_sha256", {}),
        repo_root,
        label="scientific protocol",
    )
    lock_sha256 = file_sha256(lock_path)
    result: dict[str, Any] = {
        "status": "SCIENTIFIC_PROTOCOL_VERIFIED",
        "lock_sha256": lock_sha256,
        "scientific_protocol_sha256": scientific,
    }
    if not require_implementation:
        return result

    manifest_relative = str(
        lock.get("implementation_hash_lock", {}).get("required_manifest", "")
    )
    if manifest_relative != "manifests/implementation_lock.json":
        raise ValueError("implementation-manifest contract drifted")
    manifest_path = root / manifest_relative
    manifest = _read_object(manifest_path, "implementation lock")
    expected = {
        "experiment": "residual_sph_grounding_pilot",
        "status": "IMPLEMENTATION_FROZEN_BEFORE_FORMAL_RESULTS",
        "scientific_preregistration_sha256": lock_sha256,
        "formal_cells_started_before_implementation_lock": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"implementation lock differs at {key}")
    implementation = _verify_inventory(
        manifest.get("implementation_files_sha256", {}),
        repo_root,
        label="implementation",
    )
    expected_implementation = {
        path.relative_to(repo_root).as_posix() for path in implementation_files(root)
    }
    if set(implementation) != expected_implementation:
        missing = sorted(expected_implementation.difference(implementation))
        extra = sorted(set(implementation).difference(expected_implementation))
        raise ValueError(
            f"implementation lock coverage drifted; missing={missing}, extra={extra}"
        )
    tests = manifest.get("test_attestation")
    if not isinstance(tests, Mapping) or tests.get("status") != "PASS":
        raise ValueError("implementation lock lacks a passing test attestation")
    if not isinstance(tests.get("commands"), list) or not tests["commands"]:
        raise ValueError("implementation test attestation lacks commands")
    if SHA256.fullmatch(str(tests.get("attestation_sha256", ""))) is None:
        raise ValueError("implementation test attestation digest is invalid")
    attestation_body = {
        str(key): value
        for key, value in tests.items()
        if str(key) != "attestation_sha256"
    }
    if canonical_sha256(attestation_body) != tests["attestation_sha256"]:
        raise ValueError("implementation test attestation digest mismatched")
    if int(tests.get("passed_test_count", 0)) <= 0:
        raise ValueError("implementation test attestation has no passing tests")
    result.update(
        {
            "status": "PASS",
            "implementation_lock_sha256": file_sha256(manifest_path),
            "implementation_files_sha256": implementation,
            "test_attestation": dict(tests),
        }
    )
    return result


def require_lock_sha256(observed: str, expected: str) -> None:
    if SHA256.fullmatch(str(expected)) is None or str(observed) != str(expected):
        raise PermissionError("provided preregistration SHA-256 does not match verified lock")


__all__ = ["implementation_files", "require_lock_sha256", "verify_preregistration"]
