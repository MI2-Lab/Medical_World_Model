"""Fail-closed I/O and preregistration provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: str | Path, payload: str, *, overwrite: bool) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def verify_preregistration(
    *,
    experiment_root: str | Path = EXPERIMENT_ROOT,
) -> dict[str, Any]:
    """Verify that the frozen plan predates every run entry point."""

    root = Path(experiment_root)
    lock_path = root / "configs/preregistration_lock.json"
    plan_path = root / "EXPERIMENT_PLAN.md"
    if not lock_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError("Frozen preregistration plan/lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported preregistration lock schema")
    if not lock.get("preregistered_before_new_cohort_statistics", False):
        raise ValueError("Preregistration timing assertion is not frozen")
    if not lock.get("stage_b_requires_stage_a_go", False):
        raise ValueError("Stage-B dependency is not frozen")
    actual = sha256_file(plan_path)
    if actual != lock.get("plan_sha256"):
        raise ValueError(
            "EXPERIMENT_PLAN.md changed after preregistration: "
            f"{actual} != {lock.get('plan_sha256')}"
        )
    return lock


def tracked_tree_digest(repo_root: str | Path, relative_root: str) -> tuple[int, str]:
    """Reproduce the independent audit's tracked-tree SHA-256 definition."""

    repo = Path(repo_root)
    listing = subprocess.check_output(
        ["git", "ls-files", "-z", "--", str(relative_root)], cwd=repo
    )
    paths = sorted(item for item in listing.split(b"\0") if item)
    digest = hashlib.sha256(b"tracked-prior-experiment-v1\0")
    for raw_path in paths:
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("A tracked prior path is not UTF-8") from exc
        source = repo / relative
        if not source.is_file():
            raise FileNotFoundError(f"Tracked upstream file is missing: {relative}")
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(sha256_file(source).encode("ascii"))
        digest.update(b"\0")
    return len(paths), digest.hexdigest()


def verify_upstream_contract(
    *,
    experiment_root: str | Path = EXPERIMENT_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Fail if either immutable old run or a reused contract has changed."""

    root = Path(experiment_root)
    repo = Path(repo_root)
    lock_path = root / "configs/upstream_contract_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Upstream contract lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported upstream contract lock schema")
    observed_files: dict[str, str] = {}
    for relative, expected in lock["file_sha256"].items():
        actual = sha256_file(repo / relative)
        if actual != expected:
            raise ValueError(f"Immutable upstream file changed: {relative}")
        observed_files[relative] = actual
    observed_trees: dict[str, dict[str, Any]] = {}
    for relative, expected in lock["tracked_trees"].items():
        count, digest = tracked_tree_digest(repo, relative)
        if count != int(expected["file_count"]) or digest != expected["sha256"]:
            raise ValueError(f"Immutable upstream tracked tree changed: {relative}")
        observed_trees[relative] = {"file_count": count, "sha256": digest}

    prior_src = (
        repo
        / "additional_experiments/c1b_model_ready_ftv_sanity/src"
    )
    if str(prior_src) not in sys.path:
        sys.path.insert(0, str(prior_src))
    from c1b_sanity.builder import builder_contract_sha256  # noqa: PLC0415

    semantic = builder_contract_sha256()
    if semantic != lock["builder_semantic_contract_sha256"]:
        raise ValueError("Inherited builder semantic contract changed")
    return {
        "status": "PASS",
        "builder_semantic_contract_sha256": semantic,
        "file_sha256": observed_files,
        "tracked_trees": observed_trees,
    }


def json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
