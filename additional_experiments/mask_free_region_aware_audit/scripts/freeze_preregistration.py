#!/usr/bin/env python3
"""Freeze the mask-free audit before extraction or label-dependent analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    CONFIG_PATH,
    GITIGNORE_PATH,
    IMPLEMENTATION_ERRATUM_2_PATH,
    IMPLEMENTATION_ERRATUM_PATH,
    LOCK_PATH,
    LOCK_STATUS,
    PLAN_PATH,
    PRIOR_COMPATIBILITY_REFREEZE_COMMIT,
    PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256,
    PRIOR_PREREGISTRATION_COMMIT,
    PRIOR_PREREGISTRATION_LOCK_SHA256,
    PRIVACY_CONTRACT,
    REFREEZE_2_LOCK_KEYS,
    REFREEZE_2_LOCK_STATUS,
    REFREEZE_LOCK_KEYS,
    REFREEZE_LOCK_STATUS,
    ERRATUM_2_DISCARDED_RECORD_SET_SHA256,
    canonical_sha256,
    cells,
    file_sha256,
    historical_file_bytes,
    implementation_inventory,
    load_config,
    load_goal5_lock,
    publish_json_once,
    require_erratum_plan_disclosure,
    require_erratum_2_plan_disclosure,
    require_implementation_erratum,
    require_implementation_erratum_2,
    require_prior_compatibility_refreeze,
    require_prior_preregistration,
)


RESULT_DIRECTORIES = (
    "features",
    "predictions",
    "metrics",
    "figures",
    "reports",
    "logs",
    "manifests",
)
ALLOWED_PREFREEZE_NONRESULTS = {
    "metrics/.gitkeep",
    "figures/.gitkeep",
    "reports/.gitkeep",
    "manifests/.gitkeep",
    "manifests/start_provenance.json",
    "features/.gitkeep",
    "predictions/.gitkeep",
    "logs/.gitkeep",
}


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT.parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    # Preserve the two leading porcelain-status columns on the first line.
    return result.stdout.rstrip()


def _require_git_context(config: dict[str, Any]) -> None:
    if _git("branch", "--show-current") != config["branch"]:
        raise ValueError("current branch differs from preregistered experiment branch")
    parent = str(config["start"]["parent_commit_sha"])
    if _git("rev-parse", "HEAD") != parent or _git("merge-base", "HEAD", parent) != parent:
        raise ValueError("freeze must occur at the exact preregistered parent commit")
    status = _git("status", "--porcelain", "--untracked-files=all")
    prefix = str(ROOT.relative_to(ROOT.parents[1])) + "/"
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if not path.startswith(prefix):
            raise ValueError(f"dirty path outside new experiment directory: {path}")


def _dirty_paths() -> tuple[list[str], list[str]]:
    tracked: list[str] = []
    untracked: list[str] = []
    status = _git("status", "--porcelain", "--untracked-files=all")
    for line in status.splitlines():
        code = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        (untracked if code == "??" else tracked).append(path)
    return sorted(set(tracked)), sorted(set(untracked))


def _require_refreeze_git_context(config: dict[str, Any]) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != config["branch"]:
        raise ValueError("current branch differs from the refreeze branch")
    if head != PRIOR_PREREGISTRATION_COMMIT:
        raise ValueError("refreeze must start at the exact prior preregistration commit")
    tracked, untracked = _dirty_paths()
    prefix = str(ROOT.relative_to(ROOT.parents[1])) + "/"
    if any(not path.startswith(prefix) for path in (*tracked, *untracked)):
        raise ValueError("dirty path outside the new experiment during refreeze")
    required_dirty = {
        prefix + "EXPERIMENT_PLAN.md",
        prefix + "scripts/common.py",
        prefix + "scripts/freeze_preregistration.py",
        prefix + "scripts/run_audit.py",
        prefix + "tests/test_extraction_contract.py",
        prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json",
    }
    if not required_dirty.issubset({*tracked, *untracked}):
        raise ValueError("required refreeze implementation paths are not dirty")
    forbidden = {prefix + "configs/audit.json", prefix + ".gitignore"}
    if forbidden.intersection({*tracked, *untracked}):
        raise ValueError("refreeze may not change config or privacy policy")
    return {
        "base_head": head,
        "branch": branch,
        "all_dirty_paths_confined_to_new_experiment": True,
        "tracked_paths_before_refreeze": tracked,
        "untracked_paths_before_refreeze": untracked,
    }


def _require_refreeze_2_git_context(config: dict[str, Any]) -> dict[str, Any]:
    """Require schema 3 to start exactly at the committed schema-2 anchor."""

    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != config["branch"]:
        raise ValueError("current branch differs from the second refreeze branch")
    if head != PRIOR_COMPATIBILITY_REFREEZE_COMMIT:
        raise ValueError(
            "schema-3 refreeze must start at the exact schema-2 commit"
        )
    tracked, untracked = _dirty_paths()
    prefix = str(ROOT.relative_to(ROOT.parents[1])) + "/"
    paths = {*tracked, *untracked}
    if any(not path.startswith(prefix) for path in paths):
        raise ValueError("dirty path outside the new experiment during schema-3 refreeze")
    required_dirty = {
        prefix + "EXPERIMENT_PLAN.md",
        prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json",
        prefix + "scripts/common.py",
        prefix + "scripts/freeze_preregistration.py",
        prefix + "scripts/generate_figures.py",
        prefix + "scripts/generate_report.py",
        prefix + "scripts/run_audit.py",
        prefix + "scripts/validate_results.py",
        prefix + "tests/test_analysis.py",
        prefix + "tests/test_extraction_contract.py",
        prefix + "tests/test_reporting.py",
    }
    if not required_dirty.issubset(paths):
        missing = sorted(required_dirty.difference(paths))
        raise ValueError(f"required schema-3 implementation paths are not dirty: {missing}")
    forbidden = {
        prefix + "configs/audit.json",
        prefix + ".gitignore",
        prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json",
    }
    if forbidden.intersection(paths):
        raise ValueError("schema-3 refreeze may not change config, privacy, or erratum 1")
    return {
        "base_head": head,
        "branch": branch,
        "all_dirty_paths_confined_to_new_experiment": True,
        "tracked_paths_before_refreeze": tracked,
        "untracked_paths_before_refreeze": untracked,
    }


def _pre_freeze_result_inventory() -> dict[str, int]:
    output: dict[str, int] = {}
    for directory in RESULT_DIRECTORIES:
        count = 0
        source = ROOT / directory
        if source.is_dir():
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(ROOT))
                if relative in ALLOWED_PREFREEZE_NONRESULTS:
                    continue
                count += 1
        output[f"{directory[:-1] if directory.endswith('s') else directory}_files"] = count
    # Keep semantically clear singular keys frozen in common.require_*.
    normalized = {
        "feature_files": output["feature_files"],
        "prediction_files": output["prediction_files"],
        "metric_files": output["metric_files"],
        "figure_files": output["figure_files"],
        "report_files": output["report_files"],
        "log_files": output["log_files"],
        "manifest_files": output["manifest_files"],
    }
    if any(normalized.values()):
        raise ValueError(f"result artifacts already exist before freeze: {normalized}")
    return normalized


def _require_allowed_nonresults_unchanged(commit: str) -> None:
    """Authenticate every placeholder/start record exempted from zero inventory."""

    for relative_text in sorted(ALLOWED_PREFREEZE_NONRESULTS):
        source = ROOT / relative_text
        if not source.is_file():
            raise FileNotFoundError(f"required frozen non-result is absent: {relative_text}")
        repository_relative = source.relative_to(ROOT.parents[1])
        historical = historical_file_bytes(commit, repository_relative)
        if source.read_bytes() != historical:
            raise ValueError(f"allowed non-result drifted from Git: {relative_text}")


def _selected_cells(config: dict[str, Any]) -> dict[str, Any]:
    goal5 = load_goal5_lock(config)
    selected_cells = goal5["selected_cells"]
    if set(selected_cells) != {
        f"seed_{seed}/{arm}/fold_{fold}" for seed, arm, fold in cells()
    }:
        raise ValueError("Goal5 selected-cell inventory drifted")
    for key, record in selected_cells.items():
        checkpoint = Path(str(record["checkpoint_path"])).resolve(strict=True)
        selection = Path(str(record["selection_path"])).resolve(strict=True)
        reference = Path(str(record["reference"]["path"])).resolve(strict=True)
        reference_metadata = Path(
            str(record["reference"]["metadata_path"])
        ).resolve(strict=True)
        checks = (
            (checkpoint, record["checkpoint_sha256"]),
            (selection, record["selection_sha256"]),
            (reference, record["reference"]["sha256"]),
            (reference_metadata, record["reference"]["metadata_sha256"]),
        )
        if any(file_sha256(path) != expected for path, expected in checks):
            raise ValueError(f"Goal5 selected-cell input drifted: {key}")
    return selected_cells


def build_lock(config: dict[str, Any]) -> dict[str, Any]:
    """Build the original schema-1 lock for a genuinely fresh experiment."""

    _require_git_context(config)
    if LOCK_PATH.exists():
        raise FileExistsError("preregistration lock already exists and is immutable")
    selected_cells = _selected_cells(config)
    inventory = _pre_freeze_result_inventory()
    return {
        "schema_version": 1,
        "status": LOCK_STATUS,
        "branch": config["branch"],
        "parent_commit_sha": config["start"]["parent_commit_sha"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_sha256": file_sha256(CONFIG_PATH),
        "config_canonical_sha256": canonical_sha256(config),
        "experiment_plan_sha256": file_sha256(PLAN_PATH),
        "gitignore_sha256": file_sha256(GITIGNORE_PATH),
        "implementation_sha256": implementation_inventory(),
        "goal5_preregistration_lock_sha256": config["paths"]["goal5_lock_sha256"],
        "goal5_feature_completion_sha256": config["paths"][
            "goal5_feature_completion_sha256"
        ],
        "formal_cell_count": 20,
        "selected_cells": selected_cells,
        "pre_freeze_result_inventory": inventory,
        "privacy_contract": dict(PRIVACY_CONTRACT),
    }


def _prior_lock_bytes_from_git() -> bytes:
    relative = LOCK_PATH.relative_to(ROOT.parents[1])
    payload = historical_file_bytes(PRIOR_PREREGISTRATION_COMMIT, relative)
    if hashlib.sha256(payload).hexdigest() != PRIOR_PREREGISTRATION_LOCK_SHA256:
        raise ValueError("historical prior lock SHA-256 drifted")
    return payload


def _require_working_prior_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("schema-1 lock is absent before refreeze")
    if file_sha256(LOCK_PATH) != PRIOR_PREREGISTRATION_LOCK_SHA256:
        raise ValueError("working lock is not the exact schema-1 lock to supersede")
    historical = _prior_lock_bytes_from_git()
    working = LOCK_PATH.read_bytes()
    if working != historical:
        raise ValueError("working schema-1 lock differs from Git history")
    prior = require_prior_preregistration()
    if json.loads(working.decode("utf-8")) != prior:
        raise ValueError("working schema-1 lock JSON differs from Git history")
    return prior


def _prior_compatibility_lock_bytes_from_git() -> bytes:
    relative = LOCK_PATH.relative_to(ROOT.parents[1])
    payload = historical_file_bytes(PRIOR_COMPATIBILITY_REFREEZE_COMMIT, relative)
    if (
        hashlib.sha256(payload).hexdigest()
        != PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256
    ):
        raise ValueError("historical schema-2 lock SHA-256 drifted")
    return payload


def _require_working_prior_compatibility_lock() -> dict[str, Any]:
    """Authenticate both Git and working copies of the schema-2 lock."""

    if not LOCK_PATH.is_file():
        raise FileNotFoundError("schema-2 lock is absent before schema-3 refreeze")
    if file_sha256(LOCK_PATH) != PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256:
        raise ValueError("working lock is not the exact schema-2 lock to supersede")
    historical = _prior_compatibility_lock_bytes_from_git()
    working = LOCK_PATH.read_bytes()
    if working != historical:
        raise ValueError("working schema-2 lock differs from Git history")
    prior = require_prior_compatibility_refreeze()
    if json.loads(working.decode("utf-8")) != prior:
        raise ValueError("working schema-2 lock JSON differs from Git history")
    return prior


def build_refreeze_lock(config: dict[str, Any]) -> dict[str, Any]:
    """Build schema 2 only after authenticating and emptying the failed run."""

    git_provenance = _require_refreeze_git_context(config)
    prior = _require_working_prior_lock()
    require_implementation_erratum()
    require_erratum_plan_disclosure(prior)
    if (
        file_sha256(CONFIG_PATH) != prior["config_sha256"]
        or canonical_sha256(config) != prior["config_canonical_sha256"]
        or file_sha256(GITIGNORE_PATH) != prior["gitignore_sha256"]
    ):
        raise ValueError("refreeze changed config or privacy bytes")
    selected_cells = _selected_cells(config)
    if (
        selected_cells != prior["selected_cells"]
        or config["paths"]["goal5_lock_sha256"]
        != prior["goal5_preregistration_lock_sha256"]
        or config["paths"]["goal5_feature_completion_sha256"]
        != prior["goal5_feature_completion_sha256"]
        or prior["privacy_contract"] != PRIVACY_CONTRACT
    ):
        raise ValueError("refreeze changed Goal5 cells, inputs, or privacy contract")
    inventory = _pre_freeze_result_inventory()
    payload = {
        "schema_version": 2,
        "preregistration_revision": 1,
        "status": REFREEZE_LOCK_STATUS,
        "branch": prior["branch"],
        "parent_commit_sha": prior["parent_commit_sha"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_sha256": prior["config_sha256"],
        "config_canonical_sha256": prior["config_canonical_sha256"],
        "experiment_plan_sha256": file_sha256(PLAN_PATH),
        "gitignore_sha256": prior["gitignore_sha256"],
        "implementation_sha256": implementation_inventory(),
        "goal5_preregistration_lock_sha256": prior[
            "goal5_preregistration_lock_sha256"
        ],
        "goal5_feature_completion_sha256": prior[
            "goal5_feature_completion_sha256"
        ],
        "formal_cell_count": prior["formal_cell_count"],
        "selected_cells": selected_cells,
        "pre_freeze_result_inventory": prior["pre_freeze_result_inventory"],
        "privacy_contract": dict(PRIVACY_CONTRACT),
        "prior_preregistration_commit": PRIOR_PREREGISTRATION_COMMIT,
        "prior_preregistration_lock_sha256": PRIOR_PREREGISTRATION_LOCK_SHA256,
        "implementation_erratum_sha256": file_sha256(IMPLEMENTATION_ERRATUM_PATH),
        "superseded_artifacts_reused": False,
        "scientific_contract_unchanged": True,
        "pre_refreeze_result_inventory": inventory,
        "git_provenance_before_refreeze": git_provenance,
    }
    if set(payload) != set(REFREEZE_LOCK_KEYS):
        raise AssertionError("internal schema-2 lock key drift")
    return payload


def build_refreeze_2_lock(config: dict[str, Any]) -> dict[str, Any]:
    """Build schema 3 after authenticating and emptying the completed run."""

    git_provenance = _require_refreeze_2_git_context(config)
    prior = _require_working_prior_compatibility_lock()
    require_implementation_erratum()
    require_implementation_erratum_2()
    require_erratum_2_plan_disclosure(prior)
    if (
        file_sha256(CONFIG_PATH) != prior["config_sha256"]
        or canonical_sha256(config) != prior["config_canonical_sha256"]
        or file_sha256(GITIGNORE_PATH) != prior["gitignore_sha256"]
        or file_sha256(IMPLEMENTATION_ERRATUM_PATH)
        != prior["implementation_erratum_sha256"]
    ):
        raise ValueError("schema-3 refreeze changed config, privacy, or erratum 1")
    selected_cells = _selected_cells(config)
    if (
        selected_cells != prior["selected_cells"]
        or config["paths"]["goal5_lock_sha256"]
        != prior["goal5_preregistration_lock_sha256"]
        or config["paths"]["goal5_feature_completion_sha256"]
        != prior["goal5_feature_completion_sha256"]
        or prior["privacy_contract"] != PRIVACY_CONTRACT
        or prior["superseded_artifacts_reused"] is not False
        or prior["scientific_contract_unchanged"] is not True
    ):
        raise ValueError(
            "schema-3 refreeze changed Goal5 cells, inputs, privacy, or science"
        )
    inventory = _pre_freeze_result_inventory()
    _require_allowed_nonresults_unchanged(PRIOR_COMPATIBILITY_REFREEZE_COMMIT)
    payload = {
        **prior,
        "schema_version": 3,
        "preregistration_revision": 2,
        "status": REFREEZE_2_LOCK_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_plan_sha256": file_sha256(PLAN_PATH),
        "implementation_sha256": implementation_inventory(),
        "prior_compatibility_refreeze_commit": (
            PRIOR_COMPATIBILITY_REFREEZE_COMMIT
        ),
        "prior_compatibility_refreeze_lock_sha256": (
            PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256
        ),
        "implementation_erratum_2_sha256": file_sha256(
            IMPLEMENTATION_ERRATUM_2_PATH
        ),
        "superseded_formal_run_artifact_count": 94,
        "superseded_formal_run_artifact_record_set_sha256": (
            ERRATUM_2_DISCARDED_RECORD_SET_SHA256
        ),
        "all_twenty_feature_cells_rebuild_required": True,
        "superseded_artifacts_reused": False,
        "scientific_contract_unchanged": True,
        "pre_refreeze_2_result_inventory": inventory,
        "git_provenance_before_refreeze_2": git_provenance,
    }
    if set(payload) != set(REFREEZE_2_LOCK_KEYS):
        raise AssertionError("internal schema-3 lock key drift")
    return payload


def _replace_prior_lock(payload: dict[str, Any]) -> None:
    """Atomically replace only the authenticated historical schema-1 bytes."""

    if file_sha256(LOCK_PATH) != PRIOR_PREREGISTRATION_LOCK_SHA256:
        raise ValueError("refusing to replace a non-historical lock")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{LOCK_PATH.name}.", suffix=".tmp", dir=LOCK_PATH.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        if (
            file_sha256(LOCK_PATH) != PRIOR_PREREGISTRATION_LOCK_SHA256
            or LOCK_PATH.read_bytes() != _prior_lock_bytes_from_git()
        ):
            raise ValueError("schema-1 lock changed during refreeze")
        os.replace(temporary, LOCK_PATH)
        LOCK_PATH.chmod(0o644)
        directory_descriptor = os.open(LOCK_PATH.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_prior_compatibility_lock(payload: dict[str, Any]) -> None:
    """Atomically replace only the authenticated committed schema-2 bytes."""

    if file_sha256(LOCK_PATH) != PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256:
        raise ValueError("refusing to replace a non-historical schema-2 lock")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{LOCK_PATH.name}.", suffix=".tmp", dir=LOCK_PATH.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        if (
            file_sha256(LOCK_PATH)
            != PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256
            or LOCK_PATH.read_bytes() != _prior_compatibility_lock_bytes_from_git()
        ):
            raise ValueError("schema-2 lock changed during schema-3 refreeze")
        os.replace(temporary, LOCK_PATH)
        LOCK_PATH.chmod(0o644)
        directory_descriptor = os.open(LOCK_PATH.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--execute",
        action="store_true",
        help="create an initial schema-1 lock in a fresh experiment",
    )
    action.add_argument(
        "--execute-refreeze",
        action="store_true",
        help=(
            "replace the exact authenticated prior lock with the next schema "
            "after its append-only erratum and zero-result inventory validate"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    config = load_config(CONFIG_PATH, verify_extraction_inputs=True)
    working_lock_sha256 = file_sha256(LOCK_PATH) if LOCK_PATH.is_file() else None
    if args.execute:
        action = "initial"
        payload = build_lock(config)
    elif (
        working_lock_sha256 == PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256
        and IMPLEMENTATION_ERRATUM_2_PATH.is_file()
    ):
        action = "schema3"
        payload = build_refreeze_2_lock(config)
    elif (
        working_lock_sha256 == PRIOR_PREREGISTRATION_LOCK_SHA256
        and IMPLEMENTATION_ERRATUM_PATH.is_file()
        and not IMPLEMENTATION_ERRATUM_2_PATH.exists()
    ):
        action = "schema2"
        payload = build_refreeze_lock(config)
    elif working_lock_sha256 is None and not args.execute_refreeze:
        action = "initial"
        payload = build_lock(config)
    else:
        raise ValueError("working preregistration lock/erratum chain is not refreezable")
    if not args.execute and not args.execute_refreeze:
        ready_status = {
            "initial": "READY_TO_FREEZE",
            "schema2": "READY_TO_REFREEZE_SCHEMA_2",
            "schema3": "READY_TO_REFREEZE_SCHEMA_3",
        }[action]
        print(
            json.dumps(
                {
                    "status": ready_status,
                    "schema_version": payload["schema_version"],
                    "formal_cell_count": payload["formal_cell_count"],
                    "implementation_file_count": len(payload["implementation_sha256"]),
                    "would_write": str(LOCK_PATH),
                },
                sort_keys=True,
            )
        )
        return
    if args.execute_refreeze:
        if action == "schema3":
            _replace_prior_compatibility_lock(payload)
            status = "REFROZEN_SCHEMA_3"
        elif action == "schema2":
            _replace_prior_lock(payload)
            status = "REFROZEN_SCHEMA_2"
        else:
            raise ValueError("--execute-refreeze requires an authenticated prior lock")
    else:
        publish_json_once(payload, LOCK_PATH)
        status = "FROZEN"
    print(json.dumps({"status": status, "path": str(LOCK_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
