#!/usr/bin/env python3
"""Freeze the mask-free audit before extraction or label-dependent analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    CONFIG_PATH,
    GITIGNORE_PATH,
    LOCK_PATH,
    LOCK_STATUS,
    PLAN_PATH,
    canonical_sha256,
    cells,
    file_sha256,
    implementation_inventory,
    load_config,
    load_goal5_lock,
    publish_json_once,
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
    return result.stdout.strip()


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


def build_lock(config: dict[str, Any]) -> dict[str, Any]:
    _require_git_context(config)
    if LOCK_PATH.exists():
        raise FileExistsError("preregistration lock already exists and is immutable")
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
        "privacy_contract": {
            "private_patient_artifacts_owner_only": True,
            "raw_spatial_maps_persisted": False,
            "region_definition_reads_masks_labels_ftv_or_clinical": False,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="create the immutable lock; without this flag only validate readiness",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    config = load_config(CONFIG_PATH, verify_extraction_inputs=True)
    payload = build_lock(config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "READY_TO_FREEZE",
                    "formal_cell_count": payload["formal_cell_count"],
                    "implementation_file_count": len(payload["implementation_sha256"]),
                    "would_write": str(LOCK_PATH),
                },
                sort_keys=True,
            )
        )
        return
    publish_json_once(payload, LOCK_PATH)
    print(json.dumps({"status": "FROZEN", "path": str(LOCK_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
