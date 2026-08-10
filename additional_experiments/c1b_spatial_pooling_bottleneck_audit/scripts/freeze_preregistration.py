#!/usr/bin/env python3
"""Freeze the outcome-blind pooling plan and the immutable 40-cell inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.contracts import (  # noqa: E402
    REPO_ROOT,
    UPSTREAM_COMPLETION_SHA256,
    UPSTREAM_ROOT,
    UPSTREAM_SOURCE_SHA256,
    cell_key,
    cells,
    checkpoint_path,
    file_sha256,
    reference_feature_metadata_path,
    reference_feature_path,
    reference_probe_dir,
    relative,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 drift: expected {expected}, got {observed}")
    return observed


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _assert_no_analysis_outputs() -> None:
    allowed = {".gitkeep"}
    for name in ("features", "probes", "metrics", "figures", "reports", "logs"):
        root = ROOT / name
        if not root.exists():
            continue
        unexpected = [
            path for path in root.rglob("*") if path.is_file() and path.name not in allowed
        ]
        if unexpected:
            raise FileExistsError(
                f"preregistration must precede analysis outputs; found {unexpected[0]}"
            )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_lock() -> dict[str, Any]:
    _assert_no_analysis_outputs()
    plan = ROOT / "EXPERIMENT_PLAN.md"
    config = ROOT / "configs" / "audit.json"
    if not plan.is_file() or not config.is_file():
        raise FileNotFoundError("plan/config must exist before preregistration")
    config_payload = _read_json(config)

    completion: dict[str, str] = {}
    for rel, expected in UPSTREAM_COMPLETION_SHA256.items():
        completion[rel] = _require_sha(UPSTREAM_ROOT / rel, expected, rel)

    sources: dict[str, str] = {}
    for rel, expected in UPSTREAM_SOURCE_SHA256.items():
        sources[rel] = _require_sha(REPO_ROOT / rel, expected, rel)

    old_status = _git(
        "status", "--porcelain", "--", "additional_experiments/c1b_overlap_eligibility_ftv_stageb"
    )
    if old_status:
        raise ValueError("upstream experiment has tracked or untracked drift")

    checkpoints: dict[str, dict[str, Any]] = {}
    references: dict[str, dict[str, Any]] = {}
    for seed, arm, fold in cells():
        key = cell_key(seed, arm, fold)
        checkpoint = checkpoint_path(seed, arm, fold)
        feature = reference_feature_path(seed, arm, fold)
        metadata_path = reference_feature_metadata_path(seed, arm, fold)
        probe = reference_probe_dir(seed, arm, fold)
        for required in (
            checkpoint,
            feature,
            metadata_path,
            probe / "ridge_selection.csv",
            probe / "ridge_predictions.private.csv",
            probe / "probe_metrics.csv",
            probe / "probe_metadata.json",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        checkpoint_sha = file_sha256(checkpoint)
        metadata = _read_json(metadata_path)
        expected_identity = {"seed_base": seed, "arm": arm, "fold": fold}
        for field, expected in expected_identity.items():
            if metadata.get(field) != expected:
                raise ValueError(f"reference feature identity drift at {key}/{field}")
        if metadata.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"reference feature checkpoint binding drift at {key}")
        if metadata.get("feature_sha256") != file_sha256(feature):
            raise ValueError(f"reference feature hash binding drift at {key}")
        checkpoints[key] = {
            "path": relative(checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint.stat().st_size,
            "mtime_ns": checkpoint.stat().st_mtime_ns,
        }
        references[key] = {
            "feature_path": relative(feature),
            "feature_sha256": file_sha256(feature),
            "feature_metadata_path": relative(metadata_path),
            "feature_metadata_sha256": file_sha256(metadata_path),
            "patient_order_sha256": metadata.get("patient_order_sha256"),
            "probe_outputs_sha256": {
                name: file_sha256(probe / name)
                for name in (
                    "ridge_selection.csv",
                    "ridge_predictions.private.csv",
                    "probe_metrics.csv",
                    "probe_metadata.json",
                )
            },
        }

    if len(checkpoints) != 40 or len(references) != 40:
        raise AssertionError("preregistration inventory is not exactly 40 cells")

    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_NEW_FEATURE_OR_PROBE",
        "frozen_at_utc": config_payload["frozen_at_utc"],
        "branch": _git("branch", "--show-current"),
        "source_commit": _git("rev-parse", "HEAD"),
        "plan_sha256": file_sha256(plan),
        "config_sha256": file_sha256(config),
        "upstream_tracked_tree": _git(
            "rev-parse", "HEAD:additional_experiments/c1b_overlap_eligibility_ftv_stageb"
        ),
        "upstream_completion_sha256": completion,
        "upstream_source_sha256": sources,
        "selected_checkpoints": checkpoints,
        "formal_p0_references": references,
        "formal_cell_count": len(checkpoints),
        "new_training_performed": False,
        "ftv_results_read_before_freeze": False,
        "analysis_outputs_present_before_freeze": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "PREREGISTRATION_LOCK.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration lock: {output}")
    payload = build_lock()
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "formal_cell_count": 40, "output": relative(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

