#!/usr/bin/env python3
"""Run/resume the paired 5-seed x 5-fold x 3-arm representation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import ARMS, FOLDS, SEEDS, atomic_json, expected_cells, file_sha256  # noqa: E402
from dinov3_rg.data import validate_state_archive  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    prerequisites = (
        ROOT / "target_feasibility.json",
        ROOT / "manifests/dinov3_cache_complete.json",
        ROOT / "metrics/smoke_cell.json",
    )
    for prerequisite in prerequisites:
        if not prerequisite.is_file() or json.loads(prerequisite.read_text(encoding="utf-8")).get("status") not in {"PASS", "COMPLETE"}:
            raise SystemExit(f"formal matrix prerequisite did not pass: {prerequisite}")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard contract requires num-shards >= 1 and 0 <= shard-index < num-shards")
    if args.num_shards != 1 and (args.seed is not None or args.fold is not None):
        raise SystemExit("seed/fold filters cannot be combined with matrix sharding")
    seeds = SEEDS if args.seed is None else (args.seed,)
    folds = FOLDS if args.fold is None else (args.fold,)
    pairs = tuple((seed, fold) for seed in seeds for fold in folds)
    pairs = pairs[args.shard_index :: args.num_shards]
    for seed, fold in pairs:
        # Order is part of checkpoint-selection semantics: D1 -> D2 -> D3.
        initialization: set[str] = set()
        for arm in ARMS:
            tag = f"seed{seed}_fold{fold}_{arm}"
            complete_path = ROOT / f"checkpoints/formal/{tag}/cell_complete.private.json"
            state_path = ROOT / f"features/private/states/{tag}_states.private.npz"
            if complete_path.is_file() and state_path.is_file():
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                validate_state_archive(state_path)
                initialization.add(complete["initialization_sha256"])
                print({"cell": tag, "status": "REUSED"}, flush=True)
                continue
            command = [
                sys.executable, str(ROOT / "scripts/train_cell.py"),
                "--seed", str(seed), "--fold", str(fold), "--arm", arm,
                "--device", args.device, "--workers", str(args.workers),
            ]
            if args.epochs is not None:
                command.extend(("--epochs", str(args.epochs)))
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError:
                failure_path = ROOT / f"checkpoints/formal/{tag}/cell_failed.private.json"
                failure = (
                    json.loads(failure_path.read_text(encoding="utf-8"))
                    if failure_path.is_file() else {"status": "SUBPROCESS_FAILED"}
                )
                completed_cells = sum(
                    (ROOT / f"checkpoints/formal/{cell}/cell_complete.private.json").is_file()
                    and (ROOT / f"features/private/states/{cell}_states.private.npz").is_file()
                    for cell in expected_cells()
                )
                gate = {
                    "schema_version": 1,
                    "status": "NO_GO",
                    "failed_stage": "representation_checkpoint_safety",
                    "decision_class": "GROUNDING_OPTIMIZATION_CONFLICT",
                    "failed_cell": tag,
                    "completed_cells_before_stop": completed_cells,
                    "failure": failure,
                    "pcr_outcomes_read": False,
                    "clinical_fields_read": [],
                    "outcome_fields_read": [],
                }
                atomic_json(ROOT / "metrics/representation_matrix_gate.json", gate)
                atomic_json(
                    ROOT / "decision.json",
                    {
                        "schema_version": 1,
                        "status": "TERMINATED_AT_REPRESENTATION_CHECKPOINT_SAFETY",
                        "decision_class": "GROUNDING_OPTIMIZATION_CONFLICT",
                        "failed_cell": tag,
                        "representation_cells_completed": completed_cells,
                        "mechanism_evaluation_started": False,
                        "mechanism_lock_created": False,
                        "pcr_evaluation_started": False,
                        "pcr_outcomes_read": False,
                        "interpretation": "paired checkpoint safety failed after the full epoch budget",
                    },
                )
                atomic_json(
                    ROOT / "acceptance_check.json",
                    {
                        "schema_version": 1,
                        "status": "FAIL",
                        "failed_stage": "representation_checkpoint_safety",
                        "decision_class": "GROUNDING_OPTIMIZATION_CONFLICT",
                        "representation_cells_completed": completed_cells,
                        "representation_lock_created": False,
                        "mechanism_lock_created": False,
                        "pcr_outcomes_read": False,
                    },
                )
                raise
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            initialization.add(complete["initialization_sha256"])
        if len(initialization) != 1:
            raise RuntimeError(f"paired initialization failed: seed={seed}, fold={fold}")
    if args.seed is None and args.fold is None and args.num_shards == 1:
        cells = {}
        for tag in expected_cells():
            complete = ROOT / f"checkpoints/formal/{tag}/cell_complete.private.json"
            state = ROOT / f"features/private/states/{tag}_states.private.npz"
            if not complete.is_file() or not state.is_file():
                raise RuntimeError(f"formal matrix is incomplete: {tag}")
            validate_state_archive(state)
            cells[tag] = {"completion_sha256": file_sha256(complete), "state_sha256": file_sha256(state)}
        payload = {"schema_version": 1, "status": "COMPLETE", "cells": 75, "artifacts": cells, "outcome_fields_read": [], "clinical_fields_read": []}
        atomic_json(ROOT / "metrics/representation_matrix_complete.json", payload)
        print(payload)
    elif args.num_shards > 1:
        print({
            "status": "SHARD_COMPLETE",
            "pairs": len(pairs),
            "cells": 3 * len(pairs),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        })


if __name__ == "__main__":
    main()
