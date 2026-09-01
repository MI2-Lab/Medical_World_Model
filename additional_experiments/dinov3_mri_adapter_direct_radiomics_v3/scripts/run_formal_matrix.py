#!/usr/bin/env python3
"""Run/resume the paired 50-cell fresh-seed confirmatory matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import ARMS, FOLDS, SEEDS, atomic_json, file_sha256  # noqa: E402
from dinov3_rg.data import validate_state_archive  # noqa: E402
from dinov3_rg.locking import verify_pilot_lock  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def _paths(seed: int, fold: int, arm: str) -> tuple[Path, Path, Path]:
    tag = f"seed{seed}_fold{fold}_{arm}"
    cell = ROOT / "checkpoints/formal" / tag
    return (
        cell / "cell_complete.private.json",
        cell / "cell_failed.private.json",
        ROOT / "features/private/formal_states" / f"{tag}_states.private.npz",
    )


def _terminate(decision_class: str, reason: str, completed: int) -> None:
    atomic_json(ROOT / "decision.json", {
        "schema_version": 1, "status": "TERMINATED_DURING_FORMAL_MATRIX",
        "decision_class": decision_class, "reason": reason,
        "formal_cells_completed": completed,
        "pcr_evaluation_started": False, "pcr_outcomes_read": False,
    })
    atomic_json(ROOT / "acceptance_check.json", {
        "schema_version": 1, "status": "FAIL", "failed_stage": "formal_matrix",
        "formal_cells_completed": completed, "formal_cells_required": 50,
        "evaluation_lock_created": False, "mechanism_lock_created": False,
        "pcr_outcomes_read": False, "decision_class": decision_class,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    verify_pilot_lock()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard contract")
    pairs = [(seed, fold) for seed in SEEDS for fold in FOLDS]
    selected = pairs[args.shard_index::args.num_shards]
    completed_here = 0
    for seed, fold in selected:
        for arm in ARMS:
            complete, failed, state = _paths(seed, fold, arm)
            tag = f"seed{seed}_fold{fold}_{arm}"
            if complete.is_file() and state.is_file():
                validate_state_archive(state)
                print({"cell": tag, "status": "REUSED"}, flush=True)
                completed_here += 1
                continue
            if failed.is_file():
                if arm == "RAD":
                    _terminate("GROUNDING_OPTIMIZATION_CONFLICT", tag, completed_here)
                raise SystemExit(f"formal failed sentinel already exists: {tag}")
            if arm == "RAD":
                c0_complete, _, c0_state = _paths(seed, fold, "C0")
                if not c0_complete.is_file() or not c0_state.is_file():
                    raise RuntimeError(f"paired C0 is incomplete before RAD: seed={seed}, fold={fold}")
            command = [
                sys.executable, str(ROOT / "scripts/train_cell.py"),
                "--phase", "formal", "--seed", str(seed), "--fold", str(fold),
                "--arm", arm, "--device", args.device, "--workers", str(args.workers),
            ]
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                decision = "GROUNDING_OPTIMIZATION_CONFLICT" if arm == "RAD" else "FORMAL_C0_TRAINING_FAILURE"
                _terminate(decision, tag, completed_here)
                raise SystemExit(f"formal cell failed: {tag}")
            completed_here += 1
            print({"cell": tag, "status": "COMPLETE"}, flush=True)
    if args.num_shards > 1:
        print({"status": "SHARD_COMPLETE", "pairs": len(selected), "shard_index": args.shard_index})
        return
    artifacts = {}
    for seed, fold in pairs:
        c0_payload = None
        for arm in ARMS:
            complete, failed, state = _paths(seed, fold, arm)
            tag = f"seed{seed}_fold{fold}_{arm}"
            if failed.is_file() or not complete.is_file() or not state.is_file():
                raise RuntimeError(f"formal matrix incomplete: {tag}")
            payload = json.loads(complete.read_text(encoding="utf-8"))
            validate_state_archive(state)
            if arm == "C0":
                c0_payload = payload
            else:
                if c0_payload is None or payload["base_checkpoint_sha256"] != c0_payload["checkpoint_sha256"]:
                    raise RuntimeError(f"paired base binding failed: {tag}")
                if float(payload["selected_validation"]["jepa_loss"]) > 1.05 * float(c0_payload["selected_validation"]["jepa_loss"]):
                    _terminate("GROUNDING_OPTIMIZATION_CONFLICT", tag, len(artifacts))
                    raise SystemExit("paired JEPA safety failed")
            artifacts[tag] = {
                "completion_sha256": file_sha256(complete),
                "state_sha256": file_sha256(state),
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
    output = {
        "schema_version": 1, "status": "COMPLETE", "cell_count": len(artifacts),
        "pair_count": len(pairs), "artifacts": artifacts,
        "outcome_fields_read": [], "clinical_fields_read": [],
    }
    atomic_json(ROOT / "metrics/representation_matrix_complete.json", output)
    print(output)


if __name__ == "__main__":
    main()
