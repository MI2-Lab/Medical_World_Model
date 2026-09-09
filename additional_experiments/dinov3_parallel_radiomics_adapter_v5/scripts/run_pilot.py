#!/usr/bin/env python3
"""Run the five V5 pilot folds, optionally sharded across GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--num-shards", type=int, default=1); parser.add_argument("--shard-index", type=int, default=0); args = parser.parse_args()
    if not args.num_shards >= 1 or not 0 <= args.shard_index < args.num_shards: raise SystemExit("invalid shard")
    statuses = {}
    for fold in range(args.shard_index, 5, args.num_shards):
        tag = f"seed2026_fold{fold}_P1_PARALLEL"; cell = ROOT / "checkpoints/pilot" / tag; complete = cell / "cell_complete.private.json"; state = ROOT / "features/private/pilot_states" / f"{tag}_states.private.npz"
        head_only = ROOT / "features/private/pilot_states" / f"seed2026_fold{fold}_HEAD_ONLY_states.private.npz"
        if complete.is_file() and state.is_file() and head_only.is_file() and (cell / "head_only.private.pt").is_file(): statuses[tag] = "REUSED"; continue
        command = [sys.executable, str(ROOT / "scripts/train_cell.py"), "--fold", str(fold), "--device", args.device, "--workers", str(args.workers)]
        statuses[tag] = "COMPLETE" if subprocess.run(command, check=False).returncode == 0 else "FAILED"
    if args.num_shards > 1: print(json.dumps({"status": "SHARD_COMPLETE", "shard_index": args.shard_index, "statuses": statuses}, indent=2)); return
    all_statuses = {f"seed2026_fold{fold}_P1_PARALLEL": "COMPLETE" if (ROOT / "checkpoints/pilot" / f"seed2026_fold{fold}_P1_PARALLEL/cell_complete.private.json").is_file() else "FAILED" for fold in range(5)}
    result = {"status": "COMPLETE" if all(value == "COMPLETE" for value in all_statuses.values()) else "FAIL", "cells": 5, "statuses": all_statuses, "outcome_fields_read": [], "clinical_fields_read": []}; (ROOT / "metrics").mkdir(exist_ok=True); (ROOT / "metrics/pilot_execution.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
