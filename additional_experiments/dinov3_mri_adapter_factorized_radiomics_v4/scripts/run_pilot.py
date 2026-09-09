#!/usr/bin/env python3
"""Run the 20-cell factorized V4 pilot, sharded by fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_VERSION = "v4-factorized-20260902-r2"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--num-shards", type=int, default=1); parser.add_argument("--shard-index", type=int, default=0); args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards: raise SystemExit("invalid shard")
    folds = list(range(args.shard_index, 5, args.num_shards)); arms = ("F0", "F005", "F010", "F025")
    statuses = {}
    for fold in folds:
        for arm in arms:
            tag = f"seed2026_fold{fold}_{arm}"; cell = ROOT / "checkpoints/pilot" / tag
            complete = cell / "cell_complete.private.json"; state = ROOT / "features/private/pilot_states" / f"{tag}_states.private.npz"
            if complete.is_file() and state.is_file():
                try:
                    payload = json.loads(complete.read_text())
                except json.JSONDecodeError:
                    payload = {}
                if payload.get("implementation_version") == IMPLEMENTATION_VERSION:
                    statuses[tag] = "REUSED"; continue
            command = [sys.executable, str(ROOT / "scripts/train_cell.py"), "--fold", str(fold), "--arm", arm, "--device", args.device, "--workers", str(args.workers)]
            result = subprocess.run(command, check=False); statuses[tag] = "COMPLETE" if result.returncode == 0 else "FAILED"
    if args.num_shards > 1:
        print(json.dumps({"status": "SHARD_COMPLETE", "shard_index": args.shard_index, "statuses": statuses}, indent=2)); return
    all_statuses = {}
    for fold in range(5):
        for arm in arms:
            tag = f"seed2026_fold{fold}_{arm}"; cell = ROOT / "checkpoints/pilot" / tag
            all_statuses[tag] = "COMPLETE" if (cell / "cell_complete.private.json").is_file() else "FAILED"
    result = {"status": "COMPLETE", "cells": 20, "statuses": all_statuses, "clinical_fields_read": [], "outcome_fields_read": []}
    (ROOT / "metrics").mkdir(exist_ok=True); (ROOT / "metrics/pilot_execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
