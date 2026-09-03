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
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    seeds = SEEDS if args.seed is None else (args.seed,)
    folds = FOLDS if args.fold is None else (args.fold,)
    for seed in seeds:
        for fold in folds:
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
                subprocess.run(command, check=True)
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                initialization.add(complete["initialization_sha256"])
            if len(initialization) != 1:
                raise RuntimeError(f"paired initialization failed: seed={seed}, fold={fold}")
    if args.seed is None and args.fold is None:
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


if __name__ == "__main__":
    main()
