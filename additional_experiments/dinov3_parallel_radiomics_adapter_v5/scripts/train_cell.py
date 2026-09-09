#!/usr/bin/env python3
"""Train one outcome-blind V5 parallel radiomics cell and export states."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from parallel_rg.contracts import FOLDS, PILOT_SEED, ROOT as EXPERIMENT_ROOT, V4_C0_ROOT, load_folds
from parallel_rg.training import export_states, train_cell


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--fold", type=int, choices=FOLDS, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    c0 = V4_C0_ROOT / f"seed2026_fold{args.fold}_F0/selected.private.pt"
    result = train_cell(fold=args.fold, seed=PILOT_SEED, c0_checkpoint=c0, checkpoint_root=EXPERIMENT_ROOT / "checkpoints/pilot", device=args.device, workers=args.workers)
    ispy2 = tuple(sorted(load_folds().loc[load_folds()["fold"].eq(args.fold), "patient_id"].astype(str)))
    state = export_states(fold=args.fold, c0_checkpoint=c0, rad_checkpoint=EXPERIMENT_ROOT / f"checkpoints/pilot/seed2026_fold{args.fold}_P1_PARALLEL/selected.private.pt", patient_ids=ispy2, output_path=EXPERIMENT_ROOT / f"features/private/pilot_states/seed2026_fold{args.fold}_P1_PARALLEL_states.private.npz", device=args.device, workers=args.workers)
    head_state = export_states(fold=args.fold, c0_checkpoint=c0, rad_checkpoint=EXPERIMENT_ROOT / f"checkpoints/pilot/seed2026_fold{args.fold}_P1_PARALLEL/head_only.private.pt", patient_ids=ispy2, output_path=EXPERIMENT_ROOT / f"features/private/pilot_states/seed2026_fold{args.fold}_HEAD_ONLY_states.private.npz", device=args.device, workers=args.workers)
    print({"completion": result, "state": state, "head_only_state": head_state})


if __name__ == "__main__": main()
