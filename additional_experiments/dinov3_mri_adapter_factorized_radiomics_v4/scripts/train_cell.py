#!/usr/bin/env python3
"""Train one outcome-blind V4 cell and export I-SPY2 states."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from factorized_rg.contracts import FOLDS, PILOT_SEED, PILOT_WEIGHTS, V2_D1_CHECKPOINT_ROOT, V2_SUMMARY_DIR, load_folds, split_ids
from factorized_rg.training import export_states, train_cell


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--fold", type=int, choices=FOLDS, required=True); parser.add_argument("--arm", required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot"); args = parser.parse_args()
    if args.phase != "pilot": raise SystemExit("formal V4 runner is locked until pilot gate passes")
    arm = args.arm.upper();
    if arm not in PILOT_WEIGHTS: raise SystemExit(f"unknown pilot arm {arm}")
    splits = split_ids(args.fold)
    if arm == "F0":
        base = V2_D1_CHECKPOINT_ROOT / f"seed2026_fold{args.fold}_D1/selected.private.pt"
    else:
        base = ROOT / f"checkpoints/pilot/seed2026_fold{args.fold}_F0/selected.private.pt"
        if not base.is_file(): raise SystemExit(f"F0 must finish before {arm}: {base}")
    result = train_cell(seed=PILOT_SEED, fold=args.fold, arm=arm, radiomics_weight=PILOT_WEIGHTS[arm], train_ids=splits["train"], validation_ids=splits["val"], checkpoint_root=ROOT / "checkpoints/pilot", device=args.device, workers=args.workers, base_checkpoint=base)
    # Export exactly the 808 I-SPY2 patients. The 139 I-SPY1 patients are
    # train-only and must never enter validation/test state archives.
    ispy2 = tuple(sorted(load_folds().loc[load_folds()["fold"].eq(args.fold), "patient_id"].astype(str)))
    state = export_states(checkpoint_path=ROOT / f"checkpoints/pilot/seed2026_fold{args.fold}_{arm}/selected.private.pt", patient_ids=ispy2, output_path=ROOT / f"features/private/pilot_states/seed2026_fold{args.fold}_{arm}_states.private.npz", device=args.device, workers=args.workers)
    print({"completion": result, "state": state})


if __name__ == "__main__": main()
