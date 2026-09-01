#!/usr/bin/env python3
"""Train one V3 pilot/formal cell and export all 808 image states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import (  # noqa: E402
    ARMS, FOLDS, PILOT_ARMS, PILOT_SEED, PILOT_WEIGHTS, SEEDS, TRAIN_ONLY_MANIFEST,
    V2_CHECKPOINT_ROOT, V2_SUMMARY_DIR, V2_TARGET_DIR,
)
from dinov3_rg.data import load_fold_frame, load_train_only_ids, split_patient_ids  # noqa: E402
from dinov3_rg.locking import verify_pilot_lock  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402
from dinov3_rg.training import export_cell_states, train_cell  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    preflight = ROOT / "metrics/preflight.json"
    if not preflight.is_file() or json.loads(preflight.read_text(encoding="utf-8"))["status"] != "PASS":
        raise SystemExit("V3 preflight has not passed")
    folds = load_fold_frame(); train_only = load_train_only_ids(TRAIN_ONLY_MANIFEST)
    splits = split_patient_ids(args.fold, train_only, folds)
    all_ispy2 = tuple(sorted(folds.loc[folds["fold"].eq(args.fold), "patient_id"].astype(str)))
    if args.phase == "pilot":
        arm = args.arm.upper()
        if args.seed != PILOT_SEED or arm not in PILOT_ARMS:
            raise SystemExit("pilot cell contract requires seed2026 and R025/R050/R100")
        checkpoint_root = ROOT / "checkpoints/pilot"
        state_root = ROOT / "features/private/pilot_states"
        base_dir = V2_CHECKPOINT_ROOT / f"seed2026_fold{args.fold}_D1"
        weight = PILOT_WEIGHTS[arm]
    else:
        arm = args.arm.upper()
        if args.seed not in SEEDS or arm not in ARMS:
            raise SystemExit("formal cell seed/arm contract failed")
        checkpoint_root = ROOT / "checkpoints/formal"
        state_root = ROOT / "features/private/formal_states"
        if arm == "C0":
            base_dir = None; weight = 0.0
        else:
            lock = verify_pilot_lock()
            weight = float(lock["selected_radiomics_weight"])
            base_dir = checkpoint_root / f"seed{args.seed}_fold{args.fold}_C0"
    completion = train_cell(
        seed=args.seed, fold=args.fold, arm=arm, radiomics_weight=weight,
        train_ids=splits["train"], validation_ids=splits["val"],
        summary_dir=V2_SUMMARY_DIR,
        targets_path=V2_TARGET_DIR / f"fold_{args.fold}_targets.private.npz",
        checkpoint_root=checkpoint_root, device=args.device, workers=args.workers,
        base_checkpoint=None if base_dir is None else base_dir / "selected.private.pt",
        base_completion=None if base_dir is None else base_dir / "cell_complete.private.json",
        epochs=args.epochs,
    )
    tag = f"seed{args.seed}_fold{args.fold}_{arm}"
    state = export_cell_states(
        checkpoint_path=checkpoint_root / tag / "selected.private.pt",
        patient_ids=all_ispy2, summary_dir=V2_SUMMARY_DIR,
        output_path=state_root / f"{tag}_states.private.npz",
        device=args.device, workers=args.workers,
    )
    print({"completion": completion, "state": state})


if __name__ == "__main__": main()
