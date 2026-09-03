#!/usr/bin/env python3
"""Train one outcome-blind D1/D2/D3 cell and export all 808 image states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import ARMS, FOLDS, SEEDS, TRAIN_ONLY_MANIFEST  # noqa: E402
from dinov3_rg.data import load_fold_frame, load_train_only_ids, split_patient_ids  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402
from dinov3_rg.training import export_cell_states, train_cell  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "cache/dinov3_summaries")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "features/private/fold_targets")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/formal")
    parser.add_argument("--state-root", type=Path, default=ROOT / "features/private/states")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    cache_complete = ROOT / "manifests/dinov3_cache_complete.json"
    if not cache_complete.is_file() or json.loads(cache_complete.read_text())["status"] != "COMPLETE":
        raise SystemExit("DINO summary cache is not complete")
    folds = load_fold_frame()
    train_only = load_train_only_ids(TRAIN_ONLY_MANIFEST)
    splits = split_patient_ids(args.fold, train_only, folds)
    completion = train_cell(
        seed=args.seed, fold=args.fold, arm=args.arm,
        train_ids=splits["train"], validation_ids=splits["val"],
        summary_dir=args.summary_dir,
        targets_path=args.target_dir / f"fold_{args.fold}_targets.private.npz",
        checkpoint_root=args.checkpoint_root, device=args.device,
        workers=args.workers, epochs=args.epochs,
    )
    state_path = args.state_root / f"seed{args.seed}_fold{args.fold}_{args.arm}_states.private.npz"
    all_ispy2 = tuple(sorted(folds.loc[folds["fold"].eq(args.fold), "patient_id"].astype(str)))
    state = export_cell_states(
        checkpoint_path=args.checkpoint_root / f"seed{args.seed}_fold{args.fold}_{args.arm}/selected.private.pt",
        patient_ids=all_ispy2, summary_dir=args.summary_dir, output_path=state_path,
        device=args.device, workers=args.workers,
    )
    print({"completion": completion, "state": state})


if __name__ == "__main__":
    main()
