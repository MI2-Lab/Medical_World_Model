#!/usr/bin/env python3
"""Orchestrate the frozen formal C1-C5 matrix from private paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from raw_spatial_pcr.contracts import ARMS, FOLDS, PRIMARY_TIMINGS, SEEDS, load_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root-template", help="template containing {seed} and {fold}")
    parser.add_argument("--feature-array-template", help="template containing {seed} and {fold}")
    parser.add_argument("--checkpoint-template", help="template containing {seed} and {fold}; required for C5")
    parser.add_argument("--raw-array", required=False, help="private raw .private.npy array for all C5 cells")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "predictions" / "formal")
    parser.add_argument("--arms", nargs="+", default=list(ARMS[1:]), choices=ARMS[1:])
    parser.add_argument("--timings", nargs="+", default=list(PRIMARY_TIMINGS), choices=PRIMARY_TIMINGS + ("T0_T3",))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_contract()
    if any(arm != "C5" for arm in args.arms) and not args.feature_root_template and not args.feature_array_template:
        raise SystemExit("C1-C4 require --feature-root-template or --feature-array-template")
    if "C5" in args.arms and not args.checkpoint_template:
        raise SystemExit("C5 requires --checkpoint-template")
    if "C5" in args.arms and not args.raw_array:
        raise SystemExit("C5 requires --raw-array")
    script = ROOT / "scripts" / "train_streaming_cell.py"
    for seed in args.seeds:
        for fold in args.folds:
            for arm in args.arms:
                for timing in args.timings:
                    output_dir = args.output_dir / f"seed_{seed}" / arm / f"fold_{fold}"
                    if args.resume and (output_dir / f"cell_{seed}_{fold}_{arm}_{timing}.private.csv").is_file():
                        print(f"SKIP existing {seed}/{fold}/{arm}/{timing}", flush=True)
                        continue
                    command = [sys.executable, str(script), "--manifest", str(args.manifest), "--arm", arm, "--seed", str(seed), "--fold", str(fold), "--timing", timing, "--output-dir", str(output_dir), "--device", args.device, "--max-epochs", str(args.max_epochs), "--patience", str(args.patience)]
                    if args.batch_size is not None:
                        command.extend(["--batch-size", str(args.batch_size)])
                    if arm == "C5":
                        command.extend(["--checkpoint", args.checkpoint_template.format(seed=seed, fold=fold), "--raw-array", args.raw_array])
                    elif args.feature_array_template:
                        command.extend(["--feature-array", args.feature_array_template.format(seed=seed, fold=fold)])
                    else:
                        command.extend(["--feature-dir", args.feature_root_template.format(seed=seed, fold=fold)])
                    print(" ".join(command), flush=True)
                    if not args.dry_run:
                        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
