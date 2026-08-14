#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from raw_spatial_pcr.contracts import ARMS, FOLDS, PRIMARY_TIMINGS, SEEDS, load_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Goal C matrix; data paths remain private.")
    parser.add_argument("--input-template", required=True, help="private NPZ template with {seed}, {fold}, {arm}, {timing}")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--arms", nargs="+", default=list(ARMS[1:]), choices=ARMS[1:])
    parser.add_argument("--timings", nargs="+", default=list(PRIMARY_TIMINGS), choices=PRIMARY_TIMINGS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract()
    script = ROOT / "scripts" / "train_cell.py"
    for seed in SEEDS:
        for fold in FOLDS:
            for arm in args.arms:
                for timing in args.timings:
                    input_path = args.input_template.format(seed=seed, fold=fold, arm=arm, timing=timing)
                    output_dir = args.output_dir / f"seed_{seed}" / arm / f"fold_{fold}"
                    command = [sys.executable, str(script), "--input-npz", input_path, "--arm", arm, "--seed", str(seed), "--fold", str(fold), "--timing", timing, "--output-dir", str(output_dir), "--device", args.device]
                    print(" ".join(command))
                    if not args.dry_run:
                        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

