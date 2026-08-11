#!/usr/bin/env python3
"""Aggregate the complete Stage B matrix, DiD tables, and figures 7-14."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.analysis import aggregate_stage_b  # noqa: E402
from c1b_stage_b.cli import add_gate_arguments, authorize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = authorize(args)
    summary = aggregate_stage_b(
        checkpoint_root=args.checkpoint_root,
        probe_root=args.probe_root,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        authorization=authorization,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
