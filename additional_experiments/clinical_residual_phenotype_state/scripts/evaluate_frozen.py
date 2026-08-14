#!/usr/bin/env python3
"""Run Goal-F post-freeze probes, complementarity models, and decision gates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from crps.evaluation import run_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "evaluation.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    decision = run_evaluation(config_path=args.config, overwrite=args.overwrite)
    print(decision["classification"]["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
