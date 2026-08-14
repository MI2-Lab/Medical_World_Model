#!/usr/bin/env python3
"""Freeze or verify the label-free Goal-F evaluation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from crps.evaluation_lock import freeze, verify  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the existing lock instead of creating it",
    )
    args = parser.parse_args()
    payload = verify() if args.verify else freeze()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
