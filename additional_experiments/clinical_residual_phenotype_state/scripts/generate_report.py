#!/usr/bin/env python3
"""Generate the required aggregate-only Chinese final report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from crps.reporting import generate_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = generate_report(output_path=args.output, overwrite=args.overwrite)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
