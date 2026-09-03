#!/usr/bin/env python3
"""Verify the immutable scientific lock and optional implementation lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.preregistration import verify_preregistration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scientific-only",
        action="store_true",
        help="verify protocol hashes before the implementation manifest exists",
    )
    args = parser.parse_args()
    result = verify_preregistration(
        EXPERIMENT_ROOT, require_implementation=not args.scientific_only
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
