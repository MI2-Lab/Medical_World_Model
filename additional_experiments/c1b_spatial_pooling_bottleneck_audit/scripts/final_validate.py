#!/usr/bin/env python3
"""Run fail-closed non-scientific validation for the completed C1B audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.final_validation import (  # noqa: E402
    atomic_json,
    run_final_validation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing public final-validation result atomically.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        payload = run_final_validation(ROOT, REPO_ROOT)
    except BaseException as error:
        payload = {
            "schema_version": 1,
            "status": "FAIL",
            "checks": {
                "validation_runtime": {
                    "status": "FAIL",
                    "findings": [
                        {
                            "finding": "unexpected_final_validation_exception",
                            "exception_type": type(error).__name__,
                        }
                    ],
                }
            },
            "all_checks_passed": False,
            "scientific_metrics_recomputed": False,
            "new_training_performed": None,
        }
    output = ROOT / "metrics/final_validation.json"
    try:
        atomic_json(output, payload, overwrite=args.overwrite)
    except FileExistsError:
        raise SystemExit(
            "final validation output exists; pass --overwrite to replace it"
        ) from None
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["status"] != "PASS":
        raise SystemExit("final non-scientific validation failed")


if __name__ == "__main__":
    main()
