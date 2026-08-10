#!/usr/bin/env python3
"""Scan the C1B spatial audit's public text for private-data leakage."""

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
    scan_public_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing public privacy gate atomically.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        payload = scan_public_artifacts(ROOT, REPO_ROOT)
    except BaseException as error:
        payload = {
            "schema_version": 1,
            "status": "FAIL",
            "identifier_path_or_column_findings": [
                {
                    "finding": "unexpected_privacy_scan_exception",
                    "exception_type": type(error).__name__,
                }
            ],
            "contains_sensitive_identifiers_paths_or_columns": None,
        }
    output = ROOT / "metrics/privacy_gate.json"
    try:
        atomic_json(output, payload, overwrite=args.overwrite)
    except FileExistsError:
        raise SystemExit("privacy gate exists; pass --overwrite to replace it") from None
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["status"] != "PASS":
        raise SystemExit("public artifact privacy gate failed")


if __name__ == "__main__":
    main()
