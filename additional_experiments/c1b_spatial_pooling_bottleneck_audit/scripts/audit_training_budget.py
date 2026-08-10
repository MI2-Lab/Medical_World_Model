#!/usr/bin/env python3
"""Create Table 7 from immutable history/selection files; never train."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.budget import audit_training_budget  # noqa: E402
from c1b_spatial_audit.runtime import verify_preregistration  # noqa: E402


def _atomic_csv(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_preregistration()
    if not args.execute:
        raise SystemExit("validated preregistration; pass --execute to write budget audit")
    table = ROOT / "metrics" / "table7_training_budget.csv"
    summary_path = ROOT / "metrics" / "training_budget_summary.json"
    trajectory = ROOT / "metrics" / "training_trajectories.csv"
    if any(path.exists() for path in (table, summary_path, trajectory)):
        raise FileExistsError("refusing to overwrite training-budget outputs")
    frame, summary, trajectories = audit_training_budget()
    _atomic_csv(table, frame)
    _atomic_csv(trajectory, trajectories)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "COMPLETE", "cells": len(frame)}, sort_keys=True))


if __name__ == "__main__":
    main()

