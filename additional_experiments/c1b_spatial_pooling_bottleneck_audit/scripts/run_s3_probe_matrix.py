#!/usr/bin/env python3
"""Validate or execute the exact trigger-authorized S3 FTV probe matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.probe_runner import execute_probe_plan  # noqa: E402
from c1b_spatial_audit.s3_probe_runner import (  # noqa: E402
    prepare_s3_probe_matrix,
    s3_plan_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=ROOT / "features")
    parser.add_argument("--probe-root", type=Path, default=ROOT / "probes")
    parser.add_argument(
        "--trigger-gate",
        type=Path,
        default=ROOT / "metrics" / "s3_trigger_authorization.json",
    )
    parser.add_argument(
        "--p0-equivalence-gate",
        type=Path,
        default=ROOT / "metrics" / "p0_equivalence_gate.json",
    )
    parser.add_argument(
        "--p0-probe-replication-gate",
        type=Path,
        default=ROOT / "metrics" / "p0_probe_replication_gate.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, validate the complete 100-cell matrix and write nothing.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan, records = prepare_s3_probe_matrix(
        feature_root=args.feature_root,
        probe_root=args.probe_root,
        trigger_gate_path=args.trigger_gate,
        equivalence_gate_path=args.p0_equivalence_gate,
        replication_gate_path=args.p0_probe_replication_gate,
    )
    if not args.execute:
        print(json.dumps(s3_plan_summary(plan, status="VALIDATED_NOT_EXECUTED"), sort_keys=True))
        return
    completion = execute_probe_plan(
        plan,
        records=records,
        nuisance_targets=None,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "stage": completion["stage"],
                "requested_poolings": completion["requested_poolings"],
                "executed_cell_count": completion["executed_cell_count"],
                "completion_path": completion["completion_path"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
