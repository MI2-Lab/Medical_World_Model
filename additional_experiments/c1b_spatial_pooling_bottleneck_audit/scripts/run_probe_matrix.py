#!/usr/bin/env python3
"""Validate or execute the frozen spatial FTV/nuisance probe matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.probe_runner import (  # noqa: E402
    STAGES,
    execute_probe_plan,
    parse_poolings,
    plan_summary,
    prepare_formal_probe_matrix,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="final",
        help="Frozen encoder stage; final is primary and s3 is trigger-only secondary.",
    )
    parser.add_argument(
        "--poolings",
        default="P0",
        help="Comma-separated preregistered poolings (P0-only is the first formal run).",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "features",
        help="Base exporter root containing final/ and, if triggered later, s3/.",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=ROOT / "probes",
    )
    parser.add_argument(
        "--nuisance-targets",
        type=Path,
        default=ROOT / "manifests" / "nuisance_targets.private.csv",
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, validate the complete requested matrix and write nothing.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    try:
        args.poolings = parse_poolings(args.poolings)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan, records, nuisance = prepare_formal_probe_matrix(
        stage=args.stage,
        poolings=args.poolings,
        feature_root=args.feature_root,
        probe_root=args.probe_root,
        nuisance_path=args.nuisance_targets,
        equivalence_gate_path=args.p0_equivalence_gate,
        probe_replication_gate_path=args.p0_probe_replication_gate,
    )
    if not args.execute:
        print(json.dumps(plan_summary(plan, status="VALIDATED_NOT_EXECUTED"), sort_keys=True))
        return
    completion = execute_probe_plan(
        plan,
        records=records,
        nuisance_targets=nuisance,
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
