#!/usr/bin/env python3
"""Evaluate and optionally publish the preregistered conditional-S3 trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.s3_trigger import (  # noqa: E402
    compute_s3_trigger_gate,
    write_s3_trigger_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-probes", type=Path, default=ROOT / "probes" / "final")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "audit.json")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "metrics" / "s3_trigger_authorization.json",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, compute and print the trigger without writing it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    gate = compute_s3_trigger_gate(
        final_probe_root=args.final_probes,
        config_path=args.config,
        equivalence_gate_path=args.p0_equivalence_gate,
        replication_gate_path=args.p0_probe_replication_gate,
    )
    if args.execute:
        expected = (ROOT / "metrics" / "s3_trigger_authorization.json").resolve()
        if args.output.expanduser().resolve() != expected:
            raise ValueError("formal S3 trigger must use the canonical metrics path")
        write_s3_trigger_gate(expected, gate)
    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
