#!/usr/bin/env python3
"""Aggregate the complete Stage B matrix, Tables 2-5, and Figures 4-12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.analysis import (  # noqa: E402
    aggregate_stage_b,
    validate_formal_aggregation_inputs,
)
from c1b_stage_b.cli import (  # noqa: E402
    add_data_contract_arguments,
    add_gate_arguments,
    authorize,
    data_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    add_data_contract_arguments(parser)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this explicit flag, validate the formal completion chain only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = authorize(args)
    # Validate the schema-v2 contract itself in addition to the completion-chain
    # SHA binding performed by the aggregation implementation.
    data_paths(args)
    if not args.execute:
        evidence = validate_formal_aggregation_inputs(
            checkpoint_root=args.checkpoint_root,
            feature_root=args.feature_root,
            probe_root=args.probe_root,
            authorization=authorization,
            data_contract=args.data_contract,
            data_contract_sha256=args.data_contract_sha256,
        )
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PREFLIGHT_PASS",
                    "execution_requested": False,
                    "input_sha256": evidence,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = aggregate_stage_b(
        checkpoint_root=args.checkpoint_root,
        feature_root=args.feature_root,
        probe_root=args.probe_root,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        authorization=authorization,
        data_contract=args.data_contract,
        data_contract_sha256=args.data_contract_sha256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
