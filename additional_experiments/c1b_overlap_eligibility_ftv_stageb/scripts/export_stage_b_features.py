#!/usr/bin/env python3
"""Export only online pre-projector ``r`` from one selected Stage B checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.cli import (  # noqa: E402
    add_data_contract_arguments,
    add_gate_arguments,
    authorize,
    data_paths,
    resolve_device,
)
from c1b_stage_b.contracts import ARMS  # noqa: E402
from c1b_stage_b.inputs import load_stage_b_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    add_data_contract_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = authorize(args)
    from c1b_stage_b.features import export_response_features

    data = load_stage_b_data(
        data_paths(args), authorization, verify_cache_files=False
    )
    metadata = export_response_features(
        checkpoint_path=args.checkpoint,
        arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
        data=data,
        authorization=authorization,
        output_path=args.output,
        device=resolve_device(args.device),
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
