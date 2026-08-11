#!/usr/bin/env python3
"""Run static FTV and literal natural-delta Ridge probes for one Stage B cell."""

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
)
from c1b_stage_b.inputs import load_stage_b_data  # noqa: E402
from c1b_stage_b.probes import run_ftv_probes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    add_data_contract_arguments(parser)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = authorize(args)
    data = load_stage_b_data(
        data_paths(args), authorization, verify_cache_files=False
    )
    metadata = run_ftv_probes(
        feature_path=args.features,
        records=data.ftv,
        folds=data.folds,
        authorization=authorization,
        data_provenance=data.provenance,
        output_dir=args.output_dir,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
