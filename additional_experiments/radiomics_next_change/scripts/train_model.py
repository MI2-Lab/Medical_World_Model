#!/usr/bin/env python3
"""训练单个 M0/M1/M2 fold；支持 smoke 与 lambda override。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rnc.config import load_config  # noqa: E402
from rnc.training import train_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-patients", type=int)
    parser.add_argument("--lambda-rad", type=float)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    checkpoint = train_fold(
        load_config(args.config),
        run_name=args.run_name,
        fold=args.fold,
        device_name=args.device,
        epochs_override=args.epochs,
        smoke_patients=args.smoke_patients,
        lambda_rad_override=args.lambda_rad,
        output_suffix=args.output_suffix,
    )
    print(json.dumps({"status": "完成", "checkpoint": str(checkpoint)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
