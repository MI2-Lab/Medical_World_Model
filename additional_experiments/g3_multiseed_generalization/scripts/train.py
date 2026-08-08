#!/usr/bin/env python3
"""按预注册 protocol 训练一个 G1 或 G3 fold。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.config import load_config  # noqa: E402
from dgrs.training import train_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--model-name", "--model", dest="model_name", choices=("G1", "G3"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke-patients", type=int)
    parser.add_argument("--baseline-val-base-loss", type=float)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--export-pilot-features", action="store_true")
    args = parser.parse_args()
    checkpoint = train_fold(
        load_config(args.config),
        run_name=args.run_name,
        fold=args.fold,
        model_name=args.model_name,
        device_name=args.device,
        epochs_override=args.epochs,
        smoke_patients=args.smoke_patients,
        baseline_val_base_loss=args.baseline_val_base_loss,
        baseline_checkpoint=args.baseline_checkpoint,
        workers_override=args.workers,
        export_pilot=args.export_pilot_features,
    )
    print(json.dumps({"status": "完成", "checkpoint": str(checkpoint.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
