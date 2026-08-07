#!/usr/bin/env python3
"""评估单个 best.pt：冻结 readout、transition grounding 与 shortcut 扰动。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rnc.config import load_config  # noqa: E402
from rnc.evaluation import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="待评估的 best.pt")
    parser.add_argument("--config", type=Path, required=True, help="与 checkpoint 匹配的 YAML config")
    parser.add_argument("--device", default="cuda", help="例如 cuda、cuda:1 或 cpu")
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_evaluation(
        checkpoint_path=args.checkpoint,
        config=load_config(args.config),
        device_name=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
