#!/usr/bin/env python3
"""运行一个 DGRS model×fold 的严格 image-state-only pCR readouts。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DGRS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DGRS_ROOT / "src"))

from dgrs.features import MODELS  # noqa: E402
from dgrs.pcr import (  # noqa: E402
    C_GRID,
    PENALTIES,
    run_pcr_readouts,
    synthetic_self_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--feature-root", type=Path, default=DGRS_ROOT / "features")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=DGRS_ROOT / "predictions" / "pcr_readouts",
    )
    parser.add_argument(
        "--metric-root",
        type=Path,
        default=DGRS_ROOT / "metrics" / "pcr_readouts",
    )
    parser.add_argument("--penalties", nargs="+", choices=PENALTIES, default=list(PENALTIES))
    parser.add_argument("--C-grid", nargs="+", type=float, default=list(C_GRID))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return
    if args.model is None or args.fold is None:
        parser.error("非 --self-test 模式必须显式提供 --model 与 --fold")
    result = run_pcr_readouts(
        model_name=args.model,
        fold=args.fold,
        feature_root=args.feature_root,
        prediction_root=args.prediction_root,
        metric_root=args.metric_root,
        penalties=args.penalties,
        c_grid=args.C_grid,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
