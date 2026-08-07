#!/usr/bin/env python3
"""运行一个 DGRS model×fold 的 frozen static/delta Ridge probes。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DGRS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DGRS_ROOT / "src"))

from dgrs.features import MODELS  # noqa: E402
from dgrs.probes import (  # noqa: E402
    ALPHAS,
    TARGETS,
    run_representation_probes,
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
        default=DGRS_ROOT / "predictions" / "representation_probes",
    )
    parser.add_argument(
        "--metric-root",
        type=Path,
        default=DGRS_ROOT / "metrics" / "representation_probes",
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return
    if args.model is None or args.fold is None:
        parser.error("非 --self-test 模式必须显式提供 --model 与 --fold")
    result = run_representation_probes(
        model_name=args.model,
        fold=args.fold,
        feature_root=args.feature_root,
        prediction_root=args.prediction_root,
        metric_root=args.metric_root,
        targets=args.targets,
        alphas=args.alphas,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
