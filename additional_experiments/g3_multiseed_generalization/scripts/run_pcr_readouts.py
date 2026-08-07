#!/usr/bin/env python3
"""运行一个 seed_base×G1/G3×fold 的 image-state-only pCR readouts。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DGRS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DGRS_ROOT / "src"))

from dgrs.features import DEFAULT_FOLD_MANIFEST, MODELS, SEED_BASES  # noqa: E402
from dgrs.pcr import (  # noqa: E402
    run_pcr_readouts,
    synthetic_self_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seed-base", type=int, choices=SEED_BASES)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument(
        "--prediction-root",
        type=Path,
    )
    parser.add_argument(
        "--metric-root",
        type=Path,
    )
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return
    if any(
        value is None
        for value in (
            args.model,
            args.fold,
            args.seed_base,
            args.feature_root,
            args.prediction_root,
            args.metric_root,
        )
    ):
        parser.error(
            "非 --self-test 模式必须显式提供 --model/--fold/--seed-base/"
            "--feature-root/--prediction-root/--metric-root"
        )
    result = run_pcr_readouts(
        model_name=args.model,
        fold=args.fold,
        seed_base=args.seed_base,
        feature_root=args.feature_root,
        prediction_root=args.prediction_root,
        metric_root=args.metric_root,
        fold_manifest=args.fold_manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
