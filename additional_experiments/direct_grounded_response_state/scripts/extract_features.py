#!/usr/bin/env python3
"""提取一个 DGRS model×fold 的 frozen observed response state。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DGRS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DGRS_ROOT / "src"))

from dgrs.features import (  # noqa: E402
    DEFAULT_FOLD_MANIFEST,
    MODELS,
    OSRA_ROOT,
    extract_model,
    synthetic_self_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="G1--G4 best.pt；省略时只在能够唯一解析正式 checkpoint 时自动发现",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=DGRS_ROOT / "features")
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="可选显式校验值；必须与 checkpoint data_contract.cache_root 解析到同一目录",
    )
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--osra-root", type=Path, default=OSRA_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-patients-per-split",
        type=int,
        help="仅 smoke；正式 probe/pCR 会明确拒绝 partial feature",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return
    if args.model is None or args.fold is None:
        parser.error("非 --self-test 模式必须显式提供 --model 与 --fold")
    result = extract_model(
        model_name=args.model,
        fold=args.fold,
        checkpoint=args.checkpoint,
        device_name=args.device,
        output_root=args.output_root,
        cache_root=args.cache_root,
        fold_manifest=args.fold_manifest,
        osra_root=args.osra_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_patients_per_split=args.max_patients_per_split,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
