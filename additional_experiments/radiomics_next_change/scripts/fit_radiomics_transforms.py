#!/usr/bin/env python3
"""仅用各 fold training patients 拟合并锁定 radiomics change transforms。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rnc.config import load_config  # noqa: E402
from rnc.data import patient_hash, split_ids  # noqa: E402
from rnc.training import build_bundle, ensure_radiomics_transform  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    args = parser.parse_args()
    config = load_config(args.config)
    bundle = build_bundle(config)
    folds = args.fold if args.fold else list(range(5))
    summaries = []
    for fold in folds:
        splits = split_ids(bundle, fold)
        transform = ensure_radiomics_transform(bundle, fold, splits["train"])
        if transform.train_patient_hash != patient_hash(splits["train"]):
            raise AssertionError("transform 并非由当前 fold train IDs 拟合")
        if set(splits["train"]) & (set(splits["val"]) | set(splits["test"])):
            raise AssertionError("fold train 与 val/test 有交集")
        summaries.append(transform.to_dict())
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
