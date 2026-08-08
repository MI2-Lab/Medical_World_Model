#!/usr/bin/env python3
"""为五个 outer fold 准备 train-only pooled-four-visit FTV transforms。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.config import file_sha256, load_config  # noqa: E402
from dgrs.training import build_bundle, ensure_ftv_transform, split_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    args = parser.parse_args()
    config = load_config(args.config)
    bundle = build_bundle(config)
    folds = sorted(set(args.fold or range(5)))
    rows = []
    for fold in folds:
        splits = split_ids(bundle, fold)
        path = EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
        transform = ensure_ftv_transform(bundle, fold, splits["train"], path)
        rows.append(
            {
                "fold": fold,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "train_patient_hash": transform.train_patient_hash,
                "train_patient_count": transform.train_patient_count,
                "paired_train_patient_count": transform.paired_train_patient_count,
                "valid_visit_count": transform.valid_visit_count,
                "raw_targets_sha256": transform.raw_targets_sha256,
            }
        )
    print(json.dumps({"status": "完成", "transforms": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
