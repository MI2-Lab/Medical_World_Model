#!/usr/bin/env python3
"""从单个正式 checkpoint 提取 frozen observed global/ROI feature。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from osra.extraction import extract_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=AUDIT_ROOT / "configs" / "audit.yaml")
    parser.add_argument("--model", choices=("m0", "m1", "m2"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=AUDIT_ROOT / "features")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-patients-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = extract_checkpoint(
        config_path=args.config,
        model_label=args.model,
        fold=args.fold,
        device_name=args.device,
        output_root=args.output_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_patients_per_split=args.max_patients_per_split,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

