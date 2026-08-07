#!/usr/bin/env python3
"""验证 3×5 正式 feature 文件并合并 patient-level manifest。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from osra.extraction import finalize_feature_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=AUDIT_ROOT / "configs" / "audit.yaml")
    parser.add_argument("--feature-root", type=Path, default=AUDIT_ROOT / "features")
    parser.add_argument("--output", type=Path, default=AUDIT_ROOT / "features" / "feature_manifest.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = finalize_feature_manifest(args.config, args.feature_root, args.output, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
