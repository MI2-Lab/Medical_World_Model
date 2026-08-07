#!/usr/bin/env python3
"""拟合并锁定五折 train-only static target transforms。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from osra.targets import prepare_static_transforms  # noqa: E402
from osra.common import REPO_ROOT, atomic_json  # noqa: E402


def _portable_summary_paths(result: dict) -> dict:
    """Git 可见验证摘要只保存 repo-relative transform 路径。"""

    for row in result.get("folds", []):
        for key in ("static_transform", "change_transform"):
            resolved = Path(row[key]).resolve()
            try:
                row[key] = str(resolved.relative_to(REPO_ROOT))
            except ValueError as error:
                raise ValueError(f"验证摘要不得导出仓库外 transform 路径: {resolved}") from error
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=AUDIT_ROOT / "configs" / "audit.yaml")
    parser.add_argument("--output-dir", type=Path, default=AUDIT_ROOT / "configs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = _portable_summary_paths(
        prepare_static_transforms(args.config, args.output_dir, args.overwrite)
    )
    atomic_json(AUDIT_ROOT / "metrics" / "target_transform_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
