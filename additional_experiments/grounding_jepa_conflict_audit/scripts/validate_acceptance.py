#!/usr/bin/env python3
"""只执行Grounding–JEPA conflict audit最终验收，不生成图或分析结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.delivery import validate_acceptance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="执行全部检查但不写验收JSON",
    )
    args = parser.parse_args()
    result = validate_acceptance(ROOT, write=not args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
