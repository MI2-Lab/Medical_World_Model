#!/usr/bin/env python3
"""由已封口正式分析生成14张图、15节中文报告，并可选执行验收。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.delivery import make_deliverables, validate_acceptance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="生成交付物后严格验收，并写reports/acceptance.json与metrics/acceptance_check.json",
    )
    args = parser.parse_args()
    result: dict[str, object] = {"deliverables": make_deliverables(ROOT)}
    if args.validate:
        result["acceptance"] = validate_acceptance(ROOT, write=True)
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
