#!/usr/bin/env python3
"""把正式 batch gradients 聚合为唯一 run-level 推断单位。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.aggregation import synthetic_self_test, write_aggregate_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return
    rows = write_aggregate_tables(ROOT, overwrite=False)
    print(json.dumps({"status": "ok", "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
