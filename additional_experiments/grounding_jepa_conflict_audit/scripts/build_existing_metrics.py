#!/usr/bin/env python3
"""在任何新 gradient forward 前闭环既有 25 个 run 指标。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.existing import synthetic_self_test, write_existing_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(
            json.dumps(
                {"status": "ok", "checks": synthetic_self_test()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = write_existing_metrics(ROOT, overwrite=False)
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
