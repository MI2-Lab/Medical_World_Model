#!/usr/bin/env python3
"""提取单个 seed×fold G3 checkpoint 的固定 batch conflict metrics。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.gradients import extract_run, synthetic_self_test  # noqa: E402
from gjca.contracts import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--checkpoint-kind", choices=("selected", "last"), default="selected"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(
            json.dumps(synthetic_self_test(args.device), ensure_ascii=False, indent=2)
        )
        return
    if args.seed_base is None or args.fold is None:
        parser.error("正式模式必须提供 --seed-base 与 --fold")
    path = extract_run(
        args.seed_base,
        args.fold,
        args.checkpoint_kind,
        args.device,
        overwrite=False,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
