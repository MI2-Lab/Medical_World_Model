#!/usr/bin/env python3
"""运行预注册 run-level/PASS-FAIL/layer/dynamics/fold 统计与唯一诊断。"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.analysis import (  # noqa: E402
    synthetic_self_test,
    validate_final_analysis_bundle,
    write_full_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = synthetic_self_test()
    elif args.validate_only:
        result = validate_final_analysis_bundle(ROOT)
    else:
        result = {"status": "ok", **write_full_analysis(ROOT)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
