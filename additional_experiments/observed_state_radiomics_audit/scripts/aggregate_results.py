#!/usr/bin/env python
"""聚合 Observed-State Radiomics Audit 的五折 outer-test prediction。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from osra.analysis import (  # noqa: E402
    OUTPUT_NAME_PATTERN,
    AnalysisRunConfig,
    run_analysis,
    run_self_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "严格聚合五折 OOF static/change Ridge probe；核心 cell 执行 fold 内患者分层 bootstrap。"
        )
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=AUDIT_ROOT / "predictions",
        help="probe outer-test CSV 根目录",
    )
    parser.add_argument("--output-name", default="final_analysis", help="metrics/figures 子目录名")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="核心 primary cell 的 patient bootstrap 次数（正式默认2000）",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许缺注册 cell，仅用于开发；正式结论不得使用",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许以 staging+backup 方式替换同名完整输出；默认拒绝",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="仅在系统临时目录运行合成端到端 smoke test，不写实验假结果",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    if not OUTPUT_NAME_PATTERN.fullmatch(args.output_name):
        raise ValueError("output-name 只能含字母、数字、点、下划线、短横线，且不超过128字符")
    if args.bootstrap_replicates != 2000 and not args.allow_partial:
        raise ValueError("正式 strict 聚合必须使用 --bootstrap-replicates 2000")
    summary = run_analysis(
        AnalysisRunConfig(
            prediction_dir=args.prediction_dir.resolve(),
            metric_dir=(AUDIT_ROOT / "metrics" / args.output_name).resolve(),
            figure_dir=(AUDIT_ROOT / "figures" / args.output_name).resolve(),
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            overwrite=args.overwrite,
            allow_partial=args.allow_partial,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
