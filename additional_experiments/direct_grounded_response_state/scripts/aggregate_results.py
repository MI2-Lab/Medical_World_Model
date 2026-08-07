#!/usr/bin/env python3
"""聚合 DGRS G0–G4 五折 probe、image-only pCR、history 与预注册决策。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.analysis import (  # noqa: E402
    SAFE_OUTPUT_NAME,
    AnalysisConfig,
    AnalysisInputError,
    run_analysis,
    run_self_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=EXPERIMENT_ROOT / "predictions",
        help="含 representation_probes/ 与 pcr_readouts/ 的根目录",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "training",
        help="训练器 metrics/training/{run_name}/fold_k.csv 根目录",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=EXPERIMENT_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--output-name",
        default="final",
        help="metrics/ 与 figures/ 下的安全子目录名",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="仅用于开发；允许 coverage/history 缺失并生成明确 placeholder，不能用于正式结论",
    )
    parser.add_argument("--overwrite", action="store_true", help="显式允许 staging+backup 替换同名输出")
    parser.add_argument(
        "--expected-probe-patients",
        type=int,
        default=375,
        help="每个 measurement probe OOF cell 的预期 exact patient 数",
    )
    parser.add_argument(
        "--expected-pcr-patients",
        type=int,
        default=808,
        help="每个 pCR OOF cell 的预期 exact patient 数",
    )
    parser.add_argument("--self-test", action="store_true", help="只在系统临时目录运行合成端到端测试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    if not SAFE_OUTPUT_NAME.fullmatch(args.output_name):
        raise SystemExit("output-name 只能含字母、数字、点、下划线、短横线，且不超过128字符")
    if args.bootstrap_replicates != 2000 and not args.allow_partial:
        raise SystemExit("正式聚合必须使用 2000 次 patient bootstrap")
    if args.allow_partial and not args.output_name.lower().startswith(
        ("dev_", "partial_", "selftest_")
    ):
        raise SystemExit(
            "--allow-partial 必须搭配 dev_/partial_/selftest_ 前缀 output-name，禁止写入 final"
        )
    if not args.allow_partial and (
        args.expected_probe_patients != 375 or args.expected_pcr_patients != 808
    ):
        raise SystemExit("正式聚合固定要求 probe=375、pCR=808 patients")
    try:
        summary = run_analysis(
            AnalysisConfig(
                prediction_root=args.prediction_root.resolve(),
                history_root=args.history_root.resolve(),
                checkpoint_root=args.checkpoint_root.resolve(),
                metric_dir=(EXPERIMENT_ROOT / "metrics" / args.output_name).resolve(),
                figure_dir=(EXPERIMENT_ROOT / "figures" / args.output_name).resolve(),
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
                overwrite=args.overwrite,
                allow_partial=args.allow_partial,
                expected_probe_patients=args.expected_probe_patients,
                expected_pcr_patients=args.expected_pcr_patients,
            )
        )
    except (AnalysisInputError, FileExistsError, ValueError) as exc:
        print(f"聚合失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
