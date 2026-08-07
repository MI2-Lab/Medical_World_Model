#!/usr/bin/env python3
"""聚合 G1/G3 五 training seeds×五 folds 的冻结 OOF 结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgrs.analysis import (  # noqa: E402
    EXPECTED_ANALYSIS_SEED,
    EXPECTED_CONDITIONAL_REPLICATES,
    EXPECTED_CROSSED_REPLICATES,
    SAFE_OUTPUT_NAME,
    AnalysisConfig,
    AnalysisInputError,
    run_analysis,
    run_self_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=ROOT / "predictions")
    parser.add_argument(
        "--metric-input-root",
        type=Path,
        default=ROOT / "metrics",
        help="representation_probes/ 与 pcr_readouts/ selection/summary 根目录",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=ROOT / "metrics" / "training" / "formal",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints" / "formal",
    )
    parser.add_argument("--feature-root", type=Path, default=ROOT / "features")
    parser.add_argument("--output-name", default="final")
    parser.add_argument(
        "--conditional-replicates", type=int, default=EXPECTED_CONDITIONAL_REPLICATES
    )
    parser.add_argument(
        "--crossed-replicates", type=int, default=EXPECTED_CROSSED_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=EXPECTED_ANALYSIS_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    if not SAFE_OUTPUT_NAME.fullmatch(args.output_name) or args.output_name != "final":
        raise SystemExit("冻结计划要求正式聚合 output-name 恰为 final")
    locked_roots = {
        "prediction-root": ROOT / "predictions",
        "metric-input-root": ROOT / "metrics",
        "history-root": ROOT / "metrics" / "training" / "formal",
        "checkpoint-root": ROOT / "checkpoints" / "formal",
        "feature-root": ROOT / "features",
    }
    observed_roots = {
        "prediction-root": args.prediction_root,
        "metric-input-root": args.metric_input_root,
        "history-root": args.history_root,
        "checkpoint-root": args.checkpoint_root,
        "feature-root": args.feature_root,
    }
    for label, expected in locked_roots.items():
        if observed_roots[label].resolve() != expected.resolve():
            raise SystemExit(f"冻结计划要求 {label}={expected.resolve()}")
    if args.conditional_replicates != EXPECTED_CONDITIONAL_REPLICATES:
        raise SystemExit("冻结计划要求正式 conditional patient bootstrap 恰为 2000 次")
    if args.crossed_replicates != EXPECTED_CROSSED_REPLICATES:
        raise SystemExit("冻结计划要求正式 crossed bootstrap 恰为 5000 次")
    if args.seed != EXPECTED_ANALYSIS_SEED:
        raise SystemExit("冻结计划要求 bootstrap RNG seed=20260807")
    try:
        summary = run_analysis(
            AnalysisConfig(
                prediction_root=args.prediction_root.resolve(),
                metric_input_root=args.metric_input_root.resolve(),
                history_root=args.history_root.resolve(),
                checkpoint_root=args.checkpoint_root.resolve(),
                feature_root=args.feature_root.resolve(),
                metric_dir=(ROOT / "metrics" / args.output_name).resolve(),
                figure_dir=(ROOT / "figures" / args.output_name).resolve(),
                report_path=(ROOT / "reports" / "final_report.md").resolve(),
                conditional_replicates=args.conditional_replicates,
                crossed_replicates=args.crossed_replicates,
                seed=args.seed,
                overwrite=args.overwrite,
                audit_checkpoints=True,
            )
        )
    except (AnalysisInputError, FileExistsError, ValueError, OSError) as exc:
        print(f"聚合失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
