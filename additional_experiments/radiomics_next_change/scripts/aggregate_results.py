#!/usr/bin/env python3
"""聚合 M0/M1/M2 五折评估、shortcut、radiomics grounding 与 controls。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rnc.aggregation import (  # noqa: E402
    AggregationConfig,
    AggregationInputError,
    aggregate_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-run", default="m0_final", help="M0 evaluation run name")
    parser.add_argument("--m1-run", default="m1_final", help="M1 evaluation run name")
    parser.add_argument("--m2-run", default="m2_final", help="M2 evaluation run name")
    parser.add_argument(
        "--output-tag",
        default="primary_fivefold",
        help="写入 metrics/final/<tag> 与 figures/final/<tag>；已存在则拒绝覆盖",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "仅用于 smoke/诊断；允许缺 run/fold/file/controls，" "但会写入明确缺失清单"
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="patient bootstrap 次数（至少 100）",
    )
    parser.add_argument("--seed", type=int, default=20260806, help="bootstrap 随机种子")
    parser.add_argument(
        "--controls-name",
        help="如同一 checkpoint 有多个 controls namespace，用该 output_name 消歧",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path(
            "/data/data/Preprocessed/I-SPY2/"
            "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
            "matched_patient_cv_splits_seed2026.csv"
        ),
        help="仅用于重建 fold-train radiomics mean baseline",
    )
    parser.add_argument(
        "--radiomics-raw-targets",
        type=Path,
        default=EXPERIMENT_ROOT / "data_audit" / "radiomics_transition_targets_raw.csv",
        help="只读 radiomics 相邻变化审计表",
    )
    parser.add_argument(
        "--radiomics-transform-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "configs",
        help="fold-specific radiomics transform JSON 所在目录",
    )
    args = parser.parse_args()

    config = AggregationConfig(
        run_names={"m0": args.m0_run, "m1": args.m1_run, "m2": args.m2_run},
        output_tag=args.output_tag,
        allow_partial=args.allow_partial,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        controls_name=args.controls_name,
        fold_manifest=args.fold_manifest,
        radiomics_raw_targets=args.radiomics_raw_targets,
        radiomics_transform_dir=args.radiomics_transform_dir,
    )
    try:
        summary = aggregate_results(config)
    except (AggregationInputError, FileExistsError, ValueError) as exc:
        print(f"聚合失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
