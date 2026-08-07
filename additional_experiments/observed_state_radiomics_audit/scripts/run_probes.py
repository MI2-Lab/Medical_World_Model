#!/usr/bin/env python3
"""运行一个 model×fold 的 frozen observed-state 单输出 Ridge probes。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from osra.probes import (  # noqa: E402
    CHANGE_VARIANTS,
    TASK_TYPES,
    run_probe_suite,
    synthetic_self_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=AUDIT_ROOT / "configs" / "audit.yaml"
    )
    parser.add_argument("--model", choices=("m0", "m1", "m2"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--feature-root", type=Path, default=AUDIT_ROOT / "features")
    parser.add_argument(
        "--prediction-root", type=Path, default=AUDIT_ROOT / "predictions" / "probes"
    )
    parser.add_argument(
        "--metric-root", type=Path, default=AUDIT_ROOT / "metrics" / "probes"
    )
    parser.add_argument("--output-name", default="formal")
    parser.add_argument(
        "--task",
        choices=("both", *TASK_TYPES),
        default="both",
        help="默认同时运行 static 与 change",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        help="默认 audit.yaml 全部 representation 加 canonical transition_predicted_delta",
    )
    parser.add_argument(
        "--input-variants",
        nargs="+",
        choices=CHANGE_VARIANTS,
        default=list(CHANGE_VARIANTS),
        help="只作用于 observed change representation；transition/B1 有锁定 variant",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("ftv", "ld", "sphericity", "bpe"),
        help="默认 primary+exploratory 全部 target",
    )
    parser.add_argument(
        "--b1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否加入 current-radiomics -> change 的显式 table baseline",
    )
    parser.add_argument("--transition-device", default="cpu")
    parser.add_argument("--transition-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--self-test", action="store_true", help="只运行无磁盘 synthetic 契约自检"
    )
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return
    if args.model is None or args.fold is None:
        parser.error(
            "非 --self-test 模式必须显式提供 --model 和 --fold，避免误跑全量 15 任务"
        )
    task_types = TASK_TYPES if args.task == "both" else (args.task,)
    result = run_probe_suite(
        args.config,
        args.model,
        args.fold,
        feature_root=args.feature_root,
        prediction_root=args.prediction_root,
        metric_root=args.metric_root,
        output_name=args.output_name,
        task_types=task_types,
        representations=args.representations,
        input_variants=args.input_variants,
        target_names=args.targets,
        include_b1=args.b1,
        transition_device=args.transition_device,
        transition_batch_size=args.transition_batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
