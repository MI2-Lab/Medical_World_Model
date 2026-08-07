#!/usr/bin/env python3
"""运行单 fold 的 C0/C1/C2 控制与 M0/M1/M2 radiomics Ridge probe。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rnc.config import load_config  # noqa: E402
from rnc.controls import run_control_suite, synthetic_self_test  # noqa: E402


def _required_path(
    parser: argparse.ArgumentParser, value: Path | None, flag: str
) -> Path:
    """在非 self-test 模式给出清晰的缺参错误。"""

    if value is None:
        parser.error(f"非 --self-test 模式必须提供 {flag}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ("m0", "m1", "m2"):
        parser.add_argument(
            f"--{mode}-checkpoint", type=Path, help=f"{mode.upper()} best.pt"
        )
        parser.add_argument(
            f"--{mode}-config",
            type=Path,
            help=f"与 {mode.upper()} checkpoint 匹配的 YAML",
        )
    parser.add_argument(
        "--m2-native-predictions",
        type=Path,
        help="正式 evaluate_fold 的 M2 test_predictions.csv；省略时按 checkpoint 精确定位",
    )
    parser.add_argument("--output-name", default="c0_c1_c2", help="隔离输出的安全名称")
    parser.add_argument("--device", default="cuda", help="例如 cuda、cuda:1 或 cpu")
    parser.add_argument(
        "--batch-size", "--batch", dest="batch_size", type=int, default=8
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, default=2026, help="logistic 确定性 random_state"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="仅运行无磁盘输出的 synthetic 契约自测"
    )
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return

    checkpoint_paths: dict[str, Path] = {}
    configs: dict[str, dict[str, object]] = {}
    for mode in ("m0", "m1", "m2"):
        checkpoint = _required_path(
            parser, getattr(args, f"{mode}_checkpoint"), f"--{mode}-checkpoint"
        )
        config_path = _required_path(
            parser, getattr(args, f"{mode}_config"), f"--{mode}-config"
        )
        checkpoint_paths[mode] = checkpoint
        configs[mode] = load_config(config_path)

    result = run_control_suite(
        checkpoint_paths,
        configs,
        device_name=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        output_name=args.output_name,
        random_state=args.seed,
        m2_native_predictions=args.m2_native_predictions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
