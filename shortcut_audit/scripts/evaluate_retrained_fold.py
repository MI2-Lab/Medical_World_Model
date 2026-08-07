#!/usr/bin/env python3
"""运行一个 audit-retrained fold 的正式 B--F 评估；必须显式授权。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = AUDIT_ROOT.parent
CLEAN_ROOT = REPOSITORY_ROOT / "ispy_jepa_tmi_clean"
EXPECTED_BRANCH = "feature/ispy-clean-corejepa"
EXPECTED_COMMIT = "c413ec86af04795434bdc19e65bbb006c966f379"


def _nonnegative_integer(value: str) -> int:
    try:
        output = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须为非负整数") from error
    if output < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return output


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--gpu", type=_nonnegative_integer, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--donor-output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--allow-evaluation",
        action="store_true",
        help="必须显式提供；防止检查命令意外启动正式模型评估。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.allow_evaluation:
        parser.error("评估未启动：必须显式添加 --allow-evaluation")

    branch, commit = _git("branch", "--show-current"), _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or commit != EXPECTED_COMMIT:
        raise RuntimeError(f"仓库 provenance 不一致：branch={branch}, commit={commit}")

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(CLEAN_ROOT))
    from shortcut_audit.auditlib.evaluation_driver import (  # pylint: disable=import-outside-toplevel
        evaluate_retrained_fold_b_to_f,
    )

    result = evaluate_retrained_fold_b_to_f(
        args.fold_dir.resolve(),
        fold=args.fold,
        legacy_x_cache_dir=args.cache.resolve(),
        evaluation_output_dir=args.eval_output.resolve(),
        donor_output_dir=args.donor_output.resolve(),
        device=f"cuda:{args.gpu}",
    )
    summary = {
        "status": "complete",
        "protocol": "corejepa_shortcut_audit_retraining_v1",
        "fold": result.fold,
        "gpu": args.gpu,
        "device": f"cuda:{args.gpu}",
        "branch": branch,
        "commit": commit,
        "fold_dir": str(args.fold_dir.resolve()),
        "cache": str(args.cache.resolve()),
        "evaluation_output": str(result.evaluation.output_dir),
        "donor_output": str(result.donor.output_dir),
        "donor_pairs": int(len(result.donor.mapping)),
        "donor_prediction_rows": int(len(result.donor.predictions)),
        "heldout_patients": int(len(result.donor_metadata)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
