#!/usr/bin/env python3
"""按审计协议训练一个显式 CoRe-JEPA fold；必须主动确认训练。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = AUDIT_ROOT.parent
CLEAN_ROOT = REPOSITORY_ROOT / "ispy_jepa_tmi_clean"
EXPECTED_BRANCH = "feature/ispy-clean-corejepa"
EXPECTED_COMMIT = "c413ec86af04795434bdc19e65bbb006c966f379"
DEFAULT_CONFIG = AUDIT_ROOT / "configs" / "retrain_paper_v1.yaml"
DEFAULT_OUTPUT = AUDIT_ROOT / "runs" / "retrain_paper_v1"
DEFAULT_CACHE = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_"
    "autoroi_t0fallback_minfrac05_z32_y96_x96"
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--legacy-x-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="必须显式提供；该开关防止 dry inspection 意外启动正式训练。",
    )
    args = parser.parse_args()
    if not args.allow_training:
        raise SystemExit("训练未启动：必须显式添加 --allow-training")
    branch, commit = _git("branch", "--show-current"), _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"仓库 provenance 不一致：branch={branch}, commit={commit}"
        )

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(CLEAN_ROOT))
    from corejepa.config import load_config  # pylint: disable=import-outside-toplevel
    from shortcut_audit.auditlib.provenance import (  # pylint: disable=import-outside-toplevel
        inspect_checkpoint,
    )
    from shortcut_audit.auditlib.training import (  # pylint: disable=import-outside-toplevel
        SEED2026_MANIFEST_SHA256,
        train_explicit_fold,
    )

    config = load_config(args.config.resolve())
    config.train.gpus = (args.gpu,)
    config.train.seed = int(config.train.seed) + int(args.fold)
    output_root = args.output_root.resolve()
    fold_dir = output_root / f"fold_{args.fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    launch = {
        "status": "launched",
        "protocol": "corejepa_shortcut_audit_retraining_v1",
        "fold": args.fold,
        "gpu": args.gpu,
        "seed": config.train.seed,
        "branch": branch,
        "commit": commit,
        "config": str(args.config.resolve()),
        "legacy_x_cache": str(args.legacy_x_cache.resolve()),
        "output_root": str(output_root),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (fold_dir / "launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n"
    )
    try:
        checkpoint = train_explicit_fold(
            config,
            args.fold,
            allow_training=True,
            expected_manifest_sha256=SEED2026_MANIFEST_SHA256,
            output_root=output_root,
            legacy_x_cache_dir=args.legacy_x_cache.resolve(),
        )
        _, summary = inspect_checkpoint(checkpoint)
    except Exception as error:
        launch.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        (fold_dir / "launch.json").write_text(
            json.dumps(launch, ensure_ascii=False, indent=2) + "\n"
        )
        raise
    launch.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": summary,
        }
    )
    (fold_dir / "launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(launch, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
