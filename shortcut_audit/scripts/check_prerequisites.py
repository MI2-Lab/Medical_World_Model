#!/usr/bin/env python3
"""检查 CoRe-WM shortcut audit 的不可变前置条件。

本脚本只读取仓库、配置和既有产物，并将检查结果写入
``shortcut_audit/metrics``。它不会构建 cache、训练模型或改写原始结果。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = AUDIT_ROOT.parent
CLEAN_ROOT = REPOSITORY_ROOT / "ispy_jepa_tmi_clean"
EXPECTED_BRANCH = "feature/ispy-clean-corejepa"


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def path_record(path: Path, *, expected_count: int | None = None) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
    if path.is_dir() and expected_count is not None:
        record["npz_count"] = sum(1 for _ in path.glob("*.npz"))
        record["expected_npz_count"] = expected_count
        record["complete"] = record["npz_count"] == expected_count
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "paper_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=AUDIT_ROOT / "metrics" / "prerequisite_check.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    run_dir = (CLEAN_ROOT / config["train"]["output_dir"]).resolve()

    sys.path.insert(0, str(CLEAN_ROOT))
    from corejepa.config import load_config  # pylint: disable=import-outside-toplevel
    from corejepa.training.runner import (  # pylint: disable=import-outside-toplevel
        build_splits,
        load_experiment_records,
    )

    experiment_config = load_config(config_path)
    records, n_primary = load_experiment_records(experiment_config)
    splits = build_splits(records, n_primary, experiment_config.train.split_seed)

    required_data = {
        "ispy2_root": path_record(Path(data["ispy2_root"])),
        "ispy2_labels": path_record(Path(data["ispy2_labels"])),
        "ispy1_root": path_record(Path(data["ispy1_root"])),
        "ispy1_labels": path_record(Path(data["ispy1_labels"])),
        "tensor_cache": path_record(Path(data["tensor_cache"]), expected_count=len(records)),
        "response_cache": path_record(Path(data["response_cache"])),
    }
    artifact_names = (
        "config.yaml",
        "best_corejepa.pt",
        "last_corejepa.pt",
        "history.csv",
        "frozen_states.npz",
        "splits.json",
        "flr.pkl",
        "flr_metrics.csv",
        "flr_scores.csv",
        "flr_summary.json",
    )
    run_artifacts = {name: path_record(run_dir / name) for name in artifact_names}

    branch = git_output("branch", "--show-current")
    source_split = (CLEAN_ROOT / "corejepa" / "data" / "records.py").read_text()
    has_five_fold_implementation = any(
        token in source_split for token in ("KFold", "StratifiedKFold", "GroupKFold")
    )
    critical_failures: list[str] = []
    if branch != EXPECTED_BRANCH:
        critical_failures.append(f"当前分支为 {branch}，预期为 {EXPECTED_BRANCH}")
    for name in ("ispy2_root", "ispy2_labels", "ispy1_root", "ispy1_labels"):
        if not required_data[name]["exists"]:
            critical_failures.append(f"缺少数据前置条件：{name}")
    if not required_data["tensor_cache"].get("complete", False):
        critical_failures.append("clean tensor cache 不存在或不完整")
    if not required_data["response_cache"]["exists"]:
        critical_failures.append("clean response cache 不存在")
    for name in ("best_corejepa.pt", "frozen_states.npz", "splits.json", "flr.pkl"):
        if not run_artifacts[name]["exists"]:
            critical_failures.append(f"缺少正式产物：{name}")
    if not has_five_fold_implementation:
        critical_failures.append("clean 分支未实现任务要求的五折 patient-level 协议")

    output = {
        "status": "ready" if not critical_failures else "blocked",
        "repository": {
            "root": str(REPOSITORY_ROOT),
            "branch": branch,
            "expected_branch": EXPECTED_BRANCH,
            "commit": git_output("rev-parse", "HEAD"),
        },
        "config": str(config_path),
        "records": {
            "total": len(records),
            "ispy2_primary": n_primary,
            "ispy1_pretraining_only": len(records) - n_primary,
            "implemented_split_sizes": {name: len(indices) for name, indices in splits.items()},
            "implemented_protocol": "single stratified 70/15/15 split",
            "five_fold_implementation_detected": has_five_fold_implementation,
        },
        "data": required_data,
        "run_directory": str(run_dir),
        "run_artifacts": run_artifacts,
        "critical_failures": critical_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if critical_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
