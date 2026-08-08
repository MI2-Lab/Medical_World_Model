"""本 audit 的结果前计划冻结与 fail-closed 验证。"""

from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .contracts import (
    AUDIT_ROOT,
    REPO_ROOT,
    assert_audit_config,
    assert_source_hashes,
    atomic_json,
    canonical_json_sha256,
    file_sha256,
)


OBJECTIVE_PATH = Path(
    "/home/bowen/.codex/attachments/fb7b24e3-43c5-42c6-9420-1369a5c645ea/pasted-text-1.txt"
)
OBJECTIVE_SHA256 = "5b0fa91fdefe8bd33d27902491c6ba1d6d3d9d6e50531a1807c7ec21ed0d8557"
PLAN_FREEZE = AUDIT_ROOT / "PLAN_FREEZE.json"
SOURCE_CONTRACT = AUDIT_ROOT / "SOURCE_CONTRACT.json"

CORE_PROTOCOL_FILES = (
    ".gitignore",
    "EXPERIMENT_PLAN.md",
    "configs/audit.yaml",
    "reports/asset_and_graph_inspection.md",
    "src/gjca/__init__.py",
    "src/gjca/contracts.py",
    "src/gjca/assets.py",
    "src/gjca/batches.py",
    "src/gjca/existing.py",
    "src/gjca/gradients.py",
    "src/gjca/freeze.py",
    "src/gjca/source_contract.py",
    "src/gjca/statistics.py",
    "src/gjca/phase_a.py",
    "src/gjca/diagnosis.py",
    "src/gjca/aggregation.py",
    "src/gjca/analysis.py",
    "src/gjca/delivery.py",
    "scripts/freeze_plan.py",
    "scripts/build_existing_metrics.py",
    "scripts/inspect_and_prepare.py",
    "scripts/extract_gradients.py",
    "scripts/run_gradient_matrix.py",
    "scripts/test_statistics.py",
    "scripts/run_phase_a.py",
    "scripts/aggregate_metrics.py",
    "scripts/analyze_conflicts.py",
    "scripts/make_deliverables.py",
    "scripts/validate_acceptance.py",
)

CORE_EXECUTABLE_SOURCE_FILES = tuple(
    relative
    for relative in CORE_PROTOCOL_FILES
    if relative.startswith(("src/gjca/", "scripts/"))
)

PREFREEZE_PLACEHOLDER_FILES = (
    "configs/private/.gitkeep",
    "figures/.gitkeep",
    "logs/.gitkeep",
    "metrics/raw/.gitkeep",
)

PREFREEZE_ALLOWED_FILES = frozenset(
    (*CORE_PROTOCOL_FILES, *PREFREEZE_PLACEHOLDER_FILES)
)

PREFREEZE_IGNORED_RUNTIME_CACHES = (
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _is_ignored_runtime_cache(path: Path) -> bool:
    """只豁免 .gitignore 已声明的非正式 Python/test 运行时缓存。"""
    if path.is_symlink():
        return False
    relative = path.relative_to(AUDIT_ROOT)
    return (
        "__pycache__" in relative.parts
        or ".pytest_cache" in relative.parts
        or relative.suffix == ".pyc"
    )


def _audit_file_set() -> set[str]:
    files: set[str] = set()
    for path in AUDIT_ROOT.rglob("*"):
        if (path.is_file() or path.is_symlink()) and not _is_ignored_runtime_cache(
            path
        ):
            files.add(path.relative_to(AUDIT_ROOT).as_posix())
    return files


def _assert_executable_source_file_set() -> set[str]:
    observed: set[str] = set()
    for relative_root in ("src/gjca", "scripts"):
        root = AUDIT_ROOT / relative_root
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError(f"计划冻结缺 source directory: {relative_root}")
        observed.update(
            path.relative_to(AUDIT_ROOT).as_posix()
            for path in root.rglob("*.py")
            if path.is_file() and not path.is_symlink()
        )
    expected = set(CORE_EXECUTABLE_SOURCE_FILES)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            "计划冻结 executable/source exact-set 失败: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return observed


def _assert_prefreeze_layout() -> dict[str, Any]:
    executable_sources = _assert_executable_source_file_set()
    observed = _audit_file_set()
    missing = sorted(PREFREEZE_ALLOWED_FILES - observed)
    unexpected = sorted(observed - PREFREEZE_ALLOWED_FILES)
    symlinks = sorted(
        relative for relative in observed if (AUDIT_ROOT / relative).is_symlink()
    )
    if missing:
        raise FileNotFoundError(
            f"计划冻结缺允许的 protocol/placeholder 文件: {missing}"
        )
    if unexpected or symlinks:
        raise FileExistsError(
            "计划冻结前仅允许源码、计划、asset graph report 与固定 .gitkeep；"
            f"unexpected={unexpected}, symlinks={symlinks}"
        )
    return {
        "allowed_file_count": len(PREFREEZE_ALLOWED_FILES),
        "observed_file_count": len(observed),
        "core_protocol_file_count": len(CORE_PROTOCOL_FILES),
        "executable_source_file_count": len(executable_sources),
        "exact_executable_source_set": True,
        "ignored_runtime_caches": list(PREFREEZE_IGNORED_RUNTIME_CACHES),
        "formal_outputs_present": False,
    }


def _core_hashes() -> dict[str, str]:
    _assert_executable_source_file_set()
    hashes: dict[str, str] = {}
    for relative in CORE_PROTOCOL_FILES:
        path = AUDIT_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"计划冻结缺核心文件: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def _environment() -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": devices,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }


def freeze_plan() -> Path:
    if PLAN_FREEZE.exists():
        raise FileExistsError(f"拒绝覆盖已有计划冻结: {PLAN_FREEZE}")
    prefreeze_scan = _assert_prefreeze_layout()
    if not OBJECTIVE_PATH.is_file() or file_sha256(OBJECTIVE_PATH) != OBJECTIVE_SHA256:
        raise ValueError("goal objective attachment 缺失或 SHA 漂移")
    config = assert_audit_config()
    source_hashes = assert_source_hashes()
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    expected_branch = "feature/ispy-clean-corejepa"
    expected_head = str(config["source"]["audit_start_commit"])
    if branch != expected_branch or head != expected_head:
        raise ValueError(f"计划冻结 git 起点漂移: {branch}@{head}")
    protected_status = _git(
        "status",
        "--short",
        "--",
        "additional_experiments/direct_grounded_response_state",
        "additional_experiments/g3_multiseed_generalization",
    )
    if protected_status:
        raise ValueError(f"冻结来源目录存在改动: {protected_status}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "freeze_kind": "result_blind_grounding_jepa_protocol",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "objective": {
            "path_role": "codex_goal_attachment",
            "sha256": OBJECTIVE_SHA256,
            "lines": 1314,
        },
        "git": {
            "branch": branch,
            "audit_start_commit": head,
            "source_training_commit": str(config["source"]["source_commit"]),
            "status_short_at_freeze": _git("status", "--short").splitlines(),
            "protected_source_status_clean": True,
        },
        "environment": _environment(),
        "prefreeze_scan": prefreeze_scan,
        "core_protocol_sha256": _core_hashes(),
        "upstream_source_sha256": source_hashes,
        "formal_gradient_results_observed_before_freeze": False,
        "formal_derived_outputs_present_at_freeze": False,
        "contains_patient_ids": False,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    atomic_json(PLAN_FREEZE, payload)
    assert_plan_freeze()
    return PLAN_FREEZE


def assert_plan_freeze() -> dict[str, Any]:
    if not PLAN_FREEZE.is_file():
        raise FileNotFoundError("缺 PLAN_FREEZE.json；正式步骤 fail-closed")
    import json

    payload = json.loads(PLAN_FREEZE.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("PLAN_FREEZE schema 非法")
    expected_digest = str(payload.get("payload_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    if canonical_json_sha256(unsigned) != expected_digest:
        raise ValueError("PLAN_FREEZE payload digest 不闭环")
    if payload.get("objective", {}).get("sha256") != OBJECTIVE_SHA256:
        raise ValueError("PLAN_FREEZE objective SHA 漂移")
    expected_scan = {
        "allowed_file_count": len(PREFREEZE_ALLOWED_FILES),
        "observed_file_count": len(PREFREEZE_ALLOWED_FILES),
        "core_protocol_file_count": len(CORE_PROTOCOL_FILES),
        "executable_source_file_count": len(CORE_EXECUTABLE_SOURCE_FILES),
        "exact_executable_source_set": True,
        "ignored_runtime_caches": list(PREFREEZE_IGNORED_RUNTIME_CACHES),
        "formal_outputs_present": False,
    }
    if payload.get("prefreeze_scan") != expected_scan:
        raise ValueError("PLAN_FREEZE prefreeze exact-set/结果盲 marker 漂移")
    if (
        payload.get("formal_gradient_results_observed_before_freeze") is not False
        or payload.get("formal_derived_outputs_present_at_freeze") is not False
        or payload.get("contains_patient_ids") is not False
    ):
        raise ValueError("PLAN_FREEZE 结果盲/privacy flags 漂移")
    if payload.get("core_protocol_sha256") != _core_hashes():
        raise ValueError("PLAN_FREEZE 后核心计划/实现发生漂移")
    if payload.get("upstream_source_sha256") != assert_source_hashes():
        raise ValueError("PLAN_FREEZE upstream source 发生漂移")
    assert_audit_config()
    return payload


__all__ = [
    "CORE_EXECUTABLE_SOURCE_FILES",
    "CORE_PROTOCOL_FILES",
    "OBJECTIVE_SHA256",
    "PLAN_FREEZE",
    "PREFREEZE_ALLOWED_FILES",
    "PREFREEZE_IGNORED_RUNTIME_CACHES",
    "SOURCE_CONTRACT",
    "assert_plan_freeze",
    "freeze_plan",
]
