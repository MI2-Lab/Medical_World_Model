#!/usr/bin/env python3
"""冻结新实验源码并验证正式运行前的不可变协议与数据资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.config import load_config, resolve_path  # noqa: E402


SEEDS = (2026, 3026, 4026, 5026, 6026)
PLAN_SHA256 = "394402aa8235b26f07b98a32426639a915bad80c53fc49cb053e7123e97ad06c"
MANIFEST_SHA256 = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
RAW_TARGET_FILE_SHA256 = "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
RAW_FTV_SHA256 = "41b419f6e098dade710ee8963ccd6245ce3ea9bd32687afbef1223cafde9529b"
TRANSFORM_SHA256 = {
    0: "8df48a908a5d56f76a2dd1a5f52b7189b03ce64e60743f856ef14afca07ebd5b",
    1: "6b582c2bb22e8208bc2e149eec032d179182fde212b94bcf6161bd274b38b4d4",
    2: "fcdf72ea26da1ff49efbdc937c78761e41d54640dae20289ac73a193e9cee23a",
    3: "a666b556e87c955214869547c6d54f083b8f975838c12461cc1158332532792c",
    4: "cb207a387900cc9ebc3deb7dca8e448bdbea083aae495af07fd11200008d6a9c",
}
SOURCE_FREEZE_PATH = EXPERIMENT_ROOT / "SOURCE_FREEZE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def source_files() -> list[Path]:
    # 这里只冻结会影响 50-run formal training 的代码。feature/probe/pCR/analysis
    # 各自在资产中保存独立实现 SHA，因此允许它们在 GPU 训练期间继续完成审计。
    paths = [
        EXPERIMENT_ROOT / ".gitignore",
        EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md",
        EXPERIMENT_ROOT / "PLAN_FREEZE.json",
        EXPERIMENT_ROOT / "src" / "dgrs" / "__init__.py",
        EXPERIMENT_ROOT / "src" / "dgrs" / "config.py",
        EXPERIMENT_ROOT / "src" / "dgrs" / "data.py",
        EXPERIMENT_ROOT / "src" / "dgrs" / "targets.py",
        EXPERIMENT_ROOT / "src" / "dgrs" / "model.py",
        EXPERIMENT_ROOT / "src" / "dgrs" / "training.py",
        EXPERIMENT_ROOT / "scripts" / "prepare_ftv_targets.py",
        EXPERIMENT_ROOT / "scripts" / "train.py",
        EXPERIMENT_ROOT / "scripts" / "run_training_matrix.py",
        EXPERIMENT_ROOT / "scripts" / "verify_smoke_gate.py",
    ]
    paths.extend(sorted((EXPERIMENT_ROOT / "configs").glob("*.yaml")))
    paths.extend(sorted((EXPERIMENT_ROOT / "configs").glob("ftv_transform_fold_*.json")))
    paths.append(Path(__file__).resolve())
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"冻结源码缺文件：{missing}")
    return list(dict.fromkeys(path.resolve() for path in paths))


def implementation_manifest() -> dict[str, Any]:
    rows = {
        str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files()
    }
    digest = hashlib.sha256()
    for relative, value in sorted(rows.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": 1,
        "status": "frozen before formal training",
        "experiment": "g3_multiseed_generalization",
        "scope": "formal_training_only",
        "branch": git("branch", "--show-current"),
        "start_commit": "596d6d509aaf62c5385344a83b8ed66dd301ee79",
        "plan_sha256": PLAN_SHA256,
        "source_file_count": len(rows),
        "source_files": rows,
        "implementation_sha256": digest.hexdigest(),
        "formal_training_started_at_freeze": False,
    }


def validate_protocol(*, allow_existing_formal: bool) -> dict[str, Any]:
    issues: list[str] = []
    plan = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
    if sha256(plan) != PLAN_SHA256:
        issues.append("EXPERIMENT_PLAN.md SHA 漂移")
    plan_freeze = json.loads(
        (EXPERIMENT_ROOT / "PLAN_FREEZE.json").read_text(encoding="utf-8")
    )
    if plan_freeze.get("status") != "frozen" or plan_freeze.get(
        "plan_sha256"
    ) != PLAN_SHA256:
        issues.append("PLAN_FREEZE.json 与冻结计划不一致")

    resolved_configs: dict[str, Any] = {}
    for seed in SEEDS:
        path = EXPERIMENT_ROOT / "configs" / f"seed_{seed}.yaml"
        config = load_config(path)
        resolved_configs[str(seed)] = {
            "seed_base": int(config["train"]["seed"]),
            "effective_seeds": [int(config["train"]["seed"]) + fold for fold in range(5)],
        }
        if int(config["train"]["seed"]) != seed:
            issues.append(f"seed config 错位：{path.name}")
        if float(config["loss"]["lambda_ftv"]) != 0.0:
            issues.append(f"base config lambda 必须为 0，由 G3 runner 锁定 0.25：{path.name}")
        if int(config["model"]["image_channels"]) != 7:
            issues.append(f"非 DCE7 config：{path.name}")

    transform_rows: dict[str, Any] = {}
    for fold, expected in TRANSFORM_SHA256.items():
        path = EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = sha256(path)
        transform_rows[str(fold)] = {
            "sha256": observed,
            "train_patient_count": payload.get("train_patient_count"),
            "paired_train_patient_count": payload.get("paired_train_patient_count"),
        }
        if observed != expected or int(payload.get("fold", -1)) != fold:
            issues.append(f"fold {fold} FTV transform 漂移")
        if payload.get("raw_targets_sha256") != RAW_FTV_SHA256:
            issues.append(f"fold {fold} raw FTV semantic SHA 漂移")

    base = load_config(EXPERIMENT_ROOT / "configs" / "base.yaml")
    manifest = resolve_path(base["data"]["fold_manifest"])
    cache_root = resolve_path(base["data"]["cache_root"])
    primary_labels = resolve_path(base["data"]["primary_labels"])
    extra_labels = resolve_path(base["data"]["extra_labels"])
    raw_targets = resolve_path(base["data"]["ftv_targets"])
    for path, label in (
        (manifest, "fold manifest"),
        (primary_labels, "I-SPY2 labels"),
        (extra_labels, "I-SPY1 labels"),
        (raw_targets, "raw FTV targets"),
    ):
        if not path.is_file():
            issues.append(f"缺 {label}")
    if not cache_root.is_dir():
        issues.append("缺固定 DCE cache")
    if manifest.is_file() and sha256(manifest) != MANIFEST_SHA256:
        issues.append("fold manifest SHA 漂移")
    if raw_targets.is_file() and sha256(raw_targets) != RAW_TARGET_FILE_SHA256:
        issues.append("raw FTV target CSV SHA 漂移")

    formal_best = sorted((EXPERIMENT_ROOT / "checkpoints" / "formal").glob("**/best.pt"))
    if formal_best and not allow_existing_formal:
        issues.append("源码冻结前 formal checkpoints 已存在")

    current_manifest = implementation_manifest()
    if SOURCE_FREEZE_PATH.is_file():
        frozen = json.loads(SOURCE_FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen != current_manifest:
            issues.append("SOURCE_FREEZE.json 与当前实现不一致")
    else:
        issues.append("缺 SOURCE_FREEZE.json")

    result = {
        "schema_version": 1,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "branch": git("branch", "--show-current"),
        "head_at_preflight": git("rev-parse", "HEAD"),
        "plan_sha256": sha256(plan),
        "implementation_sha256": current_manifest["implementation_sha256"],
        "seed_configs": resolved_configs,
        "ftv_transforms": transform_rows,
        "fold_manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "raw_target_file_sha256": sha256(raw_targets) if raw_targets.is_file() else None,
        "cache_present": cache_root.is_dir(),
        "formal_checkpoint_count_at_preflight": len(formal_best),
        "forbidden_models": ["G0", "G2", "G4"],
        "formal_models": ["G1", "G3"],
        "g3_lambda_ftv": 0.25,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-freeze",
        action="store_true",
        help="在 formal checkpoint 不存在时写入 SOURCE_FREEZE.json",
    )
    parser.add_argument(
        "--allow-existing-formal",
        action="store_true",
        help="训练后复核时允许 formal checkpoint 已存在",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "preflight.json",
    )
    args = parser.parse_args()
    if args.write_freeze:
        if any((EXPERIMENT_ROOT / "checkpoints" / "formal").glob("**/best.pt")):
            raise RuntimeError("formal checkpoint 已存在，拒绝重新冻结源码")
        atomic_json(SOURCE_FREEZE_PATH, implementation_manifest())
    result = validate_protocol(allow_existing_formal=args.allow_existing_formal)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
