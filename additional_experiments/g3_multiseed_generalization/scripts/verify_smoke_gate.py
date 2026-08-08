#!/usr/bin/env python3
"""机械验证两个 training seeds × fold 3 × G1/G3 的真实 cache smoke gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.features import synthetic_self_test as feature_self_test  # noqa: E402
from dgrs.pcr import synthetic_self_test as pcr_self_test  # noqa: E402
from dgrs.probes import synthetic_self_test as probe_self_test  # noqa: E402


SEEDS = (2026, 3026)
FOLD = 3
MODELS = ("G1", "G3")
PLAN_SHA256 = "394402aa8235b26f07b98a32426639a915bad80c53fc49cb053e7123e97ad06c"


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


def require(condition: bool, message: str, issues: list[str]) -> None:
    if not condition:
        issues.append(message)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def load_run(seed: int, model: str, issues: list[str]) -> dict[str, Any]:
    lower = model.lower()
    run_name = f"smoke_multiseed/seed_{seed}/{lower}"
    run_dir = EXPERIMENT_ROOT / "checkpoints" / run_name / f"fold_{FOLD}"
    metric_path = (
        EXPERIMENT_ROOT / "metrics" / "training" / run_name / f"fold_{FOLD}.csv"
    )
    paths = {
        "best": run_dir / "best.pt",
        "selection": run_dir / "selection.json",
        "resolved": run_dir / "resolved_run.json",
        "history": metric_path,
    }
    for label, path in paths.items():
        require(path.is_file(), f"{seed}/{model} 缺 {label}", issues)
    if any(not path.is_file() for path in paths.values()):
        return {}
    try:
        payload = torch.load(paths["best"], map_location="cpu", weights_only=True)
        selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
        resolved = json.loads(paths["resolved"].read_text(encoding="utf-8"))
        history = pd.read_csv(paths["history"])
    except Exception as error:  # pragma: no cover - error path is reported in artifact
        issues.append(f"{seed}/{model} 资产读取失败：{type(error).__name__}")
        return {}
    effective = seed + FOLD
    for name, obj in (("checkpoint", payload), ("selection", selection), ("resolved", resolved)):
        require(int(obj.get("seed_base", -1)) == seed, f"{seed}/{model} {name} seed_base 错", issues)
        require(int(obj.get("fold", -1)) == FOLD, f"{seed}/{model} {name} fold 错", issues)
        require(
            int(obj.get("effective_seed", -1)) == effective,
            f"{seed}/{model} {name} effective_seed 错",
            issues,
        )
    require(str(payload.get("model_name")) == model, f"{seed}/{model} checkpoint model 错", issues)
    runtime = payload.get("runtime", {})
    require(isinstance(runtime, Mapping) and runtime.get("smoke") is True, f"{seed}/{model} 非 smoke", issues)
    require(payload.get("plan_sha256") == PLAN_SHA256, f"{seed}/{model} plan SHA 漂移", issues)
    require(selection.get("test_data_used") is False, f"{seed}/{model} test 参与 selection", issues)
    require(payload.get("history_sha256") == sha256(paths["history"]), f"{seed}/{model} history SHA 未闭合", issues)
    require(payload.get("selection_sha256") == sha256(paths["selection"]), f"{seed}/{model} selection SHA 未闭合", issues)
    require(len(history) == 1, f"{seed}/{model} smoke 应只有一 epoch", issues)
    if not history.empty:
        for field, expected in (("seed_base", seed), ("fold", FOLD), ("effective_seed", effective)):
            require(history[field].eq(expected).all(), f"{seed}/{model} history {field} 错", issues)
        numeric = history[
            ["total_loss", "val_base_loss", "val_ftv_loss", "val_representation_std"]
        ].to_numpy(dtype=float)
        require(bool(pd.notna(numeric).all() and (abs(numeric) != math.inf).all()), f"{seed}/{model} history 非 finite", issues)
    contract = payload.get("architecture_contract", {})
    require(contract.get("backbone_input") == "DCE7", f"{seed}/{model} 非 DCE7", issues)
    require(contract.get("first_conv_in_channels") == 7, f"{seed}/{model} 输入通道非 7", issues)
    require(contract.get("pooling") == "gap", f"{seed}/{model} 非 GAP", issues)
    require(contract.get("roi_mask_backbone_input") is False, f"{seed}/{model} mask 进入 backbone", issues)
    require(contract.get("ftv_is_forward_input") is False, f"{seed}/{model} FTV 进入 forward", issues)
    expected_lambda = 0.25 if model == "G3" else 0.0
    require(
        float(payload.get("loss_config", {}).get("lambda_ftv", -1)) == expected_lambda,
        f"{seed}/{model} lambda 错",
        issues,
    )
    if model == "G3" and not history.empty:
        for field in (
            "train_first_valid_ftv_encoder_gradient_norm_weighted",
            "train_first_valid_ftv_response_projection_gradient_norm_weighted",
            "train_first_valid_ftv_head_gradient_norm_weighted",
        ):
            require(field in history, f"{seed}/{model} 缺 {field}", issues)
            if field in history:
                require(float(history.iloc[0][field]) > 0.0, f"{seed}/{model} {field} 非正", issues)
    return {
        "seed_base": seed,
        "fold": FOLD,
        "effective_seed": effective,
        "model": model,
        "checkpoint": relative(paths["best"]),
        "checkpoint_sha256": sha256(paths["best"]),
        "history_sha256": sha256(paths["history"]),
        "selection_sha256": sha256(paths["selection"]),
        "shared_initialization_sha256": payload.get("shared_initialization_sha256"),
        "implementation_sha256": payload.get("implementation_sha256"),
        "split_hashes": payload.get("split_hashes"),
        "ftv_transform_sha256": payload.get("ftv_transform_sha256"),
        "selected_validation_base_loss": selection.get("selected_validation_base_loss"),
        "selected_validation_ftv_loss": selection.get("selected_validation_ftv_loss"),
        "selected_representation_std": selection.get("selected_representation_std"),
        "selection_mode": selection.get("selection_mode"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "smoke_gate.json",
    )
    args = parser.parse_args()
    issues: list[str] = []
    formal = sorted((EXPERIMENT_ROOT / "checkpoints" / "formal").glob("**/best.pt"))
    require(not formal, "smoke gate 前 formal checkpoints 已存在", issues)
    runs = [load_run(seed, model, issues) for seed in SEEDS for model in MODELS]
    runs = [row for row in runs if row]
    by_key = {(row["seed_base"], row["model"]): row for row in runs}
    for seed in SEEDS:
        if (seed, "G1") not in by_key or (seed, "G3") not in by_key:
            continue
        g1, g3 = by_key[(seed, "G1")], by_key[(seed, "G3")]
        require(
            g1["shared_initialization_sha256"] == g3["shared_initialization_sha256"],
            f"seed {seed} G1/G3 shared initialization 不一致",
            issues,
        )
        require(g1["split_hashes"] == g3["split_hashes"], f"seed {seed} split 不一致", issues)
        require(
            g1["ftv_transform_sha256"] == g3["ftv_transform_sha256"],
            f"seed {seed} transform 不一致",
            issues,
        )
        require(
            g1["implementation_sha256"] == g3["implementation_sha256"],
            f"seed {seed} implementation 不一致",
            issues,
        )
    if all((seed, "G1") in by_key for seed in SEEDS):
        require(
            by_key[(SEEDS[0], "G1")]["shared_initialization_sha256"]
            != by_key[(SEEDS[1], "G1")]["shared_initialization_sha256"],
            "不同 seed 初始化未变化",
            issues,
        )

    self_tests: dict[str, Any] = {}
    for name, function in (
        ("features", feature_self_test),
        ("probes", probe_self_test),
        ("pcr", pcr_self_test),
    ):
        try:
            self_tests[name] = function()
        except Exception as error:  # pragma: no cover - error recorded for audit
            issues.append(f"{name} synthetic self-test 失败：{type(error).__name__}")
            self_tests[name] = {"status": "fail", "error_type": type(error).__name__}

    result = {
        "schema_version": 1,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "seeds": list(SEEDS),
        "fold": FOLD,
        "models": list(MODELS),
        "run_count": len(runs),
        "runs": runs,
        "synthetic_self_tests": self_tests,
        "formal_checkpoint_count_at_gate": len(formal),
        "test_used_for_selection": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
