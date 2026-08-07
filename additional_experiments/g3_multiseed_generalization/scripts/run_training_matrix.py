#!/usr/bin/env python3
"""运行锁定的 5 seeds × 5 folds × (G1,G3) paired 训练矩阵。

每个 seed/fold pair 在同一张 GPU 上串行执行 G1→G3；不同 pair 最多三卡并发。
``--resume`` 只会跳过通过 schema/hash/provenance 完整校验的 finalized run，
不会删除或覆盖任何 incomplete/mismatched 输出。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.config import atomic_json, file_sha256, json_sha256, load_config  # noqa: E402
from dgrs.training import (  # noqa: E402
    ALLOWED_MODELS,
    LOCKED_LAMBDA_FTV,
    SEED_BASES,
    git_metadata,
    implementation_sha256,
)


FOLDS = tuple(range(5))
MODELS = ALLOWED_MODELS
MANIFEST_PATH = EXPERIMENT_ROOT / "metrics" / "training" / "formal" / "training_matrix_manifest.json"
PLAN_PATH = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
TRAIN_SCRIPT = EXPERIMENT_ROOT / "scripts" / "train.py"
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_PROCESSES: dict[int, subprocess.Popen[str]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 mapping: {path}")
    return payload


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} 不一致: actual={actual!r}, expected={expected!r}")


def _run_name(seed_base: int, model: str) -> str:
    return f"formal/seed_{seed_base}/{model.lower()}"


@dataclass(frozen=True)
class RunPaths:
    checkpoint_dir: Path
    best: Path
    last: Path
    fallback: Path
    resolved: Path
    selection: Path
    claim: Path
    history: Path
    log: Path


def _run_paths(seed_base: int, fold: int, model: str) -> RunPaths:
    run_name = _run_name(seed_base, model)
    checkpoint_dir = EXPERIMENT_ROOT / "checkpoints" / run_name / f"fold_{fold}"
    return RunPaths(
        checkpoint_dir=checkpoint_dir,
        best=checkpoint_dir / "best.pt",
        last=checkpoint_dir / "last.pt",
        fallback=checkpoint_dir / "fallback.pt",
        resolved=checkpoint_dir / "resolved_run.json",
        selection=checkpoint_dir / "selection.json",
        claim=checkpoint_dir / "RUN_CLAIM.json",
        history=EXPERIMENT_ROOT / "metrics" / "training" / run_name / f"fold_{fold}.csv",
        log=EXPERIMENT_ROOT / "logs" / "training" / run_name / f"fold_{fold}.log",
    )


@dataclass(frozen=True)
class RunSummary:
    seed_base: int
    fold: int
    effective_seed: int
    model_name: str
    run_name: str
    checkpoint: str
    checkpoint_sha256: str
    implementation_sha256: str
    plan_sha256: str
    shared_initialization_sha256: str
    split_hashes_sha256: str
    ftv_transform_sha256: str
    common_model_sha256: str
    common_train_sha256: str
    common_loss_sha256: str
    selected_epoch: int
    selection_mode: str
    selected_val_state_loss: float
    selected_representation_std: float
    base_gate_pass: bool


def _expected_resolved_config(seed_base: int, model: str) -> dict[str, Any]:
    config = load_config(EXPERIMENT_ROOT / "configs" / f"seed_{seed_base}.yaml")
    config = dict(config)
    config["model"] = dict(config["model"])
    config["loss"] = dict(config["loss"])
    config["train"] = dict(config["train"])
    config["model"]["model_name"] = model
    config["loss"]["lambda_ftv"] = LOCKED_LAMBDA_FTV[model]
    return config


def _common_model_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload.get("model_config", {}))
    output.pop("model_name", None)
    output.pop("direct_ftv_grounding", None)
    return output


def _common_loss_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload.get("loss_config", {}))
    output.pop("lambda_ftv", None)
    return output


def _validate_tensor_state(payload: Mapping[str, Any], path: Path) -> None:
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint 缺 state_dict: {path}")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint tensor 类型错误: {path}:{name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint 含 nonfinite tensor: {path}:{name}")


def _validate_aux_checkpoint(
    path: Path,
    seed_base: int,
    fold: int,
    model: str,
    plan_sha: str,
    implementation_sha: str,
    history_sha: str,
    selection_sha: str,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint schema 错误: {path}")
    _assert_equal(bool(payload.get("finalized")), True, f"{path.name}.finalized")
    _assert_equal(int(payload.get("schema_version", -1)), 2, f"{path.name}.schema_version")
    _assert_equal(payload.get("model_name"), model, f"{path.name}.model_name")
    _assert_equal(int(payload.get("seed_base", -1)), seed_base, f"{path.name}.seed_base")
    _assert_equal(int(payload.get("fold", -1)), fold, f"{path.name}.fold")
    _assert_equal(int(payload.get("effective_seed", -1)), seed_base + fold, f"{path.name}.effective_seed")
    _assert_equal(payload.get("plan_sha256"), plan_sha, f"{path.name}.plan_sha256")
    _assert_equal(payload.get("implementation_sha256"), implementation_sha, f"{path.name}.implementation_sha256")
    _assert_equal(payload.get("history_sha256"), history_sha, f"{path.name}.history_sha256")
    _assert_equal(payload.get("selection_sha256"), selection_sha, f"{path.name}.selection_sha256")
    _validate_tensor_state(payload, path)


def _validate_history(
    path: Path,
    seed_base: int,
    fold: int,
    model: str,
    selected_epoch: int,
) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"history 为空: {path}")
    required = {
        "seed_base",
        "fold",
        "effective_seed",
        "epoch",
        "model_name",
        "lambda_ftv",
        "val_base_loss",
        "val_ftv_loss",
        "val_representation_std",
        "noncollapse",
        "base_gate_pass",
        "checkpoint_eligible",
        "is_selected_checkpoint",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"history 缺列 {sorted(missing)}: {path}")
    selected_rows: list[dict[str, str]] = []
    for row in rows:
        _assert_equal(int(row["seed_base"]), seed_base, "history.seed_base")
        _assert_equal(int(row["fold"]), fold, "history.fold")
        _assert_equal(int(row["effective_seed"]), seed_base + fold, "history.effective_seed")
        _assert_equal(row["model_name"], model, "history.model_name")
        if not math.isclose(float(row["lambda_ftv"]), LOCKED_LAMBDA_FTV[model], rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"history lambda_ftv 不一致: {path}")
        for field in ("val_base_loss", "val_ftv_loss", "val_representation_std"):
            if not _finite(row[field]):
                raise ValueError(f"history {field} nonfinite: {path}")
        if row["is_selected_checkpoint"].strip().lower() in {"true", "1"}:
            selected_rows.append(row)
    if len(selected_rows) != 1 or int(selected_rows[0]["epoch"]) != selected_epoch:
        raise ValueError(f"history selected epoch 不唯一/不一致: {path}")


def _validate_run(seed_base: int, fold: int, model: str) -> RunSummary:
    paths = _run_paths(seed_base, fold, model)
    required = (paths.best, paths.last, paths.resolved, paths.selection, paths.claim, paths.history, paths.log)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"run incomplete，缺少: {missing}")

    plan_sha = file_sha256(PLAN_PATH)
    implementation_sha = implementation_sha256()
    history_sha = file_sha256(paths.history)
    selection_sha = file_sha256(paths.selection)
    payload = torch.load(paths.best, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"best checkpoint schema 错误: {paths.best}")
    _assert_equal(bool(payload.get("finalized")), True, "best.finalized")
    _assert_equal(int(payload.get("schema_version", -1)), 2, "best.schema_version")
    _assert_equal(payload.get("run_name"), _run_name(seed_base, model), "best.run_name")
    _assert_equal(payload.get("model_name"), model, "best.model_name")
    _assert_equal(int(payload.get("seed_base", -1)), seed_base, "best.seed_base")
    _assert_equal(int(payload.get("fold", -1)), fold, "best.fold")
    _assert_equal(int(payload.get("effective_seed", -1)), seed_base + fold, "best.effective_seed")
    _assert_equal(payload.get("plan_sha256"), plan_sha, "best.plan_sha256")
    _assert_equal(payload.get("implementation_sha256"), implementation_sha, "best.implementation_sha256")
    _assert_equal(payload.get("history_sha256"), history_sha, "best.history_sha256")
    _assert_equal(payload.get("selection_sha256"), selection_sha, "best.selection_sha256")
    _assert_equal(Path(str(payload.get("history_path", ""))).resolve(), paths.history.resolve(), "best.history_path")
    _assert_equal(Path(str(payload.get("selection_path", ""))).resolve(), paths.selection.resolve(), "best.selection_path")
    _assert_equal(payload.get("resolved_config_sha256"), json_sha256(_expected_resolved_config(seed_base, model)), "best.resolved_config_sha256")
    _assert_equal(int(payload.get("train_config", {}).get("seed", -1)), seed_base, "best.train_config.seed")
    if not math.isclose(
        float(payload.get("loss_config", {}).get("lambda_ftv", math.nan)),
        LOCKED_LAMBDA_FTV[model],
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("best.lambda_ftv 不符合锁定 protocol")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("best 缺 runtime")
    for key, expected in (
        ("seed_base", seed_base),
        ("fold", fold),
        ("effective_seed", seed_base + fold),
        ("seed", seed_base + fold),
        ("smoke", False),
    ):
        _assert_equal(runtime.get(key), expected, f"best.runtime.{key}")
    if bool(payload.get("selected_epoch_metrics", {}).get("noncollapse")) is not True:
        raise ValueError("best selected epoch 必须 non-collapse")
    selected_metrics = payload.get("selected_epoch_metrics")
    if not isinstance(selected_metrics, Mapping):
        raise ValueError("best 缺 selected_epoch_metrics")
    for field in ("val_base_loss", "val_state_loss", "val_ftv_loss", "val_representation_std"):
        if not _finite(selected_metrics.get(field)):
            raise ValueError(f"best selected_epoch_metrics.{field} nonfinite")
    for key, expected in (
        ("seed_base", seed_base),
        ("fold", fold),
        ("effective_seed", seed_base + fold),
    ):
        _assert_equal(selected_metrics.get(key), expected, f"best.selected_epoch_metrics.{key}")
    if float(selected_metrics["val_representation_std"]) < 0.05:
        raise ValueError("best representation std < 0.05")
    expected_transform = (EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json").resolve()
    _assert_equal(Path(str(payload.get("ftv_transform_path", ""))).resolve(), expected_transform, "best.ftv_transform_path")
    _assert_equal(payload.get("ftv_transform_sha256"), file_sha256(expected_transform), "best.ftv_transform_sha256")
    split_hashes = payload.get("split_hashes")
    if not isinstance(split_hashes, Mapping) or set(split_hashes) != {"train", "val", "test", "pretrain_train"}:
        raise ValueError("best.split_hashes schema 不完整")
    if any(not isinstance(value, str) or len(value) != 64 for value in split_hashes.values()):
        raise ValueError("best.split_hashes 含非法 SHA-256")
    data_contract = payload.get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise ValueError("best 缺 data_contract")
    expected_manifest_sha = str(_expected_resolved_config(seed_base, model)["data"]["fold_manifest_sha256"])
    _assert_equal(data_contract.get("fold_manifest_sha256"), expected_manifest_sha, "best.fold_manifest_sha256")
    _assert_equal(payload.get("git", {}).get("experiment_path"), EXPERIMENT_ROOT.relative_to(REPO_ROOT).as_posix(), "best.git.experiment_path")
    _validate_tensor_state(payload, paths.best)

    selection = _read_json(paths.selection)
    _assert_equal(int(selection.get("schema_version", -1)), 1, "selection.schema_version")
    for key, expected in (
        ("run_name", _run_name(seed_base, model)),
        ("model_name", model),
        ("seed_base", seed_base),
        ("fold", fold),
        ("effective_seed", seed_base + fold),
        ("test_data_used", False),
    ):
        _assert_equal(selection.get(key), expected, f"selection.{key}")
    selection_mode = str(selection.get("selection_mode"))
    if selection_mode not in ({"primary"} if model == "G1" else {"primary", "fallback_base_gate_failed"}):
        raise ValueError(f"selection_mode 非法: {selection_mode}")
    selected_epoch = int(selection.get("selected_epoch", -1))
    _assert_equal(int(payload.get("epoch", -1)), selected_epoch, "best.epoch")
    epochs = selection.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("selection.epochs 为空")
    selected_evidence = [item for item in epochs if int(item.get("epoch", -1)) == selected_epoch]
    if len(selected_evidence) != 1:
        raise ValueError("selection selected epoch evidence 不唯一")
    for item in epochs:
        for key, expected in (
            ("seed_base", seed_base),
            ("fold", fold),
            ("effective_seed", seed_base + fold),
        ):
            _assert_equal(item.get(key), expected, f"selection.epochs.{key}")
    for selection_field, metric_field in (
        ("selected_validation_base_loss", "val_base_loss"),
        ("selected_validation_ftv_loss", "val_ftv_loss"),
        ("selected_representation_std", "val_representation_std"),
    ):
        if not math.isclose(
            float(selection.get(selection_field, math.nan)),
            float(selected_metrics[metric_field]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"selection.{selection_field} 与 best 不一致")
    if model == "G1" and not bool(selected_metrics.get("base_gate_pass")):
        raise ValueError("G1 selected checkpoint 必须 base_gate_pass")
    if model == "G3" and (selection_mode == "primary") != bool(selected_metrics.get("base_gate_pass")):
        raise ValueError("G3 selection_mode 与 selected base_gate_pass 不一致")
    _validate_history(paths.history, seed_base, fold, model, selected_epoch)

    resolved = _read_json(paths.resolved)
    for key, expected in (
        ("run_name", _run_name(seed_base, model)),
        ("model_name", model),
        ("seed_base", seed_base),
        ("fold", fold),
        ("effective_seed", seed_base + fold),
        ("seed", seed_base + fold),
        ("lambda_ftv", LOCKED_LAMBDA_FTV[model]),
        ("implementation_sha256", implementation_sha),
        ("experiment_plan_sha256", plan_sha),
    ):
        _assert_equal(resolved.get(key), expected, f"resolved.{key}")
    claim = _read_json(paths.claim)
    for key, expected in (
        ("run_name", _run_name(seed_base, model)),
        ("model_name", model),
        ("seed_base", seed_base),
        ("fold", fold),
        ("effective_seed", seed_base + fold),
    ):
        _assert_equal(claim.get(key), expected, f"claim.{key}")

    _validate_aux_checkpoint(
        paths.last,
        seed_base,
        fold,
        model,
        plan_sha,
        implementation_sha,
        history_sha,
        selection_sha,
    )
    if paths.fallback.exists():
        _validate_aux_checkpoint(
            paths.fallback,
            seed_base,
            fold,
            model,
            plan_sha,
            implementation_sha,
            history_sha,
            selection_sha,
        )

    return RunSummary(
        seed_base=seed_base,
        fold=fold,
        effective_seed=seed_base + fold,
        model_name=model,
        run_name=_run_name(seed_base, model),
        checkpoint=str(paths.best.resolve()),
        checkpoint_sha256=file_sha256(paths.best),
        implementation_sha256=implementation_sha,
        plan_sha256=plan_sha,
        shared_initialization_sha256=str(payload.get("shared_initialization_sha256", "")),
        split_hashes_sha256=json_sha256(payload.get("split_hashes", {})),
        ftv_transform_sha256=str(payload.get("ftv_transform_sha256", "")),
        common_model_sha256=json_sha256(_common_model_config(payload)),
        common_train_sha256=json_sha256(payload.get("train_config", {})),
        common_loss_sha256=json_sha256(_common_loss_config(payload)),
        selected_epoch=selected_epoch,
        selection_mode=selection_mode,
        selected_val_state_loss=float(selected_metrics["val_state_loss"]),
        selected_representation_std=float(selected_metrics["val_representation_std"]),
        base_gate_pass=bool(selected_metrics.get("base_gate_pass")),
    )


def _validate_pair(g1: RunSummary, g3: RunSummary) -> None:
    for field in (
        "seed_base",
        "fold",
        "effective_seed",
        "implementation_sha256",
        "plan_sha256",
        "shared_initialization_sha256",
        "split_hashes_sha256",
        "ftv_transform_sha256",
        "common_model_sha256",
        "common_train_sha256",
        "common_loss_sha256",
    ):
        _assert_equal(getattr(g3, field), getattr(g1, field), f"paired.{field}")
    g3_payload = torch.load(Path(g3.checkpoint), map_location="cpu", weights_only=True)
    baseline = g3_payload.get("baseline_selection_contract", {})
    _assert_equal(baseline.get("paired_model"), "G1", "G3 baseline.paired_model")
    _assert_equal(Path(str(baseline.get("baseline_checkpoint"))).resolve(), Path(g1.checkpoint), "G3 baseline.path")
    _assert_equal(baseline.get("baseline_checkpoint_sha256"), g1.checkpoint_sha256, "G3 baseline.sha256")
    if not math.isclose(
        float(baseline.get("baseline_val_base_loss", math.nan)),
        g1.selected_val_state_loss,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("G3 baseline val_state_loss 与 paired G1 不一致")


def _has_any_run_output(paths: RunPaths) -> bool:
    return paths.checkpoint_dir.exists() or paths.history.exists() or paths.log.exists()


def _training_command(seed_base: int, fold: int, model: str, g1: RunSummary | None) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(EXPERIMENT_ROOT / "configs" / f"seed_{seed_base}.yaml"),
        "--run-name",
        _run_name(seed_base, model),
        "--fold",
        str(fold),
        "--model-name",
        model,
        "--device",
        "cuda",
    ]
    if model == "G3":
        if g1 is None:
            raise ValueError("G3 command 缺 paired G1 summary")
        command.extend(("--baseline-checkpoint", g1.checkpoint))
    return command


def _run_subprocess(command: list[str], gpu: int, log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"label": label, "gpu": gpu, "command": command}, ensure_ascii=False) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES[process.pid] = process
        try:
            while True:
                try:
                    return_code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(
                        json.dumps(
                            {"status": "running", "label": label, "gpu": gpu, "seconds": round(time.monotonic() - started, 1)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.pop(process.pid, None)
    if return_code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise RuntimeError(f"{label} 训练失败 (exit={return_code}); log tail:\n" + "\n".join(tail))


def _terminate_active_processes() -> None:
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES.values())
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _pair_worker(
    seed_base: int,
    fold: int,
    gpu_pool: queue.Queue[int],
    resume: bool,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        raise RuntimeError("矩阵已因其他 pair 失败而停止")
    gpu = gpu_pool.get()
    try:
        output: dict[str, Any] = {"seed_base": seed_base, "fold": fold, "gpu": gpu, "models": {}}
        summaries: dict[str, RunSummary] = {}
        for model in MODELS:
            if stop_event.is_set():
                raise RuntimeError("矩阵已因其他 pair 失败而停止")
            paths = _run_paths(seed_base, fold, model)
            if _has_any_run_output(paths):
                if not resume:
                    raise FileExistsError(f"拒绝覆盖已有 run: {paths.checkpoint_dir}")
                summary = _validate_run(seed_base, fold, model)
                status = "skipped_validated"
            else:
                label = f"seed={seed_base}/fold={fold}/{model}"
                print(json.dumps({"status": "start", "label": label, "gpu": gpu}, ensure_ascii=False), flush=True)
                _run_subprocess(
                    _training_command(seed_base, fold, model, summaries.get("G1")),
                    gpu,
                    paths.log,
                    label,
                )
                summary = _validate_run(seed_base, fold, model)
                status = "trained"
                print(json.dumps({"status": "finished", "label": label, "gpu": gpu}, ensure_ascii=False), flush=True)
            summaries[model] = summary
            output["models"][model] = {"status": status, **asdict(summary)}
        _validate_pair(summaries["G1"], summaries["G3"])
        output["pair_validated"] = True
        return output
    except BaseException:
        stop_event.set()
        raise
    finally:
        gpu_pool.put(gpu)


def _preflight_existing(resume: bool) -> dict[tuple[int, int, str], RunSummary]:
    summaries: dict[tuple[int, int, str], RunSummary] = {}
    for seed_base in SEED_BASES:
        for fold in FOLDS:
            pair: dict[str, RunSummary] = {}
            for model in MODELS:
                paths = _run_paths(seed_base, fold, model)
                if not _has_any_run_output(paths):
                    if model == "G3" and "G1" not in pair and _has_any_run_output(_run_paths(seed_base, fold, "G1")):
                        raise AssertionError("内部 preflight 状态不一致")
                    continue
                if not resume:
                    raise FileExistsError(f"发现已有 formal 输出，默认拒绝覆盖: {paths.checkpoint_dir}")
                summary = _validate_run(seed_base, fold, model)
                pair[model] = summary
                summaries[(seed_base, fold, model)] = summary
            if "G3" in pair and "G1" not in pair:
                raise ValueError(f"seed={seed_base}, fold={fold}: 已有 G3 但缺 finalized G1")
            if set(pair) == set(MODELS):
                _validate_pair(pair["G1"], pair["G3"])
    return summaries


def _validate_gpu_ids(gpus: tuple[int, ...]) -> None:
    if not 1 <= len(gpus) <= 3 or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus 需要 1–3 个不重复 GPU id")
    if any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU id 必须非负")
    if not torch.cuda.is_available():
        raise RuntimeError("正式矩阵训练需要 CUDA")
    count = torch.cuda.device_count()
    if any(gpu >= count for gpu in gpus):
        raise ValueError(f"GPU id 超出可见设备范围 0..{count - 1}")


def _new_manifest(gpus: tuple[int, ...], resumed: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_ROOT.relative_to(REPO_ROOT).as_posix(),
        "status": "running",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "resumed": resumed,
        "seed_bases": list(SEED_BASES),
        "folds": list(FOLDS),
        "models": list(MODELS),
        "pair_order": ["G1", "G3"],
        "effective_seed_rule": "seed_base + fold",
        "locked_lambda_ftv": LOCKED_LAMBDA_FTV,
        "gpus": list(gpus),
        "max_concurrent_pairs": len(gpus),
        "plan_sha256": file_sha256(PLAN_PATH),
        "implementation_sha256": implementation_sha256(),
        "runner_sha256": file_sha256(Path(__file__)),
        "seed_config_file_sha256": {
            str(seed_base): file_sha256(EXPERIMENT_ROOT / "configs" / f"seed_{seed_base}.yaml")
            for seed_base in SEED_BASES
        },
        "resolved_run_config_sha256": {
            f"seed_{seed_base}/{model}": json_sha256(_expected_resolved_config(seed_base, model))
            for seed_base in SEED_BASES
            for model in MODELS
        },
        "ftv_transform_sha256": {
            str(fold): file_sha256(EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json")
            for fold in FOLDS
        },
        "git": git_metadata(),
        "tasks": [],
        "errors": [],
    }


def _validate_resume_manifest(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ROOT.relative_to(REPO_ROOT).as_posix(),
        "seed_bases": list(SEED_BASES),
        "folds": list(FOLDS),
        "models": list(MODELS),
        "pair_order": ["G1", "G3"],
        "effective_seed_rule": "seed_base + fold",
        "locked_lambda_ftv": LOCKED_LAMBDA_FTV,
        "plan_sha256": file_sha256(PLAN_PATH),
        "implementation_sha256": implementation_sha256(),
        "runner_sha256": file_sha256(Path(__file__)),
        "seed_config_file_sha256": {
            str(seed_base): file_sha256(EXPERIMENT_ROOT / "configs" / f"seed_{seed_base}.yaml")
            for seed_base in SEED_BASES
        },
        "resolved_run_config_sha256": {
            f"seed_{seed_base}/{model}": json_sha256(_expected_resolved_config(seed_base, model))
            for seed_base in SEED_BASES
            for model in MODELS
        },
        "ftv_transform_sha256": {
            str(fold): file_sha256(EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json")
            for fold in FOLDS
        },
    }
    for key, value in expected.items():
        _assert_equal(payload.get(key), value, f"existing manifest.{key}")


def _dry_run(gpus: tuple[int, ...]) -> None:
    if not 1 <= len(gpus) <= 3 or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus 需要 1–3 个不重复非负 GPU id")
    commands: list[dict[str, Any]] = []
    for seed_base in SEED_BASES:
        for fold in FOLDS:
            fake_g1 = RunSummary(
                seed_base,
                fold,
                seed_base + fold,
                "G1",
                _run_name(seed_base, "G1"),
                str(_run_paths(seed_base, fold, "G1").best.resolve()),
                "<validated-at-runtime>",
                implementation_sha256(),
                file_sha256(PLAN_PATH),
                "",
                "",
                "",
                "",
                "",
                "",
                -1,
                "primary",
                math.nan,
                math.nan,
                True,
            )
            commands.append(
                {
                    "seed_base": seed_base,
                    "fold": fold,
                    "effective_seed": seed_base + fold,
                    "G1": _training_command(seed_base, fold, "G1", None),
                    "G3": _training_command(seed_base, fold, "G3", fake_g1),
                }
            )
    print(
        json.dumps(
            {
                "status": "dry_run",
                "pairs": len(commands),
                "runs": len(commands) * 2,
                "gpus": list(gpus),
                "first_pair": commands[0],
                "last_pair": commands[-1],
                "writes_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1, 2), help="1–3 个可见 GPU id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅跳过通过完整 schema/hash/provenance 校验的 finalized run",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印锁定矩阵和首尾命令，不写文件")
    args = parser.parse_args()
    gpus = tuple(args.gpus)
    if args.dry_run:
        _dry_run(gpus)
        return

    _validate_gpu_ids(gpus)
    existing_manifest: dict[str, Any] | None = None
    if MANIFEST_PATH.exists():
        if not args.resume:
            raise FileExistsError(f"运行总 manifest 已存在，默认拒绝覆盖: {MANIFEST_PATH}")
        existing_manifest = _read_json(MANIFEST_PATH)
        _validate_resume_manifest(existing_manifest)
    _preflight_existing(args.resume)

    manifest = _new_manifest(gpus, args.resume)
    if existing_manifest is not None:
        manifest["created_at_utc"] = existing_manifest.get("created_at_utc", manifest["created_at_utc"])
        manifest["previous_status"] = existing_manifest.get("status")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(MANIFEST_PATH, manifest)

    stop_event = threading.Event()
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    futures: dict[Future[dict[str, Any]], tuple[int, int]] = {}
    executor = ThreadPoolExecutor(max_workers=len(gpus), thread_name_prefix="g3ms-pair")
    try:
        for seed_base in SEED_BASES:
            for fold in FOLDS:
                future = executor.submit(_pair_worker, seed_base, fold, gpu_pool, args.resume, stop_event)
                futures[future] = (seed_base, fold)
        for future in as_completed(futures):
            seed_base, fold = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                stop_event.set()
                manifest["errors"].append(
                    {"seed_base": seed_base, "fold": fold, "type": type(error).__name__, "message": str(error)}
                )
                raise
            manifest["tasks"].append(result)
            manifest["tasks"].sort(key=lambda item: (item["seed_base"], item["fold"]))
            manifest["completed_pairs"] = len(manifest["tasks"])
            manifest["updated_at_utc"] = _utc_now()
            atomic_json(MANIFEST_PATH, manifest)
        if len(manifest["tasks"]) != len(SEED_BASES) * len(FOLDS):
            raise AssertionError("矩阵 task 数不完整")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["updated_at_utc"] = manifest["completed_at_utc"]
        manifest["completed_pairs"] = len(manifest["tasks"])
        manifest["completed_runs"] = len(manifest["tasks"]) * len(MODELS)
        atomic_json(MANIFEST_PATH, manifest)
    except BaseException as error:
        stop_event.set()
        for future in futures:
            future.cancel()
        _terminate_active_processes()
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = _utc_now()
        manifest["updated_at_utc"] = manifest["failed_at_utc"]
        if not manifest["errors"] or manifest["errors"][-1].get("message") != str(error):
            manifest["errors"].append({"type": type(error).__name__, "message": str(error)})
        atomic_json(MANIFEST_PATH, manifest)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    print(
        json.dumps(
            {
                "status": "完成",
                "pairs": len(manifest["tasks"]),
                "runs": len(manifest["tasks"]) * len(MODELS),
                "manifest": str(MANIFEST_PATH.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
