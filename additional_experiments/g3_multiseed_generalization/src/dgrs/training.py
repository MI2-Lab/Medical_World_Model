"""G1/G3 multi-seed 实验的 test-blind 五折训练与 checkpoint 选择。"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .config import EXPERIMENT_ROOT, REPO_ROOT, atomic_json, file_sha256, json_sha256, resolve_path
from .data import (
    CohortBundle,
    LongitudinalDGRSDataset,
    PatientRecord,
    load_cohort_bundle,
    records_for_ids,
    split_ids,
)
from .model import DGRSOutput, DGRSWorldModel, GROUNDED_MODELS
from .targets import PooledFTVTransform, patient_hash, raw_ftv_hash


ALLOWED_MODELS = ("G1", "G3")
SEED_BASES = (2026, 3026, 4026, 5026, 6026)
LOCKED_LAMBDA_FTV = {"G1": 0.0, "G3": 0.25}
FORMAL_PROTOCOL_LOCK: dict[str, dict[str, Any]] = {
    "model": {
        "image_channels": 7,
        "base_channels": 16,
        "latent_dim": 192,
        "predictor_depth": 3,
        "predictor_heads": 4,
        "predictor_mlp_dim": 512,
        "dropout": 0.1,
    },
    "loss": {
        "sigreg": 0.09,
        "sigreg_projections": 256,
        "step_weights": [2.0, 1.0, 0.5],
    },
    "train": {
        "batch_size": 32,
        "workers": 4,
        "epochs": 12,
        "patience": 4,
        "learning_rate": 0.00005,
        "weight_decay": 0.0001,
        "ema_momentum": 0.996,
        "max_grad_norm": 5.0,
        "min_representation_std": 0.05,
        "deterministic_algorithms": False,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unavailable"

    try:
        experiment_path = EXPERIMENT_ROOT.relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"实验目录必须位于 repository 内: {EXPERIMENT_ROOT}") from error
    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "experiment_path": experiment_path,
        "experiment_status": run("git", "status", "--short", "--", experiment_path),
    }


def implementation_sha256() -> str:
    names = ("__init__.py", "config.py", "data.py", "targets.py", "model.py", "training.py")
    paths = [EXPERIMENT_ROOT / "src" / "dgrs" / name for name in names]
    paths.extend((EXPERIMENT_ROOT / "scripts" / "prepare_ftv_targets.py", EXPERIMENT_ROOT / "scripts" / "train.py"))
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
        digest.update(str(path.relative_to(EXPERIMENT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def shared_initialization_sha256(model: DGRSWorldModel) -> str:
    common: dict[str, torch.Tensor] = {}
    for prefix, module in (
        ("encoder", model.encoder),
        ("response_projection", model.response_projection),
        ("projector", model.projector),
        ("transition", model.transition),
    ):
        for name, value in module.state_dict().items():
            common[f"{prefix}.{name}"] = value
    return tensor_state_sha256(common)


def validate_run_config(config: Mapping[str, Any]) -> None:
    for section in ("data", "model", "loss", "train"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"配置缺 mapping: {section}")
    model = config["model"]
    loss = config["loss"]
    train = config["train"]
    model_name = str(model.get("model_name", "")).upper()
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"本实验 model.model_name 只允许 {ALLOWED_MODELS}")
    if int(model.get("image_channels", 7)) != 7:
        raise ValueError("G1/G3 image_channels 必须为 7")
    for name in ("base_channels", "latent_dim", "predictor_depth", "predictor_heads", "predictor_mlp_dim"):
        if isinstance(model.get(name), bool) or int(model.get(name, 0)) <= 0:
            raise ValueError(f"model.{name} 必须为正整数")
    if not 0.0 <= float(model.get("dropout", -1.0)) < 1.0:
        raise ValueError("model.dropout 必须位于 [0,1)")
    for name in ("batch_size", "epochs", "patience"):
        if isinstance(train.get(name), bool) or int(train.get(name, 0)) <= 0:
            raise ValueError(f"train.{name} 必须为正整数")
    if isinstance(train.get("workers"), bool) or int(train.get("workers", -1)) < 0:
        raise ValueError("train.workers 必须为非负整数")
    seed_base = train.get("seed")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base not in SEED_BASES:
        raise ValueError(f"train.seed 必须是预注册 seed_base 之一: {SEED_BASES}")
    for name in ("learning_rate", "weight_decay", "ema_momentum", "max_grad_norm", "min_representation_std"):
        value = float(train.get(name, math.nan))
        if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(f"train.{name} 必须为有限{'非负' if name == 'weight_decay' else '正'}数")
    if not 0.0 < float(train["ema_momentum"]) < 1.0:
        raise ValueError("EMA momentum 必须位于 (0,1)")
    for name in ("lambda_ftv", "sigreg"):
        value = float(loss.get(name, -1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{name} 必须为有限非负数")
    step_weights = loss.get("step_weights")
    if not isinstance(step_weights, list) or len(step_weights) != 3 or any(float(value) <= 0 for value in step_weights):
        raise ValueError("loss.step_weights 必须为三个正数")
    expected_lambda = LOCKED_LAMBDA_FTV[model_name]
    if not math.isclose(float(loss["lambda_ftv"]), expected_lambda, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{model_name} 的 loss.lambda_ftv 必须锁定为 {expected_lambda}")


def _seed_contract(config: Mapping[str, Any], fold: int) -> tuple[int, int]:
    seed_base = config["train"].get("seed")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base not in SEED_BASES:
        raise ValueError(f"seed_base 必须是 {SEED_BASES}")
    effective_seed = seed_base + int(fold)
    if effective_seed != int(config["train"]["seed"]) + int(fold):
        raise AssertionError("effective_seed != seed_base + fold")
    return seed_base, effective_seed


def _validated_run_tag(run_name: str) -> str:
    tag = str(run_name).strip()
    path = Path(tag)
    if not tag or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"run_name 必须是实验目录内的安全相对路径: {run_name!r}")
    return path.as_posix()


def _validate_formal_protocol_lock(config: Mapping[str, Any]) -> None:
    for section, expected_values in FORMAL_PROTOCOL_LOCK.items():
        actual_values = config.get(section)
        if not isinstance(actual_values, Mapping):
            raise ValueError(f"formal config 缺 mapping: {section}")
        for key, expected in expected_values.items():
            if actual_values.get(key) != expected:
                raise ValueError(
                    f"formal protocol 锁冲突: {section}.{key}={actual_values.get(key)!r}, expected={expected!r}"
                )


def build_bundle(config: Mapping[str, Any]) -> CohortBundle:
    data = config["data"]
    fold_path = resolve_path(data["fold_manifest"])
    expected = str(data.get("fold_manifest_sha256", ""))
    if not expected or file_sha256(fold_path) != expected:
        raise ValueError("锁定 fold manifest SHA-256 不匹配")
    return load_cohort_bundle(
        resolve_path(data["primary_labels"]),
        resolve_path(data["extra_labels"]),
        fold_path,
        resolve_path(data["ftv_targets"]),
        resolve_path(data["cache_root"]),
        resolve_path(data["ftv_overlap"]) if data.get("ftv_overlap") else None,
    )


def ensure_ftv_transform(
    bundle: CohortBundle,
    fold: int,
    train_ids: Iterable[str],
    path: str | Path | None = None,
) -> PooledFTVTransform:
    train_ids = tuple(str(value) for value in train_ids)
    path = Path(path) if path is not None else EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
    expected_patient_hash = patient_hash(train_ids)
    expected_target_hash = raw_ftv_hash(bundle.raw_ftv)
    if path.exists():
        transform = PooledFTVTransform.load(path)
        if (
            transform.fold != fold
            or transform.train_patient_hash != expected_patient_hash
            or transform.raw_targets_sha256 != expected_target_hash
        ):
            raise ValueError(f"已有 FTV transform 与 fold/train/raw target 不一致: {path}")
        return transform
    transform = PooledFTVTransform.fit(bundle.raw_ftv, train_ids, fold)
    if transform.train_patient_hash != expected_patient_hash:
        raise AssertionError("FTV transform train patient hash 内部不一致")
    transform.save(path)
    return transform


def make_loader(
    records: list[PatientRecord],
    transformed_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]],
    raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = LongitudinalDGRSDataset(records, transformed_ftv, raw_ftv)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        # 与既有 G0/M0 训练器保持完全相同的 batch policy，避免把
        # DCE8→DCE7/mask ablation 与最后一个小 batch 的变化混在一起。
        drop_last=bool(shuffle and len(dataset) >= batch_size),
        generator=generator,
    )


class SIGReg(nn.Module):
    """与 prior M0 相同的 Sketch Isotropic Gaussian regularizer。"""

    def __init__(self, projections: int = 256, knots: int = 17) -> None:
        super().__init__()
        self.projections = int(projections)
        points = torch.linspace(0, 3, knots, dtype=torch.float32)
        interval = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * interval, dtype=torch.float32)
        weights[[0, -1]] = interval
        gaussian = torch.exp(-points.square() / 2)
        self.register_buffer("points", points)
        self.register_buffer("gaussian", gaussian)
        self.register_buffer("weights", weights * gaussian)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.float()
        directions = torch.randn(state.size(-1), self.projections, device=state.device)
        directions = directions / directions.norm(dim=0).clamp_min(1e-6)
        projected = (state @ directions).unsqueeze(-1) * self.points
        error = (projected.cos().mean(-3) - self.gaussian).square() + projected.sin().mean(-3).square()
        return ((error @ self.weights) * state.size(-2)).mean()


def patient_mean_ftv_loss(
    prediction: torch.Tensor | None,
    target: torch.Tensor,
    valid: torch.Tensor,
    differentiable_zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.shape != valid.shape or target.ndim != 2 or target.size(1) != 4:
        raise ValueError(f"FTV target/mask 期望 [B,4]；实际 {tuple(target.shape)}/{tuple(valid.shape)}")
    valid = valid.bool() & torch.isfinite(target)
    patient_valid = valid.any(dim=1)
    valid_visits = valid.sum()
    patient_count = patient_valid.sum()
    if prediction is None:
        zero_count = patient_count.new_zeros(())
        return differentiable_zero.sum() * 0.0, zero_count, valid_visits.new_zeros(())
    if not bool(patient_count):
        return differentiable_zero.sum() * 0.0, patient_count, valid_visits
    if prediction.shape != target.shape:
        raise ValueError(f"FTV prediction shape 不一致: {tuple(prediction.shape)}")
    safe_target = torch.where(valid, target, prediction.detach())
    element = F.smooth_l1_loss(prediction, safe_target, reduction="none") * valid.to(prediction.dtype)
    per_patient = element.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(prediction.dtype)
    return per_patient[patient_valid].mean(), patient_count, valid_visits


class DGRSObjective(nn.Module):
    def __init__(
        self,
        model_name: str,
        lambda_ftv: float,
        sigreg_weight: float = 0.09,
        sigreg_projections: int = 256,
        step_weights: tuple[float, float, float] = (2.0, 1.0, 0.5),
    ) -> None:
        super().__init__()
        self.model_name = str(model_name).upper()
        self.lambda_ftv = float(lambda_ftv) if self.model_name in GROUNDED_MODELS else 0.0
        self.sigreg_weight = float(sigreg_weight)
        weights = torch.tensor(step_weights, dtype=torch.float32)
        self.register_buffer("step_weights", weights / weights.mean())
        self.sigreg = SIGReg(sigreg_projections)

    def forward(
        self, output: DGRSOutput, ftv_target: torch.Tensor, ftv_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized_prediction = F.layer_norm(output.predicted_next, (output.predicted_next.size(-1),))
        normalized_target = F.layer_norm(output.target_next, (output.target_next.size(-1),))
        per_step = (normalized_prediction - normalized_target).square().mean(dim=-1)
        state_loss = (per_step * self.step_weights).mean()
        sigreg_loss = self.sigreg(output.online_state.transpose(0, 1))
        base_loss = state_loss + self.sigreg_weight * sigreg_loss
        ftv_loss, ftv_patients, ftv_visits = patient_mean_ftv_loss(
            output.ftv_prediction, ftv_target, ftv_mask, output.response_state
        )
        weighted_ftv = self.lambda_ftv * ftv_loss
        total = base_loss + weighted_ftv
        stats = {
            "loss": total.detach(),
            "base_loss": base_loss.detach(),
            "state_loss": state_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "ftv_loss": ftv_loss.detach(),
            "weighted_ftv_loss": weighted_ftv.detach(),
            "ftv_patients": ftv_patients.detach().to(torch.float32),
            "ftv_valid_visits": ftv_visits.detach().to(torch.float32),
            "_base_component": base_loss,
            "_ftv_component_raw": ftv_loss,
            "_ftv_component_weighted": weighted_ftv,
        }
        return total, stats


def _gradient_norm(parameters: Iterable[torch.nn.Parameter], from_grad: bool = True) -> float:
    tensors: list[torch.Tensor] = []
    for parameter in parameters:
        value = parameter.grad if from_grad else parameter
        if value is not None:
            tensors.append(value.detach().float())
    if not tensors:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in tensors)))


def _autograd_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    values = [value.detach().float() for value in grads if value is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in values)))


def run_epoch(
    model: DGRSWorldModel,
    objective: DGRSObjective,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ema_momentum: float,
    max_grad_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    objective.train(training)
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    batches = 0
    responses: list[torch.Tensor] = []
    base_component_audited = False
    ftv_component_audited = False
    component = {
        "first_valid_base_shared_gradient_norm": 0.0,
        "first_valid_ftv_encoder_gradient_norm_raw": 0.0,
        "first_valid_ftv_response_projection_gradient_norm_raw": 0.0,
        "first_valid_ftv_head_gradient_norm_raw": 0.0,
        "first_valid_ftv_encoder_gradient_norm_weighted": 0.0,
        "first_valid_ftv_response_projection_gradient_norm_weighted": 0.0,
        "first_valid_ftv_head_gradient_norm_weighted": 0.0,
    }
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            roi_mask = batch["roi_mask"].to(device, non_blocking=True) if model.requires_roi_mask else None
            ftv_target = batch["ftv_target"].to(device, non_blocking=True)
            ftv_mask = batch["ftv_mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(image, roi_mask)
            loss, stats = objective(output, ftv_target, ftv_mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"loss 非有限: {float(loss.detach())}")
            total_grad = encoder_grad = projection_grad = head_grad = 0.0
            if training:
                shared = tuple(
                    parameter
                    for module in (model.encoder, model.response_projection, model.projector, model.transition)
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
                if not base_component_audited:
                    grads = torch.autograd.grad(
                        stats["_base_component"], shared, retain_graph=True, allow_unused=True
                    )
                    component["first_valid_base_shared_gradient_norm"] = _autograd_norm(grads)
                    base_component_audited = True
                if (
                    not ftv_component_audited
                    and model.ftv_head is not None
                    and bool(ftv_mask.any())
                ):
                    encoder_parameters = tuple(model.encoder.parameters())
                    projection_parameters = tuple(model.response_projection.parameters())
                    head_parameters = tuple(model.ftv_head.parameters())
                    raw_encoder = _autograd_norm(
                        torch.autograd.grad(
                            stats["_ftv_component_raw"], encoder_parameters, retain_graph=True, allow_unused=True
                        )
                    )
                    raw_projection = _autograd_norm(
                        torch.autograd.grad(
                            stats["_ftv_component_raw"], projection_parameters, retain_graph=True, allow_unused=True
                        )
                    )
                    raw_head = _autograd_norm(
                        torch.autograd.grad(
                            stats["_ftv_component_raw"], head_parameters, retain_graph=True, allow_unused=True
                        )
                    )
                    component["first_valid_ftv_encoder_gradient_norm_raw"] = raw_encoder
                    component["first_valid_ftv_response_projection_gradient_norm_raw"] = raw_projection
                    component["first_valid_ftv_head_gradient_norm_raw"] = raw_head
                    component["first_valid_ftv_encoder_gradient_norm_weighted"] = objective.lambda_ftv * raw_encoder
                    component["first_valid_ftv_response_projection_gradient_norm_weighted"] = objective.lambda_ftv * raw_projection
                    component["first_valid_ftv_head_gradient_norm_weighted"] = objective.lambda_ftv * raw_head
                    ftv_component_audited = True
                loss.backward()
                encoder_grad = _gradient_norm(model.encoder.parameters())
                projection_grad = _gradient_norm(model.response_projection.parameters())
                head_grad = _gradient_norm(model.ftv_head.parameters()) if model.ftv_head is not None else 0.0
                total_grad = float(clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True))
                optimizer.step()
                model.update_target(ema_momentum)
            batch_size = image.size(0)
            samples += batch_size
            batches += 1
            responses.append(output.response_state.detach().float().cpu())
            for name in ("loss", "base_loss", "state_loss", "sigreg_loss"):
                sums[name] += float(stats[name]) * batch_size
            ftv_patients = float(stats["ftv_patients"])
            ftv_visits = float(stats["ftv_valid_visits"])
            sums["ftv_loss_sum"] += float(stats["ftv_loss"]) * ftv_patients
            sums["ftv_patients"] += ftv_patients
            sums["ftv_valid_visits"] += ftv_visits
            sums["ungrounded_patients"] += batch_size - ftv_patients
            sums["roi_valid_visits"] += float(batch["roi_valid"].sum())
            sums["roi_total_visits"] += float(batch["roi_valid"].numel())
            sums["gradient_norm"] += total_grad * batch_size
            sums["encoder_gradient_norm"] += encoder_grad * batch_size
            sums["response_projection_gradient_norm"] += projection_grad * batch_size
            sums["ftv_head_gradient_norm"] += head_grad * batch_size
    if not samples:
        raise RuntimeError("DataLoader 为空")
    response = torch.cat(responses, dim=0)
    representation_std = float(response.std(dim=0, unbiased=False).mean())
    ftv_patients = sums["ftv_patients"]
    ftv_loss = sums["ftv_loss_sum"] / max(ftv_patients, 1.0)
    output_stats = {
        name: sums[name] / samples
        for name in (
            "loss",
            "base_loss",
            "state_loss",
            "sigreg_loss",
            "gradient_norm",
            "encoder_gradient_norm",
            "response_projection_gradient_norm",
            "ftv_head_gradient_norm",
        )
    }
    output_stats.update(
        {
            "ftv_loss": ftv_loss,
            "weighted_ftv_loss": objective.lambda_ftv * ftv_loss,
            "grounded_patients": ftv_patients,
            "ungrounded_patients": sums["ungrounded_patients"],
            "valid_ftv_visits": sums["ftv_valid_visits"],
            "roi_valid_visits": sums["roi_valid_visits"],
            "roi_empty_visits": sums["roi_total_visits"] - sums["roi_valid_visits"],
            "representation_std": representation_std,
            "samples": float(samples),
            "batches": float(batches),
            "seconds": time.monotonic() - start,
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
            **component,
        }
    )
    # 用全 epoch 的 patient-weighted FTV 重新构造可解释总损失。
    output_stats["objective_reconstructed"] = output_stats["base_loss"] + output_stats["weighted_ftv_loss"]
    return output_stats


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _checkpoint_payload(
    model: DGRSWorldModel,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    run_name: str,
    fold: int,
    epoch: int,
    splits: Mapping[str, list[str]],
    transform_path: Path,
    shared_init_hash: str,
    history_path: Path,
    selection_path: Path,
    baseline: Mapping[str, Any],
    runtime: Mapping[str, Any],
    epoch_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    state = model.state_dict()
    seed_base = int(runtime["seed_base"])
    effective_seed = int(runtime["effective_seed"])
    if effective_seed != seed_base + int(fold):
        raise AssertionError("checkpoint seed contract 破坏: effective_seed != seed_base + fold")
    plan_path = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
    fold_path = resolve_path(config["data"]["fold_manifest"])
    model_config = model.model_config()
    return {
        "schema_version": 2,
        "run_name": run_name,
        "model_name": model.model_name,
        "seed_base": seed_base,
        "fold": fold,
        "effective_seed": effective_seed,
        "epoch": epoch,
        "state_dict": state,
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config,
        "train_config": dict(config["train"]),
        "loss_config": dict(config["loss"]),
        "architecture_contract": model.architecture_contract(),
        "splits": dict(splits),
        "split_hashes": {name: patient_hash(ids) for name, ids in splits.items()},
        "ftv_transform_path": str(transform_path.resolve()),
        "ftv_transform_sha256": file_sha256(transform_path),
        "history_path": str(history_path.resolve()),
        "selection_path": str(selection_path.resolve()),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
        "implementation_sha256": implementation_sha256(),
        "resolved_config_sha256": json_sha256(config),
        "shared_initialization_sha256": shared_init_hash,
        "baseline_selection_contract": dict(baseline),
        "selected_epoch_metrics": dict(epoch_metrics),
        "data_contract": {
            "fold_manifest": str(fold_path),
            "fold_manifest_sha256": file_sha256(fold_path),
            "cache_root": str(resolve_path(config["data"]["cache_root"])),
            "backbone_tensor": "x[:, :7]",
            "roi_mask_tensor": "x[:, 7:8] kept separate",
            "train_patient_hash": patient_hash(splits["train"]),
            "val_patient_hash": patient_hash(splits["val"]),
            "test_patient_hash": patient_hash(splits["test"]),
            "extra_pretrain_patient_hash": patient_hash(set(splits["pretrain_train"]) - set(splits["train"])),
            "raw_ftv_sha256": raw_ftv_hash(runtime["raw_ftv"]),
        },
        "runtime": {key: value for key, value in runtime.items() if key != "raw_ftv"},
        "git": git_metadata(),
        "source_commit": str(config.get("provenance", {}).get("source_commit", "unknown")),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "determinism": {
            "seed_base": seed_base,
            "fold": int(fold),
            "effective_seed": effective_seed,
            "seed": effective_seed,
            "shared_head_rng_isolation": True,
            "fixed_patient_order_seed": effective_seed,
            "no_random_augmentation": True,
            "cross_hardware_bitwise_reproducibility_claimed": False,
        },
    }


def _baseline_from_checkpoint(path: Path) -> tuple[float, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metric = payload.get("selected_epoch_metrics", {}).get("val_base_loss")
    if metric is None:
        raise ValueError(f"baseline checkpoint 缺 selected val_base_loss: {path}")
    return float(metric), payload


def _validate_paired_baseline_contract(
    payload: Mapping[str, Any],
    path: Path,
    config: Mapping[str, Any],
    model: DGRSWorldModel,
    splits: Mapping[str, list[str]],
    train_ids: list[str],
    val_ids: list[str],
    smoke: bool,
    transform_path: Path,
    raw_target_sha256: str,
    seed_base: int,
    fold: int,
    effective_seed: int,
) -> None:
    """拒绝把 smoke、旧代码、异参或错 patient baseline 用于 5% gate。"""

    if not bool(payload.get("finalized")) or int(payload.get("schema_version", -1)) != 2:
        raise ValueError(f"paired baseline 必须是 finalized schema-v2 checkpoint: {path}")
    expected_seed_contract = {
        "seed_base": int(seed_base),
        "fold": int(fold),
        "effective_seed": int(effective_seed),
    }
    if int(payload.get("seed_base", -1)) != expected_seed_contract["seed_base"]:
        raise ValueError("paired baseline seed_base 不一致")
    if int(payload.get("effective_seed", -1)) != expected_seed_contract["effective_seed"]:
        raise ValueError("paired baseline effective_seed 不一致")
    if int(payload.get("fold", -1)) != expected_seed_contract["fold"]:
        raise ValueError("paired baseline fold 不一致")
    if str(payload.get("implementation_sha256", "")) != implementation_sha256():
        raise ValueError("paired baseline 与当前训练实现 SHA-256 不一致")
    plan_path = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
    if str(payload.get("plan_sha256", "")) != file_sha256(plan_path):
        raise ValueError("paired baseline 与当前预注册计划 SHA-256 不一致")

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("paired baseline 缺 runtime contract")
    expected_runtime = {
        "seed_base": int(seed_base),
        "fold": int(fold),
        "effective_seed": int(effective_seed),
        "seed": int(effective_seed),
        "smoke": bool(smoke),
        "effective_train_patient_hash": patient_hash(train_ids),
        "effective_validation_patient_hash": patient_hash(val_ids),
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise ValueError(f"paired baseline runtime.{key} 不一致")

    split_hashes = payload.get("split_hashes")
    if not isinstance(split_hashes, Mapping):
        raise ValueError("paired baseline 缺 split hashes")
    for name in ("train", "val", "test", "pretrain_train"):
        if str(split_hashes.get(name, "")) != patient_hash(splits[name]):
            raise ValueError(f"paired baseline canonical {name} patient hash 不一致")
    if str(payload.get("ftv_transform_sha256", "")) != file_sha256(transform_path):
        raise ValueError("paired baseline FTV transform hash 不一致")
    data_contract = payload.get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise ValueError("paired baseline 缺 data contract")
    expected_data = {
        "fold_manifest_sha256": file_sha256(resolve_path(config["data"]["fold_manifest"])),
        "cache_root": str(resolve_path(config["data"]["cache_root"])),
        "raw_ftv_sha256": raw_target_sha256,
    }
    for key, expected in expected_data.items():
        if str(data_contract.get(key, "")) != str(expected):
            raise ValueError(f"paired baseline data_contract.{key} 不一致")

    baseline_model = dict(payload.get("model_config", {}))
    candidate_model = model.model_config()
    for key in ("model_name", "direct_ftv_grounding"):
        baseline_model.pop(key, None)
        candidate_model.pop(key, None)
    if baseline_model != candidate_model:
        raise ValueError("paired baseline 共同 model config 不一致")

    baseline_train = dict(payload.get("train_config", {}))
    if baseline_train != dict(config["train"]):
        raise ValueError("paired baseline train config 不一致")
    baseline_loss = dict(payload.get("loss_config", {}))
    candidate_loss = dict(config["loss"])
    baseline_loss.pop("lambda_ftv", None)
    candidate_loss.pop("lambda_ftv", None)
    if baseline_loss != candidate_loss:
        raise ValueError("paired baseline 共同 loss config 不一致")

    selected = payload.get("selected_epoch_metrics")
    if not isinstance(selected, Mapping) or not all(
        bool(selected.get(key)) for key in ("noncollapse", "base_gate_pass", "eligible")
    ):
        raise ValueError("paired baseline selected epoch 不合格")
    selection_path = Path(str(payload.get("selection_path", "")))
    history_path = Path(str(payload.get("history_path", "")))
    if not selection_path.is_file() or file_sha256(selection_path) != str(payload.get("selection_sha256", "")):
        raise ValueError("paired baseline selection 文件/哈希不一致")
    if not history_path.is_file() or file_sha256(history_path) != str(payload.get("history_sha256", "")):
        raise ValueError("paired baseline history 文件/哈希不一致")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if bool(selection.get("test_data_used")) or selection.get("selection_mode") != "primary":
        raise ValueError("paired baseline selection 使用 test 或不是 primary")


def _smoke_subset(
    ids: list[str],
    raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]],
    size: int,
    seed: int,
) -> list[str]:
    size = min(max(2, int(size)), len(ids))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=size, replace=False).tolist()
    paired = [patient_id for patient_id in ids if patient_id in raw_ftv]
    if paired and not set(chosen).intersection(paired):
        chosen[0] = paired[0]
    return chosen


def _finalize_checkpoint(path: Path, history_path: Path, selection_path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["history_sha256"] = file_sha256(history_path)
    payload["selection_sha256"] = file_sha256(selection_path)
    payload["finalized"] = True
    torch.save(payload, path)


def export_pilot_features(
    checkpoint_path: str | Path,
    bundle: CohortBundle,
    transform: PooledFTVTransform,
    splits: Mapping[str, list[str]],
    device: torch.device,
    batch_size: int,
    workers: int,
    output_path: str | Path,
) -> Path:
    """只导出 canonical outer train+val；代码路径不接触 test IDs。"""

    from .model import load_checkpoint_for_evaluation

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖 pilot feature: {output_path}")
    train_ids, val_ids = list(splits["train"]), list(splits["val"])
    if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & set(splits["test"]):
        raise ValueError("pilot export split 泄漏")
    ids = train_ids + val_ids
    transformed = transform.transform_all(bundle.raw_ftv)
    loader = make_loader(
        records_for_ids(bundle, ids), transformed, bundle.raw_ftv, batch_size, workers, False, 0
    )
    model, payload = load_checkpoint_for_evaluation(checkpoint_path, device)
    responses: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []
    standardized: list[np.ndarray] = []
    valid_values: list[np.ndarray] = []
    patient_ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            roi_mask = batch["roi_mask"].to(device, non_blocking=True) if model.requires_roi_mask else None
            responses.append(model.encode_response(image, roi_mask).cpu().numpy().astype(np.float32))
            raw_values.append(batch["ftv_raw"].numpy().astype(np.float32))
            standardized.append(batch["ftv_target"].numpy().astype(np.float32))
            valid_values.append(batch["ftv_mask"].numpy().astype(bool))
            patient_ids.extend(str(value) for value in batch["patient_id"])
    response = np.concatenate(responses)
    if response.shape != (len(ids), 4, model.latent_dim) or patient_ids != ids:
        raise AssertionError("pilot feature patient order/shape 漂移")
    split_labels = np.asarray(["train"] * len(train_ids) + ["val"] * len(val_ids))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                response=response,
                patient_ids=np.asarray(patient_ids),
                splits=split_labels,
                ftv_raw=np.concatenate(raw_values),
                ftv_standardized=np.concatenate(standardized),
                ftv_valid=np.concatenate(valid_values),
                val_base_loss=np.asarray(float(payload["selected_epoch_metrics"]["val_base_loss"])),
                val_state_loss=np.asarray(float(payload["selected_epoch_metrics"]["val_state_loss"])),
                representation_std=np.asarray(float(response.std(axis=0).mean())),
                checkpoint=np.asarray(str(checkpoint_path.resolve())),
                checkpoint_sha256=np.asarray(file_sha256(checkpoint_path)),
                checkpoint_implementation_sha256=np.asarray(str(payload["implementation_sha256"])),
                checkpoint_run_name=np.asarray(str(payload["run_name"])),
                fold_manifest_sha256=np.asarray(str(payload["data_contract"]["fold_manifest_sha256"])),
                canonical_train_patient_hash=np.asarray(patient_hash(train_ids)),
                canonical_val_patient_hash=np.asarray(patient_hash(val_ids)),
                canonical_test_patient_hash=np.asarray(patient_hash(splits["test"])),
                transform_path=np.asarray(str(payload["ftv_transform_path"])),
                transform_sha256=np.asarray(file_sha256(payload["ftv_transform_path"])),
                model=np.asarray(model.model_name),
                model_name=np.asarray(model.model_name),
                lambda_ftv=np.asarray(float(payload["loss_config"]["lambda_ftv"])),
                seed_base=np.asarray(int(payload["seed_base"])),
                fold=np.asarray(int(payload["fold"])),
                effective_seed=np.asarray(int(payload["effective_seed"])),
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def train_fold(
    config: Mapping[str, Any],
    run_name: str,
    fold: int,
    model_name: str | None = None,
    device_name: str = "cuda",
    epochs_override: int | None = None,
    smoke_patients: int | None = None,
    lambda_ftv_override: float | None = None,
    baseline_val_base_loss: float | None = None,
    baseline_checkpoint: str | Path | None = None,
    workers_override: int | None = None,
    export_pilot: bool = False,
) -> Path:
    if fold not in range(5):
        raise ValueError("fold 必须为 0–4")
    config = copy.deepcopy(dict(config))
    config["model"] = dict(config["model"])
    config["loss"] = dict(config["loss"])
    config["train"] = dict(config["train"])
    if model_name is not None:
        config["model"]["model_name"] = str(model_name).upper()
    if workers_override is not None:
        config["train"]["workers"] = int(workers_override)
    if epochs_override is not None:
        config["train"]["epochs"] = int(epochs_override)
    model_name = str(config["model"]["model_name"]).upper()
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"本实验只允许训练 {ALLOWED_MODELS}")
    lambda_ftv = LOCKED_LAMBDA_FTV[model_name]
    if lambda_ftv_override is not None and not math.isclose(
        float(lambda_ftv_override), lambda_ftv, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(f"{model_name} 的 lambda_ftv 已锁定为 {lambda_ftv}，拒绝 override")
    config["loss"]["lambda_ftv"] = lambda_ftv
    validate_run_config(config)
    seed_base, effective_seed = _seed_contract(config, fold)
    tag = _validated_run_tag(run_name)
    if tag.startswith("formal/"):
        _validate_formal_protocol_lock(config)
        expected_tag = f"formal/seed_{seed_base}/{model_name.lower()}"
        if tag != expected_tag:
            raise ValueError(f"formal run_name 必须为 {expected_tag}")
        if any(
            value is not None
            for value in (epochs_override, workers_override, smoke_patients, lambda_ftv_override)
        ) or export_pilot:
            raise ValueError("formal run 禁止 epochs/workers/smoke/lambda/pilot override")
        if baseline_val_base_loss is not None:
            raise ValueError("formal G3 必须只通过显式 paired G1 checkpoint 取得 baseline metric")
        expected_baseline = (
            EXPERIMENT_ROOT
            / "checkpoints"
            / "formal"
            / f"seed_{seed_base}"
            / "g1"
            / f"fold_{fold}"
            / "best.pt"
        ).resolve()
        if model_name == "G3" and (
            baseline_checkpoint is None or Path(baseline_checkpoint).resolve() != expected_baseline
        ):
            raise ValueError(f"formal G3 baseline checkpoint 必须为 {expected_baseline}")

    baseline_payload: dict[str, Any] | None = None
    baseline_path: Path | None = Path(baseline_checkpoint).resolve() if baseline_checkpoint else None
    if baseline_path is not None:
        checkpoint_metric, baseline_payload = _baseline_from_checkpoint(baseline_path)
        if baseline_val_base_loss is not None and not math.isclose(
            float(baseline_val_base_loss), checkpoint_metric, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("--baseline-val-base-loss 与 baseline checkpoint 不一致")
        baseline_val_base_loss = checkpoint_metric
        if str(baseline_payload.get("model_name")) != "G1":
            raise ValueError(f"{model_name} paired baseline checkpoint 模型错误")
        if int(baseline_payload.get("fold", -1)) != fold:
            raise ValueError("paired baseline checkpoint fold 不一致")
        if int(baseline_payload.get("seed_base", -1)) != seed_base:
            raise ValueError("paired baseline checkpoint seed_base 不一致")
        if int(baseline_payload.get("effective_seed", -1)) != effective_seed:
            raise ValueError("paired baseline checkpoint effective_seed 不一致")
    if model_name == "G3":
        if baseline_val_base_loss is None or not math.isfinite(float(baseline_val_base_loss)) or float(baseline_val_base_loss) <= 0:
            raise ValueError(f"{model_name} 必须提供正的 paired --baseline-val-base-loss 或 --baseline-checkpoint")
    elif baseline_val_base_loss is not None or baseline_path is not None:
        raise ValueError("G1 baseline 训练不接受 baseline selection 参数")

    bundle = build_bundle(config)
    splits = split_ids(bundle, fold)
    transform_path = EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
    transform = ensure_ftv_transform(bundle, fold, splits["train"], transform_path)
    transformed = transform.transform_all(bundle.raw_ftv)
    set_seed(effective_seed)
    model_kwargs = dict(config["model"])
    model = DGRSWorldModel(**model_kwargs)
    shared_init_hash = shared_initialization_sha256(model)
    if baseline_payload is not None:
        baseline_hash = str(baseline_payload.get("shared_initialization_sha256", ""))
        if baseline_hash != shared_init_hash:
            raise ValueError("grounded model 与 paired baseline shared initialization hash 不一致")
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    model.to(device)
    objective = DGRSObjective(
        model_name=model_name,
        lambda_ftv=lambda_ftv,
        sigreg_weight=float(config["loss"]["sigreg"]),
        sigreg_projections=int(config["loss"]["sigreg_projections"]),
        step_weights=tuple(float(value) for value in config["loss"]["step_weights"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    train_ids = list(splits["pretrain_train"])
    val_ids = list(splits["val"])
    if smoke_patients is not None:
        train_ids = _smoke_subset(list(splits["train"]), bundle.raw_ftv, smoke_patients, effective_seed)
        val_ids = _smoke_subset(
            list(splits["val"]), bundle.raw_ftv, max(2, smoke_patients // 2), effective_seed + 1
        )
    if baseline_payload is not None and baseline_path is not None:
        _validate_paired_baseline_contract(
            baseline_payload,
            baseline_path,
            config,
            model,
            splits,
            train_ids,
            val_ids,
            smoke_patients is not None,
            transform_path,
            raw_ftv_hash(bundle.raw_ftv),
            seed_base,
            fold,
            effective_seed,
        )
    batch_size = int(config["train"]["batch_size"])
    workers = int(config["train"]["workers"])
    train_loader = make_loader(
        records_for_ids(bundle, train_ids),
        transformed,
        bundle.raw_ftv,
        batch_size,
        workers,
        True,
        effective_seed,
    )
    val_loader = make_loader(
        records_for_ids(bundle, val_ids),
        transformed,
        bundle.raw_ftv,
        batch_size,
        workers,
        False,
        effective_seed,
    )

    if smoke_patients is not None and not tag.startswith("smoke_"):
        tag = f"smoke_{tag}"
    output_dir = EXPERIMENT_ROOT / "checkpoints" / tag / f"fold_{fold}"
    metric_dir = EXPERIMENT_ROOT / "metrics" / "training" / tag
    best_path = output_dir / "best.pt"
    fallback_path = output_dir / "fallback.pt"
    last_path = output_dir / "last.pt"
    resolved_path = output_dir / "resolved_run.json"
    selection_path = output_dir / "selection.json"
    claim_path = output_dir / "RUN_CLAIM.json"
    history_path = metric_dir / f"fold_{fold}.csv"
    protected = (best_path, fallback_path, last_path, resolved_path, selection_path, claim_path, history_path)
    if any(path.exists() for path in protected):
        raise FileExistsError(f"拒绝覆盖已有 run；更换 --run-name: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"同名 run 已被占用: {output_dir}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "claimed_at_unix": time.time(),
                "run_name": tag,
                "model_name": model_name,
                "seed_base": seed_base,
                "fold": fold,
                "effective_seed": effective_seed,
            },
            stream,
        )

    baseline_contract = {
        "paired_model": "G1" if model_name == "G3" else None,
        "base_metric": "validation_normalized_next_state_loss_without_sigreg",
        "baseline_checkpoint": str(baseline_path) if baseline_path else None,
        "baseline_checkpoint_sha256": file_sha256(baseline_path) if baseline_path else None,
        "baseline_val_base_loss": baseline_val_base_loss,
        "maximum_relative_degradation": 0.05 if model_name in GROUNDED_MODELS else None,
        "allowed_val_base_loss": (
            float(baseline_val_base_loss) * 1.05 if baseline_val_base_loss is not None else None
        ),
    }
    runtime: dict[str, Any] = {
        "seed_base": seed_base,
        "fold": fold,
        "effective_seed": effective_seed,
        # 保留 seed 别名仅为向后兼容；其语义明确是 effective_seed。
        "seed": effective_seed,
        "device": str(device),
        "smoke": smoke_patients is not None,
        "smoke_patients_requested": smoke_patients,
        "effective_train_ids": train_ids,
        "effective_train_patient_hash": patient_hash(train_ids),
        "effective_validation_ids": val_ids,
        "effective_validation_patient_hash": patient_hash(val_ids),
        "transform_fit_ids": splits["train"],
        "raw_ftv": bundle.raw_ftv,
    }
    plan_path = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
    current_git = git_metadata()
    atomic_json(
        resolved_path,
        {
            "run_name": tag,
            "seed_base": seed_base,
            "fold": fold,
            "effective_seed": effective_seed,
            "model_name": model_name,
            "seed": effective_seed,
            "lambda_ftv": lambda_ftv,
            "train_samples": len(train_ids),
            "validation_samples": len(val_ids),
            "smoke_patients": smoke_patients,
            "epochs_requested": int(config["train"]["epochs"]),
            "device": str(device),
            "model_config": model.model_config(),
            "architecture_contract": model.architecture_contract(),
            "shared_initialization_sha256": shared_init_hash,
            "baseline_selection_contract": baseline_contract,
            "ftv_transform_path": str(transform_path.resolve()),
            "ftv_transform_sha256": file_sha256(transform_path),
            "experiment_plan_path": str(plan_path.resolve()),
            "experiment_plan_sha256": file_sha256(plan_path),
            "source_commit": str(config.get("provenance", {}).get("source_commit", "unknown")),
            "current_commit": current_git["commit"],
            "implementation_sha256": implementation_sha256(),
        },
    )

    min_std = float(config["train"]["min_representation_std"])
    patience = int(config["train"]["patience"])
    history: list[dict[str, Any]] = []
    selection_evidence: list[dict[str, Any]] = []
    best_metric = math.inf
    fallback_metric: tuple[float, float] = (math.inf, math.inf)
    selected_epoch: int | None = None
    fallback_epoch: int | None = None
    stale = 0
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_stats = run_epoch(
            model,
            objective,
            train_loader,
            device,
            optimizer,
            float(config["train"]["ema_momentum"]),
            float(config["train"]["max_grad_norm"]),
        )
        val_stats = run_epoch(
            model,
            objective,
            val_loader,
            device,
            None,
            float(config["train"]["ema_momentum"]),
            float(config["train"]["max_grad_norm"]),
        )
        noncollapse = math.isfinite(float(val_stats["representation_std"])) and float(val_stats["representation_std"]) >= min_std
        # Protocol gate/selection follows prior M0's normalized next-state loss.
        # SIGReg remains in the training objective but its random projections do
        # not inject noise into checkpoint selection.
        val_base_loss = float(val_stats["state_loss"])
        val_base_objective = float(val_stats["base_loss"])
        base_gate = (
            True
            if model_name not in GROUNDED_MODELS
            else val_base_loss <= float(baseline_contract["allowed_val_base_loss"])
        )
        ftv_finite = math.isfinite(float(val_stats["ftv_loss"])) and float(val_stats["grounded_patients"]) > 0
        eligible = noncollapse and base_gate and (ftv_finite if model_name in GROUNDED_MODELS else True)
        metric = float(val_stats["ftv_loss"] if model_name in GROUNDED_MODELS else val_base_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        row: dict[str, Any] = {
            "epoch": epoch,
            "seed_base": seed_base,
            "fold": fold,
            "effective_seed": effective_seed,
            "model_name": model_name,
            "model": model_name,
            "lambda_ftv": lambda_ftv,
            "learning_rate": lr,
            "total_loss": float(train_stats["loss"]),
            "base_loss": float(train_stats["base_loss"]),
            "ftv_loss": float(train_stats["ftv_loss"]),
            "weighted_ftv_loss": float(train_stats["weighted_ftv_loss"]),
            "val_base_loss": val_base_loss,
            "val_base_objective": val_base_objective,
            "val_ftv_metric": float(val_stats["ftv_loss"]),
            "val_representation_std": float(val_stats["representation_std"]),
            "representation_std": float(val_stats["representation_std"]),
            "encoder_grad_norm": float(train_stats["encoder_gradient_norm"]),
            "ftv_head_grad_norm": float(train_stats["ftv_head_gradient_norm"]),
            "grounded_patients": float(train_stats["grounded_patients"]),
            "grounded_visits": float(train_stats["valid_ftv_visits"]),
            "ungrounded_patients": float(train_stats["ungrounded_patients"]),
            "val_ftv_loss": float(val_stats["ftv_loss"]),
            "noncollapse": noncollapse,
            "base_gate_pass": base_gate,
            "checkpoint_eligible": eligible,
            "validation_selection_metric": metric,
            "is_selected_checkpoint": False,
        }
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in val_stats.items()})
        # ``val_stats['base_loss']`` 含 SIGReg，而预注册的 base gate 使用纯
        # normalized next-state ``state_loss``。保留两个清楚命名的字段，
        # 避免上面的机械前缀覆盖 gate/聚合器读取的 ``val_base_loss``。
        row["val_base_loss"] = val_base_loss
        row["val_base_objective"] = val_base_objective
        history.append(row)
        epoch_metrics = {
            "epoch": epoch,
            "seed_base": seed_base,
            "fold": fold,
            "effective_seed": effective_seed,
            "val_base_loss": val_base_loss,
            "val_base_objective": val_base_objective,
            "val_state_loss": float(val_stats["state_loss"]),
            "val_sigreg_loss": float(val_stats["sigreg_loss"]),
            "val_ftv_loss": float(val_stats["ftv_loss"]),
            "val_representation_std": float(val_stats["representation_std"]),
            "noncollapse": noncollapse,
            "base_gate_pass": base_gate,
            "eligible": eligible,
        }
        selection_evidence.append(epoch_metrics)
        payload = _checkpoint_payload(
            model,
            optimizer,
            config,
            tag,
            fold,
            epoch,
            splits,
            transform_path,
            shared_init_hash,
            history_path,
            selection_path,
            baseline_contract,
            runtime,
            epoch_metrics,
        )
        improved = False
        if eligible and metric < best_metric:
            best_metric = metric
            selected_epoch = epoch
            torch.save(payload, best_path)
            improved = True
        if noncollapse and ftv_finite:
            base_violation = max(
                0.0,
                val_base_loss - float(baseline_contract["allowed_val_base_loss"]),
            ) if model_name in GROUNDED_MODELS else 0.0
            current_fallback = (base_violation, metric)
            if current_fallback < fallback_metric:
                fallback_metric = current_fallback
                fallback_epoch = epoch
                torch.save(payload, fallback_path)
                if selected_epoch is None:
                    improved = True
        torch.save(payload, last_path)
        _write_history(history_path, history)
        stale = 0 if improved else stale + 1
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seed_base": seed_base,
                    "fold": fold,
                    "effective_seed": effective_seed,
                    "model": model_name,
                    "lambda_ftv": lambda_ftv,
                    "train_base": train_stats["base_loss"],
                    "train_ftv": train_stats["ftv_loss"],
                    "val_base": val_base_loss,
                    "val_base_objective": val_base_objective,
                    "val_ftv": val_stats["ftv_loss"],
                    "val_rep_std": val_stats["representation_std"],
                    "base_gate": base_gate,
                    "eligible": eligible,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stale >= patience:
            break

    selection_mode = "primary"
    if selected_epoch is None:
        if fallback_epoch is None or not fallback_path.exists():
            raise RuntimeError("没有 finite/non-collapse checkpoint 可用")
        shutil.copy2(fallback_path, best_path)
        selected_epoch = fallback_epoch
        selection_mode = "fallback_base_gate_failed"
    selected_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    selected_metrics = dict(selected_payload["selected_epoch_metrics"])
    for row in history:
        row["is_selected_checkpoint"] = int(row["epoch"]) == int(selected_epoch)
    _write_history(history_path, history)
    selection_payload = {
        "schema_version": 1,
        "run_name": tag,
        "seed_base": seed_base,
        "fold": fold,
        "effective_seed": effective_seed,
        "model_name": model_name,
        "selection_mode": selection_mode,
        "selection_rule": (
            "non-collapse epoch with minimum validation normalized next-state loss"
            if model_name not in GROUNDED_MODELS
            else "non-collapse and <=5% paired-baseline normalized next-state loss degradation; minimum validation FTV loss"
        ),
        "fallback_rule": "minimum normalized-next-state gate violation, then minimum validation FTV loss among non-collapse finite epochs",
        "selected_epoch": selected_epoch,
        "selected_validation_base_loss": selected_metrics["val_base_loss"],
        "selected_validation_ftv_loss": selected_metrics["val_ftv_loss"],
        "selected_representation_std": selected_metrics["val_representation_std"],
        "baseline_selection_contract": baseline_contract,
        "epochs": selection_evidence,
        "test_data_used": False,
    }
    atomic_json(selection_path, selection_payload)
    _finalize_checkpoint(best_path, history_path, selection_path)
    _finalize_checkpoint(last_path, history_path, selection_path)
    if fallback_path.exists():
        _finalize_checkpoint(fallback_path, history_path, selection_path)
    if export_pilot:
        if fold != 0:
            raise ValueError("pilot feature 只允许 fold 0")
        pilot_path = EXPERIMENT_ROOT / "metrics" / "lambda_pilot" / tag / "fold_0_train_val_features.npz"
        export_pilot_features(
            best_path,
            bundle,
            transform,
            splits,
            device,
            batch_size,
            workers,
            pilot_path,
        )
    return best_path


__all__ = [
    "DGRSObjective",
    "SIGReg",
    "build_bundle",
    "ensure_ftv_transform",
    "export_pilot_features",
    "make_loader",
    "records_for_ids",
    "run_epoch",
    "set_seed",
    "split_ids",
    "train_fold",
]
