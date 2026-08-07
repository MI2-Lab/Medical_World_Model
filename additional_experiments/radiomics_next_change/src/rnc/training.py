"""五折 image-only M0/M1/M2 训练。"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .config import EXPERIMENT_ROOT, resolve_path, save_json
from .data import (
    CohortBundle,
    LongitudinalCacheDataset,
    PatientRecord,
    load_cohort_bundle,
    patient_hash,
    split_ids,
)
from .losses import NextChangeObjective
from .model import ImageOnlyWorldModel
from .transforms import RadiomicsChangeTransform, raw_targets_hash


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=EXPERIMENT_ROOT.parents[1], text=True).strip()

    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "experiment_status": run(
            "git", "status", "--short", "--", "additional_experiments/radiomics_next_change"
        ),
    }


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    training_sources = (
        "__init__.py",
        "config.py",
        "data.py",
        "transforms.py",
        "model.py",
        "losses.py",
        "training.py",
    )
    paths = [EXPERIMENT_ROOT / "src" / "rnc" / name for name in training_sources]
    paths.append(EXPERIMENT_ROOT / "scripts" / "train_model.py")
    for path in paths:
        digest.update(str(path.relative_to(EXPERIMENT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_bundle(config: dict[str, Any]) -> CohortBundle:
    data = config["data"]
    fold_path = resolve_path(data["fold_manifest"])
    expected_fold_sha = str(data.get("fold_manifest_sha256", ""))
    if not expected_fold_sha or file_sha256(fold_path) != expected_fold_sha:
        raise ValueError("候选五折 manifest SHA256 与锁定配置不一致")
    return load_cohort_bundle(
        resolve_path(data["primary_labels"]),
        resolve_path(data["extra_labels"]),
        fold_path,
        resolve_path(data["radiomics_overlap"]),
        resolve_path(data["radiomics_targets"]),
        resolve_path(data["cache_root"]),
    )


def validate_run_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "loss", "train"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"配置缺少 mapping: {section}")
    model, loss, train = config["model"], config["loss"], config["train"]
    if model.get("mode") not in ImageOnlyWorldModel.VALID_MODES:
        raise ValueError("model.mode 非法")
    if int(model.get("radiomics_dim", -1)) != 4:
        raise ValueError("radiomics_dim 必须为 4")
    for name in ("image_channels", "base_channels", "latent_dim", "predictor_depth", "predictor_heads", "predictor_mlp_dim"):
        if isinstance(model.get(name), bool) or int(model.get(name, 0)) <= 0:
            raise ValueError(f"model.{name} 必须为正整数")
    if int(model["image_channels"]) not in (7, 8):
        raise ValueError("image_channels 必须为 7 或 8")
    if not 0.0 <= float(model.get("dropout", -1.0)) < 1.0:
        raise ValueError("model.dropout 必须位于 [0,1)")
    for name in ("batch_size", "epochs", "patience"):
        if isinstance(train.get(name), bool) or int(train.get(name, 0)) <= 0:
            raise ValueError(f"train.{name} 必须为正整数")
    if isinstance(train.get("workers"), bool) or int(train.get("workers", -1)) < 0:
        raise ValueError("train.workers 必须为非负整数")
    for name in ("learning_rate", "ema_momentum", "max_grad_norm", "min_latent_std"):
        if not math.isfinite(float(train.get(name, math.nan))) or float(train[name]) <= 0:
            raise ValueError(f"train.{name} 必须为有限正数")
    if not 0.0 < float(train["ema_momentum"]) < 1.0:
        raise ValueError("ema_momentum 必须位于 (0,1)")
    if float(loss.get("lambda_rad", -1.0)) < 0 or float(loss.get("sigreg", -1.0)) < 0:
        raise ValueError("lambda_rad/sigreg 必须非负")
    step_weights = loss.get("step_weights")
    if not isinstance(step_weights, list) or len(step_weights) != 3 or any(float(value) <= 0 for value in step_weights):
        raise ValueError("loss.step_weights 必须是三个正数")


def ensure_radiomics_transform(
    bundle: CohortBundle,
    fold: int,
    train_ids: Iterable[str],
    output_path: Path | None = None,
) -> RadiomicsChangeTransform:
    train_ids = tuple(str(value) for value in train_ids)
    output_path = output_path or EXPERIMENT_ROOT / "configs" / f"radiomics_transform_fold_{fold}.json"
    expected_hash = patient_hash(train_ids)
    if output_path.exists():
        transform = RadiomicsChangeTransform.load(output_path)
        if (
            transform.fold != fold
            or transform.train_patient_hash != expected_hash
            or transform.raw_targets_sha256 != raw_targets_hash(bundle.raw_radiomics)
        ):
            raise ValueError(f"已有 transform 与 fold/train IDs 不一致: {output_path}")
        return transform
    transform = RadiomicsChangeTransform.fit(bundle.raw_radiomics, train_ids, fold)
    if transform.train_patient_hash != expected_hash:
        raise AssertionError("transform train hash 内部不一致")
    transform.save(output_path)
    return transform


def records_for_ids(bundle: CohortBundle, patient_ids: Iterable[str]) -> list[PatientRecord]:
    lookup = bundle.by_id
    ids = [str(value) for value in patient_ids]
    missing = [patient_id for patient_id in ids if patient_id not in lookup]
    if missing:
        raise KeyError(f"未知 patient IDs: {missing[:5]}")
    return [lookup[patient_id] for patient_id in ids]


def make_loader(
    records: list[PatientRecord],
    transformed_radiomics: dict[str, tuple[np.ndarray, np.ndarray]],
    image_channels: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = LongitudinalCacheDataset(records, transformed_radiomics, image_channels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
        generator=generator,
    )


def run_epoch(
    model: ImageOnlyWorldModel,
    objective: NextChangeObjective,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ema_momentum: float,
    max_grad_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    objective.train(training)
    sums: dict[str, float] = defaultdict(float)
    total_samples = 0
    total_batches = 0
    start_time = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    context = torch.enable_grad() if training else torch.no_grad()
    component_gradient_stats = {
        "first_batch_image_task_gradient_norm": 0.0,
        "first_batch_radiomics_shared_gradient_norm_raw": 0.0,
        "first_batch_radiomics_head_gradient_norm_raw": 0.0,
        "first_batch_radiomics_shared_gradient_norm_weighted": 0.0,
        "first_batch_radiomics_head_gradient_norm_weighted": 0.0,
        "first_batch_weighted_radiomics_to_image_gradient_ratio": 0.0,
    }

    def gradient_norm_from(grads: tuple[torch.Tensor | None, ...]) -> float:
        finite = [gradient.detach().float() for gradient in grads if gradient is not None]
        if not finite:
            return 0.0
        return float(torch.sqrt(sum(gradient.square().sum() for gradient in finite)))

    with context:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            radiomics_target = batch["radiomics_target"].to(device, non_blocking=True)
            radiomics_mask = batch["radiomics_mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(image)
            loss, stats = objective(output, radiomics_target, radiomics_mask)
            gradient_norm = 0.0
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"训练 loss 非有限: {float(loss.detach())}")
                if total_batches == 0:
                    shared_parameters = tuple(
                        parameter
                        for module in (model.encoder, model.projector, model.transition)
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    )
                    image_grads = torch.autograd.grad(
                        stats["_image_component"],
                        shared_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    component_gradient_stats["first_batch_image_task_gradient_norm"] = gradient_norm_from(
                        image_grads
                    )
                    if model.radiomics_head is not None and bool(radiomics_mask.any()):
                        rad_shared_grads = torch.autograd.grad(
                            stats["_radiomics_component"],
                            shared_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        head_parameters = tuple(
                            parameter for parameter in model.radiomics_head.parameters() if parameter.requires_grad
                        )
                        rad_head_grads = torch.autograd.grad(
                            stats["_radiomics_component"],
                            head_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        raw_shared = gradient_norm_from(rad_shared_grads)
                        raw_head = gradient_norm_from(rad_head_grads)
                        weighted_shared = objective.weights.radiomics * raw_shared
                        weighted_head = objective.weights.radiomics * raw_head
                        component_gradient_stats["first_batch_radiomics_shared_gradient_norm_raw"] = raw_shared
                        component_gradient_stats["first_batch_radiomics_head_gradient_norm_raw"] = raw_head
                        component_gradient_stats[
                            "first_batch_radiomics_shared_gradient_norm_weighted"
                        ] = weighted_shared
                        component_gradient_stats[
                            "first_batch_radiomics_head_gradient_norm_weighted"
                        ] = weighted_head
                        component_gradient_stats[
                            "first_batch_weighted_radiomics_to_image_gradient_ratio"
                        ] = weighted_shared / max(
                            component_gradient_stats["first_batch_image_task_gradient_norm"], 1e-12
                        )
                loss.backward()
                gradient_norm = float(
                    clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
                )
                optimizer.step()
                model.update_target(ema_momentum)
            batch_size = image.size(0)
            total_samples += batch_size
            total_batches += 1
            direct_sum_names = {
                "radiomics_loss_sum",
                "radiomics_valid_elements",
                "raw_learned_error_sum",
                "raw_copy_error_sum",
                "normalized_learned_error_sum",
                "normalized_copy_error_sum",
                "transition_cells",
            }
            for name, value in stats.items():
                if name.startswith("_") or name in {"radiomics_loss", "weighted_radiomics_loss"}:
                    continue
                if name in direct_sum_names:
                    sums[name] += float(value)
                else:
                    sums[name] += float(value) * batch_size
            sums["paired_sample_fraction"] += float(batch["has_radiomics"].float().mean()) * batch_size
            sums["gradient_norm"] += gradient_norm * batch_size
    if total_samples == 0:
        raise RuntimeError("DataLoader 为空")
    output_stats = {
        name: value / total_samples
        for name, value in sums.items()
        if name
        not in {
            "radiomics_loss_sum",
            "radiomics_valid_elements",
            "raw_learned_error_sum",
            "raw_copy_error_sum",
            "normalized_learned_error_sum",
            "normalized_copy_error_sum",
            "transition_cells",
        }
    }
    valid_elements = sums.get("radiomics_valid_elements", 0.0)
    radiomics_loss = sums.get("radiomics_loss_sum", 0.0) / max(valid_elements, 1.0)
    output_stats["radiomics_loss"] = radiomics_loss
    output_stats["weighted_radiomics_loss"] = objective.weights.radiomics * radiomics_loss
    output_stats["radiomics_valid_elements"] = valid_elements
    transition_cells = sums.get("transition_cells", 0.0)
    raw_learned = sums.get("raw_learned_error_sum", 0.0) / max(transition_cells, 1.0)
    raw_copy = sums.get("raw_copy_error_sum", 0.0) / max(transition_cells, 1.0)
    normalized_learned = sums.get("normalized_learned_error_sum", 0.0) / max(
        transition_cells, 1.0
    )
    normalized_copy = sums.get("normalized_copy_error_sum", 0.0) / max(
        transition_cells, 1.0
    )
    output_stats.update(
        {
            "raw_next_mse": raw_learned,
            "copy_mse": raw_copy,
            "normalized_next_mse": normalized_learned,
            "normalized_copy_mse": normalized_copy,
            "aggregate_transition_gain": (raw_copy - raw_learned) / max(raw_copy, 1e-8),
            "normalized_error_aggregate_gain": (normalized_copy - normalized_learned)
            / max(normalized_copy, 1e-8),
            "transition_cells": transition_cells,
        }
    )
    output_stats.update(component_gradient_stats)
    output_stats["samples"] = float(total_samples)
    output_stats["batches"] = float(total_batches)
    output_stats["seconds"] = time.monotonic() - start_time
    output_stats["peak_gpu_memory_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    return output_stats


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(
    model: ImageOnlyWorldModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    run_name: str,
    fold: int,
    epoch: int,
    best_metric: float,
    splits: dict[str, list[str]],
    transform_path: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    fold_manifest = resolve_path(config["data"]["fold_manifest"])
    return {
        "schema_version": 1,
        "run_name": run_name,
        "fold": fold,
        "epoch": epoch,
        "best_validation_objective": best_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": config["model"],
        "train_config": config["train"],
        "loss_config": config["loss"],
        "data_contract": {
            "fold_manifest": str(fold_manifest),
            "fold_manifest_sha256": file_sha256(fold_manifest),
            "cache_root": str(resolve_path(config["data"]["cache_root"])),
            "radiomics_transform": str(transform_path),
            "radiomics_transform_sha256": file_sha256(transform_path),
            "train_patient_hash": patient_hash(splits["train"]),
            "fold_train_patient_hash": patient_hash(splits["train"]),
            "val_patient_hash": patient_hash(splits["val"]),
            "test_patient_hash": patient_hash(splits["test"]),
            "extra_pretrain_patient_hash": patient_hash(set(splits["pretrain_train"]) - set(splits["train"])),
        },
        "splits": splits,
        "runtime": runtime,
        "architecture_contract": model.architecture_contract(),
        "git": git_metadata(),
        "implementation_sha256": implementation_sha256(),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "fold_provenance_status": str(config["data"]["fold_provenance_status"]),
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "note": "固定 seed 与无随机 augmentation；未宣称跨硬件 bitwise reproducibility",
        },
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
    }


def train_fold(
    config: dict[str, Any],
    run_name: str,
    fold: int,
    device_name: str = "cuda",
    epochs_override: int | None = None,
    smoke_patients: int | None = None,
    lambda_rad_override: float | None = None,
    output_suffix: str = "",
) -> Path:
    if fold not in range(5):
        raise ValueError("fold 必须为 0–4")
    config = copy.deepcopy(config)
    validate_run_config(config)
    bundle = build_bundle(config)
    splits = split_ids(bundle, fold)
    transform_path = EXPERIMENT_ROOT / "configs" / f"radiomics_transform_fold_{fold}.json"
    transform = ensure_radiomics_transform(bundle, fold, splits["train"], transform_path)
    transformed = transform.transform_all(bundle.raw_radiomics)

    seed = int(config["train"]["seed"]) + fold
    set_seed(seed)
    train_ids = list(splits["pretrain_train"])
    val_ids = list(splits["val"])
    if smoke_patients is not None:
        rng = np.random.default_rng(seed)
        primary_train = list(splits["train"])
        n_primary = min(max(8, smoke_patients), len(primary_train))
        selected = rng.choice(primary_train, size=n_primary, replace=False).tolist()
        paired = [patient_id for patient_id in primary_train if patient_id in bundle.raw_radiomics]
        if paired and not set(selected).intersection(paired):
            selected[0] = paired[0]
        train_ids = selected
        val_ids = val_ids[: min(max(8, smoke_patients // 2), len(val_ids))]

    model_config = dict(config["model"])
    mode = str(model_config.pop("mode"))
    model = ImageOnlyWorldModel(mode=mode, **model_config)
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    model.to(device)
    lambda_rad = float(config["loss"].get("lambda_rad", 0.0))
    if lambda_rad_override is not None:
        lambda_rad = float(lambda_rad_override)
    config["loss"]["lambda_rad"] = lambda_rad
    objective = NextChangeObjective(
        mode=mode,
        lambda_rad=lambda_rad,
        sigreg_weight=float(config["loss"].get("sigreg", 0.09)),
        sigreg_projections=int(config["loss"].get("sigreg_projections", 256)),
        step_weights=tuple(float(value) for value in config["loss"].get("step_weights", [2.0, 1.0, 0.5])),
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    batch_size = int(config["train"]["batch_size"])
    workers = int(config["train"]["workers"])
    train_loader = make_loader(
        records_for_ids(bundle, train_ids), transformed, model.image_channels, batch_size, workers, True, seed
    )
    val_loader = make_loader(
        records_for_ids(bundle, val_ids), transformed, model.image_channels, batch_size, workers, False, seed
    )

    tag = f"{run_name}{output_suffix}"
    if smoke_patients is not None:
        tag = f"smoke_{tag}"
    output_dir = EXPERIMENT_ROOT / "checkpoints" / tag / f"fold_{fold}"
    metric_dir = EXPERIMENT_ROOT / "metrics" / "training" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    claim_path = output_dir / "RUN_CLAIM.json"
    history_path = metric_dir / f"fold_{fold}.csv"
    protected = (best_path, last_path, output_dir / "resolved_run.json", claim_path, history_path)
    if any(path.exists() for path in protected):
        raise FileExistsError(f"拒绝覆盖已有 checkpoint；请使用新 run/suffix: {output_dir}")
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"同名 run 已被另一进程占用: {output_dir}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "claimed_at_unix": time.time()}, stream)
    epochs = int(epochs_override or config["train"]["epochs"])
    config["train"]["epochs"] = epochs
    save_json(
        output_dir / "resolved_run.json",
        {
            "run_name": tag,
            "fold": fold,
            "mode": mode,
            "seed": seed,
            "lambda_rad": lambda_rad,
            "train_samples": len(train_ids),
            "train_ispy2_samples": sum(patient_id.startswith(("ISPY2-", "ACRIN-6698-")) for patient_id in train_ids),
            "train_ispy1_samples": sum(patient_id.startswith("ISPY1_") for patient_id in train_ids),
            "validation_samples": len(val_ids),
            "smoke_patients": smoke_patients,
            "epochs_requested": epochs,
            "device": str(device),
            "architecture_contract": model.architecture_contract(),
        },
    )
    patience = int(config["train"]["patience"])
    runtime = {
        "smoke": smoke_patients is not None,
        "smoke_patients_requested": smoke_patients,
        "effective_train_ids": train_ids,
        "effective_train_patient_hash": patient_hash(train_ids),
        "effective_validation_ids": val_ids,
        "effective_validation_patient_hash": patient_hash(val_ids),
        "transform_fit_ids": splits["train"],
        "epochs_requested": epochs,
        "lambda_rad": lambda_rad,
        "device": str(device),
    }
    history: list[dict[str, Any]] = []
    best_metric = math.inf
    stale = 0
    for epoch in range(1, epochs + 1):
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
        row: dict[str, Any] = {"epoch": epoch, "fold": fold, "mode": mode, "lambda_rad": lambda_rad}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in val_stats.items()})
        history.append(row)
        if mode == "m0":
            metric = float(val_stats["state_loss"])
        else:
            # M1/M2 的核心目标是相对 copy-current 的真实 transition 改善；
            # M2 与 M1 使用同一 test-blind image metric，radiomics 不决定停止点。
            metric = -float(val_stats["aggregate_transition_gain"])
        row["validation_selection_metric"] = metric
        eligible = float(val_stats["visit_feature_std"]) >= float(config["train"]["min_latent_std"])
        if eligible and metric < best_metric:
            best_metric = metric
            stale = 0
            torch.save(
                checkpoint_payload(
                    model, optimizer, config, tag, fold, epoch, best_metric, splits, transform_path, runtime
                ),
                best_path,
            )
        else:
            stale += 1
        torch.save(
            checkpoint_payload(
                model, optimizer, config, tag, fold, epoch, best_metric, splits, transform_path, runtime
            ),
            last_path,
        )
        write_history(history_path, history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "mode": mode,
                    "lambda_rad": lambda_rad,
                    "train_loss": train_stats["loss"],
                    "val_loss": val_stats["loss"],
                    "val_state": val_stats["state_loss"],
                    "val_delta": val_stats["delta_loss"],
                    "val_rad": val_stats["radiomics_loss"],
                    "val_gain": val_stats["aggregate_transition_gain"],
                    "val_state_std": val_stats["visit_state_std"],
                    "val_feature_std": val_stats["visit_feature_std"],
                    "train_seconds": train_stats["seconds"],
                    "val_seconds": val_stats["seconds"],
                    "eligible": eligible,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stale >= patience:
            break
    if not best_path.exists():
        raise RuntimeError("没有 checkpoint 满足 minimum latent std")
    return best_path
