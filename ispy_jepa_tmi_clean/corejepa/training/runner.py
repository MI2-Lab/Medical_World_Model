from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..config import ExperimentConfig
from ..data.condition import ConditionEncoder
from ..data.dataset import LongitudinalDCEDataset, PretrainingDataset
from ..data.imaging import load_phase_metadata
from ..data.records import PatientRecord, load_records, stratified_split
from ..data.response_targets import (
    ResponseTargetTransform,
    build_response_feature_cache,
    load_response_vectors,
)
from ..models import CoReJEPA
from .losses import PretrainingObjective


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: torch.nn.Module) -> CoReJEPA:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def select_device(gpus: tuple[int, ...]) -> tuple[torch.device, list[int]]:
    if not torch.cuda.is_available() or not gpus:
        return torch.device("cpu"), []
    available = torch.cuda.device_count()
    selected = [gpu for gpu in gpus if 0 <= gpu < available]
    if not selected:
        return torch.device("cpu"), []
    return torch.device(f"cuda:{selected[0]}"), selected


def load_experiment_records(config: ExperimentConfig) -> tuple[list[PatientRecord], int]:
    primary = load_records(config.data.ispy2_root, config.data.ispy2_labels, "ispy2", require_pcr=True)
    extra: list[PatientRecord] = []
    if config.data.ispy1_root and config.data.ispy1_labels:
        extra = load_records(config.data.ispy1_root, config.data.ispy1_labels, "ispy1", require_pcr=False)
    return primary + extra, len(primary)


def build_splits(records: list[PatientRecord], n_primary: int, seed: int) -> dict[str, list[int]]:
    primary_train, validation, test = stratified_split(records[:n_primary], seed)
    extra = list(range(n_primary, len(records)))
    return {
        "primary_train": primary_train,
        "pretrain_train": primary_train + extra,
        "validation": validation,
        "test": test,
    }


def build_datasets(
    config: ExperimentConfig,
) -> tuple[PretrainingDataset, list[PatientRecord], int, dict[str, list[int]], ConditionEncoder, ResponseTargetTransform]:
    records, n_primary = load_experiment_records(config)
    splits = build_splits(records, n_primary, config.train.split_seed)
    condition_encoder = ConditionEncoder(records)
    phase_metadata = load_phase_metadata(config.data.breastdcedl_metadata_csv)
    base = LongitudinalDCEDataset(
        records=records,
        condition_encoder=condition_encoder,
        cache_dir=config.data.tensor_cache,
        crop_size=config.data.crop_size,
        phase_policy=config.data.phase_policy,
        phase_metadata=phase_metadata,
        automatic_roi_fallback=config.data.auto_roi_fallback,
        minimum_roi_capture=config.data.min_roi_capture,
        legacy_empty_ftv_full_field=config.data.legacy_empty_ftv_full_field,
        build_missing=True,
    )
    response_cache = Path(config.data.response_cache)
    if not response_cache.exists():
        build_response_feature_cache(
            records,
            response_cache,
            config.data.auto_roi_fallback,
            config.data.legacy_empty_ftv_full_field,
            phase_metadata,
            config.data.response_phase_policy,
        )
    raw_response = load_response_vectors(response_cache, records)
    transform = ResponseTargetTransform.fit(raw_response, records, splits["pretrain_train"])
    response_vector, response_score = transform.transform(raw_response, records)
    return PretrainingDataset(base, response_vector, response_score), records, n_primary, splits, condition_encoder, transform


def make_loader(
    dataset: PretrainingDataset,
    indices: list[int],
    config: ExperimentConfig,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.workers > 0,
        drop_last=shuffle and len(indices) >= config.train.batch_size,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: torch.nn.Module,
    objective: PretrainingObjective,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ema_momentum: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    objective.train(training)
    totals: dict[str, list[float]] = defaultdict(list)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if training else "validation"):
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch["image"], batch["geometry"], batch["condition"])
            loss, stats = objective(output, batch)
            if training:
                loss.backward()
                optimizer.step()
                unwrap(model).update_target(ema_momentum)
            for name, value in stats.items():
                totals[name].append(float(value))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_payload(
    model: torch.nn.Module,
    config: ExperimentConfig,
    condition_encoder: ConditionEncoder,
    response_transform: ResponseTargetTransform,
    records: list[PatientRecord],
    n_primary: int,
    splits: dict[str, list[int]],
    epoch: int,
    validation: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": unwrap(model).state_dict(),
        "config": config.to_dict(),
        "condition": {
            "feature_names": list(condition_encoder.spec.feature_names),
            "arm_vocab": condition_encoder.spec.arm_vocab,
            "age_mean": condition_encoder.spec.age_mean,
            "age_std": condition_encoder.spec.age_std,
        },
        "response_transform": response_transform.state_dict(),
        "patient_ids": [record.patient_id for record in records],
        "n_primary": n_primary,
        "splits": splits,
        "epoch": epoch,
        "validation": validation,
    }


def train(config: ExperimentConfig) -> Path:
    """Run pCR-free CoRe-JEPA pretraining and return the best checkpoint."""

    set_seed(config.train.seed)
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")
    dataset, records, n_primary, splits, condition_encoder, response_transform = build_datasets(config)
    train_loader = make_loader(dataset, splits["pretrain_train"], config, shuffle=True)
    validation_loader = make_loader(dataset, splits["validation"], config, shuffle=False)
    device, gpu_ids = select_device(config.train.gpus)
    model: torch.nn.Module = CoReJEPA(config.model, condition_encoder.spec.dim).to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    route_weights = condition_encoder.routing_class_weights(records, splits["pretrain_train"])
    objective = PretrainingObjective(
        config.loss,
        config.train.sigreg_projections,
        torch.from_numpy(route_weights),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_prediction = float("inf")
    epochs_without_improvement = 0
    best_path, last_path = output_dir / "best_corejepa.pt", output_dir / "last_corejepa.pt"
    for epoch in range(1, config.train.epochs + 1):
        train_stats = run_epoch(model, objective, train_loader, device, optimizer, config.train.ema_momentum)
        validation_stats = run_epoch(model, objective, validation_loader, device, None, config.train.ema_momentum)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in validation_stats.items()})
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_stats['loss']:.4f} "
            f"val_loss={validation_stats['loss']:.4f} val_pred={validation_stats['prediction']:.4f} "
            f"val_state_std={validation_stats['visit_state_std']:.4f}"
        )
        payload = _checkpoint_payload(
            model,
            config,
            condition_encoder,
            response_transform,
            records,
            n_primary,
            splits,
            epoch,
            validation_stats,
        )
        torch.save(payload, last_path)
        eligible = validation_stats["visit_state_std"] >= config.train.min_latent_std
        if eligible and validation_stats["prediction"] < best_prediction:
            best_prediction = validation_stats["prediction"]
            epochs_without_improvement = 0
            torch.save(payload, best_path)
        else:
            epochs_without_improvement += 1
        _write_history(output_dir / "history.csv", history)
        if epochs_without_improvement >= config.train.patience:
            break
    if not best_path.exists():
        raise RuntimeError("No checkpoint satisfied the minimum latent standard deviation")
    export_frozen_states(config, best_path, dataset, records, n_primary, splits, condition_encoder, device)
    return best_path


@torch.no_grad()
def export_frozen_states(
    config: ExperimentConfig,
    checkpoint_path: Path,
    dataset: PretrainingDataset,
    records: list[PatientRecord],
    n_primary: int,
    splits: dict[str, list[int]],
    condition_encoder: ConditionEncoder,
    device: torch.device | None = None,
) -> Path:
    """Export frozen future states for downstream FLR fitting."""

    if device is None:
        device, _ = select_device(config.train.gpus)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CoReJEPA(config.model, condition_encoder.spec.dim).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    loader = DataLoader(
        dataset.base,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.workers,
        pin_memory=torch.cuda.is_available(),
    )
    states, image_predictions, ids = [], [], []
    for batch in tqdm(loader, leave=False, desc="export frozen states"):
        batch = move_batch(batch, device)
        response_state = model.forecast_response(batch["geometry"], batch["condition"])
        visit_state = model.encode_visits(batch["image"], batch["geometry"])
        image_prediction = model.image_transition(visit_state[:, :-1], batch["condition"])
        states.append(response_state.cpu().numpy())
        image_predictions.append(image_prediction.cpu().numpy())
        ids.extend(batch["patient_id"])
    output = Path(config.train.output_dir) / "frozen_states.npz"
    labels = np.asarray([record.pcr if record.pcr is not None else -1 for record in records], dtype=np.int64)
    np.savez_compressed(
        output,
        patient_ids=np.asarray(ids),
        future_response_state=np.concatenate(states).astype(np.float32),
        image_prediction=np.concatenate(image_predictions).astype(np.float32),
        pcr=labels,
        n_primary=np.asarray(n_primary, dtype=np.int64),
    )
    (Path(config.train.output_dir) / "splits.json").write_text(json.dumps(splits, indent=2))
    return output
