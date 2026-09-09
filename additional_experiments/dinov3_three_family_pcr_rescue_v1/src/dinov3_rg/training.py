"""Outcome-blind C0 training, radiomics fine-tuning, and state export."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .contracts import (
    EXPERIMENT_ROOT, FOLD_PATIENTS, atomic_json, canonical_sha256, file_sha256,
    load_protocol, patient_order_sha256,
)
from .data import FoldTargets, SummaryDataset, validate_state_archive
from .model import MRIAdapterWorldModel, initialization_sha256
from .objective import DirectRadiomicsObjective


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def make_loader(
    dataset: SummaryDataset, *, shuffle: bool, seed: int,
    batch_size: int = 32, workers: int = 4,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle),
        drop_last=bool(shuffle and len(dataset) >= batch_size),
        num_workers=int(workers), pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers), generator=generator,
    )


def _to_device(batch: Mapping[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _module_gradient_norm(module: nn.Module) -> float:
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def run_epoch(
    model: MRIAdapterWorldModel,
    objective: DirectRadiomicsObjective,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *, ema: float = 0.996, gradient_clip: float = 5.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training); objective.train(training)
    sums: dict[str, float] = defaultdict(float)
    samples = 0; states: list[torch.Tensor] = []; maximum_gradient = 0.0
    first_adapter = first_projection = first_head = 0.0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device); count = int(batch["summary"].size(0))
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch["summary"])
            loss, stats = objective(
                output, batch["ftv"], batch["ftv_mask"],
                batch["radiomics"], batch["radiomics_mask"],
            )
        if training:
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"non-finite gradient: {name}")
            if samples == 0:
                first_adapter = _module_gradient_norm(model.adapter)
                first_projection = _module_gradient_norm(model.adapter.response_projection)
                first_head = _module_gradient_norm(model.radiomics_head)
            norm = clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), float(gradient_clip)
            )
            maximum_gradient = max(maximum_gradient, float(norm))
            optimizer.step(); model.update_target(float(ema))
        for name, value in stats.items(): sums[name] += float(value) * count
        states.append(output.response_state.detach().float().cpu()); samples += count
    if samples == 0: raise RuntimeError("epoch loader produced no samples")
    state = torch.cat(states).reshape(-1, 192)
    result = {name: value / samples for name, value in sums.items()}
    result.update({
        "state_mean_sd": float(state.std(0, unbiased=False).mean()),
        "state_min_sd": float(state.std(0, unbiased=False).min()),
        "gradient_norm_max_preclip": maximum_gradient,
        "first_batch_adapter_gradient_norm": first_adapter,
        "first_batch_response_projection_gradient_norm": first_projection,
        "first_batch_radiomics_head_gradient_norm": first_head,
        "samples": float(samples),
    })
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError("epoch metrics are non-finite")
    return result


def reinitialize_radiomics_head(model: MRIAdapterWorldModel, seed: int) -> str:
    # The model is still on CPU here; do not initialize or perturb any CUDA RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed)); nn.init.xavier_uniform_(model.radiomics_head.weight)
        nn.init.zeros_(model.radiomics_head.bias)
    return canonical_sha256({
        "weight": model.radiomics_head.weight.detach().cpu().float().numpy().tolist(),
        "bias": model.radiomics_head.bias.detach().cpu().float().numpy().tolist(),
    })


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary); os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True); raise


def _datasets(
    train_ids: tuple[str, ...], validation_ids: tuple[str, ...],
    summary_dir: str | Path, targets_path: str | Path,
) -> tuple[SummaryDataset, SummaryDataset]:
    targets = FoldTargets.load(targets_path)
    return SummaryDataset(train_ids, summary_dir, targets), SummaryDataset(validation_ids, summary_dir, targets)


def _fit(
    *, model: MRIAdapterWorldModel, objective: DirectRadiomicsObjective,
    optimizer: torch.optim.Optimizer, train_loader: DataLoader,
    validation_loader: DataLoader, device: torch.device, effective_seed: int,
    epochs: int, patience: int, ema: float, gradient_clip: float,
    minimum_state_sd: float, maximum_jepa: float | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    best: dict[str, Any] | None = None; best_score = math.inf; stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        train = run_epoch(model, objective, train_loader, device, optimizer, ema=ema, gradient_clip=gradient_clip)
        torch.manual_seed(int(effective_seed) + 100000 + epoch)
        validation = run_epoch(model, objective, validation_loader, device, None, ema=ema, gradient_clip=gradient_clip)
        feasible = validation["state_mean_sd"] >= float(minimum_state_sd)
        if maximum_jepa is not None: feasible = feasible and validation["jepa_loss"] <= maximum_jepa
        score = validation["jepa_loss"] if objective.radiomics_weight == 0 else validation["radiomics_loss"]
        score = float(score) if feasible else math.inf
        history.append({"epoch": epoch, "train": train, "validation": validation, "feasible": feasible, "selection_score": score})
        if score < best_score - 1e-12:
            best_score = score; stale = 0
            best = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": epoch, "validation": validation}
        elif best is not None:
            stale += 1
        if best is not None and stale >= int(patience): break
    return best, history


def train_cell(
    *, seed: int, fold: int, arm: str, radiomics_weight: float,
    train_ids: Iterable[str], validation_ids: Iterable[str],
    summary_dir: str | Path, targets_path: str | Path,
    checkpoint_root: str | Path, device: str = "cuda", workers: int = 4,
    base_checkpoint: str | Path | None = None,
    base_completion: str | Path | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(); arm = str(arm).upper(); fold = int(fold); seed = int(seed)
    train_ids = tuple(map(str, train_ids)); validation_ids = tuple(map(str, validation_ids))
    effective_seed = seed + fold; set_seed(effective_seed)
    model = MRIAdapterWorldModel(dropout=float(protocol["model"]["dropout"]))
    base_sha = None; base_jepa = None
    if base_checkpoint is not None:
        checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        base_sha = file_sha256(base_checkpoint)
        if base_completion is None: raise ValueError("candidate requires base completion")
        base_payload = json.loads(Path(base_completion).read_text(encoding="utf-8"))
        base_jepa = float(base_payload["selected_validation"]["jepa_loss"])
        head_init_sha = reinitialize_radiomics_head(model, 900000 + effective_seed)
    else:
        if float(radiomics_weight) != 0.0 or arm != "C0":
            raise ValueError("only C0 may train without a base checkpoint")
        head_init_sha = canonical_sha256("C0-untrained-radiomics-head")
    model.ftv_head.requires_grad_(False)
    if float(radiomics_weight) == 0.0: model.radiomics_head.requires_grad_(False)
    init_hash = initialization_sha256(model)
    resolved = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(resolved); objective = DirectRadiomicsObjective(radiomics_weight).to(resolved)
    if radiomics_weight == 0:
        config = protocol["formal"]
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(config["baseline_learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
        total_epochs = int(config["baseline_epochs"] if epochs is None else epochs)
        patience = int(config["baseline_patience"]); maximum_jepa = None
    else:
        config = protocol["pilot"]
        head_parameters = list(model.radiomics_head.parameters())
        head_ids = {id(p) for p in head_parameters}
        shared_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        optimizer = torch.optim.AdamW([
            {"params": shared_parameters, "lr": float(config["shared_learning_rate"])},
            {"params": head_parameters, "lr": float(config["radiomics_head_learning_rate"])},
        ], weight_decay=float(config["weight_decay"]))
        total_epochs = int(config["epochs"] if epochs is None else epochs)
        patience = int(config["patience"]); maximum_jepa = float(config["maximum_jepa_ratio"]) * float(base_jepa)
    train_dataset, validation_dataset = _datasets(train_ids, validation_ids, summary_dir, targets_path)
    train_loader = make_loader(train_dataset, shuffle=True, seed=effective_seed, batch_size=32, workers=workers)
    validation_loader = make_loader(validation_dataset, shuffle=False, seed=effective_seed, batch_size=32, workers=workers)
    root = Path(checkpoint_root); cell_dir = root / f"seed{seed}_fold{fold}_{arm}"
    started = time.monotonic()
    best, history = _fit(
        model=model, objective=objective, optimizer=optimizer, train_loader=train_loader,
        validation_loader=validation_loader, device=resolved, effective_seed=effective_seed,
        epochs=total_epochs, patience=patience, ema=float(config["ema"]),
        gradient_clip=float(config["gradient_clip"]),
        minimum_state_sd=float(protocol["pilot"]["minimum_state_sd"]), maximum_jepa=maximum_jepa,
    )
    atomic_json(cell_dir / "history.private.json", {"history": history})
    if best is None:
        failure = {
            "schema_version": 1, "status": "NO_FEASIBLE_CHECKPOINT", "seed": seed,
            "fold": fold, "arm": arm, "radiomics_weight": radiomics_weight,
            "epochs_completed": len(history), "paired_c0_jepa_loss": base_jepa,
            "maximum_allowed_jepa_loss": maximum_jepa,
            "minimum_observed_jepa_loss": min(float(e["validation"]["jepa_loss"]) for e in history),
            "history_sha256": canonical_sha256(history), "outcome_fields_read": [], "clinical_fields_read": [],
        }
        atomic_json(cell_dir / "cell_failed.private.json", failure)
        raise RuntimeError(f"{seed}/{fold}/{arm} has no feasible checkpoint")
    checkpoint_path = cell_dir / "selected.private.pt"
    payload = {
        **best, "seed": seed, "fold": fold, "arm": arm, "effective_seed": effective_seed,
        "radiomics_weight": radiomics_weight, "ftv_weight": 0.0,
        "initialization_sha256": init_hash, "radiomics_head_initialization_sha256": head_init_sha,
        "base_checkpoint_sha256": base_sha, "architecture": model.architecture_contract(),
        "protocol_sha256": file_sha256(EXPERIMENT_ROOT / "configs/protocol.json"),
    }
    _atomic_torch_save(checkpoint_path, payload)
    complete = {
        "schema_version": 1, "status": "COMPLETE", "seed": seed, "fold": fold, "arm": arm,
        "effective_seed": effective_seed, "radiomics_weight": radiomics_weight, "ftv_weight": 0.0,
        "selected_epoch": int(best["epoch"]), "selected_validation": best["validation"],
        "initialization_sha256": init_hash, "radiomics_head_initialization_sha256": head_init_sha,
        "base_checkpoint_sha256": base_sha, "checkpoint_sha256": file_sha256(checkpoint_path),
        "train_patient_order_sha256": patient_order_sha256(train_ids),
        "validation_patient_order_sha256": patient_order_sha256(validation_ids),
        "history_sha256": canonical_sha256(history), "elapsed_seconds": time.monotonic() - started,
        "outcome_fields_read": [], "clinical_fields_read": [],
    }
    atomic_json(cell_dir / "cell_complete.private.json", complete)
    return complete


@torch.inference_mode()
def export_cell_states(
    *, checkpoint_path: str | Path, patient_ids: Iterable[str], summary_dir: str | Path,
    output_path: str | Path, device: str = "cuda", workers: int = 4,
) -> dict[str, Any]:
    ids = tuple(sorted(map(str, patient_ids)))
    if len(ids) != FOLD_PATIENTS: raise ValueError("state export requires all 808 I-SPY2 patients")
    resolved = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MRIAdapterWorldModel(); model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(resolved).eval(); dataset = SummaryDataset(ids, summary_dir, targets=None)
    loader = make_loader(dataset, shuffle=False, seed=0, batch_size=32, workers=workers)
    states=[]; radiomics=[]; ftv=[]; observed=[]
    for batch in loader:
        response = model.encode_response(batch["summary"].to(resolved, non_blocking=True))
        states.append(response.float().cpu().numpy())
        radiomics.append(model.radiomics_head(response).float().cpu().numpy())
        ftv.append(model.ftv_head(response).squeeze(-1).float().cpu().numpy())
        observed.extend(map(str, batch["patient_id"]))
    if tuple(observed) != ids: raise AssertionError("state export patient order drifted")
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary, patient_id=np.asarray(ids, dtype="U32"),
            state=np.concatenate(states).astype(np.float32),
            radiomics_prediction=np.concatenate(radiomics).astype(np.float32),
            ftv_prediction=np.concatenate(ftv).astype(np.float32),
            checkpoint_sha256=np.asarray(file_sha256(checkpoint_path)),
        ); os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True); raise
    _, state = validate_state_archive(destination)
    return {"status": "COMPLETE", "patients": len(ids), "shape": list(state.shape),
            "state_mean_sd": float(state.reshape(-1, 192).std(0).mean()), "sha256": file_sha256(destination)}


__all__ = [
    "export_cell_states", "make_loader", "reinitialize_radiomics_head", "run_epoch",
    "set_seed", "train_cell",
]
