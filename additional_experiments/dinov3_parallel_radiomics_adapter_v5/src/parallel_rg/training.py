from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .contracts import ROOT, V2_SUMMARY_DIR, atomic_json, load_protocol, sha256_file
from .data import SummaryDataset, RadiomicsTargets, fold_target_path
from .model import ParallelRadiomicsModel
from .objective import masked_radiomics_loss


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))


def resolve_device(value: str) -> torch.device:
    return torch.device(value if value != "cuda" or torch.cuda.is_available() else "cpu")


def make_loader(dataset: SummaryDataset, *, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=32, shuffle=shuffle, drop_last=False, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0, generator=generator)


def _grad_norm(module: nn.Module) -> float:
    values = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def _device_batch(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def run_epoch(model: ParallelRadiomicsModel, data_loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    training = optimizer is not None; model.train(training)
    totals: dict[str, float] = defaultdict(float); samples = 0; state_chunks = []
    first_adapter = first_branch = first_head = 0.0; max_grad = 0.0
    for raw in data_loader:
        batch = _device_batch(raw, device); count = int(batch["summary"].size(0))
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch["summary"]); loss, visits = masked_radiomics_loss(output["radiomics_prediction"], batch["radiomics"], batch["radiomics_mask"])
        if training:
            loss.backward()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite radiomics loss")
            if samples == 0:
                first_adapter = _grad_norm(model.rad_adapter); first_branch = _grad_norm(model.rad_branch); first_head = _grad_norm(model.radiomics_head)
            for parameter in model.rad_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()): raise FloatingPointError("non-finite radiomics gradient")
            max_grad = max(max_grad, float(clip_grad_norm_(model.rad_parameters(), 5.0))); optimizer.step()
        totals["loss"] += float(loss.detach()) * count; totals["radiomics_visits"] += float(visits.detach()) * count
        state_chunks.append(output["rad_state"].detach().float().cpu()); samples += count
    if samples == 0: raise RuntimeError("empty epoch")
    state = torch.cat(state_chunks).reshape(-1, 64)
    result = {key: value / samples for key, value in totals.items()}; result.update({"rad_state_mean_sd": float(state.std(0, unbiased=False).mean()), "maximum_gradient_preclip": max_grad, "first_rad_adapter_gradient_norm": first_adapter, "first_rad_branch_gradient_norm": first_branch, "first_radiomics_head_gradient_norm": first_head, "samples": float(samples)})
    if not all(np.isfinite(value) for value in result.values()): raise FloatingPointError("non-finite epoch metrics")
    return result


def _best_state(model: ParallelRadiomicsModel) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_cell(*, fold: int, seed: int, c0_checkpoint: str | Path, checkpoint_root: str | Path, device: str, workers: int = 4) -> dict[str, Any]:
    protocol = load_protocol(); fold = int(fold); seed = int(seed); effective_seed = seed + fold; set_seed(effective_seed)
    c0_checkpoint = Path(c0_checkpoint); model = ParallelRadiomicsModel(str(c0_checkpoint)); resolved = resolve_device(device); model.to(resolved)
    targets = RadiomicsTargets(fold_target_path(fold)); target_ids = set(targets.patient_ids)
    frame = __import__("parallel_rg.contracts", fromlist=["load_folds"]).load_folds()
    current = frame.loc[frame["fold"].eq(fold)]
    train_ids = tuple(sorted(set(current.loc[current["split"].eq("train"), "patient_id"].astype(str)) & target_ids))
    validation_ids = tuple(sorted(set(current.loc[current["split"].eq("val"), "patient_id"].astype(str)) & target_ids))
    if len(train_ids) < 100 or len(validation_ids) < 20: raise RuntimeError("insufficient fold target rows")
    train_loader = make_loader(SummaryDataset(train_ids, V2_SUMMARY_DIR, fold_target_path(fold)), shuffle=True, seed=effective_seed, workers=workers)
    validation_loader = make_loader(SummaryDataset(validation_ids, V2_SUMMARY_DIR, fold_target_path(fold)), shuffle=False, seed=effective_seed, workers=workers)
    head_lr = float(protocol["training"]["head_learning_rate"]); branch_lr = float(protocol["training"]["branch_learning_rate"]); adapter_lr = float(protocol["training"]["adapter_learning_rate"]); weight_decay = float(protocol["training"]["weight_decay"])
    model.set_head_only(); optimizer = torch.optim.AdamW(model.radiomics_head.parameters(), lr=head_lr, weight_decay=weight_decay)
    cell = Path(checkpoint_root) / f"seed{seed}_fold{fold}_P1_PARALLEL"; cell.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []; best_state = None; best_score = float("inf"); stale = 0
    for epoch in range(int(protocol["training"]["head_warmup_epochs"])):
        train = run_epoch(model, train_loader, resolved, optimizer); validation = run_epoch(model, validation_loader, resolved, None); feasible = validation["rad_state_mean_sd"] >= float(protocol["training"]["minimum_rad_state_sd"]); score = validation["loss"] if feasible else float("inf")
        history.append({"stage": "head_warmup", "epoch": epoch, "train": train, "validation": validation, "feasible": feasible, "selection_score": score})
        if score < best_score - 1e-12: best_score = score; best_state = _best_state(model); stale = 0
        else: stale += 1
        if best_state is not None and stale >= int(protocol["training"]["head_warmup_patience"]): break
    if best_state is None: raise RuntimeError("head warm-up has no feasible checkpoint")
    model.load_state_dict(best_state, strict=True); model.set_all_rad_trainable()
    head_only_checkpoint = cell / "head_only.private.pt"
    torch.save({"model_state": best_state, "seed": seed, "fold": fold, "phase": "head_only", "c0_checkpoint_sha256": sha256_file(c0_checkpoint), "protocol_sha256": sha256_file(ROOT / "configs/protocol.json")}, head_only_checkpoint)
    optimizer = torch.optim.AdamW([
        {"params": model.rad_adapter.parameters(), "lr": adapter_lr},
        {"params": model.rad_branch.parameters(), "lr": branch_lr},
        {"params": model.radiomics_head.parameters(), "lr": head_lr},
    ], weight_decay=weight_decay)
    best_state = None; best_score = float("inf"); stale = 0
    for epoch in range(int(protocol["training"]["adapter_epochs"])):
        train = run_epoch(model, train_loader, resolved, optimizer); validation = run_epoch(model, validation_loader, resolved, None); feasible = validation["rad_state_mean_sd"] >= float(protocol["training"]["minimum_rad_state_sd"]); score = validation["loss"] if feasible else float("inf")
        history.append({"stage": "adapter", "epoch": epoch, "train": train, "validation": validation, "feasible": feasible, "selection_score": score})
        if score < best_score - 1e-12: best_score = score; best_state = _best_state(model); stale = 0
        else: stale += 1
        if best_state is not None and stale >= int(protocol["training"]["adapter_patience"]): break
    if best_state is None: raise RuntimeError("independent adapter has no feasible checkpoint")
    model.load_state_dict(best_state, strict=True)
    atomic_json(cell / "history.private.json", {"history": history})
    checkpoint = cell / "selected.private.pt"; torch.save({"model_state": best_state, "seed": seed, "fold": fold, "architecture": model.architecture_contract(), "c0_checkpoint_sha256": sha256_file(c0_checkpoint), "protocol_sha256": sha256_file(ROOT / "configs/protocol.json"), "radiomics_weight": 1.0, "jepa_weight_in_rad_path": 0.0, "sigreg_weight_in_rad_path": 0.0, "ftv_weight_in_rad_path": 0.0, "temporal_weight_in_rad_path": 0.0}, checkpoint)
    complete = {"status": "COMPLETE", "implementation_version": VERSION, "seed": seed, "fold": fold, "arm": "P1_PARALLEL", "c0_checkpoint_sha256": sha256_file(c0_checkpoint), "rad_checkpoint_sha256": sha256_file(checkpoint), "head_only_checkpoint_sha256": sha256_file(head_only_checkpoint), "history_sha256": sha256_file(cell / "history.private.json"), "selected_head_warmup_epoch": int(next(row["epoch"] for row in history if row["stage"] == "head_warmup" and row["selection_score"] == min(x["selection_score"] for x in history if x["stage"] == "head_warmup"))), "selected_adapter_epoch": int(next(row["epoch"] for row in history if row["stage"] == "adapter" and row["selection_score"] == min(x["selection_score"] for x in history if x["stage"] == "adapter"))), "c0_trainable": False, "radiomics_loss_updates_c0": False, "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(cell / "cell_complete.private.json", complete); return complete


@torch.inference_mode()
def export_states(*, fold: int, c0_checkpoint: str | Path, rad_checkpoint: str | Path, patient_ids: Iterable[str], output_path: str | Path, device: str, workers: int = 4) -> dict[str, Any]:
    ids = tuple(sorted(map(str, patient_ids))); resolved = resolve_device(device); model = ParallelRadiomicsModel(str(c0_checkpoint)); payload = torch.load(rad_checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(payload["model_state"], strict=True); model.to(resolved).eval()
    dataset = SummaryDataset(ids, V2_SUMMARY_DIR, fold_target_path(fold)); loader = make_loader(dataset, shuffle=False, seed=0, workers=workers)
    c0_states=[]; rad_initial=[]; rad_states=[]; parallel_initial=[]; parallel=[]; rad_predictions=[]; observed=[]
    for raw in loader:
        output = model(raw["summary"].to(resolved, non_blocking=True)); c0 = output["c0_state"].float().cpu().numpy(); initial = output["c0_phenotype"].float().cpu().numpy(); rad = output["rad_state"].float().cpu().numpy(); prediction = output["radiomics_prediction"].float().cpu().numpy(); c0_states.append(c0); rad_initial.append(initial); rad_states.append(rad); parallel_initial.append(np.concatenate((c0, initial), -1)); parallel.append(np.concatenate((c0, rad), -1)); rad_predictions.append(prediction); observed.extend(map(str, raw["patient_id"]))
    if tuple(observed) != ids: raise AssertionError("state export order drifted")
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_name("." + destination.name + ".tmp.npz")
    np.savez_compressed(temporary, patient_id=np.asarray(ids, dtype="U64"), c0_state=np.concatenate(c0_states), rad_initial_state=np.concatenate(rad_initial), rad_state=np.concatenate(rad_states), parallel_initial_state=np.concatenate(parallel_initial), parallel_state=np.concatenate(parallel), radiomics_prediction=np.concatenate(rad_predictions), c0_checkpoint_sha256=np.asarray(sha256_file(c0_checkpoint)), rad_checkpoint_sha256=np.asarray(sha256_file(rad_checkpoint)))
    temporary.replace(destination)
    return {"status": "COMPLETE", "patients": len(ids), "c0_state_shape": list(np.concatenate(c0_states).shape), "rad_state_shape": list(np.concatenate(rad_states).shape), "parallel_state_shape": list(np.concatenate(parallel).shape), "state_sha256": sha256_file(destination)}


VERSION = "v5-parallel-radiomics-20260903-r1"
