from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .contracts import V2_D1_CHECKPOINT_ROOT, V2_SUMMARY_DIR, atomic_json, canonical_sha, load_protocol, sha256_file, private_token
from .data import SummaryDataset, fold_target_path
from .model import FactorizedWorldModel
from .objective import FactorizedObjective


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))


def loader(dataset: SummaryDataset, *, shuffle: bool, seed: int, workers: int = 4) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=32, shuffle=shuffle, drop_last=shuffle and len(dataset) >= 32, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0, generator=generator)


def _device(value: str) -> torch.device:
    return torch.device(value if value != "cuda" or torch.cuda.is_available() else "cpu")


IMPLEMENTATION_VERSION = "v4-factorized-20260902-r2"


def _load_inherited(model: FactorizedWorldModel, checkpoint_path: Path) -> str:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = payload["model_state"]
    destination = model.state_dict()
    # Candidate cells must clone the complete V4 F0 state. Only the initial
    # F0 cell inherits matching adapter weights from the older V2 D1 model.
    if "jepa_branch.net.0.weight" in source:
        model.load_state_dict(source, strict=True)
        return sha256_file(checkpoint_path)
    copied = 0
    for name, value in source.items():
        if name in destination and destination[name].shape == value.shape:
            destination[name].copy_(value)
            copied += 1
    if copied < 20:
        raise RuntimeError("V2 D1 checkpoint did not provide the expected adapter weights")
    model.load_state_dict(destination, strict=True)
    return sha256_file(checkpoint_path)


def _module_grad_norm(module: nn.Module) -> float:
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def run_epoch(model: FactorizedWorldModel, objective: FactorizedObjective, data_loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None, seed: int) -> dict[str, float]:
    training = optimizer is not None
    model.train(training); objective.train(training)
    totals: dict[str, float] = defaultdict(float); count_total = 0; states = []
    first_adapter = first_response = first_phenotype = first_head = 0.0; maximum_grad = 0.0
    for raw in data_loader:
        batch = _to_device(raw, device); count = int(batch["summary"].size(0))
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch["summary"])
            loss, stats = objective(output, batch["radiomics"], batch["radiomics_mask"])
        if training:
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"non-finite gradient: {name}")
            if count_total == 0:
                first_adapter = _module_grad_norm(model.adapter); first_response = _module_grad_norm(model.adapter.response_projection)
                first_phenotype = _module_grad_norm(model.phenotype_branch); first_head = _module_grad_norm(model.radiomics_head)
            maximum_grad = max(maximum_grad, float(clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)))
            optimizer.step(); model.update_target(float(load_protocol()["pilot"]["ema"]))
        for name, value in stats.items(): totals[name] += float(value) * count
        states.append(torch.cat((output.jepa_online.detach().float(), output.phenotype_online.detach().float()), -1).cpu()); count_total += count
    if count_total == 0: raise RuntimeError("empty epoch")
    state = torch.cat(states).reshape(-1, 192)
    result = {name: value / count_total for name, value in totals.items()}
    result.update({"state_mean_sd": float(state.std(0, unbiased=False).mean()), "jepa_mean_sd": float(state[:, :128].std(0, unbiased=False).mean()), "phenotype_mean_sd": float(state[:, 128:].std(0, unbiased=False).mean()), "maximum_gradient_preclip": maximum_grad, "first_adapter_gradient_norm": first_adapter, "first_response_gradient_norm": first_response, "first_phenotype_gradient_norm": first_phenotype, "first_radiomics_head_gradient_norm": first_head, "samples": float(count_total)})
    if not all(np.isfinite(value) for value in result.values()): raise FloatingPointError("non-finite epoch metrics")
    return result


def train_cell(*, seed: int, fold: int, arm: str, radiomics_weight: float, train_ids: Iterable[str], validation_ids: Iterable[str], checkpoint_root: str | Path, device: str, workers: int = 4, base_checkpoint: str | Path | None = None) -> dict[str, Any]:
    protocol = load_protocol(); seed = int(seed); fold = int(fold); arm = str(arm).upper(); effective_seed = seed + fold; set_seed(effective_seed)
    model = FactorizedWorldModel()
    inherited = Path(base_checkpoint) if base_checkpoint else V2_D1_CHECKPOINT_ROOT / f"seed2026_fold{fold}_D1/selected.private.pt"
    inherited_sha = _load_inherited(model, inherited)
    model.ftv_loss_weight = 0.0
    if arm == "F0":
        weight = 0.0
    else:
        weight = float(radiomics_weight)
        if weight <= 0.0: raise ValueError("candidate must have positive radiomics weight")
    model.to(_device(device)); objective = FactorizedObjective(weight).to(_device(device))
    resolved = _device(device)
    dataset_train = SummaryDataset(train_ids, V2_SUMMARY_DIR, fold_target_path(fold)); dataset_val = SummaryDataset(validation_ids, V2_SUMMARY_DIR, fold_target_path(fold))
    train_loader = loader(dataset_train, shuffle=True, seed=effective_seed, workers=workers); val_loader = loader(dataset_val, shuffle=False, seed=effective_seed, workers=workers)
    lr = float(protocol["pilot"]["shared_learning_rate"]); phenotype_lr = float(protocol["pilot"]["phenotype_learning_rate"])
    head = list(model.radiomics_head.parameters()); head_ids = {id(p) for p in head}
    if weight == 0.0: model.radiomics_head.requires_grad_(False)
    shared = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": shared, "lr": lr}]
    if weight > 0.0: groups.append({"params": head, "lr": phenotype_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(protocol["pilot"]["weight_decay"]))
    base_jepa = None
    if weight > 0.0 and base_checkpoint:
        base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        base_jepa = float(base_payload["selected_validation"]["jepa_loss"])
    maximum_jepa = None if base_jepa is None else 1.05 * base_jepa
    best = None; best_score = float("inf"); stale = 0; history = []
    for epoch in range(int(protocol["pilot"]["epochs"])):
        train_stats = run_epoch(model, objective, train_loader, resolved, optimizer, effective_seed + epoch)
        torch.manual_seed(effective_seed + 100000 + epoch)
        validation = run_epoch(model, objective, val_loader, resolved, None, effective_seed + epoch)
        feasible = validation["jepa_mean_sd"] >= float(protocol["pilot"]["minimum_branch_sd"]) and validation["phenotype_mean_sd"] >= float(protocol["pilot"]["minimum_branch_sd"])
        if maximum_jepa is not None: feasible = feasible and validation["jepa_loss"] <= maximum_jepa
        score = validation["jepa_loss"] if weight == 0.0 else validation["radiomics_loss"]
        if not feasible: score = float("inf")
        row = {"epoch": epoch, "train": train_stats, "validation": validation, "feasible": feasible, "selection_score": score}; history.append(row)
        if score < best_score - 1e-12:
            stale = 0; best_score = score; best = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": epoch, "selected_validation": validation}
        else: stale += 1
        if best is not None and stale >= int(protocol["pilot"]["patience"]): break
    cell = Path(checkpoint_root) / f"seed{seed}_fold{fold}_{arm}"; cell.mkdir(parents=True, exist_ok=True)
    atomic_json(cell / "history.private.json", {"history": history})
    if best is None:
        failure = {"status": "NO_FEASIBLE_CHECKPOINT", "seed": seed, "fold": fold, "arm": arm, "radiomics_weight": weight, "inherited_checkpoint_sha256": inherited_sha, "base_jepa_loss": base_jepa, "maximum_allowed_jepa_loss": maximum_jepa, "minimum_jepa_loss": min(float(x["validation"]["jepa_loss"]) for x in history), "history_sha256": canonical_sha(history), "outcome_fields_read": [], "clinical_fields_read": []}
        atomic_json(cell / "cell_failed.private.json", failure); raise RuntimeError(f"no feasible V4 checkpoint: {seed}/{fold}/{arm}")
    checkpoint = cell / "selected.private.pt"
    torch.save({**best, "seed": seed, "fold": fold, "arm": arm, "radiomics_weight": weight, "ftv_loss_weight": 0.0, "inherited_checkpoint_sha256": inherited_sha, "architecture": model.architecture_contract(), "protocol_sha256": sha256_file(EXPERIMENT_ROOT / "configs/protocol.json")}, checkpoint)
    complete = {"status": "COMPLETE", "implementation_version": IMPLEMENTATION_VERSION, "seed": seed, "fold": fold, "arm": arm, "radiomics_weight": weight, "ftv_loss_weight": 0.0, "selected_epoch": int(best["epoch"]), "selected_validation": best["selected_validation"], "checkpoint_sha256": sha256_file(checkpoint), "history_sha256": canonical_sha(history), "inherited_checkpoint_sha256": inherited_sha, "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(cell / "cell_complete.private.json", complete)
    return complete


@torch.inference_mode()
def export_states(*, checkpoint_path: str | Path, patient_ids: Iterable[str], output_path: str | Path, device: str, workers: int = 4) -> dict[str, Any]:
    ids = tuple(sorted(map(str, patient_ids))); model = FactorizedWorldModel(); payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False); model.load_state_dict(payload["model_state"], strict=True); resolved = _device(device); model.to(resolved).eval()
    dataset = SummaryDataset(ids, V2_SUMMARY_DIR, fold_target_path(0)); data_loader = loader(dataset, shuffle=False, seed=0, workers=workers)
    jepa=[]; phenotype=[]; full=[]; rad=[]; observed=[]
    for raw in data_loader:
        summary = raw["summary"].to(resolved, non_blocking=True); response, j, p = model.encode_online(summary); jepa.append(j.float().cpu().numpy()); phenotype.append(p.float().cpu().numpy()); full.append(response.float().cpu().numpy()); rad.append(model.radiomics_head(p).float().cpu().numpy()); observed.extend(map(str, raw["patient_id"]))
    if tuple(observed) != ids: raise AssertionError("state export patient order drifted")
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_name("." + destination.name + ".tmp.npz")
    np.savez_compressed(temporary, patient_id=np.asarray(ids, dtype="U64"), jepa_state=np.concatenate(jepa), phenotype_state=np.concatenate(phenotype), full_state=np.concatenate(full), radiomics_prediction=np.concatenate(rad), checkpoint_sha256=np.asarray(sha256_file(checkpoint_path)))
    temporary.replace(destination)
    return {"status": "COMPLETE", "patients": len(ids), "full_state_shape": list(np.concatenate(full).shape), "state_sha256": sha256_file(destination)}


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
