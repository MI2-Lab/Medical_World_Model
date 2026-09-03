"""Outcome-blind D1/D2/D3 training, checkpoint selection, and state export."""

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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .contracts import (
    EXPERIMENT_ROOT,
    FOLD_PATIENTS,
    STATE_SHAPE,
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_protocol,
    patient_order_sha256,
    validate_seed_fold_arm,
)
from .data import FoldTargets, SummaryDataset, validate_state_archive
from .model import MRIAdapterWorldModel, initialization_sha256
from .objective import GroundedJEPAObjective


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def make_loader(
    dataset: SummaryDataset,
    *,
    shuffle: bool,
    seed: int,
    batch_size: int = 32,
    workers: int = 4,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(shuffle and len(dataset) >= batch_size),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
        generator=generator,
    )


def _to_device(batch: Mapping[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def run_epoch(
    model: MRIAdapterWorldModel,
    objective: GroundedJEPAObjective,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    ema: float = 0.996,
    gradient_clip: float = 5.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    objective.train(training)
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    state_moments: list[torch.Tensor] = []
    maximum_gradient = 0.0
    first_adapter_gradient = 0.0
    first_radiomics_head_gradient = 0.0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        count = int(batch["summary"].size(0))
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = device.type == "cuda"
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=autocast
        ):
            output = model(batch["summary"])
            loss, stats = objective(
                output,
                batch["ftv"],
                batch["ftv_mask"],
                batch["radiomics"],
                batch["radiomics_mask"],
            )
        if training:
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"non-finite gradient: {name}")
            if samples == 0:
                first_adapter_gradient = float(
                    sum(
                        parameter.grad.detach().float().square().sum()
                        for parameter in model.adapter.parameters()
                        if parameter.grad is not None
                    ).sqrt()
                )
                first_radiomics_head_gradient = float(
                    sum(
                        parameter.grad.detach().float().square().sum()
                        for parameter in model.radiomics_head.parameters()
                        if parameter.grad is not None
                    ).sqrt()
                )
            norm = clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=float(gradient_clip),
            )
            maximum_gradient = max(maximum_gradient, float(norm))
            optimizer.step()
            model.update_target(float(ema))
        for name, value in stats.items():
            sums[name] += float(value) * count
        state_moments.append(output.response_state.detach().float().cpu())
        samples += count
    if samples == 0:
        raise RuntimeError("epoch loader produced no samples")
    states = torch.cat(state_moments).reshape(-1, 192)
    result = {name: value / samples for name, value in sums.items()}
    result["state_mean_sd"] = float(states.std(dim=0, unbiased=False).mean())
    result["state_min_sd"] = float(states.std(dim=0, unbiased=False).min())
    result["gradient_norm_max_preclip"] = maximum_gradient
    result["first_batch_adapter_gradient_norm"] = first_adapter_gradient
    result["first_batch_radiomics_head_gradient_norm"] = first_radiomics_head_gradient
    result["samples"] = float(samples)
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("epoch metrics are non-finite")
    return result


def _selection_reference(seed: int, fold: int, arm: str, checkpoint_root: Path) -> dict[str, Any]:
    required = {"D1": (), "D2": ("D1",), "D3": ("D1", "D2")}[arm]
    output: dict[str, Any] = {}
    for name in required:
        path = checkpoint_root / f"seed{seed}_fold{fold}_{name}" / "cell_complete.private.json"
        if not path.is_file():
            raise RuntimeError(f"{arm} requires completed paired {name}: {path}")
        output[name] = json.loads(path.read_text(encoding="utf-8"))
    return output


def checkpoint_feasible(
    arm: str,
    validation: Mapping[str, float],
    references: Mapping[str, Mapping[str, Any]],
    minimum_state_sd: float = 0.05,
) -> bool:
    if float(validation["state_mean_sd"]) < float(minimum_state_sd):
        return False
    if arm == "D1":
        return True
    d1_jepa = float(references["D1"]["selected_validation"]["jepa_loss"])
    if float(validation["jepa_loss"]) > 1.05 * d1_jepa:
        return False
    if arm == "D3":
        d2_ftv = float(references["D2"]["selected_validation"]["ftv_loss"])
        if float(validation["ftv_loss"]) > 1.05 * d2_ftv:
            return False
    return True


def selection_score(arm: str, validation: Mapping[str, float]) -> float:
    return float(validation[{"D1": "jepa_loss", "D2": "ftv_loss", "D3": "radiomics_loss"}[arm]])


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def train_cell(
    *,
    seed: int,
    fold: int,
    arm: str,
    train_ids: Iterable[str],
    validation_ids: Iterable[str],
    summary_dir: str | Path,
    targets_path: str | Path,
    checkpoint_root: str | Path,
    device: str = "cuda",
    workers: int = 4,
    epochs: int | None = None,
) -> dict[str, Any]:
    seed, fold, arm = validate_seed_fold_arm(seed, fold, arm)
    protocol = load_protocol()
    train_ids = tuple(map(str, train_ids))
    validation_ids = tuple(map(str, validation_ids))
    targets = FoldTargets.load(targets_path)
    train_dataset = SummaryDataset(train_ids, summary_dir, targets)
    validation_dataset = SummaryDataset(validation_ids, summary_dir, targets)
    effective_seed = seed + fold
    set_seed(effective_seed)
    model = MRIAdapterWorldModel(dropout=float(protocol["model"]["dropout"]))
    init_hash = initialization_sha256(model)
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    objective = GroundedJEPAObjective(arm).to(resolved_device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    train_loader = make_loader(
        train_dataset, shuffle=True, seed=effective_seed, batch_size=32, workers=workers
    )
    validation_loader = make_loader(
        validation_dataset, shuffle=False, seed=effective_seed, batch_size=32, workers=workers
    )
    root = Path(checkpoint_root)
    cell_dir = root / f"seed{seed}_fold{fold}_{arm}"
    references = _selection_reference(seed, fold, arm, root)
    total_epochs = int(protocol["training"]["epochs"] if epochs is None else epochs)
    patience = int(protocol["training"]["patience"])
    best_score = math.inf
    best_epoch = -1
    best_payload: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(total_epochs):
        train_metrics = run_epoch(
            model, objective, train_loader, resolved_device, optimizer,
            ema=float(protocol["training"]["ema"]),
            gradient_clip=float(protocol["training"]["gradient_clip"]),
        )
        # Fixed validation SIGReg randomness makes paired checkpoint comparisons reproducible.
        torch.manual_seed(effective_seed + 100000 + epoch)
        validation_metrics = run_epoch(
            model, objective, validation_loader, resolved_device, None,
            ema=float(protocol["training"]["ema"]),
            gradient_clip=float(protocol["training"]["gradient_clip"]),
        )
        feasible = checkpoint_feasible(
            arm, validation_metrics, references,
            minimum_state_sd=float(protocol["training"]["minimum_state_sd"]),
        )
        score = selection_score(arm, validation_metrics) if feasible else math.inf
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "feasible": feasible, "selection_score": score})
        if score < best_score - 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_payload = {
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "epoch": epoch,
                "validation": validation_metrics,
            }
        elif best_payload is not None:
            stale += 1
        if best_payload is not None and stale >= patience:
            break
    if best_payload is None:
        d1_jepa = None
        if arm != "D1":
            d1_jepa = float(references["D1"]["selected_validation"]["jepa_loss"])
        failure = {
            "schema_version": 1,
            "status": "NO_FEASIBLE_CHECKPOINT",
            "seed": seed,
            "fold": fold,
            "arm": arm,
            "epochs_completed": len(history),
            "minimum_observed_validation_jepa_loss": min(
                float(epoch["validation"]["jepa_loss"]) for epoch in history
            ),
            "paired_d1_validation_jepa_loss": d1_jepa,
            "maximum_allowed_validation_jepa_loss": None if d1_jepa is None else 1.05 * d1_jepa,
            "minimum_observed_state_mean_sd": min(
                float(epoch["validation"]["state_mean_sd"]) for epoch in history
            ),
            "history_sha256": canonical_sha256(history),
            "outcome_fields_read": [],
            "clinical_fields_read": [],
        }
        atomic_json(cell_dir / "history.private.json", {"history": history})
        atomic_json(cell_dir / "cell_failed.private.json", failure)
        raise RuntimeError(f"{seed}/{fold}/{arm} has no feasible checkpoint")
    checkpoint_path = cell_dir / "selected.private.pt"
    _atomic_torch_save(
        checkpoint_path,
        {
            **best_payload,
            "seed": seed,
            "fold": fold,
            "arm": arm,
            "effective_seed": effective_seed,
            "initialization_sha256": init_hash,
            "architecture": model.architecture_contract(),
            "protocol_sha256": file_sha256(EXPERIMENT_ROOT / "configs/protocol.json"),
        },
    )
    complete = {
        "schema_version": 1,
        "status": "COMPLETE",
        "seed": seed,
        "fold": fold,
        "arm": arm,
        "effective_seed": effective_seed,
        "selected_epoch": best_epoch,
        "selected_validation": best_payload["validation"],
        "initialization_sha256": init_hash,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "train_patient_order_sha256": patient_order_sha256(train_ids),
        "validation_patient_order_sha256": patient_order_sha256(validation_ids),
        "history_sha256": canonical_sha256(history),
        "elapsed_seconds": time.monotonic() - started,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    atomic_json(cell_dir / "history.private.json", {"history": history})
    atomic_json(cell_dir / "cell_complete.private.json", complete)
    return complete


@torch.inference_mode()
def export_cell_states(
    *,
    checkpoint_path: str | Path,
    patient_ids: Iterable[str],
    summary_dir: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    workers: int = 4,
) -> dict[str, Any]:
    ids = tuple(sorted(map(str, patient_ids)))
    if len(ids) != FOLD_PATIENTS:
        raise ValueError("formal state export requires all 808 I-SPY2 patients")
    resolved = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MRIAdapterWorldModel()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(resolved).eval()
    dataset = SummaryDataset(ids, summary_dir, targets=None)
    loader = make_loader(dataset, shuffle=False, seed=0, batch_size=32, workers=workers)
    states: list[np.ndarray] = []
    radiomics: list[np.ndarray] = []
    ftv: list[np.ndarray] = []
    observed_ids: list[str] = []
    for batch in loader:
        summary = batch["summary"].to(resolved, non_blocking=True)
        response = model.encode_response(summary)
        states.append(response.float().cpu().numpy())
        radiomics.append(model.radiomics_head(response).float().cpu().numpy())
        ftv.append(model.ftv_head(response).squeeze(-1).float().cpu().numpy())
        observed_ids.extend(map(str, batch["patient_id"]))
    if tuple(observed_ids) != ids:
        raise AssertionError("state export patient order drifted")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        patient_id=np.asarray(ids, dtype="U32"),
        state=np.concatenate(states).astype(np.float32),
        radiomics_prediction=np.concatenate(radiomics).astype(np.float32),
        ftv_prediction=np.concatenate(ftv).astype(np.float32),
        checkpoint_sha256=np.asarray(file_sha256(checkpoint_path)),
    )
    _, state = validate_state_archive(destination)
    return {
        "status": "COMPLETE",
        "patients": len(ids),
        "shape": list(state.shape),
        "state_mean_sd": float(state.reshape(-1, 192).std(axis=0).mean()),
        "sha256": file_sha256(destination),
    }


__all__ = [
    "checkpoint_feasible", "export_cell_states", "make_loader", "run_epoch",
    "selection_score", "set_seed", "train_cell"
]
