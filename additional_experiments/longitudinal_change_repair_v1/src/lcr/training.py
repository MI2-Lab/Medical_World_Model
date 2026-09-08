from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import ChangeDataset, FTVTransform, bundle_from_config, records
from .model import ChangeRepairModel
from rnc.data import split_ids


ROOT = Path(__file__).resolve().parents[2]

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_config(path: Path) -> dict[str, Any]:
    import yaml
    with path.open() as f: return yaml.safe_load(f)

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(target)
    if not bool(valid.any()): return pred.sum() * 0
    return F.smooth_l1_loss(pred[valid], target[valid])

def sigreg(states: torch.Tensor) -> torch.Tensor:
    # Lightweight variance guard; the original SIGReg remains a historical baseline.
    std = states.std(dim=0, unbiased=False).mean()
    return F.relu(.05 - std)

def run_epoch(model: ChangeRepairModel, loader: DataLoader, optimizer: torch.optim.Optimizer | None, arm: str, device: torch.device, config: dict, history: bool = True) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {key: 0.0 for key in ("loss", "jepa", "static", "observed", "future", "n")}
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []
    for batch in loader:
        image = batch["image"].to(device); static = batch["static"].to(device); smask = batch["static_mask"].to(device)
        change = batch["change"].to(device); cmask = batch["change_mask"].to(device)
        output = model(image, history=history)
        jepa_loss = F.mse_loss(
            F.layer_norm(output.predicted_next_state, (output.predicted_next_state.size(-1),)),
            F.layer_norm(output.target_states[:, 1:], (output.target_states.size(-1),)),
        )
        static_loss = masked_l1(output.static, static, smask)
        observed_loss = masked_l1(output.observed_change, change, cmask)
        # Stage F maps predictions for horizons T1->T2 and T2->T3 to change indices 1 and 2.
        future_loss = masked_l1(output.future_change, change[:, 1:], cmask[:, 1:])
        weights = config["loss"]
        base = weights["lambda_jepa"] * jepa_loss + weights["lambda_sigreg"] * sigreg(output.states)
        if arm == "R0": loss = base
        elif arm == "R1": loss = base + weights["lambda_static"] * static_loss
        elif arm == "R2": loss = base + weights["lambda_static"] * static_loss + weights["lambda_observed_change"] * observed_loss
        else: loss = base + weights["lambda_static"] * static_loss + weights["lambda_observed_change"] * observed_loss + weights["lambda_future_change"] * future_loss
        if training:
            optimizer.zero_grad(set_to_none=True); loss.backward(); clip_grad_norm_(model.parameters(), float(config["train"]["max_grad_norm"])); optimizer.step(); model.update_target(float(config["train"]["ema_momentum"]))
        sums["loss"] += float(loss.detach()); sums["jepa"] += float(jepa_loss.detach()); sums["static"] += float(static_loss.detach()); sums["observed"] += float(observed_loss.detach()); sums["future"] += float(future_loss.detach()); sums["n"] += 1
        with torch.no_grad():
            # R arms evaluate observed change; F arms evaluate future change only.
            pred = output.observed_change if arm.startswith("R") else output.future_change
            tgt = change if arm.startswith("R") else change[:, 1:]
            msk = cmask if arm.startswith("R") else cmask[:, 1:]
            predictions.append(pred[msk].detach().cpu().numpy()); targets.append(tgt[msk].detach().cpu().numpy())
    pred = np.concatenate(predictions) if predictions else np.empty(0); target = np.concatenate(targets) if targets else np.empty(0)
    r2 = float(1 - np.square(pred-target).sum() / max(np.square(target-target.mean()).sum(), 1e-8)) if len(target) > 1 else float("nan")
    return {key: value / max(sums["n"],1) for key,value in sums.items() if key != "n"} | {"r2": r2, "count": int(len(target))}

def train_cell(config_path: Path, arm: str, fold: int, seed_base: int, device_name: str = "cuda", smoke_patients: int | None = None, epochs_override: int | None = None) -> Path:
    if arm not in {"R0","R1","R2","F1","F2","F3","F4"}: raise ValueError("unknown arm")
    config = read_config(config_path); bundle = bundle_from_config(config); splits = split_ids(bundle, fold)
    seed = int(seed_base) + fold; seed_all(seed)
    train_ids = splits["train"] + [x.patient_id for x in bundle.extra_pretrain] if arm.startswith("R") else splits["train"]
    val_ids = splits["val"]
    if smoke_patients:
        train_ids = train_ids[:smoke_patients]; val_ids = val_ids[:max(4, smoke_patients//2)]
    transform = FTVTransform.fit(bundle.raw_radiomics, splits["train"])
    train = ChangeDataset(records(bundle, train_ids), bundle.raw_radiomics, transform); val = ChangeDataset(records(bundle, val_ids), bundle.raw_radiomics, transform)
    loader = DataLoader(train, batch_size=int(config["train"]["batch_size"]), shuffle=True, num_workers=int(config["train"]["workers"])); vloader = DataLoader(val, batch_size=int(config["train"]["batch_size"]), num_workers=int(config["train"]["workers"]))
    m = config["model"]; model = ChangeRepairModel(m["image_channels"],m["base_channels"],m["state_dim"],m["predictor_depth"],m["predictor_heads"],m["predictor_mlp_dim"],m["dropout"])
    device = torch.device(device_name if torch.cuda.is_available() else "cpu"); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    history_mode = arm in {"F2","F4"}; output = ROOT / "checkpoints" / "formal_v3" / arm / f"seed_{seed_base}" / f"fold_{fold}"; output.mkdir(parents=True, exist_ok=False)
    best = float("inf"); stale = 0; rows = []
    for epoch in range(1, int(epochs_override or config["train"]["epochs"])+1):
        tr = run_epoch(model, loader, opt, arm, device, config, history_mode); va = run_epoch(model, vloader, None, arm, device, config, history_mode)
        row = {"epoch":epoch,"train":tr,"val":va}; rows.append(row)
        metric = -va["r2"] if np.isfinite(va["r2"]) else va["observed"]
        if metric < best:
            best=metric; stale=0
            torch.save({"schema":1,"arm":arm,"fold":fold,"seed_base":seed_base,"model":m,"state":model.state_dict(),"transform":transform.__dict__,"splits":splits,"config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),"best_validation":va}, output/"best.pt")
        else: stale+=1
        if stale >= int(config["train"]["patience"]): break
    (output/"history.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)); return output/"best.pt"
