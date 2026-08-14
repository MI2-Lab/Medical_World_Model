from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics
from .models import RawC1BSupervised, SpatialReadout


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    selected_epoch: int
    train_probability: np.ndarray
    validation_probability: np.ndarray
    test_probability: np.ndarray
    attention_diagnostics: dict[str, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weight(labels: np.ndarray) -> torch.Tensor:
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("training split must contain both pCR classes")
    return torch.tensor(negatives / positives, dtype=torch.float32)


def _forward_sequence(model: nn.Module, batch: torch.Tensor, arm: str, local_mask: torch.Tensor | None) -> tuple[torch.Tensor, Any]:
    """Forward a prefix and average visit logits without changing input geometry."""
    if batch.ndim == 6:  # [B,V,C,D,H,W]
        outputs = []
        last = None
        for visit in range(batch.shape[1]):
            current = model(batch[:, visit], local_mask=local_mask) if arm == "C5" else model(batch[:, visit], local_mask=local_mask)
            outputs.append(current["logits"])
            last = current
        return torch.stack(outputs, dim=1).mean(dim=1), last
    current = model(batch, local_mask=local_mask)
    return current["logits"], current


def train_cell(
    inputs: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    arm: str,
    seed: int,
    device: str = "cuda",
    max_epochs: int = 80,
    patience: int = 10,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
) -> TrainingResult:
    """Train one seed×fold×arm×timing cell using train/validation only for selection."""
    seed_everything(seed)
    target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    split = np.asarray(split)
    if x.shape[0] != y.shape[0] or split.shape[0] != y.shape[0]:
        raise ValueError("inputs, labels, and split must align")
    if arm == "C5":
        model: nn.Module = RawC1BSupervised(input_channels=x.shape[-4] if x.ndim == 6 else x.shape[1]).to(target_device)
    else:
        if x.ndim not in {5, 6} or x.shape[-4] != 128:
            raise ValueError("C1-C4 require feature maps with 128 channels")
        model = SpatialReadout(arm, input_dim=128, width=128).to(target_device)
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "validation")
    test_idx = np.flatnonzero(split == "test")
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError("all train/validation/test splits are required")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_class_weight(y[train_idx]).to(target_device))
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -np.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    local_mask = None
    if arm in {"C2", "C3", "C4"} and x.ndim == 6:
        from .models import fixed_local_mask

        local_mask = fixed_local_mask(tuple(x.shape[-3:])).to(target_device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = _forward_sequence(model, batch_x, arm, local_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_x = torch.from_numpy(x[val_idx]).to(target_device)
            val_logits, _ = _forward_sequence(model, val_x, arm, local_mask)
            val_probability = torch.sigmoid(val_logits).cpu().numpy()
        val_auroc = classification_metrics(y[val_idx], val_probability)["auroc"]
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_auroc": float(val_auroc)})
        if val_auroc > best_validation + 1e-12:
            best_validation = val_auroc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()

    def predict(indices: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            current = torch.from_numpy(x[indices]).to(target_device)
            logits, _ = _forward_sequence(model, current, arm, local_mask)
            return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    train_probability = predict(train_idx)
    validation_probability = predict(val_idx)
    test_probability = predict(test_idx)
    return TrainingResult(model, history, best_epoch, train_probability, validation_probability, test_probability, {})

