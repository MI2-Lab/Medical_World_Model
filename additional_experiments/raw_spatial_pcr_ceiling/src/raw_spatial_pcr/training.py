from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_metrics
from .models import RawC1BSupervised, SequenceClassifier, SpatialReadout, fixed_local_mask, load_encoder_weights


TIMING_STEPS = {"T0": 1, "T0_T1": 2, "T0_T2": 3, "T0_T3": 4}
OLD_DATA_SRC = Path(__file__).resolve().parents[3] / "conditional_pcr_contrastive_ceiling" / "src"
if str(OLD_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(OLD_DATA_SRC))
from conditional_ceiling.data import CacheRecord, load_c1b_image  # noqa: E402


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    selected_epoch: int
    train_probability: np.ndarray
    validation_probability: np.ndarray
    test_probability: np.ndarray
    train_row_index: np.ndarray
    validation_row_index: np.ndarray
    test_row_index: np.ndarray
    attention_diagnostics: dict[str, float]


class PrivateManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Streaming patient dataset for raw caches or per-patient spatial maps."""

    def __init__(self, frame: pd.DataFrame, arm: str, timing: str) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.arm = arm
        self.timing = timing
        self.steps = TIMING_STEPS[timing]
        self._raw_cache: dict[int, torch.Tensor] = {}
        self._feature_array = None
        self._raw_array = None
        if arm == "C5" and "raw_array_path" in self.frame.columns:
            raw_path = Path(str(self.frame.iloc[0]["raw_array_path"])).expanduser().resolve(strict=True)
            self._raw_array = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        if arm != "C5" and "feature_array_path" in self.frame.columns:
            array_path = Path(str(self.frame.iloc[0]["feature_array_path"])).expanduser().resolve(strict=True)
            self._feature_array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        required = {"row_index", "label_pcr"}
        if arm == "C5" and self._raw_array is None:
            required.add("cache_path")
        elif arm != "C5" and self._feature_array is None:
            required.add("feature_map_path")
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"private manifest misses {missing}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.arm == "C5" and index in self._raw_cache:
            row = self.frame.iloc[index]
            return self._raw_cache[index], torch.tensor(float(row["label_pcr"]), dtype=torch.float32), torch.tensor(int(row["row_index"]), dtype=torch.int64)
        row = self.frame.iloc[index]
        if self.arm == "C5":
            if self._raw_array is not None:
                value = torch.from_numpy(np.asarray(self._raw_array[int(row["row_index"]), : self.steps], dtype=np.float32))
                self._raw_cache[index] = value
                label = torch.tensor(float(row["label_pcr"]), dtype=torch.float32)
                row_index = torch.tensor(int(row["row_index"]), dtype=torch.int64)
                return value, label, row_index
            cache_path = Path(str(row["cache_path"])).expanduser().resolve(strict=True)
            record = CacheRecord(
                patient_id=str(row["patient_id"]),
                path=cache_path,
                sha256=str(row.get("cache_sha256", "")),
                size_bytes=int(row.get("cache_size_bytes", cache_path.stat().st_size)),
                mtime_ns=int(row.get("cache_mtime_ns", cache_path.stat().st_mtime_ns)),
            )
            image = load_c1b_image(record)
            value = torch.from_numpy(np.asarray(image[: self.steps], dtype=np.float32))
            self._raw_cache[index] = value
        else:
            if self._feature_array is not None:
                value = torch.from_numpy(np.asarray(self._feature_array[int(row["row_index"]), : self.steps], dtype=np.float32))
                label = torch.tensor(float(row["label_pcr"]), dtype=torch.float32)
                row_index = torch.tensor(int(row["row_index"]), dtype=torch.int64)
                return value, label, row_index
            feature_path = Path(str(row["feature_map_path"])).expanduser().resolve(strict=True)
            with np.load(feature_path, allow_pickle=False) as payload:
                if "feature_map" not in payload.files:
                    raise ValueError(f"feature map lacks feature_map: {feature_path}")
                value = torch.from_numpy(np.asarray(payload["feature_map"][: self.steps], dtype=np.float32))
        if value.ndim != 5 or value.shape[0] != self.steps:
            raise ValueError(f"streamed sample has wrong shape: {tuple(value.shape)}")
        label = torch.tensor(float(row["label_pcr"]), dtype=torch.float32)
        row_index = torch.tensor(int(row["row_index"]), dtype=torch.int64)
        return value, label, row_index


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


def _loader(dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False, generator=generator)


def _build_model(arm: str, steps: int, device: torch.device, c5_checkpoint: str | Path | None) -> SequenceClassifier:
    if arm == "C5":
        base = RawC1BSupervised(input_channels=7, base_channels=16, latent_dim=192, dropout=0.1)
        if c5_checkpoint:
            load_encoder_weights(base, c5_checkpoint)
    else:
        base = SpatialReadout(arm, input_dim=128, width=128, dropout=0.1)
    return SequenceClassifier(base, steps=steps, embedding_dim=128, dropout=0.1).to(device)


def _forward(model: SequenceClassifier, batch: torch.Tensor, arm: str, device: torch.device, collect_attention: bool = False) -> dict[str, Any]:
    local_mask = None
    if arm in {"C2", "C3"}:
        local_mask = fixed_local_mask((14, 22, 20)).to(device)
    return model(batch.to(device), local_mask=local_mask, collect_attention=collect_attention)


def _predict(
    model: SequenceClassifier,
    dataset: Dataset,
    frame: pd.DataFrame,
    arm: str,
    device: torch.device,
    batch_size: int,
    seed: int,
    collect_attention: bool = False,
    max_attention_samples: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=seed)
    probabilities: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    diagnostic_values: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        seen = 0
        for batch, _, rows in loader:
            output = _forward(model, batch, arm, device, collect_attention=collect_attention and seen < max_attention_samples)
            probabilities.append(torch.sigmoid(output["logits"]).cpu().numpy().astype(np.float32))
            row_indices.append(rows.numpy().astype(np.int64))
            if collect_attention and seen < max_attention_samples:
                diagnostic_values.extend(_attention_stats(output, arm))
                seen += len(rows)
    diagnostics = _summarize_diagnostics(diagnostic_values)
    return np.concatenate(probabilities), np.concatenate(row_indices), diagnostics


def _attention_stats(output: dict[str, Any], arm: str) -> list[dict[str, float]]:
    if arm not in {"C2", "C3", "C4"}:
        return []
    values: list[dict[str, float]] = []
    visit_embeddings = output.get("visit_embeddings") or []
    longitudinal = 0.0
    if len(visit_embeddings) > 1:
        similarities = []
        for previous, current in zip(visit_embeddings[:-1], visit_embeddings[1:]):
            similarities.append(torch.nn.functional.cosine_similarity(previous.float(), current.float(), dim=-1).mean())
        longitudinal = float(torch.stack(similarities).mean().cpu())
    attentions = output.get("attention") or []
    flattened: list[Any] = []
    for item in attentions:
        if isinstance(item, (list, tuple)):
            flattened.extend(item)
        else:
            flattened.append(item)
    for attention in flattened:
        if attention is None or not isinstance(attention, torch.Tensor) or attention.numel() == 0:
            continue
        # C2 is [B,H,1,L]; C3/C4 are [B,H,N,N], where row zero is CLS.
        if attention.ndim == 4 and attention.shape[-2] == 1:
            probs = attention[:, :, 0, :]
        elif attention.ndim == 4:
            probs = attention[:, :, 0, 1:]
        else:
            continue
        probs = probs.float().clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = (-probs * probs.log()).sum(dim=-1) / np.log(max(int(probs.shape[-1]), 2))
        concentration = probs.max(dim=-1).values
        top_count = max(1, int(np.ceil(probs.shape[-1] * 0.10)))
        top10 = probs.topk(top_count, dim=-1).values.sum(dim=-1)
        coordinates = output.get("coordinates")
        center_mass = torch.full_like(entropy, float("nan"))
        outer_mass = torch.full_like(entropy, float("nan"))
        if isinstance(coordinates, torch.Tensor) and coordinates.ndim == 3 and coordinates.shape[1] == probs.shape[-1]:
            radius = torch.linalg.vector_norm(coordinates.float() - 0.5, dim=-1)
            center = radius <= 0.25
            center_mass = (probs * center[:, None, :].to(probs.dtype)).sum(dim=-1) if center.any() else center_mass
            outer_mass = (probs * (~center)[:, None, :].to(probs.dtype)).sum(dim=-1) if (~center).any() else outer_mass
        values.extend({"attention_entropy": float(e), "attention_concentration": float(c), "attention_concentration_top10": float(t), "center_mass": float(cm), "outer_mass": float(om), "longitudinal_embedding_cosine": longitudinal} for e, c, t, cm, om in zip(entropy.mean(dim=1).cpu(), concentration.mean(dim=1).cpu(), top10.mean(dim=1).cpu(), center_mass.mean(dim=1).cpu(), outer_mass.mean(dim=1).cpu()))
    return values


def _summarize_diagnostics(values: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(values)
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def train_streaming(
    manifest: pd.DataFrame,
    arm: str,
    seed: int,
    fold: int,
    timing: str,
    *,
    device: str = "cuda",
    max_epochs: int = 80,
    patience: int = 10,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int | None = None,
    c5_checkpoint: str | Path | None = None,
) -> TrainingResult:
    """Train one real seed×fold×arm×timing cell without materializing the cohort."""

    if timing not in TIMING_STEPS:
        raise ValueError(f"unregistered timing: {timing}")
    seed_everything(seed + fold * 100003)
    target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    cell = manifest.loc[manifest["fold"].astype(int).eq(int(fold))].copy()
    if len(cell) != 808 or set(cell["split"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError(f"fold {fold} does not contain the frozen 808-patient train/validation/test split")
    steps = TIMING_STEPS[timing]
    train_frame = cell.loc[cell["split"].eq("train")].reset_index(drop=True)
    validation_frame = cell.loc[cell["split"].eq("validation")].reset_index(drop=True)
    test_frame = cell.loc[cell["split"].eq("test")].reset_index(drop=True)
    datasets = {
        "train": PrivateManifestDataset(train_frame, arm, timing),
        "validation": PrivateManifestDataset(validation_frame, arm, timing),
        "test": PrivateManifestDataset(test_frame, arm, timing),
    }
    effective_batch_size = batch_size or (1 if arm in {"C4", "C5"} else 8)
    model = _build_model(arm, steps, target_device, c5_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_labels = train_frame["label_pcr"].to_numpy(float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_class_weight(train_labels).to(target_device))
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -np.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    train_loader = _loader(datasets["train"], batch_size=effective_batch_size, shuffle=True, seed=seed + fold)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch, arm, target_device, collect_attention=False)["logits"]
            loss = criterion(logits, labels.to(target_device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probability, _, _ = _predict(model, datasets["validation"], validation_frame, arm, target_device, effective_batch_size, seed + epoch, collect_attention=False)
        val_auroc = classification_metrics(validation_frame["label_pcr"].to_numpy(), val_probability)["auroc"]
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
    train_probability, train_rows, _ = _predict(model, datasets["train"], train_frame, arm, target_device, effective_batch_size, seed + 11)
    validation_probability, validation_rows, _ = _predict(model, datasets["validation"], validation_frame, arm, target_device, effective_batch_size, seed + 12)
    # Attention maps are a required descriptive diagnostic for the spatial
    # feature-map arms.  C5 uses the same small readout architecture, but its
    # raw-image feature maps make materializing multi-head attention tensors
    # unnecessarily expensive at test time and can trigger CUDA OOM.  The
    # prediction path is unchanged; only optional diagnostic capture is gated.
    collect_attention = arm in {"C2", "C3", "C4"}
    test_probability, test_rows, diagnostics = _predict(
        model,
        datasets["test"],
        test_frame,
        arm,
        target_device,
        effective_batch_size,
        seed + 13,
        collect_attention=collect_attention,
    )
    return TrainingResult(model, history, best_epoch, train_probability, validation_probability, test_probability, train_rows, validation_rows, test_rows, diagnostics)


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
    """Small-array compatibility path retained for unit/smoke tests."""

    frame = pd.DataFrame({"row_index": np.arange(len(labels)), "patient_id": [f"SMOKE-{i}" for i in range(len(labels))], "label_pcr": labels, "split": np.asarray(split).astype(str)})
    # This compatibility entry point intentionally preserves the old tensor API;
    # formal runs use train_streaming so raw archives never become one giant NPZ.
    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim == 5:
        x = x[:, None]
    if x.ndim != 6:
        raise ValueError("compatibility inputs must be [N,V,C,D,H,W] or [N,C,D,H,W]")
    steps = x.shape[1]
    if steps not in {1, 2, 3, 4}:
        raise ValueError("compatibility path supports one to four visits")
    seed_everything(seed)
    target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = _build_model(arm, steps, target_device, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    y = np.asarray(labels, dtype=np.float32)
    train_idx = np.flatnonzero(frame["split"].eq("train"))
    val_idx = np.flatnonzero(frame["split"].eq("validation"))
    test_idx = np.flatnonzero(frame["split"].eq("test"))
    if not all(len(idx) for idx in (train_idx, val_idx, test_idx)):
        raise ValueError("all compatibility splits are required")
    criterion = nn.BCEWithLogitsLoss(pos_weight=_class_weight(y[train_idx]).to(target_device))
    best_state = None
    best_val = -np.inf
    selected_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    train_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])), batch_size=batch_size, shuffle=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch, arm, target_device)["logits"]
            loss = criterion(logits, batch_y.to(target_device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_probability = torch.sigmoid(_forward(model, torch.from_numpy(x[val_idx]), arm, target_device)["logits"]).cpu().numpy()
        val_auroc = classification_metrics(y[val_idx], val_probability)["auroc"]
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_auroc": float(val_auroc)})
        if val_auroc > best_val + 1e-12:
            best_val, selected_epoch, stale = val_auroc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no compatibility checkpoint selected")
    model.load_state_dict(best_state)
    def predict(indices: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(_forward(model, torch.from_numpy(x[indices]), arm, target_device)["logits"]).cpu().numpy().astype(np.float32)
    return TrainingResult(model, history, selected_epoch, predict(train_idx), predict(val_idx), predict(test_idx), train_idx, val_idx, test_idx, {})
