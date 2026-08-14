"""Leak-resistant training utilities for one supervised ceiling cell.

The image model never receives clinical or treatment variables. They are used
only by :mod:`conditional_ceiling.strata` to construct an outer-train loss
mask. Checkpoint selection reads validation labels but never test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import PRIMARY_TIMINGS
from .losses import conditional_supervised_contrastive_loss


@dataclass(frozen=True)
class TrainingHyperparameters:
    epochs: int
    patience: int
    encoder_learning_rate: float
    head_learning_rate: float
    weight_decay: float = 1e-4
    bce_weight: float = 0.25
    max_grad_norm: float = 5.0

    def validate(self, arm: str) -> None:
        if arm not in {"B2", "B3"}:
            raise ValueError("image adaptation hyperparameters apply only to B2/B3")
        if self.epochs <= 0 or self.patience <= 0:
            raise ValueError("epochs and patience must be positive")
        if self.encoder_learning_rate <= 0 or self.head_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("weight decay/max_grad_norm are invalid")
        if self.bce_weight != 0.25:
            raise ValueError("B2/B3 BCE weight is frozen to exactly 0.25")


def _prefix_logits(model: Any, response: Any) -> list[Any]:
    """Return low-capacity training-only logits for T0, T0-T1, T0-T2."""

    projected = model.projection_head(response[:, :3])
    return [
        model.pcr_heads[index](projected[:, : index + 1].reshape(len(projected), -1))
        .squeeze(-1)
        for index in range(3)
    ]


def ceiling_loss(
    model: Any,
    response: Any,
    labels: Any,
    stratum_ids: Any,
    eligible_anchor: Any,
    *,
    arm: str,
    temperature: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    """Compute the primary conditional objective over T0/T1/T2.

    SupCon is evaluated on each literal observed prefix (T0, T0-T1, T0-T2)
    after projecting each visit, then averaged. No future visit can influence
    an earlier timing. B2/B3 add mean prefix BCE at exactly 0.25. B1
    intentionally has no BCE path.
    """

    import torch
    import torch.nn.functional as functional

    if arm not in {"B1", "B2", "B3"}:
        raise ValueError("ceiling_loss is defined only for supervised arms")
    if response.ndim != 3 or response.shape[1] < 3 or response.shape[2] != 192:
        raise ValueError("response must have shape [B,>=3,192]")
    embeddings = model.projection_head(response[:, :3])
    contrastive_terms = [
        conditional_supervised_contrastive_loss(
            embeddings[:, :end].reshape(len(embeddings), end * embeddings.shape[-1]),
            labels,
            stratum_ids,
            eligible_anchor=eligible_anchor,
            temperature=temperature,
        )
        for end in range(1, 4)
    ]
    contrastive = torch.stack(contrastive_terms).mean()
    bce = response.new_zeros(())
    if arm in {"B2", "B3"}:
        logits = _prefix_logits(model, response)
        bce = torch.stack(
            [functional.binary_cross_entropy_with_logits(value, labels.float()) for value in logits]
        ).mean()
    total = contrastive + (0.25 * bce if arm in {"B2", "B3"} else 0.0)
    return total, {
        "conditional_supcon": float(contrastive.detach().cpu()),
        "pcr_bce": float(bce.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def _validation_mean_auroc(model: Any, state: Any, labels: Any) -> float:
    from sklearn.metrics import roc_auc_score
    import torch

    model.eval()
    with torch.no_grad():
        logits = _prefix_logits(model, state)
        probabilities = [torch.sigmoid(value).detach().cpu().numpy() for value in logits]
    y = labels.detach().cpu().numpy().astype(np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("validation labels must contain both classes")
    return float(np.mean([roc_auc_score(y, value) for value in probabilities]))


def _linear_probe_validation_mean_auroc(
    model: Any,
    train_state: Any,
    train_labels: Any,
    validation_state: Any,
    validation_labels: Any,
) -> float:
    """Validation selection probe for B1; no BCE gradient enters the head."""

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    model.eval()
    with torch.no_grad():
        train_projected = model.projection_head(train_state[:, :3]).detach().cpu().numpy()
        validation_projected = (
            model.projection_head(validation_state[:, :3]).detach().cpu().numpy()
        )
    train_y = train_labels.detach().cpu().numpy().astype(np.int64)
    validation_y = validation_labels.detach().cpu().numpy().astype(np.int64)
    if set(np.unique(train_y)) != {0, 1} or set(np.unique(validation_y)) != {0, 1}:
        raise ValueError("B1 selection probe requires both pCR classes")
    scores: list[float] = []
    for end in range(1, 4):
        train_x = train_projected[:, :end].reshape(len(train_projected), -1)
        validation_x = validation_projected[:, :end].reshape(
            len(validation_projected), -1
        )
        scaler = StandardScaler().fit(train_x)
        classifier = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=10_000, random_state=0
        ).fit(scaler.transform(train_x), train_y)
        probability = classifier.predict_proba(scaler.transform(validation_x))[:, 1]
        scores.append(float(roc_auc_score(validation_y, probability)))
    return float(np.mean(scores))


def train_b1_from_frozen_states(
    model: Any,
    train_state: Any,
    train_labels: Any,
    train_stratum_ids: Any,
    eligible_anchor: Any,
    validation_state: Any,
    validation_labels: Any,
    *,
    epochs: int = 80,
    patience: int = 12,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Train B1 without loading images or changing LOCAL3."""

    import torch

    if epochs <= 0 or patience <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid B1 hyperparameters")
    if tuple(train_state.shape[1:]) != (4, 192):
        raise ValueError("train_state must be [N,4,192]")
    optimizer = torch.optim.AdamW(
        model.projection_head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        response = train_state
        total, components = ceiling_loss(
            model,
            response,
            train_labels,
            train_stratum_ids,
            eligible_anchor,
            arm="B1",
            temperature=temperature,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.projection_head.parameters(), 5.0)
        optimizer.step()
        score = _linear_probe_validation_mean_auroc(
            model,
            train_state,
            train_labels,
            validation_state,
            validation_labels,
        )
        history.append({"epoch": epoch, "validation_mean_auroc": score, **components})
        print(
            f"TRAIN_PROGRESS arm=B1 epoch={epoch} "
            f"validation_mean_auroc={score:.6f} loss={components['total']:.6f}",
            flush=True,
        )
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("B1 training produced no selectable epoch")
    model.load_state_dict(best_state, strict=True)
    return {
        "arm": "B1",
        "selected_epoch": best_epoch,
        "selected_validation_mean_auroc": best_score,
        "selection_timings": list(PRIMARY_TIMINGS),
        "test_labels_used": False,
        "history": history,
    }


def _parameter_groups(model: Any, arm: str, hyperparameters: TrainingHyperparameters) -> list[dict[str, Any]]:
    encoder: list[Any] = []
    heads: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.encoder"):
            encoder.append(parameter)
        else:
            heads.append(parameter)
    if not encoder or not heads:
        raise ValueError(f"{arm} must contain trainable encoder and head parameters")
    return [
        {"params": encoder, "lr": hyperparameters.encoder_learning_rate},
        {"params": heads, "lr": hyperparameters.head_learning_rate},
    ]


def train_adaptation_cell(
    model: Any,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    hyperparameters: TrainingHyperparameters,
    device: Any,
    temperature: float = 0.1,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    """Train one B2/B3 fold from images with validation-only selection.

    Batches must already be sampled from exact training strata and contain
    ``image``, ``label``, ``stratum_id``, and ``eligible_anchor``. No clinical
    tensors are accepted or passed to the model.
    """

    import torch

    hyperparameters.validate(arm)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, arm, hyperparameters),
        weight_decay=hyperparameters.weight_decay,
    )
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, hyperparameters.epochs + 1):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch - 1)
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            labels = batch["label"].to(device)
            strata = batch["stratum_id"].to(device)
            eligible = batch["eligible_anchor"].to(device)
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"]
            micro = len(images) if microbatch_size is None else int(microbatch_size)
            if micro <= 0:
                raise ValueError("microbatch_size must be positive")
            response_chunks: list[Any] = []
            for start in range(0, len(images), micro):
                image = images[start : start + micro].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    if arm == "B3":
                        from torch.utils.checkpoint import checkpoint

                        response_chunks.append(
                            checkpoint(
                                model.encode_response,
                                image,
                                use_reentrant=False,
                                preserve_rng_state=True,
                            )
                        )
                    else:
                        response_chunks.append(model.encode_response(image))
            response = torch.cat(response_chunks)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                total, _ = ceiling_loss(
                    model,
                    response,
                    labels,
                    strata,
                    eligible,
                    arm=arm,
                    temperature=temperature,
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                hyperparameters.max_grad_norm,
            )
            optimizer.step()
            loss_sum += float(total.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("training loader is empty")

        states: list[Any] = []
        labels_list: list[Any] = []
        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                states.append(model.encode_response(batch["image"].to(device)))
                labels_list.append(batch["label"].to(device))
        if not states:
            raise ValueError("validation loader is empty")
        score = _validation_mean_auroc(
            model, torch.cat(states), torch.cat(labels_list)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / batches,
                "validation_mean_auroc": score,
            }
        )
        print(
            f"TRAIN_PROGRESS arm={arm} epoch={epoch} "
            f"validation_mean_auroc={score:.6f} train_loss={loss_sum / batches:.6f}",
            flush=True,
        )
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= hyperparameters.patience:
                break
    if best_state is None:
        raise RuntimeError("adaptation training produced no selectable epoch")
    model.load_state_dict(best_state, strict=True)
    return {
        "arm": arm,
        "selected_epoch": best_epoch,
        "selected_validation_mean_auroc": best_score,
        "selection_timings": list(PRIMARY_TIMINGS),
        "test_labels_used": False,
        "history": history,
    }


def save_private_checkpoint(
    path: str | Path,
    model: Any,
    selection: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Persist a non-public checkpoint, refusing overwrite."""

    import os
    import tempfile
    import torch

    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    try:
        torch.save(
            {
                "schema_version": 1,
                "state_dict": model.state_dict(),
                "selection": dict(selection),
                "provenance": dict(provenance),
                "pcr_supervised": True,
                "world_model_claim_allowed": False,
            },
            temporary,
        )
        Path(temporary).replace(target)
        target.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)
