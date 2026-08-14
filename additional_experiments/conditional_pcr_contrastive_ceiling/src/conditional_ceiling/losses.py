"""Conditional supervised contrastive and ceiling objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from .strata import conditional_pair_masks


def conditional_supervised_contrastive_loss(
    embeddings: Any,
    labels: Any,
    stratum_ids: Any,
    eligible_anchor: Any | None = None,
    temperature: float = 0.1,
    *,
    reduction: str = "mean",
) -> Any:
    """Compute self-excluding SupCon entirely within exact strata.

    For eligible anchor ``i``, the denominator contains every ``j != i`` with
    the same stratum and contains no sample from any other stratum. Positives
    are the same-pCR members of that denominator. Eligibility additionally
    requires at least one opposite-pCR denominator member.

    Parameters
    ----------
    embeddings:
        A floating tensor of shape ``[patients, dimensions]``. Rows are
        L2-normalized inside this function.
    labels, stratum_ids:
        Aligned one-dimensional tensors.
    eligible_anchor:
        Optional precomputed boolean selection. A true value that violates the
        exact mathematical eligibility rule is rejected rather than repaired.
    temperature:
        Strictly positive finite scalar; frozen callers use ``0.1``.
    reduction:
        ``"mean"`` (default), ``"sum"``, or ``"none"`` over eligible anchors.
    """

    import torch.nn.functional as functional

    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional tensor [N,D]")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have nonzero patient and feature dimensions")
    if not embeddings.is_floating_point():
        raise ValueError("embeddings must use a floating dtype")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite")
    try:
        resolved_temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a positive finite scalar") from exc
    if not math.isfinite(resolved_temperature) or resolved_temperature <= 0.0:
        raise ValueError("temperature must be a positive finite scalar")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be mean, sum, or none")

    labels = torch.as_tensor(labels, device=embeddings.device)
    stratum_ids = torch.as_tensor(stratum_ids, device=embeddings.device)
    eligible_anchor = (
        None
        if eligible_anchor is None
        else torch.as_tensor(eligible_anchor, device=embeddings.device)
    )
    masks = conditional_pair_masks(labels, stratum_ids, eligible_anchor)
    if masks.positives.shape[0] != embeddings.shape[0]:
        raise ValueError("embedding rows must align with labels and strata")

    features = functional.normalize(embeddings, p=2, dim=1, eps=1e-12)
    logits = torch.matmul(features, features.transpose(0, 1)) / resolved_temperature
    # Masking with -inf makes cross-stratum samples mathematically absent, not
    # merely large-negative approximations in the denominator.
    denominator_logits = logits.masked_fill(~masks.denominator, -torch.inf)
    log_denominator = torch.logsumexp(denominator_logits, dim=1)
    positive_count = masks.positives.sum(dim=1)
    positive_log_probability_sum = torch.where(
        masks.positives,
        logits - log_denominator[:, None],
        torch.zeros((), dtype=logits.dtype, device=logits.device),
    ).sum(dim=1)
    per_anchor = -positive_log_probability_sum / positive_count.clamp_min(1)
    selected = per_anchor[masks.eligible_anchor]
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("conditional supervised contrastive loss is non-finite")
    if reduction == "none":
        return selected
    if reduction == "sum":
        return selected.sum()
    return selected.mean()


class ConditionalSupConLoss(nn.Module):
    """``torch.nn.Module`` wrapper with a frozen default temperature."""

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = str(reduction)

    def forward(
        self,
        embeddings: Any,
        labels: Any,
        stratum_ids: Any,
        eligible_anchor: Any | None = None,
    ) -> Any:
        return conditional_supervised_contrastive_loss(
            embeddings,
            labels,
            stratum_ids,
            eligible_anchor=eligible_anchor,
            temperature=self.temperature,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, reduction={self.reduction!r}"


@dataclass(frozen=True)
class CeilingLossOutput:
    """Scalar tensors for one supervised ceiling objective."""

    total: Any
    conditional_supcon: Any
    pcr_bce: Any
    bce_weight: float


def conditional_ceiling_loss(
    embeddings: Any,
    labels: Any,
    stratum_ids: Any,
    eligible_anchor: Any,
    *,
    arm: str,
    logits: Any | None = None,
    temperature: float = 0.1,
) -> CeilingLossOutput:
    """Apply B1 SupCon-only or B2/B3 SupCon + exactly 0.25 BCE."""

    import torch.nn.functional as functional

    name = str(arm).upper()
    if name not in {"B1", "B2", "B3"}:
        raise ValueError("conditional ceiling loss applies only to B1, B2, or B3")
    contrastive = conditional_supervised_contrastive_loss(
        embeddings,
        labels,
        stratum_ids,
        eligible_anchor=eligible_anchor,
        temperature=temperature,
    )
    weight = 0.0 if name == "B1" else 0.25
    if name == "B1":
        if logits is not None:
            raise ValueError("B1 is SupCon-only and must not receive pCR logits")
        bce = contrastive.new_zeros(())
    else:
        if logits is None:
            raise ValueError("B2/B3 require image-only pCR logits")
        logit_tensor = torch.as_tensor(logits, device=embeddings.device)
        target = torch.as_tensor(
            labels, device=embeddings.device, dtype=logit_tensor.dtype
        )
        if logit_tensor.ndim == 2 and logit_tensor.shape[1] == 1:
            logit_tensor = logit_tensor[:, 0]
        if logit_tensor.ndim != 1 or logit_tensor.shape != target.shape:
            raise ValueError("pCR logits must align one-for-one with labels")
        if not bool(torch.isfinite(logit_tensor).all()):
            raise ValueError("pCR logits must be finite")
        bce = functional.binary_cross_entropy_with_logits(logit_tensor, target)
    total = contrastive + weight * bce
    return CeilingLossOutput(total, contrastive, bce, weight)


# Concise aliases retained for orchestration and exploratory notebooks.
conditional_supcon_loss = conditional_supervised_contrastive_loss
ceiling_loss = conditional_ceiling_loss


__all__ = [
    "CeilingLossOutput",
    "ConditionalSupConLoss",
    "ceiling_loss",
    "conditional_ceiling_loss",
    "conditional_supcon_loss",
    "conditional_supervised_contrastive_loss",
]
