from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_radiomics_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2] or mask[:, 3].any():
        raise ValueError("radiomics target contract failed")
    valid = mask.bool() & torch.isfinite(target).all(-1)
    safe = torch.where(valid[..., None], target, prediction.detach())
    element = F.smooth_l1_loss(prediction, safe, reduction="none").mean(-1)
    patients = valid.any(1)
    if not bool(patients.any()): return prediction.sum() * 0.0, valid.sum()
    return ((element * valid.to(element.dtype)).sum(1) / valid.sum(1).clamp_min(1))[patients].mean(), valid.sum()
