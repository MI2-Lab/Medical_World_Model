from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .model import FactorizedOutput


class SIGReg(nn.Module):
    def __init__(self, projections: int = 256, knots: int = 17) -> None:
        super().__init__()
        points = torch.linspace(0, 3, knots)
        interval = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * interval); weights[[0, -1]] = interval
        self.register_buffer("points", points); self.register_buffer("gaussian", torch.exp(-points.square() / 2)); self.register_buffer("weights", weights * torch.exp(-points.square() / 2))
        self.projections = projections

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.float()
        directions = torch.randn(state.size(-1), self.projections, device=state.device)
        directions = directions / directions.norm(dim=0).clamp_min(1e-6)
        projected = (state @ directions).unsqueeze(-1) * self.points
        error = (projected.cos().mean(-3) - self.gaussian).square() + projected.sin().mean(-3).square()
        return ((error @ self.weights) * state.size(-2)).mean()


def masked_radiomics_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2] or mask[:, 3].any():
        raise ValueError("radiomics target contract failed")
    valid = mask.bool() & torch.isfinite(target).all(-1)
    safe = torch.where(valid[..., None], target, prediction.detach())
    element = F.smooth_l1_loss(prediction, safe, reduction="none").mean(-1)
    patients = valid.any(1)
    if not bool(patients.any()):
        return prediction.sum() * 0.0, valid.sum()
    loss = ((element * valid.to(element.dtype)).sum(1) / valid.sum(1).clamp_min(1))[patients].mean()
    return loss, valid.sum()


class FactorizedObjective(nn.Module):
    def __init__(self, radiomics_weight: float) -> None:
        super().__init__()
        self.radiomics_weight = float(radiomics_weight)
        self.sigreg_weight = 0.09
        self.register_buffer("step_weights", torch.tensor([2.0, 1.0, 0.5]) / (3.5 / 3.0))
        self.sigreg = SIGReg()

    def forward(self, output: FactorizedOutput, radiomics: torch.Tensor, radiomics_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        j_pred = F.layer_norm(output.predicted_jepa, (128,)); j_tgt = F.layer_norm(output.jepa_target[:, 1:], (128,))
        p_pred = F.layer_norm(output.predicted_phenotype, (64,)); p_tgt = F.layer_norm(output.phenotype_target[:, 1:], (64,))
        j_loss = ((j_pred - j_tgt).square().mean(-1) * self.step_weights).mean()
        p_loss = ((p_pred - p_tgt).square().mean(-1) * self.step_weights).mean()
        jepa = 0.5 * (j_loss + p_loss)
        sigreg = self.sigreg(torch.cat((output.jepa_online, output.phenotype_online), -1).transpose(0, 1))
        rad, visits = masked_radiomics_loss(output.radiomics_prediction, radiomics, radiomics_mask)
        total = jepa + self.sigreg_weight * sigreg + self.radiomics_weight * rad
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite objective")
        return total, {"loss": total.detach(), "jepa_loss": jepa.detach(), "jepa_branch_loss": j_loss.detach(), "phenotype_branch_loss": p_loss.detach(), "sigreg_loss": sigreg.detach(), "radiomics_loss": rad.detach(), "radiomics_visits": visits.detach().float()}
