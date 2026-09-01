"""JEPA, SIGReg, FTV, and direct residual-radiomics objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import ARMS
from .model import WorldModelOutput


class SIGReg(nn.Module):
    def __init__(self, projections: int = 256, knots: int = 17) -> None:
        super().__init__()
        self.projections = int(projections)
        points = torch.linspace(0, 3, knots, dtype=torch.float32)
        interval = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * interval, dtype=torch.float32)
        weights[[0, -1]] = interval
        gaussian = torch.exp(-points.square() / 2)
        self.register_buffer("points", points)
        self.register_buffer("gaussian", gaussian)
        self.register_buffer("weights", weights * gaussian)

    def forward(self, state_vbd: torch.Tensor) -> torch.Tensor:
        state = state_vbd.float()
        directions = torch.randn(state.size(-1), self.projections, device=state.device)
        directions = directions / directions.norm(dim=0).clamp_min(1e-6)
        projected = (state @ directions).unsqueeze(-1) * self.points
        error = (
            (projected.cos().mean(-3) - self.gaussian).square()
            + projected.sin().mean(-3).square()
        )
        return ((error @ self.weights) * state.size(-2)).mean()


def masked_patient_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape[:2] != visit_mask.shape or target.shape[:2] != visit_mask.shape:
        raise ValueError("grounding prediction/target/mask shape mismatch")
    if prediction.shape != target.shape:
        raise ValueError("grounding prediction and target shapes differ")
    finite = torch.isfinite(target)
    if target.ndim == 3:
        valid = visit_mask.bool() & finite.all(dim=-1)
        safe = torch.where(valid[..., None], target, prediction.detach())
        element = F.smooth_l1_loss(prediction, safe, reduction="none").mean(dim=-1)
    elif target.ndim == 2:
        valid = visit_mask.bool() & finite
        safe = torch.where(valid, target, prediction.detach())
        element = F.smooth_l1_loss(prediction, safe, reduction="none")
    else:
        raise ValueError("grounding tensors must be [B,4] or [B,4,D]")
    patients = valid.any(dim=1)
    if not bool(patients.any()):
        return prediction.sum() * 0.0, patients.sum(), valid.sum()
    per_patient = (element * valid.to(element.dtype)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return per_patient[patients].mean(), patients.sum(), valid.sum()


@dataclass(frozen=True)
class ObjectiveWeights:
    ftv: float
    radiomics: float


class GroundedJEPAObjective(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        arm = str(arm).upper()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.arm = arm
        self.weights = ObjectiveWeights(
            ftv={"D1": 0.0, "D2": 0.25, "D3": 0.25}[arm],
            radiomics={"D1": 0.0, "D2": 0.0, "D3": 0.10}[arm],
        )
        self.sigreg_weight = 0.09
        self.register_buffer("step_weights", torch.tensor([2.0, 1.0, 0.5]) / (3.5 / 3.0))
        self.sigreg = SIGReg(256)

    def forward(
        self,
        output: WorldModelOutput,
        ftv_target: torch.Tensor,
        ftv_mask: torch.Tensor,
        radiomics_target: torch.Tensor,
        radiomics_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if radiomics_mask.shape[1] != 4 or bool(radiomics_mask[:, 3].any()):
            raise ValueError("V2 radiomics grounding is restricted to T0-T2")
        prediction = F.layer_norm(output.predicted_next, (192,))
        target = F.layer_norm(output.target_next, (192,))
        per_step = (prediction - target).square().mean(dim=-1)
        jepa = (per_step * self.step_weights).mean()
        sigreg = self.sigreg(output.online_state.transpose(0, 1))
        ftv, ftv_patients, ftv_visits = masked_patient_smooth_l1(
            output.ftv_prediction, ftv_target, ftv_mask
        )
        radiomics, rad_patients, rad_visits = masked_patient_smooth_l1(
            output.radiomics_prediction, radiomics_target, radiomics_mask
        )
        total = jepa + self.sigreg_weight * sigreg + self.weights.ftv * ftv + self.weights.radiomics * radiomics
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("objective is non-finite")
        return total, {
            "loss": total.detach(),
            "jepa_loss": jepa.detach(),
            "sigreg_loss": sigreg.detach(),
            "ftv_loss": ftv.detach(),
            "radiomics_loss": radiomics.detach(),
            "ftv_patients": ftv_patients.detach().float(),
            "ftv_visits": ftv_visits.detach().float(),
            "radiomics_patients": rad_patients.detach().float(),
            "radiomics_visits": rad_visits.detach().float(),
        }


__all__ = ["GroundedJEPAObjective", "ObjectiveWeights", "SIGReg", "masked_patient_smooth_l1"]
