"""Exact LOCAL3 objective plus a patient-mean static SPH Huber term."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import FTV_WEIGHT, SIGREG_WEIGHT, arm_spec
from .model import ResidualSPHOutput
from .upstream import DGRSObjective


def patient_mean_static_loss(
    prediction: torch.Tensor | None,
    target: torch.Tensor,
    valid: torch.Tensor,
    differentiable_zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth-L1 mean across visits, then equally across valid patients."""

    if target.shape != valid.shape or target.ndim != 2 or target.size(1) != 4:
        raise ValueError("static SPH target/mask must be [B,4]")
    valid = valid.bool() & torch.isfinite(target)
    patient_valid = valid.any(dim=1)
    patient_count = patient_valid.sum()
    visit_count = valid.sum()
    if prediction is None:
        if bool(patient_count):
            raise ValueError("SPH targets are present but the model has no SPH head")
        return differentiable_zero.sum() * 0.0, patient_count, visit_count
    if prediction.shape != target.shape:
        raise ValueError("SPH prediction and target shapes differ")
    if not bool(patient_count):
        return differentiable_zero.sum() * 0.0, patient_count, visit_count
    safe_target = torch.where(valid, target, prediction.detach())
    element = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    element = element * valid.to(prediction.dtype)
    per_patient = element.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(prediction.dtype)
    return per_patient[patient_valid].mean(), patient_count, visit_count


class ResidualSPHObjective(DGRSObjective):
    def __init__(self, experimental_arm: str) -> None:
        self.experimental_arm = arm_spec(experimental_arm).name
        spec = arm_spec(self.experimental_arm)
        super().__init__(
            model_name="G3",
            lambda_ftv=FTV_WEIGHT,
            sigreg_weight=SIGREG_WEIGHT,
            sigreg_projections=256,
            step_weights=(2.0, 1.0, 0.5),
        )
        self.lambda_sph = float(spec.lambda_sph)

    def forward(
        self,
        output: ResidualSPHOutput,
        ftv_target: torch.Tensor,
        ftv_mask: torch.Tensor,
        sph_target: torch.Tensor,
        sph_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, stats = super().forward(output, ftv_target, ftv_mask)
        if self.lambda_sph == 0.0 and bool(sph_mask.any()):
            raise ValueError("S0 must receive an all-false SPH supervision mask")
        sph_loss, sph_patients, sph_visits = patient_mean_static_loss(
            output.sph_prediction,
            sph_target,
            sph_mask,
            output.response_state,
        )
        weighted_sph = self.lambda_sph * sph_loss
        total = stats["_base_component"] + self.lambda_ftv * stats["_ftv_component_raw"] + weighted_sph
        stats.update(
            {
                "loss": total.detach(),
                "sph_loss": sph_loss.detach(),
                "weighted_sph_loss": weighted_sph.detach(),
                "sph_patients": sph_patients.detach().to(torch.float32),
                "sph_valid_visits": sph_visits.detach().to(torch.float32),
                "_sph_component_raw": sph_loss,
                "_sph_component_weighted": weighted_sph,
            }
        )
        return total, stats


def build_objective(experimental_arm: str) -> nn.Module:
    return ResidualSPHObjective(experimental_arm)


__all__ = [
    "ResidualSPHObjective",
    "build_objective",
    "patient_mean_static_loss",
]
