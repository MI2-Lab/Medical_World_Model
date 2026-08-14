"""Locked A1 patch-JEPA, SIGReg, and FTV objective."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import TOKEN_DIM, TRANSITIONS, VISITS
from .model import PatchTokenOutput


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian regularizer used by the G3 model family."""

    def __init__(self, projections: int = 256, knots: int = 17) -> None:
        super().__init__()
        if int(projections) <= 0 or int(knots) < 2:
            raise ValueError("SIGReg projections must be positive and knots >=2")
        self.projections = int(projections)
        points = torch.linspace(0, 3, int(knots), dtype=torch.float32)
        interval = 3 / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * interval, dtype=torch.float32)
        weights[[0, -1]] = interval
        gaussian = torch.exp(-points.square() / 2)
        self.register_buffer("points", points)
        self.register_buffer("gaussian", gaussian)
        self.register_buffer("weights", weights * gaussian)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Regularize ``[..., B, D]`` over its patient axis ``-2``."""

        if state.ndim < 2 or state.shape[-2] <= 0 or state.shape[-1] <= 0:
            raise ValueError("SIGReg state must end in nonempty [B,D] axes")
        if not state.dtype.is_floating_point or not bool(torch.isfinite(state).all()):
            raise ValueError("SIGReg state must contain finite floating values")
        state = state.float()
        directions = torch.randn(state.size(-1), self.projections, device=state.device)
        directions = directions / directions.norm(dim=0).clamp_min(1e-6)
        projected = (state @ directions).unsqueeze(-1) * self.points
        error = (
            projected.cos().mean(-3) - self.gaussian
        ).square() + projected.sin().mean(-3).square()
        return ((error @ self.weights) * state.size(-2)).mean()


def normalized_masked_token_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token LayerNorm MSE and its locked transition-weighted average."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "prediction and target must share shape [B,3,M,128]; got "
            f"{tuple(prediction.shape)}/{tuple(target.shape)}"
        )
    if prediction.shape[1] != TRANSITIONS or prediction.shape[-1] != TOKEN_DIM:
        raise ValueError("masked token tensors violate the [B,3,M,128] contract")
    if target.requires_grad:
        raise ValueError("EMA target_masked must be stop-gradient")
    if step_weights.shape != (TRANSITIONS,):
        raise ValueError("step_weights must have shape [3]")
    normalized_prediction = F.layer_norm(prediction, (TOKEN_DIM,))
    normalized_target = F.layer_norm(target, (TOKEN_DIM,))
    per_token = (normalized_prediction - normalized_target).square().mean(dim=-1)
    # Mean over patients and sampled cells, then weighted mean over transitions.
    per_step = per_token.mean(dim=(0, 2))
    loss = (per_step * step_weights.to(per_step.dtype)).sum() / step_weights.sum()
    return loss, per_step


def patient_mean_ftv_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    differentiable_zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average visit Smooth-L1 within patient, then average valid patients."""

    if prediction.ndim != 2 or tuple(prediction.shape[1:]) != (VISITS,):
        raise ValueError("FTV prediction must have shape [B,4]")
    if target.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError("FTV target/mask must match prediction shape [B,4]")
    if not target.dtype.is_floating_point:
        raise TypeError("FTV target must be floating")
    valid_mask = valid.bool() & torch.isfinite(target)
    patient_valid = valid_mask.any(dim=1)
    patient_count = patient_valid.sum()
    valid_visits = valid_mask.sum()
    if not bool(patient_count):
        return differentiable_zero.sum() * 0.0, patient_count, valid_visits
    safe_target = torch.where(valid_mask, target, prediction.detach())
    element = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    element = element * valid_mask.to(dtype=prediction.dtype)
    per_patient = element.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1).to(
        prediction.dtype
    )
    return per_patient[patient_valid].mean(), patient_count, valid_visits


class PatchTokenObjective(nn.Module):
    """``L_patch + 0.09*SIGReg + 0.25*patient-mean FTV Smooth-L1``."""

    def __init__(
        self,
        *,
        lambda_ftv: float = 0.25,
        sigreg_weight: float = 0.09,
        sigreg_projections: int = 256,
        step_weights: tuple[float, float, float] = (2.0, 1.0, 0.5),
    ) -> None:
        super().__init__()
        if float(lambda_ftv) != 0.25:
            raise ValueError("A1 lambda_ftv is locked to 0.25")
        if float(sigreg_weight) != 0.09:
            raise ValueError("A1 SIGReg weight is locked to 0.09")
        if int(sigreg_projections) != 256:
            raise ValueError("A1 SIGReg projection count is locked to 256")
        weights = torch.tensor(step_weights, dtype=torch.float32)
        if tuple(float(value) for value in weights) != (2.0, 1.0, 0.5):
            raise ValueError("A1 transition weights are locked to (2,1,0.5)")
        self.lambda_ftv = float(lambda_ftv)
        self.sigreg_weight = float(sigreg_weight)
        self.register_buffer("step_weights", weights)
        self.sigreg = SIGReg(sigreg_projections)

    def forward(
        self,
        output: PatchTokenOutput,
        ftv_target: torch.Tensor,
        ftv_mask: torch.Tensor,
        *,
        sigreg_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patch_loss, per_step = normalized_masked_token_mse(
            output.predictions, output.target_masked, self.step_weights
        )
        if output.sigreg_state.ndim != 3 or output.sigreg_state.shape[1:] != (
            VISITS,
            TOKEN_DIM,
        ):
            raise ValueError("sigreg_state must have shape [B,4,128]")
        if sigreg_override is None:
            sigreg_loss = self.sigreg(output.sigreg_state.transpose(0, 1))
        else:
            if (
                not isinstance(sigreg_override, torch.Tensor)
                or sigreg_override.ndim != 0
            ):
                raise ValueError(
                    "sigreg_override must be a differentiable scalar tensor"
                )
            if sigreg_override.device != output.sigreg_state.device:
                raise ValueError("sigreg_override and model output must share a device")
            if not bool(torch.isfinite(sigreg_override)):
                raise ValueError("sigreg_override must be finite")
            sigreg_loss = sigreg_override
        weighted_sigreg = self.sigreg_weight * sigreg_loss
        ftv_loss, ftv_patients, ftv_visits = patient_mean_ftv_smooth_l1(
            output.ftv_prediction,
            ftv_target,
            ftv_mask,
            output.canonical_response,
        )
        weighted_ftv = self.lambda_ftv * ftv_loss
        total = patch_loss + weighted_sigreg + weighted_ftv
        batch = int(output.predictions.shape[0])
        masked_count = int(output.predictions.shape[2])
        count_device = output.predictions.device
        stats = {
            "loss": total.detach(),
            "patch_loss": patch_loss.detach(),
            "patch_loss_t0_t1": per_step[0].detach(),
            "patch_loss_t1_t2": per_step[1].detach(),
            "patch_loss_t2_t3": per_step[2].detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "weighted_sigreg_loss": weighted_sigreg.detach(),
            "ftv_loss": ftv_loss.detach(),
            "weighted_ftv_loss": weighted_ftv.detach(),
            "patients": torch.tensor(float(batch), device=count_device),
            "transitions": torch.tensor(
                float(batch * TRANSITIONS), device=count_device
            ),
            "masked_tokens": torch.tensor(
                float(batch * TRANSITIONS * masked_count), device=count_device
            ),
            "ftv_patients": ftv_patients.detach().to(torch.float32),
            "ftv_valid_visits": ftv_visits.detach().to(torch.float32),
            "sigreg_override_used": torch.tensor(
                float(sigreg_override is not None), device=count_device
            ),
            # Differentiable components used by exact logical-batch accumulation.
            "_patch_component": patch_loss,
            "_sigreg_component": sigreg_loss,
            "_sigreg_component_weighted": weighted_sigreg,
            "_ftv_component_raw": ftv_loss,
            "_ftv_component_weighted": weighted_ftv,
        }
        return total, stats

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "patch": "per_token_LayerNorm_MSE_unweighted_cells",
            "step_weights": [2.0, 1.0, 0.5],
            "step_reduction": "weighted_mean",
            "sigreg_state": "fractional_weighted_mean_projected_online_tokens",
            "sigreg_weight": self.sigreg_weight,
            "sigreg_projections": self.sigreg.projections,
            "ftv": "patient_mean_visit_SmoothL1",
            "lambda_ftv": self.lambda_ftv,
            "delta_ftv_supervision": False,
        }


# Short alias for training integrations.
A1Objective = PatchTokenObjective


__all__ = [
    "A1Objective",
    "PatchTokenObjective",
    "SIGReg",
    "normalized_masked_token_mse",
    "patient_mean_ftv_smooth_l1",
]
