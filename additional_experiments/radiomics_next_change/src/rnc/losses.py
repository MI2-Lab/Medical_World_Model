"""M0/M1/M2 预训练目标与 transition 诊断。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import WorldModelOutput


class SIGReg(nn.Module):
    """复用 clean 分支的 Sketch Isotropic Gaussian regularizer。"""

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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.float()
        directions = torch.randn(state.size(-1), self.projections, device=state.device)
        directions = directions / directions.norm(dim=0).clamp_min(1e-6)
        projected = (state @ directions).unsqueeze(-1) * self.points
        error = (projected.cos().mean(-3) - self.gaussian).square() + projected.sin().mean(-3).square()
        return ((error @ self.weights) * state.size(-2)).mean()


def masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError(f"radiomics prediction/target/mask shape 不一致: {prediction.shape}/{target.shape}/{mask.shape}")
    valid = mask.bool() & torch.isfinite(target)
    safe_target = torch.where(valid, target, prediction.detach())
    element = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    count = valid.sum()
    if not bool(count):
        return prediction.sum() * 0.0, count
    return (element * valid.to(element.dtype)).sum() / count, count


@dataclass(frozen=True)
class ObjectiveWeights:
    delta: float
    state: float
    radiomics: float
    sigreg: float = 0.09


class NextChangeObjective(nn.Module):
    def __init__(
        self,
        mode: str,
        lambda_rad: float = 0.0,
        sigreg_weight: float = 0.09,
        sigreg_projections: int = 256,
        step_weights: tuple[float, float, float] = (2.0, 1.0, 0.5),
    ) -> None:
        super().__init__()
        if mode == "m0":
            self.weights = ObjectiveWeights(0.0, 1.0, 0.0, sigreg_weight)
        elif mode == "m1_delta_only":
            self.weights = ObjectiveWeights(1.0, 0.0, 0.0, sigreg_weight)
        elif mode == "m1":
            self.weights = ObjectiveWeights(1.0, 1.0, 0.0, sigreg_weight)
        elif mode == "m2":
            self.weights = ObjectiveWeights(1.0, 1.0, float(lambda_rad), sigreg_weight)
        else:
            raise ValueError(mode)
        weights = torch.tensor(step_weights, dtype=torch.float32)
        self.register_buffer("step_weights", weights / weights.mean())
        self.sigreg = SIGReg(sigreg_projections)

    def forward(
        self,
        output: WorldModelOutput,
        radiomics_target: torch.Tensor,
        radiomics_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized_prediction = F.layer_norm(output.predicted_next, (output.predicted_next.size(-1),))
        normalized_target = F.layer_norm(output.target_next, (output.target_next.size(-1),))
        state_per_step = (normalized_prediction - normalized_target).square().mean(dim=-1)
        state_loss = (state_per_step * self.step_weights).mean()
        delta_per_step = F.smooth_l1_loss(
            output.predicted_delta, output.target_delta, reduction="none"
        ).mean(dim=-1)
        delta_loss = (delta_per_step * self.step_weights).mean()
        sigreg_loss = self.sigreg(output.online_state.transpose(0, 1))
        if output.radiomics_prediction is None:
            radiomics_loss = output.predicted_delta.sum() * 0.0
            valid_count = torch.zeros((), dtype=torch.long, device=output.predicted_delta.device)
        else:
            radiomics_loss, valid_count = masked_smooth_l1(
                output.radiomics_prediction, radiomics_target, radiomics_mask
            )
        total = (
            self.weights.delta * delta_loss
            + self.weights.state * state_loss
            + self.weights.radiomics * radiomics_loss
            + self.weights.sigreg * sigreg_loss
        )

        copy_error = (output.target_state[:, :-1] - output.target_next).square().mean(dim=-1)
        learned_error = (output.predicted_next - output.target_next).square().mean(dim=-1)
        gain = (copy_error - learned_error) / copy_error.clamp_min(1e-8)
        normalized_copy = F.layer_norm(
            output.target_state[:, :-1], (output.target_state.size(-1),)
        )
        normalized_copy_error = (normalized_copy - normalized_target).square().mean(dim=-1)
        cosine = F.cosine_similarity(output.predicted_delta, output.target_delta, dim=-1, eps=1e-8)
        stats = {
            "loss": total.detach(),
            "state_loss": state_loss.detach(),
            "delta_loss": delta_loss.detach(),
            "radiomics_loss": radiomics_loss.detach(),
            "radiomics_loss_sum": (radiomics_loss * valid_count.to(radiomics_loss.dtype)).detach(),
            "weighted_radiomics_loss": (self.weights.radiomics * radiomics_loss).detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "raw_next_mse": learned_error.mean().detach(),
            "copy_mse": copy_error.mean().detach(),
            "mean_cell_transition_gain": gain.mean().detach(),
            "raw_learned_error_sum": learned_error.sum().detach(),
            "raw_copy_error_sum": copy_error.sum().detach(),
            "normalized_learned_error_sum": state_per_step.sum().detach(),
            "normalized_copy_error_sum": normalized_copy_error.sum().detach(),
            "transition_cells": learned_error.new_tensor(learned_error.numel()).detach(),
            "predicted_delta_norm": output.predicted_delta.norm(dim=-1).mean().detach(),
            "target_delta_norm": output.target_delta.norm(dim=-1).mean().detach(),
            "delta_cosine": cosine.mean().detach(),
            "visit_state_std": output.online_state.std(unbiased=False).detach(),
            # 只沿患者 batch 维计算；不能让固定的 visit/time 差异伪装成跨患者多样性。
            "visit_feature_std": output.online_state.std(dim=0, unbiased=False).mean().detach(),
            "radiomics_valid_elements": valid_count.detach().to(torch.float32),
            # 仅供 runner 在每个 epoch 首 batch 做分项 gradient audit；不写入普通聚合。
            "_image_component": (
                self.weights.delta * delta_loss + self.weights.state * state_loss
            ),
            "_radiomics_component": radiomics_loss,
        }
        return total, stats
