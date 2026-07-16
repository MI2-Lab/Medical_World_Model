from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..config import LossConfig
from ..models import CoReJEPAOutput


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian regularizer used by LeWorldModel."""

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
        """Input ``state [visit,batch,Dz]``; output a scalar."""

        directions = torch.randn(state.size(-1), self.projections, device=state.device)
        directions = directions / directions.norm(dim=0).clamp_min(1e-6)
        projected = (state @ directions).unsqueeze(-1) * self.points
        error = (projected.cos().mean(-3) - self.gaussian).square() + projected.sin().mean(-3).square()
        return ((error @ self.weights) * state.size(-2)).mean()


def _normalized_weights(values: tuple[float, ...], reference: torch.Tensor) -> torch.Tensor:
    weights = reference.new_tensor(values)
    return weights / weights.mean().clamp_min(1e-6)


def _weighted_mean(per_sample_step: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (per_sample_step * weights.view(1, -1)).mean()


def _masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a ``[B,steps]`` loss while preserving missing target entries."""

    valid = torch.isfinite(target)
    safe_target = torch.where(valid, target, prediction.detach())
    element = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    count = valid.to(element.dtype).sum(dim=-1).clamp_min(1.0)
    output = (element * valid.to(element.dtype)).sum(dim=-1) / count
    return torch.where(valid.any(dim=-1), output, torch.zeros_like(output))


def _continuous_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    margin: float,
    minimum_difference: float,
) -> torch.Tensor:
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    prediction, target = prediction[finite], target[finite].detach()
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    target_difference = target[:, None] - target[None, :]
    prediction_difference = prediction[:, None] - prediction[None, :]
    valid = target_difference > minimum_difference
    if not bool(valid.any()):
        return prediction.new_zeros(())
    weights = target_difference[valid].clamp(min=minimum_difference, max=4.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return (F.softplus(margin - prediction_difference[valid]) * weights).mean()


def _scalar_regression_and_rank(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_weights: torch.Tensor,
    regression_weight: float,
    ranking_weight: float,
    margin: float,
    minimum_difference: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step_losses, regressions, rankings = [], [], []
    for step in range(prediction.size(1)):
        regression = F.smooth_l1_loss(prediction[:, step], target[:, step])
        ranking = _continuous_rank_loss(
            prediction[:, step], target[:, step], margin, minimum_difference
        )
        step_losses.append(regression_weight * regression + ranking_weight * ranking)
        regressions.append(regression)
        rankings.append(ranking)
    step_loss = torch.stack(step_losses)
    regression = torch.stack(regressions)
    ranking = torch.stack(rankings)
    return (
        (step_loss * step_weights).mean(),
        (regression * step_weights).mean(),
        (ranking * step_weights).mean(),
    )


def _continuous_contrast(
    features: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    target_temperature: float,
    condition: torch.Tensor | None,
    mismatch_penalty: float,
) -> torch.Tensor:
    """Soft neighborhood matching from response change to state change."""

    if features.size(0) < 2:
        return features.new_zeros(())
    features = F.normalize(features, dim=-1)
    logits = features @ features.t() / temperature
    eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(eye, -1e9)
    log_probability = F.log_softmax(logits - logits.max(dim=1, keepdim=True).values.detach(), dim=1)
    target = target.reshape(target.size(0), -1).to(features)
    target_logits = -torch.cdist(target, target) / target_temperature
    if condition is not None and mismatch_penalty > 0:
        mismatch = condition[:, None] != condition[None, :]
        target_logits = target_logits - mismatch_penalty * mismatch.to(target_logits.dtype)
    target_probability = F.softmax(target_logits.masked_fill(eye, -1e9), dim=1).detach()
    return -(target_probability * log_probability).sum(dim=1).mean()


class PretrainingObjective(nn.Module):
    """Selected paper-v1 objective. No input or term contains pCR."""

    def __init__(
        self,
        config: LossConfig,
        sigreg_projections: int,
        routing_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sigreg = SIGReg(sigreg_projections)
        if routing_class_weights is None:
            self.register_buffer("routing_class_weights", torch.empty(0))
        else:
            self.register_buffer("routing_class_weights", routing_class_weights.float())

    def forward(
        self,
        output: CoReJEPAOutput,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        config = self.config
        prediction_weights = _normalized_weights(config.prediction_steps, output.prediction)
        response_weights = _normalized_weights(config.response_steps, output.prediction)
        update_weights = _normalized_weights(config.update_steps, output.prediction)

        raw_prediction_error = (output.prediction - output.target).square().mean(dim=-1)
        normalized_prediction = F.layer_norm(output.prediction, (output.prediction.size(-1),))
        normalized_target = F.layer_norm(output.target, (output.target.size(-1),))
        prediction_error = (normalized_prediction - normalized_target).square().mean(dim=-1)
        prediction_loss = _weighted_mean(prediction_error, prediction_weights)
        sigreg_loss = self.sigreg(output.visit_state.transpose(0, 1))

        response_score = batch["response_score"].to(output.prediction)[..., 0]
        response_vector = batch["response_vector"].to(output.prediction)
        routing_target = batch["routing_target"].to(output.gate_logits.device, dtype=torch.long)

        score_loss, score_regression, score_ranking = _scalar_regression_and_rank(
            output.score_prediction,
            response_score,
            response_weights,
            config.score_regression,
            config.score_ranking,
            config.score_rank_margin,
            config.min_rank_target_difference,
        )
        vector_loss = _weighted_mean(
            _masked_smooth_l1(output.vector_prediction, response_vector), response_weights
        )

        score_update_target = torch.stack(
            (response_score[:, 0], response_score[:, 1] - response_score[:, 0]), dim=1
        )
        vector_update_target = torch.stack(
            (response_vector[:, 0], response_vector[:, 1] - response_vector[:, 0]), dim=1
        )
        update_score_loss, update_score_regression, update_score_ranking = _scalar_regression_and_rank(
            output.update_score_prediction,
            score_update_target,
            update_weights,
            config.score_regression,
            config.score_ranking,
            config.update_rank_margin,
            config.min_rank_target_difference,
        )
        update_vector_loss = _weighted_mean(
            _masked_smooth_l1(output.update_vector_prediction, vector_update_target), update_weights
        )

        state_delta = output.future_response_state[:, 1:] - output.future_response_state[:, :-1]
        score_delta = response_score[:, 1:] - response_score[:, :-1]
        contrast_per_step = []
        for step in range(state_delta.size(1)):
            contrast_per_step.append(
                _continuous_contrast(
                    state_delta[:, step],
                    score_delta[:, step],
                    config.contrast_temperature,
                    config.contrast_target_temperature,
                    routing_target,
                    config.contrast_condition_penalty,
                )
            )
        delta_contrast = (torch.stack(contrast_per_step) * update_weights).mean()

        class_weights = self.routing_class_weights
        if class_weights.numel() == 0:
            class_weights = None
        route_by_step = torch.stack(
            [
                F.cross_entropy(output.gate_logits[:, step], routing_target, weight=class_weights)
                for step in range(output.gate_logits.size(1))
            ]
        )
        route_loss = (route_by_step * prediction_weights).mean()
        probabilities = output.gate_probabilities
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_loss = entropy.mean() / math.log(probabilities.size(-1))
        mean_use = probabilities.mean(dim=(0, 1))
        balance_loss = ((mean_use - 1.0 / probabilities.size(-1)) ** 2).sum()

        total = (
            config.prediction * prediction_loss
            + config.sigreg * sigreg_loss
            + config.response_score * score_loss
            + config.state_delta_contrast * delta_contrast
            + config.update_score * update_score_loss
            + config.response_vector * vector_loss
            + config.response_vector_update * update_vector_loss
            + config.gate_route * route_loss
            + config.gate_entropy * entropy_loss
            + config.gate_balance * balance_loss
        )
        stats = {
            "loss": float(total.detach()),
            "prediction": float(prediction_loss.detach()),
            "prediction_raw_mse": float(raw_prediction_error.mean().detach()),
            "sigreg": float(sigreg_loss.detach()),
            "response_score": float(score_loss.detach()),
            "response_score_regression": float(score_regression.detach()),
            "response_score_ranking": float(score_ranking.detach()),
            "response_vector": float(vector_loss.detach()),
            "update_score": float(update_score_loss.detach()),
            "update_score_regression": float(update_score_regression.detach()),
            "update_score_ranking": float(update_score_ranking.detach()),
            "update_vector": float(update_vector_loss.detach()),
            "state_delta_contrast": float(delta_contrast.detach()),
            "gate_route": float(route_loss.detach()),
            "gate_entropy": float(entropy_loss.detach()),
            "gate_balance": float(balance_loss.detach()),
            "visit_state_std": float(output.visit_state.std(unbiased=False).detach()),
            "future_response_state_std": float(output.future_response_state.std(unbiased=False).detach()),
            "response_correction_norm": float(output.response_correction.norm(dim=-1).mean().detach()),
        }
        return total, stats
