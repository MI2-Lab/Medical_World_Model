"""Goal-F representation losses, including logical-batch noncollapse terms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import PHENOTYPE_DIM, RESPONSE_DIM, arm_spec
from .model import FactorizedOutput
from .upstream import SIGReg, patient_mean_ftv_loss


def _normalized_state_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("next-state prediction and target shapes must agree")
    if prediction.size(1) != step_weights.numel():
        raise ValueError("next-state step count differs from frozen weights")
    normalized_prediction = F.layer_norm(prediction, (prediction.size(-1),))
    normalized_target = F.layer_norm(target, (target.size(-1),))
    per_step = (normalized_prediction - normalized_target).square().mean(dim=-1)
    return (per_step * step_weights).mean()


def _matrix_view(value: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if value.ndim < 2 or int(value.size(-1)) != expected_dim:
        raise ValueError(f"state must end in {expected_dim} dimensions")
    matrix = value.reshape(-1, expected_dim).float()
    if matrix.size(0) < 2:
        raise ValueError("logical covariance losses require at least two rows")
    if not bool(torch.isfinite(matrix).all()):
        raise FloatingPointError("logical state contains non-finite values")
    return matrix


def covariance_matrix(value: torch.Tensor) -> torch.Tensor:
    matrix = value.float()
    if matrix.ndim != 2 or matrix.size(0) < 2:
        raise ValueError("covariance input must be [N,D] with N>=2")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(matrix.size(0) - 1)


def cross_covariance_penalty(
    response: torch.Tensor,
    phenotype: torch.Tensor,
) -> torch.Tensor:
    """Dimension-normalized squared Frobenius batch cross-covariance.

    Dividing by one branch width (rather than all 96x96 entries) follows the
    VICReg covariance-loss scale. It preserves a meaningful decorrelation
    gradient at logical B32 while keeping the preregistered 0.05 coefficient
    commensurate with the JEPA and phenotype objectives.
    """

    response_matrix = _matrix_view(response, RESPONSE_DIM)
    phenotype_matrix = _matrix_view(phenotype, PHENOTYPE_DIM)
    if response_matrix.size(0) != phenotype_matrix.size(0):
        raise ValueError("response and phenotype logical rows differ")
    response_centered = response_matrix - response_matrix.mean(dim=0, keepdim=True)
    phenotype_centered = phenotype_matrix - phenotype_matrix.mean(dim=0, keepdim=True)
    cross = response_centered.T @ phenotype_centered / float(response_matrix.size(0) - 1)
    return cross.square().sum() / float(RESPONSE_DIM)


def vicreg_consistency(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    invariance_weight: float = 1.0,
    variance_weight: float = 0.1,
    covariance_weight: float = 0.01,
    variance_target: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Same-visit invariance plus variance/covariance noncollapse regularization."""

    first_matrix = _matrix_view(first, PHENOTYPE_DIM)
    second_matrix = _matrix_view(second, PHENOTYPE_DIM)
    if first_matrix.shape != second_matrix.shape:
        raise ValueError("two augmented phenotype views must have identical shape")
    invariance = F.mse_loss(first_matrix, second_matrix)
    first_std = torch.sqrt(first_matrix.var(dim=0, unbiased=False) + 1e-4)
    second_std = torch.sqrt(second_matrix.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(float(variance_target) - first_std).mean()
        + F.relu(float(variance_target) - second_std).mean()
    )

    def off_diagonal_penalty(matrix: torch.Tensor) -> torch.Tensor:
        covariance = covariance_matrix(matrix)
        diagonal = torch.diagonal(covariance)
        off_diagonal = covariance - torch.diag_embed(diagonal)
        return off_diagonal.square().sum() / float(matrix.size(1))

    covariance = 0.5 * (
        off_diagonal_penalty(first_matrix)
        + off_diagonal_penalty(second_matrix)
    )
    total = (
        float(invariance_weight) * invariance
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
    )
    return total, {
        "phenotype_invariance_loss": invariance,
        "phenotype_variance_loss": variance,
        "phenotype_covariance_loss": covariance,
    }


@dataclass(frozen=True)
class LogicalLossWeights:
    lambda_ftv: float = 0.25
    lambda_phenotype_future: float = 0.5
    lambda_phenotype_consistency: float = 0.1
    lambda_crosscov: float = 0.05
    sigreg_weight: float = 0.09
    consistency_invariance: float = 1.0
    consistency_variance: float = 0.1
    consistency_covariance: float = 0.01
    variance_target: float = 1.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lambda_ftv != 0.25:
            raise ValueError("Goal F FTV grounding weight is frozen to 0.25")


class FactorizedObjective(nn.Module):
    """Separate additive per-patient and exact logical-batch components."""

    def __init__(
        self,
        arm: str,
        weights: LogicalLossWeights = LogicalLossWeights(),
        *,
        sigreg_projections: int = 256,
        step_weights: tuple[float, float, float] = (2.0, 1.0, 0.5),
    ) -> None:
        super().__init__()
        self.spec = arm_spec(arm)
        self.arm = self.spec.name
        weights.validate()
        self.weights = weights
        normalized = torch.tensor(step_weights, dtype=torch.float32)
        if normalized.shape != (3,) or bool((normalized <= 0).any()):
            raise ValueError("three positive JEPA step weights are required")
        self.register_buffer("step_weights", normalized / normalized.mean())
        self.sigreg = SIGReg(int(sigreg_projections))

    @property
    def lambda_ftv(self) -> float:
        return self.weights.lambda_ftv

    @property
    def lambda_adv(self) -> float:
        return self.spec.adversarial_weight

    def additive_components(
        self,
        output: FactorizedOutput,
        ftv_target: torch.Tensor,
        ftv_mask: torch.Tensor,
        clinical_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        response_jepa = _normalized_state_mse(
            output.predicted_response_next,
            output.target_response_online[:, 1:],
            self.step_weights,
        )
        phenotype_future = _normalized_state_mse(
            output.predicted_phenotype_next,
            output.target_phenotype_online[:, 1:],
            self.step_weights,
        )
        ftv, ftv_patients, ftv_visits = patient_mean_ftv_loss(
            output.ftv_prediction,
            ftv_target,
            ftv_mask,
            output.response_state,
        )
        if clinical_target.ndim != 2 or clinical_target.shape != (
            output.response_state.size(0),
            2,
        ):
            raise ValueError("clinical adversary targets must be [B,2]")
        if output.adversary_hr_logits is None:
            if self.spec.uses_adversary:
                raise ValueError("residualization arm omitted adversary logits")
            adversary_hr = output.phenotype_state.sum() * 0.0
            adversary_her2 = output.phenotype_state.sum() * 0.0
        else:
            if not self.spec.uses_adversary or output.adversary_her2_logits is None:
                raise ValueError("non-residual arm unexpectedly emitted adversary logits")
            visits = output.adversary_hr_logits.size(1)
            hr_target = clinical_target[:, 0, None].expand(-1, visits).reshape(-1)
            her2_target = clinical_target[:, 1, None].expand(-1, visits).reshape(-1)
            adversary_hr = F.cross_entropy(
                output.adversary_hr_logits.reshape(-1, 2), hr_target
            )
            adversary_her2 = F.cross_entropy(
                output.adversary_her2_logits.reshape(-1, 2), her2_target
            )
        adversary = 0.5 * (adversary_hr + adversary_her2)
        non_ftv = (
            response_jepa
            + self.weights.lambda_phenotype_future * phenotype_future
            + self.lambda_adv * adversary
        )
        total = non_ftv + self.lambda_ftv * ftv
        return total, {
            "response_jepa_loss": response_jepa,
            "phenotype_future_loss": phenotype_future,
            "ftv_loss": ftv,
            "weighted_ftv_loss": self.lambda_ftv * ftv,
            "adversary_loss": adversary,
            "adversary_hr_loss": adversary_hr,
            "adversary_her2_loss": adversary_her2,
            "weighted_adversary_loss": self.lambda_adv * adversary,
            "_non_ftv_component": non_ftv,
            "_ftv_component_raw": ftv,
            "ftv_patients": ftv_patients.to(torch.float32),
            "ftv_valid_visits": ftv_visits.to(torch.float32),
        }

    def logical_regularizers(
        self,
        *,
        response_state: torch.Tensor,
        phenotype_state: torch.Tensor,
        augmented_phenotype_state: torch.Tensor,
        response_online: torch.Tensor,
        sigreg_seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if sigreg_seed is None:
            sigreg = self.sigreg(response_online.transpose(0, 1))
        else:
            devices = []
            if response_online.device.type == "cuda":
                devices = [
                    torch.cuda.current_device()
                    if response_online.device.index is None
                    else int(response_online.device.index)
                ]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(sigreg_seed))
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(int(sigreg_seed))
                sigreg = self.sigreg(response_online.transpose(0, 1))
        consistency, consistency_stats = vicreg_consistency(
            phenotype_state,
            augmented_phenotype_state,
            invariance_weight=self.weights.consistency_invariance,
            variance_weight=self.weights.consistency_variance,
            covariance_weight=self.weights.consistency_covariance,
            variance_target=self.weights.variance_target,
        )
        crosscov = cross_covariance_penalty(response_state, phenotype_state)
        total = (
            self.weights.sigreg_weight * sigreg
            + self.weights.lambda_phenotype_consistency * consistency
            + self.weights.lambda_crosscov * crosscov
        )
        stats = {
            "sigreg_loss": sigreg,
            "phenotype_consistency_loss": consistency,
            "crosscov_loss": crosscov,
            "weighted_sigreg_loss": self.weights.sigreg_weight * sigreg,
            "weighted_phenotype_consistency_loss": self.weights.lambda_phenotype_consistency * consistency,
            "weighted_crosscov_loss": self.weights.lambda_crosscov * crosscov,
            "logical_regularizer_loss": total,
            **consistency_stats,
        }
        return total, stats


def logical_loss_surrogate(
    current: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    reference_gradient: Mapping[str, torch.Tensor],
    logical_loss: torch.Tensor,
    *,
    logical_batch_size: int,
) -> torch.Tensor:
    """Microbatch surrogate whose accumulated gradient equals one logical loss.

    The caller multiplies this result by ``microbatch/logical_batch``.  The
    correction multiplier therefore makes both the summed value at the
    reference point and the summed gradient exact.
    """

    if set(current) != set(reference) or set(current) != set(reference_gradient):
        raise ValueError("logical surrogate state inventories differ")
    sizes = {int(value.size(0)) for value in current.values()}
    if len(sizes) != 1:
        raise ValueError("logical surrogate tensors have inconsistent batch sizes")
    microbatch_size = sizes.pop()
    if not 0 < microbatch_size <= int(logical_batch_size):
        raise ValueError("logical surrogate batch sizes are invalid")
    correction = logical_loss.detach() * 0.0
    for name in current:
        if current[name].shape != reference[name].shape or current[name].shape != reference_gradient[name].shape:
            raise ValueError(f"logical surrogate shape mismatch for {name}")
        correction = correction + (
            (current[name] - reference[name]) * reference_gradient[name]
        ).sum()
    multiplier = float(logical_batch_size) / float(microbatch_size)
    return logical_loss.detach() + multiplier * correction


__all__ = [
    "FactorizedObjective",
    "LogicalLossWeights",
    "covariance_matrix",
    "cross_covariance_penalty",
    "logical_loss_surrogate",
    "vicreg_consistency",
]
