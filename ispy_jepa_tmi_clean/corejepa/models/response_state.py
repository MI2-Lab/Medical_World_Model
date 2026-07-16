from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .transition import ConditionedCausalTransformer


def transform_geometry(geometry: torch.Tensor) -> torch.Tensor:
    """Log-scale q columns 0 and 4 while preserving ``[...,9]`` shape."""

    output = geometry.clone()
    for column in (0, 4):
        output[..., column] = torch.log1p(output[..., column].clamp_min(0) * 1000.0) / math.log1p(1000.0)
    return output


@dataclass
class ResponseStateOutput:
    """Outputs of the factorized future response pathway."""

    future_state: torch.Tensor
    decoded_geometry: torch.Tensor
    latent_correction: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor


class FutureResponseState(nn.Module):
    """Forecast a response state from each observed lesion-descriptor prefix.

    Inputs:
        ``geometry_context [B,3,9]`` for q0/q1/q2 and
        ``condition [B,3,Cc]``.
    Outputs:
        ``future_state [B,3,Ds]`` for s-hat1/s-hat2/s-hat3,
        ``decoded_geometry [B,3,9]``, and
        ``latent_correction [B,3,Dz]``.
    """

    def __init__(
        self,
        geometry_dim: int,
        latent_dim: int,
        condition_dim: int,
        temporal_condition_dim: int,
        response_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        dropout: float,
        experts: int,
        expert_hidden_dim: int,
        gate_hidden_dim: int,
        gate_temperature: float,
        expert_scale: float,
        expert_init_std: float,
        latent_scale: float,
        film_scale: float,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.gate_temperature = float(gate_temperature)
        self.expert_scale = float(expert_scale)
        self.latent_scale = float(latent_scale)
        self.temporal_condition_dim = int(temporal_condition_dim)
        self.context = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, response_dim),
            nn.LayerNorm(response_dim),
        )
        self.dynamics = ConditionedCausalTransformer(
            response_dim,
            condition_dim,
            max(1, depth),
            heads,
            max(hidden_dim, response_dim * 2),
            dropout,
            film_scale=film_scale,
        )
        self.expert_adapter = nn.Sequential(
            nn.LayerNorm(response_dim),
            nn.Linear(response_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, response_dim * experts),
        )
        nn.init.normal_(self.expert_adapter[-1].weight, std=expert_init_std)
        nn.init.zeros_(self.expert_adapter[-1].bias)
        gate_input_dim = condition_dim - temporal_condition_dim
        self.expert_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, experts),
        )
        nn.init.zeros_(self.expert_gate[-1].weight)
        nn.init.zeros_(self.expert_gate[-1].bias)
        self.geometry_decoder = nn.Sequential(
            nn.LayerNorm(response_dim),
            nn.Linear(response_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, geometry_dim),
        )
        self.latent_adapter = nn.Sequential(
            nn.LayerNorm(response_dim + geometry_dim),
            nn.Linear(response_dim + geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.latent_adapter[-1].weight)
        nn.init.zeros_(self.latent_adapter[-1].bias)

    def forward(self, geometry_context: torch.Tensor, condition: torch.Tensor) -> ResponseStateOutput:
        context = self.context(transform_geometry(geometry_context))
        base_state = self.dynamics(context, condition)
        batch, steps, response_dim = base_state.shape
        experts = self.expert_adapter(base_state).reshape(batch, steps, self.experts, response_dim)
        patient_condition = condition[..., self.temporal_condition_dim :]
        gate_logits = self.expert_gate(patient_condition)
        gate_probabilities = F.softmax(gate_logits / self.gate_temperature, dim=-1)
        hard = F.one_hot(gate_probabilities.argmax(dim=-1), num_classes=self.experts).to(gate_probabilities.dtype)
        straight_through = hard - gate_probabilities.detach() + gate_probabilities
        expert_delta = (experts * straight_through.unsqueeze(-1)).sum(dim=2)
        future_state = base_state + self.expert_scale * expert_delta
        decoded_geometry = self.geometry_decoder(future_state)
        latent_correction = self.latent_scale * self.latent_adapter(
            torch.cat((future_state, decoded_geometry), dim=-1)
        )
        return ResponseStateOutput(
            future_state=future_state,
            decoded_geometry=decoded_geometry,
            latent_correction=latent_correction,
            gate_logits=gate_logits,
            gate_probabilities=gate_probabilities,
        )


class ScalarGuidanceHead(nn.Module):
    """Map ``[...,Ds]`` to a scalar pCR-free response target ``[...]``."""

    def __init__(self, state_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class VectorGuidanceHead(nn.Module):
    """Map ``[...,Ds]`` to an 18-D pCR-free response vector ``[...,18]``."""

    def __init__(self, state_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ResponseGuidanceHeads(nn.Module):
    """All pCR-free decoders used by IRG."""

    def __init__(self, state_dim: int, response_dim: int, dropout: float) -> None:
        super().__init__()
        self.score = ScalarGuidanceHead(state_dim, 64, dropout)
        self.update_score = ScalarGuidanceHead(state_dim, 64, dropout)
        self.vector = VectorGuidanceHead(state_dim, response_dim, 128, dropout)
        self.update_vector = VectorGuidanceHead(state_dim, response_dim, 128, dropout)
