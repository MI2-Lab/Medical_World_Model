from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from torch import nn

from ..config import ModelConfig
from .encoder import GeometryProjector, VisitEncoder3D, VisitProjector
from .response_state import FutureResponseState, ResponseGuidanceHeads
from .transition import ImageTransition


class CoReJEPAOutput(NamedTuple):
    """Named outputs with explicit dimensions.

    All dimensions are batch-first:
        ``visit_state [B,4,Dz]``;
        ``image_prediction/prediction/target [B,3,Dz]``;
        ``future_response_state [B,3,Ds]``;
        ``decoded_geometry [B,3,9]``;
        ``response_correction [B,3,Dz]``;
        ``gate_logits/gate_probabilities [B,3,E]``;
        ``score_prediction [B,3]`` and ``vector_prediction [B,3,18]``;
        ``update_score_prediction [B,2]`` and
        ``update_vector_prediction [B,2,18]``.
    """

    visit_state: torch.Tensor
    image_prediction: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    future_response_state: torch.Tensor
    decoded_geometry: torch.Tensor
    response_correction: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor
    score_prediction: torch.Tensor
    vector_prediction: torch.Tensor
    update_score_prediction: torch.Tensor
    update_vector_prediction: torch.Tensor


class CoReJEPA(nn.Module):
    """Clinical/treatment-conditioned response-state JEPA."""

    def __init__(self, config: ModelConfig, condition_dim: int, temporal_condition_dim: int = 7) -> None:
        super().__init__()
        self.config = config
        self.condition_dim = int(condition_dim)
        self.temporal_condition_dim = int(temporal_condition_dim)
        self.encoder = VisitEncoder3D(config.image_channels, config.base_channels, config.latent_dim)
        self.projector = VisitProjector(config.latent_dim)
        self.geometry_projector = GeometryProjector(config.geometry_dim, config.latent_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        self.target_geometry_projector = copy.deepcopy(self.geometry_projector)
        for module in (self.target_encoder, self.target_projector, self.target_geometry_projector):
            module.requires_grad_(False)
        self.image_transition = ImageTransition(
            config.latent_dim,
            condition_dim,
            config.predictor_depth,
            config.predictor_heads,
            config.predictor_mlp_dim,
            config.dropout,
            config.film_scale,
        )
        self.response_transition = FutureResponseState(
            geometry_dim=config.geometry_dim,
            latent_dim=config.latent_dim,
            condition_dim=condition_dim,
            temporal_condition_dim=temporal_condition_dim,
            response_dim=config.response_dim,
            hidden_dim=config.response_hidden_dim,
            depth=config.response_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
            experts=config.response_experts,
            expert_hidden_dim=config.expert_hidden_dim,
            gate_hidden_dim=config.expert_gate_hidden_dim,
            gate_temperature=config.expert_temperature,
            expert_scale=config.expert_scale,
            expert_init_std=config.expert_init_std,
            latent_scale=config.response_latent_scale,
            film_scale=config.film_scale,
        )
        self.guidance = ResponseGuidanceHeads(config.response_dim, config.response_target_dim, config.dropout)

    @staticmethod
    def _encode_sequence(
        image: torch.Tensor,
        geometry: torch.Tensor,
        encoder: nn.Module,
        projector: nn.Module,
        geometry_projector: nn.Module,
    ) -> torch.Tensor:
        if image.ndim != 6 or geometry.ndim != 3 or image.shape[:2] != geometry.shape[:2]:
            raise ValueError(f"Expected image [B,V,C,Z,Y,X] and geometry [B,V,9], got {image.shape}, {geometry.shape}")
        batch, visits = image.shape[:2]
        appearance = projector(encoder(image.reshape(batch * visits, *image.shape[2:])))
        geometry_state = geometry_projector(geometry.reshape(batch * visits, -1))
        return (appearance + geometry_state).reshape(batch, visits, -1)

    def encode_visits(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """Return online observed visit states ``z [B,4,Dz]``."""

        return self._encode_sequence(image, geometry, self.encoder, self.projector, self.geometry_projector)

    @torch.no_grad()
    def encode_targets(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """Return EMA target visit states ``z_tar [B,4,Dz]``."""

        self.target_encoder.eval()
        self.target_projector.eval()
        self.target_geometry_projector.eval()
        return self._encode_sequence(
            image,
            geometry,
            self.target_encoder,
            self.target_projector,
            self.target_geometry_projector,
        )

    def forward(self, image: torch.Tensor, geometry: torch.Tensor, condition: torch.Tensor) -> CoReJEPAOutput:
        visit_state = self.encode_visits(image, geometry)
        target = self.encode_targets(image, geometry)[:, 1:].detach()
        image_prediction = self.image_transition(visit_state[:, :-1], condition)
        response = self.response_transition(geometry[:, :-1], condition)
        prediction = image_prediction + response.latent_correction
        state_update = response.future_state[:, 1:] - response.future_state[:, :-1]
        return CoReJEPAOutput(
            visit_state=visit_state,
            image_prediction=image_prediction,
            prediction=prediction,
            target=target,
            future_response_state=response.future_state,
            decoded_geometry=response.decoded_geometry,
            response_correction=response.latent_correction,
            gate_logits=response.gate_logits,
            gate_probabilities=response.gate_probabilities,
            score_prediction=self.guidance.score(response.future_state),
            vector_prediction=self.guidance.vector(response.future_state),
            update_score_prediction=self.guidance.update_score(state_update),
            update_vector_prediction=self.guidance.update_vector(state_update),
        )

    def forecast_response(self, geometry: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return future response states ``[B,3,Ds]`` from ``q0:q2``.

        This is the exact frozen representation used by FLR. The causal rows are
        ``q0 -> s_hat1``, ``q0:q1 -> s_hat2``, and
        ``q0:q2 -> s_hat3``.
        """

        return self.response_transition(geometry[:, :-1], condition).future_state

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """EMA-update only the target visit representation modules."""

        pairs = (
            (self.encoder, self.target_encoder),
            (self.projector, self.target_projector),
            (self.geometry_projector, self.target_geometry_projector),
        )
        for online_module, target_module in pairs:
            for online, target in zip(online_module.parameters(), target_module.parameters()):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
            for online, target in zip(online_module.buffers(), target_module.buffers()):
                target.copy_(online)
