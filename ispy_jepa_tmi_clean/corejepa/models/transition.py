from __future__ import annotations

import math

import torch
from torch import nn


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


class ConditionedCausalTransformer(nn.Module):
    """Forecast one state for each causal prefix.

    Inputs:
        ``state [B,L,D]`` and ``condition [B,L,Cc]``.
    Output:
        ``hidden [B,L,D]``. Position ``t`` sees states ``0..t`` only.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        max_steps: int = 3,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.film_scale = float(film_scale)
        self.position = nn.Parameter(torch.randn(1, max_steps, state_dim) / math.sqrt(state_dim))
        self.condition_add = nn.Linear(condition_dim, state_dim)
        self.condition_film = nn.Linear(condition_dim, state_dim * 2)
        nn.init.zeros_(self.condition_film.weight)
        nn.init.zeros_(self.condition_film.bias)
        layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=heads if state_dim % heads == 0 else 1,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim))

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or condition.ndim != 3 or state.shape[:2] != condition.shape[:2]:
            raise ValueError(f"Expected state [B,L,D] and condition [B,L,C], got {state.shape}, {condition.shape}")
        length = state.size(1)
        gamma, beta = self.condition_film(condition).chunk(2, dim=-1)
        hidden = state + self.position[:, :length] + self.condition_add(condition)
        hidden = hidden * (1.0 + self.film_scale * torch.tanh(gamma)) + self.film_scale * beta
        hidden = self.transformer(hidden, mask=causal_attention_mask(length, hidden.device))
        return self.output(hidden)


class ImageTransition(nn.Module):
    """Predict the image-driven next-visit latent.

    Inputs: ``z_context [B,3,Dz]``, ``condition [B,3,Cc]``.
    Output: ``image_prediction [B,3,Dz]`` for T1/T2/T3.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        film_scale: float,
    ) -> None:
        super().__init__()
        self.dynamics = ConditionedCausalTransformer(
            latent_dim,
            condition_dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            film_scale=film_scale,
        )
        self.prediction = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, z_context: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.prediction(self.dynamics(z_context, condition))
