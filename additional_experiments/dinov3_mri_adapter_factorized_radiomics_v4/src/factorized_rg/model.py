from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from .contracts import SUMMARY_SHAPE


def sinusoidal_positions(length: int, dimension: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    out = torch.zeros(length, dimension)
    out[:, 0::2] = torch.sin(pos * div)
    out[:, 1::2] = torch.cos(pos * div)
    return out


class MRIDomainAdapter(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.summary_projection = nn.Sequential(nn.Linear(2304, 128), nn.LayerNorm(128), nn.GELU())
        self.channel_embedding = nn.Parameter(torch.randn(1, 7, 128) / math.sqrt(128))
        channel = nn.TransformerEncoderLayer(128, 4, 256, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.channel_transformer = nn.TransformerEncoder(channel, 1)
        self.state_token = nn.Parameter(torch.randn(1, 1, 128) / math.sqrt(128))
        self.register_buffer("axial_position", sinusoidal_positions(33, 128).unsqueeze(0))
        slice_layer = nn.TransformerEncoderLayer(128, 4, 512, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.slice_transformer = nn.TransformerEncoder(slice_layer, 1)
        self.response_projection = nn.Sequential(nn.Linear(128, 192), nn.LayerNorm(192))

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        if summary.ndim != 5 or tuple(summary.shape[1:]) != SUMMARY_SHAPE:
            raise ValueError(f"expected [B,4,7,32,2304], got {tuple(summary.shape)}")
        if not bool(torch.isfinite(summary).all()):
            raise ValueError("non-finite DINO summary")
        b, visits, channels, slices, _ = summary.shape
        values = self.summary_projection[0](summary.to(self.summary_projection[0].weight.dtype))
        values = self.summary_projection[1:](values)
        values = values.permute(0, 1, 3, 2, 4).reshape(-1, channels, 128)
        values = self.channel_transformer(values + self.channel_embedding).mean(1)
        values = values.reshape(b * visits, slices, 128)
        token = self.state_token.expand(b * visits, -1, -1)
        sequence = torch.cat((token, values), 1) + self.axial_position
        return self.response_projection(self.slice_transformer(sequence)[:, 0]).reshape(b, visits, 192)


class Branch(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(192, output_dim), nn.LayerNorm(output_dim))

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        return self.net(response)


class CausalTransition(nn.Module):
    def __init__(self, dimension: int, feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, 3, dimension) / math.sqrt(dimension))
        layer = nn.TransformerEncoderLayer(dimension, 4, feedforward, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 3)
        self.output = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, feedforward), nn.GELU(), nn.Linear(feedforward, dimension))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.size(1) not in (1, 2, 3):
            raise ValueError("transition expects [B,L,D], L=1..3")
        length = state.size(1)
        mask = torch.triu(torch.full((length, length), float("-inf"), device=state.device), diagonal=1)
        return self.output(self.transformer(state + self.position[:, :length], mask=mask))


@dataclass
class FactorizedOutput:
    response_state: torch.Tensor
    jepa_online: torch.Tensor
    phenotype_online: torch.Tensor
    jepa_target: torch.Tensor
    phenotype_target: torch.Tensor
    predicted_jepa: torch.Tensor
    predicted_phenotype: torch.Tensor
    radiomics_prediction: torch.Tensor


class FactorizedWorldModel(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.adapter = MRIDomainAdapter(dropout)
        self.jepa_branch = Branch(128)
        self.phenotype_branch = Branch(64)
        self.target_adapter = copy.deepcopy(self.adapter).requires_grad_(False)
        self.target_jepa_branch = copy.deepcopy(self.jepa_branch).requires_grad_(False)
        self.target_phenotype_branch = copy.deepcopy(self.phenotype_branch).requires_grad_(False)
        self.jepa_transition = CausalTransition(128, 512, dropout)
        self.phenotype_transition = CausalTransition(64, 256, dropout)
        self.radiomics_head = nn.Linear(64, 16)

    def encode_online(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        response = self.adapter(summary)
        return response, self.jepa_branch(response), self.phenotype_branch(response)

    @torch.no_grad()
    def encode_target(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        response = self.target_adapter(summary)
        return self.target_jepa_branch(response), self.target_phenotype_branch(response)

    def forward(self, summary: torch.Tensor) -> FactorizedOutput:
        response, jepa, phenotype = self.encode_online(summary)
        with torch.no_grad():
            target_jepa, target_phenotype = self.encode_target(summary)
        return FactorizedOutput(
            response, jepa, phenotype, target_jepa.detach(), target_phenotype.detach(),
            self.jepa_transition(jepa[:, :-1]), self.phenotype_transition(phenotype[:, :-1]),
            self.radiomics_head(phenotype),
        )

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        for online, target in (
            (self.adapter, self.target_adapter),
            (self.jepa_branch, self.target_jepa_branch),
            (self.phenotype_branch, self.target_phenotype_branch),
        ):
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.data.mul_(momentum).add_(op.data, alpha=1.0 - momentum)
            for ob, tb in zip(online.buffers(), target.buffers()):
                tb.copy_(ob)

    def architecture_contract(self) -> dict[str, object]:
        return {
            "forward_signature": "forward(frozen_dinov3_summary)",
            "input_shape": ["B", *SUMMARY_SHAPE],
            "response_shape": ["B", 4, 192],
            "jepa_branch": ["B", 4, 128],
            "phenotype_branch": ["B", 4, 64],
            "DINO_backbone_in_training_graph": False,
            "ftv_loss_weight": 0.0,
            "forbidden_forward_inputs": ["clinical", "pCR", "FTV", "radiomics", "ROI mask", "geometry"],
        }
