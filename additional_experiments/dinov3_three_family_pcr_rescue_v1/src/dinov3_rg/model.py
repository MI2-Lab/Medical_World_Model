"""MRI-domain adapter and image-only temporal world model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .contracts import STATE_SHAPE, SUMMARY_SHAPE


def sinusoidal_positions(length: int, dimension: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    result = torch.zeros(length, dimension, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor)
    return result


class MRIDomainAdapter(nn.Module):
    """Aggregate seven channel-specific DINO summaries over 32 axial slices."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.summary_projection = nn.Sequential(
            nn.Linear(2304, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.channel_embedding = nn.Parameter(torch.randn(1, 7, 128) / math.sqrt(128))
        channel_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_transformer = nn.TransformerEncoder(channel_layer, num_layers=1)
        self.state_token = nn.Parameter(torch.randn(1, 1, 128) / math.sqrt(128))
        self.register_buffer("axial_position", sinusoidal_positions(33, 128).unsqueeze(0))
        slice_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slice_transformer = nn.TransformerEncoder(slice_layer, num_layers=1)
        self.response_projection = nn.Sequential(nn.Linear(128, 192), nn.LayerNorm(192))

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        if summary.ndim != 5 or tuple(summary.shape[1:]) != SUMMARY_SHAPE:
            raise ValueError(
                "MRI adapter accepts only frozen [B,4,7,32,2304] slice summaries; "
                f"got {tuple(summary.shape)}"
            )
        if not bool(torch.isfinite(summary).all()):
            raise ValueError("slice summaries are non-finite")
        batch, visits, channels, slices, _ = summary.shape
        values = summary.to(dtype=self.summary_projection[0].weight.dtype)
        values = self.summary_projection(values)
        # channel attention independently at each patient/visit/slice
        values = values.permute(0, 1, 3, 2, 4).reshape(-1, channels, 128)
        values = self.channel_transformer(values + self.channel_embedding)
        values = values.mean(dim=1).reshape(batch * visits, slices, 128)
        token = self.state_token.expand(batch * visits, -1, -1)
        sequence = torch.cat((token, values), dim=1) + self.axial_position
        sequence = self.slice_transformer(sequence)
        response = self.response_projection(sequence[:, 0])
        return response.reshape(batch, visits, 192)


class VisitProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(192, 384), nn.LayerNorm(384), nn.GELU(), nn.Linear(384, 192)
        )

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        return self.network(response)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((length, length), float("-inf"), device=device), diagonal=1
    )


class CausalTransition(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, 3, 192) / math.sqrt(192))
        layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.output = nn.Sequential(
            nn.LayerNorm(192), nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.size(-1) != 192 or not 1 <= state.size(1) <= 3:
            raise ValueError("transition input must be [B,L,192], L=1..3")
        length = state.size(1)
        hidden = state + self.position[:, :length]
        hidden = self.transformer(hidden, mask=causal_mask(length, state.device))
        return self.output(hidden)


@dataclass
class WorldModelOutput:
    response_state: torch.Tensor
    online_state: torch.Tensor
    target_state: torch.Tensor
    target_next: torch.Tensor
    predicted_next: torch.Tensor
    ftv_prediction: torch.Tensor
    radiomics_prediction: torch.Tensor


class MRIAdapterWorldModel(nn.Module):
    """Forward API deliberately admits image summaries and nothing else."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.adapter = MRIDomainAdapter(dropout)
        self.projector = VisitProjector()
        self.target_adapter = copy.deepcopy(self.adapter).requires_grad_(False)
        self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.transition = CausalTransition(dropout)
        # Identical head structure exists in C0/RAD. Objective weights define arms.
        with torch.random.fork_rng(devices=[]):
            self.ftv_head = nn.Linear(192, 1)
            self.radiomics_head = nn.Linear(192, 16)

    def encode_response(self, slice_summaries: torch.Tensor) -> torch.Tensor:
        return self.adapter(slice_summaries)

    def encode_online(self, slice_summaries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        response = self.adapter(slice_summaries)
        return response, self.projector(response)

    @torch.no_grad()
    def encode_target(self, slice_summaries: torch.Tensor) -> torch.Tensor:
        self.target_adapter.eval()
        self.target_projector.eval()
        return self.target_projector(self.target_adapter(slice_summaries))

    def forward(self, slice_summaries: torch.Tensor) -> WorldModelOutput:
        if slice_summaries.ndim != 5 or slice_summaries.size(1) != 4:
            raise ValueError("training forward requires four visits")
        response, online = self.encode_online(slice_summaries)
        target = self.encode_target(slice_summaries).detach()
        return WorldModelOutput(
            response_state=response,
            online_state=online,
            target_state=target,
            target_next=target[:, 1:],
            predicted_next=self.transition(online[:, :-1]),
            ftv_prediction=self.ftv_head(response).squeeze(-1),
            radiomics_prediction=self.radiomics_head(response),
        )

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        if not 0.0 < float(momentum) < 1.0:
            raise ValueError("EMA momentum must be in (0,1)")
        for online_module, target_module in (
            (self.adapter, self.target_adapter),
            (self.projector, self.target_projector),
        ):
            for online, target in zip(online_module.parameters(), target_module.parameters()):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
            for online, target in zip(online_module.buffers(), target_module.buffers()):
                target.copy_(online)

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "forward_signature": "forward(slice_summaries)",
            "input_shape": ["B", *SUMMARY_SHAPE],
            "input_kind": "frozen_DINOv3_LOCAL_DCE7_slice_summaries",
            "response_shape": ["B", *STATE_SHAPE],
            "DINO_backbone_in_training_graph": False,
            "adapter": "shared Linear(2304,128)+LN+GELU",
            "channel_aggregation": "learned embedding + Transformer1/h4/ff256 + mean",
            "slice_aggregation": "sinusoidal axial position + state token + Transformer1/h4/ff512",
            "response_projection": "Linear(128,192)+LayerNorm",
            "transition": "causal Transformer3/h4/ff512",
            "radiomics_head": "Linear(192,16) from observed response",
            "ftv_head": "Linear(192,1) from observed response",
            "forbidden_forward_inputs": [
                "clinical", "outcome", "FTV", "radiomics", "ROI mask", "geometry"
            ],
        }


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_sha256(model: MRIAdapterWorldModel) -> str:
    return tensor_state_sha256(
        {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith("target_")
        }
    )


def assert_forward_contract() -> None:
    signature = inspect.signature(MRIAdapterWorldModel.forward)
    if tuple(signature.parameters) != ("self", "slice_summaries"):
        raise AssertionError(f"model forward signature drifted: {signature}")


assert_forward_contract()


__all__ = [
    "MRIAdapterWorldModel", "MRIDomainAdapter", "WorldModelOutput", "assert_forward_contract",
    "initialization_sha256", "sinusoidal_positions", "tensor_state_sha256"
]
