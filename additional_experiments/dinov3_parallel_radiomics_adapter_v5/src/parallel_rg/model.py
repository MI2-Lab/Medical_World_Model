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
    out = torch.zeros(length, dimension); out[:, 0::2] = torch.sin(pos * div); out[:, 1::2] = torch.cos(pos * div)
    return out


class MRIAdapter(nn.Module):
    """The trainable MRI-domain adapter used by the independent branch."""
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
        if summary.ndim != 5 or tuple(summary.shape[1:]) != SUMMARY_SHAPE or not bool(torch.isfinite(summary).all()):
            raise ValueError(f"expected finite [B,4,7,32,2304], got {tuple(summary.shape)}")
        b, visits, channels, slices, _ = summary.shape
        values = self.summary_projection[0](summary.to(self.summary_projection[0].weight.dtype)); values = self.summary_projection[1:](values)
        values = values.permute(0, 1, 3, 2, 4).reshape(-1, channels, 128)
        values = self.channel_transformer(values + self.channel_embedding).mean(1).reshape(b * visits, slices, 128)
        token = self.state_token.expand(b * visits, -1, -1); sequence = torch.cat((token, values), 1) + self.axial_position
        return self.response_projection(self.slice_transformer(sequence)[:, 0]).reshape(b, visits, 192)


class Branch64(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(192, 64), nn.LayerNorm(64))
    def forward(self, state: torch.Tensor) -> torch.Tensor: return self.net(state)


class FrozenC0(nn.Module):
    """Exact response/phenotype portion of a V4 F0 checkpoint."""
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(); self.adapter = MRIAdapter(dropout); self.phenotype_branch = Branch64()

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            response = self.adapter(summary); phenotype = self.phenotype_branch(response)
        return response, phenotype

    def freeze(self) -> None:
        self.requires_grad_(False); self.eval()


class ParallelRadiomicsModel(nn.Module):
    """Independent radiomics path. It has no trainable or gradient-connected C0 path."""
    def __init__(self, c0_checkpoint: str, dropout: float = 0.1) -> None:
        super().__init__()
        payload = torch.load(c0_checkpoint, map_location="cpu", weights_only=False)
        source = payload["model_state"]
        self.c0 = FrozenC0(dropout)
        self.c0.adapter.load_state_dict({key[8:]: value for key, value in source.items() if key.startswith("adapter.")}, strict=True)
        self.c0.phenotype_branch.load_state_dict({key[17:]: value for key, value in source.items() if key.startswith("phenotype_branch.")}, strict=True)
        self.c0.freeze()
        self.rad_adapter = copy.deepcopy(self.c0.adapter)
        self.rad_branch = copy.deepcopy(self.c0.phenotype_branch)
        self.radiomics_head = nn.Linear(64, 16)
        self.c0_checkpoint_sha256 = payload.get("protocol_sha256", "")

    def forward(self, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        # C0 is evaluated as a frozen reference; the training loss only uses rad_state.
        c0_state, c0_phenotype = self.c0(summary)
        rad_response = self.rad_adapter(summary)
        rad_state = self.rad_branch(rad_response)
        return {"c0_state": c0_state.detach(), "c0_phenotype": c0_phenotype.detach(), "rad_state": rad_state, "radiomics_prediction": self.radiomics_head(rad_state)}

    def rad_parameters(self) -> list[nn.Parameter]:
        return [parameter for module in (self.rad_adapter, self.rad_branch, self.radiomics_head) for parameter in module.parameters() if parameter.requires_grad]

    def train(self, mode: bool = True) -> "ParallelRadiomicsModel":
        super().train(mode)
        # Parent .train() would otherwise reactivate dropout in the frozen C0
        # reference.  C0 is never part of the optimization graph.
        self.c0.eval()
        return self

    def set_head_only(self) -> None:
        self.rad_adapter.requires_grad_(False); self.rad_branch.requires_grad_(False); self.radiomics_head.requires_grad_(True)

    def set_all_rad_trainable(self) -> None:
        self.c0.freeze(); self.rad_adapter.requires_grad_(True); self.rad_branch.requires_grad_(True); self.radiomics_head.requires_grad_(True)

    def architecture_contract(self) -> dict[str, object]:
        return {"forward_signature": "forward(frozen_dinov3_summary)", "c0_state": ["B", 4, 192], "rad_state": ["B", 4, 64], "parallel_state": ["B", 4, 256], "c0_trainable": False, "radiomics_loss_updates_c0": False, "forbidden_forward_inputs": ["clinical", "pCR", "FTV", "radiomics", "ROI mask", "geometry"]}
