from __future__ import annotations

import torch
from torch import nn


class SideEncoder(nn.Module):
    """Shared summary/spatial encoder; the only inputs are frozen DINO features."""

    def __init__(self, spatial: bool) -> None:
        super().__init__()
        self.spatial = spatial
        self.summary_projection = nn.Sequential(nn.Linear(2304, 128), nn.LayerNorm(128), nn.GELU())
        self.channel_embedding = nn.Parameter(torch.zeros(1, 1, 7, 128))
        nn.init.normal_(self.channel_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, batch_first=True, norm_first=True)
        self.channel_transformer = nn.TransformerEncoder(layer, 1)
        slice_layer = nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0, batch_first=True, norm_first=True)
        self.slice_transformer = nn.TransformerEncoder(slice_layer, 1)
        self.slice_position = nn.Parameter(torch.zeros(1, 32, 128))
        self.state_token = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.slice_position, std=0.02)
        nn.init.normal_(self.state_token, std=0.02)
        if spatial:
            self.spatial_projection = nn.Linear(768, 128)
            # Zero initialization makes B1's initial global path identical to B0.
            nn.init.zeros_(self.spatial_projection.weight)
            nn.init.zeros_(self.spatial_projection.bias)
        self.output = nn.Sequential(nn.Linear(128, 192), nn.LayerNorm(192))

    def forward(self, summary: torch.Tensor, spatial_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if summary.ndim != 5 or summary.shape[2:] != (7, 32, 2304):
            raise ValueError(f"summary must be [B,4,7,32,2304], got {tuple(summary.shape)}")
        batch, visits = summary.shape[:2]
        x = self.summary_projection(summary)
        if self.spatial:
            if spatial_tokens is None or spatial_tokens.shape != (*summary.shape[:4], 4, 4, 768):
                raise ValueError("spatial branch requires [B,4,7,32,4,4,768] tokens")
            x = x + self.spatial_projection(spatial_tokens).mean(dim=(-3, -2))
        x = x + self.channel_embedding.unsqueeze(3)
        x = x.reshape(batch * visits * 32, 7, 128)
        x = self.channel_transformer(x).mean(dim=1)
        x = x.reshape(batch * visits, 32, 128) + self.slice_position
        token = self.state_token.expand(batch * visits, -1, -1)
        x = torch.cat((token, x), dim=1)
        x = self.slice_transformer(x)[:, 0]
        return self.output(x).reshape(batch, visits, 192)


class BilateralPCRModel(nn.Module):
    """Direct image-only pCR model with symmetric left/right fusion."""

    def __init__(self, spatial: bool) -> None:
        super().__init__()
        self.spatial = spatial
        self.side_encoder = SideEncoder(spatial)
        self.bilateral_projection = nn.Sequential(nn.Linear(384, 192), nn.LayerNorm(192), nn.GELU())
        self.heads = nn.ModuleDict({
            "T0": nn.Linear(192, 1),
            "T0_T1": nn.Linear(192 * 3, 1),
            "T0_T2": nn.Linear(192 * 6, 1),
        })

    def encode(self, global_summary: torch.Tensor, spatial_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if global_summary.ndim != 6 or global_summary.shape[2:] != (2, 7, 32, 2304):
            raise ValueError(f"global summary must be [B,4,2,7,32,2304], got {tuple(global_summary.shape)}")
        left = self.side_encoder(global_summary[:, :, 0], None if spatial_tokens is None else spatial_tokens[:, :, 0])
        right = self.side_encoder(global_summary[:, :, 1], None if spatial_tokens is None else spatial_tokens[:, :, 1])
        symmetric = torch.cat(((left + right) * 0.5, (left - right).abs()), dim=-1)
        return self.bilateral_projection(symmetric)

    def forward(self, global_summary: torch.Tensor, spatial_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        state = self.encode(global_summary, spatial_tokens)
        prefixes = {
            "T0": state[:, 0],
            "T0_T1": torch.cat((state[:, 0], state[:, 1], state[:, 1] - state[:, 0]), dim=-1),
            "T0_T2": torch.cat((state[:, 0], state[:, 1], state[:, 2], state[:, 1] - state[:, 0], state[:, 2] - state[:, 1], state[:, 2] - state[:, 0]), dim=-1),
        }
        logits = {timing: self.heads[timing](value).squeeze(-1) for timing, value in prefixes.items()}
        return {"state": state, **{f"logit_{timing}": value for timing, value in logits.items()}}


def assert_frozen_dino_contract(model: nn.Module) -> None:
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("DINO is not part of the trainable image model")
