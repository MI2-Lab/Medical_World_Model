from __future__ import annotations

from typing import Any

import torch
from torch import nn


def fixed_local_mask(shape_dhw: tuple[int, int, int], spacing_zyx_mm: tuple[float, float, float] = (16.0, 7.2, 7.2), local_mm: float = 64.0) -> torch.Tensor:
    """Return an outcome-free central 64-mm feature-cell support mask."""

    axes = [
        (torch.arange(size, dtype=torch.float32) + 0.5 - size / 2.0) * spacing
        for size, spacing in zip(shape_dhw, spacing_zyx_mm)
    ]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    half = float(local_mm) / 2.0
    return (z.abs() <= half) & (y.abs() <= half) & (x.abs() <= half)


def flatten_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim != 5:
        raise ValueError(f"feature map must be [B,C,D,H,W], got {tuple(feature_map.shape)}")
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def gap(feature_map: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    tokens = flatten_feature_map(feature_map)
    if weights is None:
        return tokens.mean(dim=1)
    weights = weights.to(tokens.dtype).flatten(1)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(tokens.dtype).eps)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator


class FeedForward(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QueryAttentionPool(nn.Module):
    """Small one-query cross-attention pooling module for C2."""

    def __init__(self, input_dim: int, width: int = 128, heads: int = 4, blocks: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if blocks > 2:
            raise ValueError("C2 permits at most two cross-attention blocks")
        self.input_projection = nn.Linear(input_dim, width)
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.ModuleList([nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True) for _ in range(blocks)])
        self.feed_forward = nn.ModuleList([FeedForward(width, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.input_projection(tokens)
        query = self.query.expand(tokens.shape[0], -1, -1)
        weights = None
        for attention, feed_forward in zip(self.attention, self.feed_forward):
            update, weights = attention(query, values, values, key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False)
            query = query + update
            query = feed_forward(query)
        return self.norm(query[:, 0]), weights if weights is not None else torch.empty(0, device=tokens.device)


class SpatialTokenBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(x)
        attended, weights = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False)
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x, weights


class PatchTokenTransformer(nn.Module):
    """Capacity-matched token transformer for C3/C4 with physical positions."""

    def __init__(self, input_dim: int, width: int = 128, heads: int = 4, blocks: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, width)
        self.position = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([SpatialTokenBlock(width, heads, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor, coordinates: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.input_projection(tokens) + self.position(coordinates)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat((torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device), padding_mask), dim=1)
        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x, padding_mask)
            attention_maps.append(weights)
        return self.norm(x[:, 0]), attention_maps


class SpatialReadout(nn.Module):
    """C1-C4 readouts. Clinical fields are intentionally absent from this API."""

    def __init__(self, arm: str, input_dim: int = 128, width: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if arm not in {"C1", "C2", "C3", "C4"}:
            raise ValueError("SpatialReadout arm must be C1, C2, C3, or C4")
        self.arm = arm
        self.input_dim = input_dim
        self.width = width
        if arm == "C1":
            self.pool = None
            self.head = nn.Sequential(nn.LayerNorm(input_dim), nn.Dropout(dropout), nn.Linear(input_dim, 1))
        elif arm == "C2":
            self.pool = QueryAttentionPool(input_dim, width, 4, 2, dropout)
            self.head = nn.Linear(width, 1)
        else:
            self.pool = PatchTokenTransformer(input_dim, width, 4, 3, dropout)
            self.head = nn.Linear(width, 1)

    def forward(self, feature_map: torch.Tensor, local_mask: torch.Tensor | None = None, coordinates: torch.Tensor | None = None, valid_weights: torch.Tensor | None = None) -> dict[str, Any]:
        tokens = flatten_feature_map(feature_map)
        batch, _, depth, height, width = feature_map.shape
        if coordinates is None:
            coords = fixed_local_mask((depth, height, width), local_mm=10_000).nonzero(as_tuple=False).to(feature_map.device).float()
            coords = coords / torch.tensor([max(depth - 1, 1), max(height - 1, 1), max(width - 1, 1)], device=feature_map.device)
            coordinates = coords.unsqueeze(0).expand(batch, -1, -1)
        if local_mask is None:
            local_mask = fixed_local_mask((depth, height, width)).flatten().to(feature_map.device)
        elif local_mask.ndim >= 3:
            local_mask = local_mask.flatten(start_dim=local_mask.ndim - 3)
        if local_mask.ndim == 1:
            local_mask = local_mask.unsqueeze(0).expand(batch, -1)
        if self.arm == "C1":
            pooled = gap(feature_map, valid_weights)
            logits = self.head(pooled).squeeze(-1)
            return {"logits": logits, "embedding": pooled, "attention": None, "tokens": tokens}
        select_local = self.arm in {"C2", "C3"}
        selected = local_mask if select_local else torch.ones_like(local_mask, dtype=torch.bool)
        if not torch.all(selected == selected[0]):
            raise ValueError("variable spatial support masks are not supported in one batch")
        selected_indices = selected[0].nonzero(as_tuple=False).flatten()
        selected_tokens = tokens.index_select(1, selected_indices)
        selected_coords = coordinates.index_select(1, selected_indices)
        if self.arm == "C2":
            embedding, attention = self.pool(selected_tokens)
        else:
            embedding, attention = self.pool(selected_tokens, selected_coords)
        logits = self.head(embedding).squeeze(-1)
        return {"logits": logits, "embedding": embedding, "attention": attention, "tokens": selected_tokens, "coordinates": selected_coords}


class RawC1BSupervised(nn.Module):
    """C5: the repository's existing 3-D encoder with direct BCE optimization."""

    def __init__(self, input_channels: int = 7, base_channels: int = 16, latent_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        from ispy_jepa_tmi_clean.corejepa.models.encoder import VisitEncoder3D

        self.encoder = VisitEncoder3D(input_channels, base_channels, latent_dim)
        self.readout = SpatialReadout("C4", input_dim=base_channels * 8, width=128, dropout=dropout)

    def forward(self, image: torch.Tensor, local_mask: torch.Tensor | None = None) -> dict[str, Any]:
        if image.ndim != 5:
            raise ValueError(f"raw image must be [B,7,Z,Y,X], got {tuple(image.shape)}")
        feature_map = self.encoder.features[:-1](image)
        return self.readout(feature_map, local_mask=local_mask)
