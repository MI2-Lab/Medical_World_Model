from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


def fixed_local_weights(shape_dhw: tuple[int, int, int], spacing_zyx_mm: tuple[float, float, float] = (16.0, 7.2, 7.2), local_mm: float = 64.0) -> torch.Tensor:
    """Return exact frozen feature-cell overlap with the central 64-mm cube."""

    # C1B-H final features have stride eight and input voxel spacing
    # (2.0, 0.9, 0.9) in ZYX.  Expressing the formula through the cell spacing
    # keeps this helper useful for contract-shaped smoke tensors while matching
    # the audited final-grid centers for (14, 22, 20).
    input_shape = tuple(int(size) * 8 for size in shape_dhw)
    input_spacing = tuple(float(value) / 8.0 for value in spacing_zyx_mm)
    half = float(local_mm) / 2.0
    fractions: list[torch.Tensor] = []
    for input_size, spacing in zip(input_shape, input_spacing):
        index = torch.arange(shape_dhw[len(fractions)], dtype=torch.float64)
        centers = (index * 8.0 - 0.5 * (input_size - 1.0)) * spacing
        half_cell = 4.0 * spacing
        lower = torch.maximum(centers - half_cell, torch.full_like(centers, -half))
        upper = torch.minimum(centers + half_cell, torch.full_like(centers, half))
        fractions.append(((upper - lower).clamp_min(0.0) / (2.0 * half_cell)).to(torch.float32))
    z, y, x = fractions
    return z[:, None, None] * y[None, :, None] * x[None, None, :]


def fixed_local_mask(shape_dhw: tuple[int, int, int], spacing_zyx_mm: tuple[float, float, float] = (16.0, 7.2, 7.2), local_mm: float = 64.0) -> torch.Tensor:
    """Return the outcome-free support of the audited fractional local weights."""

    return fixed_local_weights(shape_dhw, spacing_zyx_mm=spacing_zyx_mm, local_mm=local_mm) > 0


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

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None, collect_attention: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        values = self.input_projection(tokens)
        query = self.query.expand(tokens.shape[0], -1, -1)
        weights = None
        for attention, feed_forward in zip(self.attention, self.feed_forward):
            update, weights = attention(query, values, values, key_padding_mask=padding_mask, need_weights=collect_attention, average_attn_weights=False)
            query = query + update
            query = feed_forward(query)
        return self.norm(query[:, 0]), weights


class SpatialTokenBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None, collect_attention: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.norm1(x)
        attended, weights = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=collect_attention, average_attn_weights=False)
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

    def forward(self, tokens: torch.Tensor, coordinates: torch.Tensor, padding_mask: torch.Tensor | None = None, collect_attention: bool = True) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        x = self.input_projection(tokens) + self.position(coordinates)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat((torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device), padding_mask), dim=1)
        attention_maps: list[torch.Tensor | None] = []
        for block in self.blocks:
            x, weights = block(x, padding_mask, collect_attention=collect_attention)
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

    def forward(self, feature_map: torch.Tensor, local_mask: torch.Tensor | None = None, coordinates: torch.Tensor | None = None, valid_weights: torch.Tensor | None = None, collect_attention: bool = True) -> dict[str, Any]:
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
            embedding, attention = self.pool(selected_tokens, collect_attention=collect_attention)
        else:
            embedding, attention = self.pool(selected_tokens, selected_coords, collect_attention=collect_attention)
        logits = self.head(embedding).squeeze(-1)
        return {"logits": logits, "embedding": embedding, "attention": attention, "tokens": selected_tokens, "coordinates": selected_coords}


class RawC1BSupervised(nn.Module):
    """C5: the repository's existing 3-D encoder with direct BCE optimization."""

    def __init__(self, input_channels: int = 7, base_channels: int = 16, latent_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        if input_channels != 7:
            raise ValueError("C5 is frozen to seven C1B-H DCE channels")
        g3_src = Path(__file__).resolve().parents[3] / "g3_multiseed_generalization" / "src"
        if str(g3_src) not in sys.path:
            sys.path.insert(0, str(g3_src))
        from dgrs.model import SpatialVisitEncoder3D

        self.encoder = SpatialVisitEncoder3D(base_channels)
        self.readout = SpatialReadout("C4", input_dim=base_channels * 8, width=128, dropout=dropout)

    def forward(self, image: torch.Tensor, local_mask: torch.Tensor | None = None, collect_attention: bool = True) -> dict[str, Any]:
        if image.ndim != 5:
            raise ValueError(f"raw image must be [B,7,Z,Y,X], got {tuple(image.shape)}")
        feature_map = self.encoder(image)
        return self.readout(feature_map, local_mask=local_mask, collect_attention=collect_attention)


class SequenceClassifier(nn.Module):
    """Timing-specific classifier over literal observed visit prefixes.

    The spatial readout is applied independently to each observed visit, then
    the embeddings are concatenated in visit order.  No future visit is ever
    passed to a timing-specific head.
    """

    def __init__(self, base: nn.Module, steps: int, embedding_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if steps not in {1, 2, 3, 4}:
            raise ValueError("steps must be one of 1, 2, 3, or 4")
        self.base = base
        self.steps = steps
        self.embedding_dim = embedding_dim
        self.head = nn.Sequential(
            nn.LayerNorm(steps * embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(steps * embedding_dim, 1),
        )

    def forward(self, batch: torch.Tensor, local_mask: torch.Tensor | None = None, collect_attention: bool = True) -> dict[str, Any]:
        if batch.ndim == 5:
            batch = batch.unsqueeze(1)
        if batch.ndim != 6 or batch.shape[1] != self.steps:
            raise ValueError(f"sequence input must be [B,{self.steps},C,D,H,W], got {tuple(batch.shape)}")
        embeddings: list[torch.Tensor] = []
        attentions: list[Any] = []
        last: dict[str, Any] | None = None
        for visit in range(self.steps):
            current = self.base(batch[:, visit], local_mask=local_mask, collect_attention=collect_attention)
            embeddings.append(current["embedding"])
            attentions.append(current.get("attention"))
            last = current
        sequence_embedding = torch.cat(embeddings, dim=1)
        logits = self.head(sequence_embedding).squeeze(-1)
        return {
            "logits": logits,
            "embedding": sequence_embedding,
            "visit_embeddings": embeddings,
            "attention": attentions,
            "tokens": None if last is None else last.get("tokens"),
            "coordinates": None if last is None else last.get("coordinates"),
        }


def load_encoder_weights(model: RawC1BSupervised, checkpoint: str | Any) -> dict[str, Any]:
    """Initialize C5 from a selected prior encoder without importing its head."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks a state_dict")
    encoder_state = {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}
    if not encoder_state:
        raise ValueError("checkpoint contains no encoder weights")
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"C5 encoder checkpoint architecture mismatch: missing={missing}, unexpected={unexpected}")
    return {"checkpoint": str(checkpoint), "encoder_keys": len(encoder_state), "selected": bool(payload.get("selected", False))}
