from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.main = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False), nn.GroupNorm(groups, out_ch), nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False), nn.GroupNorm(groups, out_ch), nn.SiLU(),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, stride, bias=False) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class VisitEncoder(nn.Module):
    def __init__(self, channels: int, base: int, width: int) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.net = nn.Sequential(
            ResidualBlock3D(channels, widths[0]), ResidualBlock3D(widths[0], widths[1], 2),
            ResidualBlock3D(widths[1], widths[2], 2), ResidualBlock3D(widths[2], widths[3], 2),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(widths[3], width), nn.LayerNorm(width),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class PairChangeHead(nn.Module):
    """Reads both observed states and their learned difference, never pixels subtracted across visits."""
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width * 3), nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, prior: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((prior, current, current - prior), dim=-1)).squeeze(-1)


class HistoryPredictor(nn.Module):
    """Capacity-matched two-slot predictor. The caller chooses real or duplicated history."""
    def __init__(self, width: int, depth: int, heads: int, mlp: int, dropout: float) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, 2, width) / width ** 0.5)
        layer = nn.TransformerEncoderLayer(width, heads, mlp, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, prior: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        sequence = torch.stack((prior, current), dim=1) + self.position
        hidden = self.transformer(sequence)
        return self.output(hidden[:, -1]).squeeze(-1)


@dataclass
class Output:
    states: torch.Tensor
    target_states: torch.Tensor
    predicted_next_state: torch.Tensor
    static: torch.Tensor
    observed_change: torch.Tensor
    future_change: torch.Tensor | None


class ChangeRepairModel(nn.Module):
    def __init__(self, channels: int = 7, base: int = 16, width: int = 192, depth: int = 4, heads: int = 8, mlp: int = 512, dropout: float = .1) -> None:
        super().__init__()
        self.encoder = VisitEncoder(channels, base, width)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.next_state_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.static_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.change_head = PairChangeHead(width)
        self.future_head = HistoryPredictor(width, depth, heads, mlp, dropout)

    def encode(self, image: torch.Tensor, target: bool = False) -> torch.Tensor:
        if image.ndim != 6 or image.shape[1] != 4:
            raise ValueError(f"expected [B,4,C,Z,Y,X], got {tuple(image.shape)}")
        module = self.target_encoder if target else self.encoder
        batch, visits = image.shape[:2]
        return module(image.reshape(batch * visits, *image.shape[2:])).reshape(batch, visits, -1)

    def forward(self, image: torch.Tensor, *, history: bool = True) -> Output:
        states = self.encode(image)
        with torch.no_grad():
            target_states = self.encode(image, target=True)
        predicted_next_state = self.next_state_head(states[:, :-1])
        static = self.static_head(states).squeeze(-1)
        observed = self.change_head(states[:, :-1].reshape(-1, states.size(-1)), states[:, 1:].reshape(-1, states.size(-1))).reshape(image.size(0), 3)
        prior = states[:, 0:2] if history else states[:, 1:3]
        current = states[:, 1:3]
        future = self.future_head(prior.reshape(-1, states.size(-1)), current.reshape(-1, states.size(-1))).reshape(image.size(0), 2)
        return Output(states, target_states, predicted_next_state, static, observed, future)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
