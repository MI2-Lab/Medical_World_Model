from __future__ import annotations

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    """Two-convolution residual block.

    Input: ``[B,Cin,Z,Y,X]``. Output: ``[B,Cout,Z',Y',X']``.
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.main = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False)
            if input_channels != output_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.main(image) + self.skip(image)


class VisitEncoder3D(nn.Module):
    """Encode one DCE8 visit.

    Input: ``image [B,8,Z,Y,X]``.
    Output: ``appearance [B,Dz]``.
    """

    def __init__(self, input_channels: int, base_channels: int, latent_dim: int) -> None:
        super().__init__()
        widths = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.features = nn.Sequential(
            ResidualBlock3D(input_channels, widths[0]),
            ResidualBlock3D(widths[0], widths[1], stride=2),
            ResidualBlock3D(widths[1], widths[2], stride=2),
            ResidualBlock3D(widths[2], widths[3], stride=2),
            nn.AdaptiveAvgPool3d(1),
        )
        self.output = nn.Sequential(nn.Flatten(), nn.Linear(widths[-1], latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(image))


class VisitProjector(nn.Module):
    """Project encoder appearance to the JEPA state space ``[B,Dz]``."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, appearance: torch.Tensor) -> torch.Tensor:
        return self.network(appearance)


class GeometryProjector(nn.Module):
    """Map ``q [B,9]`` to a visit-state residual ``[B,Dz]``."""

    def __init__(self, geometry_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.network(geometry)
