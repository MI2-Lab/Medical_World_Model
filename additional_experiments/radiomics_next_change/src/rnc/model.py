"""无 clinical/treatment/9D geometry 的 ROI-assisted image-only world model。"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    """与 clean 分支 VisitEncoder3D 相同的 3D residual block。"""

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
    """由 DCE7 或 DCE7+ROI mask 编码单次访视。"""

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


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


class ImageOnlyCausalTransition(nn.Module):
    """只读取 image-derived states 与位置编码；不接收任何表格/geometry 条件。"""

    def __init__(self, latent_dim: int, depth: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, 3, latent_dim) / math.sqrt(latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads if latent_dim % heads == 0 else 1,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or not 1 <= states.size(1) <= 3:
            raise ValueError(f"transition 期望 [B,L,D], L=1..3；实际 {states.shape}")
        length = states.size(1)
        hidden = states + self.position[:, :length]
        hidden = self.transformer(hidden, mask=causal_attention_mask(length, hidden.device))
        return self.output(hidden)


@dataclass
class WorldModelOutput:
    online_state: torch.Tensor
    target_state: torch.Tensor
    target_next: torch.Tensor
    target_delta: torch.Tensor
    predicted_next: torch.Tensor
    predicted_delta: torch.Tensor
    radiomics_prediction: torch.Tensor | None


class ImageOnlyWorldModel(nn.Module):
    """M0 direct Next-State 或 M1/M2 Next-Change。

    M2 的 radiomics head 只读取 ``predicted_delta``。模型签名不存在
    clinical、treatment、geometry 或 radiomics 输入，结构上阻断推理泄漏。
    """

    VALID_MODES = {"m0", "m1_delta_only", "m1", "m2"}

    def __init__(
        self,
        mode: str,
        image_channels: int = 8,
        base_channels: int = 16,
        latent_dim: int = 192,
        predictor_depth: int = 3,
        predictor_heads: int = 4,
        predictor_mlp_dim: int = 512,
        dropout: float = 0.1,
        radiomics_dim: int = 4,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"未知 model mode: {mode}")
        self.mode = mode
        self.image_channels = int(image_channels)
        self.latent_dim = int(latent_dim)
        self.encoder = VisitEncoder3D(image_channels, base_channels, latent_dim)
        self.projector = VisitProjector(latent_dim)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.transition = ImageOnlyCausalTransition(
            latent_dim, predictor_depth, predictor_heads, predictor_mlp_dim, dropout
        )
        self.radiomics_head = None
        if mode == "m2":
            # 隔离 auxiliary head 初始化的 RNG，使 lambda_rad=0 与 M1 的共同参数/
            # dropout/SIGReg 随机流完全可比。head 本身不使用 dropout。
            with torch.random.fork_rng(devices=[]):
                self.radiomics_head = nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, radiomics_dim),
                )

    @staticmethod
    def _encode_sequence(image: torch.Tensor, encoder: nn.Module, projector: nn.Module) -> torch.Tensor:
        if image.ndim != 6:
            raise ValueError(f"image 期望 [B,V,C,Z,Y,X]，实际 {image.shape}")
        batch, visits = image.shape[:2]
        flat = image.reshape(batch * visits, *image.shape[2:])
        return projector(encoder(flat)).reshape(batch, visits, -1)

    def encode_online(self, image: torch.Tensor) -> torch.Tensor:
        return self._encode_sequence(image, self.encoder, self.projector)

    @torch.no_grad()
    def encode_target(self, image: torch.Tensor) -> torch.Tensor:
        self.target_encoder.eval()
        self.target_projector.eval()
        return self._encode_sequence(image, self.target_encoder, self.target_projector)

    def forward(self, image: torch.Tensor) -> WorldModelOutput:
        if image.size(1) != 4:
            raise ValueError("训练 forward 必须接收 T0–T3 四访视")
        online = self.encode_online(image)
        target = self.encode_target(image).detach()
        transition_output = self.transition(online[:, :-1])
        target_current, target_next = target[:, :-1], target[:, 1:]
        target_delta = target_next - target_current
        if self.mode == "m0":
            predicted_next = transition_output
            predicted_delta = predicted_next - target_current
        else:
            predicted_delta = transition_output
            predicted_next = target_current + predicted_delta
        radiomics_prediction = (
            self.radiomics_head(predicted_delta) if self.radiomics_head is not None else None
        )
        return WorldModelOutput(
            online,
            target,
            target_next,
            target_delta,
            predicted_next,
            predicted_delta,
            radiomics_prediction,
        )

    @torch.no_grad()
    def readout_feature(self, observed_image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """由截至当前 decision point 的 MRI 构造固定 3D 特征。

        返回 ``[current target state, predicted next state, predicted delta]``。
        ``observed_image`` 可为原轨迹、repeated-T0 或 temporal-shuffle。
        """

        if self.training:
            raise RuntimeError("冻结 readout 提取前必须显式调用 model.eval()")
        if observed_image.ndim != 6 or not 1 <= observed_image.size(1) <= 3:
            raise ValueError(f"observed_image 期望 [B,L,C,Z,Y,X], L=1..3；实际 {observed_image.shape}")
        online = self.encode_online(observed_image)
        target = self.encode_target(observed_image)
        transition_output = self.transition(online)
        current = target[:, -1]
        if self.mode == "m0":
            predicted_next = transition_output[:, -1]
            predicted_delta = predicted_next - current
        else:
            predicted_delta = transition_output[:, -1]
            predicted_next = current + predicted_delta
        feature = torch.cat((current, predicted_next, predicted_delta), dim=-1)
        details = {
            "current_state": current,
            "predicted_next": predicted_next,
            "predicted_delta": predicted_delta,
        }
        if self.radiomics_head is not None:
            details["radiomics_prediction"] = self.radiomics_head(predicted_delta)
        return feature, details

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online_module, target_module in (
            (self.encoder, self.target_encoder),
            (self.projector, self.target_projector),
        ):
            for online, target in zip(online_module.parameters(), target_module.parameters()):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
            for online, target in zip(online_module.buffers(), target_module.buffers()):
                target.copy_(online)

    def architecture_contract(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "image_channels": self.image_channels,
            "roi_assisted": self.image_channels == 8,
            "inputs": ["longitudinal_dce_tensor"],
            "forbidden_inputs_absent": ["clinical", "treatment", "geometry_descriptor", "radiomics"],
            "radiomics_head_input": "predicted_image_delta" if self.radiomics_head is not None else None,
            "readout_feature": "concat(current_target_state, predicted_next_state, predicted_delta)",
        }
