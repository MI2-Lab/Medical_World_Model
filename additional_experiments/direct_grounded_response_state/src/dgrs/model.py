"""DCE7-only observed response state 与不变的 M0 Next-State transition。"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


VALID_MODELS = ("G1", "G2", "G3", "G4")
ROI_MODELS = frozenset({"G2", "G4"})
GROUNDED_MODELS = frozenset({"G3", "G4"})


class ResidualBlock3D(nn.Module):
    """与 prior M0 VisitEncoder3D 完全相同的 residual block。"""

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


class SpatialVisitEncoder3D(nn.Module):
    """只接收七个 DCE intensity/enhancement channels；签名中没有 mask。"""

    input_channels = 7

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        widths = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.output_channels = widths[-1]
        self.features = nn.Sequential(
            ResidualBlock3D(7, widths[0]),
            ResidualBlock3D(widths[0], widths[1], stride=2),
            ResidualBlock3D(widths[1], widths[2], stride=2),
            ResidualBlock3D(widths[2], widths[3], stride=2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.size(1) != 7:
            raise ValueError(f"encoder 只接受 [N,7,Z,Y,X] DCE7；实际 {tuple(image.shape)}")
        return self.features(image)


def normalized_occupancy_roi_mean(
    spatial: torch.Tensor,
    roi_mask: torch.Tensor,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只用 normalized occupancy 做定位；empty mask 严格回退 GAP。

    返回 pooled feature 与诊断用 ``roi_valid``。分母、voxel count 或任何几何量
    均不会离开本函数，也不会进入 learned module。
    """

    if spatial.ndim != 5:
        raise ValueError(f"spatial 期望 [N,C,z,y,x]；实际 {tuple(spatial.shape)}")
    if roi_mask.ndim != 5 or roi_mask.size(1) != 1 or roi_mask.size(0) != spatial.size(0):
        raise ValueError(f"roi_mask 期望 [N,1,Z,Y,X]；实际 {tuple(roi_mask.shape)}")
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正")
    if not bool(torch.isfinite(roi_mask).all()) or bool((roi_mask < 0).any()):
        raise ValueError("roi_mask 必须是有限非负 occupancy")
    occupancy = F.adaptive_avg_pool3d(roi_mask.to(dtype=spatial.dtype), spatial.shape[-3:])
    support = occupancy.sum(dim=(-3, -2, -1))
    roi_valid = support[:, 0] > 0
    roi = (spatial * occupancy).sum(dim=(-3, -2, -1)) / support.clamp_min(epsilon)
    gap = spatial.mean(dim=(-3, -2, -1))
    pooled = torch.where(roi_valid[:, None], roi, gap)
    return pooled, roi_valid


class VisitProjector(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        return self.network(response)


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


class ImageOnlyCausalTransition(nn.Module):
    """prior M0 transition；只读 projected image state 与 position。"""

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
            raise ValueError(f"transition 期望 [B,L,D], L=1..3；实际 {tuple(states.shape)}")
        length = states.size(1)
        hidden = states + self.position[:, :length]
        hidden = self.transformer(hidden, mask=causal_attention_mask(length, hidden.device))
        return self.output(hidden)


@dataclass
class DGRSOutput:
    response_state: torch.Tensor
    online_state: torch.Tensor
    target_response_state: torch.Tensor
    target_state: torch.Tensor
    target_next: torch.Tensor
    predicted_next: torch.Tensor
    ftv_prediction: torch.Tensor | None
    roi_valid: torch.Tensor | None


class DGRSWorldModel(nn.Module):
    """G1–G4；FTV target 永远不属于 forward 输入。"""

    def __init__(
        self,
        model_name: str,
        image_channels: int = 7,
        pooling: str | None = None,
        direct_ftv_grounding: bool | None = None,
        base_channels: int = 16,
        latent_dim: int = 192,
        predictor_depth: int = 3,
        predictor_heads: int = 4,
        predictor_mlp_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        model_name = str(model_name).upper()
        if model_name not in VALID_MODELS:
            raise ValueError(f"model_name 必须是 {VALID_MODELS}，实际 {model_name}")
        if int(image_channels) != 7:
            raise ValueError("G1–G4 backbone 的 image_channels 必须严格为 7")
        expected_pooling = "roi_mean" if model_name in ROI_MODELS else "gap"
        expected_grounding = model_name in GROUNDED_MODELS
        if pooling is not None and str(pooling) != expected_pooling:
            raise ValueError(f"{model_name} pooling 必须为 {expected_pooling}")
        if direct_ftv_grounding is not None and bool(direct_ftv_grounding) != expected_grounding:
            raise ValueError(f"{model_name} direct_ftv_grounding 必须为 {expected_grounding}")
        self.model_name = model_name
        self.image_channels = 7
        self.pooling = expected_pooling
        self.direct_ftv_grounding = expected_grounding
        self.base_channels = int(base_channels)
        self.latent_dim = int(latent_dim)
        self.predictor_depth = int(predictor_depth)
        self.predictor_heads = int(predictor_heads)
        self.predictor_mlp_dim = int(predictor_mlp_dim)
        self.dropout = float(dropout)

        self.encoder = SpatialVisitEncoder3D(base_channels)
        pooled_dim = self.encoder.output_channels
        self.response_projection = nn.Sequential(nn.Linear(pooled_dim, latent_dim), nn.LayerNorm(latent_dim))
        self.projector = VisitProjector(latent_dim)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_response_projection = copy.deepcopy(self.response_projection).requires_grad_(False)
        self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.transition = ImageOnlyCausalTransition(
            latent_dim, predictor_depth, predictor_heads, predictor_mlp_dim, dropout
        )
        self.ftv_head: nn.Linear | None = None
        if self.direct_ftv_grounding:
            # fork_rng 在退出时恢复公共随机流；G3/G4 的 head 不改变与 G1/G2
            # 配对模型的 dropout、SIGReg 或 DataLoader RNG 起点。
            with torch.random.fork_rng(devices=[]):
                self.ftv_head = nn.Linear(latent_dim, 1)

    @property
    def requires_roi_mask(self) -> bool:
        return self.model_name in ROI_MODELS

    def _validate_sequence_inputs(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None
    ) -> tuple[int, int, torch.Tensor | None]:
        if image.ndim != 6 or image.size(2) != 7:
            raise ValueError(f"image 期望 [B,V,7,Z,Y,X]；实际 {tuple(image.shape)}")
        batch, visits = image.shape[:2]
        if self.requires_roi_mask:
            if roi_mask is None:
                raise ValueError(f"{self.model_name} 需要分离的 roi_mask，仅用于 normalized pooling")
            expected = (batch, visits, 1, *image.shape[-3:])
            if tuple(roi_mask.shape) != expected:
                raise ValueError(f"roi_mask 期望 {expected}；实际 {tuple(roi_mask.shape)}")
        elif roi_mask is not None:
            raise ValueError(f"{self.model_name} 不接受 roi_mask；不得把 mask 偷渡进 GAP 模型")
        return batch, visits, roi_mask

    def _encode_sequence(
        self,
        image: torch.Tensor,
        roi_mask: torch.Tensor | None,
        encoder: nn.Module,
        response_projection: nn.Module,
        projector: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, visits, roi_mask = self._validate_sequence_inputs(image, roi_mask)
        flat = image.reshape(batch * visits, *image.shape[2:])
        spatial = encoder(flat)
        if self.requires_roi_mask:
            assert roi_mask is not None
            pooled, valid = normalized_occupancy_roi_mean(
                spatial, roi_mask.reshape(batch * visits, *roi_mask.shape[2:])
            )
            roi_valid: torch.Tensor | None = valid.reshape(batch, visits)
        else:
            pooled = spatial.mean(dim=(-3, -2, -1))
            roi_valid = None
        response = response_projection(pooled).reshape(batch, visits, self.latent_dim)
        projected = projector(response).reshape(batch, visits, self.latent_dim)
        return response, projected, roi_valid

    def encode_online(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self._encode_sequence(
            image, roi_mask, self.encoder, self.response_projection, self.projector
        )

    @torch.no_grad()
    def encode_target(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self.target_encoder.eval()
        self.target_response_projection.eval()
        self.target_projector.eval()
        return self._encode_sequence(
            image,
            roi_mask,
            self.target_encoder,
            self.target_response_projection,
            self.target_projector,
        )

    def encode_response(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """返回评估主表征 online pre-projector ``r``。"""

        response, _, _ = self.encode_online(image, roi_mask)
        return response

    def forward(self, image: torch.Tensor, roi_mask: torch.Tensor | None = None) -> DGRSOutput:
        if image.ndim != 6 or image.size(1) != 4:
            raise ValueError("训练 forward 必须接收 T0–T3 四访 DCE7")
        response, online, roi_valid = self.encode_online(image, roi_mask)
        target_response, target, _ = self.encode_target(image, roi_mask)
        target = target.detach()
        target_response = target_response.detach()
        predicted_next = self.transition(online[:, :-1])
        ftv_prediction = self.ftv_head(response).squeeze(-1) if self.ftv_head is not None else None
        return DGRSOutput(
            response_state=response,
            online_state=online,
            target_response_state=target_response,
            target_state=target,
            target_next=target[:, 1:],
            predicted_next=predicted_next,
            ftv_prediction=ftv_prediction,
            roi_valid=roi_valid,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 < momentum < 1.0:
            raise ValueError("EMA momentum 必须位于 (0,1)")
        for online_module, target_module in (
            (self.encoder, self.target_encoder),
            (self.response_projection, self.target_response_projection),
            (self.projector, self.target_projector),
        ):
            for online, target in zip(online_module.parameters(), target_module.parameters()):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
            for online, target in zip(online_module.buffers(), target_module.buffers()):
                target.copy_(online)

    def model_config(self) -> dict[str, Any]:
        """输出可直接 ``DGRSWorldModel(**payload)`` 的 resolved config。"""

        return {
            "model_name": self.model_name,
            "image_channels": self.image_channels,
            "pooling": self.pooling,
            "direct_ftv_grounding": self.direct_ftv_grounding,
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
            "predictor_depth": self.predictor_depth,
            "predictor_heads": self.predictor_heads,
            "predictor_mlp_dim": self.predictor_mlp_dim,
            "dropout": self.dropout,
        }

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_name": self.model_name,
            "backbone_input": "DCE7",
            "image_channels": 7,
            "first_conv_in_channels": 7,
            "roi_mask_backbone_input": False,
            "pooling": self.pooling,
            "roi_mask_use": "normalized_occupancy_roi_mean_only" if self.requires_roi_mask else "absent",
            "empty_roi_behavior": "strict_gap_fallback" if self.requires_roi_mask else None,
            "observed_response_state": "online_preprojector_r",
            "response_dim": self.latent_dim,
            "jepa_state": "projector(r)",
            "transition": "M0_direct_next_state_causal_transformer",
            "ftv_head": "Linear(response_dim,1)" if self.ftv_head is not None else None,
            "ftv_is_forward_input": False,
            "forbidden_inputs_absent": [
                "clinical",
                "treatment",
                "radiomics",
                "mask_geometry",
                "voxel_count",
                "explicit_volume",
            ],
        }


def load_checkpoint_for_evaluation(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[DGRSWorldModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_config" not in payload:
        raise ValueError(f"checkpoint schema 非法: {path}")
    model = DGRSWorldModel(**payload["model_config"])
    state = payload.get("state_dict", payload.get("model_state"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint 缺 state_dict/model_state")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


# 评估端约定的短别名。
load_checkpoint = load_checkpoint_for_evaluation


__all__ = [
    "DGRSOutput",
    "DGRSWorldModel",
    "GROUNDED_MODELS",
    "ROI_MODELS",
    "VALID_MODELS",
    "load_checkpoint",
    "load_checkpoint_for_evaluation",
    "normalized_occupancy_roi_mean",
]
