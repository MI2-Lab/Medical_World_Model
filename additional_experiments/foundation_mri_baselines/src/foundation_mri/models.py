"""Strict frozen adapters for the two pre-test-selected encoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .upstream import file_sha256


MEDICALNET_CHECKPOINT_SHA256 = (
    "5b6189cafbee2f5604a7279b62bc163365aa6a86a377e1dc260a14275cacbd84"
)
DINO_CHECKPOINT_SHA256 = (
    "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"
)
MEDICALNET_BACKBONE_PARAMETERS = 46_155_072
DINO_BACKBONE_PARAMETERS = 85_798_656
MEDICALNET_CHANNEL_EMBED_DIM = 2_048
MEDICALNET_EMBED_DIM = 7 * MEDICALNET_CHANNEL_EMBED_DIM
# Official DINO linear evaluation with one last block and avgpool_patchtokens:
# concat(final CLS, mean(final patch tokens)).
DINO_TOKEN_DIM = 768
DINO_EMBED_DIM = 2 * DINO_TOKEN_DIM


@dataclass(frozen=True)
class EncoderAudit:
    model_name: str
    checkpoint_sha256: str
    parameter_count: int
    state_entry_count: int
    representation_dim: int
    load_coverage: str
    frozen: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _conv3x3x3(
    in_planes: int, out_planes: int, *, stride: int = 1, dilation: int = 1
) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class _MedicalNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = _conv3x3x3(
            planes, planes, stride=stride, dilation=dilation
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(output + residual)


class MedicalNetEncoder(nn.Module):
    """MedicalNet's exact shortcut-B, dilated 3-D ResNet-50 backbone.

    This intentionally excludes the randomly initialised ``conv_seg`` decoder
    from the official downstream example.  The released checkpoint itself
    contains precisely these 318 backbone state entries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, dilation=2)
        self.layer4 = self._make_layer(512, 3, dilation=4)
        self.audit: EncoderAudit | None = None

    def _make_layer(
        self, planes: int, blocks: int, *, stride: int = 1, dilation: int = 1
    ) -> nn.Sequential:
        output_planes = planes * _MedicalNetBottleneck.expansion
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != output_planes:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    output_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(output_planes),
            )
        modules: list[nn.Module] = [
            _MedicalNetBottleneck(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = output_planes
        modules.extend(
            _MedicalNetBottleneck(
                self.inplanes, planes, stride=1, dilation=dilation
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*modules)

    def forward_spatial(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("MedicalNet input must be [B,1,D,H,W]")
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        if output.ndim != 5 or output.shape[1] != MEDICALNET_CHANNEL_EMBED_DIM:
            raise AssertionError("MedicalNet layer4 shape contract failed")
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial = self.forward_spatial(inputs)
        return F.adaptive_avg_pool3d(spatial, output_size=1).flatten(1)


class DINOEncoder(nn.Module):
    """Native Meta DINO ViT-B/16 with the official linear-eval feature."""

    def __init__(self, backbone: nn.Module, audit: EncoderAudit) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.audit = audit

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("DINO images must be [B,3,224,224]")
        if not torch.isfinite(images).all():
            raise FloatingPointError("DINO input contains NaN/Inf")
        # Bicubic interpolation can overshoot the upstream hard clip slightly.
        unit = images.float().clamp(-5.0, 5.0).add(5.0).div(10.0)
        return (unit - self.image_mean) / self.image_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(self.preprocess(images))
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (197, DINO_TOKEN_DIM):
            raise AssertionError(
                f"DINO token contract failed: observed {tuple(tokens.shape)}"
            )
        representation = torch.cat(
            (tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1
        )
        if tuple(representation.shape) != (images.shape[0], DINO_EMBED_DIM):
            raise AssertionError("DINO representation contract failed")
        return representation


def _freeze(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("encoder freeze failed")


def _require_checkpoint(path: str | Path, expected_sha256: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = file_sha256(source)
    if observed != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed}"
        )
    return source


def load_medicalnet_encoder(path: str | Path) -> MedicalNetEncoder:
    source = _require_checkpoint(path, MEDICALNET_CHECKPOINT_SHA256)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"state_dict"}:
        raise ValueError("MedicalNet checkpoint must contain only a state_dict")
    raw = payload["state_dict"]
    if not isinstance(raw, dict) or len(raw) != 318:
        raise ValueError("MedicalNet checkpoint must contain 318 state entries")
    if any(not str(key).startswith("module.") for key in raw):
        raise ValueError("MedicalNet checkpoint keys must have one module. prefix")
    state = {str(key).removeprefix("module."): value for key, value in raw.items()}
    if len(state) != len(raw):
        raise ValueError("MedicalNet prefix removal produced duplicate keys")

    model = MedicalNetEncoder()
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            "MedicalNet strict key coverage failed: "
            f"missing={sorted(set(expected) - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - set(expected))[:5]}"
        )
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"MedicalNet tensor shape/type mismatch for {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"MedicalNet checkpoint contains NaN/Inf in {key}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError("strict MedicalNet load unexpectedly reported missing keys")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != MEDICALNET_BACKBONE_PARAMETERS:
        raise AssertionError(f"MedicalNet parameter count drifted: {parameter_count}")
    _freeze(model)
    model.audit = EncoderAudit(
        model_name="medicalnet_resnet50_3dseg8",
        checkpoint_sha256=MEDICALNET_CHECKPOINT_SHA256,
        parameter_count=parameter_count,
        state_entry_count=len(state),
        representation_dim=MEDICALNET_EMBED_DIM,
        load_coverage="318/318 state entries; strict=True; 100% backbone parameters",
        frozen=True,
    )
    return model


def load_dino_encoder(path: str | Path) -> DINOEncoder:
    source = _require_checkpoint(path, DINO_CHECKPOINT_SHA256)
    try:
        import timm
    except ModuleNotFoundError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("timm is required for the native DINO adapter") from exc

    backbone = timm.models.vision_transformer.VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        global_pool="token",
        embed_dim=DINO_TOKEN_DIM,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 150:
        raise ValueError("native DINO checkpoint must contain 150 tensor entries")
    expected = backbone.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            "DINO strict key coverage failed: "
            f"missing={sorted(set(expected) - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - set(expected))[:5]}"
        )
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"DINO tensor shape/type mismatch for {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"DINO checkpoint contains NaN/Inf in {key}")
    result = backbone.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError("strict DINO load unexpectedly reported missing keys")
    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    if parameter_count != DINO_BACKBONE_PARAMETERS:
        raise AssertionError(f"DINO parameter count drifted: {parameter_count}")
    audit = EncoderAudit(
        model_name="dino_vitb16_imagenet1k",
        checkpoint_sha256=DINO_CHECKPOINT_SHA256,
        parameter_count=parameter_count,
        state_entry_count=len(state),
        representation_dim=DINO_EMBED_DIM,
        load_coverage="150/150 state entries; strict=True; native Meta architecture",
        frozen=True,
    )
    model = DINOEncoder(backbone, audit)
    _freeze(model)
    return model


def model_audit(model: nn.Module) -> dict[str, Any]:
    audit = getattr(model, "audit", None)
    if not isinstance(audit, EncoderAudit):
        raise ValueError("model does not expose a completed strict-load audit")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("model is not frozen in eval mode")
    return audit.as_dict()


__all__ = [
    "DINO_BACKBONE_PARAMETERS",
    "DINO_CHECKPOINT_SHA256",
    "DINO_EMBED_DIM",
    "DINOEncoder",
    "EncoderAudit",
    "MEDICALNET_BACKBONE_PARAMETERS",
    "MEDICALNET_CHECKPOINT_SHA256",
    "MEDICALNET_EMBED_DIM",
    "MedicalNetEncoder",
    "load_dino_encoder",
    "load_medicalnet_encoder",
    "model_audit",
]

