"""Strict offline adapter for the official DINOv3 ViT-B/16 checkpoint.

The official Hugging Face snapshot is treated as an immutable local artifact:
the repository revision, complete snapshot file set, byte sizes, and SHA-256
digests are all checked before ``transformers`` is imported.  No hub model ID,
credential, or network fallback is ever passed to ``from_pretrained``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


MODEL_NAME = "dinov3_vitb16_lvd1689m_posthoc"
DINOV3_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DINOV3_BACKBONE_PARAMETERS = 85_660_416
DINOV3_STATE_ENTRY_COUNT = 211
DINOV3_TOKEN_COUNT = 201
DINOV3_TOKEN_DIM = 768
DINOV3_REGISTER_TOKEN_COUNT = 4
DINOV3_PATCH_TOKEN_START = 1 + DINOV3_REGISTER_TOKEN_COUNT
DINOV3_PATCH_TOKEN_COUNT = 196
DINOV3_EMBED_DIM = 2 * DINOV3_TOKEN_DIM
DINOV3_IMAGE_SIZE = 224

# Content digests (not Hugging Face git-blob identifiers).  Keeping the
# preprocessor in the gate prevents silent drift in the frozen image adapter,
# even though preprocessing is implemented explicitly below.
DINOV3_ARTIFACTS: Mapping[str, tuple[int, str]] = {
    "config.json": (
        744,
        "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    ),
    "preprocessor_config.json": (
        585,
        "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
    ),
    "model.safetensors": (
        342_662_192,
        "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    ),
}

_REQUIRED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}
_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


@dataclass(frozen=True)
class EncoderAudit:
    """JSON-safe evidence emitted only after a strict complete model load."""

    model_name: str
    repository_id: str
    revision: str
    artifacts: Mapping[str, Mapping[str, object]]
    parameter_count: int
    state_entry_count: int
    representation_dim: int
    token_contract: str
    load_coverage: str
    frozen: bool
    offline: bool
    token_disabled: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_offline_runtime() -> None:
    """Fail closed unless hub access and implicit credentials are disabled."""

    drifted = {
        name: os.environ.get(name)
        for name, expected in _REQUIRED_OFFLINE_ENV.items()
        if os.environ.get(name) != expected
    }
    if drifted:
        required = ", ".join(
            f"{name}={expected}" for name, expected in _REQUIRED_OFFLINE_ENV.items()
        )
        raise RuntimeError(
            "strict offline DINOv3 loading requires environment flags: " + required
        )
    present_tokens = sorted(
        name for name in _TOKEN_ENV_NAMES if bool(os.environ.get(name, "").strip())
    )
    if present_tokens:
        raise RuntimeError(
            "DINOv3 loading forbids token-bearing environment variables: "
            + ", ".join(present_tokens)
        )


def validate_snapshot(snapshot_dir: str | Path) -> Path:
    """Prove the local snapshot is the one frozen for this experiment."""

    source = Path(snapshot_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"DINOv3 local snapshot directory is missing: {source}")
    if source.name != DINOV3_REVISION:
        raise ValueError(
            "DINOv3 snapshot revision mismatch: "
            f"expected directory name {DINOV3_REVISION}, observed {source.name}"
        )
    observed_names = {entry.name for entry in source.iterdir()}
    expected_names = set(DINOV3_ARTIFACTS)
    if observed_names != expected_names:
        raise ValueError(
            "DINOv3 snapshot artifact inventory mismatch: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    for name, (expected_size, expected_sha256) in DINOV3_ARTIFACTS.items():
        artifact = source / name
        if not artifact.is_file():
            raise FileNotFoundError(f"DINOv3 snapshot artifact is not a file: {name}")
        observed_size = artifact.stat().st_size
        if observed_size != int(expected_size):
            raise ValueError(
                f"DINOv3 artifact size mismatch for {name}: "
                f"expected {expected_size}, observed {observed_size}"
            )
        observed_sha256 = _file_sha256(artifact)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"DINOv3 artifact SHA-256 mismatch for {name}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )

    # The file digests are authoritative; these semantic checks make contract
    # failures easier to diagnose than a later token-shape assertion.
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "architectures": ["DINOv3ViTModel"],
        "model_type": "dinov3_vit",
        "hidden_size": DINOV3_TOKEN_DIM,
        "image_size": DINOV3_IMAGE_SIZE,
        "patch_size": 16,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "num_register_tokens": DINOV3_REGISTER_TOKEN_COUNT,
        "layer_norm_eps": 1e-5,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"DINOv3 config contract mismatch for {key}: "
                f"expected {expected!r}, observed {config.get(key)!r}"
            )
    preprocessor = json.loads(
        (source / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    if preprocessor.get("size") != {"height": 224, "width": 224}:
        raise ValueError("DINOv3 preprocessor image size must be exactly 224x224")
    if preprocessor.get("image_mean") != [0.485, 0.456, 0.406]:
        raise ValueError("DINOv3 preprocessor ImageNet mean drifted")
    if preprocessor.get("image_std") != [0.229, 0.224, 0.225]:
        raise ValueError("DINOv3 preprocessor ImageNet standard deviation drifted")
    if (
        preprocessor.get("do_resize") is not True
        or preprocessor.get("do_normalize") is not True
    ):
        raise ValueError("DINOv3 preprocessor resize/normalization contract drifted")
    return source


def _freeze(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise AssertionError("DINOv3 encoder freeze failed")


class DINOv3Encoder(nn.Module):
    """DINOv3 final CLS plus mean final patch-token representation.

    DINOv3 inserts four register tokens after CLS.  They are deliberately
    excluded from the patch mean: ``tokens[:, 5:, :]`` contains exactly the
    14x14 image patches at the frozen 224x224 input size.
    """

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
        _freeze(self)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("DINOv3 images must be a tensor [B,3,224,224]")
        if tuple(images.shape[1:]) != (3, DINOV3_IMAGE_SIZE, DINOV3_IMAGE_SIZE):
            raise ValueError("DINOv3 images must be [B,3,224,224]")
        if not images.is_floating_point():
            raise TypeError("DINOv3 images must be floating point")
        if not torch.isfinite(images).all():
            raise FloatingPointError("DINOv3 input contains NaN/Inf")
        # Exactly the unchanged DINOv1 mapping. Bicubic interpolation in the
        # upstream spatial adapter may overshoot the hard C1B clip slightly.
        unit = images.float().clamp(-5.0, 5.0).add(5.0).div(10.0)
        normalized = (unit - self.image_mean) / self.image_std
        if not torch.isfinite(normalized).all():
            raise FloatingPointError("DINOv3 preprocessing produced NaN/Inf")
        return normalized

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=self.preprocess(images), return_dict=True)
        tokens = getattr(outputs, "last_hidden_state", None)
        expected = (images.shape[0], DINOV3_TOKEN_COUNT, DINOV3_TOKEN_DIM)
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != expected:
            observed = getattr(tokens, "shape", None)
            raise AssertionError(
                f"DINOv3 token contract failed: expected {expected}, observed {observed}"
            )
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("DINOv3 tokens contain NaN/Inf")
        patches = tokens[:, DINOV3_PATCH_TOKEN_START:, :]
        if tuple(patches.shape[1:]) != (
            DINOV3_PATCH_TOKEN_COUNT,
            DINOV3_TOKEN_DIM,
        ):
            raise AssertionError("DINOv3 patch-token contract failed")
        representation = torch.cat((tokens[:, 0, :], patches.mean(dim=1)), dim=1)
        if tuple(representation.shape) != (images.shape[0], DINOV3_EMBED_DIM):
            raise AssertionError("DINOv3 representation contract failed")
        if not torch.isfinite(representation).all():
            raise FloatingPointError("DINOv3 representation contains NaN/Inf")
        return representation


def _strict_loading_info(info: Any) -> None:
    if not isinstance(info, dict):
        raise ValueError("DINOv3 loader did not return strict loading information")
    expected_keys = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if not expected_keys.issubset(info):
        raise ValueError("DINOv3 loading information schema is incomplete")
    failures = {
        key: sorted(str(value) for value in info[key])
        for key in expected_keys
        if info[key]
    }
    if failures:
        raise ValueError(f"DINOv3 strict state coverage failed: {failures}")


def load_dinov3_encoder(snapshot_dir: str | Path) -> DINOv3Encoder:
    """Load only the hash-gated local revision, with no credential fallback."""

    require_offline_runtime()
    source = validate_snapshot(snapshot_dir)
    try:
        from transformers import DINOv3ViTModel
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - env gate
        raise RuntimeError(
            "transformers with DINOv3ViTModel support is required"
        ) from exc

    loaded = DINOv3ViTModel.from_pretrained(
        source,
        local_files_only=True,
        token=False,
        revision=DINOV3_REVISION,
        use_safetensors=True,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("DINOv3 strict loader did not return model and loading info")
    backbone, loading_info = loaded
    if not isinstance(backbone, DINOv3ViTModel):
        raise TypeError("local snapshot did not resolve to DINOv3ViTModel")
    _strict_loading_info(loading_info)

    state = backbone.state_dict()
    if len(state) != DINOV3_STATE_ENTRY_COUNT:
        raise AssertionError(f"DINOv3 state entry count drifted: {len(state)}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"DINOv3 state entry is not a tensor: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"DINOv3 state contains NaN/Inf in {key}")
    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    if parameter_count != DINOV3_BACKBONE_PARAMETERS:
        raise AssertionError(f"DINOv3 parameter count drifted: {parameter_count}")

    artifacts = {
        name: {"size_bytes": int(size), "sha256": sha256}
        for name, (size, sha256) in sorted(DINOV3_ARTIFACTS.items())
    }
    audit = EncoderAudit(
        model_name=MODEL_NAME,
        repository_id=DINOV3_REPO_ID,
        revision=DINOV3_REVISION,
        artifacts=artifacts,
        parameter_count=parameter_count,
        state_entry_count=len(state),
        representation_dim=DINOV3_EMBED_DIM,
        token_contract=(
            "[B,201,768]: CLS=0; registers=1:5 excluded; "
            "mean patches=5:; concat CLS+patch_mean=1536"
        ),
        load_coverage="211/211 state entries; no missing/unexpected/mismatched keys",
        frozen=True,
        offline=True,
        token_disabled=True,
    )
    model = DINOv3Encoder(backbone, audit)
    _freeze(model)
    return model


def model_audit(model: nn.Module) -> dict[str, Any]:
    audit = getattr(model, "audit", None)
    if not isinstance(audit, EncoderAudit):
        raise ValueError("model does not expose a completed DINOv3 strict-load audit")
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("DINOv3 model is not frozen in eval mode")
    if audit.parameter_count != DINOV3_BACKBONE_PARAMETERS:
        raise ValueError("DINOv3 audit parameter count drifted")
    if audit.state_entry_count != DINOV3_STATE_ENTRY_COUNT:
        raise ValueError("DINOv3 audit state coverage drifted")
    if audit.representation_dim != DINOV3_EMBED_DIM:
        raise ValueError("DINOv3 audit representation dimension drifted")
    return audit.as_dict()


__all__ = [
    "DINOV3_ARTIFACTS",
    "DINOV3_BACKBONE_PARAMETERS",
    "DINOV3_EMBED_DIM",
    "DINOV3_IMAGE_SIZE",
    "DINOV3_PATCH_TOKEN_COUNT",
    "DINOV3_PATCH_TOKEN_START",
    "DINOV3_REGISTER_TOKEN_COUNT",
    "DINOV3_REPO_ID",
    "DINOV3_REVISION",
    "DINOV3_STATE_ENTRY_COUNT",
    "DINOV3_TOKEN_COUNT",
    "DINOV3_TOKEN_DIM",
    "DINOv3Encoder",
    "EncoderAudit",
    "MODEL_NAME",
    "load_dinov3_encoder",
    "model_audit",
    "require_offline_runtime",
    "validate_snapshot",
]
