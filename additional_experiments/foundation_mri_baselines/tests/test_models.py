from __future__ import annotations

from pathlib import Path

import torch

from foundation_mri.models import (
    DINO_BACKBONE_PARAMETERS,
    DINO_EMBED_DIM,
    MEDICALNET_BACKBONE_PARAMETERS,
    load_dino_encoder,
    load_medicalnet_encoder,
    model_audit,
)
from foundation_mri.upstream import (  # noqa: E402
    EXPECTED_STAGE_B_CONTRACTS_SHA256,
    STAGE_B_SRC,
    file_sha256,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_stage_b_contracts_source_is_hash_gated() -> None:
    assert EXPECTED_STAGE_B_CONTRACTS_SHA256 == (
        "48d7738b6764780ba2e784f826be44ac718fdbb0beb526ec31c3c5525cba4bf9"
    )
    assert file_sha256(STAGE_B_SRC / "c1b_stage_b/contracts.py") == (
        EXPECTED_STAGE_B_CONTRACTS_SHA256
    )


def test_official_checkpoints_strictly_cover_backbones() -> None:
    medicalnet = load_medicalnet_encoder(
        EXPERIMENT_ROOT / "checkpoints" / "medicalnet" / "resnet_50.pth"
    )
    dino = load_dino_encoder(
        EXPERIMENT_ROOT
        / "checkpoints"
        / "dino"
        / "dino_vitbase16_pretrain.pth"
    )
    assert model_audit(medicalnet)["parameter_count"] == MEDICALNET_BACKBONE_PARAMETERS
    assert model_audit(dino)["parameter_count"] == DINO_BACKBONE_PARAMETERS
    assert not medicalnet.training and not dino.training
    assert not any(parameter.requires_grad for parameter in medicalnet.parameters())
    assert not any(parameter.requires_grad for parameter in dino.parameters())


def test_encoder_output_contracts() -> None:
    medicalnet = load_medicalnet_encoder(
        EXPERIMENT_ROOT / "checkpoints" / "medicalnet" / "resnet_50.pth"
    )
    dino = load_dino_encoder(
        EXPERIMENT_ROOT
        / "checkpoints"
        / "dino"
        / "dino_vitbase16_pretrain.pth"
    )
    with torch.inference_mode():
        medical_map = medicalnet.forward_spatial(torch.zeros(1, 1, 32, 32, 32))
        dino_feature = dino(torch.zeros(2, 3, 224, 224))
    assert medical_map.shape == (1, 2048, 4, 4, 4)
    assert dino_feature.shape == (2, DINO_EMBED_DIM)
    assert torch.isfinite(medical_map).all()
    assert torch.isfinite(dino_feature).all()
