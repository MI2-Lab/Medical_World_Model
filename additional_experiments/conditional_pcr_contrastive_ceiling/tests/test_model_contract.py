from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from conditional_ceiling.contracts import (
    load_aligned_full_cohort,
    load_config,
    resolve_input_paths,
    validate_config,
)
from conditional_ceiling.model import (
    ConditionalCeilingModel,
    build_ceiling_model,
    load_confirmed_local3,
    validate_model_contract,
    verify_confirmed_source_root,
)


class TinyConfirmedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.features = nn.Sequential(
            nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
        )
        self.response_projection = nn.Sequential(nn.Linear(2, 192), nn.LayerNorm(192))

    def encode_response(self, image: torch.Tensor) -> torch.Tensor:
        value = image.mean(dim=(2, 3, 4, 5))
        return value.unsqueeze(-1).expand(-1, -1, 192)


def _prefixes(model: nn.Module) -> set[str]:
    return {name for name, value in model.named_parameters() if value.requires_grad}


def test_config_is_hash_locked_and_rejects_scientific_drift(tmp_path: Path) -> None:
    config = load_config()
    changed = json.loads(json.dumps(config))
    changed["matching"]["unmatched_fallback"] = True
    with pytest.raises(ValueError, match="unmatched_fallback"):
        validate_config(changed)
    copy = tmp_path / "experiment.json"
    copy.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_config(copy, expected_sha256="0" * 64)


def test_real_full_808_clinical_fold_cache_alignment() -> None:
    config = load_config()
    paths = resolve_input_paths(config)
    cohort = load_aligned_full_cohort(config, paths, verify_cache_files=True)
    assert len(cohort.patient_ids) == 808
    assert len(cohort.folds) == 4040
    assert len(cohort.cache) == 808
    assert cohort.provenance["clinical_fold_labels_exact"] is True
    assert cohort.provenance["cache_files_stat_verified"] is True


def test_projection_heads_forward_signature_and_shapes() -> None:
    model = build_ceiling_model("B2", TinyConfirmedBackbone(), head_seed=4)
    assert tuple(inspect.signature(model.forward).parameters) == ("image",)
    assert [(head.in_features, head.out_features) for head in model.pcr_heads] == [
        (64, 1),
        (128, 1),
        (192, 1),
    ]
    response = torch.randn(3, 4, 192)
    projected = model.project_response(response)
    assert projected.shape == (3, 4, 64)
    assert [value.shape for value in model.timing_logits(projected)] == [
        (3,),
        (3,),
        (3,),
    ]
    contract = model.architecture_contract()
    assert contract["forward_inputs"] == ["image"]
    assert contract["clinical_inputs"] is False
    assert contract["treatment_input"] is False
    assert contract["bce_weight"] == 0.25
    validate_model_contract(model)


def test_arm_trainability_is_exact() -> None:
    for arm in ("B0", "B1", "B2", "B3"):
        model = build_ceiling_model(arm, TinyConfirmedBackbone(), head_seed=7)
        names = _prefixes(model)
        if arm == "B0":
            assert names == set()
        elif arm == "B1":
            assert names and all(value.startswith("projection_head.") for value in names)
        elif arm == "B2":
            assert any(value.startswith("backbone.encoder.features.3.") for value in names)
            assert not any(value.startswith("backbone.encoder.features.0.") for value in names)
            assert any(value.startswith("backbone.response_projection.") for value in names)
            assert any(value.startswith("projection_head.") for value in names)
            assert any(value.startswith("pcr_heads.") for value in names)
        else:
            assert all(
                any(value.startswith(prefix) for value in names)
                for prefix in (
                    "backbone.encoder.features.0.",
                    "backbone.encoder.features.3.",
                    "backbone.response_projection.",
                    "projection_head.",
                    "pcr_heads.",
                )
            )


def test_forward_rejects_nonimage_contract() -> None:
    model = build_ceiling_model("B1", TinyConfirmedBackbone())
    with pytest.raises(TypeError):
        model(torch.empty(1), torch.empty(1))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="seven DCE channels"):
        model(torch.empty((1, 4, 8, 112, 176, 160), device="meta"))


def test_confirmed_source_hashes_and_real_checkpoint_path_load() -> None:
    paths = resolve_input_paths(load_config())
    observed = verify_confirmed_source_root(paths.confirmation_root)
    assert "lg_response_pilot/model.py" in observed
    checkpoint_path = paths.checkpoint_path(2026, 0)
    relative = checkpoint_path.relative_to(paths.confirmation_root)
    assert relative.parts == (
        "checkpoints",
        "formal_4x8",
        "seed_2026",
        "LOCAL3",
        "fold_0",
        "selected.pt",
    )
    loaded = load_confirmed_local3(
        checkpoint_path,
        paths.confirmation_root,
        expected_seed=2026,
        expected_fold=0,
    )
    assert loaded.seed == 2026 and loaded.fold == 0 and loaded.epoch > 0
    assert loaded.model.arm == "LOCAL3"
    assert len(loaded.checkpoint_sha256) == 64
    model = build_ceiling_model("B2", loaded, head_seed=2026)
    validate_model_contract(model)
    assert any(
        name.startswith("backbone.encoder.features.3.")
        for name in model.trainable_parameter_names()
    )


def test_checkpoint_loader_rejects_path_outside_confirmed_root(tmp_path: Path) -> None:
    paths = resolve_input_paths(load_config())
    fake = tmp_path / "selected.pt"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="outside"):
        load_confirmed_local3(fake, paths.confirmation_root)
