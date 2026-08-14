from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from patch_token_wm import (
    PatchTokenWorldModel,
    TransitionCondition,
    deterministic_mask_indices,
    source_to_query_block_mask,
)


SMALL_SHAPE = (16, 16, 16)
SMALL_SPACING = (4.0, 4.0, 4.0)


def small_model(seed: int = 7) -> PatchTokenWorldModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PatchTokenWorldModel(
            input_shape_zyx=SMALL_SHAPE,
            spacing_xyz_mm=SMALL_SPACING,
            require_formal_geometry=False,
        )


def condition(batch: int, arm: int = 2) -> TransitionCondition:
    clinical = torch.tensor([[1.0, 0.0, 1.0, 0.25, 0.0]]).expand(batch, -1).clone()
    return TransitionCondition.nominal(
        torch.full((batch,), arm, dtype=torch.long), clinical
    )


def test_predictor_locked_architecture_and_source_to_query_block_mask() -> None:
    model = small_model()
    assert len(model.predictor.transformer.layers) == 4
    for layer in model.predictor.transformer.layers:
        assert layer.self_attn.embed_dim == 128
        assert layer.self_attn.num_heads == 8
        assert layer.linear1.out_features == 512
        assert layer.dropout.p == pytest.approx(0.1)
    assert not any("film" in name.casefold() for name, _ in model.named_modules())
    mask = source_to_query_block_mask(5, 2)
    assert mask.shape == (7, 7)
    assert torch.isneginf(mask[:5, 5:]).all()
    assert torch.equal(mask[5:, :], torch.zeros((2, 7)))
    assert torch.equal(mask[:5, :5], torch.zeros((5, 5)))


def test_deterministic_masks_are_outcome_blind_unique_and_50_percent() -> None:
    first = deterministic_mask_indices(
        500,
        250,
        ["P-A", "P-B"],
        effective_seed=2026,
        epoch=3,
        logical_batch_index=4,
    )
    second = deterministic_mask_indices(
        500,
        250,
        ["P-A", "P-B"],
        effective_seed=2026,
        epoch=3,
        logical_batch_index=4,
    )
    changed = deterministic_mask_indices(
        500,
        250,
        ["P-A", "P-B"],
        effective_seed=2026,
        epoch=4,
        logical_batch_index=4,
    )
    assert first.shape == (2, 3, 250)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    for patient in range(2):
        for transition in range(3):
            assert first[patient, transition].unique().numel() == 250
    # The API has no outcome, FTV, mask, or target-value argument.
    import inspect

    names = tuple(inspect.signature(deterministic_mask_indices).parameters)
    assert all(
        forbidden not in "_".join(names).casefold()
        for forbidden in ("pcr", "ftv", "outcome", "target_value")
    )


def test_small_forward_shapes_condition_is_predictor_only_and_target_is_stopgrad() -> (
    None
):
    model = small_model().eval()
    image = torch.randn(
        (1, 4, 7, *SMALL_SHAPE), generator=torch.Generator().manual_seed(9)
    )
    first_condition = condition(1, arm=1)
    changed_condition = replace(
        condition(1, arm=13),
        clinical=torch.tensor([[0.0, 1.0, 0.0, -1.5, 1.0]]),
    )
    with torch.no_grad():
        first = model(
            image,
            first_condition,
            patient_ids=["patient"],
            mask_seed=2026,
        )
        changed = model(
            image,
            changed_condition,
            patient_ids=["patient"],
            mask_seed=2026,
        )
        direct_tokens = model.encode_tokens(image)
        direct_target = model.encode_target_tokens(image)
        direct_sigreg = model.encode_sigreg_state(image)
    assert first.online_tokens.shape == (1, 4, 8, 128)
    assert first.target_tokens.shape == (1, 4, 8, 128)
    assert first.target_masked.shape == (1, 3, 4, 128)
    assert first.predictions.shape == first.target_masked.shape
    assert first.mask_indices.shape == (1, 3, 4)
    assert first.sigreg_state.shape == (1, 4, 128)
    assert first.canonical_response.shape == (1, 4, 192)
    assert first.ftv_prediction.shape == (1, 4)
    assert torch.equal(first.online_tokens, direct_tokens)
    assert torch.equal(first.target_tokens, direct_target)
    assert torch.equal(first.sigreg_state, direct_sigreg)
    assert torch.equal(first.online_tokens, changed.online_tokens)
    assert torch.equal(first.target_tokens, changed.target_tokens)
    assert torch.equal(first.canonical_response, changed.canonical_response)
    assert not torch.equal(first.predictions, changed.predictions)
    assert not first.target_tokens.requires_grad
    assert not first.target_masked.requires_grad


def test_target_values_are_position_free_and_canonical_path_uses_raw_tokens() -> None:
    model = small_model().eval()
    image = torch.randn(
        (1, 1, 7, *SMALL_SHAPE), generator=torch.Generator().manual_seed(23)
    )
    with torch.no_grad():
        raw, projected = model._encode_with(
            image, model.encoder, model.token_projection
        )
        target = model.encode_target_tokens(image)
        canonical = model.encode_canonical_response(image)
    assert torch.equal(projected, target)
    assert not torch.equal(
        target[0, 0], target[0, 0] + model.physical_position_encoding
    )
    expected_raw_mean = (raw[0, 0] * model.local_weights[:, None]).sum(
        0
    ) / model.local_weights.sum()
    expected_response = model.response_projection(expected_raw_mean[None])[0]
    torch.testing.assert_close(canonical[0, 0], expected_response)
    # The 128->192 response does not consume projected token values.
    with torch.no_grad():
        model.token_projection[0].weight.zero_()
    torch.testing.assert_close(model.encode_canonical_response(image), canonical)


def test_gradient_boundaries_and_exact_ema_update() -> None:
    model = small_model().train()
    image = torch.randn(
        (1, 4, 7, *SMALL_SHAPE), generator=torch.Generator().manual_seed(31)
    )
    output = model(image, condition(1), patient_ids=["patient"], mask_seed=3026)
    output.predictions.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(
        parameter.grad is not None for parameter in model.token_projection.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert any(
        parameter.grad is not None for parameter in model.condition_encoder.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_token_projection.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.target_encoder.parameters()
    )

    online = next(model.token_projection.parameters())
    target = next(model.target_token_projection.parameters())
    old = target.detach().clone()
    with torch.no_grad():
        online.add_(0.4)
        expected = old * 0.75 + online * 0.25
    model.update_target(0.75)
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    assert not model.target_encoder.training
    assert not model.target_token_projection.training


def test_nominal_condition_validation_fails_closed() -> None:
    valid = condition(1)
    valid.validate(1, torch.device("cpu"))
    with pytest.raises(ValueError, match="delta_t"):
        replace(valid, delta_t=torch.tensor([[1.0, 2.0, 1.0]])).validate(
            1, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="temporal_bits"):
        replace(valid, temporal_bits=torch.zeros((1, 3, 7))).validate(
            1, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="age_missing"):
        replace(valid, clinical=torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.5]])).validate(
            1, torch.device("cpu")
        )


def test_formal_model_contract_without_full_mri_forward() -> None:
    model = PatchTokenWorldModel()
    contract = model.architecture_contract()
    assert model.token_count == 500
    assert model.masked_token_count == 250
    assert model.feature_shape_zyx == (14, 22, 20)
    assert contract["target_values_position_free"] is True
    assert contract["condition_target_path"] is False
    assert contract["condition_online_mri_token_path"] is False
    assert contract["condition_method"] == "one_condition_token_not_FiLM"
    assert contract["canonical_response"].startswith(
        "raw128_exact_fractional_LOCAL_mean"
    )
