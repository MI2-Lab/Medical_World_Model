from __future__ import annotations

import pytest
import torch

from patch_token_wm.model import PatchTokenOutput
from patch_token_wm.objective import (
    PatchTokenObjective,
    normalized_masked_token_mse,
    patient_mean_ftv_smooth_l1,
)


def output_fixture(batch: int = 2, masked: int = 4) -> PatchTokenOutput:
    generator = torch.Generator().manual_seed(5)
    prediction = torch.randn(
        (batch, 3, masked, 128), generator=generator, requires_grad=True
    )
    target = torch.randn((batch, 3, masked, 128), generator=generator)
    sigreg = torch.randn((batch, 4, 128), generator=generator, requires_grad=True)
    response = torch.randn((batch, 4, 192), generator=generator, requires_grad=True)
    ftv = response[..., 0]
    return PatchTokenOutput(
        online_tokens=torch.empty((batch, 4, 8, 128)),
        target_tokens=torch.empty((batch, 4, 8, 128)),
        target_masked=target,
        predictions=prediction,
        mask_indices=torch.arange(masked).reshape(1, 1, masked).expand(batch, 3, -1),
        sigreg_state=sigreg,
        canonical_response=response,
        ftv_prediction=ftv,
        local_coordinates_xyz_mm=torch.empty((8, 3)),
        local_weights=torch.ones(8),
    )


def test_normalized_masked_mse_has_locked_step_weight_semantics() -> None:
    prediction = torch.zeros((1, 3, 2, 128))
    target = torch.zeros_like(prediction)
    ramp = torch.arange(128, dtype=torch.float32)
    target[:, 0] = ramp
    target[:, 1] = 2 * ramp  # LayerNorm makes this the same normalized target.
    loss, per_step = normalized_masked_token_mse(
        prediction, target, torch.tensor((2.0, 1.0, 0.5))
    )
    assert per_step[0] == pytest.approx(per_step[1])
    assert per_step[2] == 0
    expected = (2 * per_step[0] + per_step[1]) / 3.5
    torch.testing.assert_close(loss, expected)


def test_patient_mean_ftv_is_visit_mean_then_patient_mean() -> None:
    prediction = torch.tensor([[0.0, 2.0, 99.0, 99.0], [1.0, 1.0, 1.0, 1.0]])
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    loss, patients, visits = patient_mean_ftv_smooth_l1(
        prediction, target, valid, prediction
    )
    # SmoothL1(0)=0, SmoothL1(2)=1.5 => patient 1=.75;
    # SmoothL1(1)=.5 => patient 2=.5; patient mean=.625.
    assert float(loss) == pytest.approx(0.625)
    assert int(patients) == 2
    assert int(visits) == 3


def test_full_objective_weights_components_counts_and_gradients() -> None:
    output = output_fixture()
    objective = PatchTokenObjective()
    override = output.sigreg_state.square().mean()
    ftv_target = torch.zeros((2, 4))
    ftv_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    total, stats = objective(
        output,
        ftv_target,
        ftv_mask,
        sigreg_override=override,
    )
    expected = (
        stats["_patch_component"]
        + 0.09 * stats["_sigreg_component"]
        + 0.25 * stats["_ftv_component_raw"]
    )
    torch.testing.assert_close(total, expected)
    assert objective.lambda_ftv == 0.25
    assert objective.sigreg_weight == 0.09
    assert objective.sigreg.projections == 256
    assert torch.equal(objective.step_weights, torch.tensor((2.0, 1.0, 0.5)))
    assert int(stats["patients"]) == 2
    assert int(stats["transitions"]) == 6
    assert int(stats["masked_tokens"]) == 24
    assert int(stats["ftv_patients"]) == 2
    assert int(stats["ftv_valid_visits"]) == 6
    assert int(stats["sigreg_override_used"]) == 1
    total.backward()
    assert output.predictions.grad is not None
    assert output.sigreg_state.grad is not None
    assert output.canonical_response.grad is not None
    assert output.target_masked.grad is None


def test_sigreg_default_and_empty_ftv_remain_finite_differentiable() -> None:
    output = output_fixture(batch=3)
    objective = PatchTokenObjective()
    total, stats = objective(
        output,
        torch.full((3, 4), float("nan")),
        torch.zeros((3, 4), dtype=torch.bool),
    )
    assert torch.isfinite(total)
    assert torch.isfinite(stats["sigreg_loss"])
    assert float(stats["ftv_loss"]) == 0
    assert int(stats["ftv_patients"]) == 0
    total.backward()
    assert output.predictions.grad is not None
    assert output.sigreg_state.grad is not None


def test_objective_hyperparameters_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="lambda_ftv"):
        PatchTokenObjective(lambda_ftv=0.1)
    with pytest.raises(ValueError, match="SIGReg weight"):
        PatchTokenObjective(sigreg_weight=0.1)
    with pytest.raises(ValueError, match="transition weights"):
        PatchTokenObjective(step_weights=(1.0, 1.0, 1.0))
