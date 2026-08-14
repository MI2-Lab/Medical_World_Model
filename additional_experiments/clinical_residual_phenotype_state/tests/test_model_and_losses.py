from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from crps.losses import (
    FactorizedObjective,
    cross_covariance_penalty,
    logical_loss_surrogate,
    vicreg_consistency,
)
from crps.model import (
    FactorizedOutput,
    FactorizedPhenotypeWorldModel,
    SingleQueryLocalPool,
    build_model,
    gradient_reverse,
)
from crps.training import TrainHyperparameters, select_checkpoint


def test_single_query_is_strictly_restricted_to_local_support() -> None:
    torch.manual_seed(4)
    pool = SingleQueryLocalPool(4, 8, heads=2, mlp_dim=16, dropout=0.0).eval()
    spatial = torch.randn(3, 4, 2, 2, 2)
    weights = torch.zeros(1, 1, 2, 2, 2)
    weights[..., 0, :, :] = 1.0
    first, attention = pool(spatial, weights, return_attention=True)
    changed = spatial.clone()
    changed[..., 1, :, :] += 1000.0
    second, changed_attention = pool(changed, weights, return_attention=True)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
    assert torch.allclose(attention, changed_attention, atol=1e-7, rtol=1e-7)
    flattened_weights = weights.flatten()
    assert torch.equal(
        attention[..., flattened_weights.eq(0)],
        torch.zeros_like(attention[..., flattened_weights.eq(0)]),
    )
    assert torch.allclose(attention.sum(dim=-1), torch.ones(3, 2))


def test_single_query_fractional_overlap_changes_attention_prior() -> None:
    pool = SingleQueryLocalPool(2, 4, heads=1, mlp_dim=8, dropout=0.0).eval()
    with torch.no_grad():
        pool.key.weight.zero_()
    spatial = torch.ones(1, 2, 1, 1, 2)
    weights = torch.tensor([[[[[1.0, 0.25]]]]])
    _, attention = pool(spatial, weights, return_attention=True)
    assert torch.allclose(attention[0, 0], torch.tensor([0.8, 0.2]), atol=1e-6)


def test_single_query_has_no_local_mean_bypass() -> None:
    pool = SingleQueryLocalPool(4, 4, heads=1, mlp_dim=8, dropout=0.0).eval()
    with torch.no_grad():
        pool.key.weight.zero_()
        pool.value.weight.zero_()
        for stack in (pool.output, pool.mlp):
            for module in stack:
                if isinstance(module, torch.nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
    weights = torch.ones(1, 1, 2, 2, 2)
    first = torch.randn(2, 4, 2, 2, 2)
    second = first + 50.0
    assert torch.allclose(pool(first, weights), pool(second, weights), atol=1e-7)


def test_gradient_reversal_changes_only_backward_sign() -> None:
    value = torch.tensor([1.0, -2.0], requires_grad=True)
    gradient_reverse(value, 0.25).sum().backward()
    assert torch.equal(value.grad, torch.full_like(value, -0.25))
    assert torch.equal(gradient_reverse(value.detach(), 0.5), value.detach())


def test_checkpoint_selection_uses_shared_loss_not_adversary_ce() -> None:
    base = {
        "finite": True,
        "val_response_std": 0.2,
        "val_phenotype_std": 0.2,
        "val_phenotype_effective_rank": 20.0,
        "val_augmentation_cosine": 0.8,
    }
    history = [
        {
            **base,
            "epoch": 1,
            "val_loss": 1.0,
            "val_selection_loss": 0.9,
        },
        {
            **base,
            "epoch": 2,
            "val_loss": 0.8,
            "val_selection_loss": 0.95,
        },
    ]
    selected = select_checkpoint(history, TrainHyperparameters())
    assert selected["selected_epoch"] == 1
    assert selected["selected_validation_selection_loss"] == pytest.approx(0.9)


def test_factorized_architecture_dimensions_and_adversary_scope() -> None:
    f1 = FactorizedPhenotypeWorldModel("F1", condition_dim=24)
    f2 = FactorizedPhenotypeWorldModel("F2", condition_dim=24)
    assert f1.response_projection[0].out_features == 96
    assert f1.phenotype_pool.output_dim == 96
    assert f1.ftv_head.in_features == 96
    assert f1.hr_adversary is None and f1.her2_adversary is None
    assert f2.hr_adversary is not None and f2.her2_adversary is not None
    contract = f2.architecture_contract()
    assert contract["state_dim"] == 192
    assert contract["ftv_head_input"] == "response_state_only"
    assert contract["phenotype_query_conditioned"] is False
    assert contract["treatment_adversarially_removed"] is False
    assert tuple(f2.phenotype_pool.query.shape)[2] == 1


def test_paired_f1_f2_common_initialization_is_exact() -> None:
    f1 = build_model("F1", 24, 2026)
    f2 = build_model("F2", 24, 2026)
    first = f1.state_dict()
    second = f2.state_dict()
    common = sorted(set(first) & set(second))
    assert common
    assert all(torch.equal(first[name], second[name]) for name in common)


def test_ema_updates_both_factorized_branches_but_not_transition() -> None:
    model = FactorizedPhenotypeWorldModel("F1", 24)
    target_before = next(model.target_phenotype_pool.parameters()).detach().clone()
    transition_before = next(model.response_transition.parameters()).detach().clone()
    with torch.no_grad():
        next(model.phenotype_pool.parameters()).add_(1.0)
        next(model.response_transition.parameters()).add_(1.0)
    model.update_target(0.5)
    target_after = next(model.target_phenotype_pool.parameters()).detach()
    assert not torch.equal(target_before, target_after)
    assert torch.equal(
        next(model.response_transition.parameters()).detach(), transition_before + 1.0
    )


def test_crosscov_zero_for_constant_or_constructed_uncorrelated_branch() -> None:
    response = torch.randn(8, 4, 96)
    phenotype = torch.ones(8, 4, 96)
    assert cross_covariance_penalty(response, phenotype) == pytest.approx(0.0)


def test_crosscov_uses_dimension_normalized_frobenius_scale() -> None:
    torch.manual_seed(17)
    response = torch.randn(32, 4, 96)
    phenotype = torch.randn(32, 4, 96)
    x = response.reshape(-1, 96)
    y = phenotype.reshape(-1, 96)
    cross = (x - x.mean(0)).T @ (y - y.mean(0)) / float(x.size(0) - 1)
    expected = cross.square().sum() / 96.0
    assert cross_covariance_penalty(response, phenotype) == pytest.approx(
        expected.item(), rel=1e-6
    )


def test_vicreg_penalizes_collapse_and_rewards_same_view_consistency() -> None:
    collapsed = torch.zeros(8, 4, 96)
    collapsed_loss, collapsed_stats = vicreg_consistency(collapsed, collapsed)
    diverse = torch.randn(64, 4, 96)
    diverse_loss, diverse_stats = vicreg_consistency(diverse, diverse)
    assert collapsed_stats["phenotype_variance_loss"] > diverse_stats["phenotype_variance_loss"]
    assert diverse_stats["phenotype_invariance_loss"] == pytest.approx(0.0)
    assert torch.isfinite(collapsed_loss) and torch.isfinite(diverse_loss)


def test_logical_surrogate_recovers_exact_reference_gradient() -> None:
    reference = {"x": torch.randn(6, 3)}
    gradient = {"x": 2.0 * reference["x"]}
    logical = reference["x"].square().sum()
    accumulated = torch.zeros_like(reference["x"])
    for start in (0, 2, 4):
        current = reference["x"][start : start + 2].clone().requires_grad_(True)
        surrogate = logical_loss_surrogate(
            {"x": current},
            {"x": reference["x"][start : start + 2]},
            {"x": gradient["x"][start : start + 2]},
            logical,
            logical_batch_size=6,
        )
        weighted = surrogate * (2.0 / 6.0)
        weighted.backward()
        accumulated[start : start + 2] = current.grad
    assert torch.allclose(accumulated, gradient["x"])


def _synthetic_output(batch: int = 4) -> FactorizedOutput:
    def state(visits: int, width: int) -> torch.Tensor:
        return torch.randn(batch, visits, width, requires_grad=True)

    response = state(4, 96)
    phenotype = state(4, 96)
    return FactorizedOutput(
        response_state=response,
        phenotype_state=phenotype,
        full_state=torch.cat((response, phenotype), dim=-1),
        response_online=state(4, 96),
        phenotype_online=state(4, 96),
        target_response_state=state(4, 96).detach(),
        target_phenotype_state=state(4, 96).detach(),
        target_response_online=state(4, 96).detach(),
        target_phenotype_online=state(4, 96).detach(),
        predicted_response_next=state(3, 96),
        predicted_phenotype_next=state(3, 96),
        ftv_prediction=torch.randn(batch, 4, requires_grad=True),
        adversary_hr_logits=torch.randn(batch, 4, 2, requires_grad=True),
        adversary_her2_logits=torch.randn(batch, 4, 2, requires_grad=True),
        augmented_phenotype_state=state(4, 96),
    )


def test_factorized_objective_keeps_ftv_and_adversary_separate() -> None:
    output = _synthetic_output()
    objective = FactorizedObjective("F2", sigreg_projections=8)
    target = torch.randn(4, 4)
    mask = torch.ones(4, 4, dtype=torch.bool)
    clinical = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    total, stats = objective.additive_components(output, target, mask, clinical)
    assert total.requires_grad
    assert float(stats["weighted_ftv_loss"].detach()) == pytest.approx(
        0.25 * float(stats["ftv_loss"].detach())
    )
    assert float(stats["weighted_adversary_loss"].detach()) == pytest.approx(
        0.05 * float(stats["adversary_loss"].detach())
    )
    logical, logical_stats = objective.logical_regularizers(
        response_state=output.response_state,
        phenotype_state=output.phenotype_state,
        augmented_phenotype_state=output.augmented_phenotype_state,
        response_online=output.response_online,
        sigreg_seed=7,
    )
    assert logical.requires_grad
    assert set(logical_stats) >= {
        "sigreg_loss",
        "phenotype_consistency_loss",
        "crosscov_loss",
    }


def test_factorized_output_schema_is_explicit() -> None:
    assert {field.name for field in fields(FactorizedOutput)} == {
        "response_state",
        "phenotype_state",
        "full_state",
        "response_online",
        "phenotype_online",
        "target_response_state",
        "target_phenotype_state",
        "target_response_online",
        "target_phenotype_online",
        "predicted_response_next",
        "predicted_phenotype_next",
        "ftv_prediction",
        "adversary_hr_logits",
        "adversary_her2_logits",
        "augmented_phenotype_state",
    }
