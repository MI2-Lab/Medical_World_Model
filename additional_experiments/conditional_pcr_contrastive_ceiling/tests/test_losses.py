from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from conditional_ceiling.losses import (
    conditional_ceiling_loss,
    conditional_supervised_contrastive_loss,
)


def _manual_anchor_loss(
    embeddings: torch.Tensor,
    anchor: int,
    denominator: list[int],
    positives: list[int],
    temperature: float,
) -> torch.Tensor:
    normalized = functional.normalize(embeddings, dim=1)
    similarity = normalized @ normalized.T / temperature
    log_denominator = torch.logsumexp(similarity[anchor, denominator], dim=0)
    return -torch.stack(
        [similarity[anchor, positive] - log_denominator for positive in positives]
    ).mean()


def test_conditional_supcon_matches_manual_within_stratum_formula() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-0.8, -0.2],
            [0.0, -1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 0, 0, 1])
    strata = torch.tensor([0, 0, 0, 1, 1, 1])
    eligible = torch.tensor([True, True, False, True, True, False])
    observed = conditional_supervised_contrastive_loss(
        embeddings,
        labels,
        strata,
        eligible_anchor=eligible,
        temperature=0.2,
    )
    expected = torch.stack(
        [
            _manual_anchor_loss(embeddings, 0, [1, 2], [1], 0.2),
            _manual_anchor_loss(embeddings, 1, [0, 2], [0], 0.2),
            _manual_anchor_loss(embeddings, 3, [4, 5], [4], 0.2),
            _manual_anchor_loss(embeddings, 4, [3, 5], [3], 0.2),
        ]
    ).mean()
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    observed.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_cross_stratum_samples_do_not_change_loss_or_gradient() -> None:
    first = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    second_a = torch.tensor(
        [[-1.0, 0.0], [-0.8, -0.2], [0.0, -1.0]], dtype=torch.float64
    )
    second_b = torch.tensor(
        [[100.0, 3.0], [-70.0, 90.0], [5.0, -200.0]], dtype=torch.float64
    )

    def evaluate(context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_state = first.detach().clone().requires_grad_(True)
        joined = torch.cat((anchor_state, context), dim=0)
        value = conditional_supervised_contrastive_loss(
            joined,
            torch.tensor([0, 0, 1, 0, 0, 1]),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            eligible_anchor=torch.tensor([True, True, False, False, False, False]),
            temperature=0.1,
        )
        gradient = torch.autograd.grad(value, anchor_state)[0]
        return value, gradient

    value_a, gradient_a = evaluate(second_a)
    value_b, gradient_b = evaluate(second_b)
    torch.testing.assert_close(value_a, value_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradient_a, gradient_b, rtol=0.0, atol=0.0)


def test_b1_is_supcon_only_and_b2_b3_add_exact_quarter_bce() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 1, 1])
    strata = torch.zeros(4, dtype=torch.long)
    eligible = torch.ones(4, dtype=torch.bool)
    b1 = conditional_ceiling_loss(
        embeddings, labels, strata, eligible, arm="B1"
    )
    assert b1.bce_weight == 0.0
    assert b1.pcr_bce.item() == 0.0
    torch.testing.assert_close(b1.total, b1.conditional_supcon)
    logits = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float64)
    expected_bce = functional.binary_cross_entropy_with_logits(logits, labels.double())
    for arm in ("B2", "B3"):
        output = conditional_ceiling_loss(
            embeddings, labels, strata, eligible, arm=arm, logits=logits
        )
        assert output.bce_weight == 0.25
        torch.testing.assert_close(output.pcr_bce, expected_bce)
        torch.testing.assert_close(
            output.total, output.conditional_supcon + 0.25 * expected_bce
        )
    with pytest.raises(ValueError, match="SupCon-only"):
        conditional_ceiling_loss(
            embeddings, labels, strata, eligible, arm="B1", logits=logits
        )

def test_loss_rejects_ineligible_batch_and_invalid_temperature() -> None:
    embeddings = torch.eye(3)
    labels = torch.tensor([0, 1, 0])
    strata = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError, match="no eligible"):
        conditional_supervised_contrastive_loss(embeddings, labels, strata)
    with pytest.raises(ValueError, match="temperature"):
        conditional_supervised_contrastive_loss(
            torch.eye(4),
            torch.tensor([0, 0, 1, 1]),
            torch.zeros(4, dtype=torch.long),
            temperature=math.nan,
        )
