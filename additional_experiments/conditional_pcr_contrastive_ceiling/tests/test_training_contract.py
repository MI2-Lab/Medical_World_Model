from __future__ import annotations

from types import SimpleNamespace

import torch

from conditional_ceiling.losses import conditional_supervised_contrastive_loss
from conditional_ceiling.training import (
    TrainingHyperparameters,
    ceiling_loss,
    train_adaptation_cell,
)


def test_registered_supcon_averages_literal_observed_prefixes() -> None:
    torch.manual_seed(17)
    response = torch.randn(4, 4, 192, dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1])
    strata = torch.zeros(4, dtype=torch.long)
    eligible = torch.ones(4, dtype=torch.bool)
    model = SimpleNamespace(projection_head=torch.nn.Identity())

    observed, _ = ceiling_loss(
        model, response, labels, strata, eligible, arm="B1", temperature=0.1
    )
    expected = torch.stack(
        [
            conditional_supervised_contrastive_loss(
                response[:, :end].reshape(4, end * 192),
                labels,
                strata,
                eligible,
                temperature=0.1,
            )
            for end in range(1, 4)
        ]
    ).mean()
    torch.testing.assert_close(observed, expected)

    changed_t3 = response.clone()
    changed_t3[:, 3] += 10_000
    unchanged, _ = ceiling_loss(
        model, changed_t3, labels, strata, eligible, arm="B1", temperature=0.1
    )
    torch.testing.assert_close(observed, unchanged)


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 192)


class _TinyCeiling(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.projection_head = torch.nn.Linear(192, 64)
        self.pcr_heads = torch.nn.ModuleList(
            [torch.nn.Linear(64 * end, 1) for end in range(1, 4)]
        )

    def encode_response(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone.encoder(image)


class _OneBatchLoader:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        self.batch_sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    def __iter__(self):
        yield self.batch


def test_b3_checkpointed_encoder_microbatch_path_is_finite() -> None:
    torch.manual_seed(23)
    labels = torch.tensor([0, 0, 1, 1])
    batch = {
        "image": torch.randn(4, 4, 2),
        "label": labels,
        "stratum_id": torch.zeros(4, dtype=torch.long),
        "eligible_anchor": torch.ones(4, dtype=torch.bool),
    }
    model = _TinyCeiling()
    selection = train_adaptation_cell(
        model,
        _OneBatchLoader(batch),
        _OneBatchLoader(batch),
        arm="B3",
        hyperparameters=TrainingHyperparameters(
            epochs=1,
            patience=1,
            encoder_learning_rate=1e-4,
            head_learning_rate=1e-3,
            bce_weight=0.25,
        ),
        device=torch.device("cpu"),
        microbatch_size=1,
    )
    assert selection["selected_epoch"] == 1
    assert len(selection["history"]) == 1
    assert torch.isfinite(
        torch.tensor(selection["history"][0]["train_loss"])
    )
