from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from patch_token_wm.training import (
    LOGICAL_BATCH_SIZE,
    TrainHyperparameters,
    _loader,
    logical_patient_batches,
    logical_sigreg_surrogate,
    physical_patient_batches,
    scale_logical_components,
    select_checkpoint,
)


class _TinyDataset(torch.utils.data.Dataset):
    patient_ids = ("p0",)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> int:
        return index


def test_locked_batch_contract_and_deterministic_shared_tail() -> None:
    identities = [f"p{index:03d}" for index in range(683)]
    first = logical_patient_batches(identities, effective_seed=2026, epoch=1)
    same = logical_patient_batches(identities, effective_seed=2026, epoch=1)
    different = logical_patient_batches(identities, effective_seed=2026, epoch=2)
    assert first == same
    assert first != different
    assert len(first) == 21
    assert all(len(batch) == LOGICAL_BATCH_SIZE for batch in first)
    flattened = [patient for batch in first for patient in batch]
    assert len(flattened) == 672
    assert len(set(flattened)) == 672
    physical = physical_patient_batches(first[:1], 4)
    assert len(physical) == 8
    assert tuple(patient for batch in physical for patient in batch) == first[0]


def test_hyperparameters_fail_closed_on_nonlogical_batch() -> None:
    TrainHyperparameters().validate()
    with pytest.raises(ValueError, match="equal 32"):
        TrainHyperparameters(physical_batch_size=4, accumulation_steps=4).validate()


def test_worker_loader_uses_spawn_after_cuda_initialization() -> None:
    loader = _loader(_TinyDataset(), (("p0",),), workers=1)
    assert loader.multiprocessing_context is not None
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert next(iter(loader)).tolist() == [0]


def test_exact_patient_mean_scaling() -> None:
    pieces = []
    counts = (1, 0, 3, 2, 4, 1, 2, 3)
    for index, count in enumerate(counts):
        pieces.append(
            scale_logical_components(
                torch.tensor(float(index + 1)),
                torch.tensor(float(10 + index)),
                microbatch_size=4,
                logical_batch_size=32,
                microbatch_ftv_patients=count,
                logical_ftv_patients=sum(counts),
                lambda_ftv=0.25,
            )
        )
    observed = float(torch.stack(pieces).sum())
    expected_base = float(np.mean(np.arange(1, 9)))
    expected_ftv = sum(
        (10 + index) * count for index, count in enumerate(counts)
    ) / sum(counts)
    assert observed == pytest.approx(expected_base + 0.25 * expected_ftv)


def test_sigreg_surrogate_has_exact_reference_gradient_after_accumulation() -> None:
    reference = torch.randn(32, 4, 6)
    gradient = torch.randn_like(reference)
    logical_value = torch.tensor(1.25)
    current_parts = [
        part.detach().clone().requires_grad_(True) for part in reference.chunk(8)
    ]
    scaled = []
    for index, current in enumerate(current_parts):
        surrogate = logical_sigreg_surrogate(
            current,
            reference[index * 4 : (index + 1) * 4],
            gradient[index * 4 : (index + 1) * 4],
            logical_value,
            logical_batch_size=32,
        )
        scaled.append(surrogate * (4.0 / 32.0))
    total = torch.stack(scaled).sum()
    total.backward()
    assert float(total) == pytest.approx(float(logical_value))
    for index, current in enumerate(current_parts):
        torch.testing.assert_close(current.grad, gradient[index * 4 : (index + 1) * 4])


def _row(epoch: int, patch: float, ftv: float, std: float = 0.2) -> dict[str, object]:
    return {
        "epoch": epoch,
        "val_patch_loss": patch,
        "val_ftv_loss": ftv,
        "val_representation_std": std,
        "finite": all(math.isfinite(value) for value in (patch, ftv, std)),
    }


def test_selection_uses_patch_then_ftv_then_earlier_and_rejects_collapse() -> None:
    selected = select_checkpoint(
        (
            _row(1, 0.8, 0.4),
            _row(2, 0.7, 0.3, std=0.01),
            _row(3, 0.6, 0.5),
            _row(4, 0.6 + 5e-13, 0.4),
            _row(5, 0.6 + 5e-13, 0.4),
        ),
        min_representation_std=0.05,
    )
    assert selected["selected_epoch"] == 4
    assert selected["test_data_used"] is False
    assert selected["pcr_loaded"] is False
    assert selected["epochs"][1]["checkpoint_eligible"] is False


def test_selection_refuses_all_collapsed() -> None:
    with pytest.raises(RuntimeError, match="no finite non-collapsed"):
        select_checkpoint((_row(1, 0.5, 0.2, std=0.01),), min_representation_std=0.05)
