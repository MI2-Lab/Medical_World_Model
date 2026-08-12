from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_sph.contracts import (  # noqa: E402
    ARMS,
    SPH_HEAD_PARAMETER_COUNT,
    TrainHyperparameters,
    arm_spec,
    validate_seed_fold,
)
from residual_sph.data import StaticSPHDataset  # noqa: E402
from residual_sph.losses import ResidualSPHObjective, patient_mean_static_loss  # noqa: E402
from residual_sph.model import (  # noqa: E402
    build_model,
    paired_initialization_report,
    shared_initialization_sha256,
    sph_head_sha256,
)
from residual_sph.training import (  # noqa: E402
    scale_microbatch_components,
    select_checkpoint,
    validate_s0_anchor,
)
from residual_sph.upstream import build_local_model  # noqa: E402


class _TinyBaseDataset:
    def __init__(self) -> None:
        self.patient_ids = ("complete-case", "jepa-only")
        self.transformed_ftv = {}

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "patient_id": self.patient_ids[index],
            "image": torch.zeros((4, 7, 2, 2, 2)),
            "ftv_target": torch.zeros(4),
            "ftv_mask": torch.zeros(4, dtype=torch.bool),
        }


def test_arm_matrix_and_hyperparameters_are_exact() -> None:
    assert ARMS == ("S0", "S1", "S2", "S2_L10")
    assert [(arm_spec(arm).sph_target, arm_spec(arm).lambda_sph) for arm in ARMS] == [
        (None, 0.0),
        ("raw_sph_z", 0.05),
        ("ftv_residual_sph_z", 0.05),
        ("ftv_residual_sph_z", 0.10),
    ]
    assert validate_seed_fold(2026, 4) == 2030
    TrainHyperparameters().validate()
    with pytest.raises(ValueError, match="physical"):
        TrainHyperparameters(physical_batch_size=8, accumulation_steps=4).validate()


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (2026, "e8cf59a6a2c0830359c8eaacdaf69447645e8c425bb5e97060fa5fd1146d64dd"),
        (3026, "6968468dc787eeb6f71cbbda36d0d75c17654ac76ff8f8a867cd083f79f03ab8"),
    ],
)
def test_shared_initialization_matches_confirmation(seed: int, expected: str) -> None:
    report = paired_initialization_report(seed)
    assert report["shared_initialization_sha256"] == expected
    assert all(report["checks"].values())


def test_s0_is_state_dict_identical_and_sph_adds_only_193_parameters() -> None:
    s0 = build_model("S0", 2026)
    local3 = build_local_model("LOCAL3", 2026)
    assert set(s0.state_dict()) == set(local3.state_dict())
    assert all(torch.equal(s0.state_dict()[key], value) for key, value in local3.state_dict().items())
    s1 = build_model("S1", 2026)
    s2 = build_model("S2", 2026)
    assert sph_head_sha256(s1) == sph_head_sha256(s2)
    assert sum(parameter.numel() for parameter in s1.sph_head.parameters()) == SPH_HEAD_PARAMETER_COUNT
    assert sum(parameter.numel() for parameter in s1.parameters()) - sum(parameter.numel() for parameter in s0.parameters()) == SPH_HEAD_PARAMETER_COUNT


def test_model_build_does_not_advance_public_rng() -> None:
    torch.manual_seed(77)
    expected = torch.rand(5)
    torch.manual_seed(77)
    build_model("S2", 2026)
    observed = torch.rand(5)
    assert torch.equal(observed, expected)


def test_ema_update_never_changes_sph_head() -> None:
    model = build_model("S2", 2026)
    before = {name: value.clone() for name, value in model.sph_head.state_dict().items()}
    model.update_target(0.996)
    assert all(torch.equal(before[name], value) for name, value in model.sph_head.state_dict().items())


def test_patient_mean_static_huber_is_patient_then_visit_mean() -> None:
    prediction = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, False, False, False], [True, True, False, False]])
    loss, patients, visits = patient_mean_static_loss(prediction, target, valid, prediction)
    # SmoothL1: patient 1 -> 0; patient 2 -> mean(0.5,1.5)=1; patient mean=.5.
    assert loss.item() == pytest.approx(0.5)
    assert patients.item() == 2
    assert visits.item() == 3


def test_broader_jepa_patient_has_no_imputed_sph_supervision() -> None:
    mapping = {
        "complete-case": (
            np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            np.ones(4, dtype=bool),
        )
    }
    wrapped = StaticSPHDataset(_TinyBaseDataset(), mapping, "S2")
    assert wrapped[0]["sph_mask"].all()
    assert not wrapped[1]["sph_mask"].any()
    assert torch.equal(wrapped[1]["sph_target"], torch.zeros(4))


def test_two_auxiliary_logical_scaling_matches_direct_patient_means() -> None:
    base = [torch.tensor(2.0, requires_grad=True), torch.tensor(6.0, requires_grad=True)]
    ftv = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    sph = [torch.tensor(4.0, requires_grad=True), torch.tensor(8.0, requires_grad=True)]
    loss = sum(
        scale_microbatch_components(
            base[index], ftv[index], sph[index],
            microbatch_size=16, logical_batch_size=32,
            microbatch_ftv_patients=(4, 12)[index], logical_ftv_patients=16,
            microbatch_sph_patients=(8, 8)[index], logical_sph_patients=16,
            lambda_ftv=0.25, lambda_sph=0.05,
        )
        for index in range(2)
    )
    expected = 0.5 * 2 + 0.5 * 6 + 0.25 * (0.25 * 1 + 0.75 * 3) + 0.05 * (0.5 * 4 + 0.5 * 8)
    assert loss.item() == pytest.approx(expected)


def _epochs() -> list[dict[str, object]]:
    return [
        {
            "epoch": 1, "val_state_loss": 1.00, "val_ftv_loss": 0.4,
            "val_sph_loss": 10.0, "val_representation_std": 0.2,
            "val_grounded_patients": 20, "val_loss": 1.5, "val_base_objective": 1.1,
        },
        {
            "epoch": 2, "val_state_loss": 1.01, "val_ftv_loss": 0.3,
            "val_sph_loss": 20.0, "val_representation_std": 0.2,
            "val_grounded_patients": 20, "val_loss": 1.5, "val_base_objective": 1.1,
        },
    ]


def test_selector_is_invariant_to_sph_and_unregistered_fields() -> None:
    first = select_checkpoint(_epochs(), min_representation_std=0.05, paired_s0_state_loss=1.0)
    changed = copy.deepcopy(_epochs())
    changed[0]["val_sph_loss"] = -1e9
    changed[0]["pcr"] = 1.0
    changed[0]["test_sph"] = 1.0
    second = select_checkpoint(changed, min_representation_std=0.05, paired_s0_state_loss=1.0)
    assert first["selected_epoch"] == second["selected_epoch"] == 2
    assert "validation_sph_loss" in first["selection_excludes"]


def test_s0_anchor_is_identity_and_hash_bound(tmp_path: Path) -> None:
    model = build_model("S0", 2026)
    payload = {
        "arm": "LOCAL3", "seed_base": 2026, "fold": 0, "effective_seed": 2026,
        "paired_initialization_sha256": shared_initialization_sha256(model),
        "test_data_used": False, "pcr_used": False, "delta_ftv_used": False,
        "selection_mode": "primary", "experiment_pass": True,
        "selected_validation_state_loss": 0.4,
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    state, observed = validate_s0_anchor(
        path,
        seed_base=2026,
        fold=0,
        expected_shared_initialization_sha256=shared_initialization_sha256(model),
    )
    assert state == 0.4 and observed == payload
    payload["pcr_used"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pcr_used"):
        validate_s0_anchor(
            path,
            seed_base=2026,
            fold=0,
            expected_shared_initialization_sha256=shared_initialization_sha256(model),
        )
