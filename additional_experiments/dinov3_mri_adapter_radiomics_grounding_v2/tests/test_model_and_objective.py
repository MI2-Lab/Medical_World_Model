import inspect

import pytest
import torch

from dinov3_rg.model import MRIAdapterWorldModel, initialization_sha256
from dinov3_rg.objective import GroundedJEPAObjective


def test_forward_contract_and_shapes():
    signature = inspect.signature(MRIAdapterWorldModel.forward)
    assert tuple(signature.parameters) == ("self", "slice_summaries")
    model = MRIAdapterWorldModel(dropout=0.0)
    values = torch.randn(1, 4, 7, 32, 2304)
    output = model(values)
    assert output.response_state.shape == (1, 4, 192)
    assert output.predicted_next.shape == (1, 3, 192)
    assert output.ftv_prediction.shape == (1, 4)
    assert output.radiomics_prediction.shape == (1, 4, 16)
    with pytest.raises(TypeError):
        model(values, torch.ones(1))


def test_paired_initialization_and_arm_structure():
    hashes = []
    states = []
    for _ in range(3):
        torch.manual_seed(2026)
        model = MRIAdapterWorldModel()
        hashes.append(initialization_sha256(model))
        states.append(tuple(model.state_dict()))
    assert len(set(hashes)) == 1
    assert states[0] == states[1] == states[2]


def test_radiomics_loss_reaches_adapter_and_head_only_in_d3():
    torch.manual_seed(5)
    model = MRIAdapterWorldModel(dropout=0.0)
    output = model(torch.randn(2, 4, 7, 32, 2304))
    ftv = torch.randn(2, 4)
    ftv_mask = torch.ones(2, 4, dtype=torch.bool)
    radiomics = torch.randn(2, 4, 16)
    radiomics_mask = torch.ones(2, 4, dtype=torch.bool)
    radiomics_mask[:, 3] = False
    objective = GroundedJEPAObjective("D3")
    loss, _ = objective(output, ftv, ftv_mask, radiomics, radiomics_mask)
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in model.adapter.parameters() if p.grad is not None) > 0
    assert sum(p.grad.abs().sum().item() for p in model.radiomics_head.parameters() if p.grad is not None) > 0
    assert GroundedJEPAObjective("D2").weights.radiomics == 0.0


def test_t3_radiomics_grounding_is_rejected():
    model = MRIAdapterWorldModel(dropout=0.0)
    output = model(torch.randn(1, 4, 7, 32, 2304))
    with pytest.raises(ValueError, match="T0-T2"):
        GroundedJEPAObjective("D3")(
            output,
            torch.randn(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            torch.randn(1, 4, 16),
            torch.ones(1, 4, dtype=torch.bool),
        )


def test_target_modules_are_frozen():
    model = MRIAdapterWorldModel()
    assert not any(parameter.requires_grad for parameter in model.target_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in model.target_projector.parameters())
