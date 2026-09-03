import inspect

import pytest
import torch

from dinov3_rg.contracts import ARMS, PILOT_ARMS, PILOT_WEIGHTS, SEEDS, expected_cells, load_protocol
from dinov3_rg.model import MRIAdapterWorldModel, WorldModelOutput
from dinov3_rg.objective import DirectRadiomicsObjective
from dinov3_rg.training import reinitialize_radiomics_head


def fake_output() -> WorldModelOutput:
    response = torch.randn(3, 4, 192, requires_grad=True)
    online = torch.randn(3, 4, 192, requires_grad=True)
    target = torch.randn(3, 4, 192)
    predicted = torch.randn(3, 3, 192, requires_grad=True)
    ftv = torch.randn(3, 4, requires_grad=True)
    radiomics = torch.randn(3, 4, 16, requires_grad=True)
    return WorldModelOutput(response, online, target, target[:, 1:], predicted, ftv, radiomics)


def test_v3_protocol_is_frozen_to_50_formal_cells():
    protocol = load_protocol()
    assert ARMS == ("C0", "RAD")
    assert PILOT_ARMS == ("R025", "R050", "R100")
    assert PILOT_WEIGHTS == {"R025": 0.25, "R050": 0.5, "R100": 1.0}
    assert SEEDS == (7026, 8026, 9026, 10026, 11026)
    assert len(expected_cells()) == 50
    assert protocol["loss"]["ftv"] == 0.0


def test_objective_has_no_ftv_gradient_and_rejects_t3():
    output = fake_output()
    objective = DirectRadiomicsObjective(0.5)
    ftv = torch.randn(3, 4)
    ftv_mask = torch.ones(3, 4, dtype=torch.bool)
    radiomics = torch.randn(3, 4, 16)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[:, 3] = False
    torch.manual_seed(4)
    loss, _ = objective(output, ftv, ftv_mask, radiomics, mask)
    loss.backward()
    assert objective.ftv_weight == 0.0
    assert output.ftv_prediction.grad is None
    assert output.radiomics_prediction.grad is not None
    assert output.radiomics_prediction.grad.abs().sum() > 0
    mask[:, 3] = True
    with pytest.raises(ValueError, match="T0-T2"):
        objective(fake_output(), ftv, ftv_mask, radiomics, mask)


def test_model_forward_api_and_head_initialization_are_deterministic():
    assert tuple(inspect.signature(MRIAdapterWorldModel.forward).parameters) == ("self", "slice_summaries")
    torch.manual_seed(9)
    first = MRIAdapterWorldModel(dropout=0.0)
    second = MRIAdapterWorldModel(dropout=0.0)
    second.load_state_dict(first.state_dict())
    assert reinitialize_radiomics_head(first, 902026) == reinitialize_radiomics_head(second, 902026)
    with pytest.raises(TypeError):
        first(torch.zeros(1, 4, 7, 32, 2304), torch.ones(1))
