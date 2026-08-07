"""当前 clean primary FLR 路径的结构性契约，不作为正式实验结果。"""

from __future__ import annotations

import unittest

import torch

from ispy_jepa_tmi_clean.corejepa.config import ModelConfig
from ispy_jepa_tmi_clean.corejepa.models import CoReJEPA
from shortcut_audit.auditlib.perturbations import repeated_t0_mri_only


class PrimaryReadoutArchitectureTest(unittest.TestCase):
    def test_c1_mri_replacement_cannot_change_forecast_response_state(self) -> None:
        """C1 保留 q/condition，因此当前 FLR 的输入 state 必须完全不变。"""

        torch.manual_seed(2026)
        config = ModelConfig(
            base_channels=2,
            latent_dim=16,
            predictor_depth=1,
            predictor_heads=4,
            predictor_mlp_dim=32,
            response_dim=8,
            response_hidden_dim=16,
            response_experts=2,
            expert_hidden_dim=8,
            expert_gate_hidden_dim=8,
            dropout=0.2,
        )
        model = CoReJEPA(config, condition_dim=11).eval()
        image = torch.randn(2, 4, 8, 2, 3, 2)
        geometry = torch.randn(2, 4, 9)
        condition = torch.randn(2, 3, 11)
        c1 = repeated_t0_mri_only(image, geometry, condition)

        self.assertFalse(torch.equal(c1.image[:, 1:3, :7], image[:, 1:3, :7]))
        with torch.no_grad():
            native_state = model.forecast_response(geometry, condition)
            c1_state = model.forecast_response(c1.geometry, c1.condition)

        torch.testing.assert_close(c1_state, native_state, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
