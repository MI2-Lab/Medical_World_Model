from __future__ import annotations

import unittest

import numpy as np
import torch

from shortcut_audit.auditlib.inference import (
    collect_frozen_inference,
    copy_current_latent_audit,
    paired_perturbation_latent_audit,
)
from shortcut_audit.auditlib.perturbations import repeated_t0_mri_only, swap_t1_t2


class _Response:
    def __init__(self, state: torch.Tensor, correction: torch.Tensor):
        self.future_state = state
        self.latent_correction = correction


class _Output:
    pass


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def encode_visits(self, image, geometry):
        return torch.stack((image[:, :, 0, 0, 0, 0], geometry[:, :, 0]), dim=-1)

    def encoder(self, image):
        return torch.stack((image[:, 0, 0, 0, 0], image[:, 7, 0, 0, 0]), dim=-1)

    def projector(self, state):
        return state

    def encode_targets(self, image, geometry):
        return self.encode_visits(image, geometry) + 0.5

    def image_transition(self, state, condition):
        return state + condition[..., :2]

    def response_transition(self, geometry, condition):
        state = geometry[..., :2] + condition[..., :2]
        return _Response(state, 0.1 * state)

    def forecast_response(self, geometry, condition):
        return self.response_transition(geometry[:, :-1], condition).future_state

    def forward(self, image, geometry, condition):
        output = _Output()
        output.visit_state = self.encode_visits(image, geometry)
        output.target = self.encode_targets(image, geometry)[:, 1:]
        output.image_prediction = self.image_transition(output.visit_state[:, :-1], condition)
        response = self.response_transition(geometry[:, :-1], condition)
        output.prediction = output.image_prediction + response.latent_correction
        output.future_response_state = response.future_state
        return output


def _loader():
    image = torch.arange(2 * 4 * 8 * 1 * 1 * 1, dtype=torch.float32).reshape(2, 4, 8, 1, 1, 1)
    geometry = torch.arange(2 * 4 * 9, dtype=torch.float32).reshape(2, 4, 9) / 10
    condition = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 100
    return [{"patient_id": ["A", "B"], "image": image, "geometry": geometry, "condition": condition}]


class InferenceTest(unittest.TestCase):
    def test_copy_audit_uses_combined_prediction_and_current_online_state(self) -> None:
        model = _FakeModel().eval()
        frame = copy_current_latent_audit(
            model, _loader(), device="cpu", fold=2, checkpoint="fold2.pt"
        )
        self.assertEqual(len(frame), 6)
        self.assertEqual(set(frame["transition"]), {"T0->T1", "T1->T2", "T2->T3"})
        self.assertEqual(set(frame["fold"]), {2})
        self.assertTrue(np.isfinite(frame["normalized_transition_gain"]).all())

    def test_collect_c1_keeps_response_but_native_t0_state(self) -> None:
        model = _FakeModel().eval()
        native = collect_frozen_inference(model, _loader(), device="cpu")
        c1 = collect_frozen_inference(
            model, _loader(), device="cpu", perturbation=repeated_t0_mri_only
        )
        np.testing.assert_array_equal(native.response_state, c1.response_state)
        np.testing.assert_array_equal(native.t0_state, c1.t0_state)
        self.assertEqual(native.response_state.shape, (2, 3, 2))
        self.assertEqual(native.t0_image_state.shape, (2, 2))

    def test_paired_swap_uses_native_target_and_reports_state_change(self) -> None:
        frame = paired_perturbation_latent_audit(
            _FakeModel().eval(),
            _loader(),
            swap_t1_t2,
            device="cpu",
            fold=0,
            checkpoint="fold0.pt",
            audit_condition="temporal_swap",
        )
        self.assertEqual(len(frame), 6)
        self.assertGreater(frame["response_state_l2_change"].max(), 0.0)
        self.assertTrue(np.isfinite(frame["perturbed_layer_norm_mse"]).all())

    def test_train_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval"):
            copy_current_latent_audit(
                _FakeModel().train(), _loader(), device="cpu", fold=0, checkpoint="x"
            )


if __name__ == "__main__":
    unittest.main()
