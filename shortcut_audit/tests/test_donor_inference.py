from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from shortcut_audit.auditlib.donor_inference import (
    DonorPairDataset,
    donor_prediction_frame,
    run_donor_swap_inference,
)
from shortcut_audit.auditlib.readouts import AuditReadoutConfig, fit_fold_readout


class _Base(Dataset):
    def __init__(self):
        self.ids = ["A", "B", "C", "D", "E", "F"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        offset = float(index * 100)
        image = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8, 1, 1, 1) + offset
        geometry = torch.arange(4 * 9, dtype=torch.float32).reshape(4, 9) / 10 + offset
        condition = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) / 100
        return {"patient_id": self.ids[index], "image": image, "geometry": geometry, "condition": condition}


class _Response:
    def __init__(self, state):
        self.future_state = state
        self.latent_correction = state * 0.01


class _Model(torch.nn.Module):
    def encode_visits(self, image, geometry):
        return torch.stack((image[:, :, 0, 0, 0, 0], geometry[:, :, 0]), dim=-1)

    def encode_targets(self, image, geometry):
        return self.encode_visits(image, geometry) + 0.5

    def image_transition(self, state, condition):
        return state + condition[..., :2]

    def response_transition(self, geometry, condition):
        return _Response(geometry[..., :2] + condition[..., :2])

    def forecast_response(self, geometry, condition):
        return self.response_transition(geometry[:, :-1], condition).future_state

    def forward(self, image, geometry, condition):
        class Output:
            pass
        result = Output()
        result.visit_state = self.encode_visits(image, geometry)
        result.target = self.encode_targets(image, geometry)[:, 1:]
        result.image_prediction = self.image_transition(result.visit_state[:, :-1], condition)
        response = self.response_transition(geometry[:, :-1], condition)
        result.prediction = result.image_prediction + response.latent_correction
        result.future_response_state = response.future_state
        return result


def _mapping():
    return pd.DataFrame(
        {
            "recipient_patient_id": ["E", "F"],
            "donor_patient_id": ["F", "E"],
            "fold": [0, 0],
            "audit_repetition": [1, 1],
            "matching_distance": [0.2, 0.3],
        }
    )


class DonorInferenceTest(unittest.TestCase):
    def test_pair_dataset_keeps_recipient_context_and_replaces_followups(self) -> None:
        base = _Base()
        pairs = DonorPairDataset(base, _mapping(), {p: i for i, p in enumerate(base.ids)}, expected_fold=0)
        item = pairs[0]
        recipient, donor = base[4], base[5]
        torch.testing.assert_close(item["perturbed_image"][0], recipient["image"][0])
        torch.testing.assert_close(item["perturbed_image"][1:3], donor["image"][1:3])
        torch.testing.assert_close(item["condition"], recipient["condition"])

    def test_inference_and_donor_prediction_contract(self) -> None:
        base = _Base()
        mapping = _mapping()
        pairs = DonorPairDataset(base, mapping, {p: i for i, p in enumerate(base.ids)}, expected_fold=0)
        inference = run_donor_swap_inference(
            _Model().eval(), DataLoader(pairs, batch_size=2), mapping=mapping, device="cpu", fold=0, checkpoint="fold0.pt"
        )
        self.assertEqual(inference.response_state.shape, (2, 3, 2))
        self.assertEqual(len(inference.latent_metrics), 6)
        self.assertGreater(inference.latent_metrics["response_state_l2_change"].max(), 0)

        rng = np.random.default_rng(3)
        states = rng.normal(size=(6, 3, 2)).astype(np.float32)
        labels = np.asarray([0, 1, 0, 1, 0, 1])
        ids = np.asarray(base.ids)
        bundle = fit_fold_readout(
            states,
            labels,
            ids,
            [0, 1],
            [2, 3],
            fold=0,
            test_indices=[4, 5],
            config=AuditReadoutConfig(penalties=("l2",), c_grid=(0.1,), max_iter=200),
        )
        frame = donor_prediction_frame(
            bundle, inference, {"E": 0, "F": 1}, checkpoint="fold0.pt"
        )
        self.assertEqual(len(frame), 6)
        self.assertTrue(frame["donor_patient_id"].notna().all())
        self.assertTrue(frame["repetition_id"].eq(1).all())

    def test_invalid_self_or_cross_fold_mapping_fails(self) -> None:
        base = _Base()
        mapping = _mapping()
        mapping.loc[0, "donor_patient_id"] = "E"
        with self.assertRaisesRegex(ValueError, "不得等于"):
            DonorPairDataset(base, mapping, {p: i for i, p in enumerate(base.ids)}, expected_fold=0)
        cross = _mapping()
        cross.loc[0, "fold"] = 1
        with self.assertRaisesRegex(ValueError, "非当前"):
            DonorPairDataset(base, cross, {p: i for i, p in enumerate(base.ids)}, expected_fold=0)


if __name__ == "__main__":
    unittest.main()
