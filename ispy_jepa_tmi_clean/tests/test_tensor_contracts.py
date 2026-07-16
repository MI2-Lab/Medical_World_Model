from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from corejepa.config import LossConfig, ModelConfig, ReadoutConfig
from corejepa.data.condition import ConditionEncoder
from corejepa.data.imaging import dce8_visit, mask_geometry
from corejepa.data.records import PatientRecord
from corejepa.data.response_targets import ResponseTargetTransform
from corejepa.models import CoReJEPA
from corejepa.readout import fit_frozen_landmark_readout, landmark_features
from corejepa.training.losses import PretrainingObjective


def _records() -> list[PatientRecord]:
    arms = [
        "ISPY1_NACT",
        "Paclitaxel",
        "Paclitaxel + ABT 888 + Carboplatin",
        "Paclitaxel + AMG 386",
        "Paclitaxel + AMG 386 + Trastuzumab",
        "Paclitaxel + Ganetespib",
        "Paclitaxel + Ganitumab",
        "Paclitaxel + MK-2206",
        "Paclitaxel + MK-2206 + Trastuzumab",
        "Paclitaxel + Neratinib",
        "Paclitaxel + Pembrolizumab",
        "Paclitaxel + Pertuzumab + Trastuzumab",
        "Paclitaxel + Trastuzumab",
        "T-DM1 + Pertuzumab",
    ]
    return [
        PatientRecord(
            patient_id=f"P{index}",
            cohort="ispy1" if index == 0 else "ispy2",
            arm=arm,
            hr=index % 2,
            her2=(index // 2) % 2,
            mp=(index // 3) % 2,
            age=40.0 + index,
            manifest_path=Path(f"P{index}/manifest.json"),
            pcr=index % 2,
        )
        for index, arm in enumerate(arms)
    ]


def test_paper_condition_dimension_and_prefix_mask() -> None:
    encoder = ConditionEncoder(_records())
    condition = encoder.encode(_records()[1])
    assert condition.shape == (3, 25)
    np.testing.assert_array_equal(condition[:, :3], np.eye(3, dtype=np.float32))
    np.testing.assert_array_equal(condition[0, 3:7], [1, 0, 0, 0])
    np.testing.assert_array_equal(condition[1, 3:7], [1, 1, 0, 0])
    np.testing.assert_array_equal(condition[2, 3:7], [1, 1, 1, 0])


def test_dce8_and_geometry_shapes() -> None:
    grid = np.indices((12, 14, 8)).sum(axis=0).astype(np.float32)
    dce = np.stack([grid * (1.0 + 0.1 * phase) + phase for phase in range(6)], axis=-1)
    roi = np.zeros((12, 14, 8), dtype=bool)
    roi[3:8, 4:10, 2:6] = True
    image, phases = dce8_visit(
        dce,
        roi,
        center_xyz=(5.0, 6.5, 3.5),
        crop_size_zyx=(8, 12, 12),
        phase_metadata={"pre": 0, "post_early": 2, "post_late": 5},
        phase_policy="adaptive_early_late",
    )
    assert image.shape == (8, 8, 12, 12)
    assert phases == (0, 2, 5)
    assert set(np.unique(image[-1])).issubset({0.0, 1.0})
    geometry = mask_geometry(image[-1])
    assert geometry.shape == (9,)
    assert 0 < geometry[0] < 1


def test_model_and_loss_contracts_backward() -> None:
    records = _records()
    condition_encoder = ConditionEncoder(records)
    config = ModelConfig(
        base_channels=4,
        latent_dim=32,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_dim=64,
        response_dim=16,
        response_hidden_dim=32,
        expert_hidden_dim=24,
        expert_gate_hidden_dim=24,
    )
    model = CoReJEPA(config, condition_encoder.spec.dim)
    batch_size = 3
    image = torch.randn(batch_size, 4, 8, 8, 16, 16)
    geometry = torch.rand(batch_size, 4, 9)
    condition = torch.from_numpy(np.stack([condition_encoder.encode(record) for record in records[:batch_size]]))
    output = model(image, geometry, condition)
    assert output.visit_state.shape == (batch_size, 4, 32)
    assert output.prediction.shape == (batch_size, 3, 32)
    assert output.future_response_state.shape == (batch_size, 3, 16)
    assert output.decoded_geometry.shape == (batch_size, 3, 9)
    assert output.vector_prediction.shape == (batch_size, 3, 18)
    assert output.update_vector_prediction.shape == (batch_size, 2, 18)
    objective = PretrainingObjective(LossConfig(), 16, torch.ones(6))
    loss, stats = objective(
        output,
        {
            "response_score": torch.randn(batch_size, 3, 1),
            "response_vector": torch.randn(batch_size, 3, 18),
            "routing_target": torch.tensor([0, 1, 2]),
        },
    )
    assert torch.isfinite(loss)
    assert "prediction" in stats
    loss.backward()
    assert model.encoder.features[0].main[0].weight.grad is not None


def test_frozen_landmark_readout_dimension() -> None:
    states = np.random.default_rng(0).normal(size=(5, 3, 64)).astype(np.float32)
    for landmark in range(3):
        features = landmark_features(states, landmark)
        assert features.shape == (5, 1283)


def test_response_target_transform_is_train_fitted() -> None:
    records = _records()
    raw = np.random.default_rng(2).normal(size=(len(records), 3, 18)).astype(np.float32)
    raw[0, 0, 3] = np.nan
    transform = ResponseTargetTransform.fit(raw, records, list(range(10)))
    vector, score = transform.transform(raw, records)
    assert vector.shape == (len(records), 3, 18)
    assert score.shape == (len(records), 3, 1)
    assert np.isfinite(vector).all()
    assert np.isfinite(score).all()


def test_end_to_end_frozen_readout(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    states = rng.normal(size=(30, 3, 8)).astype(np.float32)
    labels = np.arange(30, dtype=np.int64) % 2
    states[:, :, 0] += labels[:, None] * 0.5
    states_path = tmp_path / "states.npz"
    np.savez_compressed(
        states_path,
        future_response_state=states,
        pcr=labels,
        patient_ids=np.asarray([f"P{index}" for index in range(30)]),
        n_primary=np.asarray(30),
    )
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        '{"primary_train":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],'
        '"validation":[18,19,20,21,22,23],"test":[24,25,26,27,28,29]}'
    )
    summary = fit_frozen_landmark_readout(
        states_path,
        splits_path,
        tmp_path / "readout",
        ReadoutConfig(penalties=("l2",), c_grid=(0.1,), max_iter=200),
    )
    assert summary["feature_dim"] == 163
    assert (tmp_path / "readout" / "flr_metrics.csv").exists()
