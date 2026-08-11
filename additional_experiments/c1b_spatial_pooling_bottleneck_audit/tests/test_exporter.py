from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import c1b_spatial_audit.exporter as exporter  # noqa: E402
from c1b_spatial_audit.exporter import (  # noqa: E402
    C1B_POOLINGS,
    LEGACY_POOLINGS,
    compute_final_pooling_states,
    feature_asset_path,
    feature_metadata_path,
    load_audit_sidecars,
    load_feature_asset,
    validate_feature_export,
)


class _TinyEncoder(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = F.adaptive_avg_pool3d(image.mean(dim=1, keepdim=True), (1, 1, 1))
        scales = torch.linspace(0.5, 1.5, 128, device=image.device).reshape(
            1, 128, 1, 1, 1
        )
        return value * scales


class _Forbidden(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("a forbidden model branch was called")


class _SyntheticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TinyEncoder()
        self.response_projection = nn.Sequential(
            nn.Linear(128, 192), nn.LayerNorm(192)
        )
        self.projector = _Forbidden()
        self.transition = _Forbidden()
        self.target_encoder = _Forbidden()
        self.target_response_projection = _Forbidden()
        self.target_projector = _Forbidden()
        self.ftv_head = _Forbidden()
        self.eval().requires_grad_(False)


def _image(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(19)
    return torch.randn(batch, 4, 7, 8, 8, 8, generator=generator)


def _metadata_base() -> dict[str, object]:
    digest = "1" * 64
    return {
        "checkpoint_path": "/private/checkpoint.pt",
        "checkpoint_sha256": digest,
        "checkpoint_lock_key": "seed_2026/N1/fold_0",
        "reference_feature_path": "/private/reference.npz",
        "reference_feature_sha256": digest,
        "reference_feature_metadata_path": "/private/reference.json",
        "reference_feature_metadata_sha256": digest,
        "preregistration_lock_sha256": digest,
        "plan_sha256": digest,
        "config_sha256": digest,
        "sidecar_path": "/private/sidecar.npz",
        "sidecar_sha256": digest,
        "data_contract_provenance_sha256": digest,
        "checkpoint_data_provenance_sha256": digest,
        "stage_a_sentinel_sha256": digest,
        "implementation_sha256": {"exporter.py": digest},
        "device": "cuda:0",
        "batch_size": 4,
        "workers": 2,
        "feature_tensor": "full_model.encoder_output_before_gap",
        "response_projection": "frozen_online_Linear128x192_plus_LayerNorm",
        "training_performed": False,
        "projector_called": False,
        "transition_called": False,
        "target_encoder_called": False,
        "ftv_head_called": False,
        "test_labels_used": False,
    }


class FrozenExporterTests(unittest.TestCase):
    def test_compute_c1b_uses_only_encoder_and_response_projection(self) -> None:
        model = _SyntheticModel()
        image = _image()
        batch = image.shape[0]
        oracle_valid = torch.tensor(
            [[True, False, True, False], [False, True, True, False]],
            dtype=torch.bool,
        )
        output = compute_final_pooling_states(
            model,
            image,
            arm="N1",
            local_weights=torch.ones(1, 1, 1),
            valid_weights=torch.ones(batch, 4, 1, 1, 1),
            oracle_weights=oracle_valid[..., None, None, None].float(),
            oracle_valid=oracle_valid,
        )
        self.assertEqual(set(output), set(C1B_POOLINGS))
        with torch.inference_mode():
            spatial = model.encoder(image.reshape(batch * 4, 7, 8, 8, 8))
            expected_p0 = model.response_projection(
                spatial.mean((-3, -2, -1))
            ).reshape(batch, 4, 192)
        torch.testing.assert_close(output["P0"][0], expected_p0, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            output["PLOCAL+GLOBAL"][0],
            torch.cat((output["PLOCAL"][0], output["P0"][0]), dim=-1),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(tuple(output["PLOCAL+GLOBAL"][0].shape), (batch, 4, 384))
        self.assertTrue(torch.equal(output["PORACLE"][1], oracle_valid))
        self.assertEqual(int(torch.count_nonzero(output["PORACLE"][0][~oracle_valid])), 0)
        self.assertFalse(any(state.requires_grad for state, _ in output.values()))

    def test_legacy_rejects_fake_valid_or_oracle_masks(self) -> None:
        model = _SyntheticModel()
        image = _image(batch=1)
        local = torch.ones(1, 4, 1, 1, 1)
        output = compute_final_pooling_states(
            model, image, arm="L1", local_weights=local
        )
        self.assertEqual(set(output), set(LEGACY_POOLINGS))
        with self.assertRaisesRegex(ValueError, "preregistered NA"):
            compute_final_pooling_states(
                model,
                image,
                arm="L1",
                local_weights=local,
                valid_weights=torch.ones_like(local),
            )

    def test_sidecar_reorders_global_patient_axis_to_fold_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "audit_sidecars.private.npz"
            tagged = np.empty((2, 4, 1, 1, 1), dtype=np.float32)
            tagged[0] = 0.25
            tagged[1] = 0.75
            np.savez_compressed(
                sidecar,
                patient_id=np.asarray(["patient_b", "patient_a"]),
                c1b_valid_weight_final=tagged,
                c1b_oracle_weight_final=tagged,
                c1b_oracle_valid=np.ones((2, 4), dtype=bool),
                c1b_local_weight_final=np.ones((1, 1, 1), dtype=np.float32),
                legacy_local_weight_final=tagged,
            )
            loaded = load_audit_sidecars(
                sidecar,
                ("patient_a", "patient_b"),
                c1b_feature_shape_zyx=(1, 1, 1),
                legacy_feature_shape_zyx=(1, 1, 1),
                expected_oracle_valid_count=8,
            )
            self.assertEqual(loaded.patient_id, ("patient_a", "patient_b"))
            self.assertTrue(np.all(loaded.c1b_valid_weight_final[0] == np.float32(0.75)))
            self.assertTrue(np.all(loaded.c1b_valid_weight_final[1] == np.float32(0.25)))
            self.assertTrue(np.all(loaded.legacy_local_weight_final[0] == np.float32(0.75)))
            self.assertEqual(loaded.c1b_local_weight_final.shape, (1, 1, 1))

    def test_sidecar_rejects_patient_set_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "audit_sidecars.private.npz"
            weights = np.ones((2, 4, 1, 1, 1), dtype=np.float32)
            np.savez_compressed(
                sidecar,
                patient_id=np.asarray(["a", "b"]),
                c1b_valid_weight_final=weights,
                c1b_oracle_weight_final=weights,
                c1b_oracle_valid=np.ones((2, 4), dtype=bool),
                c1b_local_weight_final=np.ones((1, 1, 1), dtype=np.float32),
                legacy_local_weight_final=weights,
            )
            with self.assertRaisesRegex(ValueError, "identity set"):
                load_audit_sidecars(
                    sidecar,
                    ("a", "c"),
                    c1b_feature_shape_zyx=(1, 1, 1),
                    legacy_feature_shape_zyx=(1, 1, 1),
                    expected_oracle_valid_count=8,
                )

    def test_schema_layout_and_probe_adapter_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features" / "final" / "p0.private.npz"
            patient_ids = ("p0", "p1", "p2")
            split = ("train", "val", "test")
            state = np.arange(3 * 4 * 192, dtype=np.float32).reshape(3, 4, 192)
            valid = np.ones((3, 4), dtype=bool)
            exporter._write_feature(
                path,
                patient_ids=patient_ids,
                split_labels=split,
                state=state,
                state_valid=valid,
                arm="N1",
                seed_base=2026,
                fold=0,
                pooling="P0",
                metadata_base=_metadata_base(),
            )
            asset, metadata = validate_feature_export(
                path,
                expected_arm="N1",
                expected_seed_base=2026,
                expected_fold=0,
                expected_pooling="P0",
                expected_patient_count=3,
                verify_live_inputs=False,
            )
            self.assertEqual(asset.state.shape, (3, 4, 192))
            self.assertEqual(metadata["status"], "COMPLETE")
            self.assertEqual(metadata["sidecar_keys_used"], [])
            self.assertEqual(feature_metadata_path(path).name, "p0.private.metadata.json")

            from c1b_spatial_audit.probes import load_frozen_state_asset

            probe_asset = load_frozen_state_asset(path)
            self.assertEqual(probe_asset.feature_dim, 192)
            np.testing.assert_array_equal(probe_asset.state, state)

    def test_feature_layout_is_deterministic_and_arm_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = feature_asset_path(root, 3026, "N3", 4, "PLOCAL+GLOBAL")
            self.assertEqual(
                path,
                root.resolve()
                / "final"
                / "seed_3026"
                / "N3"
                / "fold_4"
                / "plocal_global.private.npz",
            )
            with self.assertRaisesRegex(ValueError, "undefined"):
                feature_asset_path(root, 2026, "L1", 0, "PORACLE")

    def test_loader_rejects_nonzero_invalid_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poracle.private.npz"
            state = np.zeros((3, 4, 192), dtype=np.float32)
            valid = np.zeros((3, 4), dtype=bool)
            valid[0, 0] = True
            state[1, 1, 0] = 1.0
            np.savez_compressed(
                path,
                patient_id=np.asarray(["a", "b", "c"]),
                split=np.asarray(["train", "val", "test"]),
                state=state,
                state_valid=valid,
                arm=np.asarray("N1"),
                seed_base=np.asarray(2026, dtype=np.int64),
                fold=np.asarray(0, dtype=np.int64),
                pooling=np.asarray("PORACLE"),
            )
            with self.assertRaisesRegex(ValueError, "zero placeholder"):
                load_feature_asset(path)


if __name__ == "__main__":
    unittest.main()
