from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FORMAL_STAGE_B_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
)
MODEL_READY_SRC = (
    REPO_ROOT / "additional_experiments" / "c1b_model_ready_ftv_sanity" / "src"
)
sys.path[:0] = [str(FORMAL_STAGE_B_SRC), str(MODEL_READY_SRC), str(ROOT / "src")]

from c1b_spatial_audit.pooling import (  # noqa: E402
    S3_STAGE_GEOMETRY,
    expected_feature_shape,
)
from c1b_spatial_audit.contracts import file_sha256  # noqa: E402
from c1b_spatial_audit.probe_runner import ProbeMatrixPlan  # noqa: E402
from c1b_spatial_audit.s3_exporter import (  # noqa: E402
    S3_C1B_POOLINGS,
    S3_LEGACY_POOLINGS,
    S3_REPRESENTATION_CONTRACT,
    compute_s3_pooling_states,
    s3_feature_asset_path,
)
from c1b_spatial_audit.s3_probe_runner import (  # noqa: E402
    EXPECTED_S3_FEATURE_ASSETS,
    EXPECTED_S3_PROBE_CELLS,
    S3_POOLINGS,
    s3_plan_summary,
    validate_s3_exporter_completion,
)
import c1b_spatial_audit.s3_probe_runner as s3_probe_runner  # noqa: E402
from c1b_spatial_audit.s3_trigger import (  # noqa: E402
    NOT_TRIGGERED_STATUS,
    TRIGGERED_STATUS,
    require_s3_trigger_authorization,
    write_s3_trigger_gate,
)
import c1b_spatial_audit.s3_sidecars as s3_sidecars  # noqa: E402


def trigger_payload(*, supported: bool) -> dict[str, object]:
    digest = "0" * 64
    return {
        "schema_version": 1,
        "status": NOT_TRIGGERED_STATUS if supported else TRIGGERED_STATUS,
        "s3_execution_authorized": not supported,
        "decision_contract": "final_stage_strong_oracle_recovery_false",
        "final_stage_strong_oracle_recovery": {
            "status": "SUPPORTED_IN_PILOT" if supported else "NOT_SUPPORTED_IN_PILOT",
            "supported": supported,
            "thresholds": {},
            "per_seed": {},
        },
        "final_probe_root": "probes/final",
        "final_probe_cell_count": 180,
        "final_probe_metadata_inventory_sha256": digest,
        "preregistration_lock_sha256": digest,
        "plan_sha256": digest,
        "config_sha256": digest,
        "p0_equivalence_gate_sha256": digest,
        "p0_probe_replication_gate_sha256": digest,
        "trigger_implementation_sha256": digest,
        "analysis_implementation_sha256": digest,
        "probe_adapter_sha256": digest,
        "new_training_performed": False,
        "probe_refit_performed": False,
        "patient_identifiers_present": False,
    }


class _ResizeChannels(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.channels = channels
        self.stride = stride

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            value = torch.nn.functional.avg_pool3d(value, 2, 2)
        base = value.mean(dim=1, keepdim=True)
        return base.repeat(1, self.channels, 1, 1, 1)


class _ConstantPath(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = (tensor.shape[0], 64, *(size // 2 for size in tensor.shape[-3:]))
        return torch.full(shape, self.value, dtype=tensor.dtype, device=tensor.device)


class _FullResidualS3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = _ConstantPath(2.0)
        self.skip = _ConstantPath(5.0)
        self.calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.main(tensor) + self.skip(tensor)


class _Forbidden(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        raise AssertionError("forbidden downstream module was called")


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s3 = _FullResidualS3()
        self.fourth = _Forbidden()
        self.features = nn.Sequential(
            _ResizeChannels(16, 1),
            _ResizeChannels(32, 2),
            self.s3,
            self.fourth,
        )


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.response_projection = _Forbidden()
        self.eval().requires_grad_(False)


class S3GeometryAndExtractionTests(unittest.TestCase):
    def test_frozen_s3_geometry_and_shapes(self) -> None:
        self.assertEqual(S3_STAGE_GEOMETRY.channels, 64)
        self.assertEqual(S3_STAGE_GEOMETRY.receptive_field_zyx, (23, 23, 23))
        self.assertEqual(S3_STAGE_GEOMETRY.stride_zyx, (4, 4, 4))
        self.assertEqual(S3_STAGE_GEOMETRY.padding_zyx, (11, 11, 11))
        self.assertEqual(expected_feature_shape((32, 96, 96), stage="s3"), (8, 24, 24))
        self.assertEqual(
            expected_feature_shape((112, 176, 160), stage="s3"), (28, 44, 40)
        )

    def test_full_residual_output_is_used_and_downstream_is_never_called(self) -> None:
        model = _Model()
        image = torch.randn(1, 4, 7, 8, 8, 8, dtype=torch.float32)
        local = torch.ones(1, 4, 2, 2, 2, dtype=torch.float32)
        output = compute_s3_pooling_states(model, image, arm="L1", local_weights=local)
        self.assertEqual(tuple(output), S3_LEGACY_POOLINGS)
        for state, validity in output.values():
            self.assertEqual(tuple(state.shape), (1, 4, 64))
            self.assertTrue(validity.all())
            torch.testing.assert_close(state, torch.full_like(state, 7.0))
        self.assertEqual(model.encoder.s3.calls, 1)
        self.assertEqual(model.encoder.fourth.calls, 0)
        self.assertEqual(model.response_projection.calls, 0)

    def test_c1b_oracle_is_explicit_subset_without_gap_fallback(self) -> None:
        model = _Model()
        image = torch.randn(1, 4, 7, 8, 8, 8, dtype=torch.float32)
        local = torch.ones(2, 2, 2, dtype=torch.float32)
        oracle = torch.zeros(1, 4, 2, 2, 2, dtype=torch.float32)
        oracle[:, :2] = 1.0
        validity = torch.tensor([[True, True, False, False]])
        output = compute_s3_pooling_states(
            model,
            image,
            arm="N1",
            local_weights=local,
            oracle_weights=oracle,
            oracle_valid=validity,
        )
        self.assertEqual(tuple(output), S3_C1B_POOLINGS)
        oracle_state, oracle_valid = output["PORACLE"]
        torch.testing.assert_close(oracle_state[:, :2], torch.full_like(oracle_state[:, :2], 7.0))
        self.assertTrue(torch.equal(oracle_state[:, 2:], torch.zeros_like(oracle_state[:, 2:])))
        self.assertTrue(torch.equal(oracle_valid, validity))
        self.assertEqual(model.response_projection.calls, 0)

    def test_legacy_oracle_is_rejected(self) -> None:
        model = _Model()
        image = torch.randn(1, 4, 7, 8, 8, 8, dtype=torch.float32)
        local = torch.ones(1, 4, 2, 2, 2, dtype=torch.float32)
        oracle = torch.ones_like(local)
        with self.assertRaisesRegex(ValueError, "legacy S3 PORACLE"):
            compute_s3_pooling_states(
                model,
                image,
                arm="L1",
                local_weights=local,
                oracle_weights=oracle,
                oracle_valid=torch.ones(1, 4, dtype=torch.bool),
            )


class S3TriggerTests(unittest.TestCase):
    def test_gate_authorizes_only_weak_final_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s3_trigger_authorization.json"
            path.write_text(json.dumps(trigger_payload(supported=False)), encoding="utf-8")
            gate = require_s3_trigger_authorization(path, verify_live=False)
            self.assertTrue(gate["s3_execution_authorized"])
            path.write_text(json.dumps(trigger_payload(supported=True)), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                require_s3_trigger_authorization(path, verify_live=False)

    def test_public_trigger_write_is_atomic_0644_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s3_trigger_authorization.json"
            write_s3_trigger_gate(path, trigger_payload(supported=False))
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            with self.assertRaises(FileExistsError):
                write_s3_trigger_gate(path, trigger_payload(supported=False))


class S3SidecarTests(unittest.TestCase):
    def test_private_sidecar_schema_has_no_legacy_oracle_and_reindexes(self) -> None:
        small = {
            "FORMAL_PATIENT_COUNT": 2,
            "FORMAL_ORACLE_VISIT_COUNT": 4,
            "C1B_S3_SHAPE_ZYX": (1, 1, 1),
            "LEGACY_S3_SHAPE_ZYX": (1, 1, 1),
        }
        valid = np.asarray([[True, True, True, True], [False] * 4], dtype=bool)
        oracle = np.zeros((2, 4, 1, 1, 1), dtype=np.float32)
        oracle[valid] = 1.0
        bundle = s3_sidecars.S3AuditSidecars(
            patient_id=np.asarray(["a", "b"]),
            c1b_oracle_weight_s3=oracle,
            c1b_oracle_valid=valid,
            c1b_local_weight_s3=np.ones((1, 1, 1), dtype=np.float32),
            legacy_local_weight_s3=np.ones((2, 4, 1, 1, 1), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            s3_sidecars, **small
        ), mock.patch.object(
            s3_sidecars,
            "require_s3_trigger_authorization",
            return_value=trigger_payload(supported=False),
        ):
            root = Path(directory)
            trigger = root / "s3_trigger_authorization.json"
            trigger.write_text("{}\n", encoding="utf-8")
            source = root / "source.txt"
            source.write_text("source\n", encoding="utf-8")
            sidecar = root / "audit_sidecars_s3.private.npz"
            metadata = root / "audit_sidecars_s3.private.metadata.json"
            s3_sidecars.write_s3_sidecars(
                bundle,
                sidecar_output=sidecar,
                metadata_output=metadata,
                trigger_gate=trigger,
                source_paths={"source": source},
            )
            self.assertEqual(sidecar.stat().st_mode & 0o777, 0o600)
            self.assertEqual(metadata.stat().st_mode & 0o777, 0o600)
            with np.load(sidecar, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(s3_sidecars.S3_SIDECAR_KEYS))
                self.assertNotIn("legacy_oracle_weight_s3", archive.files)
            loaded = s3_sidecars.load_s3_sidecars(
                sidecar, ["b", "a"], verify_live=False
            )
            self.assertFalse(loaded.c1b_oracle_valid[0].any())
            self.assertTrue(loaded.c1b_oracle_valid[1].all())


class S3MatrixContractTests(unittest.TestCase):
    def test_exact_feature_and_probe_inventory_is_100(self) -> None:
        paths = {
            s3_feature_asset_path(Path("/tmp/features"), seed, arm, fold, pooling)
            for seed in (2026, 3026)
            for arm in ("L1", "L3", "N1", "N3")
            for fold in range(5)
            for pooling in (
                S3_LEGACY_POOLINGS if arm.startswith("L") else S3_C1B_POOLINGS
            )
        }
        self.assertEqual(len(paths), EXPECTED_S3_FEATURE_ASSETS)
        self.assertEqual(EXPECTED_S3_FEATURE_ASSETS, 100)
        self.assertEqual(EXPECTED_S3_PROBE_CELLS, 100)

    def test_s3_probe_summary_locks_no_nuisance_and_trigger_hash(self) -> None:
        digest = "a" * 64
        plan = ProbeMatrixPlan(
            stage="s3",
            poolings=S3_POOLINGS,
            cells=tuple([None] * 100),  # summary uses the exact count only
            feature_root=Path("/tmp/features"),
            probe_root=Path("/tmp/probes"),
            preregistration_path=Path("/tmp/lock"),
            preregistration_sha256=digest,
            nuisance_path=None,
            nuisance_sha256=None,
            gate_sha256={
                "s3_trigger_authorization_sha256": digest,
                "s3_probe_runner_sha256": digest,
            },
            exporter_completion_sha256=digest,
        )
        summary = s3_plan_summary(plan, status="VALIDATED_NOT_EXECUTED")
        self.assertEqual(summary["expected_cell_count"], 100)
        self.assertEqual(summary["nuisance_cell_count"], 0)
        self.assertEqual(
            summary["gate_sha256"]["s3_trigger_authorization_sha256"], digest
        )
        self.assertEqual(summary["gate_sha256"]["s3_probe_runner_sha256"], digest)
        self.assertEqual(summary["representation_contract"], S3_REPRESENTATION_CONTRACT)
        self.assertEqual(summary["legacy_poracle"], "NA_incomplete_source_authoritative_support_1488_of_1500")

    def test_s3_export_completion_binds_exact_100_and_trigger_sha(self) -> None:
        preregistration_sha = "1" * 64
        trigger_sha = "2" * 64
        sidecar_sha = "3" * 64
        metadata_sha = "4" * 64
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            s3_probe_runner, "EXPERIMENT_ROOT", Path(directory)
        ):
            feature_root = Path(directory) / "features"
            stage_root = feature_root / "s3"
            stage_root.mkdir(parents=True)
            inventory: dict[str, str] = {}
            for index in range(100):
                path = stage_root / f"cell_{index:03d}.private.metadata.json"
                path.write_text("{}\n", encoding="utf-8")
                path.chmod(0o600)
                inventory[str(path.resolve())] = file_sha256(path)
            preflight = stage_root / "feature_export_preflight.private.json"
            preflight.write_text(
                json.dumps(
                    {
                        "status": "PREFLIGHT_PASS",
                        "stage": "s3",
                        "representation_contract": S3_REPRESENTATION_CONTRACT,
                        "cell_count": 40,
                        "expected_asset_count": 100,
                        "preregistration_lock_sha256": preregistration_sha,
                        "trigger_gate_sha256": trigger_sha,
                        "sidecar_sha256": sidecar_sha,
                        "sidecar_metadata_sha256": metadata_sha,
                    }
                ),
                encoding="utf-8",
            )
            preflight.chmod(0o600)
            completion = stage_root / "feature_export_complete.private.json"
            payload = {
                "schema_version": 1,
                "status": "COMPLETE",
                "stage": "s3",
                "representation_contract": S3_REPRESENTATION_CONTRACT,
                "run_count": 40,
                "cell_count": 40,
                "expected_asset_count": 100,
                "feature_metadata_sha256": inventory,
                "preflight_sha256": file_sha256(preflight),
                "trigger_gate_sha256": trigger_sha,
                "sidecar_sha256": sidecar_sha,
                "sidecar_metadata_sha256": metadata_sha,
                "preregistration_lock_sha256": preregistration_sha,
            }
            completion.write_text(json.dumps(payload), encoding="utf-8")
            completion.chmod(0o600)
            self.assertEqual(
                validate_s3_exporter_completion(
                    feature_root,
                    preregistration_sha256=preregistration_sha,
                    trigger_gate_sha256=trigger_sha,
                ),
                file_sha256(completion),
            )
            with self.assertRaisesRegex(ValueError, "identity/count"):
                validate_s3_exporter_completion(
                    feature_root,
                    preregistration_sha256=preregistration_sha,
                    trigger_gate_sha256="f" * 64,
                )


if __name__ == "__main__":
    unittest.main()
