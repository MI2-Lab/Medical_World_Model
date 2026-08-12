from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from lg_response_pilot.contracts import (  # noqa: E402
    ARMS,
    ARM_SPECS,
    C1B_INPUT_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    FINAL_FEATURE_CHANNELS,
    LOCAL_WINDOW_MM_XYZ,
    MODEL_KWARGS,
)
from lg_response_pilot.model import (  # noqa: E402
    LocalGlobalResponseWorldModel,
    build_model,
    build_objective,
    load_checkpoint_for_evaluation,
    paired_initialization_report,
    shared_initialization_sha256,
    transition_sha256,
)
from lg_response_pilot.pooling import (  # noqa: E402
    build_fixed_c1b_local_weights,
    derived_final_feature_shape,
    fixed_physical_local_weights,
    pooling_contract,
    weighted_average_pool,
)
from lg_response_pilot.upstream import (  # noqa: E402
    DGRSWorldModel,
    fixed_physical_local_weights as audited_fixed_physical_local_weights,
    weighted_average_pool as audited_weighted_average_pool,
)


class _FixedEncoder(nn.Module):
    def __init__(self, spatial: torch.Tensor) -> None:
        super().__init__()
        self.spatial = spatial
        self.calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if int(image.shape[0]) != int(self.spatial.shape[0]):
            raise ValueError("fixed test map batch differs from flattened visits")
        return self.spatial


def _sealed_upstream(model_name: str, seed: int) -> DGRSWorldModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DGRSWorldModel(model_name=model_name, **dict(MODEL_KWARGS))


class ArmAndUpstreamContractTest(unittest.TestCase):
    def test_arm_inventory_and_public_spec_fields(self) -> None:
        self.assertEqual(ARMS, ("GAP0", "GAP3", "LOCAL0", "LOCAL3"))
        self.assertEqual(tuple(ARM_SPECS), ARMS)
        for arm, spec in ARM_SPECS.items():
            self.assertEqual(spec.name, arm)
            self.assertIn(spec.architecture, {"GAP", "LOCAL"})
            self.assertIsInstance(spec.grounded, bool)
            self.assertEqual(spec.upstream_model_name, "G3" if spec.grounded else "G1")
        for removed_arm in ("LG0", "LG3"):
            with self.assertRaisesRegex(ValueError, "confirmation arm"):
                build_model(removed_arm, 2026)

    def test_pooling_public_functions_are_exact_audited_objects(self) -> None:
        self.assertIs(
            fixed_physical_local_weights,
            audited_fixed_physical_local_weights,
        )
        self.assertIs(weighted_average_pool, audited_weighted_average_pool)

    def test_gap_state_and_arithmetic_are_exact_upstream_parity(self) -> None:
        seed = 1729
        for arm, upstream_name in (("GAP0", "G1"), ("GAP3", "G3")):
            pilot = build_model(arm, seed).eval()
            upstream = _sealed_upstream(upstream_name, seed).eval()
            self.assertEqual(tuple(pilot.state_dict()), tuple(upstream.state_dict()))
            for name, tensor in pilot.state_dict().items():
                self.assertTrue(torch.equal(tensor, upstream.state_dict()[name]), name)

        pilot = build_model("GAP0", seed).eval()
        upstream = _sealed_upstream("G1", seed).eval()
        feature_shape = derived_final_feature_shape()
        spatial = torch.randn(
            (2, FINAL_FEATURE_CHANNELS, *feature_shape),
            generator=torch.Generator().manual_seed(91),
        )
        image = torch.empty(
            (1, 2, int(MODEL_KWARGS["image_channels"]), *C1B_INPUT_SHAPE_ZYX),
            device="meta",
        )
        pilot_output = pilot._encode_sequence(
            image,
            None,
            _FixedEncoder(spatial),
            pilot.response_projection,
            pilot.projector,
        )
        upstream_output = upstream._encode_sequence(
            image,
            None,
            _FixedEncoder(spatial),
            upstream.response_projection,
            upstream.projector,
        )
        for observed, expected in zip(pilot_output, upstream_output):
            if expected is None:
                self.assertIsNone(observed)
            else:
                self.assertTrue(torch.equal(observed, expected))

    def test_each_architecture_calls_its_encoder_exactly_once(self) -> None:
        feature_shape = derived_final_feature_shape()
        spatial = torch.zeros((1, FINAL_FEATURE_CHANNELS, *feature_shape))
        image = torch.empty(
            (1, 1, int(MODEL_KWARGS["image_channels"]), *C1B_INPUT_SHAPE_ZYX),
            device="meta",
        )
        for arm in ("GAP0", "LOCAL0"):
            with self.subTest(arm=arm):
                model = build_model(arm, 61).eval()
                encoder = _FixedEncoder(spatial)
                model._encode_sequence(
                    image,
                    None,
                    encoder,
                    model.response_projection,
                    model.projector,
                )
                self.assertEqual(encoder.calls, 1)


class SpatialPoolingContractTest(unittest.TestCase):
    def test_fixed_weights_and_weighted_pool_are_audit_exact(self) -> None:
        feature_shape = derived_final_feature_shape()
        direct = audited_fixed_physical_local_weights(
            C1B_INPUT_SHAPE_ZYX,
            feature_shape,
            C1B_SPACING_XYZ_MM,
            stage="final",
            device="cpu",
            dtype=torch.float32,
        )
        wrapped = build_fixed_c1b_local_weights()
        model = build_model("LOCAL0", 2026)
        self.assertTrue(torch.equal(wrapped, direct))
        self.assertTrue(torch.equal(model.local_pooling_weight, direct))
        self.assertTrue(bool(((direct > 0.0) & (direct < 1.0)).any()))

        spatial = torch.randn(
            (2, FINAL_FEATURE_CHANNELS, *feature_shape),
            generator=torch.Generator().manual_seed(72),
        )
        self.assertTrue(
            torch.equal(
                model.pooled_response_input(spatial),
                audited_weighted_average_pool(spatial, direct),
            )
        )

    def test_runtime_feature_shape_is_derived_and_fail_closed(self) -> None:
        shape = derived_final_feature_shape()
        model = build_model("GAP0", 8)
        valid = torch.zeros((1, FINAL_FEATURE_CHANNELS, *shape))
        self.assertEqual(model._validate_final_spatial(valid), shape)
        wrong_shape = (shape[0], shape[1], shape[2] + 1)
        wrong = torch.zeros((1, FINAL_FEATURE_CHANNELS, *wrong_shape))
        with self.assertRaisesRegex(ValueError, "actual encoder.features"):
            model.pooled_response_input(wrong)

        contract = pooling_contract()
        self.assertEqual(contract["derived_feature_shape_zyx"], list(shape))
        self.assertEqual(contract["coordinate_convention"], "tensor_ZYX_spacing_XYZ")
        json.dumps(contract, allow_nan=False)

    def test_actual_encoder_return_is_the_full_final_residual_block_map(self) -> None:
        model = build_model("GAP0", 18).to("meta").eval()
        captured: dict[str, torch.Tensor] = {}

        def capture_final(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured["output"] = output

        handle = model.encoder.features[3].register_forward_hook(capture_final)
        try:
            image = torch.empty(
                (1, int(MODEL_KWARGS["image_channels"]), *C1B_INPUT_SHAPE_ZYX),
                device="meta",
            )
            actual = model.encoder(image)
        finally:
            handle.remove()
        self.assertIs(actual, captured["output"])
        self.assertEqual(model._validate_final_spatial(actual), derived_final_feature_shape())

    def test_no_roi_or_mask_is_admitted(self) -> None:
        model = build_model("LOCAL3", 12)
        image = torch.empty(
            (1, 1, int(MODEL_KWARGS["image_channels"]), *C1B_INPUT_SHAPE_ZYX),
            device="meta",
        )
        roi = torch.empty((1, 1, 1, *C1B_INPUT_SHAPE_ZYX), device="meta")
        with self.assertRaisesRegex(ValueError, "roi_mask"):
            model._validate_sequence_inputs(image, roi)
        self.assertNotIn("roi", json.dumps(model.model_config()).lower())
        self.assertEqual(model.architecture_contract()["roi_mask_use"], "absent")


class ProjectionAndTargetContractTest(unittest.TestCase):
    def test_gap_and_local_use_identical_frozen_projection_initialization(self) -> None:
        models = {arm: build_model(arm, 3026).eval() for arm in ARMS}
        reference_linear, reference_norm = models["GAP0"].response_projection
        reference_target_linear, reference_target_norm = models[
            "GAP0"
        ].target_response_projection
        for arm, model in models.items():
            with self.subTest(arm=arm):
                linear, norm = model.response_projection
                target_linear, target_norm = model.target_response_projection
                self.assertEqual(linear.in_features, FINAL_FEATURE_CHANNELS)
                self.assertTrue(torch.equal(linear.weight, reference_linear.weight))
                self.assertTrue(torch.equal(linear.bias, reference_linear.bias))
                self.assertTrue(torch.equal(norm.weight, reference_norm.weight))
                self.assertTrue(torch.equal(norm.bias, reference_norm.bias))
                self.assertTrue(
                    torch.equal(target_linear.weight, reference_target_linear.weight)
                )
                self.assertTrue(
                    torch.equal(target_linear.bias, reference_target_linear.bias)
                )
                self.assertTrue(
                    torch.equal(target_norm.weight, reference_target_norm.weight)
                )
                self.assertTrue(
                    torch.equal(target_norm.bias, reference_target_norm.bias)
                )
                self.assertNotIn("local_global_order", model.architecture_contract())

    def test_online_target_pooling_is_symmetric_and_ema_updates_projection(self) -> None:
        shape = derived_final_feature_shape()
        spatial = torch.randn(
            (2, FINAL_FEATURE_CHANNELS, *shape),
            generator=torch.Generator().manual_seed(46),
        )
        image = torch.empty(
            (1, 2, int(MODEL_KWARGS["image_channels"]), *C1B_INPUT_SHAPE_ZYX),
            device="meta",
        )
        for arm in ("GAP3", "LOCAL3"):
            with self.subTest(arm=arm):
                model = build_model(arm, 45).eval()
                online = model._encode_sequence(
                    image,
                    None,
                    _FixedEncoder(spatial),
                    model.response_projection,
                    model.projector,
                )
                target = model._encode_sequence(
                    image,
                    None,
                    _FixedEncoder(spatial),
                    model.target_response_projection,
                    model.target_projector,
                )
                self.assertTrue(torch.equal(online[0], target[0]))
                self.assertTrue(torch.equal(online[1], target[1]))

                old_target = (
                    model.target_response_projection[0].weight.detach().clone()
                )
                with torch.no_grad():
                    model.response_projection[0].weight.add_(0.25)
                online_weight = model.response_projection[0].weight.detach().clone()
                expected = old_target.clone()
                expected.mul_(0.4).add_(online_weight, alpha=0.6)
                model.update_target(0.4)
                torch.testing.assert_close(
                    model.target_response_projection[0].weight,
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertFalse(
                    any(
                        parameter.requires_grad
                        for parameter in model.target_response_projection.parameters()
                    )
                )


class InitializationAndCheckpointTest(unittest.TestCase):
    def test_build_is_deterministic_and_does_not_advance_caller_rng(self) -> None:
        torch.manual_seed(99)
        before = torch.random.get_rng_state().clone()
        first = build_model("LOCAL0", 101)
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        second = build_model("LOCAL0", 101)
        self.assertEqual(
            shared_initialization_sha256(first),
            shared_initialization_sha256(second),
        )
        self.assertEqual(transition_sha256(first), transition_sha256(second))

    def test_paired_initialization_report_and_parameter_counts(self) -> None:
        report = paired_initialization_report(2028)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["effective_seed"], 2028)
        self.assertEqual(report["arms"], list(ARMS))
        self.assertEqual(set(report["per_arm"]), set(ARMS))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["architecture_pairs"],
            {
                "GAP0_LOCAL0": report["shared_initialization_sha256"],
                "GAP3_LOCAL3": report["shared_initialization_sha256"],
            },
        )
        self.assertEqual(len(report["shared_initialization_sha256"]), 64)
        self.assertEqual(len(report["transition_sha256"]), 64)
        for arm in ARMS:
            record = report["per_arm"][arm]
            self.assertEqual(
                record["shared_initialization_sha256"],
                report["shared_initialization_sha256"],
            )
            self.assertEqual(
                record["transition_sha256"], report["transition_sha256"]
            )
            self.assertEqual(
                record["parameter_counts"]["response_projection"],
                25_152,
            )
        json.dumps(report, allow_nan=False)

    def test_objective_grounding_and_selected_checkpoint_roundtrip(self) -> None:
        self.assertEqual(build_objective("LOCAL0").lambda_ftv, 0.0)
        self.assertEqual(build_objective("LOCAL3").lambda_ftv, 0.25)
        model = build_model("LOCAL3", 2027)
        payload = {
            "schema_version": 1,
            "stage": "local_response_state_multiseed_confirmation",
            "arm": model.arm,
            "state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "architecture_contract": model.architecture_contract(),
            "selected": True,
            "test_data_used": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.pt"
            torch.save(payload, path)
            loaded, loaded_payload = load_checkpoint_for_evaluation(path, "cpu")
            self.assertIsInstance(loaded, LocalGlobalResponseWorldModel)
            self.assertEqual(loaded.arm, "LOCAL3")
            self.assertFalse(loaded.training)
            self.assertTrue(loaded_payload["selected"])
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)

            payload["selected"] = False
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "selected, test-blind"):
                load_checkpoint_for_evaluation(path)

            payload["selected"] = True
            payload["state_dict"] = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            payload["state_dict"]["local_pooling_weight"].zero_()
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "fixed audited"):
                load_checkpoint_for_evaluation(path)


if __name__ == "__main__":
    unittest.main()
