"""Audit-only longitudinal perturbation contracts."""

from __future__ import annotations

import unittest

import numpy as np

from shortcut_audit.auditlib.perturbations import (
    predict_perturbed_context_against_native_target,
    repeated_t0_full_image_derived,
    repeated_t0_mri_only,
    replace_followups_with_donor,
    swap_t1_t2,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 允许 numpy-only 环境运行基础测试
    torch = None  # type: ignore[assignment]


def _numpy_inputs(*, batched: bool = False, offset: float = 0.0):
    image = np.arange(4 * 8 * 2 * 3 * 2, dtype=np.float32).reshape(4, 8, 2, 3, 2)
    geometry = np.arange(4 * 9, dtype=np.float32).reshape(4, 9)
    condition = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    image = image + offset
    geometry = geometry + offset
    condition = condition + offset
    if batched:
        image = np.stack((image, image + 1_000.0))
        geometry = np.stack((geometry, geometry + 1_000.0))
        condition = np.stack((condition, condition + 1_000.0))
    return image, geometry, condition


def _assert_independent_numpy(test: unittest.TestCase, result, image, geometry, condition) -> None:
    test.assertFalse(np.shares_memory(result.image, image))
    test.assertFalse(np.shares_memory(result.geometry, geometry))
    test.assertFalse(np.shares_memory(result.condition, condition))


class NumpyPerturbationTests(unittest.TestCase):
    def test_repeated_t0_c1_changes_only_mri_channels_for_batched_input(self) -> None:
        image, geometry, condition = _numpy_inputs(batched=True)
        originals = image.copy(), geometry.copy(), condition.copy()

        result = repeated_t0_mri_only(image, geometry, condition)

        self.assertEqual(result.image.shape, image.shape)
        np.testing.assert_array_equal(result.image[:, 1, :7], image[:, 0, :7])
        np.testing.assert_array_equal(result.image[:, 2, :7], image[:, 0, :7])
        np.testing.assert_array_equal(result.image[:, 1:3, 7], image[:, 1:3, 7])
        np.testing.assert_array_equal(result.image[:, (0, 3)], image[:, (0, 3)])
        np.testing.assert_array_equal(result.geometry, geometry)
        np.testing.assert_array_equal(result.condition, condition)
        _assert_independent_numpy(self, result, image, geometry, condition)

        result.image.fill(-1)
        result.geometry.fill(-1)
        result.condition.fill(-1)
        np.testing.assert_array_equal(image, originals[0])
        np.testing.assert_array_equal(geometry, originals[1])
        np.testing.assert_array_equal(condition, originals[2])

    def test_repeated_t0_c2_copies_all_channels_and_geometry(self) -> None:
        image, geometry, condition = _numpy_inputs()
        result = repeated_t0_full_image_derived(image, geometry, condition)

        np.testing.assert_array_equal(result.image[1], image[0])
        np.testing.assert_array_equal(result.image[2], image[0])
        np.testing.assert_array_equal(result.geometry[1], geometry[0])
        np.testing.assert_array_equal(result.geometry[2], geometry[0])
        np.testing.assert_array_equal(result.image[(0, 3), :], image[(0, 3), :])
        np.testing.assert_array_equal(result.geometry[(0, 3), :], geometry[(0, 3), :])
        np.testing.assert_array_equal(result.condition, condition)
        _assert_independent_numpy(self, result, image, geometry, condition)

    def test_temporal_swap_moves_complete_image_and_geometry_only(self) -> None:
        image, geometry, condition = _numpy_inputs()
        condition_before = condition.copy()
        result = swap_t1_t2(image, geometry, condition)

        np.testing.assert_array_equal(result.image[0], image[0])
        np.testing.assert_array_equal(result.image[1], image[2])
        np.testing.assert_array_equal(result.image[2], image[1])
        np.testing.assert_array_equal(result.image[3], image[3])
        np.testing.assert_array_equal(result.geometry[1], geometry[2])
        np.testing.assert_array_equal(result.geometry[2], geometry[1])
        np.testing.assert_array_equal(result.condition, condition_before)
        np.testing.assert_array_equal(condition, condition_before)
        _assert_independent_numpy(self, result, image, geometry, condition)

    def test_donor_swap_keeps_recipient_t0_t3_and_condition(self) -> None:
        recipient_image, recipient_geometry, recipient_condition = _numpy_inputs(batched=True)
        donor_image, donor_geometry, _ = _numpy_inputs(batched=True, offset=10_000.0)
        recipient_snapshots = (
            recipient_image.copy(),
            recipient_geometry.copy(),
            recipient_condition.copy(),
        )
        donor_snapshots = donor_image.copy(), donor_geometry.copy()

        result = replace_followups_with_donor(
            recipient_image,
            recipient_geometry,
            recipient_condition,
            donor_image=donor_image,
            donor_geometry=donor_geometry,
        )

        np.testing.assert_array_equal(result.image[:, 0], recipient_image[:, 0])
        np.testing.assert_array_equal(result.image[:, 1:3], donor_image[:, 1:3])
        np.testing.assert_array_equal(result.image[:, 3], recipient_image[:, 3])
        np.testing.assert_array_equal(result.geometry[:, 0], recipient_geometry[:, 0])
        np.testing.assert_array_equal(result.geometry[:, 1:3], donor_geometry[:, 1:3])
        np.testing.assert_array_equal(result.geometry[:, 3], recipient_geometry[:, 3])
        np.testing.assert_array_equal(result.condition, recipient_condition)
        _assert_independent_numpy(
            self, result, recipient_image, recipient_geometry, recipient_condition
        )
        self.assertFalse(np.shares_memory(result.image, donor_image))
        self.assertFalse(np.shares_memory(result.geometry, donor_geometry))

        np.testing.assert_array_equal(recipient_image, recipient_snapshots[0])
        np.testing.assert_array_equal(recipient_geometry, recipient_snapshots[1])
        np.testing.assert_array_equal(recipient_condition, recipient_snapshots[2])
        np.testing.assert_array_equal(donor_image, donor_snapshots[0])
        np.testing.assert_array_equal(donor_geometry, donor_snapshots[1])

    def test_invalid_contracts_fail_before_copying(self) -> None:
        image, geometry, condition = _numpy_inputs()
        with self.assertRaisesRegex(ValueError, "visit/channel"):
            repeated_t0_mri_only(image[:, :7], geometry, condition)
        with self.assertRaisesRegex(ValueError, "geometry"):
            repeated_t0_mri_only(image, geometry[:, :8], condition)
        with self.assertRaisesRegex(ValueError, "condition"):
            repeated_t0_mri_only(image, geometry, condition[:2])
        with self.assertRaisesRegex(ValueError, "形状完全一致"):
            replace_followups_with_donor(
                image,
                geometry,
                condition,
                donor_image=np.zeros((4, 8, 3, 3, 2), dtype=np.float32),
                donor_geometry=geometry,
            )


@unittest.skipIf(torch is None, "需要 torch")
class TorchPerturbationAndTargetTests(unittest.TestCase):
    def _torch_inputs(self):
        image, geometry, condition = _numpy_inputs(batched=True)
        return tuple(torch.from_numpy(value.copy()) for value in (image, geometry, condition))

    def test_torch_c1_is_clone_safe_and_preserves_dtype_device(self) -> None:
        image, geometry, condition = self._torch_inputs()
        snapshots = image.clone(), geometry.clone(), condition.clone()
        result = repeated_t0_mri_only(image, geometry, condition)

        self.assertEqual(result.image.dtype, image.dtype)
        self.assertEqual(result.image.device, image.device)
        torch.testing.assert_close(result.image[:, 1, :7], image[:, 0, :7])
        torch.testing.assert_close(result.image[:, 2, :7], image[:, 0, :7])
        torch.testing.assert_close(result.image[:, 1:3, 7], image[:, 1:3, 7])
        torch.testing.assert_close(result.geometry, geometry)
        torch.testing.assert_close(result.condition, condition)
        self.assertNotEqual(result.image.data_ptr(), image.data_ptr())
        self.assertNotEqual(result.geometry.data_ptr(), geometry.data_ptr())
        self.assertNotEqual(result.condition.data_ptr(), condition.data_ptr())

        result.image.zero_()
        result.geometry.zero_()
        result.condition.zero_()
        torch.testing.assert_close(image, snapshots[0])
        torch.testing.assert_close(geometry, snapshots[1])
        torch.testing.assert_close(condition, snapshots[2])

    def test_mixed_numpy_torch_is_rejected(self) -> None:
        image, geometry, condition = _numpy_inputs()
        with self.assertRaisesRegex(TypeError, "同一后端"):
            swap_t1_t2(torch.from_numpy(image), geometry, torch.from_numpy(condition))

    def test_prediction_uses_perturbed_context_and_native_target(self) -> None:
        class Response:
            def __init__(self, latent_correction):
                self.latent_correction = latent_correction

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))

            @staticmethod
            def _state(image, geometry):
                appearance = image[:, :, 0, 0, 0, 0]
                roi = image[:, :, 7, 0, 0, 0]
                return torch.stack((appearance, roi + geometry[:, :, 0]), dim=-1)

            def encode_targets(self, image, geometry):
                # 与 online state 故意相同，便于精确检查 target 来源。
                return self._state(image, geometry)

            def encode_visits(self, image, geometry):
                return self._state(image, geometry)

            def image_transition(self, state, condition):
                return state + condition[..., :2]

            def response_transition(self, geometry, condition):
                # 显式依赖传入的 context geometry。
                return Response(geometry[..., :2] * 0.25)

        native_image, native_geometry, condition = self._torch_inputs()
        swapped = swap_t1_t2(native_image, native_geometry, condition)
        snapshots = (
            native_image.clone(),
            native_geometry.clone(),
            swapped.image.clone(),
            swapped.geometry.clone(),
            condition.clone(),
        )
        model = FakeModel().eval()

        result = predict_perturbed_context_against_native_target(
            model,
            native_image=native_image,
            native_geometry=native_geometry,
            perturbed_image=swapped.image,
            perturbed_geometry=swapped.geometry,
            condition=condition,
        )

        expected_target = model.encode_targets(native_image, native_geometry)[:, 1:]
        swapped_target = model.encode_targets(swapped.image, swapped.geometry)[:, 1:]
        expected_state = model.encode_visits(swapped.image, swapped.geometry)[:, :-1]
        expected_image_prediction = expected_state + condition[..., :2]
        expected_correction = swapped.geometry[:, :-1, :2] * 0.25
        torch.testing.assert_close(result.native_target, expected_target)
        self.assertFalse(torch.equal(result.native_target, swapped_target))
        torch.testing.assert_close(result.image_prediction, expected_image_prediction)
        torch.testing.assert_close(result.response_correction, expected_correction)
        torch.testing.assert_close(
            result.prediction, expected_image_prediction + expected_correction
        )
        self.assertIs(result.target, result.native_target)
        self.assertFalse(result.prediction.requires_grad)
        self.assertFalse(model.training)

        for actual, expected in zip(
            (native_image, native_geometry, swapped.image, swapped.geometry, condition), snapshots
        ):
            torch.testing.assert_close(actual, expected)

    def test_prediction_supports_unbatched_inputs(self) -> None:
        class Response:
            def __init__(self, latent_correction):
                self.latent_correction = latent_correction

        class FakeModel(torch.nn.Module):
            def encode_targets(self, image, geometry):
                return geometry[..., :2]

            def encode_visits(self, image, geometry):
                return geometry[..., :2]

            def image_transition(self, state, condition):
                return state + condition[..., :2]

            def response_transition(self, geometry, condition):
                return Response(torch.zeros_like(geometry[..., :2]))

        image_np, geometry_np, condition_np = _numpy_inputs()
        image, geometry, condition = (
            torch.from_numpy(image_np),
            torch.from_numpy(geometry_np),
            torch.from_numpy(condition_np),
        )
        result = predict_perturbed_context_against_native_target(
            FakeModel().eval(),
            native_image=image,
            native_geometry=geometry,
            perturbed_image=image,
            perturbed_geometry=geometry,
            condition=condition,
        )
        self.assertEqual(result.prediction.shape, (3, 2))
        self.assertEqual(result.native_target.shape, (3, 2))

    def test_prediction_rejects_train_mode_without_changing_it(self) -> None:
        class MinimalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout()

            def encode_targets(self, image, geometry):  # pragma: no cover - 应在调用前失败
                raise AssertionError

            def encode_visits(self, image, geometry):  # pragma: no cover
                raise AssertionError

            def image_transition(self, state, condition):  # pragma: no cover
                raise AssertionError

            def response_transition(self, geometry, condition):  # pragma: no cover
                raise AssertionError

        image, geometry, condition = self._torch_inputs()
        model = MinimalModel().train()
        with self.assertRaisesRegex(ValueError, "eval"):
            predict_perturbed_context_against_native_target(
                model,
                native_image=image,
                native_geometry=geometry,
                perturbed_image=image,
                perturbed_geometry=geometry,
                condition=condition,
            )
        self.assertTrue(model.training)
        self.assertTrue(model.dropout.training)


if __name__ == "__main__":
    unittest.main()
