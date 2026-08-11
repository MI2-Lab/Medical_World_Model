from __future__ import annotations

import inspect
import unittest

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from c1b_sanity.geometry import CanonicalVolume
from c1b_sanity.registration import (
    RegistrationFailureCode,
    canonical_volume_to_sitk,
    ras_matrix_to_sitk_transform,
    register_precontrast_rigid,
    register_t1_t2_t3_to_t0,
    sitk_image_to_affine_ras,
    sitk_transform_to_ras_matrix,
)


def _canonical_volume(data: np.ndarray, affine_ras: np.ndarray) -> CanonicalVolume:
    return CanonicalVolume(
        data=np.asarray(data, dtype=np.float32),
        affine_ras=np.asarray(affine_ras, dtype=np.float64),
        original_axcodes=("R", "A", "S"),
        orientation_transform=np.asarray(((0, 1), (1, 1), (2, 1)), dtype=np.float64),
    )


def _asymmetric_phantom() -> tuple[CanonicalVolume, np.ndarray]:
    shape_xyz = np.asarray((64, 56, 48), dtype=int)
    spacing_xyz = np.asarray((1.4, 1.6, 1.8), dtype=np.float64)
    center_ras = np.asarray((12.0, -25.0, 31.0), dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(spacing_xyz)
    affine[:3, 3] = center_ras - 0.5 * (shape_xyz - 1.0) * spacing_xyz

    indices = np.indices(tuple(shape_xyz), dtype=np.float64)
    world = np.stack(
        tuple(affine[axis, axis] * indices[axis] + affine[axis, 3] for axis in range(3)),
        axis=-1,
    )
    relative = world - center_ras
    body = (
        (relative[..., 0] / 31.0) ** 2
        + (relative[..., 1] / 27.0) ** 2
        + (relative[..., 2] / 24.0) ** 2
        < 1.0
    )
    phantom = np.zeros(tuple(shape_xyz), dtype=np.float32)
    phantom[body] = (
        35.0
        + 8.0 * relative[..., 0][body] / 31.0
        + 5.0 * relative[..., 2][body] / 24.0
    )
    # Unequal, off-axis structures make reflections and symmetric local optima
    # observable while remaining connected through the broad anatomy.
    for location, amplitude, sigma in (
        ((-13.0, -7.0, 5.0), 65.0, (5.0, 7.0, 4.0)),
        ((12.0, 10.0, -7.0), 95.0, (4.0, 5.0, 6.0)),
        ((4.0, -13.0, -10.0), 45.0, (7.0, 3.0, 4.0)),
    ):
        delta = relative - np.asarray(location, dtype=np.float64)
        exponent = sum((delta[..., axis] / sigma[axis]) ** 2 for axis in range(3))
        phantom += np.asarray(amplitude * np.exp(-0.5 * exponent), dtype=np.float32)
    phantom[~body] = 0.0
    return _canonical_volume(phantom, affine), center_ras


def _known_source_to_anchor(center_ras: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_euler("xyz", (3.0, -4.0, 5.0), degrees=True).as_matrix()
    center_translation = np.asarray((3.0, -2.5, 1.8), dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center_ras + center_translation - rotation @ center_ras
    return transform


def _make_source(
    anchor: CanonicalVolume,
    source_to_anchor_ras: np.ndarray,
) -> CanonicalVolume:
    anchor_image = canonical_volume_to_sitk(anchor)
    source_image = sitk.Resample(
        anchor_image,
        anchor_image,
        ras_matrix_to_sitk_transform(source_to_anchor_ras),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    source_array = np.transpose(sitk.GetArrayFromImage(source_image), (2, 1, 0))
    return _canonical_volume(source_array, anchor.affine_ras)


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.anchor, cls.center_ras = _asymmetric_phantom()
        cls.known_transform = _known_source_to_anchor(cls.center_ras)
        cls.source = _make_source(cls.anchor, cls.known_transform)

    def test_ras_lps_image_and_transform_coordinate_roundtrips(self) -> None:
        # Include a proper oblique direction so the test is not satisfied by an
        # axis-aligned sign flip alone.
        oblique = Rotation.from_euler("xyz", (4.0, -6.0, 8.0), degrees=True).as_matrix()
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = oblique @ np.diag((0.9, 1.3, 2.1))
        affine[:3, 3] = (21.0, -17.0, 33.0)
        volume = _canonical_volume(np.arange(8 * 7 * 6).reshape(8, 7, 6), affine)
        image = canonical_volume_to_sitk(volume)
        np.testing.assert_allclose(sitk_image_to_affine_ras(image), affine, atol=1.0e-12)

        matrix = self.known_transform
        sitk_transform = ras_matrix_to_sitk_transform(matrix)
        recovered = sitk_transform_to_ras_matrix(sitk_transform)
        np.testing.assert_allclose(recovered, matrix, atol=1.0e-12)

        point_ras = np.asarray((17.5, -22.0, 29.25), dtype=np.float64)
        point_lps = np.asarray((-point_ras[0], -point_ras[1], point_ras[2]))
        mapped_lps = np.asarray(sitk_transform.TransformPoint(tuple(point_lps)))
        mapped_ras_from_sitk = np.asarray((-mapped_lps[0], -mapped_lps[1], mapped_lps[2]))
        mapped_ras_direct = matrix[:3, :3] @ point_ras + matrix[:3, 3]
        np.testing.assert_allclose(mapped_ras_from_sitk, mapped_ras_direct, atol=1.0e-12)

    def test_known_rigid_motion_is_recovered_without_reflection(self) -> None:
        result = register_precontrast_rigid(self.anchor, self.source)
        self.assertTrue(result.success, result.failure_message)
        self.assertEqual(result.failure_code, RegistrationFailureCode.NONE)
        self.assertTrue(result.converged)
        self.assertIsNotNone(result.source_to_anchor_ras)
        recovered = result.source_to_anchor_ras
        self.assertGreater(np.linalg.det(recovered[:3, :3]), 0.99999)

        rotation_error_deg = np.degrees(
            Rotation.from_matrix(
                recovered[:3, :3] @ self.known_transform[:3, :3].T
            ).magnitude()
        )
        center_homogeneous = np.append(self.center_ras, 1.0)
        center_error_mm = np.linalg.norm(
            (recovered @ center_homogeneous - self.known_transform @ center_homogeneous)[:3]
        )
        self.assertLess(rotation_error_deg, 1.0)
        self.assertLess(center_error_mm, 1.0)
        self.assertGreater(result.similarity_after, result.similarity_before + 0.20)
        self.assertGreater(result.sidecars.anatomy_dice_after, result.sidecars.anatomy_dice_before)
        self.assertAlmostEqual(
            result.sidecars.padding_fraction_after
            + result.sidecars.valid_overlap_fraction_after,
            1.0,
            places=12,
        )

    def test_fixed_seed_is_deterministic(self) -> None:
        first = register_precontrast_rigid(self.anchor, self.source)
        second = register_precontrast_rigid(self.anchor, self.source)
        self.assertTrue(first.success, first.failure_message)
        self.assertTrue(second.success, second.failure_message)
        np.testing.assert_allclose(
            first.source_to_anchor_ras,
            second.source_to_anchor_ras,
            atol=1.0e-10,
            rtol=0.0,
        )
        self.assertEqual(first.optimizer_iterations, second.optimizer_iterations)
        self.assertAlmostEqual(first.final_mattes_mi, second.final_mattes_mi, places=13)

    def test_reflection_is_rejected(self) -> None:
        reflection = np.eye(4, dtype=np.float64)
        reflection[0, 0] = -1.0
        with self.assertRaisesRegex(RuntimeError, "reflection"):
            ras_matrix_to_sitk_transform(reflection)

    def test_constant_input_fails_closed(self) -> None:
        constant = _canonical_volume(
            np.ones(self.anchor.shape_xyz, dtype=np.float32),
            self.anchor.affine_ras,
        )
        source_failure = register_precontrast_rigid(self.anchor, constant)
        self.assertFalse(source_failure.success)
        self.assertEqual(source_failure.failure_code, RegistrationFailureCode.CONSTANT_SOURCE)
        self.assertIsNone(source_failure.source_to_anchor_ras)

        anchor_failure = register_precontrast_rigid(constant, self.source)
        self.assertFalse(anchor_failure.success)
        self.assertEqual(anchor_failure.failure_code, RegistrationFailureCode.CONSTANT_ANCHOR)
        self.assertIsNone(anchor_failure.source_to_anchor_ras)

    def test_public_registration_api_has_no_outcome_bearing_inputs(self) -> None:
        allowed = {"anchor_t0", "source", "config"}
        self.assertEqual(set(inspect.signature(register_precontrast_rigid).parameters), allowed)
        longitudinal_allowed = {"t0", "t1", "t2", "t3", "config"}
        self.assertEqual(set(inspect.signature(register_t1_t2_t3_to_t0).parameters), longitudinal_allowed)
        prohibited_tokens = ("mask", "ftv", "clinical", "response", "outcome", "label")
        for parameter in allowed | longitudinal_allowed:
            self.assertFalse(any(token in parameter.lower() for token in prohibited_tokens))


if __name__ == "__main__":
    unittest.main()
