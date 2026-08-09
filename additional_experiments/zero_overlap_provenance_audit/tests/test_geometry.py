from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from zero_overlap_audit.geometry import (  # noqa: E402
    OrientedBox,
    cardinal_grid_valid_voxel_count,
    intersection_volume,
    minimum_cardinal_translation_for_count,
    minimum_cardinal_translation_for_fraction,
    minimum_distance,
    orientation_angle_deg,
    pairwise_metrics,
)


def _rotation_z(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _affine_for_box(
    center: tuple[float, float, float],
    *,
    axes: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    shape: tuple[int, int, int] = (2, 2, 2),
) -> np.ndarray:
    directions = np.eye(3, dtype=np.float64) if axes is None else np.asarray(axes)
    linear = directions @ np.diag(np.asarray(spacing, dtype=np.float64))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = linear
    affine[:3, 3] = np.asarray(center) - linear @ (
        0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
    )
    return affine


class OrientedBoxTests(unittest.TestCase):
    def test_affine_shape_uses_complete_outer_voxel_footprint(self) -> None:
        axes = _rotation_z(30.0)
        shape = (2, 4, 6)
        spacing = (2.0, 3.0, 4.0)
        center = (11.0, -7.0, 5.5)
        affine = _affine_for_box(
            center, axes=axes, spacing=spacing, shape=shape
        )
        box = OrientedBox.from_affine_shape(affine, shape)

        np.testing.assert_allclose(box.center, center, atol=1.0e-12)
        np.testing.assert_allclose(box.axes, axes, atol=1.0e-12)
        np.testing.assert_allclose(box.half_lengths, (2.0, 6.0, 12.0))
        np.testing.assert_allclose(box.fov_lengths, (4.0, 12.0, 24.0))
        self.assertAlmostEqual(box.volume, 4.0 * 12.0 * 24.0)

        corners = box.corners()
        self.assertEqual(corners.shape, (8, 3))
        local = (corners - box.center) @ box.axes
        np.testing.assert_allclose(
            np.abs(local), np.broadcast_to(box.half_lengths, local.shape), atol=1.0e-12
        )
        np.testing.assert_allclose(box.aabb_min, corners.min(axis=0))
        np.testing.assert_allclose(box.aabb_max, corners.max(axis=0))
        # All result objects can be serialized without a custom encoder.
        json.dumps(box.to_dict(), allow_nan=False)

    def test_sheared_or_invalid_affines_fail_closed(self) -> None:
        shear = np.eye(4, dtype=np.float64)
        shear[0, 1] = 0.2
        with self.assertRaisesRegex(ValueError, "shear|non-orthogonal"):
            OrientedBox.from_affine_shape(shear, (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "positive"):
            OrientedBox.from_affine_shape(np.eye(4), (2, 0, 2))
        with self.assertRaisesRegex(ValueError, "finite integers"):
            OrientedBox.from_affine_shape(np.eye(4), (2, 2.5, 2))

    def test_exact_plane_vertex_intersection_and_pairwise_metrics(self) -> None:
        box_a = OrientedBox.from_affine_shape(
            _affine_for_box((0.0, 0.0, 0.0)), (2, 2, 2)
        )
        box_b = OrientedBox.from_affine_shape(
            _affine_for_box((1.0, 0.0, 0.0)), (2, 2, 2)
        )
        self.assertAlmostEqual(intersection_volume(box_a, box_b), 4.0, places=11)
        self.assertEqual(minimum_distance(box_a, box_b), 0.0)

        metrics = pairwise_metrics(box_a, box_b)
        self.assertAlmostEqual(metrics.center_displacement_mm, 1.0)
        self.assertAlmostEqual(metrics.orientation_angle_deg, 0.0)
        self.assertAlmostEqual(metrics.minimum_separation_mm, 0.0)
        self.assertAlmostEqual(metrics.intersection_mm3, 4.0, places=11)
        self.assertAlmostEqual(metrics.overlap_fraction_a, 0.5, places=11)
        self.assertAlmostEqual(metrics.overlap_fraction_b, 0.5, places=11)
        self.assertAlmostEqual(metrics.iou, 1.0 / 3.0, places=11)
        self.assertAlmostEqual(metrics.aabb_intersection_mm3, 4.0, places=11)
        self.assertTrue(metrics.aabb_intersects)
        payload = metrics.to_dict()
        self.assertEqual(payload["oriented_intersection_mm3"], 4.0)
        self.assertEqual(payload["overlap_fraction_first"], 0.5)
        self.assertEqual(payload["overlap_fraction_second"], 0.5)
        json.dumps(payload, allow_nan=False)

    def test_rotated_cube_intersection_is_polyhedrally_exact(self) -> None:
        axis_aligned = OrientedBox.from_affine_shape(
            _affine_for_box((0.0, 0.0, 0.0)), (2, 2, 2)
        )
        rotated = OrientedBox.from_affine_shape(
            _affine_for_box((0.0, 0.0, 0.0), axes=_rotation_z(45.0)),
            (2, 2, 2),
        )
        # The Z extrusion is 2 and the intersection of the two side-2 squares
        # is 8*(sqrt(2)-1), hence 16*(sqrt(2)-1) in three dimensions.
        expected = 16.0 * (math.sqrt(2.0) - 1.0)
        self.assertAlmostEqual(
            intersection_volume(axis_aligned, rotated), expected, places=10
        )
        self.assertAlmostEqual(
            orientation_angle_deg(axis_aligned, rotated), 45.0, places=10
        )

    def test_distance_contacts_and_orientation_axis_sign_invariance(self) -> None:
        box = OrientedBox.from_affine_shape(
            _affine_for_box((0.0, 0.0, 0.0)), (2, 2, 2)
        )
        separated = OrientedBox.from_affine_shape(
            _affine_for_box((4.0, 0.0, 0.0)), (2, 2, 2)
        )
        touching = OrientedBox.from_affine_shape(
            _affine_for_box((2.0, 0.0, 0.0)), (2, 2, 2)
        )
        self.assertAlmostEqual(minimum_distance(box, separated), 2.0, places=10)
        self.assertEqual(intersection_volume(box, separated), 0.0)
        self.assertEqual(minimum_distance(box, touching), 0.0)
        self.assertEqual(intersection_volume(box, touching), 0.0)
        self.assertTrue(pairwise_metrics(box, touching).aabb_intersects)

        sign_reversed = OrientedBox(
            center=box.center,
            axes=box.axes @ np.diag((-1.0, -1.0, 1.0)),
            half_lengths=box.half_lengths,
        )
        self.assertEqual(orientation_angle_deg(box, sign_reversed), 0.0)
        thirty_degrees = OrientedBox.from_affine_shape(
            _affine_for_box((0.0, 0.0, 0.0), axes=_rotation_z(30.0)),
            (2, 2, 2),
        )
        self.assertAlmostEqual(
            orientation_angle_deg(box, thirty_degrees), 30.0, places=10
        )


class CardinalTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target_affine = np.eye(4, dtype=np.float64)
        self.target_shape = (5, 4, 3)  # 60 target voxel centres.
        # Footprint: X [9.5,11.5], Y [0.5,2.5], Z [-0.5,2.5].
        source_affine = np.eye(4, dtype=np.float64)
        source_affine[:3, 3] = (10.0, 1.0, 0.0)
        self.source = OrientedBox.from_affine_shape(source_affine, (2, 2, 3))

    def test_exact_count_and_minimum_translation_for_one_or_many_voxels(self) -> None:
        self.assertEqual(
            cardinal_grid_valid_voxel_count(
                self.target_affine, self.target_shape, self.source
            ),
            0,
        )
        one = minimum_cardinal_translation_for_count(
            self.target_affine, self.target_shape, self.source, 1
        )
        self.assertTrue(one.attainable)
        self.assertAlmostEqual(one.translation_magnitude_mm, 5.5)
        # The nearest X centre combines with all currently covered Y/Z centres.
        self.assertEqual(one.achieved_valid_voxels, 6)
        self.assertEqual(one.maximum_attainable_valid_voxels, 27)
        self.assertAlmostEqual(one.maximum_attainable_valid_fraction, 27.0 / 60.0)

        twelve = minimum_cardinal_translation_for_count(
            self.target_affine, self.target_shape, self.source, 12
        )
        self.assertTrue(twelve.attainable)
        self.assertAlmostEqual(twelve.translation_magnitude_mm, 6.5)
        self.assertEqual(twelve.achieved_valid_voxels, 12)

    def test_fraction_wrapper_and_unattainable_threshold_are_explicit(self) -> None:
        twenty_percent = minimum_cardinal_translation_for_fraction(
            self.target_affine, self.target_shape, self.source, 0.20
        )
        self.assertEqual(twenty_percent.required_valid_voxels, 12)
        self.assertAlmostEqual(twenty_percent.translation_magnitude_mm, 6.5)

        half = minimum_cardinal_translation_for_fraction(
            self.target_affine, self.target_shape, self.source, 0.50
        )
        self.assertFalse(half.attainable)
        self.assertEqual(half.required_valid_voxels, 30)
        self.assertEqual(half.maximum_attainable_valid_voxels, 27)
        self.assertIsNone(half.translation_magnitude_mm)
        self.assertIsNone(half.achieved_valid_voxels)
        self.assertIsNone(half.achieved_valid_fraction)
        json.dumps(half.to_dict(), allow_nan=False)

    def test_boundary_inclusion_and_signed_axis_permutations_are_exact(self) -> None:
        # A unit-wide source footprint [0,1] contains target centres on both
        # closed boundaries, reproducing the pipeline's inclusive mask rule.
        boundary_source_affine = np.eye(4, dtype=np.float64)
        boundary_source_affine[:3, 3] = (0.5, 0.0, 0.0)
        boundary_source = OrientedBox.from_affine_shape(
            boundary_source_affine, (1, 1, 1)
        )
        self.assertEqual(
            cardinal_grid_valid_voxel_count(
                np.eye(4), (2, 1, 1), boundary_source
            ),
            2,
        )

        target_shape = (3, 4, 5)
        target_box = OrientedBox.from_affine_shape(np.eye(4), target_shape)
        permuted = OrientedBox(
            center=target_box.center,
            axes=np.asarray(
                ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
            ),
            half_lengths=np.asarray((2.0, 1.5, 2.5)),
        )
        self.assertEqual(
            cardinal_grid_valid_voxel_count(np.eye(4), target_shape, permuted),
            math.prod(target_shape),
        )

    def test_noncardinal_source_fails_closed(self) -> None:
        rotated = OrientedBox.from_affine_shape(
            _affine_for_box((1.0, 1.0, 1.0), axes=_rotation_z(10.0)),
            (2, 2, 2),
        )
        with self.assertRaisesRegex(ValueError, "non-cardinal"):
            cardinal_grid_valid_voxel_count(np.eye(4), (3, 3, 3), rotated)
        with self.assertRaisesRegex(ValueError, "non-cardinal"):
            minimum_cardinal_translation_for_count(
                np.eye(4), (3, 3, 3), rotated, 1
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
