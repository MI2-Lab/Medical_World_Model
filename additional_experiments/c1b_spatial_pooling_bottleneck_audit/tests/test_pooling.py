from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.pooling import (  # noqa: E402
    FINAL_STAGE_GEOMETRY,
    S3_STAGE_GEOMETRY,
    apply_frozen_response_projection,
    concatenate_local_global,
    expected_feature_shape,
    fixed_physical_local_weights,
    global_average_pool,
    receptive_field_occupancy,
    weighted_average_pool,
)


class GeometryContractTests(unittest.TestCase):
    def test_preregistered_final_and_s3_contracts(self) -> None:
        self.assertEqual(FINAL_STAGE_GEOMETRY.receptive_field_zyx, (47, 47, 47))
        self.assertEqual(FINAL_STAGE_GEOMETRY.stride_zyx, (8, 8, 8))
        self.assertEqual(FINAL_STAGE_GEOMETRY.padding_zyx, (23, 23, 23))
        self.assertEqual(S3_STAGE_GEOMETRY.receptive_field_zyx, (23, 23, 23))
        self.assertEqual(S3_STAGE_GEOMETRY.stride_zyx, (4, 4, 4))
        self.assertEqual(S3_STAGE_GEOMETRY.padding_zyx, (11, 11, 11))

        self.assertEqual(
            expected_feature_shape((112, 176, 160), stage="final"),
            (14, 22, 20),
        )
        self.assertEqual(
            expected_feature_shape((32, 96, 96), stage="final"),
            (4, 12, 12),
        )
        self.assertEqual(
            expected_feature_shape((112, 176, 160), stage="s3"),
            (28, 44, 40),
        )

    def test_unregistered_or_malformed_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown/unregistered"):
            expected_feature_shape((16, 16, 16), stage="exploratory")
        for malformed in ((0, 16, 16), (16, 16), (16.0, 16, 16)):
            with self.subTest(shape=malformed), self.assertRaises(ValueError):
                expected_feature_shape(malformed)  # type: ignore[arg-type]


class ReceptiveFieldOccupancyTests(unittest.TestCase):
    def test_final_occupancy_is_exact_avg_pool_with_external_zeros(self) -> None:
        mask = torch.ones(1, 1, 48, 48, 48)
        occupancy = receptive_field_occupancy(mask, (6, 6, 6), stage="final")
        self.assertEqual(tuple(occupancy.shape), (1, 1, 6, 6, 6))
        expected_corner = torch.tensor((24.0 / 47.0) ** 3)
        torch.testing.assert_close(occupancy[0, 0, 0, 0, 0], expected_corner)
        torch.testing.assert_close(occupancy[0, 0, 3, 3, 3], torch.tensor(1.0))
        self.assertTrue(bool(((occupancy >= 0) & (occupancy <= 1)).all()))

    def test_s3_occupancy_uses_23_4_11_contract(self) -> None:
        mask = torch.ones(1, 1, 24, 24, 24, dtype=torch.float64)
        occupancy = receptive_field_occupancy(mask, (6, 6, 6), stage="s3")
        expected_corner = torch.tensor((12.0 / 23.0) ** 3, dtype=torch.float64)
        torch.testing.assert_close(occupancy[0, 0, 0, 0, 0], expected_corner)
        torch.testing.assert_close(
            occupancy,
            torch.nn.functional.avg_pool3d(
                mask,
                kernel_size=23,
                stride=4,
                padding=11,
                count_include_pad=True,
            ),
        )

    def test_occupancy_rejects_bad_shape_range_and_empty_support(self) -> None:
        valid = torch.ones(2, 1, 16, 16, 16)
        with self.assertRaisesRegex(ValueError, "convolution geometry"):
            receptive_field_occupancy(valid, (3, 2, 2))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            receptive_field_occupancy(torch.zeros_like(valid), (2, 2, 2))
        invalid_range = valid.clone()
        invalid_range[0, 0, 0, 0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "\[0,1\]"):
            receptive_field_occupancy(invalid_range, (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "\[N,1,Z,Y,X\]"):
            receptive_field_occupancy(valid[:, 0], (2, 2, 2))


class PoolingTests(unittest.TestCase):
    def test_p0_and_all_one_weighted_pool_are_identical(self) -> None:
        generator = torch.Generator().manual_seed(7)
        spatial = torch.randn(3, 5, 2, 3, 4, generator=generator)
        expected = spatial.mean(dim=(-3, -2, -1))
        torch.testing.assert_close(global_average_pool(spatial), expected)
        weights = torch.ones(1, 1, 2, 3, 4)
        torch.testing.assert_close(weighted_average_pool(spatial, weights), expected)

    def test_weighted_pool_selects_only_declared_support(self) -> None:
        spatial = torch.arange(48, dtype=torch.float32).reshape(2, 2, 2, 2, 3)
        weights = torch.zeros(2, 1, 2, 2, 3)
        weights[0, 0, 1, 0, 2] = 1.0
        weights[1, 0, 0, 1, 1] = 0.25
        pooled = weighted_average_pool(spatial, weights)
        torch.testing.assert_close(pooled[0], spatial[0, :, 1, 0, 2])
        torch.testing.assert_close(pooled[1], spatial[1, :, 0, 1, 1])

    def test_weighted_pool_fails_on_empty_out_of_range_or_shape_mismatch(self) -> None:
        spatial = torch.ones(2, 3, 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            weighted_average_pool(spatial, torch.zeros(2, 1, 2, 2, 2))
        too_large = torch.ones(2, 1, 2, 2, 2)
        too_large[0, 0, 0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "\[0,1\]"):
            weighted_average_pool(spatial, too_large)
        with self.assertRaisesRegex(ValueError, "feature grids"):
            weighted_average_pool(spatial, torch.ones(2, 1, 2, 2, 1))
        with self.assertRaisesRegex(ValueError, "only finite"):
            bad_spatial = spatial.clone()
            bad_spatial[0, 0, 0, 0, 0] = float("nan")
            weighted_average_pool(bad_spatial, torch.ones(2, 1, 2, 2, 2))


class FixedPhysicalLocalTests(unittest.TestCase):
    def test_c1b_final_weights_are_fractional_physical_cell_overlap(self) -> None:
        weights = fixed_physical_local_weights(
            (112, 176, 160),
            (14, 22, 20),
            (0.9, 0.9, 2.0),
            stage="final",
            dtype=torch.float64,
        )
        self.assertEqual(tuple(weights.shape), (1, 1, 14, 22, 20))
        self.assertTrue(bool(((weights >= 0) & (weights <= 1)).all()))
        self.assertTrue(bool(((weights > 0) & (weights < 1)).any()))
        sampling_cell_volume = (8.0 * 0.9) * (8.0 * 0.9) * (8.0 * 2.0)
        expected_sum = 64.0**3 / sampling_cell_volume
        torch.testing.assert_close(
            weights.sum(), torch.tensor(expected_sum, dtype=torch.float64)
        )

        constant = torch.arange(1, 5, dtype=torch.float64).reshape(1, 4, 1, 1, 1)
        constant = constant.expand(2, 4, 14, 22, 20)
        pooled = weighted_average_pool(constant, weights)
        torch.testing.assert_close(
            pooled,
            torch.arange(1, 5, dtype=torch.float64).expand(2, 4),
        )

    def test_s3_uses_four_voxel_sampling_cells_not_receptive_fields(self) -> None:
        weights = fixed_physical_local_weights(
            (112, 176, 160),
            (28, 44, 40),
            (0.9, 0.9, 2.0),
            stage="s3",
            dtype=torch.float64,
        )
        sampling_cell_volume = (4.0 * 0.9) * (4.0 * 0.9) * (4.0 * 2.0)
        torch.testing.assert_close(
            weights.sum(),
            torch.tensor(64.0**3 / sampling_cell_volume, dtype=torch.float64),
        )

    def test_visit_specific_spacing_and_validation(self) -> None:
        weights = fixed_physical_local_weights(
            (32, 96, 96),
            (4, 12, 12),
            torch.tensor([[1.0, 1.0, 2.0], [0.8, 1.2, 3.0]]),
        )
        self.assertEqual(tuple(weights.shape), (2, 1, 4, 12, 12))
        self.assertFalse(torch.equal(weights[0], weights[1]))
        with self.assertRaisesRegex(ValueError, "convolution geometry"):
            fixed_physical_local_weights(
                (112, 176, 160), (13, 22, 20), (0.9, 0.9, 2.0)
            )
        with self.assertRaisesRegex(ValueError, "finite positive"):
            fixed_physical_local_weights(
                (112, 176, 160), (14, 22, 20), (0.9, 0.0, 2.0)
            )


class FrozenProjectionTests(unittest.TestCase):
    @staticmethod
    def _projection() -> nn.Module:
        torch.manual_seed(19)
        return nn.Sequential(nn.Linear(128, 192), nn.LayerNorm(192)).eval()

    def test_projection_is_inference_only_and_lg_is_direct_concat(self) -> None:
        projection = self._projection()
        pooled = torch.randn(3, 128, requires_grad=True)
        with torch.no_grad():
            expected = projection(pooled)
        response = apply_frozen_response_projection(pooled, projection)
        torch.testing.assert_close(response, expected)
        self.assertEqual(tuple(response.shape), (3, 192))
        self.assertFalse(response.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in projection.parameters()))

        local = response + 1.0
        combined = concatenate_local_global(local, response)
        self.assertEqual(tuple(combined.shape), (3, 384))
        torch.testing.assert_close(combined[:, :192], local)
        torch.testing.assert_close(combined[:, 192:], response)

    def test_projection_and_concat_shape_range_checks_fail_closed(self) -> None:
        projection = self._projection()
        projection.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            apply_frozen_response_projection(torch.ones(2, 128), projection)
        projection.eval()
        with self.assertRaisesRegex(ValueError, "\[N,128\]"):
            apply_frozen_response_projection(torch.ones(2, 127), projection)
        wrong_output = nn.Linear(128, 191).eval()
        with self.assertRaisesRegex(ValueError, "192-D"):
            apply_frozen_response_projection(torch.ones(2, 128), wrong_output)

        valid = torch.ones(2, 192)
        with self.assertRaisesRegex(ValueError, "match exactly"):
            concatenate_local_global(valid, torch.ones(3, 192))
        invalid = valid.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            concatenate_local_global(invalid, valid)


if __name__ == "__main__":
    unittest.main()
