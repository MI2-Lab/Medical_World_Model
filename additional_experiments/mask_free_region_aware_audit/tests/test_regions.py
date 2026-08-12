from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mask_free_regions", ROOT / "scripts" / "regions.py")
assert SPEC is not None and SPEC.loader is not None
regions = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = regions
SPEC.loader.exec_module(regions)


class RegionGeometryTests(unittest.TestCase):
    def test_runtime_shape_and_fractional_volume_are_exact(self) -> None:
        shape = regions.expected_feature_shape()
        self.assertEqual(shape, (14, 22, 20))
        built = regions.build_region_weights(shape, dtype=torch.float64)
        cell_volume = (8 * 2.0) * (8 * 0.9) * (8 * 0.9)
        expected = {
            "R0": 64**3,
            "R1": 32**3,
            "R2": 48**3 - 32**3,
            "R3": 64**3 - 48**3,
            "S1": 24**3,
            "S2": 40**3 - 24**3,
            "S3": 64**3 - 40**3,
        }
        for name, volume in expected.items():
            self.assertAlmostEqual(
                float(built[name].sum()) * cell_volume, volume, places=8
            )
            self.assertGreater(int(torch.count_nonzero((built[name] > 0) & (built[name] < 1))), 0)
        torch.testing.assert_close(
            built["R1"] + built["R2"] + built["R3"], built["R0"], rtol=0, atol=1e-14
        )
        torch.testing.assert_close(
            built["S1"] + built["S2"] + built["S3"], built["R0"], rtol=0, atol=1e-14
        )

    def test_wrong_runtime_shape_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime feature shape"):
            regions.build_region_weights((14, 22, 19))

    def test_r0_is_bitwise_equal_to_frozen_c1b_local_sidecar_member(self) -> None:
        config = json.loads((ROOT / "configs" / "audit.json").read_text(encoding="utf-8"))
        built = regions.build_region_weights(regions.expected_feature_shape())
        # Materialize only the fixed LOCAL member; no mask-derived member is read.
        with np.load(config["paths"]["spatial_sidecar"], allow_pickle=False) as archive:
            reference = np.asarray(archive["c1b_local_weight_final"])
        self.assertTrue(np.array_equal(built["R0"].numpy()[0, 0], reference))

    def test_feature_variants_and_fixed_projection(self) -> None:
        shape = regions.expected_feature_shape()
        weights = regions.build_region_weights(shape)
        generator = torch.Generator().manual_seed(17)
        spatial = torch.randn((3, 128, *shape), generator=generator)
        first = regions.extract_region_features(spatial, weights)
        second = regions.extract_region_features(spatial, weights)
        self.assertEqual(tuple(first), regions.REGION_MEAN_KEYS)
        for name, dimension in regions.REGION_DIMENSIONS.items():
            self.assertEqual(tuple(first[name].shape), (3, dimension))
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        torch.testing.assert_close(first["R4"], torch.cat((first["R1"], first["R2"]), 1))
        torch.testing.assert_close(
            first["R5"], torch.cat((first["R1"], first["R2"], first["R3"]), 1)
        )
        q = regions.fixed_qr_projection()
        self.assertEqual(q.dtype, np.float32)
        np.testing.assert_allclose(q.T @ q, np.eye(192), rtol=0, atol=8e-7)

    def test_public_contract_has_no_patient_fields(self) -> None:
        weights = regions.build_region_weights(regions.expected_feature_shape())
        contract = regions.geometry_contract(weights)
        self.assertEqual(contract["status"], "GEOMETRY_VALID")
        self.assertFalse(contract["contains_patient_data"])
        self.assertLess(contract["maximum_physical_volume_error_mm3"], 1e-3)
        self.assertLess(contract["primary_partition_max_abs"], 1e-6)


if __name__ == "__main__":
    unittest.main()
