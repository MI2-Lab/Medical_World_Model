from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    ornt_transform,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from c1b_sanity.dce7 import (  # noqa: E402
    DCE7_CHANNEL_NAMES,
    build_visit_dce7,
    construct_dce7_xyzt,
    normalize_dce7,
    resample_dce_to_grid,
    select_phase_indices,
)
from c1b_sanity.builder import VisitInput, build_patient_dce7  # noqa: E402
from c1b_sanity.cache import load_and_validate_cache, write_cache_atomic  # noqa: E402
from c1b_sanity.geometry import (  # noqa: E402
    C1B_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    PhysicalGrid,
    audit_support_containment,
    canonical_volume_sha256,
    canonicalize_to_ras,
    load_nifti_ras,
    make_c1b_grid,
    resample_support_nearest,
    support_bbox_center_ras,
    support_centroid_ras,
    validate_source_to_anchor_transform,
    voxel_to_world,
)


def _canonical_dce(shape_xyz: tuple[int, int, int] = (4, 5, 6), phases: int = 6):
    coordinates = np.indices(shape_xyz, dtype=np.float32)
    base = coordinates[0] + 10.0 * coordinates[1] + 100.0 * coordinates[2]
    data = np.stack([base + 7.0 * phase for phase in range(phases)], axis=-1)
    affine = np.diag((1.0, 1.0, 1.0, 1.0)).astype(np.float64)
    return data, affine


class GeometryTests(unittest.TestCase):
    def test_canonical_source_hash_closes_voxel_and_affine_provenance(self) -> None:
        data, affine = _canonical_dce()
        original = canonicalize_to_ras(data, affine)
        repeat = canonicalize_to_ras(data.copy(), affine.copy())
        self.assertEqual(
            canonical_volume_sha256(original), canonical_volume_sha256(repeat)
        )
        changed_data = data.copy()
        changed_data[0, 0, 0, 0] += 1.0
        changed_voxel = canonicalize_to_ras(changed_data, affine)
        self.assertNotEqual(
            canonical_volume_sha256(original), canonical_volume_sha256(changed_voxel)
        )
        changed_affine = affine.copy()
        changed_affine[0, 3] += 0.125
        changed_geometry = canonicalize_to_ras(data, changed_affine)
        self.assertNotEqual(
            canonical_volume_sha256(original), canonical_volume_sha256(changed_geometry)
        )

    def test_loader_applies_affine_permutations_and_flips_to_array(self) -> None:
        canonical, canonical_affine = _canonical_dce()
        ras = axcodes2ornt(("R", "A", "S"))
        with tempfile.TemporaryDirectory() as directory:
            for index, source_codes in enumerate(
                (("P", "R", "I"), ("L", "S", "P"), ("S", "L", "A"))
            ):
                with self.subTest(source_codes=source_codes):
                    transform = ornt_transform(ras, axcodes2ornt(source_codes))
                    source_data = apply_orientation(canonical, transform)
                    source_affine = canonical_affine @ inv_ornt_aff(
                        transform, canonical.shape[:3]
                    )
                    path = Path(directory) / f"orientation_{index}.nii.gz"
                    nib.save(nib.Nifti1Image(source_data, source_affine), path)
                    loaded = load_nifti_ras(path)
                    np.testing.assert_array_equal(loaded.data, canonical)
                    np.testing.assert_allclose(loaded.affine_ras, canonical_affine)
                    self.assertEqual(loaded.data.shape[-1], canonical.shape[-1])
                    # A distinctive voxel retains its physical coordinate, not
                    # merely its displayed axis labels.
                    point = np.asarray((2.0, 3.0, 4.0))
                    np.testing.assert_allclose(
                        voxel_to_world(loaded.affine_ras, point),
                        voxel_to_world(canonical_affine, point),
                    )

    def test_nonfinite_data_and_unusable_affines_are_rejected(self) -> None:
        data = np.ones((2, 3, 4), dtype=np.float32)
        bad_finite = np.eye(4)
        bad_finite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            canonicalize_to_ras(data, bad_finite)
        singular = np.eye(4)
        singular[2, 2] = 0.0
        with self.assertRaisesRegex(ValueError, "singular"):
            canonicalize_to_ras(data, singular)
        data[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            canonicalize_to_ras(data, np.eye(4))

    def test_frozen_grid_has_exact_shape_spacing_fov_and_physical_center(self) -> None:
        center = (12.25, -8.5, 41.0)
        grid = make_c1b_grid(center)
        self.assertEqual(grid.shape_zyx, C1B_SHAPE_ZYX)
        self.assertEqual(grid.spacing_xyz_mm, C1B_SPACING_XYZ_MM)
        np.testing.assert_allclose(
            grid.voxel_footprint_fov_xyz_mm, (144.0, 158.4, 224.0)
        )
        center_voxel = 0.5 * (np.asarray(grid.shape_xyz) - 1.0)
        np.testing.assert_allclose(
            voxel_to_world(grid.affine_ras, center_voxel), center
        )

    def test_registration_hook_is_only_a_sampling_transform(self) -> None:
        data, affine = _canonical_dce()
        volume = canonicalize_to_ras(data, affine)
        # The source-to-anchor translation moves source world X by +10 mm.
        transform = np.eye(4)
        transform[0, 3] = 10.0
        grid = PhysicalGrid(
            shape_zyx=(6, 5, 4),
            spacing_xyz_mm=(1.0, 1.0, 1.0),
            center_ras_mm=(11.5, 2.0, 2.5),
        )
        sampled, valid, audit = resample_dce_to_grid(
            volume, grid, source_to_anchor_ras=transform
        )
        np.testing.assert_allclose(sampled, data, atol=1e-6)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(audit.input_from_output, np.eye(4), atol=1e-12)
        scale = np.diag((1.01, 1.0, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "rigid"):
            validate_source_to_anchor_transform(scale)

    def test_localization_support_uses_nearest_neighbour_sidecar(self) -> None:
        support = np.zeros((4, 5, 6), dtype=np.float32)
        support[1, 2, 3] = 1.0
        volume = canonicalize_to_ras(support, np.eye(4))
        grid = PhysicalGrid((6, 5, 4), (1.0, 1.0, 1.0), (1.5, 2.0, 2.5))
        sampled = resample_support_nearest(volume, grid)
        self.assertEqual(sampled.dtype, np.bool_)
        self.assertEqual(int(sampled.sum()), 1)
        self.assertTrue(sampled[3, 2, 1])

    def test_t0_anchor_is_physical_bbox_center_not_foreground_centroid(self) -> None:
        support = np.zeros((5, 6, 7), dtype=np.float32)
        support[0, 1:5, 2] = 1.0
        support[4, 5, 6] = 1.0
        affine = np.asarray(
            (
                (2.0, 0.0, 0.0, 10.0),
                (0.0, 3.0, 0.0, -4.0),
                (0.0, 0.0, 4.0, 7.0),
                (0, 0, 0, 1),
            ),
            dtype=np.float64,
        )
        volume = canonicalize_to_ras(support, affine)
        # Inclusive bbox midpoint is voxel (2,3,4), hence RAS (14,5,23).
        np.testing.assert_allclose(support_bbox_center_ras(volume), (14.0, 5.0, 23.0))
        self.assertFalse(
            np.allclose(support_bbox_center_ras(volume), support_centroid_ras(volume))
        )

    def test_source_domain_full_footprint_containment_and_cut(self) -> None:
        grid = PhysicalGrid((4, 4, 4), (1.0, 1.0, 1.0), (1.5, 1.5, 1.5))

        contained = np.zeros((6, 4, 4), dtype=np.float32)
        contained[2, 2, 2] = 1.0
        contained[3, 2, 2] = 1.0  # footprint ends exactly at target x=3.5
        contained_audit = audit_support_containment(
            canonicalize_to_ras(contained, np.eye(4)), grid
        )
        self.assertTrue(contained_audit.exact_full_support_containment)
        self.assertEqual(contained_audit.retained_positive_voxels, 2)
        self.assertEqual(contained_audit.physical_volume_retention, 1.0)
        self.assertTrue(contained_audit.target_boundary_touch)
        self.assertFalse(contained_audit.source_boundary_touch)

        cut = contained.copy()
        cut[3, 2, 2] = 0.0
        cut[4, 2, 2] = 1.0  # full footprint extends beyond target high face
        cut_audit = audit_support_containment(canonicalize_to_ras(cut, np.eye(4)), grid)
        self.assertFalse(cut_audit.exact_full_support_containment)
        self.assertEqual(cut_audit.retained_positive_voxels, 1)
        self.assertEqual(cut_audit.full_positive_voxels, 2)
        self.assertEqual(cut_audit.retained_positive_voxel_fraction, 0.5)
        self.assertEqual(cut_audit.physical_volume_retention, 0.5)
        self.assertLess(cut_audit.minimum_margin_mm, 0.0)

        source_edge = np.zeros((6, 4, 4), dtype=np.float32)
        source_edge[5, 1, 1] = 1.0
        source_edge_audit = audit_support_containment(
            canonicalize_to_ras(source_edge, np.eye(4)), grid
        )
        self.assertTrue(source_edge_audit.source_boundary_touch)


class DCE7Tests(unittest.TestCase):
    def test_complete_provenance_changes_for_phase_anchor_and_strategy(self) -> None:
        data, affine = _canonical_dce()
        volume = canonicalize_to_ras(data, affine)
        mask = np.zeros(volume.shape_xyz, dtype=np.float32)
        mask[1:3, 1:4, 2:4] = 1.0
        support = canonicalize_to_ras(mask, affine)

        def inputs(metadata, *, translated=False, t0_support=support):
            result = {}
            for visit in ("T0", "T1", "T2", "T3"):
                transform = None
                if translated and visit == "T1":
                    transform = np.eye(4, dtype=np.float64)
                    transform[0, 3] = 0.25
                result[visit] = VisitInput(
                    visit=visit,
                    dce=volume,
                    phase_metadata=metadata,
                    localization_support=t0_support if visit == "T0" else None,
                    source_to_anchor_ras=transform,
                )
            return result

        def small_grid(center):
            return PhysicalGrid((6, 5, 4), (1.0, 1.0, 1.0), tuple(center))

        with mock.patch("c1b_sanity.builder.make_c1b_grid", side_effect=small_grid):
            baseline = build_patient_dce7(
                "SYNTHETIC", inputs({"pre": 0, "post_early": 2, "post_late": 5})
            )
            changed_phase = build_patient_dce7(
                "SYNTHETIC", inputs({"pre": 1, "post_early": 3, "post_late": 4})
            )
            shifted_mask = np.zeros(volume.shape_xyz, dtype=np.float32)
            shifted_mask[0:2, 0:3, 1:3] = 1.0
            changed_anchor = build_patient_dce7(
                "SYNTHETIC",
                inputs(
                    {"pre": 0, "post_early": 2, "post_late": 5},
                    t0_support=canonicalize_to_ras(shifted_mask, affine),
                ),
            )
            changed_strategy = build_patient_dce7(
                "SYNTHETIC",
                inputs(
                    {"pre": 0, "post_early": 2, "post_late": 5},
                    translated=True,
                ),
                registration_strategy="C1B-R",
            )
        digests = {
            baseline.input_provenance_sha256,
            changed_phase.input_provenance_sha256,
            changed_anchor.input_provenance_sha256,
            changed_strategy.input_provenance_sha256,
        }
        self.assertEqual(len(digests), 4)

    def test_legacy_phase_policy_and_outcome_fields(self) -> None:
        long_selection = select_phase_indices(
            6,
            {
                "pre": 1.2,
                "post_early": 3.1,
                "post_late": 99,
                "pCR": 1,
                "FTV": 1234,
                "LD": 88,
            },
        )
        self.assertEqual(long_selection.indices, (1, 3, 5))
        self.assertEqual(long_selection.peak_window, (1, 2, 3, 4, 5))
        # For <=4 phases, early and late metadata are deliberately ignored.
        short_selection = select_phase_indices(
            4, {"pre": 2, "post_early": 0, "post_late": 0, "pCR": 0}
        )
        self.assertEqual(short_selection.indices, (2, 3, 3))
        single = select_phase_indices(1, {"pre": 500})
        self.assertEqual(single.indices, (0, 0, 0))
        self.assertEqual(single.peak_window, (0,))

    def test_exact_seven_channel_formulas_and_order(self) -> None:
        dce = np.asarray(
            [
                [[[2.0, 3.0, 8.0, 5.0, 1.0, 6.0]]],
                [[[-2.0, -1.0, 4.0, 0.0, 2.0, 1.0]]],
            ],
            dtype=np.float32,
        )
        channels, selection = construct_dce7_xyzt(
            dce, phase_metadata={"pre": 0, "post_early": 2, "post_late": 5}
        )
        self.assertEqual(selection.indices, (0, 2, 5))
        self.assertEqual(channels.shape, (7, 2, 1, 1))
        # First voxel: peak=8, denom=2.
        np.testing.assert_allclose(channels[:, 0, 0, 0], (2, 8, 6, 6, 4, 3, -1))
        # Second voxel: peak=4, denom=2.
        np.testing.assert_allclose(channels[:, 1, 0, 0], (-2, 4, 1, 6, 3, 3, -1.5))
        self.assertEqual(
            DCE7_CHANNEL_NAMES,
            (
                "pre",
                "early",
                "late",
                "early_minus_pre",
                "late_minus_pre",
                "peak_relative_enhancement",
                "late_minus_peak_relative_enhancement",
            ),
        )

    def test_normalization_statistics_use_only_valid_source(self) -> None:
        base = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
        channels = np.repeat(base[None], 7, axis=0)
        channels[:, 0, 0, 5] = 1000.0
        valid = np.ones((1, 1, 6), dtype=bool)
        valid[0, 0, 5] = False
        normalized, stats = normalize_dce7(channels, valid)
        np.testing.assert_allclose(stats.p01, np.full(7, 0.04), atol=1e-6)
        np.testing.assert_allclose(stats.p99, np.full(7, 3.96), atol=1e-6)
        np.testing.assert_allclose(stats.median, np.full(7, 2.0), atol=1e-6)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertLessEqual(float(np.max(normalized)), 5.0)
        self.assertGreaterEqual(float(np.min(normalized)), -5.0)

    def test_visit_tensor_is_dce7_only_and_sidecars_stay_separate(self) -> None:
        data, affine = _canonical_dce()
        volume = canonicalize_to_ras(data, affine)
        grid = PhysicalGrid((6, 5, 4), (1.0, 1.0, 1.0), (1.5, 2.0, 2.5))
        result = build_visit_dce7(
            volume,
            grid,
            phase_metadata={
                "pre": 0,
                "post_early": 2,
                "post_late": 5,
                "mask": np.ones((4, 5, 6)),
                "spacing": (99, 99, 99),
                "pCR": 1,
            },
        )
        self.assertEqual(result.tensor_czyx.shape, (7, 6, 5, 4))
        self.assertEqual(result.tensor_czyx.dtype, np.float32)
        self.assertEqual(result.valid_source_mask_zyx.shape, (6, 5, 4))
        self.assertTrue(result.valid_source_mask_zyx.all())
        self.assertNotIn("mask", DCE7_CHANNEL_NAMES)
        self.assertNotIn("spacing", DCE7_CHANNEL_NAMES)

    def test_antialias_is_enabled_for_more_than_1p5x_downsampling(self) -> None:
        data, _ = _canonical_dce(shape_xyz=(8, 10, 12))
        source_affine = np.diag((0.5, 0.5, 0.5, 1.0))
        volume = canonicalize_to_ras(data, source_affine)
        grid = PhysicalGrid((6, 5, 4), (1.0, 1.0, 1.0), (1.75, 2.25, 2.75))
        _, _, audit = resample_dce_to_grid(volume, grid)
        self.assertTrue(audit.anti_alias_applied)
        np.testing.assert_allclose(
            audit.source_samples_per_output_axis, (2.0, 2.0, 2.0)
        )
        self.assertTrue(np.all(audit.anti_alias_sigma_source_voxels > 0))

    def test_reflect_padding_is_nonsentinel_with_separate_valid_mask(self) -> None:
        data, affine = _canonical_dce(shape_xyz=(3, 3, 3))
        volume = canonicalize_to_ras(data, affine)
        grid = PhysicalGrid((5, 5, 5), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        sampled, valid_zyx, audit = resample_dce_to_grid(volume, grid)
        self.assertEqual(audit.padding_mode, "reflect")
        valid_xyz = valid_zyx.transpose(2, 1, 0)
        self.assertGreater(int(np.count_nonzero(~valid_xyz)), 0)
        padded_pre = sampled[..., 0][~valid_xyz]
        self.assertTrue(np.isfinite(padded_pre).all())
        self.assertGreater(np.unique(padded_pre).size, 1)

    def test_patient_builder_freezes_one_t0_grid_before_followups(self) -> None:
        data, affine = _canonical_dce()
        volume = canonicalize_to_ras(data, affine)
        t0_mask = np.zeros(volume.shape_xyz, dtype=np.float32)
        t0_mask[1:3, 1:4, 2:4] = 1.0
        t0_support = canonicalize_to_ras(t0_mask, affine)
        future_mask = np.zeros(volume.shape_xyz, dtype=np.float32)
        future_mask[0, 0, 0] = 1.0
        future_support = canonicalize_to_ras(future_mask, affine)
        visit_inputs = {
            visit: VisitInput(
                visit=visit,
                dce=volume,
                phase_metadata={"pre": 0, "post_early": 2, "post_late": 5},
                localization_support=t0_support if visit == "T0" else future_support,
            )
            for visit in ("T0", "T1", "T2", "T3")
        }

        def small_grid(center):
            return PhysicalGrid((6, 5, 4), (1.0, 1.0, 1.0), tuple(center))

        with mock.patch(
            "c1b_sanity.builder.make_c1b_grid", side_effect=small_grid
        ) as factory:
            patient = build_patient_dce7(
                "SYNTHETIC", visit_inputs, formal_ftv_overlap=True
            )
        factory.assert_called_once_with((1.5, 2.0, 2.5))
        self.assertEqual(
            patient.anchor_provenance, "released_t0_localization_support_bbox_center"
        )
        self.assertEqual(patient.grid.center_ras_mm, (1.5, 2.0, 2.5))
        self.assertEqual(patient.image.shape, (4, 7, 6, 5, 4))
        self.assertEqual(patient.valid_source_mask.shape, (4, 1, 6, 5, 4))
        # Follow-up supports are observed for QC (including source-edge status)
        # but cannot recenter the already-created grid.
        self.assertTrue(patient.support_qc[1].source_boundary_touch)
        base_only_inputs = {
            visit: VisitInput(
                visit=visit,
                dce=volume,
                phase_metadata={"pre": 0, "post_early": 2, "post_late": 5},
                localization_support=t0_support if visit == "T0" else None,
            )
            for visit in ("T0", "T1", "T2", "T3")
        }
        with mock.patch(
            "c1b_sanity.builder.make_c1b_grid", side_effect=small_grid
        ):
            base_only = build_patient_dce7("SYNTHETIC", base_only_inputs)
        np.testing.assert_array_equal(patient.image, base_only.image)
        np.testing.assert_array_equal(
            patient.valid_source_mask, base_only.valid_source_mask
        )
        np.testing.assert_array_equal(patient.grid.affine_ras, base_only.grid.affine_ras)
        self.assertEqual(
            tuple(base_only.support_scope),
            ("anchor_and_qc", "not_loaded", "not_loaded", "not_loaded"),
        )
        self.assertNotEqual(
            patient.input_provenance_sha256,
            base_only.input_provenance_sha256,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "patient.npz"
            write_cache_atomic(cache_path, patient, require_frozen_grid=False)
            cached, _ = load_and_validate_cache(
                cache_path,
                require_frozen_grid=False,
                expected_image=patient.image,
            )
            self.assertEqual(
                int(cached["support_exact_full_support_containment"][0]), 1
            )
            self.assertLessEqual(
                float(np.nanmax(cached["support_physical_volume_retention"])), 1.0
            )


if __name__ == "__main__":
    unittest.main()
