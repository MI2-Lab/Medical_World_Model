from __future__ import annotations

from pathlib import Path
import stat
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


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

from c1b_sanity.geometry import (  # noqa: E402
    PhysicalGrid,
    audit_support_containment,
    canonical_volume_sha256,
    canonicalize_to_ras,
)
from c1b_spatial_audit.sidecars import (  # noqa: E402
    AuditSidecars,
    C1B_FINAL_SHAPE_ZYX,
    C1B_INPUT_SHAPE_ZYX,
    LEGACY_FINAL_SHAPE_ZYX,
    LEGACY_PORACLE_STATUS,
    LEGACY_PVALID_STATUS,
    NUISANCE_COLUMNS,
    OCCUPANCY_COLUMNS,
    SIDECAR_KEYS,
    assign_occupancy_quartiles,
    nuisance_row,
    validate_reconstructed_support,
    validate_valid_source_mask,
    write_private_sidecars,
)


class ValidSourceTests(unittest.TestCase):
    def test_authoritative_mask_counts_are_exact_and_fail_closed(self) -> None:
        mask = np.zeros((4, 1, *C1B_INPUT_SHAPE_ZYX), dtype=np.uint8)
        for visit in range(4):
            mask[visit, 0, 0, 0, : visit + 1] = 1
        counts = validate_valid_source_mask(
            mask,
            expected_counts=(1, 2, 3, 4),
            expected_target_voxels=(np.prod(C1B_INPUT_SHAPE_ZYX),) * 4,
        )
        np.testing.assert_array_equal(counts, (1, 2, 3, 4))

        with self.assertRaisesRegex(ValueError, "count disagrees"):
            validate_valid_source_mask(
                mask,
                expected_counts=(1, 2, 3, 5),
                expected_target_voxels=(np.prod(C1B_INPUT_SHAPE_ZYX),) * 4,
            )
        invalid = mask.copy()
        invalid[0, 0, 0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "not binary"):
            validate_valid_source_mask(
                invalid,
                expected_counts=(1, 2, 3, 4),
                expected_target_voxels=(np.prod(C1B_INPUT_SHAPE_ZYX),) * 4,
            )


class NuisanceTests(unittest.TestCase):
    def test_geometry_targets_follow_the_frozen_outcome_free_definitions(self) -> None:
        row = pd.Series(
            {
                "source_shape_xyz_json": "[100, 50, 10]",
                "source_affine_ras_json": (
                    "[[0.5,0,0,7],[0,1,0,8],[0,0,2,9],[0,0,0,1]]"
                ),
            }
        )
        result = nuisance_row(
            "PRIVATE-ID", "T2", row, 25, 100, (1.0, 2.0, 4.0)
        )
        self.assertEqual(tuple(result), NUISANCE_COLUMNS)
        self.assertEqual(result["padding_fraction"], 0.75)
        self.assertEqual(result["valid_source_fraction"], 0.25)
        self.assertEqual(result["native_spacing_x_mm"], 0.5)
        self.assertEqual(result["acquisition_fov_x_mm"], 50.0)
        self.assertEqual(result["acquisition_fov_y_mm"], 50.0)
        self.assertEqual(result["acquisition_fov_z_mm"], 20.0)
        self.assertEqual(result["max_resample_factor"], 4.0)
        self.assertEqual(result["resize_anisotropy"], 4.0)
        self.assertFalse(any("target" in name for name in result))

    def test_pooled_qcut_is_frozen_and_nonunique_boundaries_stop(self) -> None:
        frame = pd.DataFrame(
            {
                "lesion_occupancy": np.arange(1.0, 9.0),
                "patient_id": [f"P{index}" for index in range(8)],
            }
        )
        observed = assign_occupancy_quartiles(frame)
        self.assertEqual(
            observed["occupancy_quartile"].value_counts().sort_index().to_dict(),
            {"Q1": 2, "Q2": 2, "Q3": 2, "Q4": 2},
        )
        with self.assertRaisesRegex(ValueError, "not unique"):
            assign_occupancy_quartiles(
                pd.DataFrame({"lesion_occupancy": np.ones(8)})
            )


class SupportReconstructionTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[2, 2, 2] = 1.0
        support = canonicalize_to_ras(data, np.eye(4))
        grid = PhysicalGrid(
            shape_zyx=(5, 5, 5),
            spacing_xyz_mm=(1.0, 1.0, 1.0),
            center_ras_mm=(2.0, 2.0, 2.0),
        )
        audit = audit_support_containment(support, grid)
        sampled = np.zeros((5, 5, 5), dtype=bool)
        sampled[2, 2, 2] = True
        cache = {
            "support_available": np.ones(4, dtype=np.uint8),
            "support_canonical_sha256": np.asarray(
                [canonical_volume_sha256(support)] * 4
            ),
            "support_source_positive_voxels": np.ones(4, dtype=np.int64),
            "support_retained_positive_voxels": np.ones(4, dtype=np.int64),
            "support_nn_target_positive_voxels": np.ones(4, dtype=np.int64),
            "support_retained_positive_voxel_fraction": np.ones(
                4, dtype=np.float32
            ),
            "support_physical_volume_retention": np.ones(4, dtype=np.float32),
            "support_exact_full_support_containment": np.ones(
                4, dtype=np.uint8
            ),
            "support_source_boundary_touch": np.zeros(4, dtype=np.uint8),
            "support_target_boundary_touch": np.zeros(4, dtype=np.uint8),
            "support_minimum_margin_mm": np.asarray(
                [audit.minimum_margin_mm] * 4, dtype=np.float32
            ),
            "support_source_volume_mm3": np.asarray(
                [audit.full_physical_volume_mm3] * 4, dtype=np.float32
            ),
            "support_retained_source_volume_mm3": np.asarray(
                [audit.retained_physical_volume_mm3] * 4, dtype=np.float32
            ),
        }
        reference = pd.Series(
            {
                "source_positive_voxels": audit.full_positive_voxels,
                "retained_positive_voxels": audit.retained_positive_voxels,
                "full_physical_volume_mm3": audit.full_physical_volume_mm3,
                "retained_physical_volume_mm3": audit.retained_physical_volume_mm3,
                "physical_volume_retention": audit.physical_volume_retention,
                "exact_full_support_containment": (
                    audit.exact_full_support_containment
                ),
                "source_boundary_touch": audit.source_boundary_touch,
                "target_boundary_touch": audit.target_boundary_touch,
                "minimum_margin_mm": audit.minimum_margin_mm,
            }
        )
        return support, sampled, audit, cache, reference

    def test_hash_count_and_volume_must_all_match(self) -> None:
        support, sampled, audit, cache, reference = self._fixture()
        validate_reconstructed_support(
            support=support,
            sampled_support_zyx=sampled,
            audit=audit,
            cache=cache,
            visit_index=0,
            reference_row=reference,
        )
        bad_hash = {name: value.copy() for name, value in cache.items()}
        bad_hash["support_canonical_sha256"][0] = "0" * 64
        with self.assertRaisesRegex(ValueError, "canonical support hash"):
            validate_reconstructed_support(
                support=support,
                sampled_support_zyx=sampled,
                audit=audit,
                cache=bad_hash,
                visit_index=0,
                reference_row=reference,
            )
        bad_volume = {name: value.copy() for name, value in cache.items()}
        bad_volume["support_source_volume_mm3"][0] += np.float32(1.0)
        with self.assertRaisesRegex(ValueError, "source physical volume"):
            validate_reconstructed_support(
                support=support,
                sampled_support_zyx=sampled,
                audit=audit,
                cache=bad_volume,
                visit_index=0,
                reference_row=reference,
            )


class PrivateOutputTests(unittest.TestCase):
    @staticmethod
    def _bundle() -> AuditSidecars:
        patients = np.asarray(("P0", "P1"))
        valid = np.ones((2, 4, *C1B_FINAL_SHAPE_ZYX), dtype=np.float32)
        oracle = np.zeros_like(valid)
        oracle_valid = np.zeros((2, 4), dtype=bool)
        local = np.ones(C1B_FINAL_SHAPE_ZYX, dtype=np.float32)
        legacy = np.ones((2, 4, *LEGACY_FINAL_SHAPE_ZYX), dtype=np.float32)
        nuisance = pd.DataFrame(
            [
                {
                    name: (
                        patient
                        if name == "patient_id"
                        else visit
                        if name == "visit"
                        else 0.5
                    )
                    for name in NUISANCE_COLUMNS
                }
                for patient in patients
                for visit in ("T0", "T1", "T2", "T3")
            ],
            columns=NUISANCE_COLUMNS,
        )
        occupancy = pd.DataFrame(
            [
                {
                    name: (
                        "P0"
                        if name == "patient_id"
                        else "T0"
                        if name == "visit"
                        else "Q1"
                        if name == "occupancy_quartile"
                        else 1.0
                    )
                    for name in OCCUPANCY_COLUMNS
                }
            ],
            columns=OCCUPANCY_COLUMNS,
        )
        return AuditSidecars(
            patient_id=patients,
            c1b_valid_weight_final=valid,
            c1b_oracle_weight_final=oracle,
            c1b_oracle_valid=oracle_valid,
            c1b_local_weight_final=local,
            legacy_local_weight_final=legacy,
            nuisance=nuisance,
            occupancy=occupancy,
        )

    def test_exact_npz_schema_owner_only_and_no_legacy_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = write_private_sidecars(
                self._bundle(),
                sidecar_output=root / "audit_sidecars.private.npz",
                nuisance_output=root / "nuisance.private.csv",
                occupancy_output=root / "occupancy.private.csv",
            )
            for output in outputs:
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with np.load(outputs[0], allow_pickle=False) as archive:
                self.assertEqual(tuple(archive.files), SIDECAR_KEYS)
                self.assertEqual(archive["c1b_valid_weight_final"].dtype, np.float32)
                self.assertEqual(archive["c1b_oracle_valid"].dtype, np.bool_)
                self.assertNotIn("legacy_valid_weight_final", archive.files)
                self.assertNotIn("legacy_oracle_weight_final", archive.files)
            with self.assertRaises(FileExistsError):
                write_private_sidecars(
                    self._bundle(),
                    sidecar_output=outputs[0],
                    nuisance_output=outputs[1],
                    occupancy_output=outputs[2],
                )

    def test_legacy_unavailable_statuses_are_explicit(self) -> None:
        self.assertEqual(
            LEGACY_PVALID_STATUS, "NA_no_source_authoritative_mask"
        )
        self.assertEqual(
            LEGACY_PORACLE_STATUS,
            "NA_incomplete_source_authoritative_support_1488_of_1500",
        )

    def test_formal_cli_cannot_skip_cache_archive_hashes(self) -> None:
        source = (ROOT / "scripts/build_audit_sidecars.py").read_text(encoding="utf-8")
        self.assertIn("verify_cache_archive_sha256=True", source)
        self.assertNotIn("skip-cache", source)
        self.assertNotIn("legacy_valid_weight_final", source)
        self.assertNotIn("legacy_oracle_weight_final", source)


if __name__ == "__main__":
    unittest.main()
