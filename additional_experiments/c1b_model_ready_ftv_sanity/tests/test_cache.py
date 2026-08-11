from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from c1b_sanity.builder import (  # noqa: E402
    BUILDER_CONTRACT_VERSION,
    VISITS,
    builder_contract_sha256,
    input_provenance_sha256,
)
from c1b_sanity.cache import (  # noqa: E402
    CONTENT_HASH_KEY,
    load_and_validate_cache,
    load_model_tensor,
    validate_cache_arrays,
    write_cache_atomic,
)
from c1b_sanity.dce7 import DCE7_CHANNEL_NAMES  # noqa: E402
from c1b_sanity.geometry import PhysicalGrid  # noqa: E402


def _synthetic_payload() -> dict[str, np.ndarray]:
    grid = PhysicalGrid(
        shape_zyx=(2, 3, 4),
        spacing_xyz_mm=(0.75, 1.25, 2.5),
        center_ras_mm=(11.0, -7.0, 3.5),
    )
    rng = np.random.default_rng(2026)
    image = rng.normal(size=(4, 7, *grid.shape_zyx)).astype(np.float32)
    identity = np.eye(4, dtype=np.float64)
    source_hashes = np.asarray(("0" * 64,) * 4)
    support_hashes = np.asarray(("1" * 64, "NONE", "NONE", "NONE"))
    support_scope = np.asarray(
        ("anchor_and_qc", "not_loaded", "not_loaded", "not_loaded")
    )
    phase_metadata_hashes = np.asarray(("2" * 64,) * 4)
    phase_indices = np.asarray(((0, 2, 5),) * 4, dtype=np.int16)
    phase_counts = np.full(4, 6, dtype=np.int16)
    transforms = np.repeat(identity[None], 4, axis=0)
    contract_digest = builder_contract_sha256()
    provenance_digest = input_provenance_sha256(
        patient_id="SYNTHETIC_001",
        cohort="I-SPY2",
        formal_ftv_overlap=False,
        registration_strategy="C1B-H",
        source_hashes=source_hashes,
        support_hashes=support_hashes,
        support_scope=support_scope,
        phase_metadata_hashes=phase_metadata_hashes,
        phase_counts=phase_counts,
        phase_indices=phase_indices,
        source_to_anchor_ras=transforms,
        grid=grid,
        anchor_provenance="released_t0_localization_support_bbox_center",
        contract_sha256=contract_digest,
    )
    return {
        "schema_version": np.asarray(3, dtype=np.int16),
        "patient_id": np.asarray("SYNTHETIC_001"),
        "cohort": np.asarray("I-SPY2"),
        "formal_ftv_overlap": np.asarray(0, dtype=np.uint8),
        "registration_strategy": np.asarray("C1B-H"),
        "image": image,
        "valid_source_mask": np.ones((4, 1, *grid.shape_zyx), dtype=np.uint8),
        "phase_indices": phase_indices,
        "phase_counts": phase_counts,
        "channel_names": np.asarray(DCE7_CHANNEL_NAMES),
        "visits": np.asarray(VISITS),
        "grid_affine_ras": grid.affine_ras,
        "grid_center_ras_mm": np.asarray(grid.center_ras_mm, dtype=np.float64),
        "grid_shape_zyx": np.asarray(grid.shape_zyx, dtype=np.int16),
        "grid_spacing_xyz_mm": np.asarray(grid.spacing_xyz_mm, dtype=np.float64),
        "anchor_provenance": np.asarray("released_t0_localization_support_bbox_center"),
        "normalization_p01": np.zeros((4, 7), dtype=np.float32),
        "normalization_p99": np.ones((4, 7), dtype=np.float32),
        "normalization_median": np.zeros((4, 7), dtype=np.float32),
        "normalization_scale": np.ones((4, 7), dtype=np.float32),
        "normalization_scale_source": np.full((4, 7), "iqr_div_1.349"),
        "source_samples_per_output_axis": np.ones((4, 3), dtype=np.float64),
        "anti_alias_sigma_source_voxels": np.zeros((4, 3), dtype=np.float64),
        "anti_alias_applied": np.zeros(4, dtype=np.uint8),
        "source_canonical_sha256": source_hashes,
        "support_canonical_sha256": support_hashes,
        "support_scope": support_scope,
        "phase_metadata_sha256": phase_metadata_hashes,
        "builder_contract_version": np.asarray(BUILDER_CONTRACT_VERSION),
        "builder_contract_sha256": np.asarray(contract_digest),
        "input_provenance_sha256": np.asarray(provenance_digest),
        "source_to_anchor_ras": transforms,
        "support_available": np.asarray((1, 0, 0, 0), dtype=np.uint8),
        "support_source_positive_voxels": np.asarray((5, 0, 0, 0), dtype=np.int64),
        "support_retained_positive_voxels": np.asarray((5, 0, 0, 0), dtype=np.int64),
        "support_nn_target_positive_voxels": np.asarray((5, 0, 0, 0), dtype=np.int64),
        "support_retained_positive_voxel_fraction": np.asarray(
            (1.0, np.nan, np.nan, np.nan), dtype=np.float32
        ),
        "support_physical_volume_retention": np.asarray(
            (1.0, np.nan, np.nan, np.nan), dtype=np.float32
        ),
        "support_exact_full_support_containment": np.asarray(
            (1, 0, 0, 0), dtype=np.uint8
        ),
        "support_source_boundary_touch": np.asarray((0, 0, 0, 0), dtype=np.uint8),
        "support_target_boundary_touch": np.asarray((0, 0, 0, 0), dtype=np.uint8),
        "support_minimum_margin_mm": np.asarray(
            (2.0, np.nan, np.nan, np.nan), dtype=np.float32
        ),
        "support_source_volume_mm3": np.asarray((5.0, 0.0, 0.0, 0.0), dtype=np.float32),
        "support_retained_source_volume_mm3": np.asarray(
            (5.0, 0.0, 0.0, 0.0), dtype=np.float32
        ),
        "padding_mode": np.asarray("reflect"),
        "intensity_interpolation": np.asarray("linear"),
        "support_interpolation": np.asarray("nearest"),
    }


class CacheTests(unittest.TestCase):
    def test_schema3_rejects_unknown_label_keys_and_float_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unknown = _synthetic_payload()
            unknown["pCR_future_label"] = np.asarray(1, dtype=np.int8)
            with self.assertRaisesRegex(ValueError, "forbidden or unknown"):
                write_cache_atomic(
                    Path(directory) / "unknown.npz",
                    unknown,
                    require_frozen_grid=False,
                )
            wrong_version = _synthetic_payload()
            wrong_version["schema_version"] = np.asarray(3.0, dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "integer scalar 3"):
                write_cache_atomic(
                    Path(directory) / "float-schema.npz",
                    wrong_version,
                    require_frozen_grid=False,
                )

    def test_schema3_rejects_nonrigid_t0_and_header_transforms(self) -> None:
        cases: list[tuple[str, np.ndarray, str]] = []
        nonrigid = np.repeat(np.eye(4)[None], 4, axis=0)
        nonrigid[1, 0, 0] = 1.1
        cases.append(("nonrigid", nonrigid, "must be rigid"))
        t0_motion = np.repeat(np.eye(4)[None], 4, axis=0)
        t0_motion[0, 0, 3] = 1.0
        cases.append(("t0", t0_motion, "T0 registration transform"))
        h_motion = np.repeat(np.eye(4)[None], 4, axis=0)
        h_motion[1, 0, 3] = 1.0
        cases.append(("header", h_motion, "C1B-H cache"))
        with tempfile.TemporaryDirectory() as directory:
            for name, transforms, message in cases:
                with self.subTest(name=name):
                    payload = _synthetic_payload()
                    payload["source_to_anchor_ras"] = transforms
                    with self.assertRaisesRegex(ValueError, message):
                        write_cache_atomic(
                            Path(directory) / f"{name}.npz",
                            payload,
                            require_frozen_grid=False,
                        )

    def test_atomic_cache_is_byte_deterministic_and_exactly_round_trips(self) -> None:
        payload = _synthetic_payload()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            first_validation = write_cache_atomic(
                first, payload, require_frozen_grid=False
            )
            second_validation = write_cache_atomic(
                second, payload, require_frozen_grid=False
            )
            self.assertEqual(
                first_validation.content_sha256, second_validation.content_sha256
            )
            self.assertEqual(
                first_validation.file_sha256, second_validation.file_sha256
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertTrue(
                    all(
                        item.compress_type == zipfile.ZIP_STORED
                        for item in archive.infolist()
                    )
                )
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            arrays, reload_validation = load_and_validate_cache(
                first,
                require_frozen_grid=False,
                expected_image=payload["image"],
                expected_file_sha256=first_validation.file_sha256,
            )
            np.testing.assert_array_equal(arrays["image"], payload["image"])
            np.testing.assert_array_equal(
                arrays["phase_indices"], payload["phase_indices"]
            )
            self.assertEqual(
                tuple(arrays["channel_names"].tolist()), DCE7_CHANNEL_NAMES
            )
            self.assertEqual(
                reload_validation.content_sha256, first_validation.content_sha256
            )
            # The model-facing loader returns one object: float32 DCE7 only.
            model = load_model_tensor(first, require_frozen_grid=False)
            np.testing.assert_array_equal(model, payload["image"])
            self.assertEqual(model.shape[1], 7)

    def test_hash_validation_detects_any_sidecar_or_tensor_change(self) -> None:
        payload = _synthetic_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_cache_atomic(path, payload, require_frozen_grid=False)
            arrays, _ = load_and_validate_cache(path, require_frozen_grid=False)
            arrays["phase_indices"][0, 1] += 1
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                validate_cache_arrays(arrays, require_frozen_grid=False)
            self.assertIn(CONTENT_HASH_KEY, arrays)

    def test_valid_mask_and_metadata_cannot_be_model_channels(self) -> None:
        payload = _synthetic_payload()
        self.assertEqual(payload["image"].shape[1], len(DCE7_CHANNEL_NAMES))
        self.assertEqual(payload["valid_source_mask"].shape[1], 1)
        self.assertNotEqual(
            payload["image"].shape[1],
            payload["image"].shape[1] + payload["valid_source_mask"].shape[1],
        )
        self.assertNotIn("mask", DCE7_CHANNEL_NAMES)
        self.assertNotIn("affine", DCE7_CHANNEL_NAMES)
        self.assertNotIn("spacing", DCE7_CHANNEL_NAMES)


if __name__ == "__main__":
    unittest.main()
