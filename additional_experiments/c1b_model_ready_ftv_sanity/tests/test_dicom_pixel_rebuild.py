from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from c1b_sanity.dicom_pixel_rebuild import (  # noqa: E402
    DicomPixelRebuildError,
    rebuild_classic_dce_series,
)


ROWS = 3
COLUMNS = 4
SLICES = 3
TIMES = 2
IOP = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
PIXEL_SPACING = (2.0, 1.5)  # DICOM order: row, column.
ORIGIN_LPS = np.asarray((10.0, 20.0, 30.0), dtype=float)
SLICE_SPACING = 3.0


def _expected_affine_ras() -> np.ndarray:
    affine_lps = np.asarray(
        [
            [1.5, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return np.diag((-1.0, -1.0, 1.0, 1.0)) @ affine_lps


def _write_cell(
    path: Path,
    *,
    series_uid: str,
    sop_uid: str,
    temporal_position: int,
    acquisition_time: str,
    slice_index: int,
    raw_pixels: np.ndarray,
    slope: float,
    intercept: float,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = generate_uid()
    dataset.Modality = "MR"
    dataset.Rows = ROWS
    dataset.Columns = COLUMNS
    dataset.NumberOfFrames = 1
    dataset.ImageOrientationPatient = [str(value) for value in IOP]
    ipp = ORIGIN_LPS + np.asarray((0.0, 0.0, slice_index * SLICE_SPACING))
    dataset.ImagePositionPatient = [str(value) for value in ipp]
    dataset.PixelSpacing = [str(value) for value in PIXEL_SPACING]
    dataset.SliceThickness = str(SLICE_SPACING)
    dataset.SpacingBetweenSlices = str(SLICE_SPACING)
    dataset.TemporalPositionIdentifier = temporal_position
    dataset.AcquisitionTime = acquisition_time
    dataset.RescaleSlope = str(slope)
    dataset.RescaleIntercept = str(intercept)
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelData = np.asarray(raw_pixels, dtype="<i2").tobytes(order="C")
    dataset.save_as(path, enforce_file_format=True)


def _make_series(
    root: Path,
    *,
    constant: bool = False,
    acquisition_times: tuple[str, str] = ("120000.000", "120100.000"),
) -> tuple[str, dict[tuple[int, int], np.ndarray], list[Path]]:
    root.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    expected: dict[tuple[int, int], np.ndarray] = {}
    paths: list[Path] = []
    serial = 0
    for time_index in range(TIMES):
        for slice_index in range(SLICES):
            if constant:
                raw = np.zeros((ROWS, COLUMNS), dtype=np.int16)
                slope, intercept = 1.0, 0.0
            else:
                raw = (
                    np.arange(ROWS * COLUMNS, dtype=np.int16).reshape(ROWS, COLUMNS)
                    + 100 * time_index
                    + 10 * slice_index
                )
                slope = 1.0 + 0.25 * time_index + 0.1 * slice_index
                intercept = -10.0 + 2.0 * time_index - slice_index
            # Reverse lexical file order relative to both time and slice order.
            path = root / f"cell_{TIMES * SLICES - serial:02d}.dcm"
            _write_cell(
                path,
                series_uid=series_uid,
                sop_uid=generate_uid(),
                temporal_position=time_index + 1,
                acquisition_time=acquisition_times[time_index],
                slice_index=slice_index,
                raw_pixels=raw,
                slope=slope,
                intercept=intercept,
            )
            expected[(time_index, slice_index)] = (
                raw.astype(np.float64) * slope + intercept
            ).astype(np.float32).T
            paths.append(path)
            serial += 1
    return series_uid, expected, paths


def _rebuild(series: Path, **kwargs):
    return rebuild_classic_dce_series(
        series,
        expected_shape_xyzt=(COLUMNS, ROWS, SLICES, TIMES),
        expected_spacing_xyz_mm=(1.5, 2.0, 3.0),
        reference_affine_ras=_expected_affine_ras(),
        reference_shape_xyz=(COLUMNS, ROWS, SLICES),
        **kwargs,
    )


class DicomPixelRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rebuild_scales_orders_recompares_and_writes_nifti(self) -> None:
        series = self.root / "raw"
        series_uid, expected, _ = _make_series(series)
        output = self.root / "rebuilt.nii.gz"

        result = _rebuild(series, output_nifti=output)

        self.assertEqual(result.volume_xyzt.shape, (COLUMNS, ROWS, SLICES, TIMES))
        self.assertEqual(result.volume_xyzt.dtype, np.float32)
        np.testing.assert_allclose(result.affine_ras, _expected_affine_ras(), atol=1e-12)
        for time_index in range(TIMES):
            for slice_index in range(SLICES):
                np.testing.assert_array_equal(
                    result.volume_xyzt[:, :, slice_index, time_index],
                    expected[(time_index, slice_index)],
                )

        metrics = result.metrics
        self.assertEqual(metrics.status, "PASS")
        self.assertTrue(metrics.pixel_data_read)
        self.assertTrue(metrics.pixel_rebuild_executed)
        self.assertTrue(metrics.pixel_order_verified)
        self.assertTrue(metrics.pixel_rebuild_ready)
        self.assertEqual(metrics.decoded_cell_count, SLICES * TIMES)
        self.assertEqual(metrics.verified_cell_count, SLICES * TIMES)
        self.assertEqual(metrics.cell_recomparison_max_abs_error, 0.0)
        self.assertEqual(metrics.orientation_ras, "LPS")
        self.assertLessEqual(metrics.reference_footprint_corner_hausdorff_mm, 1e-12)
        self.assertIsNone(result.private)

        public_text = json.dumps(result.public_metrics(), sort_keys=True)
        self.assertNotIn(str(series), public_text)
        self.assertNotIn(series_uid, public_text)
        self.assertNotIn("sha256", public_text.lower())

        reloaded = nib.load(output)
        self.assertEqual(reloaded.shape, result.volume_xyzt.shape)
        self.assertEqual(np.dtype(reloaded.get_data_dtype()), np.dtype(np.float32))
        qform, qcode = reloaded.get_qform(coded=True)
        sform, scode = reloaded.get_sform(coded=True)
        self.assertGreater(int(qcode), 0)
        self.assertGreater(int(scode), 0)
        np.testing.assert_allclose(qform, result.affine_ras, atol=1e-5)
        np.testing.assert_allclose(sform, result.affine_ras, atol=1e-5)
        np.testing.assert_array_equal(
            np.asarray(reloaded.dataobj, dtype=np.float32), result.volume_xyzt
        )

    def test_private_provenance_is_explicit_opt_in(self) -> None:
        series = self.root / "raw"
        series_uid, _, _ = _make_series(series)
        output = self.root / "private_rebuild.nii"

        result = _rebuild(series, output_nifti=output, include_private=True)

        self.assertIsNotNone(result.private)
        assert result.private is not None
        self.assertEqual(result.private["series_instance_uid"], series_uid)
        self.assertEqual(len(result.private["ordered_cells"]), SLICES * TIMES)
        self.assertEqual(len(result.private["rebuilt_volume_sha256"]), 64)
        self.assertEqual(len(result.private["output_nifti_sha256"]), 64)
        for cell in result.private["ordered_cells"]:
            self.assertEqual(len(cell["source_file_sha256"]), 64)
            self.assertEqual(len(cell["scaled_pixel_sha256"]), 64)
            self.assertIn("source_path", cell)

    def test_temporal_grouping_and_order_must_agree(self) -> None:
        with self.subTest("same TPI cannot map to multiple acquisition times"):
            series = self.root / "grouping"
            _, _, paths = _make_series(series)
            dataset = pydicom.dcmread(paths[1])
            dataset.AcquisitionTime = "120030.000"
            dataset.save_as(paths[1], enforce_file_format=True)
            with self.assertRaises(DicomPixelRebuildError) as caught:
                _rebuild(series)
            self.assertEqual(caught.exception.code, "TEMPORAL_GROUPING_DISAGREEMENT")

        with self.subTest("TPI order must match AcquisitionTime order"):
            series = self.root / "ordering"
            _make_series(
                series,
                acquisition_times=("120100.000", "120000.000"),
            )
            with self.assertRaises(DicomPixelRebuildError) as caught:
                _rebuild(series)
            self.assertEqual(caught.exception.code, "TEMPORAL_ORDER_DISAGREEMENT")

    def test_duplicate_cells_sops_and_mixed_series_fail_closed(self) -> None:
        with self.subTest("missing time/slice cell"):
            series = self.root / "missing_cell"
            _, _, paths = _make_series(series)
            paths[0].unlink()
            with self.assertRaises(DicomPixelRebuildError) as caught:
                _rebuild(series)
            self.assertEqual(caught.exception.code, "MISSING_TIME_SLICE_CELL")

        with self.subTest("duplicate time/slice cell"):
            series = self.root / "duplicate_cell"
            _, _, paths = _make_series(series)
            duplicate = pydicom.dcmread(paths[0])
            new_sop = generate_uid()
            duplicate.SOPInstanceUID = new_sop
            duplicate.file_meta.MediaStorageSOPInstanceUID = new_sop
            duplicate.save_as(series / "extra.dcm", enforce_file_format=True)
            with self.assertRaises(DicomPixelRebuildError) as caught:
                _rebuild(series)
            self.assertEqual(caught.exception.code, "DUPLICATE_TIME_SLICE_CELL")

        with self.subTest("duplicate SOP"):
            series = self.root / "duplicate_sop"
            _, _, paths = _make_series(series)
            shutil.copyfile(paths[0], series / "copied.dcm")
            with self.assertRaises(DicomPixelRebuildError) as caught:
                _rebuild(series)
            self.assertEqual(caught.exception.code, "DUPLICATE_SOP_INSTANCE_UID")

        with self.subTest("mixed series"):
            series = self.root / "mixed_series"
            _, _, paths = _make_series(series)
            dataset = pydicom.dcmread(paths[0])
            dataset.SeriesInstanceUID = generate_uid()
            dataset.save_as(paths[0], enforce_file_format=True)
            with self.assertRaises(DicomPixelRebuildError) as caught:
                _rebuild(series)
            self.assertEqual(caught.exception.code, "MIXED_SERIES_INSTANCE_UID")

    def test_dimension_corner_and_signal_gates_fail_closed(self) -> None:
        series = self.root / "valid"
        _make_series(series)

        with self.subTest("expected dimensions"):
            with self.assertRaises(DicomPixelRebuildError) as caught:
                rebuild_classic_dce_series(
                    series,
                    expected_shape_xyzt=(COLUMNS + 1, ROWS, SLICES, TIMES),
                    expected_spacing_xyz_mm=(1.5, 2.0, 3.0),
                    reference_affine_ras=_expected_affine_ras(),
                    reference_shape_xyz=(COLUMNS + 1, ROWS, SLICES),
                )
            self.assertEqual(caught.exception.code, "EXPECTED_DIMENSION_MISMATCH")

        with self.subTest("reference corner"):
            shifted = _expected_affine_ras().copy()
            shifted[:3, 3] += (1.0, 0.0, 0.0)
            with self.assertRaises(DicomPixelRebuildError) as caught:
                rebuild_classic_dce_series(
                    series,
                    expected_shape_xyzt=(COLUMNS, ROWS, SLICES, TIMES),
                    expected_spacing_xyz_mm=(1.5, 2.0, 3.0),
                    reference_affine_ras=shifted,
                    reference_shape_xyz=(COLUMNS, ROWS, SLICES),
                )
            self.assertEqual(caught.exception.code, "REFERENCE_CORNER_MISMATCH")

        with self.subTest("constant reconstructed signal"):
            constant_series = self.root / "constant"
            _make_series(constant_series, constant=True)
            with self.assertRaises(DicomPixelRebuildError) as caught:
                _rebuild(constant_series)
            self.assertEqual(caught.exception.code, "CONSTANT_REBUILT_VOLUME")


if __name__ == "__main__":
    unittest.main()
