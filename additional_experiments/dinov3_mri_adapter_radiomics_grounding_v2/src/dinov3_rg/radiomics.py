"""Private observable-ROI construction and PyRadiomics extraction contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

from .contracts import (
    EXPERIMENT_ROOT,
    FTV_TRANSITIONS,
    LOCKED_HASHES,
    SOURCE_INVENTORY,
    VISITS,
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_protocol,
    patient_order_sha256,
    private_patient_token,
    verify_locked_file,
)
from .cache_io import CacheEntry


GEOMETRY_SRC = (
    EXPERIMENT_ROOT.parents[1]
    / "additional_experiments/c1b_model_ready_ftv_sanity/src"
)


def _geometry_api():
    value = str(GEOMETRY_SRC)
    if value not in sys.path:
        sys.path.insert(0, value)
    from c1b_sanity.geometry import (  # type: ignore
        PhysicalGrid,
        canonical_volume_sha256,
        load_nifti_ras,
        resample_support_nearest,
    )

    return PhysicalGrid, canonical_volume_sha256, load_nifti_ras, resample_support_nearest


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def local_bounds(shape_zyx: Sequence[int] = (112, 176, 160)) -> tuple[slice, slice, slice]:
    widths = (32, 72, 72)
    starts = tuple((int(length) - width) // 2 for length, width in zip(shape_zyx, widths))
    return tuple(slice(start, start + width) for start, width in zip(starts, widths))  # type: ignore[return-value]


def ftv_wide(path: str | Path = FTV_TRANSITIONS) -> pd.DataFrame:
    source = Path(path)
    verify_locked_file(source, LOCKED_HASHES["ftv_transitions"], "FTV transition table")
    frame = pd.read_csv(
        source,
        usecols=["patient_id", "start_visit", "end_visit", "ftv_start", "ftv_end", "ftv_valid"],
        dtype={"patient_id": str, "start_visit": str, "end_visit": str},
    )
    rows: list[dict[str, Any]] = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        values = {visit: np.nan for visit in VISITS}
        valid = {visit: False for visit in VISITS}
        for row in group.itertuples(index=False):
            if bool(row.ftv_valid):
                for visit, value in ((row.start_visit, row.ftv_start), (row.end_visit, row.ftv_end)):
                    value = float(value)
                    if valid[visit] and not np.isclose(values[visit], value, atol=1e-8, rtol=0.0):
                        raise ValueError("FTV transition rows disagree on a visit")
                    values[visit], valid[visit] = value, np.isfinite(value) and value >= 0
        rows.append({"patient_id": patient_id, **{f"ftv_{v}": values[v] for v in VISITS}, **{f"ftv_valid_{v}": valid[v] for v in VISITS}})
    output = pd.DataFrame(rows)
    if len(output) != 375 or output["patient_id"].duplicated().any():
        raise ValueError("FTV target population must contain 375 patients")
    return output


def load_support_inventory(path: str | Path = SOURCE_INVENTORY) -> pd.DataFrame:
    source = Path(path)
    verify_locked_file(source, LOCKED_HASHES["source_inventory"], "source inventory")
    frame = pd.read_csv(
        source,
        usecols=["patient_id", "cohort", "visit", "formal_ftv_overlap", "ftv_mask_nifti"],
        dtype={"patient_id": str, "cohort": str, "visit": str, "ftv_mask_nifti": str},
    )
    frame = frame.loc[frame["formal_ftv_overlap"].astype(bool)].copy()
    if len(frame) != 375 * 4 or frame.duplicated(["patient_id", "visit"]).any():
        raise ValueError("formal support inventory must contain 375 complete T0-T3 patients")
    if set(frame["visit"]) != set(VISITS) or frame["ftv_mask_nifti"].isna().any():
        raise ValueError("formal support inventory is incomplete")
    return frame.sort_values(["patient_id", "visit"], kind="stable")


def build_patient_roi(
    entry: CacheEntry,
    support_rows: pd.DataFrame,
    ftv_row: pd.Series,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = Path(output_dir) / f"{private_patient_token(entry.patient_id)}.private.npz"
    if destination.exists() and not overwrite:
        return validate_roi_archive(destination, entry.patient_id)
    PhysicalGrid, canonical_volume_sha256, load_nifti_ras, resample_support_nearest = _geometry_api()
    with np.load(entry.path, allow_pickle=False) as cache:
        if str(cache["patient_id"].item()) != entry.patient_id:
            raise ValueError("C1B cache identity mismatch while building ROI")
        valid_source = np.asarray(cache["valid_source_mask"], dtype=bool)
        source_to_anchor = np.asarray(cache["source_to_anchor_ras"], dtype=np.float64)
        grid = PhysicalGrid(
            shape_zyx=tuple(np.asarray(cache["grid_shape_zyx"], dtype=int)),
            spacing_xyz_mm=tuple(np.asarray(cache["grid_spacing_xyz_mm"], dtype=float)),
            center_ras_mm=tuple(np.asarray(cache["grid_center_ras_mm"], dtype=float)),
        )
        expected_support_hash = np.asarray(cache["support_canonical_sha256"]).astype(str)
    by_visit = support_rows.set_index("visit")
    bounds = local_bounds(grid.shape_zyx)
    local_valid_source = valid_source[:, 0][(slice(None), *bounds)]
    masks = np.zeros((4, 32, 72, 72), dtype=bool)
    valid = np.zeros(4, dtype=bool)
    volumes = np.zeros(4, dtype=np.float32)
    voxel_counts = np.zeros(4, dtype=np.int64)
    slice_counts = np.zeros(4, dtype=np.int16)
    support_hashes: list[str] = []
    for index, visit in enumerate(VISITS):
        source = Path(str(by_visit.loc[visit, "ftv_mask_nifti"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        support = load_nifti_ras(source)
        observed_hash = canonical_volume_sha256(support)
        if expected_support_hash[index] and observed_hash != expected_support_hash[index]:
            raise ValueError(f"support canonical hash mismatch for {visit}")
        full = resample_support_nearest(
            support, grid, source_to_anchor_ras=source_to_anchor[index]
        )
        roi = full[bounds] & valid_source[index, 0][bounds]
        count = int(roi.sum())
        slices = int(np.count_nonzero(roi.any(axis=(1, 2))))
        masks[index] = roi
        voxel_counts[index] = count
        slice_counts[index] = slices
        volumes[index] = count * float(np.prod(grid.spacing_xyz_mm))
        valid[index] = count >= 64 and slices >= 3
        support_hashes.append(observed_hash)
    ftv = np.asarray([ftv_row[f"ftv_{visit}"] for visit in VISITS], dtype=np.float32)
    ftv_mask = np.asarray([ftv_row[f"ftv_valid_{visit}"] for visit in VISITS], dtype=bool)
    _atomic_npz(
        destination,
        patient_id=np.asarray(entry.patient_id),
        patient_token=np.asarray(private_patient_token(entry.patient_id)),
        roi_mask=masks.astype(np.uint8),
        local_valid_source_mask=local_valid_source.astype(np.uint8),
        radiomics_mask=valid.astype(np.uint8),
        local_mask_volume_mm3=volumes,
        roi_voxels=voxel_counts,
        roi_axial_slices=slice_counts,
        ftv=ftv,
        ftv_mask=ftv_mask.astype(np.uint8),
        source_cache_sha256=np.asarray(entry.sha256),
        support_hashes_sha256=np.asarray(canonical_sha256(support_hashes)),
    )
    return validate_roi_archive(destination, entry.patient_id)


def validate_roi_archive(path: str | Path, patient_id: str | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "patient_id", "patient_token", "roi_mask", "local_valid_source_mask",
            "radiomics_mask", "local_mask_volume_mm3",
            "roi_voxels", "roi_axial_slices", "ftv", "ftv_mask", "source_cache_sha256",
            "support_hashes_sha256",
        }
        if set(payload.files) != required:
            raise ValueError("ROI archive member contract drifted")
        observed_id = str(payload["patient_id"].item())
        mask = np.asarray(payload["roi_mask"], dtype=bool)
        valid_source = np.asarray(payload["local_valid_source_mask"], dtype=bool)
        valid = np.asarray(payload["radiomics_mask"], dtype=bool)
        voxels = np.asarray(payload["roi_voxels"], dtype=np.int64)
        slices = np.asarray(payload["roi_axial_slices"], dtype=np.int16)
        volume = np.asarray(payload["local_mask_volume_mm3"], dtype=np.float32)
    if patient_id is not None and observed_id != patient_id:
        raise ValueError("ROI archive identity mismatch")
    if mask.shape != (4, 32, 72, 72) or valid_source.shape != mask.shape or valid.shape != (4,):
        raise ValueError("ROI archive shape contract failed")
    if np.any(mask & ~valid_source):
        raise ValueError("observable ROI leaves valid-source support")
    if not np.array_equal(mask.sum(axis=(1, 2, 3)), voxels):
        raise ValueError("ROI voxel count mismatch")
    if not np.array_equal(np.count_nonzero(mask.any(axis=(2, 3)), axis=1), slices):
        raise ValueError("ROI axial-slice count mismatch")
    if not np.array_equal(valid, (voxels >= 64) & (slices >= 3)):
        raise ValueError("ROI validity rule drifted")
    if not np.allclose(volume, voxels * (0.9 * 0.9 * 2.0), atol=1e-4, rtol=1e-6):
        raise ValueError("ROI volume mismatch")
    return {
        "status": "PASS",
        "valid_visits": int(valid.sum()),
        "sha256": file_sha256(path),
    }


def finalize_roi_gate(patient_ids: Iterable[str], roi_dir: str | Path) -> dict[str, Any]:
    ids = tuple(map(str, patient_ids))
    masks: list[np.ndarray] = []
    voxel_counts: list[np.ndarray] = []
    slice_counts: list[np.ndarray] = []
    hashes: list[str] = []
    observed_ids: list[str] = []
    eroded_valid: list[np.ndarray] = []
    for patient_id in ids:
        path = Path(roi_dir) / f"{private_patient_token(patient_id)}.private.npz"
        validate_roi_archive(path, patient_id)
        with np.load(path, allow_pickle=False) as payload:
            observed_ids.append(str(payload["patient_id"].item()))
            masks.append(np.asarray(payload["radiomics_mask"], dtype=bool))
            voxel_counts.append(np.asarray(payload["roi_voxels"], dtype=np.int64))
            slice_counts.append(np.asarray(payload["roi_axial_slices"], dtype=np.int16))
            roi = np.asarray(payload["roi_mask"], dtype=bool)
            valid_source = np.asarray(payload["local_valid_source_mask"], dtype=bool)
        erosion_rows = []
        for visit in range(4):
            _, erosion, _ = morphology_variants(roi[visit], valid_source[visit])
            erosion_rows.append(
                int(erosion.sum()) >= 64
                and int(np.count_nonzero(erosion.any(axis=(1, 2)))) >= 3
            )
        eroded_valid.append(np.asarray(erosion_rows, dtype=bool))
        hashes.append(file_sha256(path))
    matrix = np.stack(masks)
    voxels = np.stack(voxel_counts)
    slices = np.stack(slice_counts)
    erosion = np.stack(eroded_valid)
    coverage = {visit: float(matrix[:, index].mean()) for index, visit in enumerate(VISITS)}
    coverage["overall"] = float(matrix.mean())
    thresholds = load_protocol()["radiomics"]["coverage_minimum"]
    coverage_gates = {name: coverage[name] >= float(thresholds[name]) for name in thresholds}
    from .data import load_fold_frame

    folds = load_fold_frame()
    erosion_minimum = load_protocol()["radiomics"]["erosion_outer_train_minimum"]
    erosion_fold_counts: dict[str, dict[str, int]] = {}
    erosion_fold_gates: dict[str, dict[str, bool]] = {}
    patient_index = {patient_id: index for index, patient_id in enumerate(observed_ids)}
    for fold in range(5):
        train_ids = folds.loc[
            folds["fold"].eq(fold) & folds["split"].eq("train"), "patient_id"
        ].astype(str)
        indices = np.asarray([patient_index[value] for value in train_ids if value in patient_index])
        counts = {
            visit: int(erosion[indices, VISITS.index(visit)].sum())
            for visit in erosion_minimum
        }
        erosion_fold_counts[str(fold)] = counts
        erosion_fold_gates[str(fold)] = {
            visit: counts[visit] >= int(erosion_minimum[visit]) for visit in erosion_minimum
        }
    erosion_gate = all(
        passed for fold_gates in erosion_fold_gates.values() for passed in fold_gates.values()
    )
    gates = {**coverage_gates, "erosion_outer_train_minimum": erosion_gate}
    failure_audit = {
        visit: {
            "valid": int(matrix[:, index].sum()),
            "required_for_gate": int(np.ceil(float(thresholds[visit]) * len(ids))),
            "additional_valid_needed": max(
                0,
                int(np.ceil(float(thresholds[visit]) * len(ids))) - int(matrix[:, index].sum()),
            ),
            "voxel_below_64": int((voxels[:, index] < 64).sum()),
            "axial_slices_below_3": int((slices[:, index] < 3).sum()),
            "voxel_count_q05": float(np.quantile(voxels[:, index], 0.05)),
            "voxel_count_median": float(np.median(voxels[:, index])),
            "axial_slice_count_q05": float(np.quantile(slices[:, index], 0.05)),
            "axial_slice_count_median": float(np.median(slices[:, index])),
        }
        for index, visit in enumerate(VISITS)
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "NO_GO",
        "patients": len(ids),
        "coverage": coverage,
        "thresholds": thresholds,
        "gates": gates,
        "coverage_gates": coverage_gates,
        "erosion_outer_train_minimum": erosion_minimum,
        "erosion_outer_train_counts": erosion_fold_counts,
        "erosion_outer_train_gates": erosion_fold_gates,
        "failure_audit": failure_audit,
        "patient_order_sha256": patient_order_sha256(ids),
        "ordered_private_roi_hashes_sha256": canonical_sha256(hashes),
        "model_forward_receives_roi": False,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    atomic_json(EXPERIMENT_ROOT / "metrics/roi_feasibility.json", payload)
    atomic_json(EXPERIMENT_ROOT / "metrics/radiomics_stage_a_gate.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"radiomics coverage gate failed: {coverage}")
    return payload


def morphology_variants(
    mask_zyx: np.ndarray,
    valid_source_zyx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.shape != (32, 72, 72):
        raise ValueError("radiomics mask must be [32,72,72]")
    valid_source = np.ones_like(mask) if valid_source_zyx is None else np.asarray(valid_source_zyx, dtype=bool)
    if valid_source.shape != mask.shape or np.any(mask & ~valid_source):
        raise ValueError("valid-source morphology contract failed")
    structure = np.ones((1, 3, 3), dtype=bool)
    return (
        mask,
        ndimage.binary_erosion(mask, structure=structure, iterations=1, border_value=0) & valid_source,
        ndimage.binary_dilation(mask, structure=structure, iterations=1, border_value=0) & valid_source,
    )


def make_pyradiomics_extractor():
    try:
        import radiomics
        from radiomics import featureextractor
    except ImportError as error:
        raise RuntimeError(
            "PyRadiomics is absent. Run extraction in the locked Python 3.9 side environment."
        ) from error
    if str(radiomics.__version__).lstrip("v") != "3.1.0":
        raise RuntimeError(f"PyRadiomics must be 3.1.0, got {radiomics.__version__}")
    settings = {
        "binWidth": 0.25,
        "normalize": False,
        "voxelArrayShift": 0,
        "force2D": True,
        "force2Ddimension": 0,
        "resampledPixelSpacing": None,
        "interpolator": None,
        "minimumROIDimensions": 2,
        # PyRadiomics rejects roiSize <= minimumROISize. Setting 63 implements
        # the externally locked inclusive minimum of 64 voxels.
        "minimumROISize": int(
            load_protocol()["radiomics"]["pyradiomics_minimum_roi_size_setting"]
        ),
        "correctMask": False,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for name in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        extractor.enableFeatureClassByName(name)
    return extractor


def _sitk_pair(image_zyx: np.ndarray, mask_zyx: np.ndarray):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(image_zyx, dtype=np.float32))
    mask = sitk.GetImageFromArray(np.asarray(mask_zyx, dtype=np.uint8))
    for value in (image, mask):
        value.SetSpacing((0.9, 0.9, 2.0))
        value.SetOrigin((0.0, 0.0, 0.0))
        value.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image, mask


def extract_one_mask(extractor: Any, image_zyx: np.ndarray, mask_zyx: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask_zyx, dtype=bool)
    if int(mask.sum()) < 64 or int(np.count_nonzero(mask.any(axis=(1, 2)))) < 3:
        return {}
    image, sitk_mask = _sitk_pair(image_zyx, mask)
    result = extractor.execute(image, sitk_mask, label=1)
    selected: dict[str, float] = {}
    for key, value in result.items():
        if not str(key).startswith("original_"):
            continue
        number = float(value)
        selected[str(key)] = number if np.isfinite(number) else np.nan
    if not selected or any("shape" in name.lower() for name in selected):
        raise ValueError("PyRadiomics feature-class contract failed")
    return selected


def extract_patient_radiomics(
    entry: CacheEntry,
    roi_path: str | Path,
    output_dir: str | Path,
    extractor: Any,
    *,
    expected_feature_names: Sequence[str] | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, Any], tuple[str, ...] | None]:
    destination = Path(output_dir) / f"{private_patient_token(entry.patient_id)}.private.npz"
    if destination.exists() and not overwrite:
        with np.load(destination, allow_pickle=False) as payload:
            names = tuple(payload["feature_name"].astype(str).tolist())
            observed_contract = str(payload["extraction_contract_sha256"].item())
            observed_roi_hash = str(payload["source_roi_sha256"].item())
            if "variant_valid" not in payload.files:
                raise ValueError("reused radiomics archive lacks V2 variant-valid mask")
        expected_contract = canonical_sha256(load_protocol()["radiomics"])
        if observed_contract != expected_contract or observed_roi_hash != file_sha256(roi_path):
            raise ValueError("reused radiomics archive is not bound to the current V2 contract")
        return {"status": "REUSED", "sha256": file_sha256(destination)}, names
    with np.load(entry.path, allow_pickle=False) as cache, np.load(roi_path, allow_pickle=False) as roi:
        if str(cache["patient_id"].item()) != entry.patient_id or str(roi["patient_id"].item()) != entry.patient_id:
            raise ValueError("radiomics input identity mismatch")
        image = np.asarray(cache["image"], dtype=np.float32)[(slice(None), slice(None), *local_bounds())]
        mask = np.asarray(roi["roi_mask"], dtype=bool)
        valid_source = np.asarray(roi["local_valid_source_mask"], dtype=bool)
        valid = np.asarray(roi["radiomics_mask"], dtype=bool)
    with np.load(entry.path, allow_pickle=False) as cache:
        channel_names = tuple(str(value) for value in cache["channel_names"])
    records: dict[tuple[int, int], dict[str, float]] = {}
    variant_valid = np.zeros((4, 3), dtype=bool)
    discovered = tuple(expected_feature_names) if expected_feature_names is not None else None
    for visit_index in range(4):
        if not valid[visit_index]:
            continue
        for variant_index, variant in enumerate(
            morphology_variants(mask[visit_index], valid_source[visit_index])
        ):
            variant_valid[visit_index, variant_index] = (
                int(variant.sum()) >= 64
                and int(np.count_nonzero(variant.any(axis=(1, 2)))) >= 3
            )
            combined: dict[str, float] = {}
            for channel_index, channel_name in enumerate(channel_names):
                extracted = extract_one_mask(extractor, image[visit_index, channel_index], variant)
                combined.update({f"{channel_name}::{name}": value for name, value in extracted.items()})
            if combined:
                names = tuple(sorted(combined))
                if discovered is None:
                    discovered = names
                if names != discovered:
                    raise ValueError("PyRadiomics feature-name contract drifted")
                records[(visit_index, variant_index)] = combined
    if discovered is None:
        return {"status": "NO_VALID_ROI", "sha256": ""}, None
    values = np.full((4, 3, len(discovered)), np.nan, dtype=np.float32)
    for (visit_index, variant_index), record in records.items():
        values[visit_index, variant_index] = np.asarray([record[name] for name in discovered], dtype=np.float32)
    _atomic_npz(
        destination,
        patient_id=np.asarray(entry.patient_id),
        patient_token=np.asarray(private_patient_token(entry.patient_id)),
        feature_name=np.asarray(discovered, dtype="U160"),
        value=values,
        variant_valid=variant_valid.astype(np.uint8),
        extraction_contract_sha256=np.asarray(canonical_sha256(load_protocol()["radiomics"])),
        source_cache_sha256=np.asarray(entry.sha256),
        source_roi_sha256=np.asarray(file_sha256(roi_path)),
    )
    return {"status": "WRITTEN", "sha256": file_sha256(destination)}, discovered


__all__ = [
    "build_patient_roi", "extract_one_mask", "extract_patient_radiomics", "finalize_roi_gate",
    "ftv_wide", "load_support_inventory", "local_bounds", "make_pyradiomics_extractor",
    "morphology_variants", "validate_roi_archive"
]
