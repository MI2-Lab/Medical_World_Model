#!/usr/bin/env python3
"""Run the formal Stage A5 C1B-H versus C1B-R registration sensitivity.

Registration workers receive only the frozen formal-cohort flag, DCE paths,
repair disposition, visit label, and the three permitted acquisition phase
indices.  FTV/localization paths are never included in a worker payload.  They
are read only after every registration is complete, solely to choose anonymous
small/medium/large examples for technical-review figures.

Patient-level rows, identifiers, paths, transforms, and representative mappings
are written only to ``*.private.*`` outputs covered by the experiment's
``.gitignore``.  Every non-private artifact contains aggregates or anonymous
images only.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
import SimpleITK as sitk


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from c1b_sanity.dce7 import select_phase_indices  # noqa: E402
from c1b_sanity.geometry import CanonicalVolume, load_nifti_ras  # noqa: E402
from c1b_sanity.registration import (  # noqa: E402
    RegistrationConfig,
    _whole_anatomy_mask_array,
    canonical_volume_to_sitk,
    ras_matrix_to_sitk_transform,
    register_precontrast_rigid,
)


VISITS = ("T0", "T1", "T2", "T3")
MOVING_VISITS = ("T1", "T2", "T3")
ALLOWED_PHASE_FIELDS = ("pre", "post_early", "post_late")
PRIVATE_COLUMNS = (
    "run_signature",
    "patient_id",
    "cohort",
    "visit",
    "anchor_dce_path",
    "source_dce_path",
    "anchor_repaired",
    "source_repaired",
    "phase_metadata_available",
    "phase_pre_index_anchor",
    "phase_pre_index_source",
    "phase_count_anchor",
    "phase_count_source",
    "success",
    "failure_code",
    "failure_message",
    "converged",
    "optimizer_iterations",
    "optimizer_stop_condition",
    "optimizer_mattes_mi",
    "histogram_mi_before",
    "histogram_mi_after",
    "histogram_mi_gain",
    "histogram_nmi_before",
    "histogram_nmi_after",
    "histogram_nmi_gain",
    "similarity_before",
    "similarity_after",
    "similarity_gain",
    "rotation_x_deg",
    "rotation_y_deg",
    "rotation_z_deg",
    "rotation_magnitude_deg",
    "maximum_absolute_rotation_deg",
    "translation_x_mm",
    "translation_y_mm",
    "translation_z_mm",
    "translation_magnitude_mm",
    "affine_offset_x_mm",
    "affine_offset_y_mm",
    "affine_offset_z_mm",
    "valid_overlap_fraction_before",
    "valid_overlap_fraction_after",
    "padding_fraction_before",
    "padding_fraction_after",
    "padding_fraction_delta",
    "fixed_anatomy_valid_fraction_before",
    "fixed_anatomy_valid_fraction_after",
    "anatomy_dice_before",
    "anatomy_dice_after",
    "anatomy_dice_gain",
    *(f"source_to_anchor_ras_{row}{column}" for row in range(4) for column in range(4)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_phase_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Read only the patient key and three frozen acquisition fields."""

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        required = ("pid", *ALLOWED_PHASE_FIELDS)
        missing = [field for field in required if field not in header]
        if missing:
            raise ValueError(f"phase metadata lacks required acquisition columns: {missing}")
        indices = {field: header.index(field) for field in required}
        metadata: dict[str, dict[str, str]] = {}
        for values in reader:
            patient_id = values[indices["pid"]]
            if patient_id in metadata:
                raise ValueError("phase metadata contains a duplicate patient key")
            metadata[patient_id] = {
                field: values[indices[field]] for field in ALLOWED_PHASE_FIELDS
            }
    return metadata


def _load_formal_inventory(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], str]]:
    """Return registration-safe rows and a separate post-hoc localization map."""

    registration_rows: list[dict[str, str]] = []
    localization_paths: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "patient_id",
            "cohort",
            "visit",
            "formal_ftv_overlap",
            "dce_nifti",
            "ftv_mask_nifti",
            "phase_count",
            "pixel_rebuild_required",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"private inventory lacks required columns: {missing}")
        for row in reader:
            if not _truth(row["formal_ftv_overlap"]):
                continue
            key = (row["patient_id"], row["visit"])
            localization_paths[key] = row["ftv_mask_nifti"]
            # FTV/localization paths and all nonessential inventory fields are
            # intentionally absent from the registration-safe row.
            registration_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "cohort": row["cohort"],
                    "visit": row["visit"],
                    "dce_nifti": row["dce_nifti"],
                    "phase_count": row["phase_count"],
                    "pixel_rebuild_required": row["pixel_rebuild_required"],
                }
            )
    return registration_rows, localization_paths


def _group_patients(
    rows: Iterable[dict[str, str]],
    phase_metadata: Mapping[str, Mapping[str, str]],
    repaired_root: Path,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        patient = grouped.setdefault(row["patient_id"], {})
        visit = row["visit"]
        if visit in patient:
            raise ValueError("formal inventory contains a duplicate patient/visit row")
        patient[visit] = row

    payloads: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        visits = grouped[patient_id]
        if set(visits) != set(VISITS):
            raise ValueError("formal registration cohort contains an incomplete patient")
        safe_metadata = dict(phase_metadata.get(patient_id, {}))
        visit_payloads: dict[str, dict[str, Any]] = {}
        for visit in VISITS:
            row = visits[visit]
            repaired = _truth(row["pixel_rebuild_required"])
            source_path = (
                repaired_root / row["cohort"] / patient_id / f"{visit}_dce_rebuilt.nii.gz"
                if repaired
                else Path(row["dce_nifti"])
            )
            visit_payloads[visit] = {
                "visit": visit,
                "dce_path": str(source_path),
                "repaired": repaired,
                "inventory_phase_count": int(row["phase_count"]),
            }
        payloads.append(
            {
                "patient_id": patient_id,
                "cohort": visits["T0"]["cohort"],
                "phase_metadata": safe_metadata,
                "phase_metadata_available": patient_id in phase_metadata,
                "visits": visit_payloads,
            }
        )
    return payloads


def _precontrast_volume(
    visit: Mapping[str, Any],
    phase_metadata: Mapping[str, Any],
) -> tuple[CanonicalVolume, int, int]:
    path = Path(str(visit["dce_path"]))
    if not path.is_file():
        qualifier = "repaired NIfTI" if visit["repaired"] else "source NIfTI"
        raise FileNotFoundError(f"required {qualifier} is absent: {path}")
    volume = load_nifti_ras(path)
    if volume.data.ndim == 3:
        phase_count = 1
    elif volume.data.ndim == 4:
        phase_count = int(volume.data.shape[-1])
    else:
        raise ValueError(f"DCE volume must be 3-D or 4-D, got {volume.data.shape}")
    if phase_count != int(visit["inventory_phase_count"]):
        raise ValueError(
            f"actual phase count {phase_count} disagrees with inventory "
            f"{visit['inventory_phase_count']}"
        )
    selection = select_phase_indices(phase_count, phase_metadata)
    selected = (
        np.asarray(volume.data, dtype=np.float32)
        if volume.data.ndim == 3
        else np.asarray(volume.data[..., selection.pre], dtype=np.float32)
    )
    precontrast = CanonicalVolume(
        data=np.ascontiguousarray(selected, dtype=np.float32),
        affine_ras=np.asarray(volume.affine_ras, dtype=np.float64),
        original_axcodes=volume.original_axcodes,
        orientation_transform=np.asarray(volume.orientation_transform, dtype=np.float64),
        source_path=volume.source_path,
    )
    return precontrast, int(selection.pre), phase_count


def _sitk_mask(mask_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(np.transpose(np.asarray(mask_xyz, dtype=np.uint8), (2, 1, 0)))
    image.CopyInformation(reference)
    return image


def _resample_xyz(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
    interpolator: int,
    pixel_id: int,
) -> np.ndarray:
    output = sitk.Resample(moving, fixed, transform, interpolator, 0.0, pixel_id)
    return np.transpose(sitk.GetArrayFromImage(output), (2, 1, 0))


def _histogram_information(
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int = 64,
) -> tuple[float, float]:
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) < 128:
        raise ValueError("fewer than 128 common anatomy voxels for MI/NMI")
    x = np.asarray(fixed, dtype=np.float64)[selected]
    y = np.asarray(moving, dtype=np.float64)[selected]
    x_low, x_high = np.percentile(x, (0.5, 99.5))
    y_low, y_high = np.percentile(y, (0.5, 99.5))
    if x_high <= x_low or y_high <= y_low:
        raise ValueError("common anatomy is constant for MI/NMI")
    x = np.clip(x, x_low, x_high)
    y = np.clip(y, y_low, y_high)
    joint, _, _ = np.histogram2d(
        x,
        y,
        bins=int(bins),
        range=((float(x_low), float(x_high)), (float(y_low), float(y_high))),
    )
    probability = joint / float(joint.sum())
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    positive_joint = probability > 0.0
    entropy_x = float(-np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
    entropy_y = float(-np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
    independent = px[:, None] * py[None, :]
    mutual_information = float(
        np.sum(probability[positive_joint] * np.log(probability[positive_joint] / independent[positive_joint]))
    )
    denominator = math.sqrt(entropy_x * entropy_y)
    if not math.isfinite(mutual_information) or not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("MI/NMI computation is non-finite")
    normalized_mutual_information = float(mutual_information / denominator)
    return mutual_information, normalized_mutual_information


def _information_before_after(
    anchor: CanonicalVolume,
    source: CanonicalVolume,
    source_to_anchor_ras: np.ndarray,
    minimum_anatomy_voxels: int,
) -> tuple[float, float, float, float]:
    fixed_image = canonical_volume_to_sitk(anchor)
    moving_image = canonical_volume_to_sitk(source)
    fixed_mask = _whole_anatomy_mask_array(
        anchor.data, minimum_voxels=minimum_anatomy_voxels
    )
    moving_mask = _whole_anatomy_mask_array(
        source.data, minimum_voxels=minimum_anatomy_voxels
    )
    moving_mask_image = _sitk_mask(moving_mask, moving_image)
    moving_valid_image = sitk.Image(moving_image.GetSize(), sitk.sitkUInt8)
    moving_valid_image.CopyInformation(moving_image)
    moving_valid_image += 1

    transforms = (
        sitk.Transform(3, sitk.sitkIdentity),
        ras_matrix_to_sitk_transform(np.linalg.inv(source_to_anchor_ras)),
    )
    metrics: list[tuple[float, float]] = []
    for transform in transforms:
        warped = _resample_xyz(
            moving_image, fixed_image, transform, sitk.sitkLinear, sitk.sitkFloat32
        )
        warped_mask = _resample_xyz(
            moving_mask_image,
            fixed_image,
            transform,
            sitk.sitkNearestNeighbor,
            sitk.sitkUInt8,
        ).astype(bool)
        valid = _resample_xyz(
            moving_valid_image,
            fixed_image,
            transform,
            sitk.sitkNearestNeighbor,
            sitk.sitkUInt8,
        ).astype(bool)
        metrics.append(
            _histogram_information(anchor.data, warped, fixed_mask & warped_mask & valid)
        )
    return metrics[0][0], metrics[1][0], metrics[0][1], metrics[1][1]


def _blank_private_row(
    patient: Mapping[str, Any],
    source_visit: str,
    run_signature: str,
) -> dict[str, Any]:
    anchor_spec = patient["visits"]["T0"]
    source_spec = patient["visits"][source_visit]
    return {
        field: "" for field in PRIVATE_COLUMNS
    } | {
        "run_signature": run_signature,
        "patient_id": patient["patient_id"],
        "cohort": patient["cohort"],
        "visit": source_visit,
        "anchor_dce_path": anchor_spec["dce_path"],
        "source_dce_path": source_spec["dce_path"],
        "anchor_repaired": bool(anchor_spec["repaired"]),
        "source_repaired": bool(source_spec["repaired"]),
        "phase_metadata_available": bool(patient["phase_metadata_available"]),
        "success": False,
        "converged": False,
    }


def _failure_private_row(
    patient: Mapping[str, Any],
    source_visit: str,
    run_signature: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    row = _blank_private_row(patient, source_visit, run_signature)
    row["failure_code"] = code
    row["failure_message"] = message
    return row


def _process_patient(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Worker entry point; payload is structurally incapable of carrying FTV."""

    patient = payload["patient"]
    run_signature = str(payload["run_signature"])
    config = RegistrationConfig(**payload["registration_config"])
    try:
        anchor, anchor_pre, anchor_phase_count = _precontrast_volume(
            patient["visits"]["T0"], patient["phase_metadata"]
        )
    except Exception as exc:  # fail closed across all three moving visits
        return [
            _failure_private_row(
                patient,
                visit,
                run_signature,
                "ANCHOR_LOAD_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
            )
            for visit in MOVING_VISITS
        ]

    rows: list[dict[str, Any]] = []
    for visit in MOVING_VISITS:
        row = _blank_private_row(patient, visit, run_signature)
        row["phase_pre_index_anchor"] = anchor_pre
        row["phase_count_anchor"] = anchor_phase_count
        try:
            source, source_pre, source_phase_count = _precontrast_volume(
                patient["visits"][visit], patient["phase_metadata"]
            )
            row["phase_pre_index_source"] = source_pre
            row["phase_count_source"] = source_phase_count
            result = register_precontrast_rigid(anchor, source, config)
            row["success"] = bool(result.success)
            row["failure_code"] = result.failure_code.value
            row["failure_message"] = result.failure_message or ""
            row["converged"] = bool(result.converged)
            row["optimizer_iterations"] = (
                "" if result.optimizer_iterations is None else result.optimizer_iterations
            )
            row["optimizer_stop_condition"] = result.optimizer_stop_condition or ""
            row["optimizer_mattes_mi"] = (
                "" if result.final_mattes_mi is None else result.final_mattes_mi
            )
            if not result.success or result.source_to_anchor_ras is None:
                rows.append(row)
                continue

            mi_before, mi_after, nmi_before, nmi_after = _information_before_after(
                anchor,
                source,
                result.source_to_anchor_ras,
                config.minimum_anatomy_voxels,
            )
            row.update(
                {
                    "histogram_mi_before": mi_before,
                    "histogram_mi_after": mi_after,
                    "histogram_mi_gain": mi_after - mi_before,
                    "histogram_nmi_before": nmi_before,
                    "histogram_nmi_after": nmi_after,
                    "histogram_nmi_gain": nmi_after - nmi_before,
                    "similarity_before": result.similarity_before,
                    "similarity_after": result.similarity_after,
                    "similarity_gain": result.similarity_delta,
                    "rotation_x_deg": result.rotation_xyz_deg[0],
                    "rotation_y_deg": result.rotation_xyz_deg[1],
                    "rotation_z_deg": result.rotation_xyz_deg[2],
                    "rotation_magnitude_deg": result.rotation_magnitude_deg,
                    "maximum_absolute_rotation_deg": max(
                        abs(value) for value in result.rotation_xyz_deg
                    ),
                    "translation_x_mm": result.translation_ras_mm[0],
                    "translation_y_mm": result.translation_ras_mm[1],
                    "translation_z_mm": result.translation_ras_mm[2],
                    "translation_magnitude_mm": result.translation_magnitude_mm,
                    "affine_offset_x_mm": result.affine_offset_ras_mm[0],
                    "affine_offset_y_mm": result.affine_offset_ras_mm[1],
                    "affine_offset_z_mm": result.affine_offset_ras_mm[2],
                    "valid_overlap_fraction_before": result.sidecars.valid_overlap_fraction_before,
                    "valid_overlap_fraction_after": result.sidecars.valid_overlap_fraction_after,
                    "padding_fraction_before": result.sidecars.padding_fraction_before,
                    "padding_fraction_after": result.sidecars.padding_fraction_after,
                    "padding_fraction_delta": (
                        result.sidecars.padding_fraction_after
                        - result.sidecars.padding_fraction_before
                    ),
                    "fixed_anatomy_valid_fraction_before": (
                        result.sidecars.fixed_anatomy_valid_fraction_before
                    ),
                    "fixed_anatomy_valid_fraction_after": (
                        result.sidecars.fixed_anatomy_valid_fraction_after
                    ),
                    "anatomy_dice_before": result.sidecars.anatomy_dice_before,
                    "anatomy_dice_after": result.sidecars.anatomy_dice_after,
                    "anatomy_dice_gain": (
                        result.sidecars.anatomy_dice_after
                        - result.sidecars.anatomy_dice_before
                    ),
                }
            )
            for matrix_row in range(4):
                for matrix_column in range(4):
                    row[f"source_to_anchor_ras_{matrix_row}{matrix_column}"] = (
                        result.source_to_anchor_ras[matrix_row, matrix_column]
                    )
        except Exception as exc:
            # A successful optimizer with incomplete downstream metrics is not
            # usable by Stage A and is therefore converted to a closed failure.
            row["success"] = False
            row["failure_code"] = "PAIR_OR_QC_EXCEPTION"
            row["failure_message"] = f"{type(exc).__name__}: {exc}"
            for matrix_row in range(4):
                for matrix_column in range(4):
                    row[f"source_to_anchor_ras_{matrix_row}{matrix_column}"] = ""
        rows.append(row)
    return rows


def _write_private_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (str(row["patient_id"]), str(row["visit"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PRIVATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def _read_private_rows(path: Path, run_signature: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if any(row.get("run_signature") != run_signature for row in rows):
        raise ValueError("existing private metrics use a different run signature")
    return rows


def _quantile(values: Iterable[Any], probability: float) -> float | None:
    finite = [value for item in values if (value := _safe_float(item)) is not None]
    return float(np.quantile(finite, probability)) if finite else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _safe_float(item)) is not None]
    return float(np.mean(finite)) if finite else None


def _metric_distribution(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    values = [value for row in rows if (value := _safe_float(row.get(field))) is not None]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "q05": float(np.quantile(values, 0.05)) if values else None,
        "q25": float(np.quantile(values, 0.25)) if values else None,
        "median": float(np.quantile(values, 0.50)) if values else None,
        "q75": float(np.quantile(values, 0.75)) if values else None,
        "q95": float(np.quantile(values, 0.95)) if values else None,
        "minimum": float(np.min(values)) if values else None,
        "maximum": float(np.max(values)) if values else None,
    }


def _aggregate(
    rows: list[dict[str, str]],
    expected_pairs: int,
    thresholds: Mapping[str, float],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    success = [row for row in rows if _truth(row.get("success"))]
    failures = [row for row in rows if not _truth(row.get("success"))]
    failure_counts: dict[str, int] = {}
    for row in failures:
        code = row.get("failure_code") or "UNSPECIFIED_FAILURE"
        failure_counts[code] = failure_counts.get(code, 0) + 1

    catastrophic = [
        row
        for row in success
        if float(row["translation_magnitude_mm"])
        > float(thresholds["catastrophic_translation_mm"])
        or float(row["maximum_absolute_rotation_deg"])
        > float(thresholds["catastrophic_rotation_deg"])
    ]
    nonworse = [
        row
        for row in success
        if float(row["similarity_after"])
        >= float(row["similarity_before"]) - float(thresholds["nonworse_tolerance"])
    ]
    success_rate = len(success) / expected_pairs if expected_pairs else 0.0
    catastrophic_rate_all = len(catastrophic) / expected_pairs if expected_pairs else 0.0
    catastrophic_rate_success = len(catastrophic) / len(success) if success else 1.0
    nonworse_fraction_all = len(nonworse) / expected_pairs if expected_pairs else 0.0
    nonworse_fraction_success = len(nonworse) / len(success) if success else 0.0
    similarity_gain_median = _quantile(
        (row["similarity_gain"] for row in success), 0.5
    )
    padding_delta_median = _quantile(
        (row["padding_fraction_delta"] for row in success), 0.5
    )
    padding_delta_q95 = _quantile(
        (row["padding_fraction_delta"] for row in success), 0.95
    )

    criteria = {
        "finite_transform_success_rate": {
            "status": (
                "PASS"
                if success_rate >= float(thresholds["minimum_success_rate"])
                else "FAIL"
            ),
            "observed": success_rate,
            "threshold": f">={thresholds['minimum_success_rate']}",
        },
        "catastrophic_transform_rate": {
            "status": (
                "PASS"
                if catastrophic_rate_all <= float(thresholds["maximum_catastrophic_rate"])
                else "FAIL"
            ),
            "observed_all_pairs": catastrophic_rate_all,
            "observed_successful_transforms": catastrophic_rate_success,
            "threshold": f"<={thresholds['maximum_catastrophic_rate']}",
        },
        "median_whole_anatomy_similarity_gain": {
            "status": (
                "PASS"
                if similarity_gain_median is not None
                and similarity_gain_median
                > float(thresholds["minimum_median_similarity_gain"])
                else "FAIL"
            ),
            "observed": similarity_gain_median,
            "threshold": f">{thresholds['minimum_median_similarity_gain']}",
        },
        "nonworse_moving_visit_fraction": {
            "status": (
                "PASS"
                if nonworse_fraction_all >= float(thresholds["minimum_nonworse_fraction"])
                else "FAIL"
            ),
            "observed_all_pairs_failures_count_as_not_nonworse": nonworse_fraction_all,
            "observed_successful_transforms": nonworse_fraction_success,
            "threshold": f">={thresholds['minimum_nonworse_fraction']}",
        },
        "padding_increase": {
            "status": (
                "PASS"
                if padding_delta_median is not None
                and padding_delta_q95 is not None
                and padding_delta_median
                <= float(thresholds["maximum_padding_median_increase"])
                and padding_delta_q95
                <= float(thresholds["maximum_padding_q95_increase"])
                else "FAIL"
            ),
            "observed_median": padding_delta_median,
            "observed_q95": padding_delta_q95,
            "threshold_median": f"<={thresholds['maximum_padding_median_increase']}",
            "threshold_q95": f"<={thresholds['maximum_padding_q95_increase']}",
        },
        "available_support_exact_containment": {
            "status": "PENDING_INDEPENDENT_PHYSICAL_SUPPORT_AUDIT",
            "reason": "must be merged from builder H/R physical-support audit",
        },
        "ftv_retention_q05": {
            "status": "PENDING_INDEPENDENT_PHYSICAL_SUPPORT_AUDIT",
            "reason": "must be merged from builder H/R physical-support audit",
        },
        "anatomy_lesion_residual_pattern": {
            "status": "PENDING_LANDMARK_RESIDUAL_AUDIT",
            "reason": "no landmark residuals were invented from image similarity",
        },
        "blinded_technical_review": {
            "status": "PENDING_MANUAL_REVIEW",
            "reason": "anonymous representative H/R figure generated for review",
        },
    }
    image_only_gate_names = (
        "finite_transform_success_rate",
        "catastrophic_transform_rate",
        "median_whole_anatomy_similarity_gain",
        "nonworse_moving_visit_fraction",
        "padding_increase",
    )
    image_only_pass = all(criteria[name]["status"] == "PASS" for name in image_only_gate_names)
    any_failure = any(item["status"] == "FAIL" for item in criteria.values())
    pending = any(str(item["status"]).startswith("PENDING") for item in criteria.values())
    if any_failure:
        decision = "C1B-H"
        decision_reason = "one or more preregistered registration gates failed"
    elif pending:
        decision = "HOLD_C1B-H_PENDING_REMAINING_GATES"
        decision_reason = "C1B-R cannot be selected until every pending preregistered audit passes"
    else:
        decision = "C1B-R"
        decision_reason = "all preregistered registration gates passed"

    distributions = {
        field: _metric_distribution(success, field)
        for field in (
            "optimizer_mattes_mi",
            "histogram_mi_before",
            "histogram_mi_after",
            "histogram_mi_gain",
            "histogram_nmi_before",
            "histogram_nmi_after",
            "histogram_nmi_gain",
            "similarity_before",
            "similarity_after",
            "similarity_gain",
            "translation_magnitude_mm",
            "rotation_magnitude_deg",
            "maximum_absolute_rotation_deg",
            "padding_fraction_before",
            "padding_fraction_after",
            "padding_fraction_delta",
            "valid_overlap_fraction_before",
            "valid_overlap_fraction_after",
            "anatomy_dice_before",
            "anatomy_dice_after",
            "anatomy_dice_gain",
        )
    }
    by_visit: dict[str, Any] = {}
    for visit in MOVING_VISITS:
        visit_rows = [row for row in rows if row["visit"] == visit]
        visit_success = [row for row in visit_rows if _truth(row["success"])]
        by_visit[visit] = {
            "pairs": len(visit_rows),
            "successes": len(visit_success),
            "failures": len(visit_rows) - len(visit_success),
            "success_rate": len(visit_success) / len(visit_rows) if visit_rows else 0.0,
            "median_similarity_gain": _quantile(
                (row["similarity_gain"] for row in visit_success), 0.5
            ),
            "median_translation_mm": _quantile(
                (row["translation_magnitude_mm"] for row in visit_success), 0.5
            ),
            "median_rotation_deg": _quantile(
                (row["rotation_magnitude_deg"] for row in visit_success), 0.5
            ),
        }
    repaired_rows = [
        row for row in rows if _truth(row["anchor_repaired"]) or _truth(row["source_repaired"])
    ]
    repaired_success = [row for row in repaired_rows if _truth(row["success"])]
    unrepaired_rows = [row for row in rows if row not in repaired_rows]
    unrepaired_success = [row for row in unrepaired_rows if _truth(row["success"])]

    return {
        "schema_version": 1,
        "analysis": "Stage A5 C1B-H versus C1B-R image-only registration sensitivity",
        "contains_patient_identifiers": False,
        "contains_paths": False,
        "registration_inputs": {
            "fixed": "T0 selected precontrast",
            "moving": "T1-T3 selected precontrast",
            "phase_policy": "legacy_adaptive_early_late_outcome_free",
            "automatic_mask": "whole-anatomy mask from precontrast intensity only",
            "uses_ftv_or_localization": False,
            "uses_clinical_treatment_response_or_pcr": False,
            "singular_visit_source": "repaired NIfTI required; no original fallback",
        },
        "cohort": {
            "expected_patients": expected_pairs // 3,
            "expected_pairs": expected_pairs,
            "observed_pairs": len(rows),
            "complete": len(rows) == expected_pairs,
        },
        "outcomes": {
            "successes": len(success),
            "failures": len(failures),
            "success_rate": success_rate,
            "failure_codes": dict(sorted(failure_counts.items())),
            "catastrophic_transforms": len(catastrophic),
            "catastrophic_rate_all_pairs": catastrophic_rate_all,
            "catastrophic_rate_successful_transforms": catastrophic_rate_success,
            "nonworse_pairs": len(nonworse),
            "nonworse_fraction_all_pairs_failures_count_as_not_nonworse": nonworse_fraction_all,
            "nonworse_fraction_successful_transforms": nonworse_fraction_success,
        },
        "repair_strata": {
            "at_least_one_repaired_visit": {
                "pairs": len(repaired_rows),
                "successes": len(repaired_success),
                "success_rate": (
                    len(repaired_success) / len(repaired_rows) if repaired_rows else None
                ),
            },
            "no_repaired_visit": {
                "pairs": len(unrepaired_rows),
                "successes": len(unrepaired_success),
                "success_rate": (
                    len(unrepaired_success) / len(unrepaired_rows) if unrepaired_rows else None
                ),
            },
        },
        "by_visit": by_visit,
        "distributions": distributions,
        "gate": {
            "thresholds": dict(thresholds),
            "criteria": criteria,
            "image_only_metrics_pass": image_only_pass,
            "decision": decision,
            "decision_reason": decision_reason,
        },
        "provenance": dict(provenance),
    }


def _write_public_tables(metrics_root: Path, summary: Mapping[str, Any]) -> None:
    failure_path = metrics_root / "registration_sensitivity_failure_codes.csv"
    with failure_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("failure_code", "count"))
        for code, count in summary["outcomes"]["failure_codes"].items():
            writer.writerow((code, count))

    visit_path = metrics_root / "registration_sensitivity_by_visit.csv"
    with visit_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "visit",
            "pairs",
            "successes",
            "failures",
            "success_rate",
            "median_similarity_gain",
            "median_translation_mm",
            "median_rotation_deg",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for visit, values in summary["by_visit"].items():
            writer.writerow({"visit": visit, **values})

    distribution_path = metrics_root / "registration_sensitivity_distributions.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "metric",
            "count",
            "mean",
            "minimum",
            "q05",
            "q25",
            "median",
            "q75",
            "q95",
            "maximum",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for metric, values in summary["distributions"].items():
            writer.writerow({"metric": metric, **values})


def _finite_column(rows: Iterable[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray(
        [value for row in rows if (value := _safe_float(row.get(field))) is not None],
        dtype=np.float64,
    )


def _plot_distributions(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    success = [row for row in rows if _truth(row["success"])]
    panels = (
        ("translation_magnitude_mm", "Translation magnitude (mm)", "#355C7D"),
        ("rotation_magnitude_deg", "Rotation magnitude (degrees)", "#C06C84"),
        ("similarity_gain", "Whole-anatomy NCC gain", "#6C5B7B"),
        ("histogram_nmi_gain", "Histogram NMI gain", "#2A9D8F"),
        ("padding_fraction_delta", "Padding fraction change", "#E76F51"),
        ("anatomy_dice_gain", "Anatomy Dice gain", "#457B9D"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    for axis, (field, label, color) in zip(axes.ravel(), panels):
        values = _finite_column(success, field)
        axis.hist(values, bins=36, color=color, alpha=0.88, edgecolor="white", linewidth=0.4)
        if values.size:
            axis.axvline(np.median(values), color="black", linestyle="--", linewidth=1.1)
        axis.set_xlabel(label)
        axis.set_ylabel("Successful pairs")
        axis.grid(alpha=0.18)
    figure.suptitle("Stage A5 image-only rigid-registration distributions", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def _localization_volume_mm3(item: tuple[str, str, str]) -> tuple[str, str, float | None]:
    patient_id, visit, path_text = item
    try:
        image = nib.load(path_text)
        mask = np.asanyarray(image.dataobj)
        if mask.ndim > 3:
            mask = mask[..., 0]
        count = int(np.count_nonzero(np.asarray(mask) > 0.5))
        voxel_volume = float(abs(np.linalg.det(np.asarray(image.affine)[:3, :3])))
        volume = float(count * voxel_volume)
        return patient_id, visit, volume if math.isfinite(volume) else None
    except Exception:
        return patient_id, visit, None


def _select_representatives(
    rows: list[dict[str, str]],
    localization_paths: Mapping[tuple[str, str], str],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    success = [row for row in rows if _truth(row["success"])]
    tasks = [
        (row["patient_id"], row["visit"], localization_paths[(row["patient_id"], row["visit"])])
        for row in success
    ]
    volumes: dict[tuple[str, str], float] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 12)), mp_context=context) as pool:
        for patient_id, visit, volume in pool.map(_localization_volume_mm3, tasks, chunksize=8):
            if volume is not None:
                volumes[(patient_id, visit)] = volume

    eligible = [
        row for row in success if (row["patient_id"], row["visit"]) in volumes
    ]
    volume_values = np.asarray(
        [volumes[(row["patient_id"], row["visit"])] for row in eligible], dtype=np.float64
    )
    if len(eligible) < 4:
        return [], {
            "status": "FAILED",
            "reason": "fewer than four successful pairs had readable localization masks",
            "localization_masks_read": len(volumes),
        }
    targets = {
        "small": float(np.quantile(volume_values, 0.20)),
        "medium": float(np.quantile(volume_values, 0.50)),
        "large": float(np.quantile(volume_values, 0.80)),
    }
    chosen: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for label, target in targets.items():
        candidates = sorted(
            eligible,
            key=lambda row: (
                abs(volumes[(row["patient_id"], row["visit"])] - target),
                abs(float(row["similarity_gain"]) - float(np.median(_finite_column(success, "similarity_gain")))),
                row["patient_id"],
                row["visit"],
            ),
        )
        selected = next(
            row for row in candidates if (row["patient_id"], row["visit"]) not in used
        )
        used.add((selected["patient_id"], selected["visit"]))
        chosen.append(
            {
                "slot": label,
                "localization_volume_mm3": volumes[(selected["patient_id"], selected["visit"])],
                "row": selected,
            }
        )
    high_transform = max(
        (
            row
            for row in eligible
            if (row["patient_id"], row["visit"]) not in used
        ),
        key=lambda row: (
            float(row["translation_magnitude_mm"]) / 75.0
            + float(row["maximum_absolute_rotation_deg"]) / 20.0,
            row["patient_id"],
            row["visit"],
        ),
    )
    chosen.append(
        {
            "slot": "high-transform",
            "localization_volume_mm3": volumes[(high_transform["patient_id"], high_transform["visit"])],
            "row": high_transform,
        }
    )
    audit = {
        "status": "PASS",
        "selection_use_only": True,
        "localization_was_registration_input": False,
        "localization_masks_read": len(volumes),
        "selection_quantiles": {"small": "Q20", "medium": "Q50", "large": "Q80"},
    }
    return chosen, audit


def _matrix_from_row(row: Mapping[str, Any]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    for matrix_row in range(4):
        for matrix_column in range(4):
            matrix[matrix_row, matrix_column] = float(
                row[f"source_to_anchor_ras_{matrix_row}{matrix_column}"]
            )
    return matrix


def _robust_display(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _representative_arrays(
    selection: Mapping[str, Any],
    phase_metadata: Mapping[str, Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    row = selection["row"]
    patient_id = row["patient_id"]
    metadata = phase_metadata.get(patient_id, {})
    anchor_visit = {
        "dce_path": row["anchor_dce_path"],
        "repaired": _truth(row["anchor_repaired"]),
        "inventory_phase_count": int(row["phase_count_anchor"]),
    }
    source_visit = {
        "dce_path": row["source_dce_path"],
        "repaired": _truth(row["source_repaired"]),
        "inventory_phase_count": int(row["phase_count_source"]),
    }
    anchor, _, _ = _precontrast_volume(anchor_visit, metadata)
    source, _, _ = _precontrast_volume(source_visit, metadata)
    fixed_image = canonical_volume_to_sitk(anchor)
    moving_image = canonical_volume_to_sitk(source)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    fitted = ras_matrix_to_sitk_transform(np.linalg.inv(_matrix_from_row(row)))
    h_image = _resample_xyz(
        moving_image, fixed_image, identity, sitk.sitkLinear, sitk.sitkFloat32
    )
    r_image = _resample_xyz(
        moving_image, fixed_image, fitted, sitk.sitkLinear, sitk.sitkFloat32
    )
    anatomy = _whole_anatomy_mask_array(anchor.data, minimum_voxels=128)
    slice_index = int(np.argmax(anatomy.sum(axis=(0, 1))))
    return anchor.data, h_image, r_image, slice_index


def _plot_representatives(
    selections: list[dict[str, Any]],
    phase_metadata: Mapping[str, Mapping[str, str]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 5, figsize=(14.5, 11.5), constrained_layout=True)
    column_titles = ("T0 anchor", "C1B-H source", "C1B-R source", "|T0-H|", "|T0-R|")
    for row_index, selection in enumerate(selections):
        anchor, h_image, r_image, slice_index = _representative_arrays(
            selection, phase_metadata
        )
        fixed_slice = _robust_display(anchor[:, :, slice_index]).T
        h_slice = _robust_display(h_image[:, :, slice_index]).T
        r_slice = _robust_display(r_image[:, :, slice_index]).T
        panels = (
            fixed_slice,
            h_slice,
            r_slice,
            np.abs(fixed_slice - h_slice),
            np.abs(fixed_slice - r_slice),
        )
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(
                panel,
                cmap="gray" if column_index < 3 else "magma",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks(())
            axis.set_yticks(())
            if row_index == 0:
                axis.set_title(column_titles[column_index])
            if column_index == 0:
                label = str(selection["slot"])
                axis.set_ylabel(label, fontsize=11)
        row = selection["row"]
        axes[row_index, 2].text(
            0.02,
            0.04,
            f"{row['visit']}  |  ΔNCC={float(row['similarity_gain']):+.3f}\n"
            f"t={float(row['translation_magnitude_mm']):.1f} mm, "
            f"r={float(row['rotation_magnitude_deg']):.1f}°",
            transform=axes[row_index, 2].transAxes,
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
    figure.suptitle(
        "Anonymous Stage A5 C1B-H/C1B-R technical-review cases\n"
        "Localization was used only for post-registration size-stratum selection",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    outcomes = summary["outcomes"]
    distributions = summary["distributions"]
    criteria = summary["gate"]["criteria"]
    failure_text = ", ".join(
        f"`{code}`={count}" for code, count in outcomes["failure_codes"].items()
    ) or "none"
    criterion_rows = []
    for name, item in criteria.items():
        observed = item.get("observed")
        if observed is None:
            observed = item.get("observed_all_pairs", item.get("observed_median", "—"))
        threshold = item.get("threshold", item.get("threshold_median", "—"))
        criterion_rows.append(
            f"| {name} | {item['status']} | {observed} | {threshold} |"
        )
    report = f"""# Stage A5 C1B-H vs C1B-R registration sensitivity

## Result

The preregistered strategy decision is **{summary['gate']['decision']}**: {summary['gate']['decision_reason']}.

The formal cohort contains {summary['cohort']['observed_pairs']}/{summary['cohort']['expected_pairs']} expected T1–T3→T0 pairs. Registration succeeded for {outcomes['successes']} pairs and failed closed for {outcomes['failures']} (success rate {outcomes['success_rate']:.4f}). Failure codes: {failure_text}.

## Image-only measurements

| Metric | Before median | After median | Gain/change median | Gain/change Q95 |
|---|---:|---:|---:|---:|
| Histogram MI | {distributions['histogram_mi_before']['median']} | {distributions['histogram_mi_after']['median']} | {distributions['histogram_mi_gain']['median']} | {distributions['histogram_mi_gain']['q95']} |
| Histogram NMI | {distributions['histogram_nmi_before']['median']} | {distributions['histogram_nmi_after']['median']} | {distributions['histogram_nmi_gain']['median']} | {distributions['histogram_nmi_gain']['q95']} |
| Whole-anatomy NCC | {distributions['similarity_before']['median']} | {distributions['similarity_after']['median']} | {distributions['similarity_gain']['median']} | {distributions['similarity_gain']['q95']} |
| Anatomy Dice | {distributions['anatomy_dice_before']['median']} | {distributions['anatomy_dice_after']['median']} | {distributions['anatomy_dice_gain']['median']} | {distributions['anatomy_dice_gain']['q95']} |
| Padding fraction | {distributions['padding_fraction_before']['median']} | {distributions['padding_fraction_after']['median']} | {distributions['padding_fraction_delta']['median']} | {distributions['padding_fraction_delta']['q95']} |
| Valid overlap | {distributions['valid_overlap_fraction_before']['median']} | {distributions['valid_overlap_fraction_after']['median']} | — | — |

Median translation magnitude was {distributions['translation_magnitude_mm']['median']} mm and median rotation magnitude was {distributions['rotation_magnitude_deg']['median']} degrees. Catastrophic transforms numbered {outcomes['catastrophic_transforms']} ({outcomes['catastrophic_rate_all_pairs']:.4f} of all pairs). The nonworse fraction, conservatively counting failures as not nonworse, was {outcomes['nonworse_fraction_all_pairs_failures_count_as_not_nonworse']:.4f}.

Histogram MI is the discrete joint-histogram mutual information in nats over common automatically segmented anatomy. NMI is `MI / sqrt(H_fixed * H_moving)`. Whole-anatomy similarity is masked Pearson/NCC. All masks used for these metrics and registration were derived only from each visit's selected precontrast image.

## Preregistered gates

| Criterion | Status | Observed | Threshold |
|---|---|---:|---:|
{chr(10).join(criterion_rows)}

Exact available-support containment and FTV retention are deliberately not inferred from image similarity. They remain pending the independent builder physical-support audit. The anatomy-versus-lesion residual condition requires its own landmark/residual audit. Anonymous small/medium/large/high-transform panels were generated for blinded manual review; the runner does not claim that visual gate passed.

## Leakage and privacy controls

Only the `pre`, `post_early`, and `post_late` acquisition columns were accessible to phase selection, with the frozen `0/min(2,T-1)/min(5,T-1)` defaults. Registration worker payloads structurally omit FTV/localization paths, lesion measurements, clinical variables, treatment, response, and pCR. Every flagged singular visit resolves to the repaired NIfTI with no fallback to the original singular file.

Localization masks were opened only after all transforms and pair metrics were complete, solely to choose anonymous size-stratified technical-review panels. Patient IDs, paths, per-pair transforms, failure messages, and panel mappings exist only in ignored `*.private.*` files. Public CSV/JSON outputs are aggregate and the PNGs contain no identifiers.
"""
    _atomic_text(path, report)


def _load_thresholds(config_path: Path) -> dict[str, float]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    gate = payload["registration_gate"]
    required = (
        "minimum_success_rate",
        "catastrophic_translation_mm",
        "catastrophic_rotation_deg",
        "maximum_catastrophic_rate",
        "minimum_median_similarity_gain",
        "minimum_nonworse_fraction",
        "nonworse_tolerance",
        "maximum_padding_median_increase",
        "maximum_padding_q95_increase",
    )
    return {key: float(gate[key]) for key in required}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--phase-metadata",
        type=Path,
        default=REPOSITORY_ROOT
        / "ispy_jepa_tmi_clean/data_processing/metadata/BreastDCEDL_metadata_min_crop.csv",
    )
    parser.add_argument(
        "--stage-a-config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs/stage_a.json",
    )
    parser.add_argument(
        "--repaired-root",
        type=Path,
        default=EXPERIMENT_ROOT / "repaired_volumes",
    )
    parser.add_argument(
        "--private-pair-metrics",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/registration_sensitivity_pairs.private.csv",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-representatives", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    registration_config = RegistrationConfig(maximum_iterations=600, number_of_threads=1)
    thresholds = _load_thresholds(args.stage_a_config)
    phase_metadata = _load_phase_metadata(args.phase_metadata)
    formal_rows, localization_paths = _load_formal_inventory(args.inventory)
    patients = _group_patients(formal_rows, phase_metadata, args.repaired_root)
    if len(patients) != 375:
        raise ValueError(f"formal Stage A5 cohort must contain 375 patients, found {len(patients)}")
    if args.limit_patients is not None:
        if args.limit_patients < 1:
            raise ValueError("limit-patients must be positive")
        patients = patients[: args.limit_patients]

    # Hashes and config values contain no patient identifiers or source paths.
    signature_payload = {
        "schema_version": 1,
        "registration_config": asdict(registration_config),
        "inventory_sha256": _sha256(args.inventory),
        "phase_metadata_sha256": _sha256(args.phase_metadata),
        "registration_source_sha256": _sha256(SOURCE_ROOT / "c1b_sanity/registration.py"),
        "formal_patient_count": len(patients),
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    existing = [] if args.no_resume else _read_private_rows(args.private_pair_metrics, run_signature)
    completed_keys = {(row["patient_id"], row["visit"]) for row in existing}
    pending_patients = [
        patient
        for patient in patients
        if any((patient["patient_id"], visit) not in completed_keys for visit in MOVING_VISITS)
    ]
    all_rows: list[dict[str, Any]] = list(existing)
    print(
        json.dumps(
            {
                "formal_patients": len(patients),
                "expected_pairs": len(patients) * 3,
                "resumed_pairs": len(existing),
                "pending_patients": len(pending_patients),
                "workers": args.workers,
                "maximum_iterations": registration_config.maximum_iterations,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    start = time.monotonic()
    if pending_patients:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            future_map = {
                pool.submit(
                    _process_patient,
                    {
                        "patient": patient,
                        "run_signature": run_signature,
                        "registration_config": asdict(registration_config),
                    },
                ): patient
                for patient in pending_patients
            }
            completed_patients = 0
            for future in as_completed(future_map):
                patient = future_map[future]
                try:
                    rows = future.result()
                except BaseException as exc:
                    rows = [
                        _failure_private_row(
                            patient,
                            visit,
                            run_signature,
                            "WORKER_EXCEPTION",
                            f"{type(exc).__name__}: {exc}",
                        )
                        for visit in MOVING_VISITS
                    ]
                keys = {(row["patient_id"], row["visit"]) for row in rows}
                all_rows = [
                    row
                    for row in all_rows
                    if (row["patient_id"], row["visit"]) not in keys
                ]
                all_rows.extend(rows)
                completed_patients += 1
                # Durable, sorted private checkpoint after each patient.
                _write_private_rows(args.private_pair_metrics, all_rows)
                if completed_patients % 10 == 0 or completed_patients == len(pending_patients):
                    successes = sum(_truth(row["success"]) for row in all_rows)
                    print(
                        json.dumps(
                            {
                                "completed_patients_this_run": completed_patients,
                                "total_patients_this_run": len(pending_patients),
                                "pairs_checkpointed": len(all_rows),
                                "successes_so_far": successes,
                                "elapsed_seconds": round(time.monotonic() - start, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    expected_pairs = len(patients) * 3
    if len(all_rows) != expected_pairs:
        raise RuntimeError(f"private metrics contain {len(all_rows)} rows, expected {expected_pairs}")
    _write_private_rows(args.private_pair_metrics, all_rows)
    # Re-read strings exactly as downstream aggregation will see them.
    final_rows = _read_private_rows(args.private_pair_metrics, run_signature)
    provenance = {
        **signature_payload,
        "run_signature": run_signature,
        "private_pair_metrics_sha256": _sha256(args.private_pair_metrics),
        "deterministic_sampling_seed": registration_config.random_seed,
        "simpleitk_version": sitk.Version_VersionString(),
    }
    summary = _aggregate(final_rows, expected_pairs, thresholds, provenance)

    metrics_root = EXPERIMENT_ROOT / "metrics"
    figures_root = EXPERIMENT_ROOT / "figures"
    reports_root = EXPERIMENT_ROOT / "reports"
    metrics_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(metrics_root / "registration_sensitivity_summary.json", summary)
    _write_public_tables(metrics_root, summary)
    _plot_distributions(
        final_rows, figures_root / "registration_transform_distributions.png"
    )

    representative_audit: dict[str, Any] = {"status": "SKIPPED"}
    if not args.skip_representatives:
        selections, representative_audit = _select_representatives(
            final_rows, localization_paths, args.workers
        )
        if selections:
            private_representatives = {
                "schema_version": 1,
                "contains_patient_identifiers": True,
                "selection_use_only": True,
                "localization_was_registration_input": False,
                "representatives": [
                    {
                        "slot": selection["slot"],
                        "patient_id": selection["row"]["patient_id"],
                        "visit": selection["row"]["visit"],
                        "anchor_dce_path": selection["row"]["anchor_dce_path"],
                        "source_dce_path": selection["row"]["source_dce_path"],
                        "localization_path": localization_paths[
                            (selection["row"]["patient_id"], selection["row"]["visit"])
                        ],
                        "localization_volume_mm3": selection["localization_volume_mm3"],
                        "source_to_anchor_ras": _matrix_from_row(selection["row"]).tolist(),
                    }
                    for selection in selections
                ],
            }
            _atomic_json(
                metrics_root / "registration_representatives.private.json",
                private_representatives,
            )
            _plot_representatives(
                selections,
                phase_metadata,
                figures_root / "registration_representative_hr.png",
            )
    summary["representative_figure_audit"] = representative_audit
    _atomic_json(metrics_root / "registration_sensitivity_summary.json", summary)
    _write_report(reports_root / "registration_sensitivity_report.md", summary)
    print(
        json.dumps(
            {
                "pairs": expected_pairs,
                "successes": summary["outcomes"]["successes"],
                "failures": summary["outcomes"]["failures"],
                "success_rate": summary["outcomes"]["success_rate"],
                "decision": summary["gate"]["decision"],
                "elapsed_seconds": round(time.monotonic() - start, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
