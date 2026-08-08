#!/usr/bin/env python3
"""运行 Response-Observable Multiscale Crop 的 outcome-free Stage A audit。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[3]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ispy_jepa_tmi_clean"))
from observable_crop.geometry import (  # noqa: E402
    PhysicalWindow,
    audit_prepared_support,
    bbox_footprint_in_frame,
    index_to_world,
    make_fixed_expand_window,
    make_tight_resize_window,
    make_union_window,
    orthonormal_index_basis,
    prepare_support,
    world_to_frame,
)
from observable_crop.nifti import (  # noqa: E402
    affine_max_corner_disagreement_mm,
    read_nifti_geometry,
    read_spatial_nifti_array,
)


VISITS = ("T0", "T1", "T2", "T3")
EXPECTED_WORKBOOK_SHA256 = (
    "f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc"
)
EXPECTED_OVERLAP_SHA256 = (
    "91b575c9e7e351312b8181a091bdffd2d1f61b88b5a98ac3d78d54c94b63da6b"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [finite_or_none(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(finite_or_none(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def q(values: Iterable[Any], quantile: float) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(array, quantile)) if len(array) else math.nan


def safe_spearman(left: Iterable[Any], right: Iterable[Any]) -> tuple[float, float, int]:
    x = pd.to_numeric(pd.Series(left), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(right), errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan, int(len(x))
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), int(len(x))


def require_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise SystemExit(f"缺少 {label}；请显式传参或设置对应环境变量")
    return value.expanduser().resolve(strict=True)


def load_inputs(args: argparse.Namespace, config: dict[str, Any]) -> pd.DataFrame:
    overlap = pd.read_csv(
        args.overlap,
        usecols=["patient_id", "clinical_patient_id", "has_radiomics"],
    )
    overlap = overlap.loc[overlap["has_radiomics"].astype(bool)].copy()
    overlap["patient_id"] = overlap["patient_id"].astype(str)
    if len(overlap) != int(config["cohort"]["strict_overlap_patients"]):
        raise ValueError(f"strict overlap 应为375，实际{len(overlap)}")
    if overlap["patient_id"].duplicated().any():
        raise ValueError("strict overlap patient_id 重复")

    workbook = pd.read_excel(
        args.workbook,
        sheet_name="datawith4visits",
        usecols=["CLINICAL-TRIAL-SUBJECT-ID", *(f"LD_{visit}" for visit in VISITS)],
    )
    workbook["clinical_patient_id"] = workbook["CLINICAL-TRIAL-SUBJECT-ID"].astype(int)
    if workbook["clinical_patient_id"].duplicated().any():
        raise ValueError("LD workbook ID 重复")
    merged = overlap.merge(
        workbook.drop(columns="CLINICAL-TRIAL-SUBJECT-ID"),
        on="clinical_patient_id",
        how="left",
        validate="one_to_one",
    )
    if merged[[f"LD_{visit}" for visit in VISITS]].isna().any().any():
        raise ValueError("strict overlap 存在 LD 缺失")
    return merged.sort_values("patient_id").reset_index(drop=True)


def choose_audit_affine(
    dce_geometry: Any,
    mask_geometry: Any,
    shape_xyz: tuple[int, int, int],
    tolerance_mm: float,
) -> dict[str, Any]:
    """Select an audit affine without pretending header repair fixes pixels."""

    mask_affine = mask_geometry.sform if mask_geometry.sform_valid else None
    if dce_geometry.sform_valid and mask_affine is not None:
        disagreement = affine_max_corner_disagreement_mm(
            dce_geometry.sform, mask_affine, shape_xyz
        )
        if disagreement <= tolerance_mm:
            return {
                "affine": dce_geometry.sform,
                "status": "TRUST_DCE_SFORM",
                "dce_mask_corner_disagreement_mm": disagreement,
                "geometry_auditable": True,
                "model_ready_geometry": True,
                "pixel_rebuild_required": False,
            }
        return {
            "affine": None,
            "status": "QUARANTINE_DCE_MASK_AFFINE_MISMATCH",
            "dce_mask_corner_disagreement_mm": disagreement,
            "geometry_auditable": False,
            "model_ready_geometry": False,
            "pixel_rebuild_required": True,
        }
    if dce_geometry.qform_valid and mask_affine is not None:
        disagreement = affine_max_corner_disagreement_mm(
            dce_geometry.qform, mask_affine, shape_xyz
        )
        if disagreement <= tolerance_mm:
            return {
                "affine": dce_geometry.qform,
                "status": "TRUST_DCE_QFORM_SFORM_SINGULAR",
                "dce_mask_corner_disagreement_mm": disagreement,
                "geometry_auditable": True,
                "model_ready_geometry": False,
                "pixel_rebuild_required": True,
            }
    if mask_affine is not None:
        disagreement = (
            affine_max_corner_disagreement_mm(
                dce_geometry.qform, mask_affine, shape_xyz
            )
            if dce_geometry.qform_valid
            else math.nan
        )
        return {
            "affine": mask_affine,
            "status": "MASK_SFORM_GEOMETRY_CANDIDATE_REBUILD_DICOM_PIXELS",
            "dce_mask_corner_disagreement_mm": disagreement,
            "geometry_auditable": True,
            "model_ready_geometry": False,
            "pixel_rebuild_required": True,
        }
    return {
        "affine": None,
        "status": "QUARANTINE_NO_RELIABLE_AFFINE",
        "dce_mask_corner_disagreement_mm": math.nan,
        "geometry_auditable": False,
        "model_ready_geometry": False,
        "pixel_rebuild_required": True,
    }


def physical_overlap_with_source(
    window: PhysicalWindow,
    affine: np.ndarray,
    shape_xyz: tuple[int, int, int],
) -> tuple[float, float]:
    source_low, source_high = bbox_footprint_in_frame(
        affine,
        window.frame_basis,
        (0, 0, 0),
        tuple(value - 1 for value in shape_xyz),
    )
    intersection = np.maximum(
        0.0,
        np.minimum(window.high_frame_mm, source_high)
        - np.maximum(window.low_frame_mm, source_low),
    )
    valid_volume = float(np.prod(intersection))
    fraction = float(valid_volume / np.prod(window.fov_xyz_mm))
    return valid_volume, float(np.clip(fraction, 0.0, 1.0))


def c0_window(
    legacy_row: pd.Series,
    affine: np.ndarray,
    frame_basis: np.ndarray,
) -> PhysicalWindow | None:
    starts = [legacy_row.get(f"crop_start_{axis}") for axis in "xyz"]
    if not all(pd.notna(value) for value in starts):
        return None
    start = np.asarray(starts, dtype=np.float64)
    size_xyz = np.asarray([96, 96, 32], dtype=np.float64)
    low, high = bbox_footprint_in_frame(
        affine, frame_basis, start, start + size_xyz - 1.0
    )
    return PhysicalWindow(
        contract="C0",
        view="legacy",
        frame_basis=frame_basis,
        center_frame_mm=0.5 * (low + high),
        fov_xyz_mm=high - low,
        output_shape_zyx=(32, 96, 96),
        anchor_policy="legacy_recovered_fixed_voxel_origin",
        reference_visit="T0_INDEX_PROJECTED",
        causal_deployability="LEGACY_REFERENCE",
        margin_mm=None,
    )


def build_windows(
    visits: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, list[PhysicalWindow]]:
    crop = config["physical_crop"]
    margin = float(crop["selected_margin_mm"])
    detail_fov = np.asarray(crop["detail_nominal_fov_xyz_mm"], dtype=float)
    detail_shape = tuple(int(value) for value in crop["detail_output_shape_zyx"])
    context_fov = np.asarray(crop["context_nominal_fov_xyz_mm"], dtype=float)
    context_shape = tuple(int(value) for value in crop["context_output_shape_zyx"])

    t0_affine = visits["T0"]["audit_affine"]
    t0_basis = orthonormal_index_basis(t0_affine)
    for visit in VISITS:
        visits[visit]["visit_basis"] = orthonormal_index_basis(
            visits[visit]["audit_affine"]
        )
        bbox = visits[visit]["bbox"]
        visits[visit]["visit_bounds"] = bbox_footprint_in_frame(
            visits[visit]["audit_affine"],
            visits[visit]["visit_basis"],
            bbox["min"],
            bbox["max"],
        )
        visits[visit]["t0_bounds"] = bbox_footprint_in_frame(
            visits[visit]["audit_affine"],
            t0_basis,
            bbox["min"],
            bbox["max"],
        )

    t0_low, t0_high = visits["T0"]["t0_bounds"]
    t0_center = 0.5 * (t0_low + t0_high)
    anchored_detail = make_fixed_expand_window(
        contract="C1B",
        view="detail",
        frame_basis=t0_basis,
        support_low_frame_mm=t0_low,
        support_high_frame_mm=t0_high,
        nominal_fov_xyz_mm=detail_fov,
        output_shape_zyx=detail_shape,
        margin_mm=margin,
        anchor_policy="T0_support_center_and_frame_unregistered_header_physical",
        reference_visit="T0",
        causal_deployability="T0_ANCHORED_CAUSAL",
        center_frame_mm=t0_center,
    )
    anchored_context = make_fixed_expand_window(
        contract="C2B",
        view="context",
        frame_basis=t0_basis,
        support_low_frame_mm=t0_low,
        support_high_frame_mm=t0_high,
        nominal_fov_xyz_mm=context_fov,
        output_shape_zyx=context_shape,
        margin_mm=margin,
        anchor_policy="T0_support_center_and_frame_unregistered_header_physical",
        reference_visit="T0",
        causal_deployability="T0_ANCHORED_CAUSAL",
        center_frame_mm=t0_center,
    )
    oracle = make_union_window(
        contract="C1C",
        view="detail",
        frame_basis=t0_basis,
        visit_bounds_frame_mm=[visits[visit]["t0_bounds"] for visit in VISITS],
        nominal_fov_xyz_mm=detail_fov,
        output_shape_zyx=detail_shape,
        margin_mm=margin,
    )

    output: dict[str, list[PhysicalWindow]] = {}
    for visit in VISITS:
        basis = visits[visit]["visit_basis"]
        low, high = visits[visit]["visit_bounds"]
        current_detail = make_fixed_expand_window(
            contract="C1A",
            view="detail",
            frame_basis=basis,
            support_low_frame_mm=low,
            support_high_frame_mm=high,
            nominal_fov_xyz_mm=detail_fov,
            output_shape_zyx=detail_shape,
            margin_mm=margin,
            anchor_policy="current_visit_support_center",
            reference_visit=visit,
            causal_deployability="CURRENT_VISIT_CAUSAL_TEMPORAL_RECENTER_RISK",
        )
        tight = make_tight_resize_window(
            contract="C1A-tight",
            view="detail",
            frame_basis=basis,
            support_low_frame_mm=low,
            support_high_frame_mm=high,
            output_shape_zyx=detail_shape,
            margin_mm=margin,
            reference_visit=visit,
        )
        current_context = make_fixed_expand_window(
            contract="C2A",
            view="context",
            frame_basis=basis,
            support_low_frame_mm=low,
            support_high_frame_mm=high,
            nominal_fov_xyz_mm=context_fov,
            output_shape_zyx=context_shape,
            margin_mm=margin,
            anchor_policy="current_visit_support_center",
            reference_visit=visit,
            causal_deployability="CURRENT_VISIT_CAUSAL_TEMPORAL_RECENTER_RISK",
        )
        c2a_detail = PhysicalWindow(
            **{
                **current_detail.__dict__,
                "contract": "C2A",
            }
        )
        c2b_detail = PhysicalWindow(
            **{
                **anchored_detail.__dict__,
                "contract": "C2B",
            }
        )
        output[visit] = [
            current_detail,
            tight,
            anchored_detail,
            oracle,
            c2a_detail,
            current_context,
            c2b_detail,
            anchored_context,
        ]
    return output


def manifest_bbox(visit: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    bbox = visit.get("bbox_nii_xyz_inclusive")
    if not bbox:
        raise ValueError(f"{visit.get('visit')} 缺 nonempty released support bbox")
    return {
        "min": (int(bbox["x_min"]), int(bbox["y_min"]), int(bbox["z_min"])),
        "max": (int(bbox["x_max"]), int(bbox["y_max"]), int(bbox["z_max"])),
    }


def geometry_record(
    patient_id: str,
    visit_name: str,
    visit: dict[str, Any],
    dce_geometry: Any,
    mask_geometry: Any,
    selection: dict[str, Any],
) -> dict[str, Any]:
    shape = tuple(int(value) for value in visit["dce_shape"][:3])
    spacing = tuple(float(value) for value in dce_geometry.spacing_xyz_mm)
    return {
        "patient_id": patient_id,
        "visit": visit_name,
        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],
        "raw_dce_phase_count": int(visit["dce_shape"][3]),
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "acquisition_fov_x_mm": shape[0] * spacing[0],
        "acquisition_fov_y_mm": shape[1] * spacing[1],
        "acquisition_fov_z_mm": shape[2] * spacing[2],
        "legacy_fov_x_mm": 96 * spacing[0],
        "legacy_fov_y_mm": 96 * spacing[1],
        "legacy_fov_z_mm": 32 * spacing[2],
        "dce_qform_code": dce_geometry.qform_code,
        "dce_sform_code": dce_geometry.sform_code,
        "dce_sform_valid": dce_geometry.sform_valid,
        "dce_sform_failure_reason": dce_geometry.sform_failure_reason,
        "dce_header_orientation": dce_geometry.orientation,
        "mask_header_orientation": mask_geometry.orientation,
        "mask_sform_valid": mask_geometry.sform_valid,
        "max_obliquity_deg": (
            float(dce_geometry.max_obliquity_deg)
            if dce_geometry.sform_valid
            else float(mask_geometry.max_obliquity_deg)
        ),
        "affine_decision": selection["status"],
        "dce_mask_corner_disagreement_mm": selection[
            "dce_mask_corner_disagreement_mm"
        ],
        "geometry_auditable": selection["geometry_auditable"],
        "model_ready_geometry": selection["model_ready_geometry"],
        "pixel_rebuild_required": selection["pixel_rebuild_required"],
        "axis_canonicalization": "none",
    }


def fallback_c0_record(old: pd.Series) -> dict[str, Any]:
    return {
        "full_support_voxels": int(old["full_support_voxels"]),
        "retained_support_voxels": int(old["cached_support_voxels"]),
        "retained_ftv_fraction": float(old["containment_ratio"]),
        "physical_volume_retention": float(old["containment_ratio"]),
        "exact_full_support_containment": bool(
            old["exact_full_support_containment"]
        ),
        "bbox_fully_contained": bool(old["bbox_fully_contained"]),
        "boundary_touch": bool(old["any_boundary_touch"]),
        "suspected_truncation": bool(old["suspected_truncation"]),
        "severe_truncation": bool(old["severe_truncation"]),
        "sufficient_containment": bool(old["sufficient_containment"]),
        "minimum_margin_mm": float(old["minimum_margin_mm"])
        if pd.notna(old["minimum_margin_mm"])
        else math.nan,
        "extent_retention_x": math.nan,
        "extent_retention_y": math.nan,
        "extent_retention_z": math.nan,
        "extent_retention_min_axis": float(old["whole_union_extent_retention_ratio"]),
        "surface_voxels": math.nan,
        "retained_surface_voxels": math.nan,
        "surface_voxel_retention": math.nan,
        "component_count": int(old["full_component_count"]),
        "cut_component_count": math.nan,
        "missed_component_count": math.nan,
        "lesion_physical_volume_mm3": float(old["full_support_voxels"])
        * float(old["spacing_x_mm"])
        * float(old["spacing_y_mm"])
        * float(old["spacing_z_mm"]),
        "window_physical_volume_mm3": float(old["crop_extent_x_mm"])
        * float(old["crop_extent_y_mm"])
        * float(old["crop_extent_z_mm"]),
        "context_to_lesion_volume_ratio": math.nan,
        "context_margin_x_low_mm": math.nan,
        "context_margin_x_high_mm": math.nan,
        "context_margin_y_low_mm": math.nan,
        "context_margin_y_high_mm": math.nan,
        "context_margin_z_low_mm": math.nan,
        "context_margin_z_high_mm": math.nan,
        "resize_factor_x": 1.0,
        "resize_factor_y": 1.0,
        "resize_factor_z": 1.0,
        "resize_anisotropy_ratio": 1.0,
        "output_anisotropy_ratio": float(
            max(old["spacing_x_mm"], old["spacing_y_mm"], old["spacing_z_mm"])
            / min(old["spacing_x_mm"], old["spacing_y_mm"], old["spacing_z_mm"])
        ),
    }


def audit_patient(
    row: Any,
    preprocessed_root: Path,
    legacy_by_key: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    patient_id = str(row.patient_id)
    manifest = json.loads(
        (preprocessed_root / patient_id / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_visits = {str(item["visit"]): item for item in manifest["visits"]}
    if set(manifest_visits) != set(VISITS):
        raise ValueError(f"{patient_id} manifest visits不完整")
    geom_config = config["geometry"]
    tolerance = float(geom_config["affine_match_atol_mm"])
    visits: dict[str, dict[str, Any]] = {}
    geometry_rows: list[dict[str, Any]] = []
    for visit_name in VISITS:
        visit = manifest_visits[visit_name]
        dce_geometry = read_nifti_geometry(
            visit["dce_nifti"],
            sform_spacing_rtol=float(geom_config["sform_spacing_rtol"]),
            sform_spacing_atol_mm=float(geom_config["sform_spacing_atol_mm"]),
        )
        mask_geometry = read_nifti_geometry(
            visit["ftv_mask_nifti"],
            sform_spacing_rtol=float(geom_config["sform_spacing_rtol"]),
            sform_spacing_atol_mm=float(geom_config["sform_spacing_atol_mm"]),
        )
        shape = tuple(int(value) for value in visit["dce_shape"][:3])
        manifest_dce_shape = tuple(int(value) for value in visit["dce_shape"])
        if tuple(int(value) for value in dce_geometry.shape) != manifest_dce_shape:
            raise ValueError(
                f"{patient_id}/{visit_name} DCE NIfTI shape与manifest不一致"
            )
        if len(manifest_dce_shape) != 4 or manifest_dce_shape[3] < 2:
            raise ValueError(
                f"{patient_id}/{visit_name} DCE必须为至少两phase的4-D NIfTI"
            )
        if tuple(mask_geometry.shape[:3]) != shape:
            raise ValueError(f"{patient_id}/{visit_name} DCE-mask shape mismatch")
        if not np.allclose(
            dce_geometry.spacing_xyz_mm,
            mask_geometry.spacing_xyz_mm,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"{patient_id}/{visit_name} DCE-mask spacing mismatch")
        selection = choose_audit_affine(
            dce_geometry, mask_geometry, shape, tolerance
        )
        if not selection["geometry_auditable"]:
            raise ValueError(
                f"{patient_id}/{visit_name} physical geometry quarantine: {selection['status']}"
            )
        visits[visit_name] = {
            "manifest": visit,
            "dce_geometry": dce_geometry,
            "mask_geometry": mask_geometry,
            "audit_affine": selection["affine"],
            "affine_selection": selection,
            "bbox": manifest_bbox(visit),
            "shape_xyz": shape,
        }
        geometry_rows.append(
            geometry_record(
                patient_id,
                visit_name,
                visit,
                dce_geometry,
                mask_geometry,
                selection,
            )
        )

    windows = build_windows(visits, config)
    records: list[dict[str, Any]] = []
    margin_records: list[dict[str, Any]] = []
    for visit_index, visit_name in enumerate(VISITS):
        visit_data = visits[visit_name]
        mask, meta = read_spatial_nifti_array(
            visit_data["manifest"]["ftv_mask_nifti"]
        )
        mask = np.asarray(mask > 0, dtype=bool)
        if mask.shape != visit_data["shape_xyz"]:
            raise ValueError(f"{patient_id}/{visit_name} reader shape drift: {mask.shape}")
        prepared = prepare_support(mask, visit_data["audit_affine"])
        prepared_bbox_min = tuple(
            int(value) for value in prepared.coordinates_xyz.min(axis=0)
        )
        prepared_bbox_max = tuple(
            int(value) for value in prepared.coordinates_xyz.max(axis=0)
        )
        if (
            prepared_bbox_min != tuple(visit_data["bbox"]["min"])
            or prepared_bbox_max != tuple(visit_data["bbox"]["max"])
        ):
            raise ValueError(
                f"{patient_id}/{visit_name} released bbox与实际support不一致"
            )
        source_boundary_touch = bool(
            np.any(prepared.coordinates_xyz == 0)
            or np.any(
                prepared.coordinates_xyz
                == (np.asarray(visit_data["shape_xyz"], dtype=int) - 1)[None, :]
            )
        )
        old = legacy_by_key.loc[(patient_id, visit_name)]
        reported_ld = float(getattr(row, f"LD_{visit_name}"))
        bbox = visit_data["bbox"]
        bbox_center_index = 0.5 * (
            np.asarray(bbox["min"], dtype=float) + np.asarray(bbox["max"], dtype=float)
        )
        bbox_center_world = index_to_world(
            bbox_center_index[None, :], visit_data["audit_affine"]
        )[0]
        support_centroid_index = prepared.coordinates_xyz.mean(axis=0)
        lesion_center_world = index_to_world(
            support_centroid_index[None, :], visit_data["audit_affine"]
        )[0]

        candidate_windows = list(windows[visit_name])
        legacy_window = c0_window(
            old,
            visit_data["audit_affine"],
            visit_data["visit_basis"],
        )
        if legacy_window is not None:
            candidate_windows.insert(0, legacy_window)

        audit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        for window in candidate_windows:
            key = (
                *np.round(window.frame_basis.ravel(), 8),
                *np.round(window.center_frame_mm, 6),
                *np.round(window.fov_xyz_mm, 6),
                *window.output_shape_zyx,
            )
            if key not in audit_cache:
                audit_cache[key] = audit_prepared_support(
                    prepared,
                    window,
                    retention_threshold=float(
                        config["metrics"]["retention_threshold"]
                    ),
                    severe_threshold=float(
                        config["metrics"]["severe_retention_threshold"]
                    ),
                ).to_record()
            audit = dict(audit_cache[key])
            audit["available_support_exact_containment"] = bool(
                audit["exact_full_support_containment"]
            )
            audit["source_boundary_touch"] = source_boundary_touch
            audit["source_boundary_uncensored"] = not source_boundary_touch
            # A crop can contain every available mask voxel while the source
            # acquisition may already have censored the lesion.  Preserve the
            # pre-registered crop-containment definitions and report this as a
            # separate upstream-censoring sensitivity; silently folding it into
            # suspected_truncation after seeing the data would change the gate.
            if window.contract == "C0":
                if audit["retained_support_voxels"] != int(old["cached_support_voxels"]):
                    raise ValueError(
                        f"{patient_id}/{visit_name} C0 source-domain count未复现legacy cache"
                    )
                # C0 reference must retain the exact historical definitions based
                # on the actual cached mask.  The physical source-domain audit is
                # still used for surface/components, but it must not silently
                # redefine the published legacy gate.
                audit.update(
                    {
                        "retained_support_voxels": int(old["cached_support_voxels"]),
                        "retained_ftv_fraction": float(old["containment_ratio"]),
                        "physical_volume_retention": float(old["containment_ratio"]),
                        "exact_full_support_containment": bool(
                            old["exact_full_support_containment"]
                        ),
                        "bbox_fully_contained": bool(old["bbox_fully_contained"]),
                        "boundary_touch": bool(old["any_boundary_touch"]),
                        "suspected_truncation": bool(old["suspected_truncation"]),
                        "severe_truncation": bool(old["severe_truncation"]),
                        "sufficient_containment": bool(old["sufficient_containment"]),
                        "minimum_margin_mm": float(old["minimum_margin_mm"])
                        if pd.notna(old["minimum_margin_mm"])
                        else math.nan,
                    }
                )
            valid_source_volume, valid_source_fraction = physical_overlap_with_source(
                window, visit_data["audit_affine"], visit_data["shape_xyz"]
            )
            lesion_volume = float(audit["lesion_physical_volume_mm3"])
            crop_center_world = window.center_frame_mm @ window.frame_basis.T
            lesion_center_in_crop_frame = world_to_frame(
                lesion_center_world[None, :], window.frame_basis
            )[0]
            relative = lesion_center_in_crop_frame - window.center_frame_mm
            relative_world = lesion_center_world - crop_center_world
            records.append(
                {
                    "patient_id": patient_id,
                    "visit": visit_name,
                    "visit_index": visit_index,
                    "reported_ld": reported_ld,
                    "ld_zero": reported_ld == 0.0,
                    "ld_unit_status": "LD_UNIT_NOT_EXPLICIT",
                    "support_source": "full_resolution_ftv_inclusion_region_proxy",
                    "geometry_affine_decision": visit_data["affine_selection"]["status"],
                    "geometry_model_ready": visit_data["affine_selection"][
                        "model_ready_geometry"
                    ],
                    **window.to_record(),
                    **audit,
                    "valid_source_volume_mm3_bbox": valid_source_volume,
                    "valid_source_fraction_bbox": valid_source_fraction,
                    "padding_fraction_bbox": 1.0 - valid_source_fraction,
                    "valid_context_to_lesion_volume_ratio": max(
                        valid_source_volume - lesion_volume, 0.0
                    )
                    / lesion_volume,
                    "crop_center_world_x_mm": float(crop_center_world[0]),
                    "crop_center_world_y_mm": float(crop_center_world[1]),
                    "crop_center_world_z_mm": float(crop_center_world[2]),
                    "lesion_center_world_x_mm": float(lesion_center_world[0]),
                    "lesion_center_world_y_mm": float(lesion_center_world[1]),
                    "lesion_center_world_z_mm": float(lesion_center_world[2]),
                    "lesion_center_definition": "FTV_SUPPORT_VOXEL_CENTROID",
                    "localization_bbox_center_world_x_mm": float(
                        bbox_center_world[0]
                    ),
                    "localization_bbox_center_world_y_mm": float(
                        bbox_center_world[1]
                    ),
                    "localization_bbox_center_world_z_mm": float(
                        bbox_center_world[2]
                    ),
                    "lesion_relative_x_mm": float(relative[0]),
                    "lesion_relative_y_mm": float(relative[1]),
                    "lesion_relative_z_mm": float(relative[2]),
                    "lesion_relative_distance_mm": float(np.linalg.norm(relative)),
                    "lesion_relative_world_x_mm": float(relative_world[0]),
                    "lesion_relative_world_y_mm": float(relative_world[1]),
                    "lesion_relative_world_z_mm": float(relative_world[2]),
                    "support_occupancy_physical": lesion_volume
                    / float(audit["window_physical_volume_mm3"]),
                    "approx_max_extent_largest_component_mm": float(
                        old["approx_max_extent_largest_component_mm"]
                    ),
                    "whole_union_extent_retention_ratio_legacy": float(
                        old["whole_union_extent_retention_ratio"]
                    ),
                }
            )

        if legacy_window is None:
            fallback = fallback_c0_record(old)
            records.append(
                {
                    "patient_id": patient_id,
                    "visit": visit_name,
                    "visit_index": visit_index,
                    "reported_ld": reported_ld,
                    "ld_zero": reported_ld == 0.0,
                    "ld_unit_status": "LD_UNIT_NOT_EXPLICIT",
                    "support_source": "full_resolution_ftv_inclusion_region_proxy",
                    "geometry_affine_decision": visit_data["affine_selection"]["status"],
                    "geometry_model_ready": visit_data["affine_selection"][
                        "model_ready_geometry"
                    ],
                    "contract": "C0",
                    "view": "legacy",
                    "anchor_policy": "legacy_origin_unrecoverable_fallback_metrics",
                    "reference_visit": "T0_INDEX_PROJECTED",
                    "causal_deployability": "LEGACY_REFERENCE",
                    "audit_only": False,
                    "margin_mm": math.nan,
                    "center_frame_x_mm": math.nan,
                    "center_frame_y_mm": math.nan,
                    "center_frame_z_mm": math.nan,
                    "fov_x_mm": float(old["crop_extent_x_mm"]),
                    "fov_y_mm": float(old["crop_extent_y_mm"]),
                    "fov_z_mm": float(old["crop_extent_z_mm"]),
                    "output_z": 32,
                    "output_y": 96,
                    "output_x": 96,
                    "effective_spacing_x_mm": float(old["spacing_x_mm"]),
                    "effective_spacing_y_mm": float(old["spacing_y_mm"]),
                    "effective_spacing_z_mm": float(old["spacing_z_mm"]),
                    "expanded_from_nominal": False,
                    "direct_bbox_resize": False,
                    **fallback,
                    "available_support_exact_containment": bool(
                        fallback["exact_full_support_containment"]
                    ),
                    "source_boundary_touch": source_boundary_touch,
                    "source_boundary_uncensored": not source_boundary_touch,
                    "valid_source_volume_mm3_bbox": math.nan,
                    "valid_source_fraction_bbox": math.nan,
                    "padding_fraction_bbox": math.nan,
                    "valid_context_to_lesion_volume_ratio": math.nan,
                    "crop_center_world_x_mm": math.nan,
                    "crop_center_world_y_mm": math.nan,
                    "crop_center_world_z_mm": math.nan,
                    "lesion_center_world_x_mm": float(lesion_center_world[0]),
                    "lesion_center_world_y_mm": float(lesion_center_world[1]),
                    "lesion_center_world_z_mm": float(lesion_center_world[2]),
                    "lesion_center_definition": "FTV_SUPPORT_VOXEL_CENTROID",
                    "localization_bbox_center_world_x_mm": float(
                        bbox_center_world[0]
                    ),
                    "localization_bbox_center_world_y_mm": float(
                        bbox_center_world[1]
                    ),
                    "localization_bbox_center_world_z_mm": float(
                        bbox_center_world[2]
                    ),
                    "lesion_relative_x_mm": math.nan,
                    "lesion_relative_y_mm": math.nan,
                    "lesion_relative_z_mm": math.nan,
                    "lesion_relative_distance_mm": math.nan,
                    "lesion_relative_world_x_mm": math.nan,
                    "lesion_relative_world_y_mm": math.nan,
                    "lesion_relative_world_z_mm": math.nan,
                    "support_occupancy_physical": float(
                        fallback["lesion_physical_volume_mm3"]
                        / fallback["window_physical_volume_mm3"]
                    ),
                    "approx_max_extent_largest_component_mm": float(
                        old["approx_max_extent_largest_component_mm"]
                    ),
                    "whole_union_extent_retention_ratio_legacy": float(
                        old["whole_union_extent_retention_ratio"]
                    ),
                }
            )

        visit_low, visit_high = visit_data["visit_bounds"]
        for margin in config["physical_crop"]["margin_candidates_mm"]:
            t0_basis = visits["T0"]["visit_basis"]
            t0_low, t0_high = visits["T0"]["t0_bounds"]
            t0_center = 0.5 * (t0_low + t0_high)
            margin_candidates = (
                (
                    "C1A",
                    make_fixed_expand_window(
                        contract=f"C1A-M{int(margin):02d}",
                        view="detail",
                        frame_basis=visit_data["visit_basis"],
                        support_low_frame_mm=visit_low,
                        support_high_frame_mm=visit_high,
                        nominal_fov_xyz_mm=config["physical_crop"][
                            "detail_nominal_fov_xyz_mm"
                        ],
                        output_shape_zyx=config["physical_crop"][
                            "detail_output_shape_zyx"
                        ],
                        margin_mm=float(margin),
                        anchor_policy="current_visit_support_center",
                        reference_visit=visit_name,
                        causal_deployability="MARGIN_SENSITIVITY",
                    ),
                ),
                (
                    "C1B",
                    make_fixed_expand_window(
                        contract=f"C1B-M{int(margin):02d}",
                        view="detail",
                        frame_basis=t0_basis,
                        support_low_frame_mm=t0_low,
                        support_high_frame_mm=t0_high,
                        nominal_fov_xyz_mm=config["physical_crop"][
                            "detail_nominal_fov_xyz_mm"
                        ],
                        output_shape_zyx=config["physical_crop"][
                            "detail_output_shape_zyx"
                        ],
                        margin_mm=float(margin),
                        anchor_policy=(
                            "T0_support_center_and_frame_unregistered_header_physical"
                        ),
                        reference_visit="T0",
                        causal_deployability="MARGIN_SENSITIVITY",
                        center_frame_mm=t0_center,
                    ),
                ),
            )
            for strategy, candidate in margin_candidates:
                margin_audit = audit_prepared_support(
                    prepared,
                    candidate,
                    retention_threshold=float(
                        config["metrics"]["retention_threshold"]
                    ),
                    severe_threshold=float(
                        config["metrics"]["severe_retention_threshold"]
                    ),
                )
                margin_records.append(
                    {
                        "patient_id": patient_id,
                        "visit": visit_name,
                        "reported_ld": reported_ld,
                        "strategy": strategy,
                        "margin_mm": float(margin),
                        "expanded_from_nominal": candidate.expanded_from_nominal,
                        "fov_x_mm": candidate.fov_xyz_mm[0],
                        "fov_y_mm": candidate.fov_xyz_mm[1],
                        "fov_z_mm": candidate.fov_xyz_mm[2],
                        "source_boundary_touch": source_boundary_touch,
                        "source_boundary_uncensored": not source_boundary_touch,
                        "suspected_truncation": margin_audit.suspected_truncation,
                        "severe_truncation": margin_audit.severe_truncation,
                        "sufficient_containment": margin_audit.sufficient_containment,
                        "exact_full_support_containment": (
                            margin_audit.exact_full_support_containment
                        ),
                        "minimum_margin_mm": margin_audit.minimum_margin_mm,
                        "retained_ftv_fraction": margin_audit.retained_ftv_fraction,
                        "resize_factor_x": margin_audit.resize_factor_x,
                        "resize_factor_y": margin_audit.resize_factor_y,
                        "resize_factor_z": margin_audit.resize_factor_z,
                        "resize_anisotropy_ratio": (
                            margin_audit.resize_anisotropy_ratio
                        ),
                    }
                )
    return records, geometry_rows, margin_records


def add_ld_flags(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    thresholds: dict[str, tuple[float, float]] = {}
    reference = output.loc[
        output["contract"].eq("C0") & output["view"].eq("legacy"),
        ["visit", "reported_ld"],
    ]
    for visit, subset in reference.groupby("visit", sort=False):
        thresholds[str(visit)] = (
            q(subset["reported_ld"], 0.75),
            q(subset["reported_ld"], 0.90),
        )
    output["ld_q75_visit"] = output["visit"].map(
        {visit: values[0] for visit, values in thresholds.items()}
    )
    output["ld_q90_visit"] = output["visit"].map(
        {visit: values[1] for visit, values in thresholds.items()}
    )
    output["ld_top25"] = output["reported_ld"] >= output["ld_q75_visit"]
    output["ld_top10"] = output["reported_ld"] >= output["ld_q90_visit"]
    return output


def summarize_containment(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (contract, view), group in frame.groupby(["contract", "view"], sort=False):
        scopes = [("OVERALL", group)] + [
            (visit, group.loc[group["visit"].eq(visit)]) for visit in VISITS
        ]
        for scope, subset in scopes:
            rho, pvalue, rho_n = safe_spearman(
                subset["reported_ld"], subset["minimum_margin_mm"]
            )
            rows.append(
                {
                    "contract": contract,
                    "view": view,
                    "scope": scope,
                    "n": len(subset),
                    "n_patients": subset["patient_id"].nunique(),
                    "boundary_touch_rate": float(subset["boundary_touch"].mean()),
                    "suspected_truncation_rate": float(
                        subset["suspected_truncation"].mean()
                    ),
                    "severe_truncation_rate": float(
                        subset["severe_truncation"].mean()
                    ),
                    "sufficient_containment_rate": float(
                        subset["sufficient_containment"].mean()
                    ),
                    "exact_full_support_containment_rate": float(
                        subset["exact_full_support_containment"].mean()
                    ),
                    "bbox_fully_contained_rate": float(
                        subset["bbox_fully_contained"].mean()
                    ),
                    "source_boundary_touch_rate": float(
                        subset["source_boundary_touch"].mean()
                    ),
                    "source_boundary_uncensored_rate": float(
                        subset["source_boundary_uncensored"].mean()
                    ),
                    "available_support_exact_containment_rate": float(
                        subset["available_support_exact_containment"].mean()
                    ),
                    "source_uncensored_and_sufficient_rate": float(
                        (
                            subset["source_boundary_uncensored"]
                            & subset["sufficient_containment"]
                        ).mean()
                    ),
                    "source_uncensored_and_exact_rate": float(
                        (
                            subset["source_boundary_uncensored"]
                            & subset["exact_full_support_containment"]
                        ).mean()
                    ),
                    "retained_ftv_fraction_median": q(
                        subset["retained_ftv_fraction"], 0.50
                    ),
                    "retained_ftv_fraction_q05": q(
                        subset["retained_ftv_fraction"], 0.05
                    ),
                    "retained_ftv_fraction_min": q(
                        subset["retained_ftv_fraction"], 0.00
                    ),
                    "minimum_margin_mm_median": q(
                        subset["minimum_margin_mm"], 0.50
                    ),
                    "minimum_margin_mm_q05": q(
                        subset["minimum_margin_mm"], 0.05
                    ),
                    "ld_margin_spearman": rho,
                    "ld_margin_pvalue": pvalue,
                    "ld_margin_n": rho_n,
                }
            )
    summary = pd.DataFrame(rows)
    by_visit = summary.loc[summary["scope"].isin(VISITS)].copy()
    return summary, by_visit


def summarize_large_ld(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (contract, view), group in frame.groupby(["contract", "view"], sort=False):
        for flag, label in (("ld_top25", "LD_TOP_QUARTILE"), ("ld_top10", "LD_TOP_10PCT")):
            for scope, subset in [("OVERALL", group.loc[group[flag]])] + [
                (visit, group.loc[group["visit"].eq(visit) & group[flag]])
                for visit in VISITS
            ]:
                rows.append(
                    {
                        "contract": contract,
                        "view": view,
                        "subgroup": label,
                        "scope": scope,
                        "n": len(subset),
                        "suspected_truncation_rate": float(
                            subset["suspected_truncation"].mean()
                        )
                        if len(subset)
                        else math.nan,
                        "severe_truncation_rate": float(
                            subset["severe_truncation"].mean()
                        )
                        if len(subset)
                        else math.nan,
                        "sufficient_containment_rate": float(
                            subset["sufficient_containment"].mean()
                        )
                        if len(subset)
                        else math.nan,
                        "exact_full_support_containment_rate": float(
                            subset["exact_full_support_containment"].mean()
                        )
                        if len(subset)
                        else math.nan,
                        "retained_ftv_fraction_q05": q(
                            subset["retained_ftv_fraction"], 0.05
                        ),
                        "minimum_margin_mm_median": q(
                            subset["minimum_margin_mm"], 0.50
                        ),
                        "minimum_margin_mm_min": q(
                            subset["minimum_margin_mm"], 0.00
                        ),
                        "source_boundary_censored_rate": float(
                            subset["source_boundary_touch"].mean()
                        )
                        if len(subset)
                        else math.nan,
                        "upstream_censoring_adjusted_suspected_rate": float(
                            (
                                subset["suspected_truncation"]
                                | subset["source_boundary_touch"]
                            ).mean()
                        )
                        if len(subset)
                        else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_ld_rank_sanity(frame: pd.DataFrame) -> pd.DataFrame:
    """Outcome-free LD rank sanity against the full-support physical extent.

    The C1A window is centered on the current visit's complete support.  Its
    stored six margins therefore recover the full support footprint extent
    without using a cropped or resampled mask.  This remains a proxy rank
    comparison: it is not a unit conversion and not a radiologist-target
    segmentation claim.
    """

    reference = frame.loc[
        frame["contract"].eq("C1A") & frame["view"].eq("detail")
    ].copy()
    extent_columns: list[str] = []
    for axis in "xyz":
        column = f"full_support_extent_{axis}_mm"
        reference[column] = (
            reference[f"fov_{axis}_mm"]
            - reference[f"context_margin_{axis}_low_mm"]
            - reference[f"context_margin_{axis}_high_mm"]
        )
        extent_columns.append(column)
    reference["available_full_support_max_extent_mm"] = reference[
        extent_columns
    ].max(axis=1)

    scopes: list[tuple[str, pd.DataFrame]] = [
        ("OVERALL", reference),
        ("T0_T1", reference.loc[reference["visit"].isin(["T0", "T1"])]),
        *[
            (visit, reference.loc[reference["visit"].eq(visit)])
            for visit in VISITS
        ],
    ]
    rows: list[dict[str, Any]] = []
    for scope, subset in scopes:
        bbox_rho, bbox_pvalue, bbox_sample_count = safe_spearman(
            subset["reported_ld"],
            subset["available_full_support_max_extent_mm"],
        )
        component_rho, component_pvalue, component_sample_count = safe_spearman(
            subset["reported_ld"],
            subset["approx_max_extent_largest_component_mm"],
        )
        rows.append(
            {
                "scope": scope,
                "n": len(subset),
                "ld_zero_fraction": float(subset["ld_zero"].mean()),
                "ld_full_support_bbox_extent_spearman": bbox_rho,
                "ld_full_support_bbox_extent_pvalue": bbox_pvalue,
                "ld_full_support_bbox_extent_n": bbox_sample_count,
                "ld_largest_component_extent_spearman": component_rho,
                "ld_largest_component_extent_pvalue": component_pvalue,
                "ld_largest_component_extent_n": component_sample_count,
                "full_support_bbox_extent_median_mm": q(
                    subset["available_full_support_max_extent_mm"], 0.50
                ),
                "full_support_bbox_extent_q95_mm": q(
                    subset["available_full_support_max_extent_mm"], 0.95
                ),
                "largest_component_extent_median_mm": q(
                    subset["approx_max_extent_largest_component_mm"], 0.50
                ),
                "largest_component_extent_q95_mm": q(
                    subset["approx_max_extent_largest_component_mm"], 0.95
                ),
                "extent_proxy": (
                    "FULL_SUPPORT_BBOX_AND_LARGEST_COMPONENT_APPROXIMATE_EXTENT_"
                    "NOT_RADIOLOGIST_TARGET"
                ),
                "ld_unit_status": "LD_SOURCE_UNIT_NOT_EXPLICIT_NO_CONVERSION",
            }
        )
    return pd.DataFrame(rows)


def summarize_ftv(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (contract, view, visit), subset in frame.groupby(
        ["contract", "view", "visit"], sort=False
    ):
        rows.append(
            {
                "contract": contract,
                "view": view,
                "scope": visit,
                "n": len(subset),
                "median": q(subset["retained_ftv_fraction"], 0.50),
                "q05": q(subset["retained_ftv_fraction"], 0.05),
                "q25": q(subset["retained_ftv_fraction"], 0.25),
                "minimum": q(subset["retained_ftv_fraction"], 0.00),
                "physical_volume_retention_median": q(
                    subset["physical_volume_retention"], 0.50
                ),
                "physical_volume_retention_q05": q(
                    subset["physical_volume_retention"], 0.05
                ),
                "physical_volume_retention_q25": q(
                    subset["physical_volume_retention"], 0.25
                ),
                "physical_volume_retention_minimum": q(
                    subset["physical_volume_retention"], 0.00
                ),
                "extent_retention_min_axis_median": q(
                    subset["extent_retention_min_axis"], 0.50
                ),
                "extent_retention_min_axis_q05": q(
                    subset["extent_retention_min_axis"], 0.05
                ),
                "extent_retention_min_axis_q25": q(
                    subset["extent_retention_min_axis"], 0.25
                ),
                "extent_retention_min_axis_minimum": q(
                    subset["extent_retention_min_axis"], 0.00
                ),
            }
        )
    for (contract, view), subset in frame.groupby(["contract", "view"], sort=False):
        rows.append(
            {
                "contract": contract,
                "view": view,
                "scope": "OVERALL",
                "n": len(subset),
                "median": q(subset["retained_ftv_fraction"], 0.50),
                "q05": q(subset["retained_ftv_fraction"], 0.05),
                "q25": q(subset["retained_ftv_fraction"], 0.25),
                "minimum": q(subset["retained_ftv_fraction"], 0.00),
                "physical_volume_retention_median": q(
                    subset["physical_volume_retention"], 0.50
                ),
                "physical_volume_retention_q05": q(
                    subset["physical_volume_retention"], 0.05
                ),
                "physical_volume_retention_q25": q(
                    subset["physical_volume_retention"], 0.25
                ),
                "physical_volume_retention_minimum": q(
                    subset["physical_volume_retention"], 0.00
                ),
                "extent_retention_min_axis_median": q(
                    subset["extent_retention_min_axis"], 0.50
                ),
                "extent_retention_min_axis_q05": q(
                    subset["extent_retention_min_axis"], 0.05
                ),
                "extent_retention_min_axis_q25": q(
                    subset["extent_retention_min_axis"], 0.25
                ),
                "extent_retention_min_axis_minimum": q(
                    subset["extent_retention_min_axis"], 0.00
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_morphology(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (contract, view), subset in frame.groupby(["contract", "view"], sort=False):
        available = subset["surface_voxel_retention"].notna()
        rows.append(
            {
                "contract": contract,
                "view": view,
                "n": len(subset),
                "surface_available_fraction": float(available.mean()),
                "surface_retention_median": q(
                    subset["surface_voxel_retention"], 0.50
                ),
                "surface_retention_q05": q(subset["surface_voxel_retention"], 0.05),
                "surface_retention_minimum": q(
                    subset["surface_voxel_retention"], 0.00
                ),
                "bbox_containment_rate": float(
                    subset["bbox_fully_contained"].mean()
                ),
                "source_boundary_censored_rate": float(
                    subset["source_boundary_touch"].mean()
                ),
                "fully_observable_surface_rate": float(
                    (
                        subset["surface_voxel_retention"].eq(1.0)
                        & subset["source_boundary_uncensored"]
                    ).mean()
                ),
                "any_cut_component_rate": float(
                    subset["cut_component_count"].fillna(0).gt(0).mean()
                ),
                "any_missed_component_rate": float(
                    subset["missed_component_count"].fillna(0).gt(0).mean()
                ),
                "cut_components_total": int(
                    subset["cut_component_count"].fillna(0).sum()
                ),
                "missed_components_total": int(
                    subset["missed_component_count"].fillna(0).sum()
                ),
                "component_count_median": q(subset["component_count"], 0.50),
                "proxy_semantics": "FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION",
            }
        )
    return pd.DataFrame(rows)


def summarize_context(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (contract, view), subset in frame.groupby(["contract", "view"], sort=False):
        six_margin_table = subset[
            [
                "context_margin_x_low_mm",
                "context_margin_x_high_mm",
                "context_margin_y_low_mm",
                "context_margin_y_high_mm",
                "context_margin_z_low_mm",
                "context_margin_z_high_mm",
            ]
        ]
        six_margins = six_margin_table.min(axis=1)
        median_margins = six_margin_table.median(axis=1)
        rows.append(
            {
                "contract": contract,
                "view": view,
                "n": len(subset),
                "minimum_context_margin_median_mm": q(six_margins, 0.50),
                "minimum_context_margin_q05_mm": q(six_margins, 0.05),
                "minimum_context_margin_min_mm": q(six_margins, 0.00),
                "median_context_margin_median_mm": q(median_margins, 0.50),
                "median_context_margin_q05_mm": q(median_margins, 0.05),
                "median_context_margin_min_mm": q(median_margins, 0.00),
                "valid_context_to_lesion_volume_ratio_median": q(
                    subset["valid_context_to_lesion_volume_ratio"], 0.50
                ),
                "valid_context_to_lesion_volume_ratio_q05": q(
                    subset["valid_context_to_lesion_volume_ratio"], 0.05
                ),
                "valid_source_fraction_median": q(
                    subset["valid_source_fraction_bbox"], 0.50
                ),
                "padding_fraction_median": q(subset["padding_fraction_bbox"], 0.50),
                "padding_fraction_q95": q(subset["padding_fraction_bbox"], 0.95),
                "padding_fraction_gt_50pct_rate": float(
                    subset["padding_fraction_bbox"].gt(0.50).mean()
                ),
                "support_occupancy_median": q(
                    subset["support_occupancy_physical"], 0.50
                ),
                "support_occupancy_q05": q(
                    subset["support_occupancy_physical"], 0.05
                ),
                "fov_x_mm_median": q(subset["fov_x_mm"], 0.50),
                "fov_y_mm_median": q(subset["fov_y_mm"], 0.50),
                "fov_z_mm_median": q(subset["fov_z_mm"], 0.50),
            }
        )
    return pd.DataFrame(rows)


def summarize_resampling(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (contract, view), subset in frame.groupby(["contract", "view"], sort=False):
        resize_factors = subset[
            ["resize_factor_x", "resize_factor_y", "resize_factor_z"]
        ].to_numpy(float)
        resize_anisotropy = resize_factors.max(axis=1) / resize_factors.min(axis=1)
        row: dict[str, Any] = {
            "contract": contract,
            "view": view,
            "n": len(subset),
            "expanded_from_nominal_rate": float(
                subset["expanded_from_nominal"].mean()
            ),
            "direct_bbox_resize": bool(subset["direct_bbox_resize"].all()),
            "resize_anisotropy_median": q(resize_anisotropy, 0.50),
            "resize_anisotropy_q95": q(resize_anisotropy, 0.95),
            "resize_anisotropy_max": q(resize_anisotropy, 1.00),
            "output_spacing_anisotropy_median": q(
                subset["output_anisotropy_ratio"], 0.50
            ),
            "output_spacing_anisotropy_q95": q(
                subset["output_anisotropy_ratio"], 0.95
            ),
            "output_spacing_anisotropy_max": q(
                subset["output_anisotropy_ratio"], 1.00
            ),
            "extreme_axis_factor_gt2_rate": float(
                subset[["resize_factor_x", "resize_factor_y", "resize_factor_z"]]
                .gt(2.0)
                .any(axis=1)
                .mean()
            ),
            "extreme_axis_factor_lt0_5_rate": float(
                subset[["resize_factor_x", "resize_factor_y", "resize_factor_z"]]
                .lt(0.5)
                .any(axis=1)
                .mean()
            ),
            "support_occupancy_cv": float(
                subset["support_occupancy_physical"].std(ddof=0)
                / max(subset["support_occupancy_physical"].mean(), 1e-12)
            ),
        }
        for axis in "xyz":
            column = subset[f"resize_factor_{axis}"]
            for label, quantile in (
                ("min", 0.0),
                ("q05", 0.05),
                ("median", 0.50),
                ("q95", 0.95),
                ("max", 1.0),
            ):
                row[f"resize_factor_{axis}_{label}"] = q(column, quantile)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_temporal(frame: pd.DataFrame) -> pd.DataFrame:
    patient_rows: list[dict[str, Any]] = []
    for (contract, view, patient_id), subset in frame.groupby(
        ["contract", "view", "patient_id"], sort=False
    ):
        subset = subset.sort_values("visit_index")
        if len(subset) != 4:
            raise ValueError(f"{contract}/{view}/{patient_id} visits不为4")
        crop = subset[
            [
                "crop_center_world_x_mm",
                "crop_center_world_y_mm",
                "crop_center_world_z_mm",
            ]
        ].to_numpy(float)
        lesion = subset[
            [
                "lesion_center_world_x_mm",
                "lesion_center_world_y_mm",
                "lesion_center_world_z_mm",
            ]
        ].to_numpy(float)
        relative = subset[
            [
                "lesion_relative_world_x_mm",
                "lesion_relative_world_y_mm",
                "lesion_relative_world_z_mm",
            ]
        ].to_numpy(float)
        crop_drift = np.linalg.norm(crop - crop[0], axis=1)
        lesion_drift = np.linalg.norm(lesion - lesion[0], axis=1)
        relative_change = np.linalg.norm(relative - relative[0], axis=1)
        valid = np.isfinite(crop_drift) & np.isfinite(lesion_drift)
        follow_ratio = np.divide(
            crop_drift,
            lesion_drift,
            out=np.full(4, np.nan),
            where=lesion_drift > 1e-6,
        )
        fov = subset[["fov_x_mm", "fov_y_mm", "fov_z_mm"]].to_numpy(float)
        spacing = subset[
            [
                "effective_spacing_x_mm",
                "effective_spacing_y_mm",
                "effective_spacing_z_mm",
            ]
        ].to_numpy(float)
        resize = subset[
            ["resize_factor_x", "resize_factor_y", "resize_factor_z"]
        ].to_numpy(float)
        valid_source_fraction = subset["valid_source_fraction_bbox"].to_numpy(float)
        valid_context_ratio = subset[
            "valid_context_to_lesion_volume_ratio"
        ].to_numpy(float)
        support_occupancy = subset["support_occupancy_physical"].to_numpy(float)

        def maximum_relative_change(values: np.ndarray) -> float:
            baseline = np.asarray(values[0], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                change = np.abs(np.asarray(values, dtype=float) / baseline - 1.0)
            return float(np.nanmax(change)) if np.any(np.isfinite(change)) else math.nan

        patient_rows.append(
            {
                "contract": contract,
                "view": view,
                "patient_id": patient_id,
                "crop_center_drift_max_mm": float(np.nanmax(crop_drift)),
                "lesion_center_drift_max_mm": float(np.nanmax(lesion_drift)),
                "relative_position_change_max_mm": float(
                    np.nanmax(relative_change)
                ),
                "crop_follow_lesion_ratio_median": float(
                    np.nanmedian(follow_ratio[1:])
                )
                if np.any(np.isfinite(follow_ratio[1:]))
                else math.nan,
                "fov_max_relative_change": float(
                    np.nanmax(np.abs(fov / fov[0] - 1.0))
                ),
                "spacing_max_relative_change": float(
                    np.nanmax(np.abs(spacing / spacing[0] - 1.0))
                ),
                "resize_factor_max_relative_change": float(
                    np.nanmax(np.abs(resize / resize[0] - 1.0))
                ),
                "valid_source_fraction_range": float(
                    np.nanmax(valid_source_fraction)
                    - np.nanmin(valid_source_fraction)
                )
                if np.any(np.isfinite(valid_source_fraction))
                else math.nan,
                "valid_context_ratio_max_relative_change": (
                    maximum_relative_change(valid_context_ratio)
                ),
                "support_occupancy_max_relative_change": (
                    maximum_relative_change(support_occupancy)
                ),
                "valid_temporal_geometry": bool(np.all(valid)),
            }
        )
    patient = pd.DataFrame(patient_rows)
    rows: list[dict[str, Any]] = []
    for (contract, view), subset in patient.groupby(["contract", "view"], sort=False):
        rows.append(
            {
                "contract": contract,
                "view": view,
                "n_patients": len(subset),
                "valid_temporal_geometry_fraction": float(
                    subset["valid_temporal_geometry"].mean()
                ),
                "crop_center_drift_median_mm": q(
                    subset["crop_center_drift_max_mm"], 0.50
                ),
                "crop_center_drift_q95_mm": q(
                    subset["crop_center_drift_max_mm"], 0.95
                ),
                "crop_center_drift_max_mm": q(
                    subset["crop_center_drift_max_mm"], 1.00
                ),
                "lesion_center_drift_median_mm": q(
                    subset["lesion_center_drift_max_mm"], 0.50
                ),
                "lesion_center_drift_q95_mm": q(
                    subset["lesion_center_drift_max_mm"], 0.95
                ),
                "relative_position_change_median_mm": q(
                    subset["relative_position_change_max_mm"], 0.50
                ),
                "crop_follow_lesion_ratio_median": q(
                    subset["crop_follow_lesion_ratio_median"], 0.50
                ),
                "fov_max_relative_change_q95": q(
                    subset["fov_max_relative_change"], 0.95
                ),
                "effective_spacing_max_relative_change_q95": q(
                    subset["spacing_max_relative_change"], 0.95
                ),
                "resize_factor_max_relative_change_median": q(
                    subset["resize_factor_max_relative_change"], 0.50
                ),
                "resize_factor_max_relative_change_q95": q(
                    subset["resize_factor_max_relative_change"], 0.95
                ),
                "resize_factor_max_relative_change_max": q(
                    subset["resize_factor_max_relative_change"], 1.00
                ),
                "valid_source_fraction_range_median": q(
                    subset["valid_source_fraction_range"], 0.50
                ),
                "valid_source_fraction_range_q95": q(
                    subset["valid_source_fraction_range"], 0.95
                ),
                "valid_context_ratio_relative_change_q95": q(
                    subset["valid_context_ratio_max_relative_change"], 0.95
                ),
                "support_occupancy_relative_change_q95": q(
                    subset["support_occupancy_max_relative_change"], 0.95
                ),
                "temporal_interpretation": (
                    "RECENTERING_REMOVES_LESION_MOTION"
                    if contract in {"C1A", "C1A-tight", "C2A"}
                    else "FIXED_T0_WINDOW_PRESERVES_HEADER_FRAME_MOTION"
                    if contract in {"C1B", "C2B"}
                    else "AUDIT_ONLY_OR_LEGACY"
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_geometry(
    geometry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    measurement_columns = [
        "shape_x",
        "shape_y",
        "shape_z",
        "raw_dce_phase_count",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "legacy_fov_x_mm",
        "legacy_fov_y_mm",
        "legacy_fov_z_mm",
        "acquisition_fov_x_mm",
        "acquisition_fov_y_mm",
        "acquisition_fov_z_mm",
    ]
    rows: list[dict[str, Any]] = []
    for scope, subset in [("OVERALL", geometry)] + [
        (visit, geometry.loc[geometry["visit"].eq(visit)]) for visit in VISITS
    ]:
        for measurement in measurement_columns:
            rows.append(
                {
                    "scope": scope,
                    "measurement": measurement,
                    "n": subset[measurement].notna().sum(),
                    "median": q(subset[measurement], 0.50),
                    "q25": q(subset[measurement], 0.25),
                    "q75": q(subset[measurement], 0.75),
                    "minimum": q(subset[measurement], 0.00),
                    "maximum": q(subset[measurement], 1.00),
                }
            )
    summary = pd.DataFrame(rows)
    variation: list[dict[str, Any]] = []
    for axis in "xyz":
        grouped = geometry.groupby("patient_id")[f"spacing_{axis}_mm"]
        ranges = grouped.max() - grouped.min()
        ratios = grouped.max() / grouped.min()
        variation.append(
            {
                "axis": axis.upper(),
                "patients": geometry["patient_id"].nunique(),
                "patients_with_visit_variation": int((ranges > 1e-6).sum()),
                "fraction_with_visit_variation": float((ranges > 1e-6).mean()),
                "range_median_mm": q(ranges, 0.50),
                "range_q95_mm": q(ranges, 0.95),
                "range_max_mm": q(ranges, 1.00),
                "max_to_min_ratio_q95": q(ratios, 0.95),
                "max_to_min_ratio_max": q(ratios, 1.00),
            }
        )
    return summary, pd.DataFrame(variation)


def summarize_margins(
    margins: pd.DataFrame,
    minimum_public_cell: int,
) -> pd.DataFrame:
    work = margins.copy()
    ld_reference = work[
        ["patient_id", "visit", "reported_ld"]
    ].drop_duplicates(["patient_id", "visit"])
    thresholds = {
        str(visit): (
            q(subset["reported_ld"], 0.75),
            q(subset["reported_ld"], 0.90),
        )
        for visit, subset in ld_reference.groupby("visit", sort=False)
    }
    work["ld_q75_visit"] = work["visit"].map(
        {visit: values[0] for visit, values in thresholds.items()}
    )
    work["ld_q90_visit"] = work["visit"].map(
        {visit: values[1] for visit, values in thresholds.items()}
    )
    work["ld_top25"] = work["reported_ld"] >= work["ld_q75_visit"]
    work["ld_top10"] = work["reported_ld"] >= work["ld_q90_visit"]
    rows: list[dict[str, Any]] = []
    for (strategy, margin), subset in work.groupby(["strategy", "margin_mm"]):
        top25_by_visit = [
            group.loc[group["ld_top25"], "suspected_truncation"].mean()
            for _, group in subset.groupby("visit", sort=False)
        ]
        top10_by_visit = [
            group.loc[group["ld_top10"], "suspected_truncation"].mean()
            for _, group in subset.groupby("visit", sort=False)
        ]
        resize = subset[
            ["resize_factor_x", "resize_factor_y", "resize_factor_z"]
        ]
        expanded_patient_count = int(
            subset.loc[subset["expanded_from_nominal"], "patient_id"].nunique()
        )
        expanded_reportable = expanded_patient_count >= minimum_public_cell
        rows.append(
            {
                "strategy": strategy,
                "margin_mm": margin,
                "n": len(subset),
                "n_patients": subset["patient_id"].nunique(),
                "exact_full_support_containment_rate": float(
                    subset["exact_full_support_containment"].mean()
                ),
                "source_boundary_uncensored_rate": float(
                    subset["source_boundary_uncensored"].mean()
                ),
                "sufficient_containment_rate": float(
                    subset["sufficient_containment"].mean()
                ),
                "suspected_truncation_rate": float(
                    subset["suspected_truncation"].mean()
                ),
                "large_ld_top_quartile_worst_visit_suspected_rate": float(
                    np.nanmax(top25_by_visit)
                ),
                "large_ld_top_10pct_worst_visit_suspected_rate": float(
                    np.nanmax(top10_by_visit)
                ),
                "retained_ftv_fraction_q05": q(
                    subset["retained_ftv_fraction"], 0.05
                ),
                "minimum_margin_q05_mm": q(subset["minimum_margin_mm"], 0.05),
                "expanded_from_nominal_rate": float(
                    subset["expanded_from_nominal"].mean()
                )
                if expanded_reportable
                else math.nan,
                "expanded_patient_count_public": (
                    str(expanded_patient_count)
                    if expanded_reportable
                    else f"<{minimum_public_cell}"
                ),
                "expanded_count_status": (
                    "REPORTED"
                    if expanded_reportable
                    else f"SUPPRESSED_LT_{minimum_public_cell}"
                ),
                "extreme_axis_factor_gt2_rate": float(
                    resize.gt(2.0).any(axis=1).mean()
                ),
                "resize_anisotropy_max": float(
                    (
                        resize.max(axis=1)
                        / resize.min(axis=1)
                    ).max()
                ),
                "fov_x_mm_median": q(subset["fov_x_mm"], 0.50),
                "fov_y_mm_median": q(subset["fov_y_mm"], 0.50),
                "fov_z_mm_median": q(subset["fov_z_mm"], 0.50),
                "fov_volume_liter_median": q(
                    subset["fov_x_mm"]
                    * subset["fov_y_mm"]
                    * subset["fov_z_mm"]
                    / 1e6,
                    0.50,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_grid_selection_basis(
    frame: pd.DataFrame,
    geometry: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Document the outcome-free data basis for spacing and nominal FOV."""

    margin = float(config["physical_crop"]["selected_margin_mm"])
    adaptive = frame.loc[
        frame["contract"].eq("C1A") & frame["view"].eq("detail")
    ].copy()
    anchored = frame.loc[
        frame["contract"].eq("C1B") & frame["view"].eq("detail")
    ].copy()
    rows: list[dict[str, Any]] = []
    for axis_index, axis in enumerate("xyz"):
        adaptive_extent = (
            adaptive[f"fov_{axis}_mm"]
            - adaptive[f"context_margin_{axis}_low_mm"]
            - adaptive[f"context_margin_{axis}_high_mm"]
        )
        adaptive_required = adaptive_extent + 2.0 * margin
        anchored_exact = (
            anchored[f"fov_{axis}_mm"]
            - 2.0
            * anchored[
                [
                    f"context_margin_{axis}_low_mm",
                    f"context_margin_{axis}_high_mm",
                ]
            ].min(axis=1)
        )
        anchored_required_by_patient = (
            pd.DataFrame(
                {
                    "patient_id": anchored["patient_id"],
                    "exact": anchored_exact,
                    "with_margin": anchored_exact + 2.0 * margin,
                }
            )
            .groupby("patient_id", sort=False)[["exact", "with_margin"]]
            .max()
        )
        native_spacing = geometry[f"spacing_{axis}_mm"]
        acquisition_fov = geometry[f"acquisition_fov_{axis}_mm"]
        rows.append(
            {
                "axis": axis.upper(),
                "native_spacing_median_mm": q(native_spacing, 0.50),
                "native_spacing_q90_mm": q(native_spacing, 0.90),
                "native_spacing_q95_mm": q(native_spacing, 0.95),
                "detail_target_spacing_mm": float(
                    config["physical_crop"]["detail_common_spacing_xyz_mm"][
                        axis_index
                    ]
                ),
                "context_target_spacing_mm": float(
                    config["physical_crop"]["context_effective_spacing_xyz_mm"][
                        axis_index
                    ]
                ),
                "visit_adaptive_bbox_plus_margin_q95_mm": q(
                    adaptive_required, 0.95
                ),
                "visit_adaptive_bbox_extent_median_mm": q(
                    adaptive_extent, 0.50
                ),
                "visit_adaptive_bbox_extent_q95_mm": q(
                    adaptive_extent, 0.95
                ),
                "visit_adaptive_bbox_extent_q99_mm": q(
                    adaptive_extent, 0.99
                ),
                "visit_adaptive_bbox_extent_max_mm": q(
                    adaptive_extent, 1.00
                ),
                "visit_adaptive_bbox_plus_margin_q99_mm": q(
                    adaptive_required, 0.99
                ),
                "visit_adaptive_bbox_plus_margin_max_mm": q(
                    adaptive_required, 1.00
                ),
                "t0_anchored_all_visit_exact_q95_mm": q(
                    anchored_required_by_patient["exact"], 0.95
                ),
                "t0_anchored_all_visit_exact_q99_mm": q(
                    anchored_required_by_patient["exact"], 0.99
                ),
                "t0_anchored_all_visit_margin_q95_mm": q(
                    anchored_required_by_patient["with_margin"], 0.95
                ),
                "t0_anchored_all_visit_margin_q99_mm": q(
                    anchored_required_by_patient["with_margin"], 0.99
                ),
                "acquisition_fov_q05_mm": q(acquisition_fov, 0.05),
                "acquisition_fov_median_mm": q(acquisition_fov, 0.50),
                "acquisition_fov_q95_mm": q(acquisition_fov, 0.95),
                "detail_nominal_fov_mm": float(
                    config["physical_crop"]["detail_nominal_fov_xyz_mm"][
                        axis_index
                    ]
                ),
                "context_nominal_fov_mm": float(
                    config["physical_crop"]["context_nominal_fov_xyz_mm"][
                        axis_index
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_tensor_footprint(config: dict[str, Any]) -> pd.DataFrame:
    """Raw float32 input footprint before activations or optimizer state."""

    visits = len(config["cohort"]["visits"])
    channels = int(config["legacy"]["dce_channels"])
    legacy_voxels = int(np.prod(config["legacy"]["crop_size_zyx"]))
    detail_voxels = int(
        np.prod(config["physical_crop"]["detail_output_shape_zyx"])
    )
    context_voxels = int(
        np.prod(config["physical_crop"]["context_output_shape_zyx"])
    )
    rows: list[dict[str, Any]] = []
    for contract, views, spatial_voxels in (
        ("C0", "legacy", legacy_voxels),
        ("C1B", "detail", detail_voxels),
        ("C2B", "detail+context", detail_voxels + context_voxels),
    ):
        values = visits * channels * spatial_voxels
        rows.append(
            {
                "contract": contract,
                "views": views,
                "visits": visits,
                "dce_channels": channels,
                "spatial_voxels_per_visit": spatial_voxels,
                "float32_values_per_patient": values,
                "float32_megabytes_per_patient": values * 4 / 1e6,
                "relative_to_legacy": spatial_voxels / legacy_voxels,
                "excludes_activations_optimizer_and_padding_mask": True,
            }
        )
    return pd.DataFrame(rows)


def decide_gate(
    frame: pd.DataFrame,
    containment: pd.DataFrame,
    large_ld: pd.DataFrame,
    ftv: pd.DataFrame,
    morphology: pd.DataFrame,
    resampling: pd.DataFrame,
    temporal: pd.DataFrame,
    geometry: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    # A multiscale contract is only as safe as its least-safe branch.  C2 must
    # therefore aggregate detail and context for containment, FTV/surface,
    # temporal, geometry and resampling checks; evaluating only the detail row
    # would hide the deliberately coarser context branch's distortion burden.
    candidates = (
        ("C1A", ("detail",)),
        ("C1B", ("detail",)),
        ("C2A", ("detail", "context")),
        ("C2B", ("detail", "context")),
    )
    results: dict[str, Any] = {}
    for contract, views in candidates:
        overall_rows = containment.loc[
            containment["contract"].eq(contract)
            & containment["view"].isin(views)
            & containment["scope"].eq("OVERALL")
        ]
        top25_rows = large_ld.loc[
            large_ld["contract"].eq(contract)
            & large_ld["view"].isin(views)
            & large_ld["scope"].isin(VISITS)
            & large_ld["subgroup"].eq("LD_TOP_QUARTILE")
        ]
        top10_rows = large_ld.loc[
            large_ld["contract"].eq(contract)
            & large_ld["view"].isin(views)
            & large_ld["scope"].isin(VISITS)
            & large_ld["subgroup"].eq("LD_TOP_10PCT")
        ]
        ftv_rows = ftv.loc[
            ftv["contract"].eq(contract)
            & ftv["view"].isin(views)
            & ftv["scope"].eq("OVERALL")
        ]
        morphology_rows = morphology.loc[
            morphology["contract"].eq(contract)
            & morphology["view"].isin(views)
        ]
        resize_rows = resampling.loc[
            resampling["contract"].eq(contract)
            & resampling["view"].isin(views)
        ]
        temporal_rows = temporal.loc[
            temporal["contract"].eq(contract)
            & temporal["view"].isin(views)
        ]
        subset = frame.loc[
            frame["contract"].eq(contract) & frame["view"].isin(views)
        ]
        expected_rows = len(views)
        for name, rows in (
            ("containment", overall_rows),
            ("FTV", ftv_rows),
            ("morphology", morphology_rows),
            ("resampling", resize_rows),
            ("temporal", temporal_rows),
        ):
            if len(rows) != expected_rows:
                raise ValueError(
                    f"{contract} {name} rows应为{expected_rows}，实际{len(rows)}"
                )
        expected_large_ld_rows = len(views) * len(VISITS)
        for name, rows in (
            ("large-LD Q75", top25_rows),
            ("large-LD Q90", top10_rows),
        ):
            if len(rows) != expected_large_ld_rows:
                raise ValueError(
                    f"{contract} {name} rows应为{expected_large_ld_rows}，"
                    f"实际{len(rows)}"
                )
        sufficient_rate = float(
            overall_rows["sufficient_containment_rate"].min()
        )
        exact_rate = float(
            overall_rows["exact_full_support_containment_rate"].min()
        )
        top25_suspected = float(
            top25_rows["suspected_truncation_rate"].max()
        )
        top10_suspected = float(
            top10_rows["suspected_truncation_rate"].max()
        )
        top25_upstream_adjusted = float(
            top25_rows["upstream_censoring_adjusted_suspected_rate"].max()
        )
        top10_upstream_adjusted = float(
            top10_rows["upstream_censoring_adjusted_suspected_rate"].max()
        )
        ftv_q05 = float(ftv_rows["q05"].min())
        surface_q05 = float(morphology_rows["surface_retention_q05"].min())
        max_resize_factor = float(
            resize_rows[
                [
                    "resize_factor_x_max",
                    "resize_factor_y_max",
                    "resize_factor_z_max",
                ]
            ].to_numpy(float).max()
        )
        max_anisotropy = float(resize_rows["resize_anisotropy_max"].max())
        max_obliquity = float(geometry["max_obliquity_deg"].max())
        fixed_t0 = contract in {"C1B", "C2B"}
        checks = {
            "overall_sufficient_containment": bool(
                sufficient_rate >= float(gate["overall_sufficient_containment_min"])
            ),
            "overall_exact_full_support": bool(
                exact_rate >= float(gate["overall_exact_full_support_min"])
            ),
            "large_ld_top_quartile": bool(
                top25_suspected
                <= float(gate["top_quartile_suspected_truncation_max"])
            ),
            "large_ld_top_10pct": bool(
                top10_suspected <= float(gate["top_10pct_suspected_truncation_max"])
            ),
            "ftv_retention_q05": bool(
                ftv_q05 >= float(gate["ftv_retention_q05_min"])
            ),
            "morphology_surface_q05": bool(surface_q05 >= 0.95),
            "source_boundary_observability": bool(
                overall_rows["source_uncensored_and_exact_rate"].min()
                >= float(gate["overall_exact_full_support_min"])
            ),
            "upstream_acquisition_large_ld_sensitivity": bool(
                top25_upstream_adjusted
                <= float(gate["top_quartile_suspected_truncation_max"])
                and top10_upstream_adjusted
                <= float(gate["top_10pct_suspected_truncation_max"])
            ),
            "no_extreme_downsampling": bool(
                max_resize_factor <= float(gate["max_effective_axis_scale_factor"])
            ),
            "anisotropy": bool(
                max_anisotropy <= float(gate["max_anisotropy_ratio"])
            ),
            "temporal_no_recentering_normalization": bool(
                fixed_t0
                and temporal_rows["crop_center_drift_q95_mm"].max() <= 1e-4
                and temporal_rows["fov_max_relative_change_q95"].max() <= 1e-8
            ),
            "temporal_frame_validated": bool(
                fixed_t0
                and config["geometry"][
                    "image_only_rigid_registration_sensitivity_completed"
                ]
            ),
            "orientation_contract_validated": bool(
                config["geometry"][
                    "production_orientation_canonicalization_validated"
                ]
            ),
            "geometry_model_ready": bool(subset["geometry_model_ready"].all()),
            "model_input_pipeline_validated": bool(
                config["physical_crop"][
                    "model_ready_3d_dce7_pipeline_validated"
                ]
            ),
            "causally_deployable": bool(
                contract in {"C1A", "C1B", "C2A", "C2B"}
                and not subset["audit_only"].any()
            ),
            "no_direct_geometry_input": bool(
                not config["leakage"]["mask_is_model_input"]
                and not config["leakage"]["crop_scale_metadata_is_model_input"]
                and not config["leakage"]["bbox_geometry_is_model_input"]
            ),
        }
        results[contract] = {
            "contract": contract,
            "view": "+".join(views),
            "evaluated_views": list(views),
            "large_ld_gate_scope": "WORST_VISIT_AND_VIEW",
            "large_ld_primary_definition": (
                "PRE_REGISTERED_CROP_SUSPECTED_TRUNCATION_ON_AVAILABLE_SUPPORT"
            ),
            "source_boundary_sensitivity_does_not_redefine_primary_metrics": True,
            "checks": checks,
            "passed": bool(all(checks.values())),
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "observed": {
                "overall_sufficient_containment_rate": sufficient_rate,
                "overall_exact_full_support_rate": exact_rate,
                "top_quartile_suspected_truncation_rate": top25_suspected,
                "top_10pct_suspected_truncation_rate": top10_suspected,
                "top_quartile_upstream_censoring_adjusted_suspected_rate": (
                    top25_upstream_adjusted
                ),
                "top_10pct_upstream_censoring_adjusted_suspected_rate": (
                    top10_upstream_adjusted
                ),
                "ftv_retention_q05": ftv_q05,
                "surface_retention_q05": surface_q05,
                "max_resize_factor": max_resize_factor,
                "max_resize_anisotropy_ratio": max_anisotropy,
                "max_cardinal_obliquity_deg": max_obliquity,
                "source_uncensored_and_exact_rate": float(
                    overall_rows["source_uncensored_and_exact_rate"].min()
                ),
                "geometry_model_ready_fraction": float(
                    subset["geometry_model_ready"].mean()
                ),
            },
        }

    passed = [name for name, result in results.items() if result["passed"]]
    containment_pass = [
        name
        for name, result in results.items()
        if all(
            result["checks"][key]
            for key in (
                "overall_sufficient_containment",
                "overall_exact_full_support",
                "large_ld_top_quartile",
                "large_ld_top_10pct",
                "ftv_retention_q05",
                "morphology_surface_q05",
                "source_boundary_observability",
            )
        )
    ]
    if passed:
        decision = "INPUT-CONTRACT GO"
        reason = "至少一个 causally deployable candidate 通过全部预注册 gate"
    elif containment_pass:
        decision = "INPUT-CONTRACT PARTIAL"
        reason = (
            "available-support crop observability子门通过，但上游acquisition边界、"
            "geometry、temporal、resampling或model-input pipeline仍有未解决项"
        )
    else:
        decision = "INPUT-CONTRACT NO-GO"
        reason = "没有deployable candidate可靠通过lesion/large-LD/FTV observability gate"
    return {
        "schema_version": 1,
        "stage": "A_INPUT_OBSERVABILITY_ONLY",
        "decision": decision,
        "reason": reason,
        "candidate_results": results,
        "passing_candidates": passed,
        "containment_passing_candidates": containment_pass,
        "oracle_union_is_never_deployable": True,
        "source_boundary_policy": (
            "REPORT_AS_UPSTREAM_CENSORING_SENSITIVITY_WITHOUT_POSTHOC_"
            "REDEFINING_PRE_REGISTERED_CROP_METRICS"
        ),
        "stage_b_authorized": bool(decision == "INPUT-CONTRACT GO"),
        "ld_grounding_authorized": False,
        "thresholds": gate,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
            )
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_physical_geometry_report(
    geometry: pd.DataFrame,
    summary: pd.DataFrame,
    variation: pd.DataFrame,
    grid_basis: pd.DataFrame,
    config: dict[str, Any],
    path: Path,
) -> None:
    status = geometry["affine_decision"].value_counts().to_dict()
    orientation = geometry["mask_header_orientation"].value_counts().to_dict()
    obliquity_flag = float(
        config["geometry"]["cardinal_obliquity_qc_flag_deg"]
    )
    minimum_public_cell = int(config["metrics"]["minimum_public_cell"])
    obliquity_count = int(
        geometry["max_obliquity_deg"].gt(obliquity_flag).sum()
    )
    obliquity_count_public = (
        str(obliquity_count)
        if obliquity_count >= minimum_public_cell
        else f"<{minimum_public_cell}"
    )
    selected = summary.loc[
        summary["measurement"].isin(
            [
                "shape_x",
                "shape_y",
                "shape_z",
                "raw_dce_phase_count",
                "spacing_x_mm",
                "spacing_y_mm",
                "spacing_z_mm",
                "legacy_fov_x_mm",
                "legacy_fov_y_mm",
                "legacy_fov_z_mm",
                "acquisition_fov_x_mm",
                "acquisition_fov_y_mm",
                "acquisition_fov_z_mm",
            ]
        )
    ].copy()
    report = f"""# 物理几何审计

## 结论

已完成375人、1,500个visit的shape、spacing、axis order、qform/sform、orientation及current crop physical FOV复核。历史reader的1,500个visit均采用index `XYZ`、无axis transpose，但它不读取affine，因此过去的shape/spacing match不能被解释为world-space registration。

- mask orientation计数：`{json.dumps(orientation, ensure_ascii=False, sort_keys=True)}`；同一患者四访方向一致。
- affine decision计数：`{json.dumps(status, ensure_ascii=False, sort_keys=True)}`。
- 当前仍沿native/T0 index basis采样，尚未执行跨患者anatomical orientation canonicalization。LAS/RPS等方向虽已审计，却不能仅凭common spacing视为common anatomical orientation；production必须冻结统一方向策略或验证native-orientation contract。
- DCE sform奇异：{int((~geometry['dce_sform_valid']).sum())}/1,500；这些visit必须按raw DICOM重建pixel order后才可写model-ready cache。
- 当前header-only audit可计算physical containment，但model-ready geometry比例仅为{geometry['model_ready_geometry'].mean():.1%}；不能把mask sform candidate误称为已修复DCE image。
- obliquity>{obliquity_flag:g}°的QC flag为{obliquity_count_public}/1,500，最大{geometry['max_obliquity_deg'].max():.2f}°。physical affine sampler支持正交oblique acquisition，因此这是QC/sensitivity flag，不把斜扫本身误判为几何失败。

## Spacing与legacy physical FOV

下表的IQR为Q25–Q75；spacing/FOV使用mm，shape与phase count为无量纲计数。

{markdown_table(selected, ['scope','measurement','n','median','q25','q75','minimum','maximum'])}

固定`32×96×96 voxel`并不对应固定物理视野。总体median约为X={q(geometry['legacy_fov_x_mm'],.5):.2f}、Y={q(geometry['legacy_fov_y_mm'],.5):.2f}、Z={q(geometry['legacy_fov_z_mm'],.5):.2f} mm；范围分别为{q(geometry['legacy_fov_x_mm'],0):.2f}–{q(geometry['legacy_fov_x_mm'],1):.2f}、{q(geometry['legacy_fov_y_mm'],0):.2f}–{q(geometry['legacy_fov_y_mm'],1):.2f}、{q(geometry['legacy_fov_z_mm'],0):.2f}–{q(geometry['legacy_fov_z_mm'],1):.2f} mm。

## Full-support physical bbox extent

下表来自完整FTV inclusion support的source-domain physical bbox；它是保守proxy，远端碎片/多灶会放大extent，不等同radiologist LD target。

{markdown_table(grid_basis, ['axis','visit_adaptive_bbox_extent_median_mm','visit_adaptive_bbox_extent_q95_mm','visit_adaptive_bbox_extent_q99_mm','visit_adaptive_bbox_extent_max_mm'])}

## Visit-to-visit spacing variation

{markdown_table(variation, list(variation.columns))}

## Physical frame限制

T0-anchored结果使用validated/repaired-header RAS-mm frame，但没有把T1–T3做image-only rigid registration。它能审计crop-induced recentering与coverage，却不能把patient repositioning和biological lesion displacement分开。所有后续报告均以`HEADER_PHYSICAL_UNREGISTERED`标记这一限制。
"""
    path.write_text(report, encoding="utf-8")


def write_containment_report(
    containment: pd.DataFrame,
    large_ld: pd.DataFrame,
    ftv: pd.DataFrame,
    margins: pd.DataFrame,
    ld_rank: pd.DataFrame,
    selected_margin_mm: float,
    path: Path,
) -> None:
    core = containment.loc[
        containment["scope"].eq("OVERALL")
        & containment["contract"].isin(["C0", "C1A", "C1B", "C1C", "C2A", "C2B"])
    ].copy()
    large = large_ld.loc[
        large_ld["scope"].eq("OVERALL")
        & large_ld["contract"].isin(["C0", "C1A", "C1B", "C1C", "C2A", "C2B"])
        & (
            (large_ld["view"].eq("legacy"))
            | (large_ld["view"].eq("detail"))
        )
    ].copy()
    by_visit = containment.loc[
        containment["scope"].isin(VISITS)
        & containment["contract"].isin(["C0", "C1A", "C1B", "C1C", "C2B"])
        & containment["view"].isin(["legacy", "detail"])
    ].copy()
    margin_rank = containment.loc[
        containment["scope"].eq("OVERALL")
        & containment["contract"].isin(["C0", "C1A", "C1B", "C1C", "C2B"])
        & containment["view"].isin(["legacy", "detail"])
    ].copy()
    ftv_overall = ftv.loc[
        ftv["scope"].eq("OVERALL")
        & ftv["contract"].isin(["C0", "C1A", "C1B", "C1C", "C2A", "C2B"])
    ].copy()
    source_reference = core.loc[
        core["contract"].eq("C1B") & core["view"].eq("detail")
    ].iloc[0]
    source_touch_count = int(
        round(
            float(source_reference["source_boundary_touch_rate"])
            * int(source_reference["n"])
        )
    )
    report = f"""# 病灶包含性审计

## 口径

新contract在source physical domain检查完整voxel footprint，不用resampled mask voxel count替代full-support retention。FTV inclusion support只作为observability proxy；它不是radiologist target lesion的dense segmentation。LD保持source raw unit，只按每个visit分别计算Q75/Q90并用`>=`包含ties。

预注册的`boundary_touch/suspected/sufficient/exact`只衡量**现有acquisition内可用support相对crop**的保留，不在看到结果后改写。另行加入上游acquisition-boundary sensitivity：{source_touch_count}/{int(source_reference['n'])}个visit的support接触source image face。它不被悄悄并入primary crop rate，但作为完整GO的独立blocker；因此“available-support crop通过”不能被表述为clinical whole lesion已经确定完整可观察。

## Overall

{markdown_table(core, ['contract','view','n','boundary_touch_rate','suspected_truncation_rate','severe_truncation_rate','sufficient_containment_rate','exact_full_support_containment_rate','source_boundary_touch_rate','source_uncensored_and_exact_rate','retained_ftv_fraction_q05','minimum_margin_mm_q05'])}

## 按 visit

{markdown_table(by_visit, ['contract','view','scope','n','suspected_truncation_rate','severe_truncation_rate','sufficient_containment_rate','exact_full_support_containment_rate','retained_ftv_fraction_q05','minimum_margin_mm_q05'])}

## Large-LD subgroup

{markdown_table(large, ['contract','view','subgroup','n','suspected_truncation_rate','upstream_censoring_adjusted_suspected_rate','source_boundary_censored_rate','severe_truncation_rate','sufficient_containment_rate','exact_full_support_containment_rate','retained_ftv_fraction_q05','minimum_margin_mm_median'])}

`suspected_truncation_rate`是预注册的crop-specific primary；`upstream_censoring_adjusted_suspected_rate`是将source-face touch并入后的保守敏感性。后者用于阻止过强的end-to-end observability声明，不追溯改写primary metric或margin选择。

## FTV-specific observability

voxel retention与physical-volume retention均在source domain计算；extent retention为保留support physical extent相对完整support的最小轴比例。

{markdown_table(ftv_overall, ['contract','view','n','median','q05','q25','minimum','physical_volume_retention_median','physical_volume_retention_q05','physical_volume_retention_q25','physical_volume_retention_minimum','extent_retention_min_axis_median','extent_retention_min_axis_q05','extent_retention_min_axis_q25','extent_retention_min_axis_minimum'])}

## LD rank sanity

reported LD保持source raw unit，不做mm换算。这里同时报告完整FTV inclusion support physical bbox最大轴与largest-component approximate extent；前者受远端碎片影响，后者忽略其他component，二者都只作rank proxy，均不等同radiologist target lesion。

{markdown_table(ld_rank, list(ld_rank.columns))}

LD与minimum crop margin的overall秩相关如下；T0–T3逐visit结果保存在公开`containment_summary.csv`。

{markdown_table(margin_rank, ['contract','view','scope','ld_margin_spearman','ld_margin_pvalue','ld_margin_n'])}

## Margin sensitivity

margin候选在任何pCR/model结果之前冻结。四档均实际审计C1A与C1B；选择规则是在所有observability gate通过后，取使overflow/scale change最少的最小候选，因此主分析使用{selected_margin_mm:g}mm。该选择不声称{selected_margin_mm:g}mm是唯一最优margin。

{markdown_table(margins, list(margins.columns))}

## 解释边界

`C1C`使用T0–T3 support union，只是offline available-support observability upper bound。它即使达到100% crop containment，也不能消除source-boundary uncertainty，更不能作为T0 causal input。`C1A-tight`通过tight bbox resize获得containment，但其occupancy/blur与lesion geometry耦合，只作size-normalization sensitivity。
"""
    path.write_text(report, encoding="utf-8")


def write_image_context_report(
    context: pd.DataFrame,
    resampling: pd.DataFrame,
    temporal: pd.DataFrame,
    morphology: pd.DataFrame,
    grid_basis: pd.DataFrame,
    tensor_footprint: pd.DataFrame,
    config: dict[str, Any],
    path: Path,
) -> None:
    basis = grid_basis.set_index("axis")
    selected_margin = float(config["physical_crop"]["selected_margin_mm"])
    preview_path = path.parents[1] / "metrics" / "image_quality_preview.csv"
    if preview_path.is_file():
        preview = pd.read_csv(preview_path)
        preview_table = markdown_table(preview, list(preview.columns))
    else:
        preview_table = "尚未生成真实图像preview聚合。"
    report = f"""# 图像质量与上下文分析

## Grid选择的数据依据

{markdown_table(grid_basis, list(grid_basis.columns))}

detail spacing的X/Y=0.9mm接近native P90（{basis.loc['X','native_spacing_q90_mm']:.3f}/{basis.loc['Y','native_spacing_q90_mm']:.3f}mm），Z=2.0mm等于native median；它比context的1.375/1.5/3.0mm更细。detail nominal FOV的X/Y={basis.loc['X','detail_nominal_fov_mm']:.1f}/{basis.loc['Y','detail_nominal_fov_mm']:.1f}mm，高于P95的T0-anchored四访最坏bbox+{selected_margin:g}mm margin需求（{basis.loc['X','t0_anchored_all_visit_margin_q95_mm']:.1f}/{basis.loc['Y','t0_anchored_all_visit_margin_q95_mm']:.1f}mm）；Z={basis.loc['Z','detail_nominal_fov_mm']:.1f}mm位于P95 exact需求{basis.loc['Z','t0_anchored_all_visit_exact_q95_mm']:.1f}mm与P95加margin需求{basis.loc['Z','t0_anchored_all_visit_margin_q95_mm']:.1f}mm之间。这是outcome-free coverage/detail折中，不是已证明最优的model grid；其padding与downsampling必须继续作为PARTIAL blocker审计。

未来production预处理必须仅用training split或外部protocol冻结spacing/FOV，再原样应用于validation/test；本轮全375人统计只用于outcome-free Stage A设计审计。

## Context与padding

`valid_context_to_lesion_volume_ratio`只把acquisition内有效source视为context，不把crop超出acquisition的zero padding算作额外context。

{markdown_table(context, list(context.columns))}

zero padding即使不作为显式mask输入，也会在图像中形成可见边界，并可能成为visit/geometry cue；因此`no direct geometry metadata`不等于没有间接geometry cue。

## Resampling distortion

resize factor定义为`effective output spacing / native spacing`；大于1为downsampling，小于1为upsampling。`resize_anisotropy`是三轴resize factor的max/min，才用于distortion gate；`output_spacing_anisotropy`只描述输出voxel spacing。`C1A-tight`的effective spacing随lesion bbox改变，可能标准化absolute size；fixed-FOV策略仅在明确标记的overflow病例改变scale。

{markdown_table(resampling, list(resampling.columns))}

## Temporal consistency

{markdown_table(temporal, list(temporal.columns))}

Visit-adaptive方案使crop center跟随每访support bbox center，因而删除bbox平移；表中的voxel-centroid relative change在共同world frame计算，仍保留centroid相对bbox center的变化。T0-anchored方案的window center/FOV在四访保持不变，但尚未完成image-only rigid registration，仍混有patient repositioning。

## Morphology readiness

{markdown_table(morphology, list(morphology.columns))}

surface/component统计针对高度碎片化的FTV inclusion proxy，只回答“support surface是否被crop切断”，不把proxy surface解释为真实tumor boundary irregularity或正式sphericity target。

## 真实图像与normalization sensitivity

{preview_table}

该表覆盖5个去标识case的真实raw-DCE二维物理中心平面，并对first-post-minus-pre执行legacy P01/P99 clipping + median/IQR normalization（zero padding计入统计）。它是代表性sensitivity，不是完整3-D DCE7验收；variable raw phase到production DCE7的phase selection、统一anatomical orientation、全部7通道3-D单次重采样、anti-alias、归一化与cache round-trip仍未实现，继续作为model-ready blocker。

## Tensor footprint与实现边界

{markdown_table(tensor_footprint, list(tensor_footprint.columns))}

本轮已实现source-domain physical geometry、C0/C1/C2 window、完整cohort audit与真实2-D preview；尚未生成可训练的3-D DCE7 volume/cache。表中仅是float32 input本体，未计activation、gradient或optimizer，因此不能把geometry contract误称为已验证的same-encoder训练可行性。

## 强度归一化与合规边界

legacy pipeline在crop后做每通道P01/P99 clipping与median/IQR normalization。改变FOV/context会改变这些统计，因此model-ready builder必须冻结“先physical resample/crop，再按view独立沿用legacy normalization”的顺序，并在matched control中保持一致。preview不参与gate或candidate选择。

三张代表图不含ID、路径或敏感PNG metadata，并已对本轮12张公开图做人工视觉复核。自动验证器不对任意未来PNG执行像素OCR，因此这仍只是本轮技术性去标识与人工图审结论。若把derived MRI图发布到仓库外，仍须按I-SPY2数据使用协议单独确认再分发权限。
"""
    path.write_text(report, encoding="utf-8")


def refresh_image_context_report(
    experiment_root: Path = EXPERIMENT_ROOT,
) -> None:
    """Rebuild the public context report after the independent preview step."""

    metrics = experiment_root / "metrics"
    config = json.loads(
        (experiment_root / "configs" / "stage_a.json").read_text(
            encoding="utf-8"
        )
    )
    write_image_context_report(
        pd.read_csv(metrics / "context_summary.csv"),
        pd.read_csv(metrics / "resampling_summary.csv"),
        pd.read_csv(metrics / "temporal_consistency.csv"),
        pd.read_csv(metrics / "morphology_readiness.csv"),
        pd.read_csv(metrics / "grid_selection_basis.csv"),
        pd.read_csv(metrics / "tensor_footprint_estimate.csv"),
        config,
        experiment_root / "reports" / "image_quality_context_analysis.md",
    )


def write_final_report(
    gate: dict[str, Any],
    containment: pd.DataFrame,
    large_ld: pd.DataFrame,
    ftv: pd.DataFrame,
    morphology: pd.DataFrame,
    context: pd.DataFrame,
    resampling: pd.DataFrame,
    temporal: pd.DataFrame,
    geometry: pd.DataFrame,
    tensor_footprint: pd.DataFrame,
    config: dict[str, Any],
    path: Path,
) -> None:
    decision = gate["decision"]
    candidates = gate["candidate_results"]
    # Do not reward visit recentering merely because it makes containment
    # trivially perfect. Prefer a passing candidate; otherwise report the
    # pre-registered T0-anchored world-model candidate C1B.
    best_name = gate["passing_candidates"][0] if gate["passing_candidates"] else "C1B"
    best = candidates[best_name]
    c0 = containment.loc[
        containment["contract"].eq("C0")
        & containment["view"].eq("legacy")
        & containment["scope"].eq("OVERALL")
    ].iloc[0]
    c1b = candidates["C1B"]["observed"]
    c2b = candidates["C2B"]["observed"]
    c1a = candidates["C1A"]["observed"]
    oracle = containment.loc[
        containment["contract"].eq("C1C")
        & containment["view"].eq("detail")
        & containment["scope"].eq("OVERALL")
    ].iloc[0]
    t1b = temporal.loc[
        temporal["contract"].eq("C1B") & temporal["view"].eq("detail")
    ].iloc[0]
    t1a = temporal.loc[
        temporal["contract"].eq("C1A") & temporal["view"].eq("detail")
    ].iloc[0]
    ftv_c1b = ftv.loc[
        ftv["contract"].eq("C1B")
        & ftv["view"].eq("detail")
        & ftv["scope"].eq("OVERALL")
    ].iloc[0]
    morphology_c1b = morphology.loc[
        morphology["contract"].eq("C1B") & morphology["view"].eq("detail")
    ].iloc[0]
    context_c1b = context.loc[
        context["contract"].eq("C1B") & context["view"].eq("detail")
    ].iloc[0]
    context_c2b = context.loc[
        context["contract"].eq("C2B") & context["view"].eq("context")
    ].iloc[0]
    resize_c2b_context = resampling.loc[
        resampling["contract"].eq("C2B")
        & resampling["view"].eq("context")
    ].iloc[0]
    source_c1b = containment.loc[
        containment["contract"].eq("C1B")
        & containment["view"].eq("detail")
        & containment["scope"].eq("OVERALL")
    ].iloc[0]
    source_touch_count = int(
        round(
            float(source_c1b["source_boundary_touch_rate"])
            * int(source_c1b["n"])
        )
    )
    c1b_large = large_ld.loc[
        large_ld["contract"].eq("C1B")
        & large_ld["view"].eq("detail")
        & large_ld["scope"].isin(VISITS)
    ]
    upstream_q75 = float(
        c1b_large.loc[
            c1b_large["subgroup"].eq("LD_TOP_QUARTILE"),
            "upstream_censoring_adjusted_suspected_rate",
        ].max()
    )
    upstream_q90 = float(
        c1b_large.loc[
            c1b_large["subgroup"].eq("LD_TOP_10PCT"),
            "upstream_censoring_adjusted_suspected_rate",
        ].max()
    )
    detail_footprint = tensor_footprint.loc[
        tensor_footprint["contract"].eq("C1B")
    ].iloc[0]
    multiscale_footprint = tensor_footprint.loc[
        tensor_footprint["contract"].eq("C2B")
    ].iloc[0]
    crop_large_pass = bool(
        candidates["C1B"]["checks"]["large_ld_top_quartile"]
        and candidates["C1B"]["checks"]["large_ld_top_10pct"]
    )
    upstream_large_pass = bool(
        candidates["C1B"]["checks"][
            "upstream_acquisition_large_ld_sensitivity"
        ]
    )
    failed_checks = "、".join(best["failed_checks"]) or "无"
    margin_mm = float(config["physical_crop"]["selected_margin_mm"])
    dicom_gate_path = path.parents[1] / "metrics" / "dicom_geometry_repair_gate.json"
    if dicom_gate_path.is_file():
        dicom_gate = json.loads(dicom_gate_path.read_text(encoding="utf-8"))
        dicom_summary = (
            f"独立raw-DICOM header复核覆盖"
            f"{dicom_gate['audited_singular_visit_count']}/"
            f"{dicom_gate['expected_singular_visit_count']}个异常visit："
            f"header geometry={'PASS' if dicom_gate['header_geometry_audit_pass'] else 'FAIL'}，"
            f"但pixel rebuild={str(dicom_gate['pixel_rebuild_executed']).lower()}、"
            f"pixel order verified={str(dicom_gate['pixel_order_verified']).lower()}、"
            f"model-ready={str(dicom_gate['model_ready']).lower()}。"
        )
    else:
        dicom_summary = "raw-DICOM header复核产物尚不存在；geometry blocker维持未解除。"
    report = f"""# Response-Observable Multiscale Crop 最终报告

## 最终判断：{decision}

{gate['reason']}。预注册的world-model主要方向为`{best_name}`（{margin_mm:g} mm margin）：available-support sufficient={best['observed']['overall_sufficient_containment_rate']:.1%}、exact={best['observed']['overall_exact_full_support_rate']:.1%}、FTV Q05={best['observed']['ftv_retention_q05']:.3f}。完整候选门控未通过项为：`{failed_checks}`。

这里的`PARTIAL`严格限定为 **available-support crop containment显著改善**，不等于end-to-end clinical whole-lesion observability已经成立。{source_touch_count}/{int(source_c1b['n'])}个visit的FTV inclusion support接触原始image face；将其作为上游censoring sensitivity后，C1B最坏visit的large-LD Q75/Q90疑似不可完整观察率为{upstream_q75:.1%}/{upstream_q90:.1%}，上游敏感性门控{'通过' if upstream_large_pass else '未通过'}。该敏感性不事后改写预注册crop-specific指标，但会阻止GO。

Stage B状态：**{'允许最小FTV-only sanity' if gate['stage_b_authorized'] else '未授权、未执行'}**。无论本轮Decision如何，FTV+LD dual grounding、transition、pCR/clinical/treatment supervision均未执行。

{dicom_summary}详见 [DICOM几何修复审计](dicom_geometry_repair_audit.md)。

## 逐条回答

1. **current crop physical FOV variability**：固定`32×96×96 voxel`的pooled median为X/Y/Z={q(geometry['legacy_fov_x_mm'],.5):.1f}/{q(geometry['legacy_fov_y_mm'],.5):.1f}/{q(geometry['legacy_fov_z_mm'],.5):.1f} mm，范围为{q(geometry['legacy_fov_x_mm'],0):.1f}–{q(geometry['legacy_fov_x_mm'],1):.1f}/{q(geometry['legacy_fov_y_mm'],0):.1f}–{q(geometry['legacy_fov_y_mm'],1):.1f}/{q(geometry['legacy_fov_z_mm'],0):.1f}–{q(geometry['legacy_fov_z_mm'],1):.1f} mm；{geometry.groupby('patient_id')['spacing_x_mm'].nunique().gt(1).mean():.1%}患者四访X spacing变化。
2. **C1 adaptive能否解决truncation**：在现有acquisition内可用support口径下，visit-adaptive C1A sufficient/exact为{c1a['overall_sufficient_containment_rate']:.1%}/{c1a['overall_exact_full_support_rate']:.1%}，保留T0坐标的C1B为{c1b['overall_sufficient_containment_rate']:.1%}/{c1b['overall_exact_full_support_rate']:.1%}，均相对C0的{c0['sufficient_containment_rate']:.1%}/{c0['exact_full_support_containment_rate']:.1%}显著改善；计入source-face uncertainty后，C1B source-uncensored且exact为{c1b['source_uncensored_and_exact_rate']:.1%}。C1A依赖逐visit bbox重定位，不能忽略temporal normalization风险。
3. **C2是否提供额外context**：C2B context的有效source context/lesion volume ratio中位数由detail的{context_c1b['valid_context_to_lesion_volume_ratio_median']:.1f}增至{context_c2b['valid_context_to_lesion_volume_ratio_median']:.1f}，但padding中位数也由{context_c1b['padding_fraction_median']:.1%}增至{context_c2b['padding_fraction_median']:.1%}；{resize_c2b_context['extreme_axis_factor_gt2_rate']:.1%}的context visit存在任一轴downsampling factor `>2`（max={c2b['max_resize_factor']:.2f}）。所以它提供更多几何context，却尚未证明是额外“有效信号”，当前不优先于C1B。
4. **large lesion是否仍系统性截断**：预注册的available-support crop-specific最坏visit Q75/Q90为{c1b['top_quartile_suspected_truncation_rate']:.1%}/{c1b['top_10pct_suspected_truncation_rate']:.1%}，{'通过' if crop_large_pass else '未通过'}5%/10%阈值，已远低于legacy T0/T1的99–100%。但上游censoring sensitivity为{upstream_q75:.1%}/{upstream_q90:.1%}并{'通过' if upstream_large_pass else '未通过'}；所以不能声称large lesion的端到端observability已经可靠。
5. **FTV support是否完整可观察**：对available support，C1B overall Q05={c1b['ftv_retention_q05']:.3f}、exact={c1b['overall_exact_full_support_rate']:.1%}、minimum={ftv_c1b['minimum']:.3f}；source-uncensored且exact为{c1b['source_uncensored_and_exact_rate']:.1%}。oracle-union也只给出available-support exact={oracle['exact_full_support_containment_rate']:.1%}，不能消除原始image边缘的不确定性。
6. **morphology/surface readiness**：C1B surface Q05={c1b['surface_retention_q05']:.3f}，但minimum={morphology_c1b['surface_retention_minimum']:.3f}、fully-observable surface={morphology_c1b['fully_observable_surface_rate']:.1%}、cut/missed-component visit比例为{morphology_c1b['any_cut_component_rate']:.1%}/{morphology_c1b['any_missed_component_rate']:.1%}。它达到crop-level群体门槛，但proxy surface不能提升为真实tumor morphology真值。
7. **visit-adaptive是否破坏longitudinal geometry**：是。C1A crop-center max drift median={t1a['crop_center_drift_median_mm']:.2f} mm，window跟随每访support bbox center，从而删除bbox平移；实际voxel centroid相对bbox center的变化仍保留并单独报告，但C1A不能作为唯一world-model coordinate view。
8. **T0-anchored是否更适合world model**：crop-induced geometry方面更合适；C1B window drift Q95={t1b['crop_center_drift_q95_mm']:.6f}mm、FOV relative change Q95={t1b['fov_max_relative_change_q95']:.6f}。但header frame未做image-only rigid registration，patient repositioning仍是限制。
9. **observable且causally deployable candidate**：当前没有通过完整门控的candidate。C1B是最佳causal方向，只通过available-support crop containment与direct-leakage子门；上游acquisition sensitivity、DICOM pixel rebuild、orientation contract、resampling、temporal-frame validation和完整3-D model-input pipeline仍阻止GO。C2B的context分支重采样更严重；C1C明确禁止部署。
10. **是否有资格进入FTV-only retraining**：{'是，仅允许matched G1/N1后再做lambda_FTV=0.25的G3/N3。' if gate['stage_b_authorized'] else '否；Stage B未授权。必须先完成raw-DICOM pixel rebuild/order验收、统一orientation策略、image-only registration sensitivity、极端downsampling处置以及3-D DCE7 phase selection/resampling/normalization/cache round-trip验证。'}
11. **observability是否改善FTV R²/optimization stability**：本轮Stage B未执行，不能声称改善或不改善。
12. **是否有资格重新测试LD grounding**：否。本Goal明确不自动执行Stage C；即使input GO，也应先完成FTV-only representation sanity。

## Candidate gate

```json
{json.dumps(finite_or_none(candidates), ensure_ascii=False, indent=2, sort_keys=True)}
```

## 最终input recommendation

- 主要deployable方向：T0-anchored fixed physical detail（C1B），但状态只是设计候选，不是model-ready输入。真实5-case二维preview通过finite/non-constant sanity；它不替代完整3-D DCE7验收。
- audit upper bound：C1C ORACLE-UNION，永不进入训练loader。
- 资源边界：C1B四访DCE7 float32 input约{detail_footprint['float32_megabytes_per_patient']:.0f} MB/患者（legacy的{detail_footprint['relative_to_legacy']:.1f}倍）；C2B约{multiscale_footprint['float32_megabytes_per_patient']:.0f} MB（{multiscale_footprint['relative_to_legacy']:.1f}倍），均未计activations/optimizer。C2B context的高padding与重采样风险不支持升级为首选。
- 必须先完成：72个奇异DCE sform visit的raw-DICOM pixel rebuild验收、source-edge病例复核、T0 frame的image-only rigid-registration sensitivity，以及完整3-D DCE7 builder/normalization/cache验收；不得用future lesion mask修补C1B miss。
- model batch只含DCE7 image tensor；所有geometry metadata均留在sidecar。

本实验的结论不是“为了LD把crop变大”，而是：current fixed voxel-space crop不保证clinically meaningful response target可从图像观察，必须先建立Response-Observable Image State。
"""
    path.write_text(report, encoding="utf-8")


def save_figures(
    frame: pd.DataFrame,
    containment: pd.DataFrame,
    large_ld: pd.DataFrame,
    geometry: pd.DataFrame,
    temporal: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "C0": "#767676",
        "C1A": "#4c78a8",
        "C1B": "#f58518",
        "C1C": "#54a24b",
        "C2A": "#b279a2",
        "C2B": "#e45756",
    }
    selections = [
        ("C0", "legacy"),
        ("C1A", "detail"),
        ("C1B", "detail"),
        ("C1C", "detail"),
        ("C2B", "context"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.15
    x = np.arange(4)
    for index, (contract, view) in enumerate(selections):
        subset = containment.loc[
            containment["contract"].eq(contract)
            & containment["view"].eq(view)
            & containment["scope"].isin(VISITS)
        ].set_index("scope")
        values = [subset.loc[visit, "sufficient_containment_rate"] for visit in VISITS]
        ax.bar(
            x + (index - 2) * width,
            values,
            width,
            label=f"{contract}/{view}",
            color=colors[contract],
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, label="95% gate")
    ax.set_xticks(x, VISITS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Sufficient containment rate")
    ax.set_title("C0/C1/C2 containment by visit")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "01_containment_by_visit.png", dpi=180)
    plt.close(fig)

    overall = containment.loc[
        containment["scope"].eq("OVERALL")
        & containment.apply(
            lambda row: (row["contract"], row["view"]) in selections, axis=1
        )
    ].copy()
    labels = [f"{row.contract}\n{row.view}" for row in overall.itertuples()]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(
        labels,
        overall["suspected_truncation_rate"],
        color=[colors[value] for value in overall["contract"]],
    )
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Suspected truncation rate")
    ax.set_title("Suspected truncation by input contract")
    fig.tight_layout()
    fig.savefig(output_dir / "02_suspected_truncation_by_contract.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(
        labels,
        overall["exact_full_support_containment_rate"],
        color=[colors[value] for value in overall["contract"]],
    )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact full-support retention")
    ax.set_title("Exact retention by input contract")
    fig.tight_layout()
    fig.savefig(output_dir / "03_exact_retention_by_contract.png", dpi=180)
    plt.close(fig)

    large = large_ld.loc[
        large_ld["scope"].isin(VISITS)
        & large_ld.apply(
            lambda row: (row["contract"], row["view"]) in selections, axis=1
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    contracts = [f"{c}/{v}" for c, v in selections]
    x = np.arange(len(contracts))
    for offset, subgroup in ((-0.18, "LD_TOP_QUARTILE"), (0.18, "LD_TOP_10PCT")):
        subset = (
            large.loc[large["subgroup"].eq(subgroup)]
            .groupby(["contract", "view"], sort=False)[
                "suspected_truncation_rate"
            ]
            .max()
        )
        values = [subset.loc[pair] for pair in selections]
        ax.bar(x + offset, values, 0.36, label=subgroup)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.10, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, contracts, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Suspected truncation rate")
    ax.set_title(
        "Large-LD available-support crop-specific truncation — worst visit"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_large_ld_subgroup_truncation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [
            geometry["legacy_fov_x_mm"],
            geometry["legacy_fov_y_mm"],
            geometry["legacy_fov_z_mm"],
        ],
        tick_labels=["X", "Y", "Z"],
        showfliers=False,
    )
    ax.set_ylabel("Physical FOV (mm)")
    ax.set_title("Legacy 32×96×96 crop physical FOV")
    fig.tight_layout()
    fig.savefig(output_dir / "05_physical_fov_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [geometry["spacing_x_mm"], geometry["spacing_y_mm"], geometry["spacing_z_mm"]],
        tick_labels=["X", "Y", "Z"],
        showfliers=False,
    )
    ax.set_ylabel("Native spacing (mm)")
    ax.set_title("MRI spacing distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "06_spacing_distribution.png", dpi=180)
    plt.close(fig)

    resize_detail = frame.loc[
        frame["contract"].eq("C1B") & frame["view"].eq("detail")
    ]
    resize_context = frame.loc[
        frame["contract"].eq("C2B") & frame["view"].eq("context")
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [
            resize_detail["resize_factor_x"],
            resize_detail["resize_factor_y"],
            resize_detail["resize_factor_z"],
            resize_context["resize_factor_x"],
            resize_context["resize_factor_y"],
            resize_context["resize_factor_z"],
        ],
        tick_labels=[
            "C1B detail\nX",
            "C1B detail\nY",
            "C1B detail\nZ",
            "C2B context\nX",
            "C2B context\nY",
            "C2B context\nZ",
        ],
        showfliers=False,
    )
    ax.axhline(1.0, color="black", linewidth=1)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Effective spacing / native spacing")
    ax.set_title("C1B detail vs C2B context resize factors")
    fig.tight_layout()
    fig.savefig(output_dir / "07_resize_factor_distribution.png", dpi=180)
    plt.close(fig)

    temp = temporal.loc[
        temporal.apply(
            lambda row: (row["contract"], row["view"])
            in [("C0", "legacy"), ("C1A", "detail"), ("C1B", "detail"), ("C2A", "detail"), ("C2B", "detail")],
            axis=1,
        )
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    temp_labels = [f"{row.contract}/{row.view}" for row in temp.itertuples()]
    ax.bar(temp_labels, temp["crop_center_drift_median_mm"], color="#4c78a8")
    ax.scatter(
        np.arange(len(temp)),
        temp["lesion_center_drift_median_mm"],
        color="#e45756",
        label="lesion center drift",
        zorder=3,
    )
    ax.set_ylabel("Patient max drift, median (mm)")
    ax.set_title("Temporal crop-center drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "08_temporal_crop_center_drift.png", dpi=180)
    plt.close(fig)

    retention_data = []
    retention_labels = []
    for contract, view in selections:
        subset = frame.loc[
            frame["contract"].eq(contract) & frame["view"].eq(view),
            "retained_ftv_fraction",
        ]
        retention_data.append(subset)
        retention_labels.append(f"{contract}\n{view}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(retention_data, tick_labels=retention_labels, showfliers=False)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Retained FTV fraction")
    ax.set_title("FTV support retention distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "09_ftv_retention_distribution.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    dgrs_root = os.environ.get("DGRS_DATA_ROOT")
    raw_root = os.environ.get("ISPY2_RAW_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "stage_a.json"
    )
    parser.add_argument(
        "--overlap",
        type=Path,
        default=REPO_ROOT
        / "additional_experiments"
        / "radiomics_next_change"
        / "data_audit"
        / "radiomics_patient_overlap.csv",
    )
    parser.add_argument(
        "--legacy-table",
        type=Path,
        default=REPO_ROOT
        / "additional_experiments"
        / "ftv_ld_dual_grounding_pilot"
        / "metrics"
        / "crop_containment_patient_visit.csv",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path(dgrs_root) / "I-SPY2" if dgrs_root else None,
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path(raw_root) / "Multi-feature-MRI-NACT-Data.xlsx"
        if raw_root
        else None,
    )
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = require_path(args.config, "Stage A config")
    args.overlap = require_path(args.overlap, "strict overlap")
    args.legacy_table = require_path(args.legacy_table, "legacy patient-visit audit table")
    args.preprocessed_root = require_path(
        args.preprocessed_root, "DGRS_DATA_ROOT/I-SPY2"
    )
    args.workbook = require_path(
        args.workbook, "ISPY2_RAW_ROOT/Multi-feature-MRI-NACT-Data.xlsx"
    )
    output_root = args.output_root.expanduser().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if file_sha256(args.workbook) != EXPECTED_WORKBOOK_SHA256:
        raise ValueError("LD workbook SHA256 drift")
    if file_sha256(args.overlap) != EXPECTED_OVERLAP_SHA256:
        raise ValueError("strict overlap SHA256 drift")

    core_destinations = [
        output_root / "metrics" / "containment_summary.csv",
        output_root / "metrics" / "stage_a_gate.json",
        output_root / "reports" / "final_report.md",
    ]
    if not args.overwrite and any(path.exists() for path in core_destinations):
        raise FileExistsError("Stage A输出已存在；如需重跑请显式使用--overwrite")

    cohort = load_inputs(args, config)
    legacy = pd.read_csv(args.legacy_table)
    if len(legacy) != int(config["cohort"]["expected_patient_visits"]):
        raise ValueError("legacy patient-visit table行数漂移")
    legacy["patient_id"] = legacy["patient_id"].astype(str)
    legacy_by_key = legacy.set_index(["patient_id", "visit"], verify_integrity=True)

    all_records: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    for index, row in enumerate(cohort.itertuples(index=False), start=1):
        records, geometry_rows, margin_rows = audit_patient(
            row, args.preprocessed_root, legacy_by_key, config
        )
        all_records.extend(records)
        all_geometry.extend(geometry_rows)
        all_margins.extend(margin_rows)
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"processed {index}/{len(cohort)} patients", flush=True)

    frame = add_ld_flags(pd.DataFrame(all_records))
    geometry = pd.DataFrame(all_geometry)
    margins_private = pd.DataFrame(all_margins)
    expected_contract_rows = int(config["cohort"]["expected_patient_visits"]) * 9
    if len(frame) != expected_contract_rows:
        raise ValueError(f"contract rows应为{expected_contract_rows}，实际{len(frame)}")
    if len(geometry) != int(config["cohort"]["expected_patient_visits"]):
        raise ValueError("geometry rows漂移")

    containment, containment_by_visit = summarize_containment(frame)
    large_ld = summarize_large_ld(frame)
    ld_rank = summarize_ld_rank_sanity(frame)
    ftv = summarize_ftv(frame)
    morphology = summarize_morphology(frame)
    context = summarize_context(frame)
    resampling = summarize_resampling(frame)
    temporal = summarize_temporal(frame)
    physical, spacing_variation = summarize_geometry(geometry)
    margin_summary = summarize_margins(
        margins_private,
        int(config["metrics"]["minimum_public_cell"]),
    )
    grid_basis = summarize_grid_selection_basis(frame, geometry, config)
    tensor_footprint = summarize_tensor_footprint(config)
    gate = decide_gate(
        frame,
        containment,
        large_ld,
        ftv,
        morphology,
        resampling,
        temporal,
        geometry,
        config,
    )

    metrics = output_root / "metrics"
    reports = output_root / "reports"
    figures = output_root / "figures"
    manifests = output_root / "manifests"
    for directory in (metrics, reports, figures, manifests):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_csv(metrics / "patient_visit_contracts.csv", frame)
    atomic_csv(metrics / "patient_level_geometry.csv", geometry)
    atomic_csv(metrics / "containment_summary.csv", containment)
    atomic_csv(metrics / "containment_by_contract_visit.csv", containment_by_visit)
    atomic_csv(metrics / "large_ld_subgroups.csv", large_ld)
    atomic_csv(metrics / "ld_rank_sanity.csv", ld_rank)
    atomic_csv(metrics / "ftv_retention_summary.csv", ftv)
    atomic_csv(metrics / "morphology_readiness.csv", morphology)
    atomic_csv(metrics / "context_summary.csv", context)
    atomic_csv(metrics / "resampling_summary.csv", resampling)
    atomic_csv(metrics / "temporal_consistency.csv", temporal)
    atomic_csv(metrics / "physical_geometry_by_visit.csv", physical)

    qc_rows = []
    for status, count in geometry["affine_decision"].value_counts().items():
        qc_rows.append({"category": "affine_decision", "label": status, "count": count})
    for orientation, count in geometry["mask_header_orientation"].value_counts().items():
        qc_rows.append(
            {"category": "mask_orientation", "label": orientation, "count": count}
        )
    for row in spacing_variation.itertuples(index=False):
        qc_rows.append(
            {
                "category": "visit_spacing_variation",
                "label": row.axis,
                "count": row.patients_with_visit_variation,
                "fraction": row.fraction_with_visit_variation,
                "q95": row.range_q95_mm,
                "maximum": row.range_max_mm,
            }
        )
    atomic_csv(metrics / "physical_geometry_summary.csv", pd.DataFrame(qc_rows))
    atomic_csv(metrics / "margin_candidate_summary.csv", margin_summary)
    atomic_csv(metrics / "grid_selection_basis.csv", grid_basis)
    atomic_csv(metrics / "tensor_footprint_estimate.csv", tensor_footprint)
    atomic_json(metrics / "stage_a_gate.json", gate)

    recommendation = {
        "schema_version": 1,
        "decision": gate["decision"],
        "primary_deployable_candidate": (
            gate["passing_candidates"][0]
            if gate["passing_candidates"]
            else "C1B_PENDING_FAILED_GATES"
        ),
        "multiscale_candidate": "C2B_PENDING_CONTEXT_VALUE_AND_FAILED_GATES",
        "audit_upper_bound": "C1C_ORACLE_UNION_AUDIT_ONLY",
        "stage_b_authorized": gate["stage_b_authorized"],
        "stage_b_executed": False,
        "ld_grounding_executed": False,
        "required_before_model_ready_cache": [
            "rebuild_72_singular_sform_visits_from_raw_DICOM_and_verify_pixel_order",
            "review_source_image_boundary_censoring_in_large_LD_cases",
            "freeze_anatomical_orientation_and_header_or_image_rigid_registration_policy",
            "resolve_any_extreme_resampling_gate_failure",
            "validate_full_3D_DCE7_phase_selection_resampling_normalization_and_cache_roundtrip",
            "keep_mask_and_geometry_sidecars_out_of_model_batch",
        ],
    }
    atomic_json(metrics / "input_recommendation.json", recommendation)
    atomic_csv(
        metrics / "stage_execution_status.csv",
        pd.DataFrame(
            [
                {
                    "stage": "A_INPUT_OBSERVABILITY",
                    "status": "COMPLETED",
                    "decision": gate["decision"],
                },
                {
                    "stage": "B_MINIMAL_FTV_REPRESENTATION_SANITY",
                    "status": "NOT_EXECUTED"
                    if not gate["stage_b_authorized"]
                    else "AUTHORIZED_NOT_AUTOMATICALLY_EXECUTED",
                    "decision": "CONDITIONAL_ON_EXPLICIT_FOLLOWUP",
                },
                {
                    "stage": "C_FTV_LD_DUAL_GROUNDING",
                    "status": "NOT_AUTHORIZED_NOT_EXECUTED",
                    "decision": "OUT_OF_SCOPE",
                },
            ]
        ),
    )

    provenance_sources = [
        EXPERIMENT_ROOT / "configs" / "stage_a.json",
        EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md",
        EXPERIMENT_ROOT / "scripts" / "run_stage_a.py",
        EXPERIMENT_ROOT / "scripts" / "run_dicom_geometry_audit.py",
        EXPERIMENT_ROOT / "scripts" / "make_previews.py",
        EXPERIMENT_ROOT / "src" / "observable_crop" / "__init__.py",
        EXPERIMENT_ROOT / "src" / "observable_crop" / "geometry.py",
        EXPERIMENT_ROOT / "src" / "observable_crop" / "dicom_geometry.py",
        EXPERIMENT_ROOT / "src" / "observable_crop" / "nifti.py",
    ]
    source_hashes = {
        source.relative_to(EXPERIMENT_ROOT).as_posix(): file_sha256(source)
        for source in provenance_sources
    }
    atomic_json(
        manifests / "stage_a_provenance.json",
        {
            "schema_version": 1,
            "repository_head_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "experiment_worktree_dirty": bool(
                git_output(
                    "status",
                    "--porcelain",
                    "--",
                    str(EXPERIMENT_ROOT.relative_to(REPO_ROOT)),
                )
            ),
            "experiment_source_sha256": source_hashes,
            "config_sha256": file_sha256(args.config),
            "overlap_sha256": file_sha256(args.overlap),
            "workbook_sha256": file_sha256(args.workbook),
            "legacy_table_sha256": file_sha256(args.legacy_table),
            "preprocessed_root": "${DGRS_DATA_ROOT}/I-SPY2",
            "workbook": "${ISPY2_RAW_ROOT}/Multi-feature-MRI-NACT-Data.xlsx",
            "patients": len(cohort),
            "patient_visits": len(geometry),
            "contract_visit_rows": len(frame),
            "pcr_clinical_treatment_read": False,
            "model_training_executed": False,
        },
    )

    write_physical_geometry_report(
        geometry,
        physical,
        spacing_variation,
        grid_basis,
        config,
        reports / "physical_geometry_audit.md",
    )
    write_containment_report(
        containment,
        large_ld,
        ftv,
        margin_summary,
        ld_rank,
        float(config["physical_crop"]["selected_margin_mm"]),
        reports / "containment_audit.md",
    )
    write_image_context_report(
        context,
        resampling,
        temporal,
        morphology,
        grid_basis,
        tensor_footprint,
        config,
        reports / "image_quality_context_analysis.md",
    )
    write_final_report(
        gate,
        containment,
        large_ld,
        ftv,
        morphology,
        context,
        resampling,
        temporal,
        geometry,
        tensor_footprint,
        config,
        reports / "final_report.md",
    )
    save_figures(frame, containment, large_ld, geometry, temporal, figures)
    print(
        json.dumps(
            {
                "decision": gate["decision"],
                "stage_b_authorized": gate["stage_b_authorized"],
                "containment_passing_candidates": gate[
                    "containment_passing_candidates"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
