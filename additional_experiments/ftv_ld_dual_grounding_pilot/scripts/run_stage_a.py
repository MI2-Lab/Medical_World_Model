#!/usr/bin/env python3
"""运行 outcome-free LD crop-containment audit。

公开输出只含聚合统计。含 patient_id 的逐访视表由实验目录 .gitignore 排除。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
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

from corejepa.data.nifti import read_spatial_nifti  # noqa: E402
from ftv_ld_pilot.geometry import (  # noqa: E402
    approx_max_extent_mm,
    bbox_xyz,
    geometry_metrics,
    recover_origin,
)


VISITS = ("T0", "T1", "T2", "T3")
MIN_PUBLIC_CELL = 5
EXPECTED_WORKBOOK_SHA256 = (
    "f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc"
)
EXPECTED_OVERLAP_SHA256 = (
    "91b575c9e7e351312b8181a091bdffd2d1f61b88b5a98ac3d78d54c94b63da6b"
)
PATIENT_PATTERN = re.compile(r"^(?:ISPY2-|ACRIN-6698-)(\d{6})$")


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
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            finite_or_none(payload), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def require_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise SystemExit(f"缺少 {label}；请通过命令行或环境变量提供")
    return value.expanduser().resolve(strict=True)


def projected_center(
    center_xyz: tuple[float, float, float],
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    output = []
    for center, source_length, target_length in zip(
        center_xyz, source_shape, target_shape, strict=True
    ):
        fraction = center / max(source_length - 1, 1)
        output.append(
            float(
                np.clip(
                    fraction * max(target_length - 1, 0), 0, max(target_length - 1, 0)
                )
            )
        )
    return tuple(output)


def center_to_start(
    center_xyz: tuple[float, float, float], crop_size_zyx: tuple[int, int, int]
) -> tuple[int, int, int]:
    size_z, size_y, size_x = crop_size_zyx
    center_int = tuple(int(round(value)) for value in center_xyz)
    return (
        center_int[0] - size_x // 2,
        center_int[1] - size_y // 2,
        center_int[2] - size_z // 2,
    )


def bbox_midpoint(visit: dict[str, Any]) -> tuple[float, float, float]:
    bbox = visit.get("bbox_nii_xyz_inclusive")
    if not bbox:
        raise ValueError("T0 manifest 缺 released bbox")
    return (
        0.5 * (float(bbox["x_min"]) + float(bbox["x_max"])),
        0.5 * (float(bbox["y_min"]) + float(bbox["y_max"])),
        0.5 * (float(bbox["z_min"]) + float(bbox["z_max"])),
    )


def touch_flags(mask_zyx: np.ndarray) -> dict[str, bool]:
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"cached ROI 期望 ZYX 3-D，实际 {mask.shape}")
    flags = {
        "touch_x_low": bool(mask[:, :, 0].any()),
        "touch_x_high": bool(mask[:, :, -1].any()),
        "touch_y_low": bool(mask[:, 0, :].any()),
        "touch_y_high": bool(mask[:, -1, :].any()),
        "touch_z_low": bool(mask[0, :, :].any()),
        "touch_z_high": bool(mask[-1, :, :].any()),
    }
    flags["any_boundary_touch"] = any(flags.values())
    return flags


def safe_spearman(
    left: Iterable[float], right: Iterable[float]
) -> tuple[float, float, int]:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan, int(len(x))
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), int(len(x))


def q(values: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    return float(np.quantile(clean, quantile)) if clean.size else math.nan


def summarize_scope(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    n = len(frame)
    origin_ok = frame["origin_exact_match"].astype(bool)
    origin_unique = frame["origin_recovery_status"].eq("unique")
    origin_ambiguous = frame["origin_recovery_status"].eq("multiple")
    diagnostic = frame["diagnostic_support_available"].astype(bool)
    exact_full = frame["exact_full_support_containment"].astype(bool)
    bbox_full = frame["bbox_fully_contained"].astype(bool)
    spacing_reliable = frame["spacing_reliable_for_index_geometry"].astype(bool)
    return {
        "scope": scope,
        "n": n,
        "n_patients": int(frame["patient_id"].nunique()),
        "diagnostic_support_n": int(diagnostic.sum()),
        "diagnostic_support_fraction": float(diagnostic.mean()) if n else math.nan,
        "origin_exact_n": int(origin_ok.sum()),
        "origin_exact_fraction": float(origin_ok.mean()) if n else math.nan,
        "origin_unique_n": int(origin_unique.sum()),
        "origin_unique_fraction": float(origin_unique.mean()) if n else math.nan,
        "origin_ambiguous_n": int(origin_ambiguous.sum()),
        "origin_ambiguous_fraction": float(origin_ambiguous.mean()) if n else math.nan,
        "spacing_reliable_n": int(spacing_reliable.sum()),
        "spacing_reliable_fraction": float(spacing_reliable.mean()) if n else math.nan,
        "complete_miss_n": int(frame["complete_miss"].sum()),
        "complete_miss_rate": float(frame["complete_miss"].mean()) if n else math.nan,
        "boundary_touch_n": int(frame["any_boundary_touch"].sum()),
        "boundary_touch_rate": (
            float(frame["any_boundary_touch"].mean()) if n else math.nan
        ),
        "suspected_truncation_n": int(frame["suspected_truncation"].sum()),
        "suspected_truncation_rate": (
            float(frame["suspected_truncation"].mean()) if n else math.nan
        ),
        "severe_truncation_n": int(frame["severe_truncation"].sum()),
        "severe_truncation_rate": (
            float(frame["severe_truncation"].mean()) if n else math.nan
        ),
        "sufficient_containment_n": int(frame["sufficient_containment"].sum()),
        "sufficient_containment_rate": (
            float(frame["sufficient_containment"].mean()) if n else math.nan
        ),
        "exact_full_support_containment_n": int(exact_full.sum()),
        "exact_full_support_containment_rate": (
            float(exact_full.mean()) if n else math.nan
        ),
        "bbox_fully_contained_n": int(bbox_full.sum()),
        "bbox_fully_contained_rate": float(bbox_full.mean()) if n else math.nan,
        "containment_ratio_median": q(frame["containment_ratio"], 0.50),
        "containment_ratio_q05": q(frame["containment_ratio"], 0.05),
        "whole_union_extent_retention_median": q(
            frame["whole_union_extent_retention_ratio"], 0.50
        ),
        "whole_union_extent_retention_q05": q(
            frame["whole_union_extent_retention_ratio"], 0.05
        ),
        "full_whole_union_approx_extent_mm_median": q(
            frame["approx_max_extent_whole_union_mm"], 0.50
        ),
        "full_largest_component_approx_extent_mm_median": q(
            frame["approx_max_extent_largest_component_mm"], 0.50
        ),
        "minimum_margin_vox_median": q(
            frame.loc[origin_ok, "minimum_margin_vox"], 0.50
        ),
        "minimum_margin_vox_q05": q(frame.loc[origin_ok, "minimum_margin_vox"], 0.05),
        "minimum_margin_mm_median": q(frame.loc[origin_ok, "minimum_margin_mm"], 0.50),
        "minimum_margin_mm_q05": q(frame.loc[origin_ok, "minimum_margin_mm"], 0.05),
        "ld_zero_n": int(frame["ld_zero"].sum()),
        "ld_zero_fraction": float(frame["ld_zero"].mean()) if n else math.nan,
        "reported_ld_median_raw_unit": q(frame["reported_ld"], 0.50),
        "crop_extent_x_mm_median": q(frame["crop_extent_x_mm"], 0.50),
        "crop_extent_x_mm_min": q(frame["crop_extent_x_mm"], 0.00),
        "crop_extent_x_mm_max": q(frame["crop_extent_x_mm"], 1.00),
        "crop_extent_y_mm_median": q(frame["crop_extent_y_mm"], 0.50),
        "crop_extent_y_mm_min": q(frame["crop_extent_y_mm"], 0.00),
        "crop_extent_y_mm_max": q(frame["crop_extent_y_mm"], 1.00),
        "crop_extent_z_mm_median": q(frame["crop_extent_z_mm"], 0.50),
        "crop_extent_z_mm_min": q(frame["crop_extent_z_mm"], 0.00),
        "crop_extent_z_mm_max": q(frame["crop_extent_z_mm"], 1.00),
    }


def load_inputs(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = pd.read_csv(
        args.overlap,
        usecols=[
            "patient_id",
            "clinical_patient_id",
            "has_radiomics",
            "legacy_dce8_cache",
        ],
    )
    overlap["patient_id"] = overlap["patient_id"].astype(str)
    overlap = overlap.loc[overlap["has_radiomics"].astype(bool)].copy()
    if len(overlap) != config["cohort"]["strict_overlap_patients"]:
        raise ValueError(f"strict overlap 应为 375，实际 {len(overlap)}")
    if overlap["patient_id"].duplicated().any():
        raise ValueError("strict overlap patient_id 重复")
    for row in overlap.itertuples(index=False):
        match = PATIENT_PATTERN.fullmatch(str(row.patient_id))
        if not match or int(match.group(1)) != int(row.clinical_patient_id):
            raise ValueError(f"patient mapping fail-closed: {row.patient_id}")

    workbook = pd.read_excel(args.workbook, sheet_name="datawith4visits")
    required = {"CLINICAL-TRIAL-SUBJECT-ID", *(f"LD_{visit}" for visit in VISITS)}
    if missing := required.difference(workbook.columns):
        raise ValueError(f"workbook 缺列：{sorted(missing)}")
    workbook = workbook[
        ["CLINICAL-TRIAL-SUBJECT-ID", *(f"LD_{visit}" for visit in VISITS)]
    ].copy()
    workbook["clinical_patient_id"] = workbook["CLINICAL-TRIAL-SUBJECT-ID"].astype(int)
    if workbook["clinical_patient_id"].duplicated().any():
        raise ValueError("workbook clinical ID 重复")
    merged = overlap.merge(
        workbook.drop(columns="CLINICAL-TRIAL-SUBJECT-ID"),
        on="clinical_patient_id",
        how="left",
        validate="one_to_one",
    )
    if merged[[f"LD_{visit}" for visit in VISITS]].isna().any().any():
        raise ValueError("375 overlap 出现 LD 缺失")
    return merged.sort_values("patient_id").reset_index(drop=True), workbook


def manifest_dce_spatial_contract(
    visit: dict[str, Any],
) -> tuple[tuple[int, int, int], tuple[float, float, float], str]:
    """Reproduce the clean reader's index-order heuristic from manifest metadata."""

    shape = tuple(int(value) for value in visit["dce_shape"][:3])
    pixdim = [float(value) for value in visit["dce_pixdim"]]
    spacing = tuple(abs(value) for value in pixdim[1:4])
    slice_first = (
        min(shape) > 0
        and min(spacing) > 0
        and shape[0] < shape[1]
        and shape[0] < shape[2]
        and shape[1] >= 128
        and shape[2] >= 128
        and spacing[0] > 1.5 * max(spacing[1], spacing[2])
    )
    if slice_first:
        return (
            (shape[1], shape[2], shape[0]),
            (spacing[1], spacing[2], spacing[0]),
            "slice_first_to_xyz",
        )
    return shape, spacing, "none"


def extent_ratio(numerator: Any, denominator: Any) -> float:
    if numerator is None or denominator is None:
        return math.nan
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.nan
    # The fixed-direction Feret proxy is not strictly monotone for a subset:
    # a cropped set can select a different extremal pair and exceed the full
    # proxy by tiny numerical amounts.  Physical retention cannot exceed one,
    # so the descriptive ratio is explicitly bounded while both raw extents
    # remain available as separate columns.
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def audit_patient(
    row: Any,
    preprocessed_root: Path,
    crop_size_zyx: tuple[int, int, int],
    origin_radius: int,
) -> list[dict[str, Any]]:
    patient_id = str(row.patient_id)
    manifest_path = preprocessed_root / patient_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visits = {str(item["visit"]): item for item in manifest["visits"]}
    if set(visits) != set(VISITS):
        raise ValueError(f"{patient_id} manifest visit 不完整")

    cache_path = Path(str(row.legacy_dce8_cache)).expanduser().resolve(strict=True)
    with np.load(cache_path, allow_pickle=False) as archive:
        if archive.files != ["x"]:
            raise ValueError(
                f"{patient_id} legacy cache contract 漂移：{archive.files}"
            )
        cached = np.asarray(archive["x"], dtype=np.float32)
    if cached.shape != (4, 8, 32, 96, 96):
        raise ValueError(f"{patient_id} cache shape 漂移：{cached.shape}")
    cached_roi = cached[:, 7] > 0.5
    if not np.all((cached[:, 7] == 0.0) | (cached[:, 7] == 1.0)):
        raise ValueError(f"{patient_id} cache ROI 非二值")

    t0_center = bbox_midpoint(visits["T0"])
    t0_shape, _, _ = manifest_dce_spatial_contract(visits["T0"])
    output: list[dict[str, Any]] = []
    for visit_index, visit_name in enumerate(VISITS):
        visit = visits[visit_name]
        mask_path = Path(str(visit["ftv_mask_nifti"])).expanduser().resolve(strict=True)
        mask, meta = read_spatial_nifti(mask_path)
        full = np.asarray(mask > 0, dtype=bool)
        if full.ndim != 3 or not full.any():
            raise ValueError(
                f"{patient_id}/{visit_name} full FTV support 为空或维度错误"
            )
        spacing_xyz = tuple(abs(float(value)) for value in meta["pixdim"][1:4])
        if (
            len(spacing_xyz) != 3
            or not all(np.isfinite(spacing_xyz))
            or min(spacing_xyz) <= 0
        ):
            raise ValueError(f"{patient_id}/{visit_name} spacing 非法：{spacing_xyz}")
        dce_shape_xyz, dce_spacing_xyz, dce_axis_handling = (
            manifest_dce_spatial_contract(visit)
        )
        shape_match = dce_shape_xyz == tuple(int(value) for value in full.shape)
        spacing_match = bool(
            np.allclose(dce_spacing_xyz, spacing_xyz, rtol=1e-6, atol=1e-6)
        )
        axis_handling_match = dce_axis_handling == str(
            meta.get("axis_canonicalization", "unknown")
        )
        if not shape_match or not spacing_match or not axis_handling_match:
            raise ValueError(
                f"{patient_id}/{visit_name} DCE-mask index/spacing contract mismatch"
            )
        center = projected_center(
            t0_center, t0_shape, tuple(int(value) for value in full.shape)
        )
        clean_start = center_to_start(center, crop_size_zyx)
        actual = cached_roi[visit_index]
        recovery = recover_origin(
            full,
            actual,
            clean_start,
            crop_size_zyx,
            radius=origin_radius,
        )
        chosen = recovery.get("chosen_start")
        recovery_status = str(recovery.get("status", "no_match"))
        origin_exact = recovery_status in {"unique", "multiple"} and chosen is not None
        geometry = (
            geometry_metrics(
                full, tuple(int(value) for value in chosen), spacing_xyz, crop_size_zyx
            )
            if origin_exact
            else {}
        )
        extent = approx_max_extent_mm(full, spacing_xyz)
        cached_xyz = np.transpose(actual, (2, 1, 0))
        cached_extent = approx_max_extent_mm(cached_xyz, spacing_xyz)
        full_box = bbox_xyz(full)
        cached_box = bbox_xyz(cached_xyz)
        flags = touch_flags(actual)
        full_count = int(full.sum())
        cached_count = int(actual.sum())
        if cached_count > full_count:
            raise ValueError(
                f"{patient_id}/{visit_name} cached support 超过 full support"
            )
        ratio = float(cached_count / full_count)
        diagnostic = full_count > 0
        complete_miss = cached_count == 0
        suspected = bool(flags["any_boundary_touch"] or ratio < 0.99 or not diagnostic)
        severe = bool(ratio < 0.90)
        sufficient = bool(
            diagnostic and ratio >= 0.99 and not flags["any_boundary_touch"]
        )
        exact_full_support = cached_count == full_count

        if origin_exact:
            if int(geometry["contained_voxels"]) != cached_count:
                raise AssertionError(
                    f"{patient_id}/{visit_name} geometry/cache count mismatch"
                )
            if not math.isclose(
                float(geometry["containment_ratio"]), ratio, abs_tol=1e-12
            ):
                raise AssertionError(
                    f"{patient_id}/{visit_name} geometry/cache ratio mismatch"
                )
            geometry_touch = geometry["boundary_touch"]
            for face in ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high"):
                if bool(geometry_touch[face]) != bool(flags[f"touch_{face}"]):
                    raise AssertionError(
                        f"{patient_id}/{visit_name} boundary mismatch: {face}"
                    )

        min_margin_vox = geometry.get("min_margin_voxel", math.nan)
        min_margin_mm = geometry.get("min_margin_mm", math.nan)
        bbox_fully_contained = bool(origin_exact and min_margin_vox >= 0)

        record: dict[str, Any] = {
            "patient_id": patient_id,
            "clinical_patient_id": int(row.clinical_patient_id),
            "visit": visit_name,
            "visit_index": visit_index,
            "reported_ld": float(getattr(row, f"LD_{visit_name}")),
            "ld_zero": bool(float(getattr(row, f"LD_{visit_name}")) == 0.0),
            "ld_unit_status": "LD_UNIT_NOT_EXPLICIT",
            "ld_zero_semantics": "AMBIGUOUS_ZERO_SEMANTICS",
            "support_source": "full_resolution_ftv_inclusion_region_proxy",
            "diagnostic_support_available": diagnostic,
            "source_shape_x": int(full.shape[0]),
            "source_shape_y": int(full.shape[1]),
            "source_shape_z": int(full.shape[2]),
            "spacing_x_mm": spacing_xyz[0],
            "spacing_y_mm": spacing_xyz[1],
            "spacing_z_mm": spacing_xyz[2],
            "dce_mask_shape_match": shape_match,
            "dce_mask_spacing_match": spacing_match,
            "dce_mask_axis_handling_match": axis_handling_match,
            "spacing_reliable_for_index_geometry": True,
            "physical_geometry_scope": "matched_spacing_index_space_affine_not_used",
            "axis_canonicalization": str(meta.get("axis_canonicalization", "unknown")),
            "clean_center_x": center[0],
            "clean_center_y": center[1],
            "clean_center_z": center[2],
            "clean_start_x": clean_start[0],
            "clean_start_y": clean_start[1],
            "clean_start_z": clean_start[2],
            "origin_recovery_method": str(recovery.get("method", "unresolved")),
            "origin_recovery_status": recovery_status,
            "origin_recovery_ambiguous": recovery_status == "multiple",
            "origin_candidate_count": int(recovery.get("candidate_count", 0)),
            "origin_exact_match": origin_exact,
            "crop_start_x": int(chosen[0]) if origin_exact else math.nan,
            "crop_start_y": int(chosen[1]) if origin_exact else math.nan,
            "crop_start_z": int(chosen[2]) if origin_exact else math.nan,
            "full_support_voxels": full_count,
            "cached_support_voxels": cached_count,
            "containment_ratio": ratio,
            "complete_miss": complete_miss,
            "exact_full_support_containment": exact_full_support,
            "extent_observable_conservative": exact_full_support,
            "bbox_fully_contained": bbox_fully_contained,
            **flags,
            "suspected_truncation": suspected,
            "severe_truncation": severe,
            "sufficient_containment": sufficient,
        }
        if full_box is not None:
            for axis in "xyz":
                record[f"full_bbox_{axis}_min"] = full_box[f"{axis}_min"]
                record[f"full_bbox_{axis}_max"] = full_box[f"{axis}_max"]
            for axis, value in zip("xyz", full_box["extent_xyz_voxel"], strict=True):
                record[f"full_extent_{axis}_vox"] = int(value)
                record[f"full_extent_{axis}_mm"] = float(
                    value * spacing_xyz["xyz".index(axis)]
                )
        if cached_box is not None:
            for axis in "xyz":
                record[f"cached_bbox_in_crop_{axis}_min"] = cached_box[f"{axis}_min"]
                record[f"cached_bbox_in_crop_{axis}_max"] = cached_box[f"{axis}_max"]
        record["minimum_margin_vox"] = min_margin_vox
        record["minimum_margin_mm"] = min_margin_mm
        margins_vox = geometry.get("signed_margins_voxel", {})
        margins_mm = geometry.get("signed_margins_mm", {})
        padding_vox = geometry.get("padding_voxel", {})
        for face in ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high"):
            record[f"margin_{face}_vox"] = margins_vox.get(face, math.nan)
            record[f"margin_{face}_mm"] = margins_mm.get(face, math.nan)
            record[f"padding_{face}_vox"] = padding_vox.get(face, math.nan)

        record["full_component_count"] = extent["component_count"]
        record["full_largest_component_voxels"] = extent[
            "largest_component_voxel_count"
        ]
        record["approx_max_extent_whole_union_mm"] = extent[
            "whole_union_approx_max_extent_mm"
        ]
        record["approx_max_extent_largest_component_mm"] = extent[
            "largest_component_approx_max_extent_mm"
        ]
        record["cached_approx_max_extent_whole_union_mm"] = cached_extent[
            "whole_union_approx_max_extent_mm"
        ]
        record["cached_approx_max_extent_largest_component_mm"] = cached_extent[
            "largest_component_approx_max_extent_mm"
        ]
        record["whole_union_extent_retention_ratio"] = extent_ratio(
            cached_extent["whole_union_approx_max_extent_mm"],
            extent["whole_union_approx_max_extent_mm"],
        )
        record["largest_component_extent_retention_ratio"] = extent_ratio(
            cached_extent["largest_component_approx_max_extent_mm"],
            extent["largest_component_approx_max_extent_mm"],
        )
        # Crop physical extent does not depend on origin.
        size_z, size_y, size_x = crop_size_zyx
        record["crop_extent_x_mm"] = float(size_x * spacing_xyz[0])
        record["crop_extent_y_mm"] = float(size_y * spacing_xyz[1])
        record["crop_extent_z_mm"] = float(size_z * spacing_xyz[2])
        if origin_exact:
            size_xyz = (size_x, size_y, size_z)
            for axis_index, axis in enumerate("xyz"):
                start = int(chosen[axis_index])
                end_exclusive = start + size_xyz[axis_index]
                record[f"crop_end_{axis}_exclusive"] = end_exclusive
                record[f"crop_end_{axis}_inclusive"] = end_exclusive - 1
                record[f"effective_source_{axis}_low"] = max(start, 0)
                record[f"effective_source_{axis}_high_exclusive"] = min(
                    end_exclusive, int(full.shape[axis_index])
                )
        output.append(record)
    return output


def build_timepoint_summary(patient_visit: pd.DataFrame) -> pd.DataFrame:
    rows = [
        summarize_scope(patient_visit.loc[patient_visit["visit"].eq(visit)], visit)
        for visit in VISITS
    ]
    return pd.DataFrame(rows)


def build_overall_summary(patient_visit: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "ALL_VISITS": patient_visit,
        "T0_T1": patient_visit.loc[patient_visit["visit"].isin(("T0", "T1"))],
        "T0_T1_T2": patient_visit.loc[patient_visit["visit"].isin(("T0", "T1", "T2"))],
        "T3": patient_visit.loc[patient_visit["visit"].eq("T3")],
    }
    rows = [summarize_scope(frame, name) for name, frame in scopes.items()]
    for name, frame in scopes.items():
        origin = frame.loc[frame["origin_exact_match"].astype(bool)]
        unique_origin = frame.loc[frame["origin_recovery_status"].eq("unique")]
        ld_margin_rho, ld_margin_p, ld_margin_n = safe_spearman(
            origin["reported_ld"], origin["minimum_margin_mm"]
        )
        ld_ratio_rho, ld_ratio_p, ld_ratio_n = safe_spearman(
            frame["reported_ld"], frame["containment_ratio"]
        )
        ld_extent_rho, ld_extent_p, ld_extent_n = safe_spearman(
            frame["reported_ld"], frame["approx_max_extent_largest_component_mm"]
        )
        unique_ld_margin_rho, unique_ld_margin_p, unique_ld_margin_n = safe_spearman(
            unique_origin["reported_ld"], unique_origin["minimum_margin_mm"]
        )
        row = next(item for item in rows if item["scope"] == name)
        row.update(
            {
                "ld_margin_spearman": ld_margin_rho,
                "ld_margin_spearman_p": ld_margin_p,
                "ld_margin_spearman_n": ld_margin_n,
                "ld_margin_spearman_unique_only": unique_ld_margin_rho,
                "ld_margin_spearman_unique_only_p": unique_ld_margin_p,
                "ld_margin_spearman_unique_only_n": unique_ld_margin_n,
                "ld_containment_ratio_spearman": ld_ratio_rho,
                "ld_containment_ratio_spearman_p": ld_ratio_p,
                "ld_containment_ratio_spearman_n": ld_ratio_n,
                "ld_approx_extent_spearman_exploratory": ld_extent_rho,
                "ld_approx_extent_spearman_p_exploratory": ld_extent_p,
                "ld_approx_extent_spearman_n": ld_extent_n,
                "ld_approx_extent_status": "RANK_ONLY_LD_UNIT_UNCONFIRMED_NOT_FORMAL_CALIBRATION",
            }
        )
    return pd.DataFrame(rows)


def add_ld_quantile_flags(patient_visit: pd.DataFrame) -> pd.DataFrame:
    output = patient_visit.copy()
    output["ld_q75_visit"] = math.nan
    output["ld_q90_visit"] = math.nan
    output["ld_top25"] = False
    output["ld_top10"] = False
    for visit in VISITS:
        mask = output["visit"].eq(visit)
        q75 = q(output.loc[mask, "reported_ld"], 0.75)
        q90 = q(output.loc[mask, "reported_ld"], 0.90)
        output.loc[mask, "ld_q75_visit"] = q75
        output.loc[mask, "ld_q90_visit"] = q90
        output.loc[mask, "ld_top25"] = output.loc[mask, "reported_ld"].ge(q75)
        output.loc[mask, "ld_top10"] = output.loc[mask, "reported_ld"].ge(q90)
    return output


def build_ld_quantile_summary(patient_visit: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        visit: patient_visit.loc[patient_visit["visit"].eq(visit)] for visit in VISITS
    }
    scopes["T0_T1"] = patient_visit.loc[patient_visit["visit"].isin(("T0", "T1"))]
    scopes["ALL_VISITS"] = patient_visit
    rows: list[dict[str, Any]] = []
    for scope, frame in scopes.items():
        for group, column in (
            ("TOP_25_PERCENT", "ld_top25"),
            ("TOP_10_PERCENT", "ld_top10"),
        ):
            subset = frame.loc[frame[column].astype(bool)]
            rows.append(
                {
                    "scope": scope,
                    "ld_group": group,
                    "n": len(subset),
                    "n_patients": int(subset["patient_id"].nunique()),
                    "reported_ld_min_raw_unit": (
                        float(subset["reported_ld"].min()) if len(subset) else math.nan
                    ),
                    "reported_ld_median_raw_unit": q(subset["reported_ld"], 0.50),
                    "boundary_touch_n": int(subset["any_boundary_touch"].sum()),
                    "boundary_touch_rate": (
                        float(subset["any_boundary_touch"].mean())
                        if len(subset)
                        else math.nan
                    ),
                    "suspected_truncation_n": int(subset["suspected_truncation"].sum()),
                    "suspected_truncation_rate": (
                        float(subset["suspected_truncation"].mean())
                        if len(subset)
                        else math.nan
                    ),
                    "severe_truncation_n": int(subset["severe_truncation"].sum()),
                    "severe_truncation_rate": (
                        float(subset["severe_truncation"].mean())
                        if len(subset)
                        else math.nan
                    ),
                    "containment_ratio_median": q(subset["containment_ratio"], 0.50),
                    "exact_full_support_containment_n": int(
                        subset["exact_full_support_containment"].sum()
                    ),
                    "exact_full_support_containment_rate": (
                        float(subset["exact_full_support_containment"].mean())
                        if len(subset)
                        else math.nan
                    ),
                    "bbox_fully_contained_n": int(subset["bbox_fully_contained"].sum()),
                    "bbox_fully_contained_rate": (
                        float(subset["bbox_fully_contained"].mean())
                        if len(subset)
                        else math.nan
                    ),
                    "whole_union_extent_retention_median": q(
                        subset["whole_union_extent_retention_ratio"], 0.50
                    ),
                    "minimum_margin_mm_median": q(
                        subset.loc[
                            subset["origin_exact_match"].astype(bool),
                            "minimum_margin_mm",
                        ],
                        0.50,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_ld_distribution_summary(patient_visit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for visit in (*VISITS, "ALL_VISITS"):
        frame = (
            patient_visit
            if visit == "ALL_VISITS"
            else patient_visit.loc[patient_visit["visit"].eq(visit)]
        )
        for label, subset in (
            (
                "SUFFICIENT_CONTAINMENT",
                frame.loc[frame["sufficient_containment"].astype(bool)],
            ),
            (
                "SUSPECTED_TRUNCATION",
                frame.loc[frame["suspected_truncation"].astype(bool)],
            ),
        ):
            suppressed = len(subset) < MIN_PUBLIC_CELL
            rows.append(
                {
                    "scope": visit,
                    "containment_group": label,
                    "n": len(subset),
                    "small_cell_suppressed": suppressed,
                    "ld_median_raw_unit": (
                        math.nan if suppressed else q(subset["reported_ld"], 0.50)
                    ),
                    "ld_q25_raw_unit": (
                        math.nan if suppressed else q(subset["reported_ld"], 0.25)
                    ),
                    "ld_q75_raw_unit": (
                        math.nan if suppressed else q(subset["reported_ld"], 0.75)
                    ),
                    "ld_zero_fraction": (
                        math.nan if suppressed else float(subset["ld_zero"].mean())
                    ),
                }
            )
    return pd.DataFrame(rows)


def decide_gate(
    patient_visit: pd.DataFrame,
    summary: pd.DataFrame,
    quantiles: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    early = summary.set_index("scope").loc["T0_T1"]
    combined = summary.set_index("scope").loc["ALL_VISITS"]
    top25 = quantiles.loc[
        quantiles["scope"].eq("T0_T1") & quantiles["ld_group"].eq("TOP_25_PERCENT")
    ].iloc[0]
    criteria = {
        "t0_t1_suspected_truncation": {
            "observed": float(early["suspected_truncation_rate"]),
            "operator": "<=",
            "threshold": float(gate["primary_suspected_truncation_rate_max"]),
            "pass": bool(
                early["suspected_truncation_rate"]
                <= gate["primary_suspected_truncation_rate_max"]
            ),
        },
        "t0_t1_top_quartile_suspected_truncation": {
            "observed": float(top25["suspected_truncation_rate"]),
            "operator": "<=",
            "threshold": float(
                gate["primary_top_quartile_suspected_truncation_rate_max"]
            ),
            "pass": bool(
                top25["suspected_truncation_rate"]
                <= gate["primary_top_quartile_suspected_truncation_rate_max"]
            ),
        },
        "all_visit_sufficient_containment": {
            "observed": float(combined["sufficient_containment_rate"]),
            "operator": ">=",
            "threshold": float(gate["combined_sufficient_containment_rate_min"]),
            "pass": bool(
                combined["sufficient_containment_rate"]
                >= gate["combined_sufficient_containment_rate_min"]
            ),
        },
        "t0_t1_ld_margin_systematic_association": {
            "observed": float(early["ld_margin_spearman"]),
            "operator": ">",
            "threshold": float(gate["strong_systematic_ld_margin_spearman_threshold"]),
            "pass": bool(
                np.isfinite(early["ld_margin_spearman"])
                and early["ld_margin_spearman"]
                > gate["strong_systematic_ld_margin_spearman_threshold"]
            ),
        },
        "exact_origin_recovery": {
            "observed": float(combined["origin_exact_fraction"]),
            "operator": ">=",
            "threshold": float(gate["minimum_exact_origin_recovery_fraction"]),
            "pass": bool(
                combined["origin_exact_fraction"]
                >= gate["minimum_exact_origin_recovery_fraction"]
            ),
        },
    }
    diagnostic_fraction = float(early["diagnostic_support_fraction"])
    all_pass = all(bool(item["pass"]) for item in criteria.values())
    complete_support = bool(
        patient_visit["diagnostic_support_available"].all()
        and patient_visit["origin_exact_match"].all()
    )
    if all_pass and complete_support:
        decision = "GO"
    elif (
        all_pass
        and gate["go_with_caveat_allowed"]
        and diagnostic_fraction
        >= gate["minimum_t0_t1_diagnostic_support_fraction_for_caveat"]
    ):
        decision = "GO_WITH_CAVEAT"
    else:
        decision = "NO_GO"
    failed = [name for name, item in criteria.items() if not item["pass"]]
    return {
        "schema_version": 1,
        "stage": "A_LD_CROP_CONTAINMENT_AUDIT",
        "decision": decision,
        "stage_b_authorized": decision in {"GO", "GO_WITH_CAVEAT"},
        "stop_code": (
            None
            if decision in {"GO", "GO_WITH_CAVEAT"}
            else "LD_NOT_OBSERVABLE_UNDER_CURRENT_CROP"
        ),
        "criteria": criteria,
        "failed_criteria": failed,
        "cohort": {
            "patients": int(patient_visit["patient_id"].nunique()),
            "patient_visits": int(len(patient_visit)),
            "t0_t1_patient_visits": int(
                patient_visit["visit"].isin(("T0", "T1")).sum()
            ),
        },
        "target_semantics": {
            "ld_unit_status": "LD_UNIT_NOT_EXPLICIT",
            "zero_semantics": "AMBIGUOUS_ZERO_SEMANTICS",
            "formal_reported_ld_to_mm_calibration": "NOT_PERFORMED",
            "extent_rank_sanity": "EXPLORATORY_ONLY",
        },
        "support_semantics": {
            "source": "full-resolution FTV inclusion region",
            "manual_dense_lesion_segmentation": False,
            "reported_ld_target_replaced_by_segmentation_extent": False,
        },
        "warnings": [
            "FTV inclusion support is a target-observability proxy, not a manual dense lesion segmentation.",
            "Legacy builder source and saved crop origins are unavailable; origins are recovered by exact mask matching.",
            "Reported LD unit and zero semantics are not explicit in the source workbook.",
        ],
        "post_review_sensitivity_not_a_gate_change": {
            "all_visit_exact_full_support_containment_rate": float(
                combined["exact_full_support_containment_rate"]
            ),
            "t0_t1_exact_full_support_containment_rate": float(
                early["exact_full_support_containment_rate"]
            ),
            "all_visit_bbox_fully_contained_rate": float(
                combined["bbox_fully_contained_rate"]
            ),
            "t0_t1_bbox_fully_contained_rate": float(
                early["bbox_fully_contained_rate"]
            ),
            "origin_ambiguous_fraction": float(combined["origin_ambiguous_fraction"]),
            "t0_t1_ld_margin_spearman_unique_only": float(
                early["ld_margin_spearman_unique_only"]
            ),
        },
    }


def save_figures(
    patient_visit: pd.DataFrame,
    by_timepoint: pd.DataFrame,
    by_quantile: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#3b82f6", "#14b8a6", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    values = by_timepoint.set_index("scope").loc[list(VISITS), "boundary_touch_rate"]
    ax.bar(VISITS, values, color=colors)
    ax.axhline(
        0.10, color="black", linestyle="--", linewidth=1.2, label="T0/T1 gate 10%"
    )
    ax.set_ylabel("Boundary-touch rate")
    ax.set_ylim(0, min(1.0, max(0.15, float(values.max()) * 1.15)))
    ax.set_title("Boundary-touch rate by visit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "01_boundary_touch_rate_by_visit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data = [
        patient_visit.loc[
            patient_visit["visit"].eq(visit)
            & patient_visit["origin_exact_match"].astype(bool),
            "minimum_margin_mm",
        ].dropna()
        for visit in VISITS
    ]
    boxes = ax.boxplot(data, tick_labels=VISITS, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors, strict=False):
        patch.set_facecolor(color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Minimum signed margin (mm)")
    ax.set_title("Lesion-support-to-crop margin by visit")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_margin_distribution_by_visit.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=False, sharey=False)
    for ax, visit in zip(axes.ravel(), VISITS, strict=True):
        subset = patient_visit.loc[
            patient_visit["visit"].eq(visit)
            & patient_visit["origin_exact_match"].astype(bool)
        ]
        if len(subset):
            plot = ax.hexbin(
                subset["reported_ld"],
                subset["minimum_margin_mm"],
                gridsize=14,
                mincnt=MIN_PUBLIC_CELL,
                cmap="viridis",
            )
            fig.colorbar(plot, ax=ax, label="bin count")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.9)
        ax.set_title(visit)
        ax.set_xlabel("Reported LD (raw source unit)")
        ax.set_ylabel("Minimum margin (mm)")
    fig.suptitle("Reported LD vs crop margin (aggregate hexbin)")
    fig.tight_layout()
    fig.savefig(figures_dir / "03_ld_vs_margin_hexbin.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    positions: list[float] = []
    data = []
    tick_positions = []
    for index, visit in enumerate(VISITS):
        current = patient_visit.loc[patient_visit["visit"].eq(visit)]
        contained = current.loc[
            current["sufficient_containment"].astype(bool), "reported_ld"
        ]
        truncated = current.loc[
            current["suspected_truncation"].astype(bool), "reported_ld"
        ]
        if len(contained) < MIN_PUBLIC_CELL:
            contained = pd.Series(dtype=float)
        if len(truncated) < MIN_PUBLIC_CELL:
            truncated = pd.Series(dtype=float)
        base = index * 3.0
        positions.extend((base + 0.8, base + 1.8))
        data.extend((contained, truncated))
        tick_positions.append(base + 1.3)
    boxes = ax.boxplot(
        data, positions=positions, widths=0.7, showfliers=False, patch_artist=True
    )
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor("#22c55e" if index % 2 == 0 else "#ef4444")
    ax.set_xticks(tick_positions, VISITS)
    ax.set_ylabel("Reported LD (raw source unit)")
    ax.set_title("LD distribution: sufficient containment vs suspected truncation")
    ax.legend(
        [boxes["boxes"][0], boxes["boxes"][1]],
        ["sufficient", "suspected"],
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "04_contained_vs_truncated_ld_distribution.png", dpi=180)
    plt.close(fig)

    plot_frame = by_quantile.loc[by_quantile["scope"].isin((*VISITS, "T0_T1"))].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    scopes = [*VISITS, "T0_T1"]
    x = np.arange(len(scopes))
    width = 0.36
    for offset, group, color in (
        (-width / 2, "TOP_25_PERCENT", "#f59e0b"),
        (width / 2, "TOP_10_PERCENT", "#dc2626"),
    ):
        values = [
            float(
                plot_frame.loc[
                    plot_frame["scope"].eq(scope) & plot_frame["ld_group"].eq(group),
                    "suspected_truncation_rate",
                ].iloc[0]
            )
            for scope in scopes
        ]
        ax.bar(x + offset, values, width, label=group.replace("_", " "), color=color)
    ax.axhline(
        0.20,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="early top-quartile gate 20%",
    )
    ax.set_xticks(x, scopes)
    ax.set_ylabel("Suspected-truncation rate")
    ax.set_title("Large-LD subgroup truncation")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "05_large_ld_subgroup_truncation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, truncated in zip(axes, (False, True), strict=True):
        ax.add_patch(
            plt.Rectangle(
                (0.08, 0.08), 0.84, 0.84, fill=False, linewidth=2, color="#64748b"
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (0.24, 0.20), 0.52, 0.60, fill=False, linewidth=2.5, color="#2563eb"
            )
        )
        center_x = 0.48 if not truncated else 0.69
        width_ellipse = 0.34 if not truncated else 0.40
        ax.add_patch(
            matplotlib.patches.Ellipse(
                (center_x, 0.50), width_ellipse, 0.30, color="#ef4444", alpha=0.55
            )
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Sufficient containment" if not truncated else "Boundary touch / truncation"
        )
    fig.suptitle("Generic crop-containment schematic (no patient image)")
    fig.tight_layout()
    fig.savefig(figures_dir / "06_privacy_safe_containment_schematic.png", dpi=180)
    plt.close(fig)


def format_pct(value: Any) -> str:
    return (
        "NA"
        if value is None or not np.isfinite(float(value))
        else f"{100 * float(value):.1f}%"
    )


def write_report(
    by_timepoint: pd.DataFrame,
    summary: pd.DataFrame,
    by_quantile: pd.DataFrame,
    gate: dict[str, Any],
    report_path: Path,
) -> None:
    table_rows = []
    indexed = by_timepoint.set_index("scope")
    for visit in VISITS:
        row = indexed.loc[visit]
        table_rows.append(
            f"| {visit} | {int(row['n'])} | {format_pct(row['boundary_touch_rate'])} | "
            f"{format_pct(row['suspected_truncation_rate'])} | {format_pct(row['severe_truncation_rate'])} | "
            f"{format_pct(row['sufficient_containment_rate'])} | "
            f"{format_pct(row['exact_full_support_containment_rate'])} | "
            f"{row['minimum_margin_mm_median']:.2f} | "
            f"{row['minimum_margin_mm_q05']:.2f} | {format_pct(row['ld_zero_fraction'])} |"
        )
    all_row = summary.set_index("scope").loc["ALL_VISITS"]
    early = summary.set_index("scope").loc["T0_T1"]
    top25 = by_quantile.loc[
        by_quantile["scope"].eq("T0_T1") & by_quantile["ld_group"].eq("TOP_25_PERCENT")
    ].iloc[0]
    decision = gate["decision"]
    failed = "、".join(gate["failed_criteria"]) if gate["failed_criteria"] else "无"
    conclusion = (
        "当前 crop 通过预注册 gate，可在限定 caveat 下进入 Stage B。"
        if gate["stage_b_authorized"]
        else "当前 crop 未通过预注册 gate；LD 在该 input contract 下不能被视为充分可观察，Stage B 必须停止。"
    )
    next_step = (
        "按已冻结的 matched 2-seed×5-fold protocol 进入 B2 dual grounding。"
        if gate["stage_b_authorized"]
        else "优先扩大固定 crop、使用覆盖完整 lesion bbox 并保留 context 的 adaptive crop，或采用 lesion/context multi-scale representation；修改 input 后重新执行同一 containment gate。"
    )
    report = f"""# LD Crop-Containment Audit 报告

## 1. 结论

Stage A 决策为 **{decision}**。{conclusion}

- stop code：`{gate.get('stop_code') or 'NONE'}`
- 未通过条件：{failed}
- 375 人、1,500 个 patient×visit 全部有 nonempty full-resolution FTV inclusion support；该 support 是 FTV workflow proxy，不是手工 dense lesion segmentation。
- actual legacy mask 的 bitwise origin reconstruction fraction 为 {format_pct(all_row['origin_exact_fraction'])}；unique 为 {format_pct(all_row['origin_unique_fraction'])}、ambiguous 为 {format_pct(all_row['origin_ambiguous_fraction'])}。无法恢复的行没有伪造 margin。

## 2. LD semantics

LD 来源于 site radiologist MRI report，字段 `LD_T0`–`LD_T3` 分别映射 T0 pre-NAC、T1 early NAC、T2 inter-regimen、T3 pre-surgery。精确工作簿和同源资料没有明示单位，因此状态为 `LD_UNIT_NOT_EXPLICIT`；不能把值擅自换成 mm。0 是真实编码值，但来源不能区分 complete response、non-measurable、below detection 或 encoding floor，故状态为 `AMBIGUOUS_ZERO_SEMANTICS`。

本轮严格 overlap 中，T0/T1/T2/T3 的 LD zero fraction 见下表；T2/T3 floor 不参与放宽 containment gate。

## 3. 真实 crop 与计算协议

当前 cache 为 `[4,8,32,96,96]`，前七通道为 DCE7，第八通道是 binary localization support。crop 是固定 `(Z,Y,X)=(32,96,96)` voxel；crop 前不做 spacing harmonization，物理视野随 scanner/visit 变化。clean 公式以 released T0 bbox center 投影到后续 visit，但 legacy builder 未保存 origin 且存在历史舍入差异，所以本轮围绕 clean start 搜索并要求 full mask crop 与 actual cache mask bitwise exact match。1,500/1,500 visit 的 DCE 与 FTV mask 在实际 reader 的 index order 下 shape、spacing 和 slice-first handling 一致；mm 量仅解释为 matched-spacing index-space geometry，不声称 world-space affine registration。

Containment ratio 直接取 `actual cached support voxels / full support voxels`。`suspected_truncation` 定义为任一 boundary touch、ratio<0.99 或 support 不可审计；`severe_truncation` 为 ratio<0.90；`sufficient_containment` 要求 ratio≥0.99 且无 boundary touch。cache mask 为空而 full mask 非空时按 complete miss、ratio=0、severe truncation 处理。独立代码复核另加不改变 gate 的保守敏感性：`exact_full_support_containment` 要求一个 full-support voxel 都不丢，并比较 cached/full fixed-direction maximum-extent proxy。

## 4. 按访视结果

| Visit | n | Boundary touch | Suspected | Severe | Sufficient | Exact full support | Median margin mm | Q05 margin mm | LD zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

所有 visit 合并：boundary touch {format_pct(all_row['boundary_touch_rate'])}，suspected truncation {format_pct(all_row['suspected_truncation_rate'])}，severe truncation {format_pct(all_row['severe_truncation_rate'])}，sufficient containment {format_pct(all_row['sufficient_containment_rate'])}。更保守的 exact full-support retention 只有 {format_pct(all_row['exact_full_support_containment_rate'])}，whole-union approximate extent retention 中位数为 {all_row['whole_union_extent_retention_median']:.3f}。

## 5. Early visit 与 large-LD audit

T0/T1 合并 suspected truncation 为 {format_pct(early['suspected_truncation_rate'])}（gate ≤10%）；T0/T1 top-quartile LD 为 {format_pct(top25['suspected_truncation_rate'])}（gate ≤20%），其中 exact full-support retention 为 {format_pct(top25['exact_full_support_containment_rate'])}。T0/T1 pooled `Spearman(LD, minimum margin)` 为 {early['ld_margin_spearman']:.3f}，unique-only sensitivity 为 {early['ld_margin_spearman_unique_only']:.3f}。full-support largest-component approximate extent 与 reported LD 的 Spearman 为 {early['ld_approx_extent_spearman_exploratory']:.3f}；LD 单位未确认不影响秩，但这些值不提供物理校准，也不能证明 radiologist target lesion 与 FTV support 完全一致。

![Boundary touch](../figures/01_boundary_touch_rate_by_visit.png)

![Margin distribution](../figures/02_margin_distribution_by_visit.png)

![LD vs margin](../figures/03_ld_vs_margin_hexbin.png)

![Contained vs truncated](../figures/04_contained_vs_truncated_ld_distribution.png)

![Large LD](../figures/05_large_ld_subgroup_truncation.png)

![Schematic](../figures/06_privacy_safe_containment_schematic.png)

## 6. Gate

预注册五项 gate 的机器可读结果在 `metrics/crop_containment_gate.json`。决策未使用 pCR、clinical、treatment 或 test performance，也未修改阈值。保守 extent sensitivity 是独立代码复核后、查看 gate 汇总前加入的非 gate 指标；它只会加强或限定解释，不会把 NO-GO 改成 GO。

## 7. 局限性

1. FTV inclusion region由 inverse bit-coded analysis mask派生，不能等同 radiologist 所测 target lesion 的 dense segmentation；multifocal whole-union extent 尤其可能大于单病灶 LD。
2. legacy builder source与保存的 crop origin缺失；origin 由 actual cached mask exact matching反推，未恢复行只保留 origin-independent containment ratio/boundary结果。
3. 当前 legacy pipeline 按 index order/pixdim 而非 world-space affine registration 工作；本轮显式验证 DCE-mask shape/spacing/axis handling 一致，但不把它扩大解释为 affine registration QC。
4. reported LD 单位未确认，因此 segmentation-derived `approx_max_extent_mm` 只做 exploratory rank sanity，绝不替代 grounding target。
5. LD–margin hexbin 对每个 bin 使用 `n≥5` suppression；公开表对小于 5 的 LD distribution cell 抑制敏感统计。
6. T2/T3 LD zero floor分别明显增大；其临床语义仍不明确。

## 8. 下一步

{next_step}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)


def parse_args() -> argparse.Namespace:
    dgrs_root = (
        Path(os.environ["DGRS_DATA_ROOT"]) if os.environ.get("DGRS_DATA_ROOT") else None
    )
    raw_root = (
        Path(os.environ["ISPY2_RAW_ROOT"]) if os.environ.get("ISPY2_RAW_ROOT") else None
    )
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
        "--preprocessed-root",
        type=Path,
        default=(dgrs_root / "I-SPY2") if dgrs_root else None,
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=(raw_root / "Multi-feature-MRI-NACT-Data.xlsx") if raw_root else None,
    )
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = require_path(args.config, "Stage A config")
    args.overlap = require_path(args.overlap, "strict overlap")
    args.preprocessed_root = require_path(
        args.preprocessed_root, "DGRS_DATA_ROOT/I-SPY2"
    )
    args.workbook = require_path(
        args.workbook, "ISPY2_RAW_ROOT/Multi-feature-MRI-NACT-Data.xlsx"
    )
    output_root = args.output_root.expanduser().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    crop_size = tuple(int(value) for value in config["input_contract"]["crop_size_zyx"])
    radius = int(config["input_contract"]["legacy_origin_search_radius_vox"])

    workbook_hash = file_sha256(args.workbook)
    overlap_hash = file_sha256(args.overlap)
    if workbook_hash != EXPECTED_WORKBOOK_SHA256:
        raise ValueError(f"workbook SHA drift: {workbook_hash}")
    if overlap_hash != EXPECTED_OVERLAP_SHA256:
        raise ValueError(f"overlap SHA drift: {overlap_hash}")

    destinations = [
        output_root / "metrics" / "crop_containment_patient_visit.csv",
        output_root / "metrics" / "crop_containment_summary.csv",
        output_root / "metrics" / "crop_containment_by_timepoint.csv",
        output_root / "metrics" / "crop_containment_by_ld_quantile.csv",
        output_root / "metrics" / "crop_containment_gate.json",
        output_root / "metrics" / "ld_containment_distribution.csv",
        output_root / "metrics" / "stage_a_input_provenance.json",
        output_root / "reports" / "crop_containment_report.md",
        *(
            output_root / "figures" / name
            for name in (
                "01_boundary_touch_rate_by_visit.png",
                "02_margin_distribution_by_visit.png",
                "03_ld_vs_margin_hexbin.png",
                "04_contained_vs_truncated_ld_distribution.png",
                "05_large_ld_subgroup_truncation.png",
                "06_privacy_safe_containment_schematic.png",
            )
        ),
    ]
    if not args.overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("输出已存在；如需重跑请显式使用 --overwrite")

    matched, workbook = load_inputs(args, config)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(matched.itertuples(index=False), start=1):
        records.extend(audit_patient(row, args.preprocessed_root, crop_size, radius))
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"processed {index}/{len(matched)} patients", flush=True)
    patient_visit = pd.DataFrame(records)
    if len(patient_visit) != config["cohort"]["expected_patient_visits"]:
        raise ValueError(f"patient-visit count 漂移：{len(patient_visit)}")
    patient_visit = add_ld_quantile_flags(patient_visit)
    by_timepoint = build_timepoint_summary(patient_visit)
    summary = build_overall_summary(patient_visit)
    by_quantile = build_ld_quantile_summary(patient_visit)
    distribution = build_ld_distribution_summary(patient_visit)
    gate = decide_gate(patient_visit, summary, by_quantile, config)

    atomic_csv(
        output_root / "metrics" / "crop_containment_patient_visit.csv", patient_visit
    )
    atomic_csv(output_root / "metrics" / "crop_containment_summary.csv", summary)
    atomic_csv(
        output_root / "metrics" / "crop_containment_by_timepoint.csv", by_timepoint
    )
    atomic_csv(
        output_root / "metrics" / "crop_containment_by_ld_quantile.csv", by_quantile
    )
    atomic_csv(
        output_root / "metrics" / "ld_containment_distribution.csv", distribution
    )
    atomic_json(output_root / "metrics" / "crop_containment_gate.json", gate)
    atomic_json(
        output_root / "metrics" / "stage_a_input_provenance.json",
        {
            "schema_version": 1,
            "source_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "workbook": {
                "public_label": "${ISPY2_RAW_ROOT}/Multi-feature-MRI-NACT-Data.xlsx",
                "sha256": workbook_hash,
            },
            "overlap": {
                "public_label": "radiomics_patient_overlap.csv",
                "sha256": overlap_hash,
            },
            "preprocessed_root": "${DGRS_DATA_ROOT}/I-SPY2",
            "legacy_cache": "${DGRS_DATA_ROOT}/I-SPY2/_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96",
            "patients": int(len(matched)),
            "patient_visits": int(len(patient_visit)),
            "workbook_rows": int(len(workbook)),
            "pcr_read_for_stage_a": False,
        },
    )
    save_figures(patient_visit, by_timepoint, by_quantile, output_root / "figures")
    write_report(
        by_timepoint,
        summary,
        by_quantile,
        gate,
        output_root / "reports" / "crop_containment_report.md",
    )
    print(
        json.dumps(
            {"decision": gate["decision"], "failed_criteria": gate["failed_criteria"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
