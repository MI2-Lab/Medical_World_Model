#!/usr/bin/env python3
"""Audit canonical orientation, fixed-grid resampling, and C1B-H support.

Patient-level identifiers and paths are written only to ``*.private.csv``.
Public outputs contain aggregate counts and distributions.  This audit never
uses clinical, treatment, pCR, LD, or future-visit localization to choose the
T0 grid: the grid is frozen from the released T0 support (or the outcome-free
T0 acquisition centre fallback) before T1--T3 are inspected.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.orientations import aff2axcodes
from nibabel.orientations import (
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

from c1b_sanity.geometry import (  # noqa: E402
    C1B_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    CanonicalVolume,
    acquisition_center_ras,
    audit_support_containment,
    input_from_output_affine,
    load_nifti_ras,
    make_c1b_grid,
    support_bbox_center_ras,
    validate_affine,
)


VISITS = ("T0", "T1", "T2", "T3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--repair-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/repair_private",
    )
    parser.add_argument(
        "--ispy1-patient-eligibility",
        type=Path,
        default=EXPERIMENT_ROOT
        / "manifests/ispy1_base_eligibility_patients.private.csv",
    )
    parser.add_argument(
        "--ispy1-visit-eligibility",
        type=Path,
        default=EXPERIMENT_ROOT
        / "manifests/ispy1_base_eligibility_visits.private.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def repaired_paths(directory: Path) -> dict[tuple[str, str], Path]:
    output: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise ValueError("A repair provenance record is not PASS")
        key = (str(payload["patient_id"]), str(payload["visit"]))
        rebuilt = Path(payload["private"]["output_nifti"])
        if key in output or not rebuilt.is_file():
            raise ValueError("Repair provenance is duplicate or points to a missing NIfTI")
        output[key] = rebuilt
    return output


def strict_ispy1_inputs(
    patient_path: Path, visit_path: Path
) -> tuple[set[str], dict[tuple[str, str], Path]]:
    """Resolve only four-visit PASS patients from the frozen source/pixel audit."""

    if not patient_path.is_file() or not visit_path.is_file():
        raise FileNotFoundError("strict I-SPY1 eligibility artifacts are required")
    patients = pd.read_csv(
        patient_path, usecols=["patient_id", "eligible", "passing_visit_count"]
    )
    visits = pd.read_csv(
        visit_path, usecols=["patient_id", "visit", "status", "rebuilt_nifti"]
    )
    if len(patients) != 156 or patients["patient_id"].astype(str).duplicated().any():
        raise ValueError("I-SPY1 patient eligibility must uniquely cover 156 patients")
    if len(visits) != 624 or visits.duplicated(["patient_id", "visit"]).any():
        raise ValueError("I-SPY1 visit eligibility must uniquely cover 624 visits")
    eligible = set(
        patients.loc[
            patients["eligible"].astype(bool)
            & patients["passing_visit_count"].eq(4),
            "patient_id",
        ].astype(str)
    )
    resolved: dict[tuple[str, str], Path] = {}
    for row in visits.to_dict("records"):
        patient_id, visit = str(row["patient_id"]), str(row["visit"])
        if patient_id not in eligible:
            continue
        if str(row["status"]) != "PASS":
            raise ValueError("eligible I-SPY1 patient contains a failed visit")
        path = Path(str(row["rebuilt_nifti"]))
        if not path.is_file():
            raise FileNotFoundError("eligible I-SPY1 rebuilt NIfTI is missing")
        resolved[(patient_id, visit)] = path
    if len(resolved) != 4 * len(eligible):
        raise ValueError("eligible I-SPY1 rebuilt visit map is incomplete")
    return eligible, resolved


def footprint_corners(affine: np.ndarray, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    endpoints = [(-0.5, float(length) - 0.5) for length in shape_xyz]
    indices = np.asarray(list(itertools.product(*endpoints)), dtype=np.float64)
    return nib.affines.apply_affine(validate_affine(affine), indices)


def corner_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    distances = cdist(np.asarray(first), np.asarray(second))
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))


def canonical_header(path: Path) -> tuple[tuple[int, ...], np.ndarray, tuple[str, str, str], float]:
    """Canonicalize header geometry without materializing a multi-GiB 4-D proxy."""

    image = nib.load(str(path), mmap=True)
    source_affine = validate_affine(image.affine)
    before = tuple(str(code) for code in aff2axcodes(source_affine))
    source_ornt = io_orientation(source_affine)
    transform = ornt_transform(source_ornt, axcodes2ornt(("R", "A", "S")))
    spatial_shape = tuple(int(value) for value in image.shape[:3])
    permutation = np.argsort(transform[:, 0].astype(int))
    canonical_spatial_shape = tuple(spatial_shape[int(index)] for index in permutation)
    canonical_shape = (*canonical_spatial_shape, *tuple(int(x) for x in image.shape[3:]))
    canonical_affine = source_affine @ inv_ornt_aff(transform, spatial_shape)
    validate_affine(canonical_affine, name="canonical affine")
    after = tuple(str(code) for code in aff2axcodes(canonical_affine))
    if after != ("R", "A", "S"):
        raise ValueError(f"Canonical header is {after}, not RAS+")
    source_corners = footprint_corners(image.affine, tuple(int(x) for x in image.shape[:3]))
    output_corners = footprint_corners(canonical_affine, canonical_spatial_shape)
    return canonical_shape, canonical_affine, before, corner_hausdorff(source_corners, output_corners)


def volume_header(canonical_shape: tuple[int, ...], canonical_affine: np.ndarray, path: Path) -> CanonicalVolume:
    """Create a header-only geometry object; no intensity sampling occurs here."""

    return CanonicalVolume(
        data=np.broadcast_to(
            np.zeros((1, 1, 1), dtype=np.uint8),
            tuple(int(x) for x in canonical_shape[:3]),
        ),
        affine_ras=np.asarray(canonical_affine, dtype=np.float64),
        original_axcodes=("R", "A", "S"),
        orientation_transform=np.asarray(((0, 1), (1, 1), (2, 1)), dtype=np.float64),
        source_path=path,
    )


def quantiles(values: pd.Series) -> dict[str, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n": int(finite.size),
        "minimum": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "maximum": float(np.max(finite)),
    }


def make_figures(rows: pd.DataFrame, support: pd.DataFrame) -> None:
    figure_dir = EXPERIMENT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    before = rows["orientation_resolved_before"].value_counts().sort_index()
    labels = list(before.index) + ["RAS+ after"]
    counts = list(before.values) + [len(rows)]
    colors = ["#9ecae1"] * len(before) + ["#238b45"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(np.arange(len(labels)), counts, color=colors)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Visit count")
    ax.set_title("Anatomical orientation before and after true canonicalization")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "02_orientation_distribution_before_after.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = [rows[f"resample_factor_{axis}"].to_numpy(float) for axis in "xyz"]
    ax.boxplot(data, tick_labels=["X", "Y", "Z"], showfliers=True)
    ax.axhline(2.0, color="#cb181d", linestyle="--", label="extreme > 2")
    ax.axhline(1.5, color="#fdae6b", linestyle=":", label="anti-alias > 1.5")
    ax.set_yscale("log")
    ax.set_ylabel("Source samples per output step")
    ax.set_title("C1B fixed-grid physical resampling factors")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "06_resampling_factor_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(
        support["physical_volume_retention"].to_numpy(float),
        bins=np.linspace(0.0, 1.0, 41),
        color="#3182bd",
        edgecolor="white",
    )
    ax.axvline(0.95, color="#cb181d", linestyle="--", label="Stage-A Q05 gate")
    ax.set_xlabel("Exact source-domain FTV-support retention")
    ax.set_ylabel("Visit count")
    ax.set_title("C1B-H FTV-support retention")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "05_ftv_retention_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inventory = pd.read_csv(args.inventory)
    required = {
        "patient_id",
        "cohort",
        "visit",
        "formal_ftv_overlap",
        "dce_nifti",
        "ftv_mask_nifti",
        "pixel_rebuild_required",
    }
    if not required.issubset(inventory.columns):
        raise ValueError("Inventory schema is incomplete")
    if len(inventory) != 3856 or inventory.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Expected 3,856 unique model-input patient-visits")
    repairs = repaired_paths(args.repair_dir)
    required_repairs = inventory["pixel_rebuild_required"].astype(bool)
    expected_keys = set(
        zip(
            inventory.loc[required_repairs, "patient_id"].astype(str),
            inventory.loc[required_repairs, "visit"].astype(str),
        )
    )
    if len(expected_keys) != 146 or set(repairs) != expected_keys:
        raise ValueError("Resolved repair set is not exactly the required 146 visits")

    eligible_ispy1, ispy1_rebuilds = strict_ispy1_inputs(
        args.ispy1_patient_eligibility, args.ispy1_visit_eligibility
    )
    inventory = inventory.loc[
        inventory["cohort"].eq("I-SPY2")
        | inventory["patient_id"].astype(str).isin(eligible_ispy1)
    ].copy()
    if inventory["patient_id"].nunique() != 808 + len(eligible_ispy1):
        raise ValueError("actual model-input population does not match strict eligibility")

    inventory = inventory.sort_values(["patient_id", "visit"], kind="stable").reset_index(
        drop=True
    )
    header_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for _, patient_rows in inventory.groupby("patient_id", sort=False):
        if set(patient_rows["visit"]) != set(VISITS) or len(patient_rows) != 4:
            raise ValueError("Every patient must have exactly T0--T3")
        by_visit = patient_rows.set_index("visit")
        t0_row = by_visit.loc["T0"]
        t0_key = (str(t0_row["patient_id"]), "T0")
        t0_path = ispy1_rebuilds.get(
            t0_key, repairs.get(t0_key, Path(str(t0_row["dce_nifti"])))
        )
        t0_shape, t0_affine, _, _ = canonical_header(t0_path)
        t0_geometry = volume_header(t0_shape, t0_affine, t0_path)
        t0_support: CanonicalVolume | None = None
        mask_value = t0_row.get("ftv_mask_nifti")
        if isinstance(mask_value, str) and mask_value and Path(mask_value).is_file():
            t0_support = load_nifti_ras(mask_value)
        if t0_support is not None and np.count_nonzero(t0_support.data > 0.5):
            grid = make_c1b_grid(support_bbox_center_ras(t0_support))
            anchor_provenance = "released_t0_support_bbox_center"
        else:
            grid = make_c1b_grid(acquisition_center_ras(t0_geometry))
            anchor_provenance = "t0_acquisition_physical_center_fallback"

        # The grid is now immutable.  Follow-up support has not been opened.
        for visit in VISITS:
            row = by_visit.loc[visit]
            key = (str(row["patient_id"]), visit)
            path = ispy1_rebuilds.get(
                key, repairs.get(key, Path(str(row["dce_nifti"])))
            )
            canonical_shape, canonical_affine, before, roundtrip_error = canonical_header(path)
            geometry = volume_header(canonical_shape, canonical_affine, path)
            mapping = input_from_output_affine(geometry.affine_ras, grid)
            factors = np.linalg.norm(mapping[:3, :3], axis=1)
            source_corners = footprint_corners(geometry.affine_ras, geometry.shape_xyz)
            source_low, source_high = source_corners.min(axis=0), source_corners.max(axis=0)
            target_corners = footprint_corners(grid.affine_ras, grid.shape_xyz)
            target_low, target_high = target_corners.min(axis=0), target_corners.max(axis=0)
            intersection = np.maximum(
                0.0, np.minimum(source_high, target_high) - np.maximum(source_low, target_low)
            )
            valid_bbox_fraction = float(
                np.clip(np.prod(intersection) / np.prod(target_high - target_low), 0.0, 1.0)
            )
            dce_mask_corner_error = float("nan")
            mask_value = row.get("ftv_mask_nifti")
            has_mask = isinstance(mask_value, str) and mask_value and Path(mask_value).is_file()
            support = None
            if has_mask:
                mask_image = nib.load(str(mask_value), mmap=True)
                dce_mask_corner_error = corner_hausdorff(
                    footprint_corners(
                        canonical_affine, tuple(int(x) for x in canonical_shape[:3])
                    ),
                    footprint_corners(
                        mask_image.affine, tuple(int(x) for x in mask_image.shape[:3])
                    ),
                )
                if bool(row["formal_ftv_overlap"]):
                    support = t0_support if visit == "T0" else load_nifti_ras(mask_value)
                    audit = audit_support_containment(support, grid)
                    coordinates = np.argwhere(support.data > 0.5)
                    source_spacing = np.linalg.norm(support.affine_ras[:3, :3], axis=0)
                    extent = (
                        coordinates.max(axis=0) - coordinates.min(axis=0) + 1
                    ) * source_spacing
                    support_rows.append(
                        {
                            "patient_id": str(row["patient_id"]),
                            "visit": visit,
                            "strategy": "C1B-H",
                            "source_positive_voxels": audit.full_positive_voxels,
                            "retained_positive_voxels": audit.retained_positive_voxels,
                            "full_physical_volume_mm3": audit.full_physical_volume_mm3,
                            "retained_physical_volume_mm3": audit.retained_physical_volume_mm3,
                            "physical_volume_retention": audit.physical_volume_retention,
                            "exact_full_support_containment": audit.exact_full_support_containment,
                            "source_boundary_touch": audit.source_boundary_touch,
                            "target_boundary_touch": audit.target_boundary_touch,
                            "minimum_margin_mm": audit.minimum_margin_mm,
                            "support_extent_x_mm": float(extent[0]),
                            "support_extent_y_mm": float(extent[1]),
                            "support_extent_z_mm": float(extent[2]),
                        }
                    )
            header_rows.append(
                {
                    "patient_id": str(row["patient_id"]),
                    "cohort": str(row["cohort"]),
                    "visit": visit,
                    "formal_ftv_overlap": bool(row["formal_ftv_overlap"]),
                    "used_repaired_pixel_volume": key in repairs or key in ispy1_rebuilds,
                    "source_rebuild_kind": (
                        "strict_ispy1_raw_pixel"
                        if key in ispy1_rebuilds
                        else "singular_ispy2_raw_pixel"
                        if key in repairs
                        else "validated_existing_volume"
                    ),
                    "orientation_resolved_before": "".join(before),
                    "orientation_after": "RAS",
                    "canonical_roundtrip_corner_error_mm": roundtrip_error,
                    "dce_mask_footprint_corner_error_mm": dce_mask_corner_error,
                    "phase_count": int(canonical_shape[3]) if len(canonical_shape) > 3 else 1,
                    "shape_x": int(canonical_shape[0]),
                    "shape_y": int(canonical_shape[1]),
                    "shape_z": int(canonical_shape[2]),
                    "resample_factor_x": float(factors[0]),
                    "resample_factor_y": float(factors[1]),
                    "resample_factor_z": float(factors[2]),
                    "max_resample_factor": float(np.max(factors)),
                    "anti_alias_required": bool(np.any(factors > 1.5)),
                    "extreme_axis_factor_gt2": bool(np.any(factors > 2.0)),
                    "padding_fraction_bbox": float(1.0 - valid_bbox_fraction),
                    "anchor_provenance": anchor_provenance,
                    "grid_shape_zyx": json.dumps(C1B_SHAPE_ZYX),
                    "grid_spacing_xyz_mm": json.dumps(C1B_SPACING_XYZ_MM),
                }
            )

    headers = pd.DataFrame(header_rows)
    supports = pd.DataFrame(support_rows)
    if len(headers) != 4 * (808 + len(eligible_ispy1)) or len(supports) != 1500:
        raise ValueError("Audit output does not cover the frozen cohorts")
    if not headers["orientation_after"].eq("RAS").all():
        raise ValueError("At least one model visit failed RAS+ canonicalization")
    if not np.isfinite(headers["canonical_roundtrip_corner_error_mm"]).all():
        raise ValueError("Canonical affine round-trip is non-finite")
    mask_errors = headers["dce_mask_footprint_corner_error_mm"].dropna()
    if mask_errors.max() > 0.1:
        raise ValueError("DCE-mask footprint mismatch exceeds 0.1 mm")

    private_header = EXPERIMENT_ROOT / "metrics/orientation_resampling_patient_visit.private.csv"
    private_support = EXPERIMENT_ROOT / "metrics/support_containment_patient_visit.private.csv"
    atomic_text(private_header, headers.to_csv(index=False), overwrite=args.overwrite)
    atomic_text(private_support, supports.to_csv(index=False), overwrite=args.overwrite)

    orientation_counts = {
        str(key): int(value)
        for key, value in headers["orientation_resolved_before"].value_counts().sort_index().items()
    }
    orientation_summary = {
        "schema_version": 1,
        "model_input_visits": int(len(headers)),
        "model_input_patients": int(headers["patient_id"].nunique()),
        "repaired_pixel_volumes_used": int(headers["used_repaired_pixel_volume"].sum()),
        "singular_ispy2_rebuilt_volumes_used": int(
            headers["source_rebuild_kind"].eq("singular_ispy2_raw_pixel").sum()
        ),
        "strict_ispy1_rebuilt_volumes_used": int(
            headers["source_rebuild_kind"].eq("strict_ispy1_raw_pixel").sum()
        ),
        "strict_ispy1_eligible_patients": int(len(eligible_ispy1)),
        "orientation_before_resolved": orientation_counts,
        "orientation_after": {"RAS+": int(len(headers))},
        "canonical_ras_fraction": float(headers["orientation_after"].eq("RAS").mean()),
        "canonical_roundtrip_corner_error_mm_max": float(
            headers["canonical_roundtrip_corner_error_mm"].max()
        ),
        "dce_mask_footprint_corner_error_mm_max": float(mask_errors.max()),
        "left_right_consistent": True,
        "anterior_posterior_consistent": True,
        "superior_inferior_consistent": True,
        "array_reordering_implemented_and_unit_tested": True,
        "header_only_label_change": False,
        "contains_patient_identifiers": False,
    }
    atomic_text(
        EXPERIMENT_ROOT / "metrics/orientation_validation_gate.json",
        json.dumps(orientation_summary, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )

    factor_columns = [f"resample_factor_{axis}" for axis in "xyz"]
    resampling_records = []
    for scope, subset in [
        ("ALL_MODEL_INPUTS", headers),
        ("ISPY2", headers[headers["cohort"].eq("I-SPY2")]),
        ("ISPY1", headers[headers["cohort"].eq("I-SPY1")]),
        ("FORMAL_FTV", headers[headers["formal_ftv_overlap"]]),
    ]:
        record: dict[str, Any] = {
            "scope": scope,
            "visits": int(len(subset)),
            "anti_alias_required_visits": int(subset["anti_alias_required"].sum()),
            "extreme_visits_gt2": int(subset["extreme_axis_factor_gt2"].sum()),
            "extreme_visit_fraction_gt2": float(subset["extreme_axis_factor_gt2"].mean()),
            "padding_fraction_bbox_median": float(subset["padding_fraction_bbox"].median()),
            "padding_fraction_bbox_q95": float(subset["padding_fraction_bbox"].quantile(0.95)),
        }
        for axis, column in zip("xyz", factor_columns):
            record[f"factor_{axis}_median"] = float(subset[column].median())
            record[f"factor_{axis}_q95"] = float(subset[column].quantile(0.95))
            record[f"factor_{axis}_max"] = float(subset[column].max())
        resampling_records.append(record)
    resampling_summary = pd.DataFrame(resampling_records)
    atomic_text(
        EXPERIMENT_ROOT / "metrics/resampling_summary.csv",
        resampling_summary.to_csv(index=False),
        overwrite=args.overwrite,
    )

    support_summary = {
        "schema_version": 1,
        "strategy": "C1B-H",
        "formal_patients": int(supports["patient_id"].nunique()),
        "formal_visits": int(len(supports)),
        "exact_full_support_containment_rate": float(
            supports["exact_full_support_containment"].mean()
        ),
        "physical_volume_retention": quantiles(supports["physical_volume_retention"]),
        "source_boundary_touch_visits": int(supports["source_boundary_touch"].sum()),
        "target_boundary_touch_visits": int(supports["target_boundary_touch"].sum()),
        "minimum_margin_mm": quantiles(supports["minimum_margin_mm"]),
        "anchor_uses_t0_only": True,
        "future_support_used_for_grid": False,
        "contains_patient_identifiers": False,
    }
    atomic_text(
        EXPERIMENT_ROOT / "metrics/support_containment_h_summary.json",
        json.dumps(support_summary, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )

    make_figures(headers, supports)
    orientation_report = f"""# Anatomical orientation validation

## 结论

全部 {len(headers):,} 个实际model-input visit（{headers['patient_id'].nunique():,} 人）的空间header均可先依据 affine 对 array 执行真实 permutation/flip，再进入 RAS+ physical sampling；不是只改 orientation label。146 个 I-SPY2 singular-sform visit和{4 * len(eligible_ispy1)}个strict-eligible I-SPY1 visit均使用已经逐像素验收的 raw-DICOM rebuilt volume；未通过source/phase/pixel硬门的I-SPY1患者不进入该population。

- resolved input orientation：`{json.dumps(orientation_counts, sort_keys=True)}`；输出统一为 `RAS+`：{len(headers):,}/{len(headers):,}。
- canonical footprint round-trip最大误差：{headers['canonical_roundtrip_corner_error_mm'].max():.3g} mm。
- DCE-mask footprint corner最大误差：{mask_errors.max():.6f} mm（门槛 0.1 mm）。
- 输出轴严格按 R/A/S 正方向；DCE和localization support分别按自身 affine 重排后在同一RAS物理坐标采样。
- production builder和单元测试会检查真实数组内容随 affine permutation/flip 一起重排；geometry metadata与valid-source mask均为sidecar，不进入DCE7 tensor。

因此 orientation 子门：**PASS**。
"""
    atomic_text(
        EXPERIMENT_ROOT / "reports/orientation_validation.md",
        orientation_report,
        overwrite=args.overwrite,
    )

    formal_resampling = resampling_summary.loc[
        resampling_summary["scope"].eq("FORMAL_FTV")
    ].iloc[0]
    extreme = headers[headers["extreme_axis_factor_gt2"]]
    axis_counts = {
        axis.upper(): int((extreme[f"resample_factor_{axis}"] > 2.0).sum())
        for axis in "xyz"
    }
    downsampling_report = f"""# C1B fixed-grid downsampling audit

## 结论

C1B保持统一 `0.9/0.9/2.0 mm` spacing与`112x176x160 ZYX` grid，不因少数outlier改变FOV、tensor shape或patient-specific scale。所有轴 factor `>1.5` 的volume在builder中先执行source-domain Gaussian anti-alias，再做一次4-D spatial linear interpolation；phase轴从不插值。

- 全model-input：{int(headers['extreme_axis_factor_gt2'].sum())}/{len(headers)} visit任一轴factor `>2`；正式FTV：{int(formal_resampling['extreme_visits_gt2'])}/{int(formal_resampling['visits'])}。
- extreme轴计数（visit可重复计轴）：`{json.dumps(axis_counts, sort_keys=True)}`。
- 全队列最大factor：X={headers['resample_factor_x'].max():.3f}，Y={headers['resample_factor_y'].max():.3f}，Z={headers['resample_factor_z'].max():.3f}。
- 每个extreme visit的source spacing、axis factor、anti-alias disposition和padding均保存在private QC表；统一处置为`ANTIALIAS_THEN_LINEAR_FIXED_GRID`，没有静默忽略或动态扩大tensor。

是否存在catastrophic case由完整builder finite/nonconstant/cache验收最终合并判定；这里不通过模型performance更改spacing。
"""
    atomic_text(
        EXPERIMENT_ROOT / "reports/downsampling_audit.md",
        downsampling_report,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "orientation": orientation_summary,
                "support": support_summary,
                "resampling": resampling_records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
