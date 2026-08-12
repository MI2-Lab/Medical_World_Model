#!/usr/bin/env python3
"""Run the BPE source-observability audit without reading outcome values.

The script deliberately loads only allowlisted geometry/source-availability
columns.  The BPE scalar workbook is inspected for schema/missingness and ID
mapping; BPE values are never retained, summarized, sorted, or used in gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pydicom


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_core import (  # noqa: E402
    INTERVALS,
    VISITS,
    canonical_json_sha256,
    decide_scientific_classification,
    deterministic_case_sample,
    grid_center_from_affine,
    is_axis_aligned_ras,
    physical_bounds_xyz,
    physical_extent_xyz,
    sha256_file,
    summarize_numeric,
    validate_affine,
)


NOT_EVALUABLE = "NOT_EVALUABLE_SOURCE_ROI_NOT_AVAILABLE"
SOURCE_STATUS = "SOURCE_ROI_NOT_AVAILABLE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "audit.json",
        help="Frozen audit configuration",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_sources(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = config["paths"]
    pairs = (
        ("source_workbook", "source_workbook_sha256"),
        ("matched_transition_table", "matched_transition_table_sha256"),
        ("c1b_visit_inventory", "c1b_visit_inventory_sha256"),
        ("frozen_c1b_grids", "frozen_c1b_grids_sha256"),
        ("c1b_cache_inventory", "c1b_cache_inventory_sha256"),
        ("orientation_inventory", "orientation_inventory_sha256"),
        ("orientation_gate", "orientation_gate_sha256"),
        ("prior_bpe_fov_firewall", "prior_bpe_fov_firewall_sha256"),
        ("goal6_feature_inventory", "goal6_feature_inventory_sha256"),
        ("raw_source_inventory", "raw_source_inventory_sha256"),
    )
    result = {}
    for path_key, hash_key in pairs:
        configured_path = Path(paths[path_key])
        path = resolve_path(paths[path_key])
        require(path.is_file(), f"required source missing: {path_key}")
        actual = sha256_file(path)
        require(actual == paths[hash_key], f"SHA-256 mismatch for {path_key}")
        result[path_key] = {
            "path_scope": "external" if configured_path.is_absolute() else "repository",
            "sha256": actual,
            "status": "VERIFIED",
        }
    repair_directory = resolve_path(paths["repair_record_directory"])
    require(repair_directory.is_dir(), "repair record directory missing")
    repair_digest = hashlib.sha256(b"c1b-repair-record-set-v1\0")
    repair_files = sorted(repair_directory.glob("*.json"), key=lambda item: item.name)
    require(len(repair_files) == 146, "repair record count drifted")
    for path in repair_files:
        repair_digest.update(path.name.encode("utf-8"))
        repair_digest.update(b"\0")
        repair_digest.update(sha256_file(path).encode("ascii"))
        repair_digest.update(b"\0")
    observed = repair_digest.hexdigest()
    require(observed == paths["repair_record_set_sha256"], "repair record set SHA-256 mismatch")
    result["repair_record_set"] = {
        "record_count": len(repair_files),
        "sha256": observed,
        "status": "VERIFIED",
    }
    return result


def canonical_trial_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("missing trial ID")
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if len(text) != 6 or not text.isdigit():
        raise ValueError("trial ID does not satisfy the exact six-digit contract")
    return text


def read_workbook_availability(config: Mapping[str, Any]) -> pd.DataFrame:
    contract = config["source_contract"]
    source = resolve_path(config["paths"]["source_workbook"])
    identifier = "CLINICAL-TRIAL-SUBJECT-ID"
    fields = list(contract["workbook_fields"])
    # Only the ID and four BPE source-value fields are parsed.  Values are
    # immediately reduced to finite/missing flags and then dropped.
    frame = pd.read_excel(
        source,
        sheet_name=config["paths"]["source_workbook_sheet"],
        usecols=[identifier, *fields],
    )
    require(len(frame) == config["population"]["workbook_expected_patients"], "workbook row count drifted")
    require(list(frame.columns) == [identifier, *fields], "workbook BPE schema/order drifted")
    output = pd.DataFrame({"trial_id": frame[identifier].map(canonical_trial_id)})
    require(not output["trial_id"].duplicated().any(), "duplicate workbook trial ID")
    for field in fields:
        numeric = pd.to_numeric(frame[field], errors="raise").to_numpy(dtype=float)
        output[f"{field}_available"] = np.isfinite(numeric)
    output["bpe_complete"] = output.drop(columns=["trial_id"]).all(axis=1)
    return output


def read_primary_mapping(config: Mapping[str, Any]) -> pd.DataFrame:
    source = resolve_path(config["paths"]["matched_transition_table"])
    frame = pd.read_csv(
        source,
        usecols=["patient_id", "trial_id", "transition", "start_visit", "end_visit", "bpe_valid"],
        dtype={"patient_id": str, "trial_id": str, "transition": str, "start_visit": str, "end_visit": str},
    )
    frame["trial_id"] = frame["trial_id"].map(canonical_trial_id)
    require(
        set(zip(frame["start_visit"], frame["end_visit"], strict=False))
        == {("T0", "T1"), ("T1", "T2"), ("T2", "T3")},
        "transition endpoint set drifted",
    )
    require(frame["bpe_valid"].astype(bool).all(), "matched table contains invalid BPE endpoints")
    mapping = frame[["patient_id", "trial_id"]].drop_duplicates()
    require(not mapping["patient_id"].duplicated().any(), "patient maps to multiple trial IDs")
    require(not mapping["trial_id"].duplicated().any(), "trial ID maps to multiple patients")
    require(len(mapping) == config["population"]["primary_expected_patients"], "primary population drifted")
    return mapping.sort_values("patient_id").reset_index(drop=True)


def parse_json_array(value: object, *, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    result = np.asarray(json.loads(str(value)), dtype=dtype)
    if result.shape != shape:
        raise RuntimeError(f"JSON array has shape {result.shape}, expected {shape}")
    return result


def read_geometry(config: Mapping[str, Any], patient_ids: set[str]) -> pd.DataFrame:
    source = resolve_path(config["paths"]["c1b_visit_inventory"])
    columns = [
        "patient_id",
        "cohort",
        "visit",
        "resolved_dce_nifti",
        "source_shape_xyz_json",
        "source_affine_ras_json",
        "grid_contract_sha256",
        "geometry_contract_sha256",
        "valid_source_voxels",
        "target_grid_voxels",
        "valid_source_fraction",
        "has_valid_source_overlap",
        "eligibility_evidence_scope",
    ]
    frame = pd.read_csv(source, usecols=columns, dtype={"patient_id": str, "visit": str})
    frame = frame.loc[frame["patient_id"].isin(patient_ids)].copy()
    require(len(frame) == len(patient_ids) * len(VISITS), "primary geometry rows are incomplete")
    require(set(frame["visit"]) == set(VISITS), "visit set drifted")
    require((frame.groupby("patient_id")["visit"].nunique() == len(VISITS)).all(), "patient visit coverage drifted")
    require(frame["cohort"].eq("I-SPY2").all(), "primary population is not I-SPY2")
    require(frame["resolved_dce_nifti"].map(lambda value: Path(value).is_file()).all(), "reconstructed DCE unavailable")
    return frame.sort_values(["patient_id", "visit"]).reset_index(drop=True)


def read_grids(config: Mapping[str, Any], patient_ids: set[str]) -> pd.DataFrame:
    source = resolve_path(config["paths"]["frozen_c1b_grids"])
    columns = [
        "patient_id",
        "cohort",
        "anchor_provenance",
        "grid_center_x_ras_mm",
        "grid_center_y_ras_mm",
        "grid_center_z_ras_mm",
        "grid_shape_zyx_json",
        "grid_spacing_xyz_mm_json",
        "grid_affine_ras_json",
        "grid_contract_sha256",
    ]
    frame = pd.read_csv(source, usecols=columns, dtype={"patient_id": str})
    frame = frame.loc[frame["patient_id"].isin(patient_ids)].copy()
    require(len(frame) == len(patient_ids), "frozen grid coverage is incomplete")
    require(not frame["patient_id"].duplicated().any(), "duplicate frozen grid")
    return frame.sort_values("patient_id").reset_index(drop=True)


def read_cache_availability(config: Mapping[str, Any], patient_ids: set[str]) -> pd.DataFrame:
    source = resolve_path(config["paths"]["c1b_cache_inventory"])
    frame = pd.read_csv(source, usecols=["patient_id", "cohort", "cache_path"], dtype={"patient_id": str})
    frame = frame.loc[frame["patient_id"].isin(patient_ids)].copy()
    require(len(frame) == len(patient_ids), "C1B cache inventory coverage is incomplete")
    frame["cache_available"] = frame["cache_path"].map(lambda value: Path(value).is_file())
    return frame


def read_orientation(config: Mapping[str, Any], patient_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    source = resolve_path(config["paths"]["orientation_inventory"])
    columns = [
        "patient_id",
        "cohort",
        "visit",
        "orientation_resolved_before",
        "orientation_after",
        "canonical_roundtrip_corner_error_mm",
        "dce_mask_footprint_corner_error_mm",
        "phase_count",
        "shape_x",
        "shape_y",
        "shape_z",
        "anchor_provenance",
        "grid_shape_zyx",
        "grid_spacing_xyz_mm",
    ]
    frame = pd.read_csv(source, usecols=columns, dtype={"patient_id": str, "visit": str})
    frame = frame.loc[frame["patient_id"].isin(patient_ids)].copy()
    require(len(frame) == len(patient_ids) * len(VISITS), "orientation rows are incomplete")
    gate = load_json(resolve_path(config["paths"]["orientation_gate"]))
    require(gate["canonical_ras_fraction"] == 1, "upstream RAS gate did not pass")
    require(bool(gate["array_reordering_implemented_and_unit_tested"]), "upstream used header-only orientation")
    require(frame["orientation_after"].eq("RAS").all(), "primary orientation is not RAS+")
    return frame, gate


def grid_audit(config: Mapping[str, Any], grids: pd.DataFrame) -> dict[str, Any]:
    contract = config["fov_contracts"]["F1"]
    expected_shape_zyx = np.asarray(contract["shape_zyx"], dtype=int)
    expected_shape_xyz = expected_shape_zyx[::-1]
    expected_spacing = np.asarray(contract["spacing_xyz_mm"], dtype=float)
    expected_extent = np.asarray(contract["extent_xyz_mm"], dtype=float)
    max_center_error = 0.0
    max_extent_error = 0.0
    axis_aligned = 0
    affine_valid = 0
    for row in grids.itertuples(index=False):
        shape_zyx = parse_json_array(row.grid_shape_zyx_json, shape=(3,), dtype=int)
        spacing = parse_json_array(row.grid_spacing_xyz_mm_json, shape=(3,), dtype=float)
        affine = parse_json_array(row.grid_affine_ras_json, shape=(4, 4), dtype=float)
        validate_affine(affine, name="frozen C1B affine")
        affine_valid += 1
        require(np.array_equal(shape_zyx, expected_shape_zyx), "C1B shape drifted")
        require(np.allclose(spacing, expected_spacing, atol=1e-9, rtol=0.0), "C1B spacing drifted")
        extent = physical_extent_xyz(expected_shape_xyz, affine)
        max_extent_error = max(max_extent_error, float(np.max(np.abs(extent - expected_extent))))
        center = grid_center_from_affine(expected_shape_xyz, affine)
        stated = np.asarray(
            [row.grid_center_x_ras_mm, row.grid_center_y_ras_mm, row.grid_center_z_ras_mm],
            dtype=float,
        )
        max_center_error = max(max_center_error, float(np.max(np.abs(center - stated))))
        axis_aligned += int(is_axis_aligned_ras(affine))
    require(max_center_error <= 1e-6, "frozen center/affine mismatch")
    require(max_extent_error <= 1e-6, "frozen extent/affine mismatch")
    require(axis_aligned == len(grids), "frozen C1B grid is not RAS+ axis aligned")
    return {
        "patients": int(len(grids)),
        "affine_valid_patients": affine_valid,
        "axis_aligned_ras_patients": axis_aligned,
        "maximum_center_consistency_error_mm": max_center_error,
        "maximum_extent_consistency_error_mm": max_extent_error,
        "shape_zyx": expected_shape_zyx.tolist(),
        "spacing_xyz_mm": expected_spacing.tolist(),
        "extent_xyz_mm": expected_extent.tolist(),
        "orientation": "RAS+",
        "status": "PASS",
    }


def acquisition_support_rows(geometry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in geometry.itertuples(index=False):
        shape = parse_json_array(row.source_shape_xyz_json, shape=(3,), dtype=int)
        affine = parse_json_array(row.source_affine_ras_json, shape=(4, 4), dtype=float)
        validate_affine(affine, name="source affine")
        low, high = physical_bounds_xyz(shape, affine)
        extent = high - low
        rows.append(
            {
                "patient_id": row.patient_id,
                "visit": row.visit,
                "source_shape_x": int(shape[0]),
                "source_shape_y": int(shape[1]),
                "source_shape_z": int(shape[2]),
                "source_low_x_ras_mm": float(low[0]),
                "source_low_y_ras_mm": float(low[1]),
                "source_low_z_ras_mm": float(low[2]),
                "source_high_x_ras_mm": float(high[0]),
                "source_high_y_ras_mm": float(high[1]),
                "source_high_z_ras_mm": float(high[2]),
                "source_extent_x_mm": float(extent[0]),
                "source_extent_y_mm": float(extent[1]),
                "source_extent_z_mm": float(extent[2]),
                "source_voxels": int(np.prod(shape)),
                "source_affine_axis_aligned_ras": bool(is_axis_aligned_ras(affine)),
                "valid_source_fraction_on_c1b": float(row.valid_source_fraction),
                "has_valid_source_overlap": bool(row.has_valid_source_overlap),
            }
        )
    return pd.DataFrame(rows)


def raw_series_availability(
    config: Mapping[str, Any], geometry: pd.DataFrame
) -> pd.DataFrame:
    """Resolve every reconstructed source back to an existing raw DICOM series."""

    roots = {key: Path(value) for key, value in config["paths"]["raw_dicom_roots"].items()}
    source_inventory = pd.read_csv(
        resolve_path(config["paths"]["raw_source_inventory"]),
        usecols=[
            "patient_id",
            "cohort",
            "visit",
            "raw_dce_series_json",
            "raw_series_exists",
            "pixel_rebuild_required",
        ],
        dtype={"patient_id": str, "visit": str},
    )
    require(
        not source_inventory.duplicated(["patient_id", "visit"]).any(),
        "raw source inventory contains duplicate patient/visit rows",
    )
    selected = geometry[["patient_id", "visit"]].merge(
        source_inventory,
        on=["patient_id", "visit"],
        how="left",
        validate="one_to_one",
    )
    require(len(selected) == len(geometry), "raw source inventory join changed row count")
    require(selected["cohort"].eq("I-SPY2").all(), "raw source inventory cohort mismatch")

    rows = []
    for row in selected.itertuples(index=False):
        raw_value = json.loads(str(row.raw_dce_series_json))
        require(isinstance(raw_value, str), "formal I-SPY2 visit must resolve to one raw DCE series")
        directory = Path(raw_value).resolve()
        key = "ACRIN-6698" if row.patient_id.startswith("ACRIN-6698-") else "ISPY2"
        require(directory.is_relative_to(roots[key].resolve()), "raw series escaped its frozen collection root")
        require(row.patient_id in directory.parts, "raw series path identity mismatch")
        first_dicom = next(directory.glob("*.dcm"), None) if directory.is_dir() else None
        rows.append(
            {
                "patient_id": row.patient_id,
                "visit": row.visit,
                "raw_series_directory": str(directory),
                "raw_series_available": bool(
                    bool(row.raw_series_exists) and first_dicom is not None
                ),
                "first_dicom": str(first_dicom) if first_dicom is not None else "",
                "source_series_mapping_status": "FROZEN_MODEL_INPUT_INVENTORY",
                "pixel_rebuild_required": bool(row.pixel_rebuild_required),
            }
        )
    return pd.DataFrame(rows)


def _dicom_text(dataset: pydicom.Dataset, name: str) -> str:
    value = getattr(dataset, name, None)
    if value is None:
        return ""
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return "|".join(str(item) for item in value)
    return str(value)


def dicom_header_laterality_audit(raw_series: pd.DataFrame) -> pd.DataFrame:
    """Read only geometry/laterality tags from one instance per formal series."""

    tags = [
        "Laterality",
        "ImageLaterality",
        "PatientPosition",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "Rows",
        "Columns",
        "FrameOfReferenceUID",
    ]
    rows = []
    for row in raw_series.itertuples(index=False):
        require(bool(row.raw_series_available), "cannot audit missing raw DICOM series")
        dataset = pydicom.dcmread(
            row.first_dicom,
            stop_before_pixels=True,
            specific_tags=tags,
        )
        values = {name: _dicom_text(dataset, name) for name in tags}
        rows.append(
            {
                "patient_id": row.patient_id,
                "visit": row.visit,
                "laterality_tag": values["Laterality"],
                "image_laterality_tag": values["ImageLaterality"],
                "patient_position": values["PatientPosition"],
                "iop_present": bool(values["ImageOrientationPatient"]),
                "ipp_present": bool(values["ImagePositionPatient"]),
                "pixel_spacing_present": bool(values["PixelSpacing"]),
                "rows_present": bool(values["Rows"]),
                "columns_present": bool(values["Columns"]),
                "frame_of_reference_uid_present": bool(values["FrameOfReferenceUID"]),
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="")
    if ".private." in path.name:
        path.chmod(0o600)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def aggregate_extent_table(acquisition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis in "xyz":
        stats = summarize_numeric(acquisition[f"source_extent_{axis}_mm"])
        for statistic in ("minimum", "q05", "q50", "q90", "q95", "q99", "maximum"):
            rows.append(
                {
                    "support": "F2_RECONSTRUCTED_SOURCE_IMAGE_SUPPORT",
                    "axis": axis.upper(),
                    "statistic": statistic.upper(),
                    "extent_mm": stats[statistic],
                    "n_visits": stats["n"],
                    "interpretation": "separate_marginal_reconstructed_source_AABB_statistic_not_raw_footprint_equivalence_or_BPE_support",
                }
            )
    return pd.DataFrame(rows)


def audit_population_table(
    config: Mapping[str, Any],
    workbook: pd.DataFrame,
    mapping: pd.DataFrame,
    geometry: pd.DataFrame,
    caches: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    matched_trials = set(mapping["trial_id"])
    return pd.DataFrame(
        [
            {
                "stage": "SOURCE_WORKBOOK",
                "patients": len(workbook),
                "visits": len(workbook) * len(VISITS),
                "included": True,
                "reason": "source inventory",
            },
            {
                "stage": "BPE_COMPLETE_WORKBOOK",
                "patients": int(workbook["bpe_complete"].sum()),
                "visits": int(workbook["bpe_complete"].sum()) * len(VISITS),
                "included": True,
                "reason": "finite source scalar availability only",
            },
            {
                "stage": "WORKBOOK_ONLY_EXCLUDED",
                "patients": int((~workbook["trial_id"].isin(matched_trials)).sum()),
                "visits": 0,
                "included": False,
                "reason": "NO_COMPLETE_FOUR_VISIT_C1B_MATCH",
            },
            {
                "stage": "PRIMARY_MATCHED_COHORT",
                "patients": mapping["patient_id"].nunique(),
                "visits": len(geometry),
                "included": True,
                "reason": "BPE-complete exact MRI/C1B match",
            },
            {
                "stage": "RECONSTRUCTED_DCE_AVAILABLE",
                "patients": geometry.loc[geometry["resolved_dce_nifti"].map(lambda value: Path(value).is_file()), "patient_id"].nunique(),
                "visits": int(geometry["resolved_dce_nifti"].map(lambda value: Path(value).is_file()).sum()),
                "included": True,
                "reason": "technical source availability",
            },
            {
                "stage": "C1B_CACHE_AVAILABLE",
                "patients": int(caches["cache_available"].sum()),
                "visits": int(caches["cache_available"].sum()) * len(VISITS),
                "included": True,
                "reason": "technical cache availability",
            },
            {
                "stage": "RAW_DICOM_SELECTED_SERIES_AVAILABLE",
                "patients": raw.loc[raw["raw_series_available"], "patient_id"].nunique(),
                "visits": int(raw["raw_series_available"].sum()),
                "included": True,
                "reason": "selected raw series availability; extent geometry audited on reconstructed source image",
            },
            {
                "stage": "AUTHORITATIVE_BPE_SOURCE_ROI_AVAILABLE",
                "patients": 0,
                "visits": 0,
                "included": False,
                "reason": SOURCE_STATUS,
            },
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    require(config["source_contract"]["source_roi_status"] == SOURCE_STATUS, "unexpected source ROI state")
    source_verification = verify_sources(config)

    workbook = read_workbook_availability(config)
    mapping = read_primary_mapping(config)
    joined = mapping.merge(workbook[["trial_id", "bpe_complete"]], on="trial_id", how="left", validate="one_to_one")
    require(joined["bpe_complete"].eq(True).all(), "primary population is not BPE complete")
    patient_ids = set(mapping["patient_id"])

    geometry = read_geometry(config, patient_ids)
    grids = read_grids(config, patient_ids)
    caches = read_cache_availability(config, patient_ids)
    orientation, orientation_gate = read_orientation(config, patient_ids)
    raw = raw_series_availability(config, geometry)
    dicom_headers = dicom_header_laterality_audit(raw)
    acquisition = acquisition_support_rows(geometry)
    grid_summary = grid_audit(config, grids)
    require(caches["cache_available"].all(), "one or more primary C1B caches are unavailable")
    require(raw["raw_series_available"].all(), "one or more selected raw DICOM series are unavailable")
    require(geometry["has_valid_source_overlap"].astype(bool).all(), "primary cohort contains zero C1B/source overlap")

    source_roi_available = config["source_contract"]["authoritative_source_roi_manifest"] is not None
    code, classification = decide_scientific_classification(
        source_roi_available=source_roi_available,
        local_gate_pass=None,
        c1b_gate_pass=None,
        acquisition_source_complete=None,
    )
    require(code == "D", "fail-closed classification was not D")

    selected, selection_sha = deterministic_case_sample(
        patient_ids,
        int(config["laterality_audit_sample_size"]),
        salt="bpe-fov-audit-v1",
    )
    laterality_private = orientation.loc[
        orientation["patient_id"].isin(selected),
        [
            "patient_id",
            "visit",
            "orientation_resolved_before",
            "orientation_after",
            "canonical_roundtrip_corner_error_mm",
            "dce_mask_footprint_corner_error_mm",
            "anchor_provenance",
        ],
    ].copy()
    laterality_private = laterality_private.merge(
        dicom_headers,
        on=["patient_id", "visit"],
        how="left",
        validate="one_to_one",
    )
    laterality_private["source_roi_available"] = False
    laterality_private["lesion_side_mapping_available"] = False
    laterality_private["contralateral_side_verified"] = False
    laterality_private["status"] = "RAS_GEOMETRY_PASS_SOURCE_LATERALITY_NOT_AUDITABLE"

    patient_laterality = dicom_headers.groupby("patient_id")["laterality_tag"].agg(
        lambda values: sorted({str(value) for value in values if str(value)})
    )
    laterality_public = pd.DataFrame(
        [
            {
                "audit_scope": "outcome_blind_geometry_sample",
                "sample_patients": len(selected),
                "sample_visits": len(laterality_private),
                "selection_sha256": selection_sha,
                "orientation_after_ras_fraction": float(laterality_private["orientation_after"].eq("RAS").mean()),
                "maximum_roundtrip_corner_error_mm": float(laterality_private["canonical_roundtrip_corner_error_mm"].max()),
                "maximum_dce_mask_footprint_corner_error_mm": float(laterality_private["dce_mask_footprint_corner_error_mm"].max()),
                "population_laterality_tag_present_visits": int(dicom_headers["laterality_tag"].ne("").sum()),
                "population_laterality_tag_missing_visits": int(dicom_headers["laterality_tag"].eq("").sum()),
                "population_image_laterality_tag_present_visits": int(dicom_headers["image_laterality_tag"].ne("").sum()),
                "population_geometry_tags_complete_visits": int(dicom_headers[["iop_present", "ipp_present", "pixel_spacing_present", "rows_present", "columns_present", "frame_of_reference_uid_present"]].all(axis=1).sum()),
                "population_patients_with_conflicting_known_laterality_tag": int(patient_laterality.map(len).gt(1).sum()),
                "source_roi_available": False,
                "lesion_to_contralateral_mapping_available": False,
                "left_right_source_mapping_conclusion": "NOT_AUDITABLE_SOURCE_ROI_AND_HASH_BOUND_LATERALITY_NOT_AVAILABLE",
                "risk": "DICOM Laterality is incomplete and is not hash-bound to lesion side or historical BPE source selection; RAS conversion alone cannot verify contralateral assignment",
                "status": "GEOMETRY_PASS_SOURCE_LATERALITY_UNRESOLVED",
            }
        ]
    )

    coverage = pd.DataFrame(
        [
            {
                "fov_contract": fov,
                "support_definition": config["fov_contracts"][fov]["name"],
                "eligible_patients": len(patient_ids),
                "eligible_visits": len(geometry),
                "source_roi_available_visits": 0,
                "source_occupancy_evaluable_visits": 0,
                "source_occupancy_ge_0_99_visits": pd.NA,
                "source_occupancy_ge_0_99_fraction": pd.NA,
                "zero_source_overlap_visits": pd.NA,
                "contralateral_breast_coverage": pd.NA,
                "observability_gate": "NOT_EVALUABLE",
                "status": NOT_EVALUABLE,
            }
            for fov in ("F0", "F1", "F2")
        ]
    )
    boundary = pd.DataFrame(
        [
            {
                "fov_contract": fov,
                "eligible_visits": len(geometry),
                "evaluable_visits": 0,
                "touch_x_visits": pd.NA,
                "touch_y_visits": pd.NA,
                "touch_z_visits": pd.NA,
                "touch_any_visits": pd.NA,
                "no_boundary_touch_fraction": pd.NA,
                "status": NOT_EVALUABLE,
            }
            for fov in ("F0", "F1", "F2")
        ]
    )
    margin = pd.DataFrame(
        [
            {
                "fov_contract": fov,
                "metric": "source_roi_to_nearest_fov_boundary_mm",
                "n": 0,
                "minimum_mm": pd.NA,
                "q05_mm": pd.NA,
                "q50_mm": pd.NA,
                "q95_mm": pd.NA,
                "maximum_mm": pd.NA,
                "status": NOT_EVALUABLE,
            }
            for fov in ("F0", "F1", "F2")
        ]
    )
    longitudinal = pd.DataFrame(
        [
            {
                "interval": interval,
                "eligible_patients": len(patient_ids),
                "evaluable_patients": 0,
                "centroid_displacement_q50_mm": pd.NA,
                "centroid_displacement_q95_mm": pd.NA,
                "source_volume_ratio_q50": pd.NA,
                "source_volume_variation_q95_percent": pd.NA,
                "laterality_consistent_patients": pd.NA,
                "roi_temporal_anchor": "NOT_DOCUMENTED",
                "status": NOT_EVALUABLE,
            }
            for interval in INTERVALS
        ]
    )

    f0_extent = np.asarray(config["fov_contracts"]["F0"]["support_xyz_mm"], dtype=float)
    f1 = config["fov_contracts"]["F1"]
    f1_shape_xyz = np.asarray(f1["shape_zyx"], dtype=np.int64)[::-1]
    f0_voxels_at_c1b_spacing = int(np.prod(np.ceil(f0_extent / np.asarray(f1["spacing_xyz_mm"], dtype=float))))
    f1_voxels = int(np.prod(f1_shape_xyz))
    f1_visit_bytes = int(f1_voxels * int(f1["channels"]) * int(f1["dtype_bytes"]))
    f2_voxel_stats = summarize_numeric(acquisition["source_voxels"])
    cost = pd.DataFrame(
        [
            {
                "contract": "F0_LOCAL_READOUT_SUPPORT",
                "role": "readout_support_not_full_tensor",
                "extent_x_mm": 64.0,
                "extent_y_mm": 64.0,
                "extent_z_mm": 64.0,
                "target_spacing_xyz_mm": "0.9|0.9|2.0_actual_full_C1B_input",
                "tensor_dimensions_zyx": "112|176|160_actual_input;32|72|72_hypothetical_support_only",
                "voxels_per_visit": f1_voxels,
                "channels": 7,
                "float32_memory_mib_per_visit": f1_visit_bytes / (1024**2),
                "relative_voxel_cost_vs_C1B": 1.0,
                "downsampling_factor": 1.0,
                "status": f"READOUT_ONLY_ACTUAL_INPUT_IS_F1;hypothetical_support_voxels={f0_voxels_at_c1b_spacing}",
            },
            {
                "contract": "F1_FULL_C1B_H",
                "role": "existing_model_tensor_support",
                "extent_x_mm": 144.0,
                "extent_y_mm": 158.4,
                "extent_z_mm": 224.0,
                "target_spacing_xyz_mm": "0.9|0.9|2.0",
                "tensor_dimensions_zyx": "112|176|160",
                "voxels_per_visit": f1_voxels,
                "channels": 7,
                "float32_memory_mib_per_visit": f1_visit_bytes / (1024**2),
                "relative_voxel_cost_vs_C1B": 1.0,
                "downsampling_factor": 1.0,
                "status": "EXISTING_FROZEN_CONTRACT",
            },
            {
                "contract": "F2_RECONSTRUCTED_SOURCE_SUPPORT_MARGINAL_Q50",
                "role": "separate_marginal_summaries_not_one_realizable_tensor",
                "extent_x_mm": summarize_numeric(acquisition["source_extent_x_mm"])["q50"],
                "extent_y_mm": summarize_numeric(acquisition["source_extent_y_mm"])["q50"],
                "extent_z_mm": summarize_numeric(acquisition["source_extent_z_mm"])["q50"],
                "target_spacing_xyz_mm": "native_variable",
                "tensor_dimensions_zyx": "not_one_realizable_geometry",
                "voxels_per_visit": f2_voxel_stats["q50"],
                "channels": pd.NA,
                "float32_memory_mib_per_visit": f2_voxel_stats["q50"] * 4 / (1024**2),
                "relative_voxel_cost_vs_C1B": f2_voxel_stats["q50"] / f1_voxels,
                "downsampling_factor": pd.NA,
                "status": "RECONSTRUCTED_SOURCE_SUPPORT_ONLY;RAW_FOOTPRINT_EQUIVALENCE_NOT_RECOMPUTED;NOT_BPE_CONTEXT_CANDIDATE",
            },
            {
                "contract": "CXT1_FULL_C1B_CONTEXT",
                "role": "candidate_only_if_F1_passes",
                "extent_x_mm": pd.NA,
                "extent_y_mm": pd.NA,
                "extent_z_mm": pd.NA,
                "target_spacing_xyz_mm": pd.NA,
                "tensor_dimensions_zyx": pd.NA,
                "voxels_per_visit": pd.NA,
                "channels": pd.NA,
                "float32_memory_mib_per_visit": pd.NA,
                "relative_voxel_cost_vs_C1B": pd.NA,
                "downsampling_factor": pd.NA,
                "status": "NOT_AUTHORIZED_F1_NOT_EVALUABLE",
            },
            {
                "contract": "CXT2_BILATERAL_BREAST_PHYSICAL_CONTEXT",
                "role": "candidate_only_if_F1_fails_and_F2_passes",
                "extent_x_mm": pd.NA,
                "extent_y_mm": pd.NA,
                "extent_z_mm": pd.NA,
                "target_spacing_xyz_mm": pd.NA,
                "tensor_dimensions_zyx": pd.NA,
                "voxels_per_visit": pd.NA,
                "channels": pd.NA,
                "float32_memory_mib_per_visit": pd.NA,
                "relative_voxel_cost_vs_C1B": pd.NA,
                "downsampling_factor": pd.NA,
                "status": "NOT_DEFINED_SOURCE_ROI_AND_BREAST_SUPPORT_NOT_AVAILABLE",
            },
        ]
    )

    population = audit_population_table(config, workbook, mapping, geometry, caches, raw)
    acquisition_summary = aggregate_extent_table(acquisition)
    acquisition_valid_stats = summarize_numeric(acquisition["valid_source_fraction_on_c1b"])
    source_geometry_summary = {
        "reconstructed_source_image_aabb_extent_marginal_xyz_mm": {
            axis: summarize_numeric(acquisition[f"source_extent_{axis}_mm"])
            for axis in "xyz"
        },
        "reconstructed_source_voxels_per_visit_marginal": summarize_numeric(acquisition["source_voxels"]),
        "source_affine_axis_aligned_ras_visits": int(acquisition["source_affine_axis_aligned_ras"].sum()),
        "source_affine_oblique_visits": int((~acquisition["source_affine_axis_aligned_ras"]).sum()),
        "raw_dicom_selected_series_available_visits": int(raw["raw_series_available"].sum()),
        "raw_to_reconstructed_full_series_footprint_equivalence_recomputed": False,
        "c1b_target_grid_valid_source_fraction": acquisition_valid_stats,
        "c1b_zero_valid_source_overlap_visits": int((~acquisition["has_valid_source_overlap"]).sum()),
        "interpretation": "reconstructed source image and C1B sampling support only; marginal AABB summaries are not a raw-DICOM footprint-equivalence audit, BPE ROI coverage, or one realizable tensor geometry",
    }
    orientation_summary = {
        "formal_patients": len(patient_ids),
        "formal_visits": len(orientation),
        "orientation_before": {str(key): int(value) for key, value in orientation["orientation_resolved_before"].value_counts().sort_index().items()},
        "orientation_after": {str(key): int(value) for key, value in orientation["orientation_after"].value_counts().sort_index().items()},
        "maximum_roundtrip_corner_error_mm": float(orientation["canonical_roundtrip_corner_error_mm"].max()),
        "maximum_dce_mask_footprint_corner_error_mm": float(orientation["dce_mask_footprint_corner_error_mm"].max()),
        "array_reordering_implemented_and_unit_tested": bool(orientation_gate["array_reordering_implemented_and_unit_tested"]),
        "left_right_consistent_upstream": bool(orientation_gate["left_right_consistent"]),
        "source_laterality_mapping_status": "NOT_AUDITABLE",
        "status": "RAS_GEOMETRY_PASS_BPE_LATERALITY_UNRESOLVED",
    }

    decision = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "outcome_blind": True,
        "outcome_fields_read": [],
        "clinical_treatment_fields_read": [],
        "bpe_values_used_for_selection_or_geometry": False,
        "source_roi_status": SOURCE_STATUS,
        "source_roi_available": False,
        "population": {
            "source_workbook_patients": len(workbook),
            "bpe_complete_workbook_patients": int(workbook["bpe_complete"].sum()),
            "matched_primary_patients": len(patient_ids),
            "matched_primary_visits": len(geometry),
            "workbook_only_excluded_patients": int((~workbook["trial_id"].isin(set(mapping["trial_id"]))).sum()),
            "reconstructed_dce_available_visits": int(geometry["resolved_dce_nifti"].map(lambda value: Path(value).is_file()).sum()),
            "c1b_cache_available_patients": int(caches["cache_available"].sum()),
            "raw_dicom_selected_series_available": int(raw["raw_series_available"].sum()),
        },
        "fov_gates": {
            "F0_LOCAL": {"status": "NOT_EVALUABLE", "reason": SOURCE_STATUS},
            "F1_FULL_C1B_H": {"status": "NOT_EVALUABLE", "reason": SOURCE_STATUS},
            "F2_ACQUISITION": {"status": "NOT_EVALUABLE", "reason": SOURCE_STATUS},
        },
        "coverage": {
            "F0_LOCAL": None,
            "F1_FULL_C1B_H": None,
            "F2_ACQUISITION": None,
        },
        "boundary_touch": None,
        "physical_margin": None,
        "contralateral_breast_coverage": None,
        "longitudinal_source_geometry": None,
        "classification_code": code,
        "classification": classification,
        "next_stage_input": "PAUSE_BPE",
        "context_candidate": None,
        "multiscale_context_recommended": False,
        "f2_geometry_evidence": {
            "reconstructed_source_affine_available_visits": len(acquisition),
            "selected_raw_dicom_series_available_visits": int(raw["raw_series_available"].sum()),
            "raw_to_reconstructed_full_series_footprint_equivalence_recomputed": False,
            "extent_statistics_are_separate_marginals": True,
        },
        "local_context_phenotype_representation_pilot_authorized": False,
        "next_action": "recover_authoritative_source_ROI_and_hash_bound_lesion_to_contralateral_mapping_then_rerun_same_geometry_gates",
        "prohibitions": [
            "do_not_call_LOCAL_or_C1B_disjoint",
            "do_not_call_representation_failure",
            "do_not_infer_acquisition_completeness_from_image_support",
            "do_not_create_proxy_FGT_ROI_and_label_it_as_source",
            "do_not_design_BPE_architecture_from_current_run",
        ],
        "provenance": {
            **config["provenance"],
            "source_verification": source_verification,
            "configuration_sha256": sha256_file(args.config),
        },
        "delivery": config["delivery"],
    }
    decision["decision_sha256"] = canonical_json_sha256(decision)

    private_columns = [
        "patient_id",
        "visit",
        "source_shape_x",
        "source_shape_y",
        "source_shape_z",
        "source_low_x_ras_mm",
        "source_low_y_ras_mm",
        "source_low_z_ras_mm",
        "source_high_x_ras_mm",
        "source_high_y_ras_mm",
        "source_high_z_ras_mm",
        "source_extent_x_mm",
        "source_extent_y_mm",
        "source_extent_z_mm",
        "source_voxels",
        "source_affine_axis_aligned_ras",
        "valid_source_fraction_on_c1b",
        "has_valid_source_overlap",
    ]
    write_csv(ROOT / "manifests" / "acquisition_support.private.csv", acquisition[private_columns])
    write_csv(ROOT / "manifests" / "raw_series_availability.private.csv", raw)
    write_csv(ROOT / "manifests" / "laterality_sample.private.csv", laterality_private)
    write_csv(ROOT / "manifests" / "audit_population.csv", population)
    write_csv(ROOT / "metrics" / "coverage_table.csv", coverage)
    write_csv(ROOT / "metrics" / "boundary_touch_table.csv", boundary)
    write_csv(ROOT / "metrics" / "physical_margin_distribution.csv", margin)
    write_csv(ROOT / "metrics" / "laterality_audit.csv", laterality_public)
    write_csv(ROOT / "metrics" / "longitudinal_geometry.csv", longitudinal)
    write_csv(ROOT / "metrics" / "context_candidate_cost_table.csv", cost)
    write_csv(ROOT / "metrics" / "acquisition_image_support_extent.csv", acquisition_summary)
    write_json(ROOT / "metrics" / "source_geometry_summary.json", source_geometry_summary)
    write_json(ROOT / "metrics" / "c1b_grid_validation.json", grid_summary)
    write_json(ROOT / "metrics" / "orientation_validation.json", orientation_summary)
    write_json(ROOT / "manifests" / "input_provenance.json", source_verification)
    write_json(ROOT / "metrics" / "decision.json", decision)
    print(json.dumps({"status": "COMPLETE", "classification": classification, "patients": len(patient_ids), "visits": len(geometry)}, sort_keys=True))


if __name__ == "__main__":
    main()
