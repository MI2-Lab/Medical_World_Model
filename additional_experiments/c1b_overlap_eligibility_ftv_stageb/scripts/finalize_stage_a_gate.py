#!/usr/bin/env python3
"""Finalize the preregistered 15-item Stage-A gate.

This entry point is deliberately fail closed.  It derives the population from
the current private technical-eligibility manifests, intersects inherited
formal QC evidence with that population, and binds the complete cache evidence
by SHA-256.  Only all 15 PASS items can create ``STAGE_A_GO.json``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
PRIOR_RELATIVE = Path("additional_experiments/c1b_model_ready_ftv_sanity")
AUDIT_RELATIVE = Path("additional_experiments/zero_overlap_provenance_audit")
VISITS = ("T0", "T1", "T2", "T3")

SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = EXPERIMENT_ROOT / "src"
PRIOR_SRC_ROOT = REPO_ROOT / PRIOR_RELATIVE / "src"
for source in (SCRIPT_ROOT, SRC_ROOT, PRIOR_SRC_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from audit_public_artifacts import scan_public_artifacts  # noqa: E402
from c1b_overlap_stageb.io import (  # noqa: E402
    atomic_text,
    json_text,
    sha256_file,
    verify_preregistration,
    verify_upstream_contract,
)
from c1b_overlap_stageb.eligibility import (  # noqa: E402
    frozen_grid_contract_sha256,
    geometry_contract_sha256,
)
from c1b_sanity.geometry import make_c1b_grid  # noqa: E402


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def is_sha256(value: Any) -> bool:
    return bool(SHA256_PATTERN.fullmatch(str(value).strip().lower()))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("required JSON is not an object")
    return payload


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float = -1.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return output if np.isfinite(output) else default


def _hash_matches(mapping: Any, name: str, path: Path) -> bool:
    return bool(
        isinstance(mapping, dict)
        and is_sha256(mapping.get(name))
        and str(mapping[name]).lower() == sha256_file(path)
    )


def _add_provenance(
    output: dict[str, str], path: Path, *, experiment_root: Path, repo_root: Path
) -> None:
    if not path.is_file():
        return
    try:
        label = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        try:
            label = path.resolve().relative_to(experiment_root.resolve()).as_posix()
        except ValueError:
            label = f"runtime_code/{path.name}"
    output[label] = sha256_file(path)


def _stage_b_artifacts(experiment_root: Path) -> int:
    paths: set[Path] = set()
    for name in ("checkpoints", "features", "predictions"):
        base = experiment_root / name
        if base.is_dir():
            paths.update(
                path
                for path in base.rglob("*")
                if path.is_file() and path.name != ".gitkeep"
            )
    metrics = experiment_root / "metrics"
    if metrics.is_dir():
        for path in metrics.rglob("*"):
            relative = path.relative_to(metrics).as_posix().lower()
            if path.is_file() and (
                relative.startswith(("stage_b/", "training/", "final/"))
                or Path(relative).name.startswith(("table2_", "table3_", "table4_", "table5_"))
            ):
                paths.add(path)
    figures = experiment_root / "figures"
    if figures.is_dir():
        paths.update(
            path
            for path in figures.iterdir()
            if path.is_file() and re.match(r"(?:0[4-9]|1[0-2])_", path.name)
        )
    return len(paths)


def _empty_eligibility() -> dict[str, Any]:
    return {
        "loaded": False,
        "frozen": False,
        "outcome_free": False,
        "programmatic": False,
        "positive_overlap": False,
        "candidate_patients": 0,
        "eligible_patients": 0,
        "excluded_patients": 0,
        "candidate_visits": 0,
        "valid_visits": 0,
        "zero_overlap_visits": 0,
        "eligible_visits": 0,
        "eligible_ids": set(),
        "eligible_visit_counts": {},
        "patient_manifest_sha256": None,
        "visit_manifest_sha256": None,
        "grid_manifest_sha256": None,
        "grid_geometry_closure": False,
        "error_code": "ELIGIBILITY_EVIDENCE_MISSING_OR_INVALID",
    }


def _verify_grid_geometry_closure(
    *,
    grids: pd.DataFrame,
    visits: pd.DataFrame,
    grid_summary: dict[str, Any],
    summary: dict[str, Any],
    grid_path: Path,
    preregistration: dict[str, Any] | None,
) -> bool:
    """Recompute every frozen-grid and source/grid geometry digest."""

    required = {
        "patient_id",
        "cohort",
        "grid_shape_zyx_json",
        "grid_spacing_xyz_mm_json",
        "grid_affine_ras_json",
        "grid_contract_sha256",
        "grid_center_x_ras_mm",
        "grid_center_y_ras_mm",
        "grid_center_z_ras_mm",
    }
    if not required.issubset(grids) or grids["patient_id"].astype(str).duplicated().any():
        return False
    visit_ids = set(visits["patient_id"].astype(str))
    if set(grids["patient_id"].astype(str)) != visit_ids:
        return False
    visit_cohorts = (
        visits.groupby(visits["patient_id"].astype(str))["cohort"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    contracts: dict[str, tuple[str, np.ndarray, tuple[int, ...]]] = {}
    for row in grids.to_dict("records"):
        patient_id = str(row["patient_id"])
        try:
            shape_zyx = tuple(
                int(value) for value in json.loads(str(row["grid_shape_zyx_json"]))
            )
            spacing_xyz = tuple(
                float(value)
                for value in json.loads(str(row["grid_spacing_xyz_mm_json"]))
            )
            affine = np.asarray(
                json.loads(str(row["grid_affine_ras_json"])), dtype=np.float64
            )
            rebuilt = make_c1b_grid(
                (
                    float(row["grid_center_x_ras_mm"]),
                    float(row["grid_center_y_ras_mm"]),
                    float(row["grid_center_z_ras_mm"]),
                )
            )
            digest = frozen_grid_contract_sha256(
                patient_id=patient_id,
                cohort=str(row["cohort"]),
                grid_shape_zyx=shape_zyx,
                grid_spacing_xyz_mm=spacing_xyz,
                grid_affine_ras=affine,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            visit_cohorts.get(patient_id) != {str(row["cohort"])}
            or
            shape_zyx != (112, 176, 160)
            or spacing_xyz != (0.9, 0.9, 2.0)
            or affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or shape_zyx != rebuilt.shape_zyx
            or spacing_xyz != rebuilt.spacing_xyz_mm
            or not np.allclose(affine, rebuilt.affine_ras, atol=1e-12, rtol=0.0)
            or digest != str(row["grid_contract_sha256"])
        ):
            return False
        contracts[patient_id] = (digest, rebuilt.affine_ras, shape_zyx)
    for row in visits.to_dict("records"):
        try:
            digest, grid_affine, shape_zyx = contracts[str(row["patient_id"])]
            source_shape = tuple(
                int(value) for value in json.loads(str(row["source_shape_xyz_json"]))
            )
            source_affine = np.asarray(
                json.loads(str(row["source_affine_ras_json"])), dtype=np.float64
            )
            geometry_digest = geometry_contract_sha256(
                source_shape_xyz=source_shape,
                source_affine_ras=source_affine,
                grid_shape_xyz=tuple(reversed(shape_zyx)),
                grid_affine_ras=grid_affine,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            str(row["grid_contract_sha256"]) != digest
            or str(row["geometry_contract_sha256"]) != geometry_digest
            or int(row["target_grid_voxels"])
            != int(np.prod(shape_zyx, dtype=np.int64))
        ):
            return False
    grid_hash = sha256_file(grid_path)
    return bool(
        str(grid_summary.get("status")) == "PASS"
        and _safe_int(grid_summary.get("candidate_grid_rows")) == len(grids)
        and str(grid_summary.get("private_grid_manifest_sha256", "")).lower()
        == grid_hash
        and _hash_matches(
            summary.get("private_manifest_sha256", {}), grid_path.name, grid_path
        )
        and preregistration is not None
        and str(grid_summary.get("preregistration_plan_sha256"))
        == str(preregistration.get("plan_sha256"))
        and not as_bool(grid_summary.get("computes_eligibility", True))
        and not as_bool(grid_summary.get("evaluates_followup_overlap", True))
    )


def _collect_eligibility(
    *,
    experiment_root: Path,
    repo_root: Path,
    preregistration: dict[str, Any] | None,
    provenance: dict[str, str],
) -> dict[str, Any]:
    result = _empty_eligibility()
    summary_path = experiment_root / "metrics/technical_eligibility_summary.json"
    patient_path = experiment_root / "manifests/technical_eligibility_patients.private.csv"
    visit_path = experiment_root / "manifests/technical_eligibility_visits.private.csv"
    inventory_path = experiment_root / "manifests/eligible_model_input_inventory.private.csv"
    grid_path = experiment_root / "manifests/frozen_c1b_grids.private.csv"
    grid_summary_path = experiment_root / "metrics/frozen_c1b_grid_materialization.json"
    required_paths = (
        summary_path,
        patient_path,
        visit_path,
        inventory_path,
        grid_path,
        grid_summary_path,
    )
    try:
        if not all(path.is_file() for path in required_paths):
            return result
        summary = _read_json(summary_path)
        patients = pd.read_csv(patient_path)
        visits = pd.read_csv(visit_path)
        eligible_inventory = pd.read_csv(inventory_path)
        grids = pd.read_csv(grid_path)
        grid_summary = _read_json(grid_summary_path)
        patient_required = {
            "patient_id",
            "cohort",
            "candidate_visit_count",
            "valid_visit_count",
            "zero_overlap_visit_count",
            "minimum_valid_source_voxels",
            "eligible",
            "exclusion_reason",
        }
        visit_required = {
            "patient_id",
            "cohort",
            "visit",
            "valid_source_voxels",
            "target_grid_voxels",
            "has_valid_source_overlap",
            "eligibility_evidence_scope",
            "source_shape_xyz_json",
            "source_affine_ras_json",
            "grid_contract_sha256",
            "geometry_contract_sha256",
        }
        if not patient_required.issubset(patients) or not visit_required.issubset(visits):
            return result
        if not visit_required.issubset(eligible_inventory):
            return result

        patients = patients.copy()
        visits = visits.copy()
        eligible_inventory = eligible_inventory.copy()
        for frame in (patients, visits, eligible_inventory):
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame["cohort"] = frame["cohort"].astype(str)
        visits["visit"] = visits["visit"].astype(str)
        eligible_inventory["visit"] = eligible_inventory["visit"].astype(str)
        if (
            patients["patient_id"].duplicated().any()
            or visits.duplicated(["patient_id", "visit"]).any()
            or eligible_inventory.duplicated(["patient_id", "visit"]).any()
            or set(visits["visit"]) != set(VISITS)
        ):
            return result

        visits["_valid"] = pd.to_numeric(
            visits["valid_source_voxels"], errors="raise"
        ).astype(np.int64)
        visits["_target"] = pd.to_numeric(
            visits["target_grid_voxels"], errors="raise"
        ).astype(np.int64)
        if (visits["_valid"] < 0).any() or (visits["_target"] < 1).any():
            return result
        visits["_positive"] = visits["_valid"].gt(0)
        if not (
            visits["has_valid_source_overlap"].map(as_bool) == visits["_positive"]
        ).all():
            return result

        derived = (
            visits.groupby("patient_id", sort=True)
            .agg(
                cohort=("cohort", "first"),
                cohort_count=("cohort", "nunique"),
                candidate_visit_count=("visit", "size"),
                unique_visit_count=("visit", "nunique"),
                valid_visit_count=("_positive", "sum"),
                zero_overlap_visit_count=("_positive", lambda values: int((~values).sum())),
                minimum_valid_source_voxels=("_valid", "min"),
            )
            .reset_index()
        )
        derived["eligible"] = derived["zero_overlap_visit_count"].eq(0)
        structure_ok = bool(
            derived["cohort_count"].eq(1).all()
            and derived["candidate_visit_count"].eq(len(VISITS)).all()
            and derived["unique_visit_count"].eq(len(VISITS)).all()
            and len(derived) == len(patients)
            and set(derived["patient_id"]) == set(patients["patient_id"])
        )
        comparison = patients.merge(
            derived,
            on="patient_id",
            how="outer",
            validate="one_to_one",
            suffixes=("_recorded", "_derived"),
            indicator=True,
        )
        recorded_matches = bool(
            structure_ok
            and comparison["_merge"].eq("both").all()
            and comparison["cohort_recorded"].astype(str).eq(
                comparison["cohort_derived"].astype(str)
            ).all()
            and comparison["candidate_visit_count_recorded"].astype(int).eq(
                comparison["candidate_visit_count_derived"].astype(int)
            ).all()
            and comparison["valid_visit_count_recorded"].astype(int).eq(
                comparison["valid_visit_count_derived"].astype(int)
            ).all()
            and comparison["zero_overlap_visit_count_recorded"].astype(int).eq(
                comparison["zero_overlap_visit_count_derived"].astype(int)
            ).all()
            and comparison["minimum_valid_source_voxels_recorded"].astype(int).eq(
                comparison["minimum_valid_source_voxels_derived"].astype(int)
            ).all()
            and comparison["eligible_recorded"].map(as_bool).eq(
                comparison["eligible_derived"].map(as_bool)
            ).all()
        )

        eligible_ids = set(
            derived.loc[derived["eligible"], "patient_id"].astype(str)
        )
        eligible_keys = set(
            zip(
                visits.loc[visits["patient_id"].isin(eligible_ids), "patient_id"],
                visits.loc[visits["patient_id"].isin(eligible_ids), "visit"],
                strict=True,
            )
        )
        inventory_keys = set(
            zip(
                eligible_inventory["patient_id"],
                eligible_inventory["visit"],
                strict=True,
            )
        )
        inventory_ok = bool(
            inventory_keys == eligible_keys
            and eligible_inventory["patient_id"].isin(eligible_ids).all()
            and eligible_inventory["has_valid_source_overlap"].map(as_bool).all()
            and pd.to_numeric(
                eligible_inventory["valid_source_voxels"], errors="raise"
            ).gt(0).all()
        )
        if inventory_ok:
            compare_columns = [
                "patient_id",
                "cohort",
                "visit",
                "valid_source_voxels",
                "target_grid_voxels",
                "grid_contract_sha256",
                "geometry_contract_sha256",
            ]
            expected_inventory = visits.loc[
                visits["patient_id"].isin(eligible_ids), compare_columns
            ].copy()
            observed_inventory = eligible_inventory[compare_columns].copy()
            expected_inventory = expected_inventory.sort_values(
                ["patient_id", "visit"], kind="stable"
            ).reset_index(drop=True)
            observed_inventory = observed_inventory.sort_values(
                ["patient_id", "visit"], kind="stable"
            ).reset_index(drop=True)
            inventory_ok = bool(expected_inventory.equals(observed_inventory))
        grid_geometry_closure = _verify_grid_geometry_closure(
            grids=grids,
            visits=visits,
            grid_summary=grid_summary,
            summary=summary,
            grid_path=grid_path,
            preregistration=preregistration,
        )

        private_hashes = summary.get("private_manifest_sha256", {})
        private_hash_ok = bool(
            _hash_matches(private_hashes, patient_path.name, patient_path)
            and _hash_matches(private_hashes, visit_path.name, visit_path)
            and _hash_matches(private_hashes, inventory_path.name, inventory_path)
        )
        source_paths = {
            "candidate_inventory": repo_root
            / PRIOR_RELATIVE
            / "manifests/model_input_inventory.private.csv",
            "candidate_key_manifest": repo_root
            / PRIOR_RELATIVE
            / "metrics/orientation_resampling_patient_visit.private.csv",
            "ispy1_visit_source_eligibility": repo_root
            / PRIOR_RELATIVE
            / "manifests/ispy1_base_eligibility_visits.private.csv",
        }
        source_hashes = summary.get("source_provenance_sha256", {})
        source_hash_ok = bool(
            isinstance(source_hashes, dict)
            and all(
                path.is_file()
                and str(source_hashes.get(name, "")).lower() == sha256_file(path)
                for name, path in source_paths.items()
            )
        )
        candidate_patients = int(len(derived))
        candidate_visits = int(len(visits))
        eligible_patients = int(len(eligible_ids))
        valid_visits = int(visits["_positive"].sum())
        zero_visits = candidate_visits - valid_visits
        summary_counts_ok = bool(
            _safe_int(summary.get("candidate_patients")) == candidate_patients
            and _safe_int(summary.get("candidate_visits")) == candidate_visits
            and _safe_int(summary.get("eligible_patients")) == eligible_patients
            and _safe_int(summary.get("excluded_patients"))
            == candidate_patients - eligible_patients
            and _safe_int(summary.get("valid_visits")) == valid_visits
            and _safe_int(summary.get("zero_overlap_visits")) == zero_visits
        )
        allowed = {
            "imaging_source",
            "raw_or_rebuilt_source_geometry",
            "frozen_c1b_h_physical_grid",
            "valid_source_overlap",
        }
        forbidden_lists_empty = all(
            summary.get(name) == []
            for name in (
                "clinical_treatment_subtype_fields_read",
                "lesion_ftv_ld_sph_bpe_fields_read",
                "model_loss_representation_performance_fields_read",
                "outcome_pcr_fields_read",
            )
        )
        forbidden_column = re.compile(
            r"(?:ftv|\bld\b|sph|bpe|pcr|outcome|clinical|treatment|subtype|"
            r"model_loss|representation|performance)",
            re.IGNORECASE,
        )
        columns_safe = not any(
            forbidden_column.search(str(column))
            for column in (*patients.columns, *visits.columns)
        )
        outcome_free = bool(
            forbidden_lists_empty
            and columns_safe
            and set(summary.get("eligibility_input_allowlist", [])) == allowed
            and not as_bool(summary.get("contains_patient_identifiers", True))
        )
        programmatic = bool(
            recorded_matches
            and inventory_ok
            and summary_counts_ok
            and private_hash_ok
            and source_hash_ok
            and grid_geometry_closure
            and not as_bool(summary.get("hardcoded_population_result", True))
            and summary.get("patient_specific_rules") == []
            and as_bool(summary.get("run_is_new_and_does_not_amend_prior_no_go"))
            and str(summary.get("status")) == "PASS"
        )
        positive_overlap = bool(
            programmatic
            and visits.loc[visits["patient_id"].isin(eligible_ids), "_positive"].all()
            and len(eligible_keys) == len(eligible_ids) * len(VISITS)
        )
        plan_hash = None if preregistration is None else preregistration.get("plan_sha256")
        frozen = bool(
            preregistration is not None
            and as_bool(summary.get("preregistered_before_eligibility_results"))
            and str(summary.get("preregistration_plan_sha256")) == str(plan_hash)
        )
        visit_counts: dict[str, list[int]] = {}
        order = {visit: index for index, visit in enumerate(VISITS)}
        eligible_visit_frame = visits.loc[visits["patient_id"].isin(eligible_ids)].copy()
        eligible_visit_frame["_order"] = eligible_visit_frame["visit"].map(order)
        for patient_id, group in eligible_visit_frame.groupby("patient_id", sort=True):
            group = group.sort_values("_order")
            visit_counts[str(patient_id)] = [int(value) for value in group["_valid"]]

        for path in (*required_paths, *source_paths.values()):
            _add_provenance(
                provenance, path, experiment_root=experiment_root, repo_root=repo_root
            )
        result.update(
            {
                "loaded": True,
                "frozen": frozen,
                "outcome_free": outcome_free,
                "programmatic": programmatic,
                "positive_overlap": positive_overlap,
                "candidate_patients": candidate_patients,
                "eligible_patients": eligible_patients,
                "excluded_patients": candidate_patients - eligible_patients,
                "candidate_visits": candidate_visits,
                "valid_visits": valid_visits,
                "zero_overlap_visits": zero_visits,
                "eligible_visits": len(eligible_keys),
                "eligible_ids": eligible_ids,
                "eligible_visit_counts": visit_counts,
                "patient_manifest_sha256": sha256_file(patient_path),
                "visit_manifest_sha256": sha256_file(visit_path),
                "grid_manifest_sha256": sha256_file(grid_path),
                "grid_geometry_closure": grid_geometry_closure,
                "error_code": None,
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _empty_eligibility()
    return result


def _empty_formal() -> dict[str, Any]:
    return {
        "loaded": False,
        "structure_ok": False,
        "formal_patients_before_intersection": 0,
        "formal_patients_after_intersection": 0,
        "formal_patients_excluded": 0,
        "formal_visits_after_intersection": 0,
        "containment_rate": None,
        "retention_q05": None,
        "containment_pass": False,
        "retention_pass": False,
        "grounding_pass": False,
        "grounding_observable_visits": 0,
        "grounding_ineligible_visits": 0,
        "error_code": "FORMAL_EVIDENCE_MISSING_OR_INVALID",
    }


def _collect_formal(
    *,
    experiment_root: Path,
    repo_root: Path,
    eligibility: dict[str, Any],
    upstream_ok: bool,
    config: dict[str, Any] | None,
    provenance: dict[str, str],
) -> dict[str, Any]:
    result = _empty_formal()
    prior = repo_root / PRIOR_RELATIVE
    support_path = prior / "metrics/support_containment_patient_visit.private.csv"
    support_summary_path = prior / "metrics/support_containment_h_summary.json"
    grounding_path = prior / "manifests/grounding_observability_manifest.private.csv"
    grounding_summary_path = prior / "manifests/grounding_observability_summary.json"
    try:
        if not upstream_ok or not eligibility["loaded"] or config is None:
            return result
        if not all(
            path.is_file()
            for path in (
                support_path,
                support_summary_path,
                grounding_path,
                grounding_summary_path,
            )
        ):
            return result
        support = pd.read_csv(
            support_path,
            usecols=[
                "patient_id",
                "visit",
                "strategy",
                "physical_volume_retention",
                "exact_full_support_containment",
                "source_boundary_touch",
            ],
        )
        support_summary = _read_json(support_summary_path)
        grounding = pd.read_csv(
            grounding_path,
            usecols=[
                "patient_id",
                "visit",
                "source_boundary_touch",
                "ftv_measurement_valid",
                "grounding_observable_mask",
            ],
        )
        grounding_summary = _read_json(grounding_summary_path)
        for frame in (support, grounding):
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame["visit"] = frame["visit"].astype(str)
        retention = pd.to_numeric(support["physical_volume_retention"], errors="raise")
        support["_retention"] = retention
        support["_containment"] = support["exact_full_support_containment"].map(as_bool)
        support_before_ids = set(support["patient_id"])
        original_structure = bool(
            not support.duplicated(["patient_id", "visit"]).any()
            and not grounding.duplicated(["patient_id", "visit"]).any()
            and support["strategy"].astype(str).eq("C1B-H").all()
            and support.groupby("patient_id")["visit"].nunique().eq(len(VISITS)).all()
            and support["visit"].isin(VISITS).all()
            and grounding["visit"].isin(VISITS).all()
            and np.isfinite(retention).all()
            and retention.between(0.0, 1.0).all()
            and set(zip(support["patient_id"], support["visit"], strict=True))
            == set(zip(grounding["patient_id"], grounding["visit"], strict=True))
            and _safe_int(support_summary.get("formal_patients"))
            == len(support_before_ids)
            and _safe_int(support_summary.get("formal_visits")) == len(support)
            and str(support_summary.get("strategy")) == "C1B-H"
            and abs(
                _safe_float(support_summary.get("exact_full_support_containment_rate"))
                - float(support["_containment"].mean())
            )
            <= 1e-12
            and abs(
                _safe_float(
                    support_summary.get("physical_volume_retention", {}).get("q05")
                )
                - float(retention.quantile(0.05, interpolation="linear"))
            )
            <= 1e-12
            and _hash_matches(
                {"manifest": grounding_summary.get("private_manifest_sha256")},
                "manifest",
                grounding_path,
            )
            and _safe_int(grounding_summary.get("formal_patients"))
            == grounding["patient_id"].nunique()
            and _safe_int(grounding_summary.get("formal_visits")) == len(grounding)
        )
        eligible_ids = eligibility["eligible_ids"]
        support_selected = support.loc[support["patient_id"].isin(eligible_ids)].copy()
        grounding_selected = grounding.loc[
            grounding["patient_id"].isin(eligible_ids)
        ].copy()
        selected_ids = set(support_selected["patient_id"])
        selected_keys = set(
            zip(support_selected["patient_id"], support_selected["visit"], strict=True)
        )
        grounding_keys = set(
            zip(grounding_selected["patient_id"], grounding_selected["visit"], strict=True)
        )
        selected_structure = bool(
            original_structure
            and selected_keys == grounding_keys
            and len(support_selected) == len(selected_ids) * len(VISITS)
            and support_selected.groupby("patient_id")["visit"].nunique().eq(
                len(VISITS)
            ).all()
        )
        containment_rate = (
            float(support_selected["_containment"].mean())
            if len(support_selected)
            else -1.0
        )
        retention_q05 = (
            float(
                support_selected["_retention"].quantile(
                    0.05, interpolation="linear"
                )
            )
            if len(support_selected)
            else -1.0
        )
        grounding_mask = grounding_selected["grounding_observable_mask"].map(as_bool)
        source_touch = grounding_selected["source_boundary_touch"].map(as_bool)
        measurement_valid = grounding_selected["ftv_measurement_valid"].map(as_bool)
        grounding_pass = bool(
            selected_structure
            and str(grounding_summary.get("scope"))
            == "grounding_loss_eligibility_only"
            and not as_bool(grounding_summary.get("is_model_input", True))
            and as_bool(grounding_summary.get("does_not_filter_base_training"))
            and (grounding_mask == (measurement_valid & ~source_touch)).all()
        )
        containment_threshold = _safe_float(config.get("formal_containment_minimum"))
        retention_threshold = _safe_float(config.get("ftv_retention_q05_minimum"))
        for path in (
            support_path,
            support_summary_path,
            grounding_path,
            grounding_summary_path,
        ):
            _add_provenance(
                provenance, path, experiment_root=experiment_root, repo_root=repo_root
            )
        result.update(
            {
                "loaded": True,
                "structure_ok": selected_structure,
                "formal_patients_before_intersection": len(support_before_ids),
                "formal_patients_after_intersection": len(selected_ids),
                "formal_patients_excluded": len(support_before_ids) - len(selected_ids),
                "formal_visits_after_intersection": len(support_selected),
                "containment_rate": containment_rate,
                "retention_q05": retention_q05,
                "containment_pass": bool(
                    selected_structure
                    and containment_threshold == 0.95
                    and containment_rate >= containment_threshold
                ),
                "retention_pass": bool(
                    selected_structure
                    and retention_threshold == 0.95
                    and retention_q05 >= retention_threshold
                ),
                "grounding_pass": grounding_pass,
                "grounding_observable_visits": int(grounding_mask.sum()),
                "grounding_ineligible_visits": int((~grounding_mask).sum()),
                "error_code": None,
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _empty_formal()
    return result


def _empty_cache() -> dict[str, Any]:
    return {
        "loaded": False,
        "schema": None,
        "identity_complete": False,
        "content_complete": False,
        "roundtrip_hash_pass": False,
        "manual_corrections_absent": False,
        "catastrophic_cases": None,
        "geometry_excluded": False,
        "eligible_patients": 0,
        "eligible_visits": 0,
        "completed_patients": 0,
        "completed_visits": 0,
        "completion_fraction": 0.0,
        "exact_overlap_fraction": 0.0,
        "finite_fraction": 0.0,
        "nonconstant_fraction": 0.0,
        "shape_fraction": 0.0,
        "orientation_fraction": 0.0,
        "phase_fraction": 0.0,
        "grid_fraction": 0.0,
        "provenance_fraction": 0.0,
        "roundtrip_fraction": 0.0,
        "loader_fraction": 0.0,
        "private_metrics_sha256": None,
        "cache_inventory_sha256": None,
        "live_cache_file_hash_fraction": 0.0,
        "cache_files_independent": False,
        "private_permissions_ok": False,
        "error_code": "CACHE_EVIDENCE_MISSING_OR_INVALID",
    }


def _mean_bool(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or len(frame) == 0:
        return -1.0
    return float(frame[column].map(as_bool).mean())


def _resolve_cache_path(value: Any, experiment_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else experiment_root / path


def _live_cache_hash_check(item: tuple[Path, str]) -> bool:
    """Hash one live file and reject mutation during the read."""

    path, expected = item
    if not path.is_file():
        return False
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    )
    return bool(stable and digest == expected.lower())


def _owner_only(path: Path) -> bool:
    return bool(path.exists() and stat.S_IMODE(path.stat().st_mode) & 0o077 == 0)


def _cache_public_schema(payload: dict[str, Any]) -> str | None:
    """Recognize only the schema emitted by the formal cache runner."""

    formal_v1 = {
        "stage",
        "eligible_patients",
        "eligible_visits",
        "completed_cache_patients",
        "completed_cache_visits",
        "cache_completion_fraction",
        "private_artifact_sha256",
        "cache_roundtrip_pass_fraction",
        "exact_valid_source_count_match_fraction",
        "frozen_grid_center_match_fraction",
        "finite_fraction",
        "whole_visit_nonconstant_fraction",
        "phase_indices_in_range_fraction",
        "model_loader_only_dce7_fraction",
        "geometry_metadata_is_model_input",
        "base_only_later_visit_supports_loaded",
    }
    if formal_v1.issubset(payload):
        return "formal_runner_v1"
    return None


def _collect_cache(
    *,
    experiment_root: Path,
    repo_root: Path,
    eligibility: dict[str, Any],
    upstream_semantic_hash: str | None,
    provenance: dict[str, str],
) -> dict[str, Any]:
    result = _empty_cache()
    gate_path = experiment_root / "metrics/model_input_pipeline_h_all_gate.json"
    private_path = experiment_root / "metrics/model_input_pipeline_h_all.private.csv"
    inventory_path = experiment_root / "manifests/model_input_cache_inventory.private.csv"
    try:
        if not eligibility["loaded"] or not all(
            path.is_file() for path in (gate_path, private_path, inventory_path)
        ):
            return result
        gate = _read_json(gate_path)
        schema = _cache_public_schema(gate)
        if schema is None:
            return result
        metrics = pd.read_csv(private_path)
        inventory = pd.read_csv(inventory_path)
        required = {
            "patient_id",
            "cohort",
            "strategy",
            "scope",
            "selection_reasons",
            "cache_path",
            "cache_content_sha256",
            "cache_file_sha256",
            "cache_schema_version",
            "cache_patient_identity_match",
            "cache_complete_input_contract_match",
            "builder_contract_sha256",
            "input_provenance_sha256",
            "base_only_later_support_loaded_count",
            "shape_valid",
            "dtype_float32",
            "finite",
            "whole_visit_nonconstant",
            "phase_indices_in_range",
            "canonical_orientation",
            "grid_shape_zyx",
            "grid_spacing_xyz_mm",
            "valid_source_voxels_json",
            "eligibility_valid_source_voxels_json",
            "exact_valid_source_count_match",
            "frozen_grid_center_match",
            "cache_roundtrip_pass",
            "model_loader_only_dce7",
        }
        inventory_required = {
            "patient_id",
            "cohort",
            "cache_path",
            "cache_file_sha256",
            "cache_content_sha256",
            "builder_contract_sha256",
            "input_provenance_sha256",
        }
        if not required.issubset(metrics) or not inventory_required.issubset(inventory):
            return result
        for frame in (metrics, inventory):
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame["cohort"] = frame["cohort"].astype(str)
        eligible_ids = eligibility["eligible_ids"]
        metrics_ids = set(metrics["patient_id"])
        inventory_ids = set(inventory["patient_id"])
        expected_cache_root = (experiment_root / "cache/c1b_h").resolve()
        paths = metrics["cache_path"].map(
            lambda value: _resolve_cache_path(value, experiment_root)
        )
        filenames_match = all(
            path.name == f"{hashlib.sha256(patient_id.encode('utf-8')).hexdigest()}.npz"
            for patient_id, path in zip(metrics["patient_id"], paths, strict=True)
        )
        paths_present = bool(
            paths.map(lambda path: path.is_file()).all()
            and paths.map(lambda path: path.resolve().parent == expected_cache_root).all()
        )
        path_records = list(
            zip(paths.tolist(), metrics["cache_file_sha256"].astype(str), strict=True)
        )
        if path_records:
            workers = min(8, len(path_records))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                live_hash_matches = list(
                    executor.map(_live_cache_hash_check, path_records)
                )
            live_hash_fraction = float(np.mean(live_hash_matches))
        else:
            live_hash_fraction = 0.0
        cache_files_independent = bool(
            path_records
            and all(path.stat().st_nlink == 1 for path, _digest in path_records)
        )
        private_csvs = sorted(
            path
            for base in (experiment_root / "manifests", experiment_root / "metrics")
            if base.is_dir()
            for path in base.rglob("*.private.csv")
        )
        private_permissions_ok = bool(
            _owner_only(experiment_root / "cache")
            and _owner_only(experiment_root / "cache/c1b_h")
            and path_records
            and all(_owner_only(path) for path, _digest in path_records)
            and private_csvs
            and all(_owner_only(path) for path in private_csvs)
        )
        joined = metrics[list(inventory_required)].merge(
            inventory[list(inventory_required)],
            on=["patient_id", "cohort"],
            how="outer",
            validate="one_to_one",
            suffixes=("_metrics", "_inventory"),
            indicator=True,
        )
        inventory_match = bool(
            joined["_merge"].eq("both").all()
            and all(
                joined[f"{column}_metrics"].astype(str).eq(
                    joined[f"{column}_inventory"].astype(str)
                ).all()
                for column in inventory_required - {"patient_id", "cohort"}
            )
        )
        hashes_valid = bool(
            metrics["cache_file_sha256"].map(is_sha256).all()
            and metrics["cache_content_sha256"].map(is_sha256).all()
            and metrics["input_provenance_sha256"].map(is_sha256).all()
            and metrics["builder_contract_sha256"].map(is_sha256).all()
        )
        counts_match = True
        for row in metrics.itertuples(index=False):
            expected = eligibility["eligible_visit_counts"].get(str(row.patient_id))
            try:
                recorded = [
                    int(value)
                    for value in json.loads(str(row.valid_source_voxels_json))
                ]
                copied = [
                    int(value)
                    for value in json.loads(
                        str(row.eligibility_valid_source_voxels_json)
                    )
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                counts_match = False
                break
            if (
                expected is None
                or len(recorded) != len(VISITS)
                or recorded != copied
                or copied != expected
                or any(value <= 0 for value in recorded)
            ):
                counts_match = False
                break
        shape_fraction = _mean_bool(metrics, "shape_valid")
        orientation_fraction = float(
            metrics["canonical_orientation"].astype(str).eq("RAS+").mean()
        )
        phase_fraction = _mean_bool(metrics, "phase_indices_in_range")
        grid_fraction = min(
            _mean_bool(metrics, "frozen_grid_center_match"),
            float(metrics["grid_shape_zyx"].astype(str).eq("112x176x160").mean()),
            float(
                metrics["grid_spacing_xyz_mm"]
                .astype(str)
                .eq("0.9x0.9x2.0")
                .mean()
            ),
        )
        provenance_fraction = _mean_bool(metrics, "cache_complete_input_contract_match")
        exact_fraction = _mean_bool(metrics, "exact_valid_source_count_match")
        roundtrip_fraction = _mean_bool(metrics, "cache_roundtrip_pass")
        finite_fraction = _mean_bool(metrics, "finite")
        nonconstant_fraction = _mean_bool(metrics, "whole_visit_nonconstant")
        loader_fraction = _mean_bool(metrics, "model_loader_only_dce7")
        completed_patients = _safe_int(gate.get("completed_cache_patients"))
        completed_visits = _safe_int(gate.get("completed_cache_visits"))
        completion_fraction = _safe_float(gate.get("cache_completion_fraction"))
        artifact_hashes = gate.get("private_artifact_sha256", {})
        public_hash_ok = bool(
            _hash_matches(artifact_hashes, private_path.name, private_path)
            and _hash_matches(artifact_hashes, inventory_path.name, inventory_path)
            and str(
                artifact_hashes.get(
                    "technical_eligibility_patients.private.csv", ""
                )
            ).lower()
            == str(eligibility["patient_manifest_sha256"])
            and str(
                artifact_hashes.get(
                    "technical_eligibility_visits.private.csv", ""
                )
            ).lower()
            == str(eligibility["visit_manifest_sha256"])
        )
        public_fractions_ok = bool(
            _safe_float(gate.get("cache_roundtrip_pass_fraction"))
            == roundtrip_fraction
            and _safe_float(gate.get("exact_valid_source_count_match_fraction"))
            == exact_fraction
            and _safe_float(gate.get("frozen_grid_center_match_fraction"))
            == _mean_bool(metrics, "frozen_grid_center_match")
            and _safe_float(gate.get("finite_fraction")) == finite_fraction
            and _safe_float(gate.get("whole_visit_nonconstant_fraction"))
            == nonconstant_fraction
            and _safe_float(gate.get("phase_indices_in_range_fraction"))
            == phase_fraction
            and _safe_float(gate.get("model_loader_only_dce7_fraction"))
            == loader_fraction
        )
        eligibility_hash_ok = public_hash_ok
        # The formal runner fixes transforms to None for every visit and
        # selects every row only by the general eligibility rule.  These
        # properties are also bound into each input-provenance digest.
        manual_absent = bool(
            metrics["strategy"].astype(str).eq("H").all()
            and metrics["selection_reasons"]
            .astype(str)
            .eq("technical_eligibility_all")
            .all()
        )
        catastrophic = 0 if counts_match else 1
        geometry_excluded = bool(
            loader_fraction == 1.0
            and not as_bool(gate.get("geometry_metadata_is_model_input", True))
            and _safe_int(gate.get("base_only_later_visit_supports_loaded")) == 0
        )

        eligible_patients = _safe_int(gate.get("eligible_patients"))
        eligible_visits = _safe_int(gate.get("eligible_visits"))
        identity_complete = bool(
            len(metrics) == len(eligible_ids)
            and metrics_ids == eligible_ids == inventory_ids
            and not metrics["patient_id"].duplicated().any()
            and not inventory["patient_id"].duplicated().any()
            and not metrics["cache_path"].astype(str).duplicated().any()
            and inventory_match
            and filenames_match
            and paths_present
            and public_hash_ok
            and eligibility_hash_ok
            and eligible_patients == eligibility["eligible_patients"]
            and eligible_visits == eligibility["eligible_visits"]
            and completed_patients == eligibility["eligible_patients"]
            and completed_visits == eligibility["eligible_visits"]
            and completion_fraction == 1.0
            and eligibility["grid_geometry_closure"]
        )
        content_complete = bool(
            identity_complete
            and str(gate.get("status")) == "PASS"
            and str(gate.get("strategy")) == "C1B-H"
            and not as_bool(gate.get("stage_b_authorized", True))
            and not as_bool(gate.get("contains_patient_identifiers", True))
            and metrics["scope"].astype(str).eq("all").all()
            and metrics["strategy"].astype(str).eq("H").all()
            and metrics["cache_schema_version"].eq(3).all()
            and metrics["dtype_float32"].map(as_bool).all()
            and counts_match
            and exact_fraction == 1.0
            and finite_fraction == 1.0
            and nonconstant_fraction == 1.0
            and shape_fraction == 1.0
            and orientation_fraction == 1.0
            and phase_fraction == 1.0
            and grid_fraction == 1.0
            and provenance_fraction == 1.0
            and public_fractions_ok
        )
        roundtrip_hash_pass = bool(
            content_complete
            and roundtrip_fraction == 1.0
            and hashes_valid
            and live_hash_fraction == 1.0
            and cache_files_independent
            and private_permissions_ok
            and upstream_semantic_hash is not None
            and metrics["builder_contract_sha256"]
            .astype(str)
            .eq(str(upstream_semantic_hash))
            .all()
        )
        for path in (gate_path, private_path, inventory_path):
            _add_provenance(
                provenance, path, experiment_root=experiment_root, repo_root=repo_root
            )
        result.update(
            {
                "loaded": True,
                "schema": schema,
                "identity_complete": identity_complete,
                "content_complete": content_complete,
                "roundtrip_hash_pass": roundtrip_hash_pass,
                "manual_corrections_absent": manual_absent,
                "catastrophic_cases": catastrophic,
                "geometry_excluded": geometry_excluded,
                "eligible_patients": eligible_patients,
                "eligible_visits": eligible_visits,
                "completed_patients": completed_patients,
                "completed_visits": completed_visits,
                "completion_fraction": completion_fraction,
                "exact_overlap_fraction": exact_fraction,
                "finite_fraction": finite_fraction,
                "nonconstant_fraction": nonconstant_fraction,
                "shape_fraction": shape_fraction,
                "orientation_fraction": orientation_fraction,
                "phase_fraction": phase_fraction,
                "grid_fraction": grid_fraction,
                "provenance_fraction": provenance_fraction,
                "roundtrip_fraction": roundtrip_fraction,
                "loader_fraction": loader_fraction,
                "private_metrics_sha256": sha256_file(private_path),
                "cache_inventory_sha256": sha256_file(inventory_path),
                "live_cache_file_hash_fraction": live_hash_fraction,
                "cache_files_independent": cache_files_independent,
                "private_permissions_ok": private_permissions_ok,
                "error_code": None,
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _empty_cache()
    return result


def check(
    number: int, name: str, passed: bool, observed: Any, requirement: str
) -> dict[str, Any]:
    return {
        "item": int(number),
        "gate": name,
        "status": "PASS" if bool(passed) else "FAIL",
        "observed": observed,
        "requirement": requirement,
    }


def _build_gates(evidence: dict[str, Any], *, privacy_ok: bool) -> list[dict[str, Any]]:
    eligibility = evidence["eligibility"]
    inherited = evidence["inherited"]
    formal = evidence["formal"]
    cache = evidence["cache"]
    upstream_ok = evidence["upstream_ok"]
    return [
        check(
            1,
            "eligibility_rule_frozen_before_stage_b",
            evidence["preregistration_ok"]
            and eligibility["frozen"]
            and evidence["stage_b_artifacts_before_gate"] == 0,
            {
                "preregistration_verified": evidence["preregistration_ok"],
                "eligibility_output_bound_to_plan": eligibility["frozen"],
                "stage_b_artifacts_before_gate": evidence[
                    "stage_b_artifacts_before_gate"
                ],
            },
            "规则在 eligibility/Stage-B 结果之前冻结，且 finalizer 前无 Stage-B 产物",
        ),
        check(
            2,
            "eligibility_outcome_free_and_public_private",
            eligibility["outcome_free"] and privacy_ok,
            {
                "outcome_forbidden_field_lists_empty": eligibility["outcome_free"],
                "public_privacy_scan": "PASS" if privacy_ok else "FAIL",
            },
            "eligibility 仅读 source/geometry/grid/overlap；公开文本无 ID、UID 或绝对路径",
        ),
        check(
            3,
            "eligible_cohort_mechanically_determined",
            eligibility["programmatic"],
            {
                "candidate_patients": eligibility["candidate_patients"],
                "eligible_patients": eligibility["eligible_patients"],
                "excluded_patients": eligibility["excluded_patients"],
            },
            "从完整 candidate manifest 机械执行通用四访 AND；无硬编码人数或病例规则",
        ),
        check(
            4,
            "all_eligible_visits_positive_overlap",
            eligibility["positive_overlap"],
            {
                "eligible_visits": eligibility["eligible_visits"],
                "candidate_valid_visits": eligibility["valid_visits"],
                "candidate_zero_overlap_visits": eligibility[
                    "zero_overlap_visits"
                ],
                "frozen_grid_and_geometry_digest_closure": eligibility[
                    "grid_geometry_closure"
                ],
            },
            "每名 eligible patient 的 T0-T3 均有 valid_source_voxels > 0",
        ),
        check(
            5,
            "dicom_repair_contract",
            upstream_ok and inherited["dicom_pass"],
            inherited["dicom_observed"],
            "旧 raw-DICOM PixelData/geometry repair 证据 hash 不变且持续 PASS",
        ),
        check(
            6,
            "true_ras_orientation_contract",
            upstream_ok and inherited["orientation_pass"],
            inherited["orientation_observed"],
            "完整 candidate population 持续为真实 array-reordered RAS+，非 header relabel",
        ),
        check(
            7,
            "c1b_h_strategy_frozen",
            upstream_ok and inherited["strategy_pass"],
            inherited["strategy_observed"],
            "唯一正式策略仍为 C1B-H；旧决策与不可变树 hash 闭合",
        ),
        check(
            8,
            "formal_ftv_support_containment",
            upstream_ok and formal["containment_pass"],
            {
                "formal_patients_after_eligibility_intersection": formal[
                    "formal_patients_after_intersection"
                ],
                "formal_visits_after_intersection": formal[
                    "formal_visits_after_intersection"
                ],
                "exact_containment_rate": formal["containment_rate"],
            },
            "旧 formal support 先与新 eligible IDs 取交集，再机械重聚合；rate >= 0.95",
        ),
        check(
            9,
            "ftv_retention_q05",
            upstream_ok and formal["retention_pass"],
            {
                "formal_visits_after_intersection": formal[
                    "formal_visits_after_intersection"
                ],
                "physical_volume_retention_q05": formal["retention_q05"],
            },
            "交集后 formal physical FTV retention Q05 >= 0.95",
        ),
        check(
            10,
            "grounding_observability_loss_side_only",
            upstream_ok and formal["grounding_pass"],
            {
                "formal_observable_visits": formal[
                    "grounding_observable_visits"
                ],
                "formal_ineligible_visits": formal[
                    "grounding_ineligible_visits"
                ],
                "is_model_input": False if formal["grounding_pass"] else None,
            },
            "grounding_observable_mask 仅为 loss-side metadata，不筛 base training",
        ),
        check(
            11,
            "complete_dce7_cache",
            cache["identity_complete"] and cache["content_complete"],
            {
                "cache_evidence_schema": cache["schema"],
                "eligible_patients": eligibility["eligible_patients"],
                "completed_patients": cache["completed_patients"],
                "eligible_visits": eligibility["eligible_visits"],
                "completed_visits": cache["completed_visits"],
                "completion_fraction": cache["completion_fraction"],
                "finite_fraction": cache["finite_fraction"],
                "nonconstant_fraction": cache["nonconstant_fraction"],
                "shape_fraction": cache["shape_fraction"],
                "orientation_fraction": cache["orientation_fraction"],
                "phase_fraction": cache["phase_fraction"],
                "grid_fraction": cache["grid_fraction"],
            },
            "eligible patient 行集精确一致；四访 DCE7 cache 完成率与全部 QC 均为 100%",
        ),
        check(
            12,
            "cache_roundtrip_and_hash",
            cache["roundtrip_hash_pass"],
            {
                "roundtrip_fraction": cache["roundtrip_fraction"],
                "input_provenance_fraction": cache["provenance_fraction"],
                "eligibility_exact_count_match_fraction": cache[
                    "exact_overlap_fraction"
                ],
                "private_tables_sha256_bound": cache["identity_complete"],
                "live_cache_file_hash_fraction": cache[
                    "live_cache_file_hash_fraction"
                ],
                "cache_files_have_single_link": cache["cache_files_independent"],
                "cache_and_private_assets_owner_only": cache[
                    "private_permissions_ok"
                ],
            },
            "cache reload/round-trip、文件/content/input provenance hash 与 eligibility count 闭合",
        ),
        check(
            13,
            "no_patient_specific_manual_correction",
            upstream_ok
            and inherited["no_manual_repair_pass"]
            and eligibility["programmatic"]
            and cache["manual_corrections_absent"],
            {
                "patient_specific_eligibility_rules": 0
                if eligibility["programmatic"]
                else None,
                "manual_transform_trials": inherited["manual_transform_trials"],
                "cache_manual_corrections": 0
                if cache["manual_corrections_absent"]
                else None,
            },
            "无 patient-specific flip、translation、recenter、registration repair 或 known-case rule",
        ),
        check(
            14,
            "no_unresolved_catastrophic_resampling",
            eligibility["positive_overlap"]
            and cache["content_complete"]
            and cache["catastrophic_cases"] == 0,
            {
                "eligible_zero_overlap_visits": 0
                if eligibility["positive_overlap"]
                else None,
                "unresolved_catastrophic_resampling_cases": cache[
                    "catastrophic_cases"
                ],
            },
            "eligible cohort 无 unresolved catastrophic overlap/resampling case",
        ),
        check(
            15,
            "geometry_metadata_excluded_from_model_tensor",
            upstream_ok
            and inherited["valid_source_mask_not_input"]
            and cache["geometry_excluded"]
            and cache["loader_fraction"] == 1.0,
            {
                "model_loader_image_only_fraction": cache["loader_fraction"],
                "geometry_metadata_is_model_input": False
                if cache["geometry_excluded"]
                else None,
                "valid_source_mask_is_model_input": inherited[
                    "valid_source_mask_is_model_input"
                ],
            },
            "model tensor 仅含 DCE7 image；geometry/mask/support/provenance 均为 sidecar",
        ),
    ]


def _collect_inherited(
    *,
    experiment_root: Path,
    repo_root: Path,
    eligibility: dict[str, Any],
    upstream_ok: bool,
    provenance: dict[str, str],
) -> dict[str, Any]:
    empty = {
        "dicom_pass": False,
        "dicom_observed": {"status": "UNAVAILABLE"},
        "orientation_pass": False,
        "orientation_observed": {"status": "UNAVAILABLE"},
        "strategy_pass": False,
        "strategy_observed": {"status": "UNAVAILABLE"},
        "no_manual_repair_pass": False,
        "manual_transform_trials": None,
        "valid_source_mask_not_input": False,
        "valid_source_mask_is_model_input": None,
    }
    prior = repo_root / PRIOR_RELATIVE
    audit = repo_root / AUDIT_RELATIVE
    paths = {
        "dicom": prior / "metrics/dicom_pixel_rebuild_gate.json",
        "orientation": prior / "metrics/orientation_validation_gate.json",
        "strategy": prior / "metrics/registration_strategy_decision.json",
        "prior_no_go": prior / "STAGE_A_NO_GO.json",
        "audit": audit / "AUDIT_NOT_REPAIRABLE.json",
    }
    try:
        if not upstream_ok or not all(path.is_file() for path in paths.values()):
            return empty
        payloads = {name: _read_json(path) for name, path in paths.items()}
        dicom = payloads["dicom"]
        repairs = dicom.get("all_model_input_singular_visits", {})
        dicom_pass = bool(
            str(dicom.get("status")) == "PASS"
            and _safe_int(repairs.get("visits")) >= 0
            and _safe_int(repairs.get("passed")) == _safe_int(repairs.get("visits"))
            and _safe_int(repairs.get("failed")) == 0
            and _safe_float(repairs.get("pixel_order_verified_fraction")) == 1.0
            and _safe_float(repairs.get("max_cell_error")) == 0.0
            and _safe_float(repairs.get("max_footprint_corner_error_mm")) <= 0.1
            and as_bool(repairs.get("all_finite_nonconstant"))
            and as_bool(repairs.get("all_qform_sform_valid"))
            and _safe_int(repairs.get("decoded_cells"))
            == _safe_int(repairs.get("verified_cells"))
        )
        orientation = payloads["orientation"]
        orientation_pass = bool(
            eligibility["loaded"]
            and _safe_float(orientation.get("canonical_ras_fraction")) == 1.0
            and _safe_int(orientation.get("model_input_patients"))
            == eligibility["candidate_patients"]
            and _safe_int(orientation.get("model_input_visits"))
            == eligibility["candidate_visits"]
            and as_bool(orientation.get("array_reordering_implemented_and_unit_tested"))
            and not as_bool(orientation.get("header_only_label_change", True))
            and as_bool(orientation.get("left_right_consistent"))
            and as_bool(orientation.get("anterior_posterior_consistent"))
            and as_bool(orientation.get("superior_inferior_consistent"))
            and _safe_float(
                orientation.get("canonical_roundtrip_corner_error_mm_max")
            )
            <= 0.1
            and _safe_float(
                orientation.get("dce_mask_footprint_corner_error_mm_max")
            )
            <= 0.1
        )
        strategy = payloads["strategy"]
        safe = strategy.get("safe_phase_resample", {})
        prior_no_go = payloads["prior_no_go"]
        provenance_audit = payloads["audit"]
        strategy_pass = bool(
            as_bool(strategy.get("decision_frozen"))
            and str(strategy.get("chosen_strategy")) == "H"
            and as_bool(strategy.get("manual_review_complete"))
            and as_bool(strategy.get("manual_review_pass"))
            and as_bool(strategy.get("r_rejected"))
            and str(prior_no_go.get("status")) == "NO-GO"
            and not as_bool(prior_no_go.get("stage_b_authorized", True))
            and str(prior_no_go.get("chosen_input_strategy")) == "C1B-H"
            and str(provenance_audit.get("decision")) == "AUDIT-NOT-REPAIRABLE"
            and not as_bool(provenance_audit.get("repair_allowed", True))
        )
        no_manual = bool(
            _safe_int(provenance_audit.get("manual_transform_trials")) == 0
            and not as_bool(
                provenance_audit.get("registration_transform_used_for_repair", True)
            )
            and not as_bool(provenance_audit.get("prior_stage_a_decision_modified", True))
            and not as_bool(provenance_audit.get("c1b_crop_contract_modified", True))
        )
        mask_not_input = not as_bool(safe.get("valid_source_mask_is_model_input", True))
        for path in paths.values():
            _add_provenance(
                provenance, path, experiment_root=experiment_root, repo_root=repo_root
            )
        return {
            "dicom_pass": dicom_pass,
            "dicom_observed": {
                "repaired_visits": repairs.get("passed"),
                "failed_visits": repairs.get("failed"),
                "max_cell_error": repairs.get("max_cell_error"),
            },
            "orientation_pass": orientation_pass,
            "orientation_observed": {
                "candidate_patients": orientation.get("model_input_patients"),
                "candidate_visits": orientation.get("model_input_visits"),
                "canonical_ras_fraction": orientation.get("canonical_ras_fraction"),
            },
            "strategy_pass": strategy_pass,
            "strategy_observed": {
                "chosen_strategy": strategy.get("chosen_strategy"),
                "decision_frozen": strategy.get("decision_frozen"),
                "prior_stage_a_status": prior_no_go.get("status"),
                "provenance_audit_decision": provenance_audit.get("decision"),
            },
            "no_manual_repair_pass": no_manual,
            "manual_transform_trials": provenance_audit.get("manual_transform_trials"),
            "valid_source_mask_not_input": mask_not_input,
            "valid_source_mask_is_model_input": safe.get(
                "valid_source_mask_is_model_input"
            ),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return empty


def _table_text(gates: list[dict[str, Any]], evidence: dict[str, Any]) -> str:
    eligibility = evidence["eligibility"]
    formal = evidence["formal"]
    cache = evidence["cache"]
    frame = pd.DataFrame(
        {
            "item": [row["item"] for row in gates],
            "gate": [row["gate"] for row in gates],
            "status": [row["status"] for row in gates],
            "candidate_patients": eligibility["candidate_patients"],
            "eligible_patients": eligibility["eligible_patients"],
            "excluded_patients": eligibility["excluded_patients"],
            "candidate_visits": eligibility["candidate_visits"],
            "valid_visits": eligibility["valid_visits"],
            "zero_overlap_visits": eligibility["zero_overlap_visits"],
            "cache_completed_patients": cache["completed_patients"],
            "cache_completion_fraction": cache["completion_fraction"],
            "formal_containment_rate": formal["containment_rate"],
            "ftv_retention_q05": formal["retention_q05"],
            "observed_json": [
                json.dumps(row["observed"], sort_keys=True, ensure_ascii=False)
                for row in gates
            ],
            "requirement": [row["requirement"] for row in gates],
        }
    )
    return frame.to_csv(index=False)


def _formal_public(evidence: dict[str, Any]) -> dict[str, Any]:
    formal = evidence["formal"]
    return {
        "schema_version": 1,
        "status": "PASS" if formal["structure_ok"] else "FAIL",
        "method": "formal_patient_ids_intersect_technical_eligible_ids_then_reaggregate",
        "formal_patients_before_intersection": formal[
            "formal_patients_before_intersection"
        ],
        "formal_patients_after_intersection": formal[
            "formal_patients_after_intersection"
        ],
        "formal_patients_excluded": formal["formal_patients_excluded"],
        "formal_visits_after_intersection": formal["formal_visits_after_intersection"],
        "exact_full_support_containment_rate": formal["containment_rate"],
        "physical_volume_retention_q05": formal["retention_q05"],
        "grounding_observable_visits": formal["grounding_observable_visits"],
        "grounding_ineligible_visits": formal["grounding_ineligible_visits"],
        "contains_patient_identifiers": False,
    }


def _payload(
    evidence: dict[str, Any],
    gates: list[dict[str, Any]],
    privacy: dict[str, Any] | None,
) -> dict[str, Any]:
    status = "GO" if len(gates) == 15 and all(
        row["status"] == "PASS" for row in gates
    ) else "NO-GO"
    eligibility = evidence["eligibility"]
    cache = evidence["cache"]
    formal = evidence["formal"]
    return {
        "schema_version": 1,
        "stage": "A",
        "status": status,
        "chosen_input_strategy": "C1B-H",
        "stage_b_authorized": status == "GO",
        "thresholds_relaxed": False,
        "eligibility_rule_frozen_before_stage_b": gates[0]["status"] == "PASS",
        "eligibility_rule": "AND over T0,T1,T2,T3: valid_source_voxels > 0",
        "eligible_population_patients": eligibility["eligible_patients"],
        "eligible_population_visits": eligibility["eligible_visits"],
        "candidate_population_patients": eligibility["candidate_patients"],
        "candidate_population_visits": eligibility["candidate_visits"],
        "technical_eligibility_manifest_sha256": eligibility[
            "patient_manifest_sha256"
        ],
        "technical_eligibility_visit_manifest_sha256": eligibility[
            "visit_manifest_sha256"
        ],
        "cache_completion_fraction": cache["completion_fraction"],
        "eligible_cache_manifest_sha256": cache["cache_inventory_sha256"],
        "c1b_cache_table_sha256": cache["private_metrics_sha256"],
        "formal_population_after_eligibility_intersection": {
            "patients": formal["formal_patients_after_intersection"],
            "visits": formal["formal_visits_after_intersection"],
            "containment_rate": formal["containment_rate"],
            "ftv_retention_q05": formal["retention_q05"],
        },
        "prior_stage_a_no_go_immutable": evidence["upstream_ok"],
        "prior_audit_not_repairable_immutable": evidence["upstream_ok"],
        "upstream_contracts_and_tracked_trees_immutable": evidence["upstream_ok"],
        "public_privacy_scan": {
            "status": None if privacy is None else privacy.get("status"),
            "scanned_public_text_artifacts": 0
            if privacy is None
            else privacy.get("scanned_public_text_artifacts", 0),
            "finding_count": 0
            if privacy is None
            else len(privacy.get("identifier_or_path_findings", [])),
        },
        "gates": gates,
        "provenance_sha256": dict(sorted(evidence["provenance"].items())),
        "contains_patient_identifiers": False,
    }


def _report_text(payload: dict[str, Any]) -> str:
    gates = payload["gates"]
    formal = payload["formal_population_after_eligibility_intersection"]
    status = payload["status"]
    authorization = "允许" if payload["stage_b_authorized"] else "禁止"
    rows = "\n".join(
        f"| {row['item']} | `{row['gate']}` | {row['status']} | "
        f"`{json.dumps(row['observed'], ensure_ascii=False, sort_keys=True)}` | {row['requirement']} |"
        for row in gates
    )
    failed = [str(row["item"]) for row in gates if row["status"] != "PASS"]
    failed_text = "无" if not failed else "、".join(failed)
    return f"""# Stage A：Four-Visit overlap eligibility + C1B-H model-ready gate

## 结论

`STAGE_A = {status}`；15 项 hard gate 中 {sum(row['status'] == 'PASS' for row in gates)}/15 PASS，失败项：{failed_text}。因此{authorization}启动 Stage B。该结论属于预注册的全新 run，不追溯修改旧 `STAGE_A_NO_GO`，也不改变独立 provenance audit 的 `AUDIT-NOT-REPAIRABLE`。

## Table 1：Technical eligibility + Stage-A QC

- candidate patients：{payload['candidate_population_patients']}
- eligible patients：{payload['eligible_population_patients']}
- excluded patients：{payload['candidate_population_patients'] - payload['eligible_population_patients']}
- candidate visits：{payload['candidate_population_visits']}
- eligible visits：{payload['eligible_population_visits']}
- cache completion：{payload['cache_completion_fraction']}
- formal intersection：{formal['patients']} patients / {formal['visits']} visits
- formal containment：{formal['containment_rate']}
- FTV retention Q05：{formal['ftv_retention_q05']}

| # | Gate | 状态 | 观测（仅聚合） | 冻结要求 |
|---:|---|---|---|---|
{rows}

## 审计边界

Formal containment、retention 与 grounding evidence 均先用 technical-eligible patient set 机械取交集，再从逐访私有证据重聚合；eligibility 本身未读取这些 lesion/FTV 字段。旧 DICOM、RAS+、C1B-H、containment、retention、grounding contracts 以及两个旧实验 tracked tree 均须通过预注册 hash lock。公开表、报告和 sentinel 只含聚合计数与 SHA-256，不含 patient identifier、UID、源路径或逐病例坐标。
"""


def _write_figure(
    path: Path,
    *,
    eligible_patients: int,
    completed_patients: int,
    eligible_visits: int,
    completed_visits: int,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    panels = (
        (axes[0], "Patients", eligible_patients, completed_patients),
        (axes[1], "Visits", eligible_visits, completed_visits),
    )
    for axis, title, eligible, completed in panels:
        bars = axis.bar(
            ["Eligible", "Cache complete"],
            [eligible, completed],
            color=["#4c78a8", "#2ca25f"],
        )
        axis.bar_label(bars, padding=3)
        axis.set_title(title)
        axis.set_ylim(0, max(1, eligible, completed) * 1.12)
        axis.grid(axis="y", alpha=0.25)
    completion = completed_patients / eligible_patients if eligible_patients else 0.0
    fig.suptitle(f"Stage-A C1B-H cache completion QC ({completion:.1%})")
    fig.tight_layout()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        fig.savefig(temporary_name, dpi=180)
        os.replace(temporary_name, path)
    finally:
        plt.close(fig)
        Path(temporary_name).unlink(missing_ok=True)


def finalize_stage_a(
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
    repo_root: Path = REPO_ROOT,
    overwrite: bool = False,
    preregistration_verifier: Callable[..., dict[str, Any]] = verify_preregistration,
    upstream_verifier: Callable[..., dict[str, Any]] = verify_upstream_contract,
    privacy_scanner: Callable[..., dict[str, Any]] = scan_public_artifacts,
) -> dict[str, Any]:
    """Evaluate, write aggregate artifacts, and return the sentinel payload."""

    experiment_root = Path(experiment_root)
    repo_root = Path(repo_root)
    provenance: dict[str, str] = {}
    config_path = experiment_root / "configs/stage_a.json"
    preregistration: dict[str, Any] | None = None
    preregistration_ok = False
    try:
        preregistration = preregistration_verifier(experiment_root=experiment_root)
        preregistration_ok = bool(
            as_bool(preregistration.get("preregistered_before_new_cohort_statistics"))
            and as_bool(preregistration.get("stage_b_requires_stage_a_go"))
            and is_sha256(preregistration.get("plan_sha256"))
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        preregistration = None
    upstream: dict[str, Any] | None = None
    upstream_ok = False
    try:
        upstream = upstream_verifier(
            experiment_root=experiment_root, repo_root=repo_root
        )
        saved_path = experiment_root / "metrics/upstream_contract_verification.json"
        saved = _read_json(saved_path)
        upstream_ok = bool(
            str(upstream.get("status")) == "PASS"
            and str(saved.get("status")) == "PASS"
            and saved.get("file_sha256") == upstream.get("file_sha256")
            and saved.get("tracked_trees") == upstream.get("tracked_trees")
            and str(saved.get("builder_semantic_contract_sha256"))
            == str(upstream.get("builder_semantic_contract_sha256"))
            and as_bool(saved.get("prior_stage_a_no_go_immutable"))
            and as_bool(saved.get("prior_audit_not_repairable_immutable"))
            and preregistration is not None
            and str(saved.get("preregistration_plan_sha256"))
            == str(preregistration.get("plan_sha256"))
        )
        _add_provenance(
            provenance,
            saved_path,
            experiment_root=experiment_root,
            repo_root=repo_root,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        upstream = None
    try:
        config = _read_json(config_path)
        config_ok = bool(
            str(config.get("strategy")) == "C1B-H"
            and config.get("eligibility_required_visits") == list(VISITS)
            and _safe_float(config.get("formal_containment_minimum")) == 0.95
            and _safe_float(config.get("ftv_retention_q05_minimum")) == 0.95
            and _safe_float(config.get("require_cache_completion_fraction")) == 1.0
        )
        if not config_ok:
            config = None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        config = None
    for path in (
        config_path,
        experiment_root / "configs/preregistration_lock.json",
        experiment_root / "configs/upstream_contract_lock.json",
        experiment_root / "EXPERIMENT_PLAN.md",
        Path(__file__).resolve(),
        SCRIPT_ROOT / "audit_public_artifacts.py",
    ):
        _add_provenance(
            provenance, path, experiment_root=experiment_root, repo_root=repo_root
        )

    eligibility = _collect_eligibility(
        experiment_root=experiment_root,
        repo_root=repo_root,
        preregistration=preregistration,
        provenance=provenance,
    )
    inherited = _collect_inherited(
        experiment_root=experiment_root,
        repo_root=repo_root,
        eligibility=eligibility,
        upstream_ok=upstream_ok,
        provenance=provenance,
    )
    formal = _collect_formal(
        experiment_root=experiment_root,
        repo_root=repo_root,
        eligibility=eligibility,
        upstream_ok=upstream_ok,
        config=config,
        provenance=provenance,
    )
    semantic_hash = (
        None if upstream is None else upstream.get("builder_semantic_contract_sha256")
    )
    cache = _collect_cache(
        experiment_root=experiment_root,
        repo_root=repo_root,
        eligibility=eligibility,
        upstream_semantic_hash=None
        if semantic_hash is None
        else str(semantic_hash),
        provenance=provenance,
    )
    evidence = {
        "preregistration_ok": preregistration_ok,
        "upstream_ok": upstream_ok,
        "stage_b_artifacts_before_gate": _stage_b_artifacts(experiment_root),
        "eligibility": eligibility,
        "inherited": inherited,
        "formal": formal,
        "cache": cache,
        "provenance": provenance,
    }

    # First scan live artifacts, then include every would-be aggregate output in
    # memory.  Privacy is item 2, not a newly invented sixteenth hard gate.
    live_privacy = privacy_scanner(root=experiment_root)
    gates = _build_gates(
        evidence, privacy_ok=str(live_privacy.get("status")) == "PASS"
    )
    provisional = _payload(evidence, gates, live_privacy)
    table = _table_text(gates, evidence)
    formal_text = json_text(_formal_public(evidence))
    report = _report_text(provisional)
    sentinel_name = (
        "STAGE_A_GO.json" if provisional["status"] == "GO" else "STAGE_A_NO_GO.json"
    )
    virtual = {
        "metrics/stage_a_gate.json": json_text(provisional),
        "metrics/table1_technical_eligibility_stage_a_qc.csv": table,
        "metrics/formal_eligibility_reaggregation.json": formal_text,
        "reports/stage_a_gate_report.md": report,
        sentinel_name: json_text(provisional),
        (
            "STAGE_A_NO_GO.json"
            if sentinel_name == "STAGE_A_GO.json"
            else "STAGE_A_GO.json"
        ): None,
    }
    privacy = privacy_scanner(root=experiment_root, virtual_text=virtual)
    gates = _build_gates(evidence, privacy_ok=str(privacy.get("status")) == "PASS")
    payload = _payload(evidence, gates, privacy)
    table = _table_text(gates, evidence)
    formal_text = json_text(_formal_public(evidence))
    report = _report_text(payload)
    sentinel_name = (
        "STAGE_A_GO.json" if payload["status"] == "GO" else "STAGE_A_NO_GO.json"
    )
    virtual = {
        "metrics/stage_a_gate.json": json_text(payload),
        "metrics/table1_technical_eligibility_stage_a_qc.csv": table,
        "metrics/formal_eligibility_reaggregation.json": formal_text,
        "reports/stage_a_gate_report.md": report,
        sentinel_name: json_text(payload),
        (
            "STAGE_A_NO_GO.json"
            if sentinel_name == "STAGE_A_GO.json"
            else "STAGE_A_GO.json"
        ): None,
    }
    final_privacy = privacy_scanner(root=experiment_root, virtual_text=virtual)
    if str(final_privacy.get("status")) != str(privacy.get("status")):
        gates = _build_gates(
            evidence, privacy_ok=str(final_privacy.get("status")) == "PASS"
        )
        payload = _payload(evidence, gates, final_privacy)
        table = _table_text(gates, evidence)
        report = _report_text(payload)
        privacy = final_privacy
    else:
        privacy = final_privacy

    active_sentinel = (
        experiment_root / "STAGE_A_GO.json"
        if payload["status"] == "GO"
        else experiment_root / "STAGE_A_NO_GO.json"
    )
    stale_sentinel = (
        experiment_root / "STAGE_A_NO_GO.json"
        if payload["status"] == "GO"
        else experiment_root / "STAGE_A_GO.json"
    )
    outputs = (
        experiment_root / "metrics/stage_a_gate.json",
        experiment_root / "metrics/table1_technical_eligibility_stage_a_qc.csv",
        experiment_root / "metrics/formal_eligibility_reaggregation.json",
        experiment_root / "metrics/public_artifact_privacy_gate.json",
        experiment_root / "reports/stage_a_gate_report.md",
        experiment_root / "figures/03_cache_completion_qc.png",
        active_sentinel,
    )
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError("Stage-A output already exists; pass --overwrite")
    atomic_text(
        experiment_root / "metrics/stage_a_gate.json",
        json_text(payload),
        overwrite=overwrite,
    )
    atomic_text(
        experiment_root / "metrics/table1_technical_eligibility_stage_a_qc.csv",
        table,
        overwrite=overwrite,
    )
    atomic_text(
        experiment_root / "metrics/formal_eligibility_reaggregation.json",
        formal_text,
        overwrite=overwrite,
    )
    atomic_text(
        experiment_root / "metrics/public_artifact_privacy_gate.json",
        json_text(privacy),
        overwrite=overwrite,
    )
    atomic_text(
        experiment_root / "reports/stage_a_gate_report.md",
        report,
        overwrite=overwrite,
    )
    _write_figure(
        experiment_root / "figures/03_cache_completion_qc.png",
        eligible_patients=eligibility["eligible_patients"],
        completed_patients=cache["completed_patients"],
        eligible_visits=eligibility["eligible_visits"],
        completed_visits=cache["completed_visits"],
        overwrite=overwrite,
    )
    stale_sentinel.unlink(missing_ok=True)
    atomic_text(active_sentinel, json_text(payload), overwrite=overwrite)
    return payload


def main() -> None:
    args = parse_args()
    payload = finalize_stage_a(overwrite=args.overwrite)
    print(json_text(payload), end="")


if __name__ == "__main__":
    main()
