#!/usr/bin/env python3
"""Build and validate the complete eligible C1B-H schema-3 cache cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = EXPERIMENT_ROOT / "src"
PRIOR_ROOT = REPO_ROOT / "additional_experiments/c1b_model_ready_ftv_sanity"
PRIOR_SRC = PRIOR_ROOT / "src"
for source in (SRC_ROOT, PRIOR_SRC):
    if str(source) not in os.sys.path:
        os.sys.path.insert(0, str(source))

from c1b_overlap_stageb.io import (  # noqa: E402
    atomic_text,
    json_text,
    sha256_file,
    verify_preregistration,
    verify_upstream_contract,
)
from c1b_sanity.builder import VISITS, builder_contract_sha256  # noqa: E402
from c1b_sanity.cache import load_and_validate_cache, load_model_tensor  # noqa: E402


def _load_prior_runner() -> Any:
    path = PRIOR_ROOT / "scripts/build_validate_model_inputs.py"
    spec = importlib.util.spec_from_file_location("immutable_prior_cache_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the immutable prior cache runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIOR_RUNNER = _load_prior_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eligibility-patients",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/technical_eligibility_patients.private.csv",
    )
    parser.add_argument(
        "--eligibility-visits",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/technical_eligibility_visits.private.csv",
    )
    parser.add_argument(
        "--frozen-grids",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/frozen_c1b_grids.private.csv",
    )
    parser.add_argument(
        "--prior-inventory",
        type=Path,
        default=PRIOR_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--ispy1-visits",
        type=Path,
        default=PRIOR_ROOT / "manifests/ispy1_base_eligibility_visits.private.csv",
    )
    parser.add_argument(
        "--phase-metadata",
        type=Path,
        default=REPO_ROOT
        / "ispy_jepa_tmi_clean/data_processing/metadata/BreastDCEDL_metadata_min_crop.csv",
    )
    parser.add_argument(
        "--prior-cache-dir", type=Path, default=PRIOR_ROOT / "cache/c1b_h"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=EXPERIMENT_ROOT / "cache/c1b_h"
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite-metrics", action="store_true")
    parser.add_argument(
        "--no-prior-hardlinks",
        action="store_true",
        help="Do not hardlink immutable prior cache files into the new cache root.",
    )
    return parser.parse_args()


def _patient_token(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()


def _phase_map(path: Path) -> dict[str, dict[str, float | None]]:
    frame = pd.read_csv(path, usecols=["pid", "pre", "post_early", "post_late"])
    frame["pid"] = frame["pid"].astype(str)
    if frame["pid"].duplicated().any():
        raise ValueError("Acquisition phase table has duplicate patients")
    output: dict[str, dict[str, float | None]] = {}
    for row in frame.itertuples(index=False):
        output[str(row.pid)] = {
            "pre": None if pd.isna(row.pre) else float(row.pre),
            "post_early": None if pd.isna(row.post_early) else float(row.post_early),
            "post_late": None if pd.isna(row.post_late) else float(row.post_late),
        }
    return output


def _strict_ispy1_phase_map(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    frame = pd.read_csv(
        path,
        usecols=[
            "patient_id",
            "visit",
            "status",
            "pre_index",
            "early_index",
            "late_index",
        ],
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["visit"] = frame["visit"].astype(str)
    if frame.duplicated(["patient_id", "visit"]).any():
        raise ValueError("I-SPY1 phase/source manifest has duplicate visits")
    output: dict[tuple[str, str], dict[str, int]] = {}
    for row in frame.itertuples(index=False):
        if str(row.status) != "PASS":
            continue
        output[(str(row.patient_id), str(row.visit))] = {
            "pre": int(row.pre_index),
            "post_early": int(row.early_index),
            "post_late": int(row.late_index),
        }
    return output


def _prepare_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(
        args.eligibility_patients,
        usecols=["patient_id", "cohort", "eligible", "zero_overlap_visit_count"],
    )
    patients["patient_id"] = patients["patient_id"].astype(str)
    if patients["patient_id"].duplicated().any():
        raise ValueError("Technical eligibility patient manifest is duplicate")
    eligible = patients.loc[
        patients["eligible"].astype(bool) & patients["zero_overlap_visit_count"].eq(0)
    ].copy()

    visits = pd.read_csv(
        args.eligibility_visits,
        usecols=[
            "patient_id",
            "cohort",
            "visit",
            "resolved_dce_nifti",
            "valid_source_voxels",
            "target_grid_voxels",
            "has_valid_source_overlap",
            "grid_contract_sha256",
        ],
    )
    visits["patient_id"] = visits["patient_id"].astype(str)
    visits["visit"] = visits["visit"].astype(str)
    selected = visits.loc[visits["patient_id"].isin(set(eligible["patient_id"]))].copy()
    if (
        selected.duplicated(["patient_id", "visit"]).any()
        or len(selected) != len(eligible) * len(VISITS)
        or not selected["has_valid_source_overlap"].astype(bool).all()
        or not selected["valid_source_voxels"].gt(0).all()
    ):
        raise ValueError("Eligible visit manifest is not a complete positive-overlap cohort")

    inventory = pd.read_csv(
        args.prior_inventory,
        usecols=[
            "patient_id",
            "cohort",
            "visit",
            "formal_ftv_overlap",
            "ftv_mask_nifti",
        ],
    )
    inventory["patient_id"] = inventory["patient_id"].astype(str)
    inventory["visit"] = inventory["visit"].astype(str)
    selected = selected.merge(
        inventory,
        on=["patient_id", "cohort", "visit"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not selected["_merge"].eq("both").all():
        raise ValueError("Eligible visit keys do not resolve in frozen input inventory")
    selected = selected.drop(columns="_merge")

    grids = pd.read_csv(
        args.frozen_grids,
        usecols=[
            "patient_id",
            "cohort",
            "grid_center_x_ras_mm",
            "grid_center_y_ras_mm",
            "grid_center_z_ras_mm",
            "grid_contract_sha256",
        ],
    )
    grids["patient_id"] = grids["patient_id"].astype(str)
    grids = grids.loc[grids["patient_id"].isin(set(eligible["patient_id"]))].copy()
    if grids["patient_id"].duplicated().any() or len(grids) != len(eligible):
        raise ValueError("Frozen grids do not exactly cover the eligible population")
    return eligible, selected, grids


def _prepare_prior_links(
    *,
    eligible_ids: set[str],
    source_dir: Path,
    target_dir: Path,
) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for patient_id in sorted(eligible_ids):
        name = f"{_patient_token(patient_id)}.npz"
        source = source_dir / name
        target = target_dir / name
        if target.exists():
            continue
        if not source.is_file():
            continue
        os.link(source, target)
        linked += 1
    return linked


def _build_checked(task: dict[str, Any]) -> dict[str, Any]:
    result = PRIOR_RUNNER.build_one(task)
    cache_path = Path(task["cache_path"])
    arrays, validation = load_and_validate_cache(cache_path)
    actual_counts = np.asarray(arrays["valid_source_mask"], dtype=np.uint8).sum(
        axis=(1, 2, 3, 4), dtype=np.int64
    )
    expected_counts = np.asarray(task["expected_valid_source_voxels"], dtype=np.int64)
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError("Cache valid-source counts disagree with technical eligibility")
    expected_center = np.asarray(task["expected_grid_center_ras_mm"], dtype=np.float64)
    cached_center = np.asarray(arrays["grid_center_ras_mm"], dtype=np.float64)
    if not np.allclose(cached_center, expected_center, atol=1e-8, rtol=0.0):
        raise ValueError("Cache grid disagrees with the pre-materialized frozen grid")
    if str(arrays["builder_contract_sha256"].item()) != builder_contract_sha256():
        raise ValueError("Cache builder semantic contract drifted")
    model = load_model_tensor(cache_path)
    if model.shape != (4, 7, 112, 176, 160) or model.dtype != np.float32:
        raise ValueError("Model-only loader contract is invalid")
    result.update(
        {
            "valid_source_voxels_json": json.dumps(actual_counts.tolist()),
            "eligibility_valid_source_voxels_json": json.dumps(expected_counts.tolist()),
            "exact_valid_source_count_match": True,
            "frozen_grid_center_match": True,
            "cache_roundtrip_pass": True,
            "model_loader_only_dce7": True,
            "cache_file_sha256": validation.file_sha256,
            "cache_content_sha256": validation.content_sha256,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    preregistration = verify_preregistration()
    verify_upstream_contract()
    eligible, visits, grids = _prepare_rows(args)
    eligible_ids = set(eligible["patient_id"].astype(str))
    links_created = 0
    if not args.no_prior_hardlinks:
        links_created = _prepare_prior_links(
            eligible_ids=eligible_ids,
            source_dir=args.prior_cache_dir,
            target_dir=args.cache_dir,
        )

    phases = _phase_map(args.phase_metadata)
    strict_phases = _strict_ispy1_phase_map(args.ispy1_visits)
    grid_map = grids.set_index("patient_id").to_dict("index")
    tasks: list[dict[str, Any]] = []
    for patient_id, group in visits.groupby("patient_id", sort=True):
        patient_id = str(patient_id)
        if len(group) != len(VISITS) or set(group["visit"]) != set(VISITS):
            raise ValueError("An eligible cache task is not complete T0-T3")
        cohorts = set(group["cohort"].astype(str))
        formal_values = set(group["formal_ftv_overlap"].astype(bool))
        if len(cohorts) != 1 or len(formal_values) != 1:
            raise ValueError("Cache task cohort/formal membership is inconsistent")
        cohort = next(iter(cohorts))
        formal = bool(next(iter(formal_values)))
        row_map: dict[str, dict[str, Any]] = {}
        metadata: dict[str, dict[str, Any] | None] = {}
        expected_counts: list[int] = []
        for visit in VISITS:
            row = group.loc[group["visit"].eq(visit)].iloc[0].to_dict()
            mask = row.get("ftv_mask_nifti")
            mask_path = str(mask) if isinstance(mask, str) and mask else ""
            row["ftv_mask_nifti"] = mask_path if visit == "T0" or formal else ""
            row_map[visit] = row
            metadata[visit] = strict_phases.get(
                (patient_id, visit), phases.get(patient_id)
            )
            expected_counts.append(int(row["valid_source_voxels"]))
        grid_row = grid_map[patient_id]
        cache_path = args.cache_dir / f"{_patient_token(patient_id)}.npz"
        tasks.append(
            {
                "patient_id": patient_id,
                "cohort": cohort,
                "formal_ftv_overlap": formal,
                "rows": row_map,
                "phase_metadata_by_visit": metadata,
                "transforms": {visit: None for visit in VISITS},
                "strategy": "H",
                "scope": "all",
                "selection_reasons": ["technical_eligibility_all"],
                "cache_path": str(cache_path),
                "overwrite": False,
                "duplicate_check": False,
                "expected_prior_cache_file_sha256": "",
                "seed": int(_patient_token(patient_id)[:8], 16),
                "expected_valid_source_voxels": expected_counts,
                "expected_grid_center_ras_mm": [
                    float(grid_row["grid_center_x_ras_mm"]),
                    float(grid_row["grid_center_y_ras_mm"]),
                    float(grid_row["grid_center_z_ras_mm"]),
                ],
            }
        )

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_build_checked, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "completed_patients": completed,
                            "eligible_patients": len(tasks),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )

    metrics = pd.DataFrame(results).sort_values("patient_id", kind="stable")
    if len(metrics) != len(eligible) or metrics["patient_id"].duplicated().any():
        raise ValueError("Cache output does not exactly cover eligible patients")
    required = (
        metrics["shape_valid"].astype(bool)
        & metrics["dtype_float32"].astype(bool)
        & metrics["finite"].astype(bool)
        & metrics["whole_visit_nonconstant"].astype(bool)
        & metrics["phase_indices_in_range"].astype(bool)
        & metrics["cache_patient_identity_match"].astype(bool)
        & metrics["cache_complete_input_contract_match"].astype(bool)
        & metrics["exact_valid_source_count_match"].astype(bool)
        & metrics["frozen_grid_center_match"].astype(bool)
        & metrics["cache_roundtrip_pass"].astype(bool)
        & metrics["model_loader_only_dce7"].astype(bool)
        & metrics["cache_schema_version"].eq(3)
        & metrics["base_only_later_support_loaded_count"].eq(0)
    )
    if not required.all():
        raise ValueError("At least one eligible cache failed the frozen Stage-A contract")

    private_metrics = EXPERIMENT_ROOT / "metrics/model_input_pipeline_h_all.private.csv"
    private_inventory = EXPERIMENT_ROOT / "manifests/model_input_cache_inventory.private.csv"
    private_metrics.parent.mkdir(parents=True, exist_ok=True)
    private_metrics.parent.chmod(0o700)
    inventory = metrics[
        [
            "patient_id",
            "cohort",
            "cache_path",
            "cache_file_sha256",
            "cache_content_sha256",
            "builder_contract_sha256",
            "input_provenance_sha256",
        ]
    ].copy()
    atomic_text(
        private_metrics,
        metrics.to_csv(index=False),
        overwrite=args.overwrite_metrics,
    )
    atomic_text(
        private_inventory,
        inventory.to_csv(index=False),
        overwrite=args.overwrite_metrics,
    )

    elapsed = float(time.monotonic() - started)
    public = {
        "schema_version": 1,
        "stage": "A",
        "strategy": "C1B-H",
        "status": "PASS",
        "eligible_patients": int(len(eligible)),
        "eligible_visits": int(len(visits)),
        "completed_cache_patients": int(len(metrics)),
        "completed_cache_visits": int(len(metrics) * len(VISITS)),
        "cache_completion_fraction": float(len(metrics) / len(eligible)),
        "cache_schema_version": 3,
        "builder_contract_sha256": builder_contract_sha256(),
        "cache_roundtrip_pass_fraction": float(metrics["cache_roundtrip_pass"].mean()),
        "exact_valid_source_count_match_fraction": float(
            metrics["exact_valid_source_count_match"].mean()
        ),
        "frozen_grid_center_match_fraction": float(
            metrics["frozen_grid_center_match"].mean()
        ),
        "finite_fraction": float(metrics["finite"].mean()),
        "whole_visit_nonconstant_fraction": float(
            metrics["whole_visit_nonconstant"].mean()
        ),
        "phase_indices_in_range_fraction": float(
            metrics["phase_indices_in_range"].mean()
        ),
        "model_loader_only_dce7_fraction": float(
            metrics["model_loader_only_dce7"].mean()
        ),
        "geometry_metadata_is_model_input": False,
        "base_only_later_visit_supports_loaded": int(
            metrics["base_only_later_support_loaded_count"].sum()
        ),
        "prior_cache_files_linked_this_run": int(links_created),
        "cache_reused_after_current_contract_rebuild": int(metrics["cache_reused"].sum()),
        "preregistration_plan_sha256": preregistration["plan_sha256"],
        "private_artifact_sha256": {
            "model_input_pipeline_h_all.private.csv": sha256_file(private_metrics),
            "model_input_cache_inventory.private.csv": sha256_file(private_inventory),
            "technical_eligibility_patients.private.csv": sha256_file(
                args.eligibility_patients
            ),
            "technical_eligibility_visits.private.csv": sha256_file(
                args.eligibility_visits
            ),
        },
        "elapsed_seconds": elapsed,
        "stage_b_authorized": False,
        "contains_patient_identifiers": False,
    }
    atomic_text(
        EXPERIMENT_ROOT / "metrics/model_input_pipeline_h_all_gate.json",
        json_text(public),
        overwrite=args.overwrite_metrics,
    )
    report = f"""# Eligible C1B-H model-input cache validation

本轮只对 technical-eligibility runner 产生的 {len(eligible)} 名 eligible patients（{len(visits)} visits）执行冻结 C1B-H production builder。{len(metrics)}/{len(eligible)} patient caches 已完成 schema-3 原子写入/复用、完整当前 input-contract rebuild、reload 与 model-only loader round-trip。

- exact eligibility valid-source count match：{int(metrics['exact_valid_source_count_match'].sum())}/{len(metrics)} patients（全部四访）；
- frozen grid center match：{int(metrics['frozen_grid_center_match'].sum())}/{len(metrics)}；
- finite/nonconstant/phase/shape/schema/provenance：{int(required.sum())}/{len(required)}；
- cache completion：{len(metrics)}/{len(eligible)} = 100%；
- model loader 仅返回 `[4,7,112,176,160] float32` DCE7，geometry/valid-source/support/phase/provenance 均为 sidecar。

此 cache 子门 PASS 仍不单独授权 Stage B；必须由 15 项 Stage-A finalizer 写出唯一 `STAGE_A_GO.json`。
"""
    atomic_text(
        EXPERIMENT_ROOT / "reports/model_input_pipeline_validation.md",
        report,
        overwrite=args.overwrite_metrics,
    )
    print(json_text(public), end="")


if __name__ == "__main__":
    main()

