#!/usr/bin/env python3
"""Run preregistered four-visit valid-source-overlap eligibility."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
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

from c1b_overlap_stageb.eligibility import (  # noqa: E402
    VISITS,
    build_patient_eligibility,
    canonical_header_geometry,
    count_valid_source_voxels,
    frozen_grid_contract_sha256,
    geometry_contract_sha256,
)
from c1b_overlap_stageb.io import (  # noqa: E402
    atomic_text,
    json_text,
    sha256_file,
    verify_preregistration,
    verify_upstream_contract,
)
from c1b_sanity.geometry import (  # noqa: E402
    input_from_output_affine,
    make_c1b_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PRIOR_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--grids",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/frozen_c1b_grids.private.csv",
    )
    parser.add_argument(
        "--candidate-keys",
        type=Path,
        default=PRIOR_ROOT / "metrics/orientation_resampling_patient_visit.private.csv",
    )
    parser.add_argument(
        "--ispy1-visits",
        type=Path,
        default=PRIOR_ROOT / "manifests/ispy1_base_eligibility_visits.private.csv",
    )
    parser.add_argument(
        "--repair-dir", type=Path, default=PRIOR_ROOT / "manifests/repair_private"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _candidate_visits(args: argparse.Namespace) -> pd.DataFrame:
    # Structural allowlist: no lesion, FTV, LD, pCR, clinical, treatment,
    # subtype, loss, representation, or downstream-performance column can enter.
    inventory = pd.read_csv(
        args.inventory,
        usecols=[
            "patient_id",
            "cohort",
            "visit",
            "dce_nifti",
            "pixel_rebuild_required",
        ],
    )
    inventory["patient_id"] = inventory["patient_id"].astype(str)
    inventory["visit"] = inventory["visit"].astype(str)
    if inventory.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Candidate inventory contains duplicate visits")

    # The prior final model-input population is inherited through keys only.
    # Old padding/overlap values are structurally excluded by usecols.
    keys = pd.read_csv(
        args.candidate_keys,
        usecols=["patient_id", "cohort", "visit"],
    )
    keys["patient_id"] = keys["patient_id"].astype(str)
    keys["visit"] = keys["visit"].astype(str)
    if keys.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Frozen candidate-key manifest contains duplicate visits")
    selected = keys.merge(
        inventory,
        on=["patient_id", "cohort", "visit"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not selected["_merge"].eq("both").all():
        raise ValueError("Frozen candidate keys do not resolve in source inventory")
    selected = selected.drop(columns="_merge")

    strict = pd.read_csv(
        args.ispy1_visits,
        usecols=["patient_id", "visit", "status", "rebuilt_nifti"],
    )
    strict["patient_id"] = strict["patient_id"].astype(str)
    strict["visit"] = strict["visit"].astype(str)
    if strict.duplicated(["patient_id", "visit"]).any():
        raise ValueError("I-SPY1 visit source table contains duplicates")
    strict_map: dict[tuple[str, str], Path] = {}
    for row in strict.itertuples(index=False):
        if str(row.status) != "PASS":
            continue
        rebuilt = Path(str(row.rebuilt_nifti))
        if rebuilt.is_file():
            strict_map[(str(row.patient_id), str(row.visit))] = rebuilt

    repairs: dict[tuple[str, str], Path] = {}
    for path in sorted(args.repair_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise ValueError("Inherited raw-DICOM repair is not PASS")
        key = (str(payload["patient_id"]), str(payload["visit"]))
        rebuilt = Path(payload["private"]["output_nifti"])
        if key in repairs or not rebuilt.is_file():
            raise ValueError("Inherited repair map is duplicate/incomplete")
        repairs[key] = rebuilt

    resolved: list[str] = []
    for row in selected.itertuples(index=False):
        key = (str(row.patient_id), str(row.visit))
        if str(row.cohort) == "I-SPY1":
            if key not in strict_map:
                raise ValueError("Candidate I-SPY1 visit lacks its mandatory PASS rebuild")
            source = strict_map[key]
        elif bool(row.pixel_rebuild_required):
            if key not in repairs:
                raise ValueError("Required I-SPY2 DICOM repair is missing")
            source = repairs[key]
        else:
            source = Path(str(row.dce_nifti))
        if not source.is_file():
            raise FileNotFoundError("A candidate imaging source is missing")
        resolved.append(str(source.resolve()))
    selected["resolved_dce_nifti"] = resolved
    return selected.sort_values(["patient_id", "visit"], kind="stable").reset_index(drop=True)


def _grid_map(
    path: Path,
    expected_cohorts: dict[str, str],
) -> dict[str, dict[str, Any]]:
    # Explicit geometry-only allowlist.  No T0 support path/provenance input is
    # available to the eligibility computation.
    frame = pd.read_csv(
        path,
        usecols=[
            "patient_id",
            "cohort",
            "grid_center_x_ras_mm",
            "grid_center_y_ras_mm",
            "grid_center_z_ras_mm",
            "grid_shape_zyx_json",
            "grid_spacing_xyz_mm_json",
            "grid_affine_ras_json",
            "grid_contract_sha256",
        ],
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    if (
        frame["patient_id"].duplicated().any()
        or set(frame["patient_id"]) != set(expected_cohorts)
    ):
        raise ValueError("Frozen C1B grid manifest does not exactly cover candidates")
    output: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        patient_id = str(row["patient_id"])
        cohort = str(row["cohort"])
        if cohort != expected_cohorts[patient_id]:
            raise ValueError("Frozen-grid cohort disagrees with candidate cohort")
        shape = tuple(int(value) for value in json.loads(str(row["grid_shape_zyx_json"])))
        spacing = tuple(
            float(value) for value in json.loads(str(row["grid_spacing_xyz_mm_json"]))
        )
        affine = np.asarray(json.loads(str(row["grid_affine_ras_json"])), dtype=np.float64)
        observed = frozen_grid_contract_sha256(
            patient_id=patient_id,
            cohort=cohort,
            grid_shape_zyx=shape,
            grid_spacing_xyz_mm=spacing,
            grid_affine_ras=affine,
        )
        if observed != str(row["grid_contract_sha256"]):
            raise ValueError("Frozen-grid contract digest is invalid")
        output[patient_id] = {
            key: value for key, value in row.items() if key not in {"patient_id", "cohort"}
        }
    return output


def _evaluate_visit(task: dict[str, Any]) -> dict[str, Any]:
    source_shape, source_affine = canonical_header_geometry(task["resolved_dce_nifti"])
    center = (
        float(task["grid_center_x_ras_mm"]),
        float(task["grid_center_y_ras_mm"]),
        float(task["grid_center_z_ras_mm"]),
    )
    grid = make_c1b_grid(center)
    recorded_shape = tuple(json.loads(str(task["grid_shape_zyx_json"])))
    recorded_spacing = tuple(json.loads(str(task["grid_spacing_xyz_mm_json"])))
    recorded_affine = np.asarray(json.loads(str(task["grid_affine_ras_json"])), dtype=np.float64)
    if recorded_shape != grid.shape_zyx or not np.allclose(
        recorded_spacing, grid.spacing_xyz_mm, atol=0.0, rtol=0.0
    ) or not np.allclose(recorded_affine, grid.affine_ras, atol=1e-12, rtol=0.0):
        raise ValueError("Frozen-grid row is internally inconsistent")
    mapping = input_from_output_affine(source_affine, grid)
    valid = count_valid_source_voxels(mapping, grid.shape_xyz, source_shape)
    target = int(np.prod(grid.shape_xyz, dtype=np.int64))
    return {
        "patient_id": str(task["patient_id"]),
        "cohort": str(task["cohort"]),
        "visit": str(task["visit"]),
        "resolved_dce_nifti": str(task["resolved_dce_nifti"]),
        "source_shape_xyz_json": json.dumps(list(source_shape)),
        "source_affine_ras_json": json.dumps(source_affine.tolist()),
        "grid_contract_sha256": str(task["grid_contract_sha256"]),
        "geometry_contract_sha256": geometry_contract_sha256(
            source_shape_xyz=source_shape,
            source_affine_ras=source_affine,
            grid_shape_xyz=grid.shape_xyz,
            grid_affine_ras=grid.affine_ras,
        ),
        "valid_source_voxels": valid,
        "target_grid_voxels": target,
        "valid_source_fraction": float(valid / target),
        "has_valid_source_overlap": bool(valid > 0),
        "eligibility_evidence_scope": "source_geometry_x_frozen_c1b_h_grid_only",
    }


def _make_figures(visits: pd.DataFrame, patients: pd.DataFrame) -> None:
    output = EXPERIMENT_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    candidate = int(len(patients))
    eligible = int(patients["eligible"].sum())
    excluded = candidate - eligible

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = ["Candidate\npatients", "Four-visit overlap\neligible", "Excluded"]
    values = [candidate, eligible, excluded]
    colors = ["#4c78a8", "#2ca25f", "#de2d26"]
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Patient count")
    ax.set_title("Pre-registered four-visit technical eligibility flow")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "01_technical_eligibility_flow.png", dpi=180)
    plt.close(fig)

    fractions = visits["valid_source_fraction"].to_numpy(dtype=float)
    positive = fractions[fractions > 0]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(positive, bins=bins, color="#4c78a8", edgecolor="white")
    zeros = int(np.count_nonzero(fractions == 0))
    ax.text(
        0.98,
        0.95,
        f"zero-overlap visits = {zeros}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set_xlabel("Exact valid-source target-voxel fraction")
    ax.set_ylabel("Visit count")
    ax.set_title("C1B-H valid-source overlap distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "02_valid_source_overlap_distribution.png", dpi=180)
    plt.close(fig)


def _assert_public_privacy(texts: list[str], private_ids: set[str]) -> None:
    for payload in texts:
        leaked = [patient_id for patient_id in private_ids if patient_id and patient_id in payload]
        if leaked:
            raise ValueError("A public technical-eligibility artifact contains patient IDs")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    lock = verify_preregistration()
    verify_upstream_contract()
    candidates = _candidate_visits(args)
    patient_ids = set(candidates["patient_id"].astype(str))
    cohort_counts = candidates.groupby("patient_id")["cohort"].nunique()
    if not cohort_counts.eq(1).all():
        raise ValueError("A candidate patient's cohort changes across visits")
    expected_cohorts = (
        candidates.drop_duplicates("patient_id")
        .set_index("patient_id")["cohort"]
        .astype(str)
        .to_dict()
    )
    grids = _grid_map(args.grids, expected_cohorts)
    tasks = []
    for row in candidates.to_dict("records"):
        task = dict(row)
        task.update(grids[str(row["patient_id"])])
        tasks.append(task)

    started = time.monotonic()
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_evaluate_visit, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed == 1 or completed % 250 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "completed_visits": completed,
                            "total_visits": len(futures),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    visits = pd.DataFrame(records).sort_values(
        ["patient_id", "visit"], kind="stable"
    )
    patients = build_patient_eligibility(visits)

    visit_path = EXPERIMENT_ROOT / "manifests/technical_eligibility_visits.private.csv"
    patient_path = EXPERIMENT_ROOT / "manifests/technical_eligibility_patients.private.csv"
    eligible_inventory_path = (
        EXPERIMENT_ROOT / "manifests/eligible_model_input_inventory.private.csv"
    )
    visit_path.parent.mkdir(parents=True, exist_ok=True)
    visit_path.parent.chmod(0o700)
    atomic_text(visit_path, visits.to_csv(index=False), overwrite=args.overwrite)
    atomic_text(patient_path, patients.to_csv(index=False), overwrite=args.overwrite)
    eligible_ids = set(patients.loc[patients["eligible"], "patient_id"].astype(str))
    eligible_inventory = visits.loc[visits["patient_id"].isin(eligible_ids)].copy()
    atomic_text(
        eligible_inventory_path,
        eligible_inventory.to_csv(index=False),
        overwrite=args.overwrite,
    )

    exclusion_reasons = {
        str(key): int(value)
        for key, value in patients.loc[~patients["eligible"], "exclusion_reason"]
        .value_counts()
        .sort_index()
        .items()
    }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "run_is_new_and_does_not_amend_prior_no_go": True,
        "preregistered_before_eligibility_results": True,
        "preregistration_plan_sha256": lock["plan_sha256"],
        "eligibility_rule": "AND over T0,T1,T2,T3: valid_source_voxels > 0",
        "candidate_patients": int(len(patients)),
        "eligible_patients": int(patients["eligible"].sum()),
        "excluded_patients": int((~patients["eligible"]).sum()),
        "candidate_visits": int(len(visits)),
        "valid_visits": int(visits["has_valid_source_overlap"].sum()),
        "zero_overlap_visits": int((~visits["has_valid_source_overlap"]).sum()),
        "exclusion_reason_aggregate": exclusion_reasons,
        "minimum_valid_source_voxels": int(visits["valid_source_voxels"].min()),
        "eligibility_input_allowlist": [
            "imaging_source",
            "raw_or_rebuilt_source_geometry",
            "frozen_c1b_h_physical_grid",
            "valid_source_overlap",
        ],
        "lesion_ftv_ld_sph_bpe_fields_read": [],
        "outcome_pcr_fields_read": [],
        "clinical_treatment_subtype_fields_read": [],
        "model_loss_representation_performance_fields_read": [],
        "patient_specific_rules": [],
        "hardcoded_population_result": False,
        "private_manifest_sha256": {
            "technical_eligibility_patients.private.csv": sha256_file(patient_path),
            "technical_eligibility_visits.private.csv": sha256_file(visit_path),
            "eligible_model_input_inventory.private.csv": sha256_file(
                eligible_inventory_path
            ),
            "frozen_c1b_grids.private.csv": sha256_file(args.grids),
        },
        "source_provenance_sha256": {
            "candidate_inventory": sha256_file(args.inventory),
            "candidate_key_manifest": sha256_file(args.candidate_keys),
            "ispy1_visit_source_eligibility": sha256_file(args.ispy1_visits),
        },
        "contains_patient_identifiers": False,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    public_json = json_text(summary)
    report = f"""# Four-Visit Valid-Source-Overlap Technical Eligibility Amendment

## 结论

这是在独立 provenance audit 判定 `AUDIT-NOT-REPAIRABLE` 后启动的**新预注册 run**，不是对旧 `STAGE_A_NO_GO` 的 post-hoc 修改。eligibility rule 已在本轮任何 cohort 结果、representation、training 或 FTV probe 之前冻结，plan SHA-256 为 `{lock['plan_sha256']}`。

通用程序从完整 candidate model-input population 机械运行 `all candidates → four-visit AND → eligible population`，未写入已知失败 patient、预期分母或目标排除数。实际结果为：

- candidate patients：{summary['candidate_patients']}
- eligible patients：{summary['eligible_patients']}
- excluded patients：{summary['excluded_patients']}
- candidate visits：{summary['candidate_visits']}
- valid-source-overlap visits：{summary['valid_visits']}
- zero-overlap visits：{summary['zero_overlap_visits']}
- exclusion reasons：`{json.dumps(exclusion_reasons, sort_keys=True)}`

Eligibility runner 只读取 imaging source、raw/rebuilt source geometry、预先物化的 frozen C1B-H physical grid 与 exact valid-source overlap。它没有读取 FTV、LD、SPH、BPE、lesion response、pCR、clinical、treatment、subtype、model loss、representation metric 或 downstream performance。

逐 patient/visit 的 ID、source path、affine 与 exact count 仅保存在 private manifests；公开产物只包含聚合计数和 SHA-256。旧 `STAGE_A_NO_GO` 与旧 `AUDIT-NOT-REPAIRABLE` 保持不可变。本 amendment 只确定新 Stage-A population，并不单独授权 Stage B。
"""
    _assert_public_privacy([public_json, report], patient_ids)
    atomic_text(
        EXPERIMENT_ROOT / "metrics/technical_eligibility_summary.json",
        public_json,
        overwrite=args.overwrite,
    )
    atomic_text(
        EXPERIMENT_ROOT / "reports/technical_eligibility_amendment.md",
        report,
        overwrite=args.overwrite,
    )
    _make_figures(visits, patients)
    print(public_json, end="")


if __name__ == "__main__":
    main()
