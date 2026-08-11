#!/usr/bin/env python3
"""Materialize the already-frozen C1B-H physical grids for eligibility.

This is deliberately separate from the eligibility runner.  It recreates the
inherited T0 anchor/grid contract without evaluating any follow-up overlap or
eligibility outcome.  The downstream runner receives only grid geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

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

from c1b_overlap_stageb.io import (  # noqa: E402
    atomic_text,
    json_text,
    sha256_file,
    verify_preregistration,
    verify_upstream_contract,
)
from c1b_overlap_stageb.eligibility import frozen_grid_contract_sha256  # noqa: E402
from c1b_sanity.geometry import (  # noqa: E402
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
        default=PRIOR_ROOT / "manifests/model_input_inventory.private.csv",
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
    parser.add_argument(
        "--prior-cache-dir", type=Path, default=PRIOR_ROOT / "cache/c1b_h"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/frozen_c1b_grids.private.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _candidate_inventory(
    inventory_path: Path,
    candidate_key_path: Path,
) -> pd.DataFrame:
    # This grid-only program is the sole new entry point allowed to read the
    # frozen T0 localization path.  It reads no FTV value or follow-up support.
    inventory = pd.read_csv(
        inventory_path,
        usecols=[
            "patient_id",
            "cohort",
            "visit",
            "dce_nifti",
            "ftv_mask_nifti",
            "pixel_rebuild_required",
        ],
    )
    inventory["patient_id"] = inventory["patient_id"].astype(str)
    inventory["visit"] = inventory["visit"].astype(str)
    if inventory.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Candidate inventory has duplicate patient/visit rows")
    # Read only prior frozen cohort keys.  In particular, the old bbox overlap
    # statistic is neither loaded nor used to select the new eligible cohort.
    keys = pd.read_csv(
        candidate_key_path,
        usecols=["patient_id", "cohort", "visit", "anchor_provenance"],
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
        raise ValueError("Frozen candidate keys do not resolve in the source inventory")
    selected = selected.drop(columns="_merge")
    sizes = selected.groupby("patient_id")["visit"].agg(
        lambda values: (len(values), set(values))
    )
    if any(size != (len(VISITS), set(VISITS)) for size in sizes):
        raise ValueError("Candidate model-input population is not complete T0-T3")
    return selected


def _resolved_t0_sources(
    *,
    inventory: pd.DataFrame,
    ispy1_visit_path: Path,
    repair_dir: Path,
) -> dict[str, Path]:
    repairs: dict[tuple[str, str], Path] = {}
    for path in sorted(repair_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise ValueError("Inherited DICOM repair record is not PASS")
        key = (str(payload["patient_id"]), str(payload["visit"]))
        rebuilt = Path(payload["private"]["output_nifti"])
        if key in repairs or not rebuilt.is_file():
            raise ValueError("Inherited DICOM repair map is duplicate/incomplete")
        repairs[key] = rebuilt

    strict = pd.read_csv(
        ispy1_visit_path,
        usecols=["patient_id", "visit", "status", "rebuilt_nifti"],
    )
    strict["patient_id"] = strict["patient_id"].astype(str)
    strict["visit"] = strict["visit"].astype(str)
    if strict.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Inherited I-SPY1 visit table is duplicate")
    strict_map: dict[tuple[str, str], Path] = {}
    for row in strict.itertuples(index=False):
        if str(row.status) != "PASS":
            continue
        rebuilt = Path(str(row.rebuilt_nifti))
        if rebuilt.is_file():
            strict_map[(str(row.patient_id), str(row.visit))] = rebuilt

    output: dict[str, Path] = {}
    for row in inventory.loc[inventory["visit"].eq("T0")].itertuples(index=False):
        key = (str(row.patient_id), "T0")
        if str(row.cohort) == "I-SPY1":
            if key not in strict_map:
                raise ValueError("Candidate I-SPY1 T0 lacks its mandatory PASS rebuild")
            source = strict_map[key]
        elif bool(row.pixel_rebuild_required):
            if key not in repairs:
                raise ValueError("Required I-SPY2 T0 DICOM repair is missing")
            source = repairs[key]
        else:
            source = Path(str(row.dce_nifti))
        if not source.is_file():
            raise FileNotFoundError("A frozen T0 source is missing")
        output[str(row.patient_id)] = source
    return output


def _acquisition_center(path: Path) -> tuple[float, float, float]:
    image = nib.load(str(path), mmap=True)
    affine = validate_affine(image.affine, name="T0 source affine")
    shape = np.asarray(image.shape[:3], dtype=np.float64)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError("T0 source shape is invalid")
    center = nib.affines.apply_affine(affine, 0.5 * (shape - 1.0))
    return tuple(float(value) for value in center)


def _file_set_digest(paths: list[Path], *, domain: bytes) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _prior_cache_centers(cache_dir: Path) -> tuple[dict[str, np.ndarray], str]:
    output: dict[str, np.ndarray] = {}
    proof = hashlib.sha256(b"prior-schema3-c1b-h-cache-grid-proof-v1\0")
    for path in sorted(cache_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            schema = int(arrays["schema_version"].item())
            patient_id = str(arrays["patient_id"].item())
            strategy = str(arrays["registration_strategy"].item())
            builder = str(arrays["builder_contract_sha256"].item())
            input_provenance = str(arrays["input_provenance_sha256"].item())
            center = np.asarray(arrays["grid_center_ras_mm"], dtype=np.float64)
        if (
            schema != 3
            or strategy != "C1B-H"
            or builder != "bf2533d50a1e2bd13ae9b2f60f5118a5494060b8ab6955feb29c46a5d1ab1750"
            or len(input_provenance) != 64
            or any(character not in "0123456789abcdef" for character in input_provenance)
            or patient_id in output
            or center.shape != (3,)
            or not np.isfinite(center).all()
        ):
            raise ValueError("Prior cache grid-centre provenance is invalid")
        output[patient_id] = center
        stat = path.stat()
        record = {
            "builder_contract_sha256": builder,
            "cache_name": path.name,
            "file_size": int(stat.st_size),
            "input_provenance_sha256": input_provenance,
            "patient_id_sha256": hashlib.sha256(patient_id.encode("utf-8")).hexdigest(),
            "schema_version": schema,
            "strategy": strategy,
        }
        proof.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        proof.update(b"\0")
    return output, proof.hexdigest()


def main() -> None:
    args = parse_args()
    lock = verify_preregistration()
    verify_upstream_contract()
    inventory = _candidate_inventory(args.inventory, args.candidate_keys)
    sources = _resolved_t0_sources(
        inventory=inventory,
        ispy1_visit_path=args.ispy1_visits,
        repair_dir=args.repair_dir,
    )
    prior_centers, prior_cache_proof = _prior_cache_centers(args.prior_cache_dir)

    records: list[dict[str, Any]] = []
    matched_prior = 0
    for patient_id, group in inventory.groupby("patient_id", sort=True):
        t0 = group.loc[group["visit"].eq("T0")]
        if len(t0) != 1:
            raise ValueError("Every candidate must have one T0")
        row = t0.iloc[0]
        inherited_anchors = set(group["anchor_provenance"].astype(str))
        if len(inherited_anchors) != 1:
            raise ValueError("Inherited frozen anchor provenance changes across visits")
        inherited_anchor = next(iter(inherited_anchors))
        mask_text = row["ftv_mask_nifti"]
        mask_path = Path(str(mask_text)) if isinstance(mask_text, str) and mask_text else None
        if mask_path is not None and mask_path.is_file():
            support = load_nifti_ras(mask_path)
            if int(np.count_nonzero(support.data > 0.5)) > 0:
                center = support_bbox_center_ras(support)
                anchor = "inherited_released_t0_support_bbox_center"
                expected_anchor = "released_t0_support_bbox_center"
            else:
                center = _acquisition_center(sources[str(patient_id)])
                anchor = "inherited_t0_acquisition_physical_center_fallback"
                expected_anchor = "t0_acquisition_physical_center_fallback"
        else:
            center = _acquisition_center(sources[str(patient_id)])
            anchor = "inherited_t0_acquisition_physical_center_fallback"
            expected_anchor = "t0_acquisition_physical_center_fallback"
        if inherited_anchor != expected_anchor:
            raise ValueError("Recreated anchor category disagrees with prior frozen audit")
        grid = make_c1b_grid(center)
        cached = prior_centers.get(str(patient_id))
        if cached is not None:
            if not np.allclose(cached, np.asarray(center), atol=1e-8, rtol=0.0):
                raise ValueError("Recreated frozen grid disagrees with prior schema-3 cache")
            matched_prior += 1
        records.append(
            {
                "patient_id": str(patient_id),
                "cohort": str(row["cohort"]),
                "anchor_provenance": anchor,
                "grid_center_x_ras_mm": float(center[0]),
                "grid_center_y_ras_mm": float(center[1]),
                "grid_center_z_ras_mm": float(center[2]),
                "grid_shape_zyx_json": json.dumps(list(grid.shape_zyx)),
                "grid_spacing_xyz_mm_json": json.dumps(list(grid.spacing_xyz_mm)),
                "grid_affine_ras_json": json.dumps(grid.affine_ras.tolist()),
                "grid_contract_sha256": frozen_grid_contract_sha256(
                    patient_id=str(patient_id),
                    cohort=str(row["cohort"]),
                    grid_shape_zyx=grid.shape_zyx,
                    grid_spacing_xyz_mm=grid.spacing_xyz_mm,
                    grid_affine_ras=grid.affine_ras,
                ),
            }
        )
    grids = pd.DataFrame(records).sort_values("patient_id", kind="stable")
    if grids["patient_id"].duplicated().any() or set(prior_centers) - set(grids["patient_id"]):
        raise ValueError("Frozen-grid materialization has inconsistent patient coverage")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.chmod(0o700)
    atomic_text(args.output, grids.to_csv(index=False), overwrite=args.overwrite)

    public = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "materialize_inherited_frozen_c1b_h_grids_before_eligibility",
        "computes_eligibility": False,
        "evaluates_followup_overlap": False,
        "outcome_clinical_treatment_model_fields_read": [],
        "t0_localization_used_only_to_recreate_frozen_grid": True,
        "candidate_grid_rows": int(len(grids)),
        "prior_schema3_cache_grids_checked": int(len(prior_centers)),
        "prior_schema3_cache_grid_matches": int(matched_prior),
        "prior_schema3_cache_contract_proof_sha256": prior_cache_proof,
        "grid_shape_zyx": [112, 176, 160],
        "grid_spacing_xyz_mm": [0.9, 0.9, 2.0],
        "preregistration_plan_sha256": lock["plan_sha256"],
        "private_grid_manifest_sha256": sha256_file(args.output),
        "source_provenance_sha256": {
            "candidate_inventory": sha256_file(args.inventory),
            "candidate_key_manifest": sha256_file(args.candidate_keys),
            "ispy1_visit_source_eligibility": sha256_file(args.ispy1_visits),
            "repair_record_set": _file_set_digest(
                list(args.repair_dir.glob("*.json")),
                domain=b"c1b-repair-record-set-v1",
            ),
            "prior_builder_py": sha256_file(PRIOR_SRC / "c1b_sanity/builder.py"),
            "prior_dce7_py": sha256_file(PRIOR_SRC / "c1b_sanity/dce7.py"),
            "prior_geometry_py": sha256_file(PRIOR_SRC / "c1b_sanity/geometry.py"),
        },
        "contains_patient_identifiers": False,
    }
    atomic_text(
        EXPERIMENT_ROOT / "metrics/frozen_c1b_grid_materialization.json",
        json_text(public),
        overwrite=args.overwrite,
    )
    print(json_text(public), end="")


if __name__ == "__main__":
    main()
