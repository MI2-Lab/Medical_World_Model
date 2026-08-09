#!/usr/bin/env python3
"""Create private row-level and public aggregate Stage-A manifests.

The public files intentionally contain no patient identifiers or local paths.
Clinical/treatment/pCR columns are never requested from their source tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
VISITS = ("T0", "T1", "T2", "T3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("C1B_DATA_ROOT", "/data/data/Preprocessed")),
    )
    parser.add_argument(
        "--prior-contract-table",
        type=Path,
        default=REPO_ROOT
        / "additional_experiments/response_observable_multiscale_crop/metrics/patient_visit_contracts.csv",
    )
    parser.add_argument(
        "--overlap",
        type=Path,
        default=REPO_ROOT
        / "additional_experiments/radiomics_next_change/data_audit/radiomics_patient_overlap.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def finite_affine(affine: np.ndarray) -> bool:
    return bool(
        affine.shape == (4, 4)
        and np.isfinite(affine).all()
        and np.linalg.matrix_rank(affine[:3, :3]) == 3
        and np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
    )


def manifest_ids(data_root: Path, cohort: str) -> list[str]:
    labels = data_root / cohort / "clinical_labels_complete4visits.csv"
    if cohort == "I-SPY2":
        frame = pd.read_csv(labels, usecols=["patient_id"])
    else:
        frame = pd.read_csv(labels, usecols=["patient_id", "complete_4visits"])
        frame = frame.loc[frame["complete_4visits"].astype(bool)]
    ids = frame["patient_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{cohort} label table contains duplicate patient IDs")
    return ids


def overlap_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, usecols=["patient_id", "has_radiomics"])
    selected = frame.loc[frame["has_radiomics"].astype(bool), "patient_id"].astype(str)
    if selected.duplicated().any():
        raise ValueError("FTV overlap table contains duplicate selected patient IDs")
    return set(selected)


def source_edge_table(path: Path, expected_ids: set[str]) -> pd.DataFrame:
    columns = [
        "patient_id",
        "visit",
        "contract",
        "view",
        "source_boundary_touch",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.loc[frame["contract"].eq("C1B") & frame["view"].eq("detail")].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Prior C1B contract table has duplicate patient/visit rows")
    if set(frame["patient_id"]) != expected_ids or len(frame) != len(expected_ids) * 4:
        raise ValueError("Prior C1B contract table does not exactly cover the formal FTV cohort")
    frame["source_boundary_touch"] = frame["source_boundary_touch"].astype(bool)
    return frame[["patient_id", "visit", "source_boundary_touch"]]


def raw_series_exists(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_dir()
    if isinstance(value, list) and value:
        return all(Path(item).is_dir() for item in value)
    return False


def build_inventory(data_root: Path, overlap: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in ("I-SPY2", "I-SPY1"):
        for patient_id in manifest_ids(data_root, cohort):
            manifest_path = data_root / cohort / patient_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            visits = manifest.get("visits", [])
            if [item.get("visit") for item in visits] != list(VISITS):
                raise ValueError(f"Noncanonical visit sequence in private manifest for {cohort}")
            for visit in visits:
                dce_path = Path(visit["dce_nifti"])
                if not dce_path.is_file():
                    raise FileNotFoundError("A private DCE NIfTI path is missing")
                image = nib.load(str(dce_path), mmap=False)
                affine = np.asarray(image.affine, dtype=np.float64)
                affine_valid = finite_affine(affine)
                orientation = "UNKNOWN" if not affine_valid else "".join(nib.aff2axcodes(affine))
                ftv_path = visit.get("ftv_mask_nifti")
                has_support = bool(
                    isinstance(ftv_path, str)
                    and Path(ftv_path).is_file()
                    and int(visit.get("ftv_voxels") or 0) > 0
                )
                rows.append(
                    {
                        "patient_id": patient_id,
                        "cohort": cohort,
                        "visit": str(visit["visit"]),
                        "formal_ftv_overlap": patient_id in overlap,
                        "dce_nifti": str(dce_path.resolve()),
                        "ftv_mask_nifti": str(Path(ftv_path).resolve()) if isinstance(ftv_path, str) else "",
                        "raw_dce_series_json": json.dumps(visit.get("raw_dce_series")),
                        "raw_series_exists": raw_series_exists(visit.get("raw_dce_series")),
                        "phase_count": int(visit.get("n_times") or image.shape[-1]),
                        "shape_json": json.dumps([int(value) for value in image.shape]),
                        "affine_valid": affine_valid,
                        "orientation_before": orientation,
                        "pixel_rebuild_required": cohort == "I-SPY2" and not affine_valid,
                        "t0_anchor_source": (
                            "released_t0_support"
                            if str(visit["visit"]) == "T0" and has_support
                            else "acquisition_center_fallback"
                            if str(visit["visit"]) == "T0"
                            else "reuse_t0_grid"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    overlap = overlap_ids(args.overlap)
    if len(overlap) != 375:
        raise ValueError(f"Expected 375 formal FTV patients, found {len(overlap)}")
    edges = source_edge_table(args.prior_contract_table, overlap)
    edges["ftv_measurement_valid"] = True
    edges["grounding_observable_mask"] = ~edges["source_boundary_touch"]
    edges["ineligibility_reason"] = np.where(
        edges["source_boundary_touch"], "SOURCE_ACQUISITION_BOUNDARY_TOUCH", ""
    )
    edges = edges.sort_values(["patient_id", "visit"], kind="stable").reset_index(drop=True)

    inventory = build_inventory(args.data_root, overlap)
    expected = {"I-SPY2": 808 * 4, "I-SPY1": 156 * 4}
    observed = inventory["cohort"].value_counts().to_dict()
    if observed != expected:
        raise ValueError(f"Model-input inventory count mismatch: {observed} != {expected}")
    if not inventory["raw_series_exists"].all():
        raise FileNotFoundError("At least one model-input visit lacks its raw series")

    private_grounding = EXPERIMENT_ROOT / "manifests/grounding_observability_manifest.private.csv"
    private_inventory = EXPERIMENT_ROOT / "manifests/model_input_inventory.private.csv"
    atomic_text(private_grounding, edges.to_csv(index=False), args.overwrite)
    atomic_text(private_inventory, inventory.to_csv(index=False), args.overwrite)

    edge_by_visit = (
        edges.groupby("visit", sort=False)["source_boundary_touch"].sum().astype(int).to_dict()
    )
    grounding_summary = {
        "schema_version": 1,
        "definition": "1 unless FTV inclusion support touches any source acquisition voxel face",
        "scope": "grounding_loss_eligibility_only",
        "is_model_input": False,
        "does_not_filter_base_training": True,
        "formal_patients": 375,
        "formal_visits": 1500,
        "observable_visits": int(edges["grounding_observable_mask"].sum()),
        "ineligible_visits": int(edges["source_boundary_touch"].sum()),
        "affected_patients": int(edges.loc[edges["source_boundary_touch"], "patient_id"].nunique()),
        "ineligible_by_visit": {visit: int(edge_by_visit.get(visit, 0)) for visit in VISITS},
        "private_manifest_sha256": sha256_file(private_grounding),
        "contains_patient_identifiers": False,
    }
    atomic_text(
        EXPERIMENT_ROOT / "manifests/grounding_observability_summary.json",
        json.dumps(grounding_summary, indent=2, sort_keys=True) + "\n",
        args.overwrite,
    )

    invalid = inventory.loc[inventory["pixel_rebuild_required"]]
    formal_invalid = invalid.loc[invalid["formal_ftv_overlap"]]
    orientation_counts = Counter(inventory["orientation_before"].astype(str))
    phase_counts: dict[str, dict[str, int]] = {}
    for cohort, subset in inventory.groupby("cohort", sort=False):
        phase_counts[str(cohort)] = {
            str(int(key)): int(value)
            for key, value in subset["phase_count"].value_counts().sort_index().items()
        }
    inventory_summary = {
        "schema_version": 1,
        "patients": {"I-SPY2": 808, "I-SPY1": 156, "total": 964},
        "visits": {"I-SPY2": 3232, "I-SPY1": 624, "total": 3856},
        "formal_ftv_visits": 1500,
        "singular_model_input_visits": int(len(invalid)),
        "singular_formal_ftv_visits": int(len(formal_invalid)),
        "singular_base_only_extension_visits": int(len(invalid) - len(formal_invalid)),
        "orientation_before": dict(sorted(orientation_counts.items())),
        "phase_count_by_cohort": phase_counts,
        "t0_anchor_patients": {
            "released_support": int(
                inventory["t0_anchor_source"].eq("released_t0_support").sum()
            ),
            "acquisition_center_fallback": int(
                inventory["t0_anchor_source"].eq("acquisition_center_fallback").sum()
            ),
        },
        "private_inventory_sha256": sha256_file(private_inventory),
        "contains_patient_identifiers": False,
        "clinical_treatment_pcr_columns_read": False,
    }
    atomic_text(
        EXPERIMENT_ROOT / "manifests/model_input_inventory_summary.json",
        json.dumps(inventory_summary, indent=2, sort_keys=True) + "\n",
        args.overwrite,
    )
    print(json.dumps({"grounding": grounding_summary, "inventory": inventory_summary}, indent=2))


if __name__ == "__main__":
    main()
