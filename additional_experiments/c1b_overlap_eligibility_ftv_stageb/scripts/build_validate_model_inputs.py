#!/usr/bin/env python3
"""Build and validate the new-run, eligibility-scoped C1B-H DCE7 cache.

The old experiment is immutable.  Its hash-pinned ``c1b_sanity`` builder and
schema-3 cache implementation are imported read-only.  A valid old cache may
be made visible in this run with an atomic hard link, but this program never
opens an existing cache for writing and never replaces an existing cache.

This is a Stage-A program only.  It cannot authorize or launch Stage B.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
PRIOR_ROOT = REPO_ROOT / "additional_experiments/c1b_model_ready_ftv_sanity"
SRC_ROOT = EXPERIMENT_ROOT / "src"
PRIOR_SRC = PRIOR_ROOT / "src"
for source_root in (SRC_ROOT, PRIOR_SRC):
    if str(source_root) not in os.sys.path:
        os.sys.path.insert(0, str(source_root))

from c1b_overlap_stageb.eligibility import (  # noqa: E402
    VISITS,
    build_patient_eligibility,
    frozen_grid_contract_sha256,
)
from c1b_overlap_stageb.io import (  # noqa: E402
    atomic_text,
    json_text,
    sha256_file,
    verify_preregistration,
    verify_upstream_contract,
)
from c1b_sanity.builder import (  # noqa: E402
    VisitInput,
    build_patient_dce7,
    builder_contract_sha256,
)
from c1b_sanity.cache import (  # noqa: E402
    load_and_validate_cache,
    load_model_tensor,
    write_cache_atomic,
)
from c1b_sanity.dce7 import (  # noqa: E402
    DCE7_CHANNEL_NAMES,
    phase_metadata_sha256,
    select_phase_indices,
)
from c1b_sanity.geometry import (  # noqa: E402
    C1B_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    canonical_volume_sha256,
    load_nifti_ras,
)


LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eligibility-patients",
        type=Path,
        default=EXPERIMENT_ROOT
        / "manifests/technical_eligibility_patients.private.csv",
    )
    parser.add_argument(
        "--eligibility-visits",
        type=Path,
        default=EXPERIMENT_ROOT
        / "manifests/technical_eligibility_visits.private.csv",
    )
    parser.add_argument(
        "--eligible-inventory",
        type=Path,
        default=EXPERIMENT_ROOT
        / "manifests/eligible_model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--source-inventory",
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate and reuse existing new-run caches (default: enabled).",
    )
    parser.add_argument(
        "--reuse-prior-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate then atomically hard-link matching immutable old caches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite only generated CSV/JSON/report artifacts; never caches.",
    )
    return parser.parse_args()


def patient_token(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()


def _boolean_series(series: pd.Series, *, name: str) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    if not lowered.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"{name} must contain explicit booleans")
    return lowered.isin({"true", "1"})


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["patient_id"] = output["patient_id"].astype(str)
    if "visit" in output:
        output["visit"] = output["visit"].astype(str)
    if "cohort" in output:
        output["cohort"] = output["cohort"].astype(str)
    return output


def load_technical_eligibility(
    patient_path: Path,
    visit_path: Path,
    eligible_inventory_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and mechanically revalidate the new private eligibility outputs."""

    patients = _normalise_keys(
        pd.read_csv(
            patient_path,
            usecols=[
                "patient_id",
                "cohort",
                "candidate_visit_count",
                "valid_visit_count",
                "zero_overlap_visit_count",
                "minimum_valid_source_voxels",
                "eligible",
                "exclusion_reason",
            ],
        )
    )
    visits = _normalise_keys(
        pd.read_csv(
            visit_path,
            usecols=[
                "patient_id",
                "cohort",
                "visit",
                "resolved_dce_nifti",
                "grid_contract_sha256",
                "geometry_contract_sha256",
                "valid_source_voxels",
                "target_grid_voxels",
                "has_valid_source_overlap",
            ],
        )
    )
    if patients["patient_id"].duplicated().any():
        raise ValueError("Technical patient eligibility contains duplicate IDs")
    if visits.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Technical visit eligibility contains duplicate keys")
    patients["eligible"] = _boolean_series(patients["eligible"], name="eligible")
    visits["has_valid_source_overlap"] = _boolean_series(
        visits["has_valid_source_overlap"], name="has_valid_source_overlap"
    )
    visits["valid_source_voxels"] = pd.to_numeric(
        visits["valid_source_voxels"], errors="raise"
    ).astype(np.int64)
    visits["target_grid_voxels"] = pd.to_numeric(
        visits["target_grid_voxels"], errors="raise"
    ).astype(np.int64)
    if not visits["grid_contract_sha256"].astype(str).str.fullmatch(LOWER_SHA256).all():
        raise ValueError("Eligibility grid provenance is not SHA-256 closed")
    if not visits["geometry_contract_sha256"].astype(str).str.fullmatch(
        LOWER_SHA256
    ).all():
        raise ValueError("Eligibility geometry provenance is not SHA-256 closed")
    if not visits["has_valid_source_overlap"].eq(
        visits["valid_source_voxels"].gt(0)
    ).all():
        raise ValueError("Eligibility overlap flag disagrees with its exact count")

    recomputed = build_patient_eligibility(visits)
    stated = patients.sort_values("patient_id", kind="stable").reset_index(drop=True)
    columns = [
        "patient_id",
        "cohort",
        "candidate_visit_count",
        "valid_visit_count",
        "zero_overlap_visit_count",
        "minimum_valid_source_voxels",
        "eligible",
    ]
    pd.testing.assert_frame_equal(
        stated[columns],
        recomputed[columns],
        check_dtype=False,
        check_like=False,
    )
    eligible_ids = set(stated.loc[stated["eligible"], "patient_id"])
    eligible_visits = visits.loc[visits["patient_id"].isin(eligible_ids)].copy()
    if len(eligible_visits) != len(eligible_ids) * len(VISITS):
        raise ValueError("Eligible population is not exactly four visits per patient")
    if not eligible_visits["valid_source_voxels"].gt(0).all():
        raise ValueError("An eligible visit has zero valid-source voxels")

    frozen_inventory = _normalise_keys(
        pd.read_csv(
            eligible_inventory_path,
            usecols=[
                "patient_id",
                "cohort",
                "visit",
                "resolved_dce_nifti",
                "grid_contract_sha256",
                "geometry_contract_sha256",
                "valid_source_voxels",
                "target_grid_voxels",
            ],
        )
    )
    compare_columns = [
        "patient_id",
        "cohort",
        "visit",
        "resolved_dce_nifti",
        "grid_contract_sha256",
        "geometry_contract_sha256",
        "valid_source_voxels",
        "target_grid_voxels",
    ]
    left = eligible_visits[compare_columns].sort_values(
        ["patient_id", "visit"], kind="stable"
    ).reset_index(drop=True)
    right = frozen_inventory[compare_columns].sort_values(
        ["patient_id", "visit"], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    return stated, left


def verify_eligibility_hash_closure(
    patient_path: Path, visit_path: Path, eligible_inventory_path: Path
) -> dict[str, str]:
    summary_path = EXPERIMENT_ROOT / "metrics/technical_eligibility_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError("Technical eligibility did not PASS")
    expected = summary.get("private_manifest_sha256", {})
    observed = {
        patient_path.name: sha256_file(patient_path),
        visit_path.name: sha256_file(visit_path),
        eligible_inventory_path.name: sha256_file(eligible_inventory_path),
    }
    for name, digest in observed.items():
        if expected.get(name) != digest:
            raise ValueError(f"Technical eligibility manifest changed: {name}")
    return observed


def phase_map(path: Path) -> dict[str, dict[str, float | None]]:
    frame = pd.read_csv(path, usecols=["pid", "pre", "post_early", "post_late"])
    frame["pid"] = frame["pid"].astype(str)
    if frame["pid"].duplicated().any():
        raise ValueError("Acquisition phase table has duplicate patient IDs")
    output: dict[str, dict[str, float | None]] = {}
    for row in frame.itertuples(index=False):
        output[str(row.pid)] = {
            "pre": None if pd.isna(row.pre) else float(row.pre),
            "post_early": None if pd.isna(row.post_early) else float(row.post_early),
            "post_late": None if pd.isna(row.post_late) else float(row.post_late),
        }
    return output


def ispy1_phase_map(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    frame = _normalise_keys(
        pd.read_csv(
            path,
            usecols=[
                "patient_id",
                "visit",
                "status",
                "pre_index",
                "early_index",
                "late_index",
                "rebuilt_nifti",
            ],
        )
    )
    if frame.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Inherited I-SPY1 phase table contains duplicate visits")
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


def _optional_path(value: Any) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError("A required localization support is missing")
    return str(path.resolve())


def make_tasks(
    eligible_visits: pd.DataFrame,
    *,
    source_inventory_path: Path,
    phase_metadata_path: Path,
    ispy1_visit_path: Path,
    cache_dir: Path,
    prior_cache_dir: Path,
    resume: bool,
    reuse_prior_cache: bool,
) -> list[dict[str, Any]]:
    source = _normalise_keys(
        pd.read_csv(
            source_inventory_path,
            usecols=[
                "patient_id",
                "cohort",
                "visit",
                "formal_ftv_overlap",
                "ftv_mask_nifti",
            ],
        )
    )
    if source.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Inherited source inventory contains duplicate visits")
    source["formal_ftv_overlap"] = _boolean_series(
        source["formal_ftv_overlap"], name="formal_ftv_overlap"
    )
    merged = eligible_visits.merge(
        source,
        on=["patient_id", "cohort", "visit"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Eligible visits do not resolve in inherited source inventory")
    merged = merged.drop(columns="_merge")
    patient_phases = phase_map(phase_metadata_path)
    strict_phases = ispy1_phase_map(ispy1_visit_path)
    tasks: list[dict[str, Any]] = []
    for patient_id, group in merged.groupby("patient_id", sort=True):
        if len(group) != len(VISITS) or set(group["visit"]) != set(VISITS):
            raise ValueError("An eligible patient is not exactly T0-T3")
        cohorts = set(group["cohort"])
        formal_values = set(group["formal_ftv_overlap"].astype(bool))
        if len(cohorts) != 1 or len(formal_values) != 1:
            raise ValueError("Patient cohort/support scope changes across visits")
        cohort = next(iter(cohorts))
        formal = bool(next(iter(formal_values)))
        rows: dict[str, dict[str, Any]] = {}
        per_visit_phases: dict[str, Mapping[str, Any] | None] = {}
        for row in group.to_dict("records"):
            visit = str(row["visit"])
            source_path = Path(str(row["resolved_dce_nifti"]))
            if not source_path.is_file():
                raise FileNotFoundError("An eligible DCE source is missing")
            support = _optional_path(row.get("ftv_mask_nifti"))
            if visit != VISITS[0] and not formal:
                support = None
            if formal and support is None:
                raise ValueError("Formal FTV-overlap cache is missing a support mask")
            rows[visit] = {
                **row,
                "resolved_dce_nifti": str(source_path.resolve()),
                "support_nifti": support,
            }
            strict = strict_phases.get((str(patient_id), visit))
            per_visit_phases[visit] = (
                strict if strict is not None else patient_phases.get(str(patient_id))
            )
        token = patient_token(str(patient_id))
        tasks.append(
            {
                "patient_id": str(patient_id),
                "cohort": str(cohort),
                "formal_ftv_overlap": formal,
                "rows": rows,
                "phase_metadata_by_visit": per_visit_phases,
                "cache_path": str((cache_dir / f"{token}.npz").resolve()),
                "prior_cache_path": str(
                    (prior_cache_dir / f"{token}.npz").resolve()
                ),
                "resume": bool(resume),
                "reuse_prior_cache": bool(reuse_prior_cache),
            }
        )
    if not tasks:
        raise ValueError("Technical eligibility selected no patients")
    return tasks


def _hardlink_atomic(source: Path, destination: Path) -> None:
    """Install a read-only-validated file by link without writing its inode."""

    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing cache {destination}")
    if source.is_symlink() or not source.is_file():
        raise ValueError("Prior cache reuse requires a regular non-symlink file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".link", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.link(source, temporary, follow_symlinks=False)
        if not os.path.samefile(source, temporary):
            raise OSError("Prepared cache is not a hard link to the validated source")
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _current_source_hashes(task: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source_hashes: list[str] = []
    support_hashes: list[str] = []
    for visit in VISITS:
        row = task["rows"][visit]
        source_hashes.append(
            canonical_volume_sha256(load_nifti_ras(row["resolved_dce_nifti"]))
        )
        support = row["support_nifti"]
        support_hashes.append(
            "NONE"
            if support is None
            else canonical_volume_sha256(load_nifti_ras(support))
        )
    return np.asarray(source_hashes), np.asarray(support_hashes)


def _validate_cache_qc(
    path: Path,
    task: Mapping[str, Any],
    *,
    expected_image: np.ndarray | None = None,
    expected_source_hashes: np.ndarray | None = None,
    expected_support_hashes: np.ndarray | None = None,
) -> dict[str, Any]:
    arrays, validation = load_and_validate_cache(path, expected_image=expected_image)
    patient_id = str(task["patient_id"])
    cohort = str(task["cohort"])
    if validation.patient_id != patient_id:
        raise ValueError("Cache identity disagrees with its tokenized destination")
    if str(arrays["cohort"].item()) != cohort:
        raise ValueError("Cache cohort disagrees with technical eligibility")
    if str(arrays["registration_strategy"].item()) != "C1B-H":
        raise ValueError("Only the frozen C1B-H cache is allowed")
    if bool(arrays["formal_ftv_overlap"].item()) != bool(
        task["formal_ftv_overlap"]
    ):
        raise ValueError("Cache support scope disagrees with inherited inventory")

    expected_counts = np.asarray(
        [int(task["rows"][visit]["valid_source_voxels"]) for visit in VISITS],
        dtype=np.int64,
    )
    observed_counts = np.asarray(arrays["valid_source_mask"], dtype=np.uint8).reshape(
        len(VISITS), -1
    ).sum(axis=1, dtype=np.int64)
    count_matches = observed_counts == expected_counts
    if not count_matches.all():
        raise ValueError("Cache valid-source mask count disagrees with eligibility")
    if np.any(observed_counts <= 0):
        raise ValueError("Eligible cache contains a catastrophic zero-overlap visit")

    grid_digest = frozen_grid_contract_sha256(
        patient_id=patient_id,
        cohort=cohort,
        grid_shape_zyx=arrays["grid_shape_zyx"],
        grid_spacing_xyz_mm=arrays["grid_spacing_xyz_mm"],
        grid_affine_ras=arrays["grid_affine_ras"],
    )
    expected_grid_digests = {
        str(task["rows"][visit]["grid_contract_sha256"]) for visit in VISITS
    }
    grid_match = expected_grid_digests == {grid_digest}
    if not grid_match:
        raise ValueError("Cache physical grid disagrees with frozen eligibility grid")

    phase_counts = np.asarray(arrays["phase_counts"], dtype=np.int64)
    phase_indices = np.asarray(arrays["phase_indices"], dtype=np.int64)
    expected_phase_indices = np.asarray(
        [
            select_phase_indices(
                int(phase_counts[index]), task["phase_metadata_by_visit"][visit]
            ).indices
            for index, visit in enumerate(VISITS)
        ],
        dtype=np.int64,
    )
    expected_phase_hashes = np.asarray(
        [
            phase_metadata_sha256(task["phase_metadata_by_visit"][visit])
            for visit in VISITS
        ]
    )
    phase_match = bool(
        np.array_equal(phase_indices, expected_phase_indices)
        and np.array_equal(arrays["phase_metadata_sha256"], expected_phase_hashes)
        and tuple(str(value) for value in arrays["channel_names"])
        == tuple(DCE7_CHANNEL_NAMES)
        and tuple(str(value) for value in arrays["visits"]) == tuple(VISITS)
    )
    if not phase_match:
        raise ValueError("Cache phase/DCE7 semantics disagree with frozen contract")

    if expected_source_hashes is None or expected_support_hashes is None:
        expected_source_hashes, expected_support_hashes = _current_source_hashes(task)
    source_match = bool(
        np.array_equal(arrays["source_canonical_sha256"], expected_source_hashes)
        and np.array_equal(arrays["support_canonical_sha256"], expected_support_hashes)
    )
    provenance_match = bool(
        source_match
        and str(arrays["builder_contract_sha256"].item())
        == builder_contract_sha256()
        and LOWER_SHA256.fullmatch(str(arrays["input_provenance_sha256"].item()))
    )
    if not provenance_match:
        raise ValueError("Cache does not match current hash-pinned input provenance")

    image = np.asarray(arrays["image"])
    model_image = load_model_tensor(path)
    loader_only_image = bool(
        isinstance(model_image, np.ndarray)
        and model_image.dtype == np.float32
        and np.array_equal(model_image, image)
    )
    if not loader_only_image:
        raise ValueError("Model loader did not return exactly the image tensor")
    nonconstant_by_visit = []
    for visit_index in range(len(VISITS)):
        nonconstant_by_visit.append(
            any(float(np.ptp(image[visit_index, channel])) > 1e-6 for channel in range(3))
        )
    shape_match = tuple(image.shape) == (
        len(VISITS),
        len(DCE7_CHANNEL_NAMES),
        *C1B_SHAPE_ZYX,
    )
    orientation_match = bool(
        np.allclose(
            np.diag(np.asarray(arrays["grid_affine_ras"]))[:3],
            np.asarray(C1B_SPACING_XYZ_MM),
            atol=0.0,
            rtol=0.0,
        )
    )
    output = {
        "cache_content_sha256": validation.content_sha256,
        "cache_file_sha256": validation.file_sha256,
        "builder_contract_sha256": str(arrays["builder_contract_sha256"].item()),
        "input_provenance_sha256": str(arrays["input_provenance_sha256"].item()),
        "eligibility_valid_source_voxels_json": json.dumps(expected_counts.tolist()),
        "cache_valid_source_voxels_json": json.dumps(observed_counts.tolist()),
        "valid_source_count_matches_json": json.dumps(count_matches.tolist()),
        "exact_valid_source_visit_matches": int(count_matches.sum()),
        "exact_roundtrip_pass": loader_only_image,
        "finite": bool(np.isfinite(image).all()),
        "nonconstant": bool(all(nonconstant_by_visit)),
        "shape_match": bool(shape_match),
        "orientation_match": orientation_match,
        "phase_contract_match": phase_match,
        "grid_contract_match": bool(grid_match),
        "provenance_match": provenance_match,
        "model_loader_returns_only_image": loader_only_image,
        "zero_overlap_visits": int(np.count_nonzero(observed_counts <= 0)),
        "cache_schema_version": int(arrays["schema_version"].item()),
    }
    required = [
        output["exact_roundtrip_pass"],
        output["finite"],
        output["nonconstant"],
        output["shape_match"],
        output["orientation_match"],
        output["phase_contract_match"],
        output["grid_contract_match"],
        output["provenance_match"],
        output["model_loader_returns_only_image"],
        output["exact_valid_source_visit_matches"] == len(VISITS),
        output["zero_overlap_visits"] == 0,
    ]
    if not all(required):
        raise ValueError("Cache failed a required model-input QC")
    return output


def build_one(task: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    patient_id = str(task["patient_id"])
    destination = Path(str(task["cache_path"]))
    prior = Path(str(task["prior_cache_path"]))
    origin: str
    if destination.exists():
        if not task["resume"]:
            raise FileExistsError("Cache exists and --no-resume was requested")
        qc = _validate_cache_qc(destination, task)
        origin = "resumed_new_run_cache"
    elif task["reuse_prior_cache"] and prior.is_file():
        # All validation is read-only and precedes link creation.  The linked
        # inode is never passed to write_cache_atomic.
        qc = _validate_cache_qc(prior, task)
        _hardlink_atomic(prior, destination)
        if not os.path.samefile(prior, destination):
            raise OSError("New-run cache did not retain the validated hard link")
        origin = "validated_prior_cache_hardlink"
    else:
        visits: dict[str, VisitInput] = {}
        for visit in VISITS:
            row = task["rows"][visit]
            visits[visit] = VisitInput(
                visit=visit,
                dce=row["resolved_dce_nifti"],
                phase_metadata=task["phase_metadata_by_visit"][visit],
                localization_support=row["support_nifti"],
                source_to_anchor_ras=None,
            )
        patient = build_patient_dce7(
            patient_id,
            visits,
            cohort=str(task["cohort"]),
            formal_ftv_overlap=bool(task["formal_ftv_overlap"]),
            registration_strategy="C1B-H",
        )
        if destination.exists():
            raise FileExistsError("Concurrent process created the cache destination")
        write_cache_atomic(destination, patient)
        qc = _validate_cache_qc(
            destination,
            task,
            expected_image=patient.image,
            expected_source_hashes=patient.source_canonical_sha256,
            expected_support_hashes=patient.support_canonical_sha256,
        )
        origin = "built_from_eligible_sources"
    return {
        "patient_id": patient_id,
        "cohort": str(task["cohort"]),
        "status": "PASS",
        "cache_path": str(destination.resolve()),
        "cache_origin": origin,
        "failure_type": "",
        "failure_message": "",
        "elapsed_seconds": float(time.monotonic() - started),
        **qc,
    }


def _safe_build_one(task: dict[str, Any]) -> dict[str, Any]:
    try:
        return build_one(task)
    except Exception as error:  # private row; public output receives a count only
        return {
            "patient_id": str(task["patient_id"]),
            "cohort": str(task["cohort"]),
            "status": "FAIL",
            "cache_path": str(Path(str(task["cache_path"])).resolve()),
            "cache_origin": "",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "elapsed_seconds": np.nan,
        }


def _fraction(metrics: pd.DataFrame, column: str) -> float:
    values = metrics.get(column, pd.Series(False, index=metrics.index)).fillna(False)
    return float(values.astype(bool).mean())


def build_public_gate(
    metrics: pd.DataFrame,
    *,
    eligible_visits: int,
    private_metrics_sha256: str,
    cache_inventory_sha256: str,
    eligibility_manifest_sha256: Mapping[str, str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    eligible_patients = int(len(metrics))
    passed = metrics["status"].eq("PASS")
    completed_patients = int(passed.sum())
    completed_visits = completed_patients * len(VISITS)
    exact_visit_matches = int(
        pd.to_numeric(
            metrics.get(
                "exact_valid_source_visit_matches", pd.Series(0, index=metrics.index)
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    all_qc = all(
        _fraction(metrics, column) == 1.0
        for column in (
            "exact_roundtrip_pass",
            "finite",
            "nonconstant",
            "shape_match",
            "orientation_match",
            "phase_contract_match",
            "grid_contract_match",
            "provenance_match",
            "model_loader_returns_only_image",
        )
    )
    complete = (
        eligible_patients > 0
        and completed_patients == eligible_patients
        and completed_visits == int(eligible_visits)
        and exact_visit_matches == int(eligible_visits)
    )
    status = "PASS" if complete and all_qc else "FAIL"
    origins = metrics.loc[passed, "cache_origin"].value_counts()
    failure_types = metrics.get(
        "failure_type", pd.Series("", index=metrics.index, dtype=str)
    )
    failures = failure_types.loc[~passed].replace("", "UnknownError")
    return {
        "schema_version": 1,
        "status": status,
        "strategy": "C1B-H",
        "scope": "all",
        "cache_schema_version": 3,
        "builder_contract_sha256": builder_contract_sha256(),
        "eligible_patients": eligible_patients,
        "eligible_visits": int(eligible_visits),
        "completed_patients": completed_patients,
        "completed_visits": completed_visits,
        "completion_fraction": float(completed_patients / eligible_patients),
        "exact_valid_source_count_match_fraction": float(
            exact_visit_matches / int(eligible_visits)
        ),
        "exact_roundtrip_pass_fraction": _fraction(metrics, "exact_roundtrip_pass"),
        "finite_fraction": _fraction(metrics, "finite"),
        "nonconstant_fraction": _fraction(metrics, "nonconstant"),
        "shape_match_fraction": _fraction(metrics, "shape_match"),
        "orientation_match_fraction": _fraction(metrics, "orientation_match"),
        "phase_contract_match_fraction": _fraction(metrics, "phase_contract_match"),
        "grid_contract_match_fraction": _fraction(metrics, "grid_contract_match"),
        "provenance_match_fraction": _fraction(metrics, "provenance_match"),
        "model_loader_returns_only_image": _fraction(
            metrics, "model_loader_returns_only_image"
        )
        == 1.0,
        "sidecars_are_model_inputs": False,
        "geometry_metadata_is_model_input": False,
        "reused_prior_cache_patients": int(
            origins.get("validated_prior_cache_hardlink", 0)
        ),
        "resumed_cache_patients": int(origins.get("resumed_new_run_cache", 0)),
        "newly_built_patients": int(origins.get("built_from_eligible_sources", 0)),
        "unresolved_catastrophic_resampling_cases": int(
            (~passed).sum()
            + pd.to_numeric(
                metrics.get("zero_overlap_visits", pd.Series(0, index=metrics.index)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "patient_specific_manual_corrections": [],
        "registration_transform_used_for_repair": False,
        "failure_type_aggregate": {
            str(key): int(value) for key, value in failures.value_counts().items()
        },
        "private_metrics_sha256": private_metrics_sha256,
        "cache_inventory_sha256": cache_inventory_sha256,
        "eligibility_manifest_sha256": dict(eligibility_manifest_sha256),
        "contains_patient_identifiers": False,
        "stage_b_authorized": False,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _assert_public_privacy(payloads: Iterable[str], patient_ids: Iterable[str]) -> None:
    identifiers = tuple(str(value) for value in patient_ids if str(value))
    for payload in payloads:
        if any(identifier in payload for identifier in identifiers):
            raise ValueError("A public cache artifact contains a patient identifier")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    verify_preregistration()
    inherited = verify_upstream_contract()
    if inherited["builder_semantic_contract_sha256"] != builder_contract_sha256():
        raise ValueError("Imported builder does not match the hash-pinned contract")
    eligibility_hashes = verify_eligibility_hash_closure(
        args.eligibility_patients,
        args.eligibility_visits,
        args.eligible_inventory,
    )
    patients, eligible_visits = load_technical_eligibility(
        args.eligibility_patients,
        args.eligibility_visits,
        args.eligible_inventory,
    )
    eligible_ids = set(patients.loc[patients["eligible"], "patient_id"])
    tasks = make_tasks(
        eligible_visits,
        source_inventory_path=args.source_inventory,
        phase_metadata_path=args.phase_metadata,
        ispy1_visit_path=args.ispy1_visits,
        cache_dir=args.cache_dir,
        prior_cache_dir=args.prior_cache_dir,
        resume=args.resume,
        reuse_prior_cache=args.reuse_prior_cache,
    )
    if {str(task["patient_id"]) for task in tasks} != eligible_ids:
        raise ValueError("Cache tasks do not exactly equal the eligible patient set")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_safe_build_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "completed_patients": completed,
                            "eligible_patients": len(futures),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    metrics = pd.DataFrame(records).sort_values("patient_id", kind="stable")
    if set(metrics["patient_id"].astype(str)) != eligible_ids or len(metrics) != len(
        eligible_ids
    ):
        raise ValueError("Private cache metrics do not exactly cover eligible patients")

    metrics_path = EXPERIMENT_ROOT / "metrics/model_input_pipeline_h_all.private.csv"
    inventory_path = EXPERIMENT_ROOT / "manifests/model_input_cache_inventory.private.csv"
    for column in (
        "cache_file_sha256",
        "cache_content_sha256",
        "builder_contract_sha256",
        "input_provenance_sha256",
    ):
        if column not in metrics:
            metrics[column] = ""
    atomic_text(metrics_path, metrics.to_csv(index=False), overwrite=args.overwrite)
    cache_inventory = metrics[
        [
            "patient_id",
            "cohort",
            "status",
            "cache_path",
            "cache_origin",
            "cache_file_sha256",
            "cache_content_sha256",
            "builder_contract_sha256",
            "input_provenance_sha256",
        ]
    ].copy()
    atomic_text(
        inventory_path, cache_inventory.to_csv(index=False), overwrite=args.overwrite
    )
    public = build_public_gate(
        metrics,
        eligible_visits=len(eligible_visits),
        private_metrics_sha256=sha256_file(metrics_path),
        cache_inventory_sha256=sha256_file(inventory_path),
        eligibility_manifest_sha256=eligibility_hashes,
        elapsed_seconds=time.monotonic() - started,
    )
    public_text = json_text(public)
    report = f"""# C1B-H model-input cache validation

## 结论

本程序只处理本轮 technical eligibility private manifests 机械确定的 eligible population。完成 {public['completed_patients']}/{public['eligible_patients']} 人、{public['completed_visits']}/{public['eligible_visits']} visit；cache completion={public['completion_fraction']:.6f}，逐 visit exact valid-source count match={public['exact_valid_source_count_match_fraction']:.6f}，结论为 **{public['status']}**。

旧实验 builder/cache 由 upstream SHA-256 lock 固定。旧 cache 仅在完整只读 schema/hash、当前 source/support、phase、grid、provenance 与 eligibility exact-count 验证后，以原子 hard link 放入本轮 `cache/c1b_h`；程序从不原地写或替换既有 cache inode。其余 patient 从 eligible imaging sources 构建。model loader 只返回 float32 DCE7 image；valid-source mask、affine、grid、phase、support 与 provenance 均为 sidecar，不进入 model tensor。

本文件不授权也不启动 Stage B；最终 Stage-A GO/NO-GO 由独立 finalizer 汇总全部预注册 gate 后决定。
"""
    _assert_public_privacy([public_text, report], eligible_ids)
    atomic_text(
        EXPERIMENT_ROOT / "metrics/model_input_pipeline_h_all_gate.json",
        public_text,
        overwrite=args.overwrite,
    )
    atomic_text(
        EXPERIMENT_ROOT / "reports/model_input_pipeline_validation.md",
        report,
        overwrite=args.overwrite,
    )
    print(public_text, end="")
    if public["status"] != "PASS":
        raise RuntimeError("Eligible C1B-H cache is incomplete or failed QC")


if __name__ == "__main__":
    main()
