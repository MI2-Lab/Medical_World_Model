#!/usr/bin/env python3
"""Build production-like C1B DCE7 caches and validate exact round-trips.

The model-facing loader returns only ``image [4,7,112,176,160]``.  Every
affine, transform, valid-source mask, phase index, support metric, and patient
identifier remains a private sidecar.  Phase metadata is read with an explicit
three-field acquisition-only allowlist; no outcome-bearing column is loaded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

from c1b_sanity.builder import (  # noqa: E402
    VISITS,
    VisitInput,
    build_patient_dce7,
    builder_contract_sha256,
)
from c1b_sanity.cache import (  # noqa: E402
    content_sha256,
    load_and_validate_cache,
    load_model_tensor,
    write_cache_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("validation", "all"), default="validation")
    parser.add_argument("--strategy", choices=("H", "R"), default="H")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated manifests/metrics/reports; existing valid caches are reused.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Independently rebuild selected patient caches instead of validating/reusing them.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--phase-metadata",
        type=Path,
        default=REPO_ROOT
        / "ispy_jepa_tmi_clean/data_processing/metadata/BreastDCEDL_metadata_min_crop.csv",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path(
            "/data/data/Preprocessed/I-SPY2/"
            "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
            "matched_patient_cv_splits_seed2026.csv"
        ),
    )
    parser.add_argument(
        "--registration-transforms",
        type=Path,
        default=None,
        help="Private CSV with patient_id, visit, status, and m00..m33; required for R.",
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
    return parser.parse_args()


def atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def patient_token(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catastrophic_source_overlap_preflight(
    inventory: pd.DataFrame,
    *,
    validation_ids: set[str],
    reasons: dict[str, set[str]],
    scope: str,
    strategy: str,
    overwrite: bool,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> None:
    """Fail closed before cache work when a frozen visit has zero grid overlap.

    The orientation/resampling audit is a header-only audit over the entire
    frozen model population.  Exact key/cohort coverage is checked before its
    padding statistic is trusted.  Public outputs contain counts and hashes
    only; patient keys and cache paths remain in private CSVs.
    """

    audit_path = experiment_root / "metrics/orientation_resampling_patient_visit.private.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(
            "Complete header-only orientation/resampling audit is missing"
        )
    audit = pd.read_csv(
        audit_path,
        usecols=[
            "patient_id",
            "visit",
            "cohort",
            "padding_fraction_bbox",
        ],
    )
    expected = inventory[["patient_id", "visit", "cohort"]].copy()
    for frame in (audit, expected):
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["visit"] = frame["visit"].astype(str)
        frame["cohort"] = frame["cohort"].astype(str)
    if audit.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Header-only overlap audit has duplicate patient/visit rows")
    if expected.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Frozen selected inventory has duplicate patient/visit rows")
    coverage = expected.merge(
        audit,
        on=["patient_id", "visit"],
        how="outer",
        validate="one_to_one",
        suffixes=("_expected", "_audit"),
        indicator=True,
    )
    if (
        len(expected) != 3792
        or expected["patient_id"].nunique() != 948
        or not coverage["_merge"].eq("both").all()
        or not coverage["cohort_expected"].eq(coverage["cohort_audit"]).all()
        or not np.isfinite(audit["padding_fraction_bbox"]).all()
        or not audit["padding_fraction_bbox"].between(0.0, 1.0).all()
    ):
        raise ValueError(
            "Header-only overlap audit does not exactly cover the frozen 948x4 population"
        )

    zero_overlap = audit.loc[audit["padding_fraction_bbox"].eq(1.0)].copy()
    if zero_overlap.empty:
        return

    chosen_ids = (
        set(map(str, validation_ids))
        if scope == "validation"
        else set(expected["patient_id"])
    )
    selected_inventory = expected.loc[expected["patient_id"].isin(chosen_ids)]
    if (
        selected_inventory["patient_id"].nunique() != len(chosen_ids)
        or len(selected_inventory) != 4 * len(chosen_ids)
    ):
        raise ValueError("Chosen cache scope is not an exact four-visit patient cohort")

    cache_root = experiment_root / "cache" / f"c1b_{strategy.lower()}"
    selection = (
        expected.drop_duplicates("patient_id")[["patient_id", "cohort"]]
        .loc[lambda frame: frame["patient_id"].isin(chosen_ids)]
        .copy()
    )
    selection["selected_for_validation"] = selection["patient_id"].isin(validation_ids)
    selection["selection_reasons"] = selection["patient_id"].map(
        lambda value: "|".join(sorted(reasons.get(str(value), {"all_model_inputs"})))
    )
    selection["cache_path"] = selection["patient_id"].map(
        lambda value: str((cache_root / f"{patient_token(str(value))}.npz").resolve())
    )
    selection = selection.sort_values("patient_id", kind="stable")

    failures = zero_overlap.loc[
        zero_overlap["patient_id"].isin(chosen_ids),
        ["patient_id", "cohort", "visit", "padding_fraction_bbox"],
    ].copy()
    # A catastrophic visit anywhere in the frozen population blocks both the
    # validation closure and every later all-scope/cache action.  Retain the
    # private failing key even if deterministic validation sampling would not
    # otherwise have selected it.
    if failures.empty:
        failures = zero_overlap[
            ["patient_id", "cohort", "visit", "padding_fraction_bbox"]
        ].copy()
    failures["failure_code"] = "ZERO_VALID_SOURCE_OVERLAP"
    failures["valid_source_voxels"] = 0
    failures["target_grid_voxels"] = 112 * 176 * 160
    failures["evidence"] = "header_padding_fraction_bbox_eq_1_and_builder_zero_valid_source"
    failures = failures.sort_values(["patient_id", "visit"], kind="stable")

    selection_path = experiment_root / "manifests/cache_validation_selection.private.csv"
    failure_path = (
        experiment_root
        / "metrics"
        / f"model_input_pipeline_{strategy.lower()}_{scope}_failures.private.csv"
    )
    atomic_text(selection_path, selection.to_csv(index=False), overwrite=overwrite)
    atomic_text(failure_path, failures.to_csv(index=False), overwrite=overwrite)
    selected_cache_count = int(selection["cache_path"].map(lambda value: Path(value).is_file()).sum())
    public = {
        "schema_version": 1,
        "cache_schema_version": 3,
        "builder_contract_sha256": builder_contract_sha256(),
        "strategy": f"C1B-{strategy}",
        "scope": scope,
        "status": "FAIL",
        "failure_stage": "catastrophic_source_overlap_preflight",
        "failure_codes": {"ZERO_VALID_SOURCE_OVERLAP": int(len(zero_overlap))},
        "frozen_population_patients": 948,
        "frozen_population_visits": 3792,
        "header_audit_rows": int(len(audit)),
        "header_audit_exact_population_coverage": True,
        "selected_patients": int(len(selection)),
        "selected_visits": int(4 * len(selection)),
        "zero_valid_source_overlap_visits": int(len(zero_overlap)),
        "minimum_valid_source_voxels": 0,
        "atomic_caches_present_for_selected_patients": selected_cache_count,
        "cache_contract_completed": False,
        "full_scope_cache_build_forbidden": True,
        "stage_b_authorized": False,
        "thresholds_relaxed": False,
        "private_evidence_sha256": {
            "failure_table": file_sha256(failure_path),
            "selection_manifest": file_sha256(selection_path),
            "complete_header_audit": file_sha256(audit_path),
        },
        "contains_patient_identifiers": False,
    }
    gate_path = (
        experiment_root
        / "metrics"
        / f"model_input_pipeline_{strategy.lower()}_{scope}_gate.json"
    )
    atomic_text(
        gate_path,
        json.dumps(public, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )
    report = f"""# Model-input pipeline validation

## 结论

完整的 948 人、3792 visit header-only overlap audit 在冻结人口中发现 {len(zero_overlap)} 个 `ZERO_VALID_SOURCE_OVERLAP` visit（valid-source voxel = 0）。因此 production-like cache validation **FAIL**；不得改变冻结人口、不得继续 full-scope cache、不得启动 Stage B。

- 本次 validation 目标为 {len(selection)} 人；已有 {selected_cache_count}/{len(selection)} 个患者 cache 通过原子写入，但 cohort-level schema-3 cache contract 未闭合，不能据此判 PASS。
- 失败病例标识与路径只存在 private failure table/selection manifest；公开报告仅保留计数与 SHA-256 闭包。
- 未放宽阈值，也未事后排除病例。
"""
    atomic_text(
        experiment_root / "reports/model_input_pipeline_validation.md",
        report,
        overwrite=overwrite,
    )
    phase_report = experiment_root / "reports/dce7_phase_contract.md"
    atomic_text(
        phase_report,
        "# DCE7 phase contract\n\n"
        "DCE7语义实现已由单元测试覆盖；但冻结cohort的schema-3 cache contract"
        "未完成，因此本轮不得判PASS。\n",
        overwrite=overwrite,
    )
    raise RuntimeError(
        "ZERO_VALID_SOURCE_OVERLAP: fail-closed before cache workers; Stage B forbidden"
    )


def repair_map() -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for path in sorted((EXPERIMENT_ROOT / "manifests/repair_private").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise ValueError("A raw-DICOM rebuild record did not pass")
        key = (str(payload["patient_id"]), str(payload["visit"]))
        rebuilt = str(Path(payload["private"]["output_nifti"]).resolve())
        if key in output or not Path(rebuilt).is_file():
            raise ValueError("Repair map is duplicate or incomplete")
        output[key] = rebuilt
    if len(output) != 146:
        raise ValueError(f"Expected 146 complete-cohort repairs, found {len(output)}")
    return output


def phase_map(path: Path) -> dict[str, dict[str, float | None]]:
    # Deliberately do not inspect the file first: usecols prevents pCR,
    # treatment, clinical, volume, and split columns from entering memory.
    frame = pd.read_csv(path, usecols=["pid", "pre", "post_early", "post_late"])
    if frame["pid"].astype(str).duplicated().any():
        raise ValueError("Acquisition phase table has duplicate patient IDs")
    output: dict[str, dict[str, float | None]] = {}
    for row in frame.itertuples(index=False):
        output[str(row.pid)] = {
            "pre": None if pd.isna(row.pre) else float(row.pre),
            "post_early": None if pd.isna(row.post_early) else float(row.post_early),
            "post_late": None if pd.isna(row.post_late) else float(row.post_late),
        }
    return output


def transform_map(path: Path | None, strategy: str) -> dict[tuple[str, str], np.ndarray]:
    if strategy == "H":
        if path is not None:
            raise ValueError("C1B-H must not receive a registration-transform file")
        return {}
    if path is None:
        raise ValueError("C1B-R requires --registration-transforms")
    frame = pd.read_csv(path)
    required = {"patient_id", "visit", "status"} | {
        f"m{i}{j}" for i in range(4) for j in range(4)
    }
    if not required.issubset(frame.columns):
        raise ValueError("Registration transform CSV schema is incomplete")
    if frame.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Registration transform CSV has duplicates")
    output: dict[tuple[str, str], np.ndarray] = {}
    for row in frame.to_dict("records"):
        if str(row["visit"]) == "T0":
            continue
        if str(row["status"]) != "PASS":
            raise ValueError("Selected C1B-R contains a failed transform")
        matrix = np.asarray(
            [[row[f"m{i}{j}"] for j in range(4)] for i in range(4)], dtype=np.float64
        )
        output[(str(row["patient_id"]), str(row["visit"]))] = matrix
    return output


def select_validation_ids(
    inventory: pd.DataFrame,
    ispy1_visits: dict[tuple[str, str], dict[str, Any]],
) -> tuple[set[str], dict[str, set[str]]]:
    reasons: dict[str, set[str]] = {}

    def add(ids: Iterable[str], reason: str) -> None:
        for patient_id in ids:
            reasons.setdefault(str(patient_id), set()).add(reason)

    add(
        inventory.loc[inventory["pixel_rebuild_required"].astype(bool), "patient_id"].astype(str),
        "geometry_pixel_repair",
    )
    grounding = pd.read_csv(
        EXPERIMENT_ROOT / "manifests/grounding_observability_manifest.private.csv",
        usecols=["patient_id", "source_boundary_touch"],
    )
    add(
        grounding.loc[grounding["source_boundary_touch"].astype(bool), "patient_id"].astype(str),
        "source_edge",
    )
    support_path = EXPERIMENT_ROOT / "metrics/support_containment_patient_visit.private.csv"
    if not support_path.is_file():
        raise FileNotFoundError("Run orientation/resampling/support audit before cache selection")
    support = pd.read_csv(
        support_path, usecols=["patient_id", "full_physical_volume_mm3"]
    )
    patient_volume = support.groupby("patient_id")["full_physical_volume_mm3"].max()
    cutoff = float(patient_volume.quantile(0.90))
    add(patient_volume[patient_volume >= cutoff].index.astype(str), "large_support_top_decile")

    resampling = pd.read_csv(
        EXPERIMENT_ROOT / "metrics/orientation_resampling_patient_visit.private.csv",
        usecols=["patient_id", "extreme_axis_factor_gt2"],
    )
    add(
        resampling.loc[
            resampling["extreme_axis_factor_gt2"].astype(bool), "patient_id"
        ].astype(str),
        "extreme_resampling_gt2",
    )

    # Read only split-control fields; label_pcr is intentionally excluded.
    folds = pd.read_csv(
        Path(
            "/data/data/Preprocessed/I-SPY2/"
            "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
            "matched_patient_cv_splits_seed2026.csv"
        ),
        usecols=["patient_id", "fold", "split"],
    )
    rng = np.random.default_rng(2026)
    for fold in range(5):
        candidates = np.asarray(
            sorted(
                folds.loc[
                    folds["fold"].eq(fold) & folds["split"].eq("val"), "patient_id"
                ].astype(str).unique()
            )
        )
        chosen = rng.choice(candidates, size=min(5, len(candidates)), replace=False)
        add(chosen, f"fold_{fold}_deterministic_random")

    # The extra base-only cohort has no outer-test role, but its fallback T0
    # anchor and broader phase range must still be represented in validation.
    ispy1 = np.asarray(
        sorted(inventory.loc[inventory["cohort"].eq("I-SPY1"), "patient_id"].astype(str).unique())
    )
    if len(ispy1):
        add(rng.choice(ispy1, size=min(10, len(ispy1)), replace=False), "ispy1_base_fallback")
    # Every accepted cross-phase geometry correction is rare and therefore
    # belongs in the deterministic validation closure, not merely in a random
    # I-SPY1 sample.  The reason remains private because it is patient-level.
    add(
        (
            patient_id
            for (patient_id, _visit), record in ispy1_visits.items()
            if int(record["resampled_phase_count"]) > 0
        ),
        "ispy1_safe_phase_resampling",
    )
    return set(reasons), reasons


def apply_ispy1_eligibility(
    inventory: pd.DataFrame,
    patient_path: Path,
    visit_path: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]], int]:
    """Filter only on frozen source/pixel QC and resolve rebuilt visit inputs."""

    if not patient_path.is_file() or not visit_path.is_file():
        raise FileNotFoundError(
            "Strict I-SPY1 source/pixel eligibility must finish before model-input build"
        )
    patients = pd.read_csv(
        patient_path, usecols=["patient_id", "eligible", "passing_visit_count"]
    )
    visits = pd.read_csv(
        visit_path,
        usecols=[
            "patient_id",
            "visit",
            "status",
            "phase_count",
            "pre_index",
            "early_index",
            "late_index",
            "resampled_phase_count",
            "rebuilt_nifti",
        ],
    )
    if patients["patient_id"].astype(str).duplicated().any() or len(patients) != 156:
        raise ValueError("I-SPY1 patient eligibility must uniquely cover all 156 candidates")
    if visits.duplicated(["patient_id", "visit"]).any() or len(visits) != 624:
        raise ValueError("I-SPY1 visit eligibility must uniquely cover all 624 visits")
    eligible_ids = set(
        patients.loc[
            patients["eligible"].astype(bool) & patients["passing_visit_count"].eq(4),
            "patient_id",
        ].astype(str)
    )
    visit_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in visits.to_dict("records"):
        key = (str(row["patient_id"]), str(row["visit"]))
        if key[0] not in eligible_ids:
            continue
        if str(row["status"]) != "PASS":
            raise ValueError("An eligible I-SPY1 patient contains a failed visit")
        rebuilt = Path(str(row["rebuilt_nifti"]))
        if not rebuilt.is_file():
            raise FileNotFoundError("An eligible I-SPY1 rebuilt NIfTI is missing")
        visit_lookup[key] = {
            "rebuilt_nifti": str(rebuilt.resolve()),
            "phase_count": int(row["phase_count"]),
            "phase_metadata": {
                "pre": int(row["pre_index"]),
                "post_early": int(row["early_index"]),
                "post_late": int(row["late_index"]),
            },
            "resampled_phase_count": int(row["resampled_phase_count"]),
        }
    if len(visit_lookup) != len(eligible_ids) * 4:
        raise ValueError("Eligible I-SPY1 visit sidecar is incomplete")
    keep = inventory["cohort"].eq("I-SPY2") | inventory["patient_id"].astype(str).isin(
        eligible_ids
    )
    return inventory.loc[keep].copy(), visit_lookup, int(len(eligible_ids))


def _padding_audit(
    image: np.ndarray,
    valid_source: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ks_values: list[float] = []
    padded_zero: list[float] = []
    valid_zero: list[float] = []
    padded_count = 0
    for visit in range(4):
        valid = valid_source[visit, 0].astype(bool)
        padded = ~valid
        padded_count += int(padded.sum())
        all_zero = np.all(image[visit] == 0.0, axis=0)
        if padded.any():
            padded_zero.append(float(all_zero[padded].mean()))
            valid_zero.append(float(all_zero[valid].mean()))
            valid_indices = np.flatnonzero(valid.ravel())
            padded_indices = np.flatnonzero(padded.ravel())
            sample_size = min(4096, len(valid_indices), len(padded_indices))
            selected_valid = rng.choice(valid_indices, sample_size, replace=False)
            selected_padded = rng.choice(padded_indices, sample_size, replace=False)
            flattened = image[visit].reshape(7, -1)
            for channel in range(7):
                ks_values.append(
                    float(
                        ks_2samp(
                            flattened[channel, selected_valid],
                            flattened[channel, selected_padded],
                            method="asymp",
                        ).statistic
                    )
                )
    return {
        "padded_voxels": padded_count,
        "padding_valid_ks_median": float(np.median(ks_values)) if ks_values else 0.0,
        "padding_valid_ks_max": float(np.max(ks_values)) if ks_values else 0.0,
        "padding_all_channels_exact_zero_fraction": float(np.mean(padded_zero))
        if padded_zero
        else 0.0,
        "valid_all_channels_exact_zero_fraction": float(np.mean(valid_zero))
        if valid_zero
        else 0.0,
    }


def build_one(task: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()
    patient_id = str(task["patient_id"])
    rows = task["rows"]
    transforms = task["transforms"]
    visits: dict[str, VisitInput] = {}
    for visit in VISITS:
        row = rows[visit]
        support = row["ftv_mask_nifti"] if row["ftv_mask_nifti"] else None
        transform = (
            np.asarray(transforms[visit], dtype=np.float64)
            if transforms.get(visit) is not None
            else None
        )
        visits[visit] = VisitInput(
            visit=visit,
            dce=row["resolved_dce_nifti"],
            phase_metadata=task["phase_metadata_by_visit"].get(visit),
            localization_support=support,
            source_to_anchor_ras=transform,
        )
    cache_path = Path(task["cache_path"])
    builder_kwargs = {
        "cohort": str(task["cohort"]),
        "formal_ftv_overlap": bool(task["formal_ftv_overlap"]),
        "registration_strategy": f"C1B-{task['strategy']}",
    }
    if cache_path.exists() and not task["overwrite"]:
        arrays, validation = load_and_validate_cache(cache_path)
        if validation.patient_id != patient_id:
            raise ValueError("Cache patient identity does not match its tokenized target")
        # Rebuild the current contract in memory.  Comparing the complete
        # deterministic payload closes DCE/support identity, phase metadata,
        # strategy/transforms, anchor/grid, builder semantics, and every QC
        # sidecar—not merely the four source volumes.
        current_patient = build_patient_dce7(
            patient_id, visits, **builder_kwargs
        )
        if content_sha256(current_patient.cache_payload()) != validation.content_sha256:
            raise ValueError("Reused cache does not match the complete current input contract")
        if not np.array_equal(arrays["image"], current_patient.image):
            raise ValueError("Reused cache image differs from the current builder output")
        image = arrays["image"]
        valid_source = arrays["valid_source_mask"]
        reused = True
    else:
        patient = build_patient_dce7(patient_id, visits, **builder_kwargs)
        validation = write_cache_atomic(cache_path, patient)
        image = patient.image
        valid_source = patient.valid_source_mask
        # The public/model loader cannot return any sidecar.
        if not np.array_equal(load_model_tensor(cache_path), image):
            raise ValueError("Model-only cache loader failed exact equality")
        arrays, _ = load_and_validate_cache(cache_path, expected_image=image)
        reused = False

    if validation.patient_id != patient_id:
        raise ValueError("Cache patient identity does not match its tokenized target")

    deterministic_duplicate_match = True
    expected_prior_hash = str(task.get("expected_prior_cache_file_sha256", ""))
    if expected_prior_hash and validation.file_sha256 != expected_prior_hash:
        raise ValueError("Reused cache no longer matches the completed validation proof")
    if task["duplicate_check"]:
        duplicate = cache_path.with_suffix(".duplicate.npz")
        if duplicate.exists():
            duplicate.unlink()
        try:
            # Rebuild from raw/repaired volume, not from an existing NPZ.
            duplicate_patient = build_patient_dce7(
                patient_id, visits, **builder_kwargs
            )
            duplicate_validation = write_cache_atomic(duplicate, duplicate_patient)
            deterministic_duplicate_match = (
                duplicate_validation.file_sha256 == validation.file_sha256
                and duplicate_validation.content_sha256 == validation.content_sha256
            )
            if not deterministic_duplicate_match:
                raise ValueError("Independent builder/cache repeat is not byte deterministic")
        finally:
            duplicate.unlink(missing_ok=True)

    phase_indices = np.asarray(arrays["phase_indices"], dtype=np.int64)
    phase_counts = np.asarray(arrays["phase_counts"], dtype=np.int64)
    channel_std = image.reshape(4, 7, -1).std(axis=-1, dtype=np.float64)
    audit = _padding_audit(image, valid_source, int(task["seed"]))
    result: dict[str, Any] = {
        "patient_id": patient_id,
        "cohort": str(task["cohort"]),
        "strategy": str(task["strategy"]),
        "scope": str(task["scope"]),
        "selection_reasons": "|".join(task["selection_reasons"]),
        "cache_path": str(cache_path.resolve()),
        "cache_content_sha256": validation.content_sha256,
        "cache_file_sha256": validation.file_sha256,
        "cache_reused": reused,
        "cache_schema_version": int(arrays["schema_version"].item()),
        "cache_patient_identity_match": validation.patient_id == patient_id,
        "cache_current_source_hash_match": True,
        "cache_complete_input_contract_match": True,
        "builder_contract_sha256": str(arrays["builder_contract_sha256"].item()),
        "input_provenance_sha256": str(arrays["input_provenance_sha256"].item()),
        "formal_ftv_overlap": bool(arrays["formal_ftv_overlap"].item()),
        "support_scope": "|".join(str(value) for value in arrays["support_scope"]),
        "base_only_later_support_loaded_count": int(
            (not bool(arrays["formal_ftv_overlap"].item()))
            * int(np.asarray(arrays["support_available"])[1:].sum())
        ),
        "deterministic_duplicate_match": deterministic_duplicate_match,
        "deterministic_duplicate_check_performed": bool(task["duplicate_check"]),
        "determinism_proof_source": (
            "independent_rebuild_this_run"
            if task["duplicate_check"]
            else "prior_validation_hash_closure"
            if expected_prior_hash
            else "not_in_validation_subset"
        ),
        "shape_valid": tuple(image.shape) == (4, 7, 112, 176, 160),
        "dtype_float32": image.dtype == np.float32,
        "finite": bool(np.isfinite(image).all()),
        "whole_visit_nonconstant": bool(np.all(channel_std[:, :3].max(axis=1) > 1e-6)),
        "constant_channel_count": int(np.count_nonzero(channel_std <= 1e-6)),
        "phase_indices_json": json.dumps(phase_indices.tolist()),
        "phase_counts_json": json.dumps(phase_counts.tolist()),
        "phase_indices_in_range": bool(
            np.all(phase_indices >= 0) and np.all(phase_indices < phase_counts[:, None])
        ),
        "canonical_orientation": "RAS+",
        "grid_shape_zyx": "112x176x160",
        "grid_spacing_xyz_mm": "0.9x0.9x2.0",
        "elapsed_seconds": float(time.monotonic() - start),
        **audit,
    }
    return result


def phase_contract_report(metrics: pd.DataFrame, missing_metadata_patients: int) -> str:
    counts: dict[int, int] = {}
    for value in metrics["phase_counts_json"]:
        for count in json.loads(value):
            counts[int(count)] = counts.get(int(count), 0) + 1
    return f"""# DCE7 phase contract

## 冻结定义

正式builder从既有代码复核并固定为七个通道：`pre`、`early`、`late`、`early-pre`、`late-pre`、`(peak-pre)/max(abs(pre),1)`、`(late-peak)/max(abs(pre),1)`；`peak`只在postcontrast phase 1..last逐voxel取最大值。

- phase mapping只允许读取 acquisition fields `pid/pre/post_early/post_late`；读取时使用明确`usecols`，pCR、clinical、treatment、FTV、LD从未进入内存。
- `T<=4`：early=`pre+1`（clip），late=last；`T>4`：使用既有patient acquisition metadata，缺失时固定outcome-free default `0/min(2,T-1)/min(5,T-1)`。
- metadata缺失：{missing_metadata_patients}人；不按outcome或lesion appearance补相位。
- 已验证cache内phase-count分布（visit）：`{json.dumps(dict(sorted(counts.items())))}`；所有indices均在各自phase count范围内。
- 全部raw phase先一起做同一个4-D spatial resampling；phase轴为identity，不做temporal interpolation。DCE7 derived channels在resampling后按上述公式一次构造。

因此 phase semantics 子门：**PASS**。
"""


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    inventory = pd.read_csv(args.inventory)
    if len(inventory) != 3856 or inventory.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Inventory is not the frozen 964x4 cohort")
    if inventory["formal_ftv_overlap"].isna().any() or not set(
        inventory["formal_ftv_overlap"].unique()
    ).issubset({True, False, np.bool_(True), np.bool_(False)}):
        raise ValueError("formal FTV-overlap membership must be an explicit boolean")
    repairs = repair_map()
    phases = phase_map(args.phase_metadata)
    transforms = transform_map(args.registration_transforms, args.strategy)
    inventory, ispy1_visits, eligible_ispy1_count = apply_ispy1_eligibility(
        inventory,
        args.ispy1_patient_eligibility,
        args.ispy1_visit_eligibility,
    )
    inventory["resolved_dce_nifti"] = [
        ispy1_visits.get((str(row.patient_id), str(row.visit)), {}).get(
            "rebuilt_nifti",
            repairs.get((str(row.patient_id), str(row.visit)), str(row.dce_nifti)),
        )
        for row in inventory.itertuples(index=False)
    ]
    for path in inventory["resolved_dce_nifti"]:
        if not Path(path).is_file():
            raise FileNotFoundError("A resolved model-input volume is missing")
    if args.strategy == "R":
        expected_transform_keys = {
            (str(row.patient_id), str(row.visit))
            for row in inventory.itertuples(index=False)
            if str(row.visit) != "T0"
        }
        if set(transforms) != expected_transform_keys:
            raise ValueError(
                "C1B-R transform manifest must exactly cover every selected T1-T3 visit"
            )

    validation_ids, reasons = select_validation_ids(inventory, ispy1_visits)
    all_ids = set(inventory["patient_id"].astype(str))
    chosen_ids = validation_ids if args.scope == "validation" else all_ids
    if not validation_ids.issubset(chosen_ids):
        raise AssertionError("Validation subset must be contained in every all-scope run")

    # This complete, header-only source-overlap check must precede cache-root
    # creation and worker submission.  A zero-overlap visit is a frozen-cohort
    # model-readiness failure, never an eligibility rule.
    catastrophic_source_overlap_preflight(
        inventory,
        validation_ids=validation_ids,
        reasons=reasons,
        scope=args.scope,
        strategy=args.strategy,
        overwrite=args.overwrite,
    )

    cache_root = EXPERIMENT_ROOT / "cache" / f"c1b_{args.strategy.lower()}"
    cache_root.mkdir(parents=True, exist_ok=True)
    prior_validation_hashes: dict[str, str] = {}
    prior_validation_metrics = (
        EXPERIMENT_ROOT
        / "metrics"
        / f"model_input_pipeline_{args.strategy.lower()}_validation.private.csv"
    )
    if args.scope == "all" and args.rebuild_cache and prior_validation_metrics.is_file():
        raise ValueError(
            "Refusing to overwrite full caches under an existing validation proof; "
            "rebuild and close validation first, then run all scope without --rebuild-cache"
        )
    if args.scope == "all" and prior_validation_metrics.is_file():
        prior = pd.read_csv(
            prior_validation_metrics,
            usecols=[
                "patient_id",
                "cache_file_sha256",
                "deterministic_duplicate_match",
                "deterministic_duplicate_check_performed",
                "cache_schema_version",
                "builder_contract_sha256",
                "input_provenance_sha256",
                "cache_complete_input_contract_match",
            ],
        )
        if (
            prior["patient_id"].astype(str).duplicated().any()
            or set(prior["patient_id"].astype(str)) != validation_ids
            or not prior["deterministic_duplicate_match"].astype(bool).all()
            or not prior["deterministic_duplicate_check_performed"].astype(bool).all()
            or not prior["cache_complete_input_contract_match"].astype(bool).all()
            or not prior["cache_schema_version"].eq(3).all()
            or set(prior["builder_contract_sha256"].astype(str))
            != {builder_contract_sha256()}
            or not prior["input_provenance_sha256"].astype(str).str.fullmatch(
                r"[0-9a-f]{64}"
            ).all()
        ):
            raise ValueError("Prior validation determinism proof is incomplete")
        prior_validation_hashes = dict(
            zip(
                prior["patient_id"].astype(str),
                prior["cache_file_sha256"].astype(str),
                strict=True,
            )
        )
    tasks: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    missing_metadata = 0
    for patient_id, group in inventory.groupby("patient_id", sort=True):
        patient_id = str(patient_id)
        if patient_id not in chosen_ids:
            continue
        if set(group["visit"]) != set(VISITS) or len(group) != 4:
            raise ValueError("Patient visit inventory is incomplete")
        cohorts = set(group["cohort"].astype(str))
        formal_values = set(group["formal_ftv_overlap"].astype(bool))
        if len(cohorts) != 1 or len(formal_values) != 1:
            raise ValueError("Patient cohort/formal membership is inconsistent across visits")
        patient_cohort = next(iter(cohorts))
        patient_formal = bool(next(iter(formal_values)))
        metadata = phases.get(patient_id)
        per_visit_metadata: dict[str, dict[str, Any] | None] = {}
        for visit in VISITS:
            strict = ispy1_visits.get((patient_id, visit))
            per_visit_metadata[visit] = (
                dict(strict["phase_metadata"]) if strict is not None else metadata
            )
        if all(value is None for value in per_visit_metadata.values()):
            missing_metadata += 1
        row_map: dict[str, dict[str, Any]] = {}
        patient_transforms: dict[str, list[list[float]] | None] = {}
        for row in group.to_dict("records"):
            visit = str(row["visit"])
            mask = row.get("ftv_mask_nifti")
            mask_path = mask if isinstance(mask, str) and mask else ""
            # T0 localization alone freezes the common longitudinal grid.  A
            # later-visit support is loaded only for retrospective containment
            # QC in the formal 375-patient FTV-overlap cohort; it can never
            # alter the anchor, tensor, or base-only population.  In
            # particular, released empty T1-T3 masks outside that cohort are
            # neither localization inputs nor a reason to drop a base patient.
            row["ftv_mask_nifti"] = (
                mask_path
                if visit == "T0" or bool(row.get("formal_ftv_overlap", False))
                else ""
            )
            row_map[visit] = row
            matrix = transforms.get((patient_id, visit))
            patient_transforms[visit] = matrix.tolist() if matrix is not None else None
        selected_reasons = sorted(reasons.get(patient_id, {"all_model_inputs"}))
        cache_path = cache_root / f"{patient_token(patient_id)}.npz"
        tasks.append(
            {
                "patient_id": patient_id,
                "cohort": patient_cohort,
                "formal_ftv_overlap": patient_formal,
                "rows": row_map,
                "phase_metadata_by_visit": per_visit_metadata,
                "transforms": patient_transforms,
                "strategy": args.strategy,
                "scope": args.scope,
                "selection_reasons": selected_reasons,
                "cache_path": str(cache_path),
                "overwrite": args.rebuild_cache,
                "duplicate_check": (
                    patient_id in validation_ids
                    and patient_id not in prior_validation_hashes
                ),
                "expected_prior_cache_file_sha256": prior_validation_hashes.get(
                    patient_id, ""
                ),
                "seed": int(patient_token(patient_id)[:8], 16),
            }
        )
        selection_records.append(
            {
                "patient_id": patient_id,
                "cohort": patient_cohort,
                "selected_for_validation": patient_id in validation_ids,
                "selection_reasons": "|".join(selected_reasons),
                "cache_path": str(cache_path.resolve()),
            }
        )
    if not tasks:
        raise ValueError("No patients selected")

    results: list[dict[str, Any]] = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(futures),
                            "elapsed_seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )

    metrics = pd.DataFrame(results).sort_values("patient_id", kind="stable")
    selection = pd.DataFrame(selection_records).sort_values("patient_id", kind="stable")
    required_boolean = (
        metrics["shape_valid"]
        & metrics["dtype_float32"]
        & metrics["finite"]
        & metrics["whole_visit_nonconstant"]
        & metrics["phase_indices_in_range"]
        & metrics["cache_patient_identity_match"]
        & metrics["cache_current_source_hash_match"]
        & metrics["cache_complete_input_contract_match"]
        & metrics["cache_schema_version"].eq(3)
        & metrics["base_only_later_support_loaded_count"].eq(0)
        & metrics["deterministic_duplicate_match"]
    )
    if not required_boolean.all():
        raise ValueError("At least one production cache failed a required validation")
    private_metrics = (
        EXPERIMENT_ROOT
        / "metrics"
        / f"model_input_pipeline_{args.strategy.lower()}_{args.scope}.private.csv"
    )
    private_selection = EXPERIMENT_ROOT / "manifests/cache_validation_selection.private.csv"
    atomic_text(private_metrics, metrics.to_csv(index=False), overwrite=args.overwrite)
    atomic_text(private_selection, selection.to_csv(index=False), overwrite=args.overwrite)

    public = {
        "schema_version": 1,
        "cache_schema_version": 3,
        "builder_contract_sha256": builder_contract_sha256(),
        "strategy": f"C1B-{args.strategy}",
        "scope": args.scope,
        "patients": int(len(metrics)),
        "visits": int(len(metrics) * 4),
        "eligible_ispy1_base_patients": eligible_ispy1_count,
        "validation_subset_patients": int(selection["selected_for_validation"].sum()),
        "geometry_repair_patients_in_validation": int(
            selection["selection_reasons"].str.contains("geometry_pixel_repair").sum()
        ),
        "source_edge_patients_in_validation": int(
            selection["selection_reasons"].str.contains("source_edge").sum()
        ),
        "large_support_patients_in_validation": int(
            selection["selection_reasons"].str.contains("large_support_top_decile").sum()
        ),
        "fold_random_patients_in_validation": int(
            selection["selection_reasons"].str.contains("fold_").sum()
        ),
        "safe_phase_resampling_patients_in_validation": int(
            selection["selection_reasons"]
            .str.contains("ispy1_safe_phase_resampling")
            .sum()
        ),
        "cache_exact_roundtrip_pass_fraction": float(required_boolean.mean()),
        "cache_patient_identity_match_fraction": float(
            metrics["cache_patient_identity_match"].mean()
        ),
        "cache_current_source_hash_match_fraction": float(
            metrics["cache_current_source_hash_match"].mean()
        ),
        "cache_complete_input_contract_match_fraction": float(
            metrics["cache_complete_input_contract_match"].mean()
        ),
        "byte_deterministic_validation_fraction": float(
            metrics.loc[
                metrics["patient_id"].isin(validation_ids), "deterministic_duplicate_match"
            ].mean()
        ),
        "validation_duplicate_rebuilds_this_run": int(
            metrics["deterministic_duplicate_check_performed"].sum()
        ),
        "validation_prior_hash_proofs_reused": int(
            metrics["determinism_proof_source"]
            .eq("prior_validation_hash_closure")
            .sum()
        ),
        "finite_fraction": float(metrics["finite"].mean()),
        "whole_visit_nonconstant_fraction": float(metrics["whole_visit_nonconstant"].mean()),
        "phase_indices_in_range_fraction": float(metrics["phase_indices_in_range"].mean()),
        "constant_derived_channels_total": int(metrics["constant_channel_count"].sum()),
        "padding_all_channels_exact_zero_fraction_median": float(
            metrics["padding_all_channels_exact_zero_fraction"].median()
        ),
        "valid_all_channels_exact_zero_fraction_median": float(
            metrics["valid_all_channels_exact_zero_fraction"].median()
        ),
        "padding_valid_ks_median": float(metrics["padding_valid_ks_median"].median()),
        "padding_valid_ks_q95": float(metrics["padding_valid_ks_max"].quantile(0.95)),
        "padding_mode": "reflect",
        "zero_or_fixed_sentinel_padding": False,
        "model_loader_returns_only_dce7": True,
        "anchor_support_scope": "T0_only",
        "later_visit_support_scope": "formal_ftv_overlap_containment_qc_only",
        "later_visit_support_affects_grid_or_tensor": False,
        "base_only_later_visit_supports_loaded": int(
            metrics["base_only_later_support_loaded_count"].sum()
        ),
        "sidecars_are_model_inputs": False,
        "clinical_treatment_pcr_ld_columns_read": False,
        "missing_acquisition_phase_metadata_patients": int(missing_metadata),
        "contains_patient_identifiers": False,
        "elapsed_seconds": float(time.monotonic() - start),
    }
    output_json = (
        EXPERIMENT_ROOT
        / "metrics"
        / f"model_input_pipeline_{args.strategy.lower()}_{args.scope}_gate.json"
    )
    atomic_text(output_json, json.dumps(public, indent=2, sort_keys=True) + "\n", overwrite=args.overwrite)
    atomic_text(
        EXPERIMENT_ROOT / "reports/dce7_phase_contract.md",
        phase_contract_report(metrics, missing_metadata),
        overwrite=args.overwrite,
    )
    report = f"""# Model-input pipeline validation

## 结论

本次对 {len(metrics)} 人、{len(metrics) * 4} visit实际执行完整production-like链：raw/repaired DCE → true RAS+ canonicalization → frozen phase mapping → optional frozen registration operator → fixed physical C1B resampling → DCE7 → valid-source-only normalization → deterministic cache → reload。所有缓存均exact float32 round-trip通过；validation subset独立重建第二次后byte hash一致。

- 策略：`C1B-{args.strategy}`；scope：`{args.scope}`。
- validation subset：{int(selection['selected_for_validation'].sum())}人，覆盖全部geometry repair、source-edge、large-support top decile、每fold deterministic random病例、I-SPY1 acquisition-centre fallback及全部safe phase-resampling病例。
- shape/dtype/finite/nonconstant/phase-index/cache gate：{int(required_boolean.sum())}/{len(required_boolean)} PASS。
- 每个schema-v3 cache内锁定T0–T3 canonical DCE/support、phase metadata、support scope、H/R transforms、T0 grid与builder semantic contract；patient identity、当前source及完整input-provenance closure均为{int(required_boolean.sum())}/{len(required_boolean)} PASS。
- model loader只返回`image [4,7,112,176,160] float32`；affine、valid-source、transform、support、phase metadata均无法成为model channel。
- 共同grid仅可由T0 support（或I-SPY1 T0 acquisition-centre fallback）冻结；T1–T3 support只在正式FTV-overlap cohort做事后containment QC，base-only follow-up support读取数为0，不能改变grid或tensor。
- padding为current-volume reflection，不是zero/fixed sentinel。padded全7通道恰为0的患者内比例median={public['padding_all_channels_exact_zero_fraction_median']:.6f}，valid区对应={public['valid_all_channels_exact_zero_fraction_median']:.6f}；padding-vs-valid KS仅作sidecar context audit，不进入模型或selection。
- phase metadata读取字段严格限于`pid/pre/post_early/post_late`；未读取clinical、treatment、pCR、FTV或LD来构建图像。

因此本scope完整builder/cache round-trip：**PASS**。
"""
    atomic_text(
        EXPERIMENT_ROOT / "reports/model_input_pipeline_validation.md",
        report,
        overwrite=args.overwrite,
    )
    print(json.dumps(public, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
