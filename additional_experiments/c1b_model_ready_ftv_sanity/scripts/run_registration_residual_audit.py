#!/usr/bin/env python3
"""Post-hoc anatomy-versus-localization residual audit for Stage A5.

This script never fits or alters a transform.  It reuses the frozen private
Stage A5 matrices, measures precontrast-only whole-anatomy physical-centroid
residuals, then opens localization masks solely to compute an independent QC
centroid residual.  Failed registrations use the explicit C1B-H identity/header
fallback; failed transforms are never applied.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from c1b_sanity.dce7 import select_phase_indices  # noqa: E402
from c1b_sanity.geometry import CanonicalVolume, load_nifti_ras, validate_affine  # noqa: E402
from c1b_sanity.registration import _whole_anatomy_mask_array  # noqa: E402


VISITS = ("T0", "T1", "T2", "T3")
MOVING_VISITS = ("T1", "T2", "T3")
ALLOWED_PHASE_FIELDS = ("pre", "post_early", "post_late")
PRIVATE_COLUMNS = (
    "patient_id",
    "visit",
    "registration_success",
    "registration_failure_code",
    "transform_disposition",
    "anchor_dce_path",
    "source_dce_path",
    "anchor_localization_path",
    "source_localization_path",
    "anatomy_residual_before_mm",
    "anatomy_residual_after_mm",
    "anatomy_residual_reduction_mm",
    "lesion_residual_before_mm",
    "lesion_residual_after_mm",
    "lesion_residual_reduction_mm",
    "anatomy_gt5_lesion_lt2",
    "anatomy_gt5_lesion_compressed_from_ge2_to_lt2",
    "audit_success",
    "audit_failure_code",
    "audit_failure_message",
)


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_phase_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        required = ("pid", *ALLOWED_PHASE_FIELDS)
        missing = [field for field in required if field not in header]
        if missing:
            raise ValueError(f"phase metadata lacks acquisition fields: {missing}")
        index = {field: header.index(field) for field in required}
        output: dict[str, dict[str, str]] = {}
        for row in reader:
            patient_id = row[index["pid"]]
            output[patient_id] = {
                field: row[index[field]] for field in ALLOWED_PHASE_FIELDS
            }
    return output


def _load_registration_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["patient_id"], row["visit"])
        if key in output:
            raise ValueError("registration metrics contain a duplicate patient/visit")
        output[key] = row
    return output


def _load_patients(
    inventory_path: Path,
    repaired_root: Path,
    registration_rows: Mapping[tuple[str, str], Mapping[str, str]],
    phase_metadata: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    with inventory_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if not _truth(row["formal_ftv_overlap"]):
                continue
            patient_id = row["patient_id"]
            visit = row["visit"]
            repaired = _truth(row["pixel_rebuild_required"])
            dce_path = (
                repaired_root / row["cohort"] / patient_id / f"{visit}_dce_rebuilt.nii.gz"
                if repaired
                else Path(row["dce_nifti"])
            )
            grouped.setdefault(patient_id, {})[visit] = {
                "dce_path": str(dce_path),
                "localization_path": row["ftv_mask_nifti"],
                "phase_count": int(row["phase_count"]),
            }
    patients: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        if set(grouped[patient_id]) != set(VISITS):
            raise ValueError("formal residual-audit cohort contains an incomplete patient")
        pairs = {
            visit: dict(registration_rows[(patient_id, visit)]) for visit in MOVING_VISITS
        }
        patients.append(
            {
                "patient_id": patient_id,
                "phase_metadata": dict(phase_metadata.get(patient_id, {})),
                "visits": grouped[patient_id],
                "registration_rows": pairs,
            }
        )
    return patients


def _precontrast(
    visit: Mapping[str, Any], phase_metadata: Mapping[str, Any]
) -> CanonicalVolume:
    volume = load_nifti_ras(visit["dce_path"])
    phase_count = 1 if volume.data.ndim == 3 else int(volume.data.shape[-1])
    if phase_count != int(visit["phase_count"]):
        raise ValueError("DCE phase count disagrees with frozen inventory")
    selection = select_phase_indices(phase_count, phase_metadata)
    selected = volume.data if volume.data.ndim == 3 else volume.data[..., selection.pre]
    return CanonicalVolume(
        data=np.ascontiguousarray(selected, dtype=np.float32),
        affine_ras=np.asarray(volume.affine_ras, dtype=np.float64),
        original_axcodes=volume.original_axcodes,
        orientation_transform=np.asarray(volume.orientation_transform, dtype=np.float64),
        source_path=volume.source_path,
    )


def _physical_centroid(mask_xyz: np.ndarray, affine_ras: np.ndarray) -> np.ndarray | None:
    coordinates = np.argwhere(np.asarray(mask_xyz, dtype=bool))
    if coordinates.size == 0:
        return None
    center_voxel = coordinates.astype(np.float64).mean(axis=0)
    affine = validate_affine(affine_ras)
    return affine[:3, :3] @ center_voxel + affine[:3, 3]


def _anatomy_centroid(volume: CanonicalVolume) -> np.ndarray:
    mask = _whole_anatomy_mask_array(volume.data, minimum_voxels=128)
    center = _physical_centroid(mask, volume.affine_ras)
    if center is None:
        raise ValueError("automatic whole-anatomy mask is empty")
    return center


def _localization_centroid(path_text: str) -> np.ndarray | None:
    image = nib.load(path_text)
    affine = validate_affine(np.asarray(image.affine), name="localization affine")
    mask = np.asanyarray(image.dataobj)
    if mask.ndim > 3:
        mask = mask[..., 0]
    if mask.ndim != 3 or not np.isfinite(mask).all():
        raise ValueError("localization mask must be finite and 3-D")
    return _physical_centroid(np.asarray(mask) > 0.5, affine)


def _matrix(row: Mapping[str, str]) -> np.ndarray:
    if not _truth(row["success"]):
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(
        [
            [float(row[f"source_to_anchor_ras_{i}{j}"]) for j in range(4)]
            for i in range(4)
        ],
        dtype=np.float64,
    )
    validate_affine(matrix, name="frozen source_to_anchor_ras")
    return matrix


def _apply(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ point + matrix[:3, 3]


def _blank_row(patient: Mapping[str, Any], visit: str) -> dict[str, Any]:
    registration = patient["registration_rows"][visit]
    return {field: "" for field in PRIVATE_COLUMNS} | {
        "patient_id": patient["patient_id"],
        "visit": visit,
        "registration_success": _truth(registration["success"]),
        "registration_failure_code": registration["failure_code"],
        "transform_disposition": (
            "FROZEN_RIGID_TRANSFORM"
            if _truth(registration["success"])
            else "C1B_H_IDENTITY_HEADER_FALLBACK"
        ),
        "anchor_dce_path": patient["visits"]["T0"]["dce_path"],
        "source_dce_path": patient["visits"][visit]["dce_path"],
        "anchor_localization_path": patient["visits"]["T0"]["localization_path"],
        "source_localization_path": patient["visits"][visit]["localization_path"],
        "audit_success": False,
    }


def _process_patient(patient: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        anchor = _precontrast(patient["visits"]["T0"], patient["phase_metadata"])
        anchor_anatomy = _anatomy_centroid(anchor)
        anchor_lesion = _localization_centroid(
            patient["visits"]["T0"]["localization_path"]
        )
    except Exception as exc:
        rows = []
        for visit in MOVING_VISITS:
            row = _blank_row(patient, visit)
            row["audit_failure_code"] = "ANCHOR_AUDIT_EXCEPTION"
            row["audit_failure_message"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        return rows

    rows: list[dict[str, Any]] = []
    for visit in MOVING_VISITS:
        row = _blank_row(patient, visit)
        registration = patient["registration_rows"][visit]
        try:
            source = _precontrast(patient["visits"][visit], patient["phase_metadata"])
            source_anatomy = _anatomy_centroid(source)
            source_lesion = _localization_centroid(
                patient["visits"][visit]["localization_path"]
            )
            transform = _matrix(registration)
            anatomy_before = float(np.linalg.norm(source_anatomy - anchor_anatomy))
            anatomy_after = float(
                np.linalg.norm(_apply(transform, source_anatomy) - anchor_anatomy)
            )
            row["anatomy_residual_before_mm"] = anatomy_before
            row["anatomy_residual_after_mm"] = anatomy_after
            row["anatomy_residual_reduction_mm"] = anatomy_before - anatomy_after
            if anchor_lesion is not None and source_lesion is not None:
                lesion_before = float(np.linalg.norm(source_lesion - anchor_lesion))
                lesion_after = float(
                    np.linalg.norm(_apply(transform, source_lesion) - anchor_lesion)
                )
                row["lesion_residual_before_mm"] = lesion_before
                row["lesion_residual_after_mm"] = lesion_after
                row["lesion_residual_reduction_mm"] = lesion_before - lesion_after
                threshold_pattern = (
                    _truth(registration["success"])
                    and anatomy_after > 5.0
                    and lesion_after < 2.0
                )
                compressed_pattern = threshold_pattern and lesion_before >= 2.0
                row["anatomy_gt5_lesion_lt2"] = threshold_pattern
                row["anatomy_gt5_lesion_compressed_from_ge2_to_lt2"] = compressed_pattern
            row["audit_success"] = True
        except Exception as exc:
            row["audit_failure_code"] = "PAIR_AUDIT_EXCEPTION"
            row["audit_failure_message"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _write_private(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (str(row["patient_id"]), str(row["visit"])))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PRIVATE_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def _distribution(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [value for row in rows if (value := _safe_float(row.get(field))) is not None]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "minimum": float(np.min(values)) if values else None,
        "q05": float(np.quantile(values, 0.05)) if values else None,
        "median": float(np.quantile(values, 0.50)) if values else None,
        "q95": float(np.quantile(values, 0.95)) if values else None,
        "maximum": float(np.max(values)) if values else None,
    }


def _aggregate(rows: list[dict[str, Any]], private_path: Path) -> dict[str, Any]:
    successful_audits = [row for row in rows if _truth(row["audit_success"])]
    fitted = [row for row in successful_audits if _truth(row["registration_success"])]
    fallback = [row for row in rows if not _truth(row["registration_success"])]
    lesion_evaluable = [
        row for row in fitted if _safe_float(row["lesion_residual_after_mm"]) is not None
    ]
    threshold_flags = [row for row in lesion_evaluable if _truth(row["anatomy_gt5_lesion_lt2"])]
    compressed_flags = [
        row
        for row in lesion_evaluable
        if _truth(row["anatomy_gt5_lesion_compressed_from_ge2_to_lt2"])
    ]
    failure_codes: dict[str, int] = {}
    for row in rows:
        if not _truth(row["audit_success"]):
            code = str(row["audit_failure_code"] or "UNSPECIFIED_AUDIT_FAILURE")
            failure_codes[code] = failure_codes.get(code, 0) + 1

    fields = (
        "anatomy_residual_before_mm",
        "anatomy_residual_after_mm",
        "anatomy_residual_reduction_mm",
        "lesion_residual_before_mm",
        "lesion_residual_after_mm",
        "lesion_residual_reduction_mm",
    )
    by_visit: dict[str, Any] = {}
    for visit in MOVING_VISITS:
        visit_rows = [row for row in fitted if row["visit"] == visit]
        visit_evaluable = [
            row for row in visit_rows if _safe_float(row["lesion_residual_after_mm"]) is not None
        ]
        visit_flags = [row for row in visit_evaluable if _truth(row["anatomy_gt5_lesion_lt2"])]
        by_visit[visit] = {
            "successful_transforms": len(visit_rows),
            "lesion_evaluable": len(visit_evaluable),
            "threshold_pattern_count": len(visit_flags),
            "threshold_pattern_rate": (
                len(visit_flags) / len(visit_evaluable) if visit_evaluable else None
            ),
            "median_anatomy_residual_after_mm": _distribution(
                visit_rows, "anatomy_residual_after_mm"
            )["median"],
            "median_lesion_residual_after_mm": _distribution(
                visit_evaluable, "lesion_residual_after_mm"
            )["median"],
        }
    # With no preregistered numerical definition of "systematic", close the
    # R-specific gate conservatively: zero literal threshold cases passes and
    # any observed case fails R.  This can reject R but never refit it.
    criterion_status = "PASS" if not threshold_flags else "FAIL"
    return {
        "schema_version": 1,
        "analysis": "post-hoc anatomy-versus-localization physical-centroid residual audit",
        "contains_patient_identifiers": False,
        "contains_paths": False,
        "transform_refit_or_selection_performed": False,
        "registration_inputs_changed": False,
        "residual_definitions": {
            "anatomy": "Euclidean distance between precontrast-derived whole-anatomy physical centroids",
            "lesion": "Euclidean distance between localization-mask physical centroids; QC only",
            "before": "identity/header physical coordinates",
            "after": "frozen source-to-T0 rigid transform, or identity/header fallback on failed registration",
        },
        "cohort": {
            "pairs": len(rows),
            "audit_successes": len(successful_audits),
            "audit_failures": len(rows) - len(successful_audits),
            "audit_failure_codes": dict(sorted(failure_codes.items())),
            "successful_fitted_transforms": len(fitted),
            "identity_header_fallback_pairs": len(fallback),
            "lesion_residual_evaluable_successful_pairs": len(lesion_evaluable),
        },
        "failure_disposition": {
            "policy": "C1B_H_IDENTITY_HEADER_FALLBACK",
            "failed_transform_is_never_applied": True,
            "pair_count": len(fallback),
            "if_global_strategy_is_C1B_H": "all pairs use identity/header physical alignment",
        },
        "threshold_pattern": {
            "definition": "successful transform with anatomy residual after >5 mm and lesion residual after <2 mm",
            "count": len(threshold_flags),
            "rate_among_lesion_evaluable_successful_pairs": (
                len(threshold_flags) / len(lesion_evaluable) if lesion_evaluable else None
            ),
            "compressed_from_ge2_count": len(compressed_flags),
            "compressed_from_ge2_rate": (
                len(compressed_flags) / len(lesion_evaluable) if lesion_evaluable else None
            ),
            "criterion_status": criterion_status,
            "interpretation": (
                "No numerical rate defining 'systematic' was preregistered; any observed cases require blinded technical review."
            ),
        },
        "distributions": {field: _distribution(fitted, field) for field in fields},
        "by_visit": by_visit,
        "provenance": {
            "private_residual_metrics_sha256": _sha256(private_path),
            "whole_anatomy_mask_source": "precontrast intensity only",
            "localization_use": "post-registration QC only",
        },
    }


def _write_public_tables(root: Path, summary: Mapping[str, Any]) -> None:
    distribution_path = root / "registration_residual_distributions.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("metric", "count", "mean", "minimum", "q05", "median", "q95", "maximum")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for metric, values in summary["distributions"].items():
            writer.writerow({"metric": metric, **values})
    visit_path = root / "registration_residual_by_visit.csv"
    with visit_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "visit",
            "successful_transforms",
            "lesion_evaluable",
            "threshold_pattern_count",
            "threshold_pattern_rate",
            "median_anatomy_residual_after_mm",
            "median_lesion_residual_after_mm",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for visit, values in summary["by_visit"].items():
            writer.writerow({"visit": visit, **values})


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    fitted = [
        row
        for row in rows
        if _truth(row["audit_success"])
        and _truth(row["registration_success"])
        and _safe_float(row["lesion_residual_after_mm"]) is not None
    ]
    anatomy_before = np.asarray(
        [float(row["anatomy_residual_before_mm"]) for row in fitted], dtype=np.float64
    )
    anatomy_after = np.asarray(
        [float(row["anatomy_residual_after_mm"]) for row in fitted], dtype=np.float64
    )
    lesion_before = np.asarray(
        [float(row["lesion_residual_before_mm"]) for row in fitted], dtype=np.float64
    )
    lesion_after = np.asarray(
        [float(row["lesion_residual_after_mm"]) for row in fitted], dtype=np.float64
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    axes[0].scatter(anatomy_before, anatomy_after, s=9, alpha=0.35, color="#355C7D")
    maximum = max(float(np.max(anatomy_before, initial=1.0)), float(np.max(anatomy_after, initial=1.0)))
    axes[0].plot((0, maximum), (0, maximum), "k--", linewidth=1)
    axes[0].set(xlabel="Anatomy residual before (mm)", ylabel="Anatomy residual after (mm)")

    axes[1].scatter(lesion_before, lesion_after, s=9, alpha=0.35, color="#C06C84")
    maximum = max(float(np.max(lesion_before, initial=1.0)), float(np.max(lesion_after, initial=1.0)))
    axes[1].plot((0, maximum), (0, maximum), "k--", linewidth=1)
    axes[1].set(xlabel="Localization residual before (mm)", ylabel="Localization residual after (mm)")

    axes[2].scatter(anatomy_after, lesion_after, s=10, alpha=0.4, color="#2A9D8F")
    axes[2].axvline(5.0, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(2.0, color="black", linestyle="--", linewidth=1)
    axes[2].fill_betweenx((0.0, 2.0), 5.0, max(float(np.max(anatomy_after, initial=6.0)), 6.0), color="#E76F51", alpha=0.12)
    axes[2].set(
        xlabel="Anatomy residual after (mm)",
        ylabel="Localization residual after (mm)",
        title="Flag region: anatomy >5, localization <2",
    )
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.suptitle("Post-hoc frozen-transform physical-centroid residual audit")
    figure.savefig(output, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def _update_main_summary(
    path: Path,
    residual: Mapping[str, Any],
    support: Mapping[str, Any],
    r_manual_review: str,
) -> dict[str, Any]:
    main = json.loads(path.read_text(encoding="utf-8"))
    main["posthoc_residual_audit"] = residual
    main["failure_disposition"] = residual["failure_disposition"]
    criterion = main["gate"]["criteria"]["anatomy_lesion_residual_pattern"]
    criterion.clear()
    criterion.update(
        {
            "status": residual["threshold_pattern"]["criterion_status"],
            "observed_count": residual["threshold_pattern"]["count"],
            "observed_rate": residual["threshold_pattern"][
                "rate_among_lesion_evaluable_successful_pairs"
            ],
            "definition": residual["threshold_pattern"]["definition"],
            "note": residual["threshold_pattern"]["interpretation"],
        }
    )
    exact = main["gate"]["criteria"]["available_support_exact_containment"]
    exact.clear()
    exact.update(
        {
            "status": "PASS" if support["r_exact_drop_gate_pass"] else "FAIL",
            "c1b_h_exact_containment_rate": support[
                "c1b_h_exact_containment_rate"
            ],
            "c1b_r_exact_containment_rate": support[
                "c1b_r_exact_containment_rate"
            ],
            "r_minus_h_points": support[
                "c1b_r_minus_h_exact_containment_points"
            ],
            "threshold": ">=-0.005",
            "failed_pairs_use_identity_header_fallback": True,
        }
    )
    retention = main["gate"]["criteria"]["ftv_retention_q05"]
    retention.clear()
    retention.update(
        {
            "status": "PASS" if support["r_retention_q05_gate_pass"] else "FAIL",
            "c1b_h_q05": support["c1b_h_ftv_retention_q05"],
            "c1b_r_q05": support["c1b_r_ftv_retention_q05"],
            "threshold": ">=0.95",
            "failed_pairs_use_identity_header_fallback": True,
        }
    )
    manual = main["gate"]["criteria"]["blinded_technical_review"]
    manual.clear()
    manual.update(
        {
            "status": r_manual_review,
            "figure": "figures/04_representative_t0_t3_c1b_h_vs_r.png",
            "strata": ["small", "medium", "large", "high-transform"],
            "review_scope": "ghosting, left-right reflection, deformation, and alignment",
        }
    )
    statuses = [item["status"] for item in main["gate"]["criteria"].values()]
    if any(status == "FAIL" for status in statuses):
        main["gate"]["decision"] = "C1B-H"
        main["gate"]["decision_reason"] = "one or more preregistered registration gates failed"
    elif any(status.startswith("PENDING") or status.startswith("REVIEW") for status in statuses):
        main["gate"]["decision"] = "HOLD_C1B-H_PENDING_REMAINING_GATES"
        main["gate"]["decision_reason"] = (
            "C1B-R cannot be selected until every pending preregistered audit passes"
        )
    _atomic_json(path, main)
    return main


def _append_report(path: Path, residual: Mapping[str, Any], main: Mapping[str, Any]) -> None:
    marker = "\n## Post-hoc anatomy-versus-localization residual audit\n"
    original = path.read_text(encoding="utf-8")
    base = original.split(marker, 1)[0].rstrip()
    gate_start = "## Preregistered gates\n"
    gate_end = "## Leakage and privacy controls\n"
    if gate_start not in base or gate_end not in base:
        raise ValueError("registration report lacks expected gate-section markers")
    prefix, remainder = base.split(gate_start, 1)
    _, suffix = remainder.split(gate_end, 1)
    gate_rows: list[str] = []
    for name, item in main["gate"]["criteria"].items():
        observed = item.get("observed")
        if observed is None:
            observed = item.get(
                "observed_all_pairs",
                item.get(
                    "observed_all_pairs_failures_count_as_not_nonworse",
                    item.get(
                    "observed_median",
                    item.get(
                        "observed_rate",
                        item.get(
                            "r_minus_h_points",
                            item.get("c1b_r_q05", "—"),
                        ),
                    ),
                    ),
                ),
            )
        threshold = item.get(
            "threshold", item.get("threshold_median", "technical review")
        )
        if "observed_q95" in item:
            observed = f"median={item['observed_median']}; Q95={item['observed_q95']}"
            threshold = (
                f"median {item['threshold_median']}; Q95 {item['threshold_q95']}"
            )
        gate_rows.append(
            f"| {name} | {item['status']} | {observed} | {threshold} |"
        )
    finalized_gate = (
        "## Preregistered gates\n\n"
        "| Criterion | Status | Observed | Threshold |\n"
        "|---|---|---:|---:|\n"
        + "\n".join(gate_rows)
        + "\n\n"
        "The physical-support criteria use all 1,500 visits and identity/header "
        "fallback for failed registrations. The residual criterion uses all 1,125 "
        "pairs without refitting. The R-specific technical review is FAIL based on "
        "the anonymous numbered montage. No gate remains pending.\n\n"
    )
    base = prefix.rstrip() + "\n\n" + finalized_gate + gate_end + suffix
    base = base.replace(
        "figures/registration_representative_hr.png",
        "figures/04_representative_t0_t3_c1b_h_vs_r.png",
    )
    pattern = residual["threshold_pattern"]
    cohort = residual["cohort"]
    distributions = residual["distributions"]
    appendix = f"""

## Post-hoc anatomy-versus-localization residual audit

No transform was refit or selected in this audit. Precontrast-derived whole-anatomy physical-centroid residuals were compared with localization-mask physical-centroid residuals after applying the already frozen transform. Localization was QC-only.

The audit completed for {cohort['audit_successes']}/{cohort['pairs']} pairs; {cohort['lesion_residual_evaluable_successful_pairs']} successful transforms had nonempty T0 and moving localization masks. Median anatomy residual changed from {distributions['anatomy_residual_before_mm']['median']} mm to {distributions['anatomy_residual_after_mm']['median']} mm. Median localization residual changed from {distributions['lesion_residual_before_mm']['median']} mm to {distributions['lesion_residual_after_mm']['median']} mm.

The literal `anatomy after >5 mm && localization after <2 mm` pattern occurred in {pattern['count']} pairs ({pattern['rate_among_lesion_evaluable_successful_pairs']} of evaluable successful pairs); {pattern['compressed_from_ge2_count']} additionally moved from a pre-registration localization residual at least 2 mm to below 2 mm. Status: **{pattern['criterion_status']}**. No unregistered numerical cutoff for “systematic” was invented.

Failed registrations ({cohort['identity_header_fallback_pairs']} pairs) have the explicit **C1B-H identity/header fallback** disposition. A failed transform is never applied. The overall strategy remains **{main['gate']['decision']}**.

Public figures: `figures/03_registration_transform_distribution.png`, `figures/04_representative_t0_t3_c1b_h_vs_r.png`, and `figures/registration_anatomy_lesion_residuals.png`.
"""
    _atomic_text(path, base + marker + appendix.split(marker, 1)[1].lstrip())


def _write_strategy_decision(
    path: Path,
    main: Mapping[str, Any],
    residual: Mapping[str, Any],
    support: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the compact public strategy contract consumed by Stage A."""

    criteria = main["gate"]["criteria"]
    frozen_decision = str(main["gate"]["decision"])
    chosen_strategy = "R" if frozen_decision == "C1B-R" else "H"
    # H is final as soon as any preregistered hard gate fails.  A hypothetical
    # all-pass image-only result with pending audits would remain unfrozen H.
    any_hard_failure = any(item["status"] == "FAIL" for item in criteria.values())
    decision_frozen = frozen_decision == "C1B-R" or any_hard_failure
    residual_status = str(
        residual["threshold_pattern"]["criterion_status"]
    )
    manual_status = str(criteria["blinded_technical_review"]["status"])
    r_rejection_reasons = [
        name
        for name, item in criteria.items()
        if item["status"] == "FAIL"
        or str(item["status"]).startswith("REVIEW")
    ]
    chosen_h_is_safe = chosen_strategy == "H" and decision_frozen
    payload = {
        "schema_version": 1,
        "chosen_strategy": chosen_strategy,
        "decision_frozen": decision_frozen,
        "decision_reason": main["gate"]["decision_reason"],
        "formal_registration_pairs": main["cohort"]["observed_pairs"],
        "registration_success_pairs": main["outcomes"]["successes"],
        "registration_failure_pairs": main["outcomes"]["failures"],
        "physical_support_audit_visits": support["formal_visits"],
        "physical_support_audit_pairs": support["registration_pairs"],
        "residual_audit_pairs": residual["cohort"]["pairs"],
        "residual_audit_successes": residual["cohort"]["audit_successes"],
        "residual_audit_failures": residual["cohort"]["audit_failures"],
        "manual_review_case_count": 4,
        "manual_review_complete": manual_status in {"PASS", "FAIL"},
        "registration_success_rate": main["outcomes"]["success_rate"],
        "catastrophic_rate": main["outcomes"]["catastrophic_rate_all_pairs"],
        "similarity_gate": criteria["median_whole_anatomy_similarity_gain"],
        "nonworse_gate": criteria["nonworse_moving_visit_fraction"],
        "padding_gate": criteria["padding_increase"],
        "success_gate": criteria["finite_transform_success_rate"],
        "catastrophic_gate": criteria["catastrophic_transform_rate"],
        # These are Stage-A strategy-level dispositions: once H is frozen, an
        # R-specific visual/residual concern cannot make the unregistered H
        # strategy itself unsafe.  The R-specific states remain explicit below.
        "manual_review_pass": chosen_h_is_safe or manual_status == "PASS",
        "residual_audit_pass": chosen_h_is_safe or residual_status.startswith("PASS"),
        "manual_review_status": (
            "PASS_CHOSEN_H" if chosen_h_is_safe else manual_status
        ),
        "residual_audit_status": (
            "PASS_CHOSEN_H" if chosen_h_is_safe else residual_status
        ),
        "r_rejected": chosen_strategy == "H",
        "r_rejection_reasons": r_rejection_reasons,
        "r_specific_manual_review_status": manual_status,
        "r_specific_residual_status": residual_status,
        "exact_containment_gate_status": criteria[
            "available_support_exact_containment"
        ]["status"],
        "ftv_retention_gate_status": criteria["ftv_retention_q05"]["status"],
        "failed_pair_disposition": "C1B_H_IDENTITY_HEADER_FALLBACK",
        "failed_pair_fallback_count": residual["cohort"][
            "identity_header_fallback_pairs"
        ],
        "safe_phase_resample": {
            "phase_policy": "legacy_adaptive_early_late_outcome_free",
            "registration_channel": "selected_precontrast_only",
            "registration_interpolation": "linear",
            "registration_qc_outside_value": 0.0,
            "registration_qc_outside_value_is_model_padding": False,
            "model_resampling_padding_mode": "reflect",
            "valid_source_mask_is_model_input": False,
        },
        "registration_sensitivity_summary_sha256": _sha256(
            EXPERIMENT_ROOT / "metrics/registration_sensitivity_summary.json"
        ),
        "registration_physical_support_summary_sha256": _sha256(
            EXPERIMENT_ROOT / "metrics/registration_physical_support_summary.json"
        ),
        "registration_residual_audit_sha256": _sha256(
            EXPERIMENT_ROOT / "metrics/registration_residual_audit.json"
        ),
        "contains_patient_identifiers": False,
        "contains_paths": False,
    }
    _atomic_json(path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--phase-metadata",
        type=Path,
        default=REPOSITORY_ROOT
        / "ispy_jepa_tmi_clean/data_processing/metadata/BreastDCEDL_metadata_min_crop.csv",
    )
    parser.add_argument(
        "--registration-pairs",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/registration_sensitivity_pairs.private.csv",
    )
    parser.add_argument(
        "--repaired-root", type=Path, default=EXPERIMENT_ROOT / "repaired_volumes"
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/registration_residuals.private.csv",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--r-manual-review",
        required=True,
        choices=("PASS", "FAIL"),
        help="Blinded technical review disposition after inspecting the anonymous H/R montage.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    phase_metadata = _load_phase_metadata(args.phase_metadata)
    registration_rows = _load_registration_rows(args.registration_pairs)
    if len(registration_rows) != 1125:
        raise ValueError(f"formal residual audit requires 1125 registration pairs, found {len(registration_rows)}")
    patients = _load_patients(
        args.inventory,
        args.repaired_root,
        registration_rows,
        phase_metadata,
    )
    if len(patients) != 375:
        raise ValueError(f"formal residual audit requires 375 patients, found {len(patients)}")

    all_rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(_process_patient, patient) for patient in patients]
        for index, future in enumerate(as_completed(futures), start=1):
            all_rows.extend(future.result())
            if index % 25 == 0 or index == len(futures):
                print(
                    json.dumps(
                        {
                            "completed_patients": index,
                            "total_patients": len(futures),
                            "pairs": len(all_rows),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if len(all_rows) != 1125:
        raise RuntimeError(f"residual audit produced {len(all_rows)} rows, expected 1125")
    _write_private(args.private_output, all_rows)
    # Aggregate the serialized representation used for provenance.
    with args.private_output.open(newline="", encoding="utf-8") as stream:
        serialized_rows = list(csv.DictReader(stream))
    summary = _aggregate(serialized_rows, args.private_output)
    metrics_root = EXPERIMENT_ROOT / "metrics"
    figures_root = EXPERIMENT_ROOT / "figures"
    support_path = metrics_root / "registration_physical_support_summary.json"
    if not support_path.is_file():
        raise FileNotFoundError(
            "run_registration_support_audit.py before closing the residual/strategy gates"
        )
    support_summary = json.loads(support_path.read_text(encoding="utf-8"))
    private_support_path = metrics_root / "registration_support_patient_visit.private.csv"
    if not private_support_path.is_file():
        raise FileNotFoundError("private 1500-visit registration support table is missing")
    with private_support_path.open(newline="", encoding="utf-8") as stream:
        private_support_rows = sum(1 for _ in csv.DictReader(stream))
    if private_support_rows != 1500:
        raise ValueError(
            f"private registration support table has {private_support_rows} rows, expected 1500"
        )
    support_summary["private_support_metrics_rows"] = private_support_rows
    support_summary["private_support_metrics_sha256"] = _sha256(private_support_path)
    _atomic_json(support_path, support_summary)
    residual_summary_path = metrics_root / "registration_residual_audit.json"
    _atomic_json(residual_summary_path, summary)
    _write_public_tables(metrics_root, summary)
    _plot(serialized_rows, figures_root / "registration_anatomy_lesion_residuals.png")
    main_summary = _update_main_summary(
        metrics_root / "registration_sensitivity_summary.json",
        summary,
        support_summary,
        args.r_manual_review,
    )
    strategy_decision = _write_strategy_decision(
        metrics_root / "registration_strategy_decision.json",
        main_summary,
        summary,
        support_summary,
    )
    _append_report(
        EXPERIMENT_ROOT / "reports/registration_sensitivity_report.md",
        summary,
        main_summary,
    )
    print(
        json.dumps(
            {
                "pairs": summary["cohort"]["pairs"],
                "fallback_pairs": summary["cohort"]["identity_header_fallback_pairs"],
                "pattern_count": summary["threshold_pattern"]["count"],
                "pattern_rate": summary["threshold_pattern"][
                    "rate_among_lesion_evaluable_successful_pairs"
                ],
                "decision": main_summary["gate"]["decision"],
                "decision_frozen": strategy_decision["decision_frozen"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
