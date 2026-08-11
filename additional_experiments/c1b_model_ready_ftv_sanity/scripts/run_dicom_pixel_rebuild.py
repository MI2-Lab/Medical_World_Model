#!/usr/bin/env python3
"""Rebuild every singular I-SPY2 model-input visit from raw PixelData.

Patient-level paths, identifiers, UIDs, and hashes are written only beneath
gitignored private locations. Public CSV/JSON/Markdown contain aggregates.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from c1b_sanity.dicom_pixel_rebuild import (  # noqa: E402
    DicomPixelRebuildError,
    rebuild_classic_dce_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "repaired_volumes",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/repair_private",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _audit_name(patient_id: str, visit: str) -> str:
    token = hashlib.sha256(f"{patient_id}|{visit}".encode("utf-8")).hexdigest()
    return f"{token}.json"


def _load_raw_series(value: str) -> Path:
    decoded = json.loads(value)
    if not isinstance(decoded, str):
        raise ValueError("I-SPY2 classic DCE raw series must be one directory")
    path = Path(decoded)
    if not path.is_dir():
        raise FileNotFoundError("raw DICOM series directory is missing")
    return path


def _existing_pass(audit_path: Path, output_path: Path) -> dict[str, Any] | None:
    if not audit_path.is_file() or not output_path.is_file():
        return None
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "PASS":
        return None
    private = payload.get("private") or {}
    if private.get("output_nifti_sha256") != sha256_file(output_path):
        return None
    return payload


def rebuild_one(job: dict[str, Any]) -> dict[str, Any]:
    patient_id = str(job["patient_id"])
    visit = str(job["visit"])
    output_path = Path(job["output_path"])
    audit_path = Path(job["audit_path"])
    if not bool(job["overwrite"]):
        cached = _existing_pass(audit_path, output_path)
        if cached is not None:
            return cached
        if audit_path.exists() and not bool(job["retry_failures"]):
            previous = json.loads(audit_path.read_text(encoding="utf-8"))
            if previous.get("status") == "FAIL":
                return previous

    started = time.time()
    private_identity = {
        "patient_id": patient_id,
        "visit": visit,
        "cohort": str(job["cohort"]),
        "formal_ftv_overlap": bool(job["formal_ftv_overlap"]),
    }
    try:
        dce = nib.load(str(job["dce_nifti"]), mmap=False)
        reference = nib.load(str(job["ftv_mask_nifti"]), mmap=False)
        expected_shape = tuple(int(value) for value in dce.shape)
        if len(expected_shape) != 4:
            raise ValueError("expected DCE NIfTI is not four-dimensional")
        reference_shape = tuple(int(value) for value in reference.shape[:3])
        reference_affine = np.asarray(reference.affine, dtype=np.float64)
        reference_spacing = tuple(
            float(value) for value in np.linalg.norm(reference_affine[:3, :3], axis=0)
        )
        result = rebuild_classic_dce_series(
            _load_raw_series(str(job["raw_dce_series_json"])),
            expected_shape_xyzt=expected_shape,
            reference_affine_ras=reference_affine,
            reference_shape_xyz=reference_shape,
            expected_spacing_xyz_mm=reference_spacing,
            output_nifti=output_path,
            overwrite_output=bool(job["overwrite"]),
            include_private=True,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            **private_identity,
            "elapsed_seconds": time.time() - started,
            "public_metrics": result.public_metrics(),
            "private": result.private,
        }
    except Exception as exc:  # persist fail-closed evidence before propagating
        code = exc.code if isinstance(exc, DicomPixelRebuildError) else type(exc).__name__
        payload = {
            "schema_version": 1,
            "status": "FAIL",
            **private_identity,
            "elapsed_seconds": time.time() - started,
            "error_code": str(code),
            "error_message": str(exc),
        }
    atomic_json(audit_path, payload)
    return payload


def public_row(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("public_metrics") or {}
    nifti = metrics.get("nifti_write") or {}
    return {
        "scope": "FORMAL_72" if payload.get("formal_ftv_overlap") else "BASE_ONLY_EXTENSION_74",
        "visit": str(payload.get("visit")),
        "status": str(payload.get("status")),
        "error_code": str(payload.get("error_code", "")),
        "file_count": int(metrics.get("file_count", 0)),
        "slice_count": int(metrics.get("slice_count", 0)),
        "timepoint_count": int(metrics.get("timepoint_count", 0)),
        "decoded_cell_count": int(metrics.get("decoded_cell_count", 0)),
        "verified_cell_count": int(metrics.get("verified_cell_count", 0)),
        "pixel_order_verified": bool(metrics.get("pixel_order_verified", False)),
        "finite_fraction": float(metrics.get("finite_fraction", float("nan"))),
        "nonconstant": bool(metrics.get("nonconstant", False)),
        "cell_recomparison_max_abs_error": float(
            metrics.get("cell_recomparison_max_abs_error", float("nan"))
        ),
        "center_corner_error_mm": float(
            metrics.get("reference_center_corner_hausdorff_mm", float("nan"))
        ),
        "footprint_corner_error_mm": float(
            metrics.get("reference_footprint_corner_hausdorff_mm", float("nan"))
        ),
        "qform_code": int(nifti.get("qform_code", 0)),
        "sform_code": int(nifti.get("sform_code", 0)),
        "elapsed_seconds": float(payload.get("elapsed_seconds", float("nan"))),
    }


def aggregate(rows: pd.DataFrame, scope: str) -> dict[str, Any]:
    subset = rows if scope == "ALL" else rows.loc[rows["scope"].eq(scope)]
    passed = subset.loc[subset["status"].eq("PASS")]

    def finite_max(column: str) -> float | None:
        values = pd.to_numeric(passed[column], errors="coerce")
        return float(values.max()) if values.notna().any() else None

    return {
        "scope": scope,
        "visits": int(len(subset)),
        "passed": int(subset["status"].eq("PASS").sum()),
        "failed": int(subset["status"].ne("PASS").sum()),
        "dicom_files": int(passed["file_count"].sum()),
        "decoded_cells": int(passed["decoded_cell_count"].sum()),
        "verified_cells": int(passed["verified_cell_count"].sum()),
        "pixel_order_verified_fraction": float(passed["pixel_order_verified"].mean())
        if len(passed)
        else 0.0,
        "max_cell_error": finite_max("cell_recomparison_max_abs_error"),
        "max_center_corner_error_mm": finite_max("center_corner_error_mm"),
        "max_footprint_corner_error_mm": finite_max("footprint_corner_error_mm"),
        "all_finite_nonconstant": bool(
            len(passed) == len(subset)
            and passed["finite_fraction"].eq(1.0).all()
            and passed["nonconstant"].all()
        ),
        "all_qform_sform_valid": bool(
            len(passed) == len(subset)
            and passed["qform_code"].gt(0).all()
            and passed["sform_code"].gt(0).all()
        ),
    }


def write_public_outputs(rows: pd.DataFrame, inventory_hash: str) -> dict[str, Any]:
    metrics = EXPERIMENT_ROOT / "metrics"
    reports = EXPERIMENT_ROOT / "reports"
    metrics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    # Per-visit public rows have no identifier/path but are still randomized to
    # avoid preserving private input order.
    public = rows.sample(frac=1.0, random_state=2026).reset_index(drop=True)
    public.to_csv(metrics / "dicom_pixel_rebuild_visit_qc.csv", index=False)
    aggregates = [aggregate(rows, scope) for scope in ("FORMAL_72", "BASE_ONLY_EXTENSION_74", "ALL")]
    pd.DataFrame(aggregates).to_csv(metrics / "dicom_pixel_rebuild_summary.csv", index=False)
    formal, extension, overall = aggregates
    gate = {
        "schema_version": 1,
        "status": "PASS"
        if formal["passed"] == 72
        and formal["failed"] == 0
        and overall["failed"] == 0
        and formal["pixel_order_verified_fraction"] == 1.0
        else "FAIL",
        "required_formal_visits": 72,
        "formal": formal,
        "base_only_extension": extension,
        "all_model_input_singular_visits": overall,
        "inventory_sha256": inventory_hash,
        "patient_identifiers_in_public_outputs": False,
    }
    atomic_json(metrics / "dicom_pixel_rebuild_gate.json", gate)
    report = f"""# Raw-DICOM PixelData 重建报告

## 结论

正式 375 人范围内要求的 72 个 singular-sform visit：**{formal['passed']}/72 PASS**。为了不让 matched base-training population 的新输入臂接收坏 geometry，另对 375 人之外发现的 74 个 visit 应用完全相同的 fail-closed rebuild：**{extension['passed']}/74 PASS**。综合 gate：**{gate['status']}**。

## 验收

| scope | visits | pass | fail | DICOM cells | verified cells | max cell error | max footprint corner error (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 正式72 | {formal['visits']} | {formal['passed']} | {formal['failed']} | {formal['decoded_cells']} | {formal['verified_cells']} | {formal['max_cell_error']} | {formal['max_footprint_corner_error_mm']} |
| base-only扩展74 | {extension['visits']} | {extension['passed']} | {extension['failed']} | {extension['decoded_cells']} | {extension['verified_cells']} | {extension['max_cell_error']} | {extension['max_footprint_corner_error_mm']} |
| 全部 | {overall['visits']} | {overall['passed']} | {overall['failed']} | {overall['decoded_cells']} | {overall['verified_cells']} | {overall['max_cell_error']} | {overall['max_footprint_corner_error_mm']} |

每个 series 均要求完整且唯一的 TPI/AcquisitionTime x IPP-slice cell、逐文件 scaling、finite/nonconstant float32 volume、第二次独立 PixelData decode 后逐 cell exact compare、与 reference mask 的 center/footprint corner误差不超过0.1 mm，并在写出后验证 qform/sform。患者级路径、UID、cell hash和输出 hash只保存在 gitignored private sidecar；公开文件无患者身份。
"""
    (reports / "dicom_pixel_rebuild_report.md").write_text(report, encoding="utf-8")
    return gate


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    inventory = pd.read_csv(args.inventory)
    inventory = inventory.loc[inventory["pixel_rebuild_required"].astype(bool)].copy()
    if len(inventory) != 146:
        raise ValueError(f"Expected 146 model-input rebuild visits, found {len(inventory)}")
    if int(inventory["formal_ftv_overlap"].astype(bool).sum()) != 72:
        raise ValueError("Expected exactly 72 formal FTV rebuild visits")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.audit_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        patient_id, visit = str(row.patient_id), str(row.visit)
        jobs.append(
            {
                **row._asdict(),
                "output_path": str(
                    args.output_root / str(row.cohort) / patient_id / f"{visit}_dce_rebuilt.nii.gz"
                ),
                "audit_path": str(args.audit_root / _audit_name(patient_id, visit)),
                "overwrite": args.overwrite,
                "retry_failures": args.retry_failures,
            }
        )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(rebuild_one, job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            payload = future.result()
            results.append(payload)
            completed += 1
            print(
                f"[{completed:03d}/{len(jobs)}] {payload['status']} "
                f"scope={'formal' if payload.get('formal_ftv_overlap') else 'extension'} "
                f"visit={payload.get('visit')} elapsed={payload.get('elapsed_seconds', 0):.1f}s",
                flush=True,
            )

    rows = pd.DataFrame([public_row(payload) for payload in results])
    gate = write_public_outputs(rows, sha256_file(args.inventory))
    print(json.dumps(gate, indent=2, sort_keys=True))
    if gate["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
