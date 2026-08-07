#!/usr/bin/env python3
"""Preprocess I-SPY1 DCE-MRI into the manifest format used by the JEPA code."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom

from preprocess_one_patient import read_nifti, resolve_dcm2niix, run_dcm2niix, write_nifti_like
from path_defaults import ispy1_preprocessed_root, ispy1_raw_root


DEFAULT_RAW_ROOT = ispy1_raw_root()
DEFAULT_OUTPUT_ROOT = ispy1_preprocessed_root()
DEFAULT_CLINICAL_XLSX = DEFAULT_RAW_ROOT / "I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx"
VISITS = ("T0", "T1", "T2", "T3")


@dataclass(frozen=True)
class SeriesHeader:
    path: Path
    n_dicoms: int
    description: str
    study_date: str
    series_number: int
    acquisition_time: str
    rows: int
    columns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clinical-xlsx", type=Path, default=DEFAULT_CLINICAL_XLSX)
    parser.add_argument("--dcm2niix", type=Path, default=None)
    parser.add_argument("--patient-id", action="append", default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only preprocess patients whose existing manifest is missing or incomplete.",
    )
    return parser.parse_args()


def first_dicom(series_dir: Path) -> Path | None:
    for path in sorted(series_dir.glob("*.dcm")):
        return path
    return None


def dicom_count(series_dir: Path) -> int:
    return sum(1 for _ in series_dir.glob("*.dcm"))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def read_series_header(series_dir: Path) -> SeriesHeader | None:
    dicom = first_dicom(series_dir)
    if dicom is None:
        return None
    ds = pydicom.dcmread(str(dicom), stop_before_pixels=True, force=True)
    return SeriesHeader(
        path=series_dir,
        n_dicoms=dicom_count(series_dir),
        description=str(getattr(ds, "SeriesDescription", "") or ""),
        study_date=str(getattr(ds, "AcquisitionDate", "") or getattr(ds, "StudyDate", "") or ""),
        series_number=safe_int(getattr(ds, "SeriesNumber", 0)),
        acquisition_time=str(getattr(ds, "AcquisitionTime", "") or getattr(ds, "ContentTime", "") or ""),
        rows=safe_int(getattr(ds, "Rows", 0)),
        columns=safe_int(getattr(ds, "Columns", 0)),
    )


def read_patient_headers(patient_dir: Path) -> dict[Path, list[SeriesHeader]]:
    studies: dict[Path, list[SeriesHeader]] = {}
    for study_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
        headers = []
        for series_dir in sorted(path for path in study_dir.iterdir() if path.is_dir()):
            header = read_series_header(series_dir)
            if header is not None:
                headers.append(header)
        if headers:
            studies[study_dir] = headers
    return studies


def study_sort_key(item: tuple[Path, list[SeriesHeader]]) -> tuple[str, str]:
    study_dir, headers = item
    dates = [header.study_date for header in headers if header.study_date]
    return (min(dates) if dates else "", study_dir.name)


def excluded_dce_description(description: str) -> bool:
    text = description.lower()
    excluded_terms = (
        "segmentation",
        "breast tissue",
        "subtract",
        "subtraction",
        ": ser",
        ": pe1",
        " pe1",
        "t2",
        "scout",
        "locator",
        "loc.",
        "diffusion",
        "dwssfse",
        "fiesta",
        "pjn",
    )
    return any(term in text for term in excluded_terms)


def raw_dce_score(header: SeriesHeader) -> int:
    text = header.description.lower()
    if excluded_dce_description(header.description):
        return -1
    score = 0
    if "dynamic" in text or "3dfgre" in text:
        score += 1000
    if "ir-spgr" in text or "spgr" in text:
        score += 800
    if "fl3d_t1_sag_ca" in text:
        score += 600
    if "pass" in text:
        score += 400
    if header.n_dicoms >= 80:
        score += 200
    if header.n_dicoms >= 20:
        score += 50
    return score


def select_dce(headers: list[SeriesHeader]) -> tuple[str, list[SeriesHeader]]:
    candidates = [header for header in headers if raw_dce_score(header) > 0]
    if not candidates:
        raise RuntimeError("No raw DCE-like series found.")

    long_candidates = [header for header in candidates if header.n_dicoms >= 80]
    if long_candidates:
        best = max(long_candidates, key=lambda header: (raw_dce_score(header), header.n_dicoms))
        return "dynamic", [best]

    phase_candidates = [header for header in candidates if header.n_dicoms >= 10]
    groups: dict[tuple[int, int, int], list[SeriesHeader]] = {}
    for header in phase_candidates:
        groups.setdefault((header.n_dicoms, header.rows, header.columns), []).append(header)
    grouped_candidates = [
        sorted(
            group,
            key=lambda header: (header.series_number, header.acquisition_time, header.path.name),
        )
        for group in groups.values()
        if len(group) >= 2
    ]
    if grouped_candidates:
        phase_candidates = max(
            grouped_candidates,
            key=lambda group: (
                len(group),
                group[0].rows * group[0].columns,
                group[0].n_dicoms,
                sum(raw_dce_score(header) for header in group),
            ),
        )
    else:
        phase_candidates = sorted(
            phase_candidates,
            key=lambda header: (header.series_number, header.acquisition_time, header.path.name),
        )
    phase_candidates = sorted(
        phase_candidates,
        key=lambda header: (header.series_number, header.acquisition_time, header.path.name),
    )
    if len(phase_candidates) >= 2:
        return "phase_stack", phase_candidates
    raise RuntimeError("Insufficient DCE phase series.")


def converted_nifti(tmp_dir: Path) -> tuple[Path, Path]:
    nii_files = sorted(tmp_dir.glob("*.nii"))
    json_files = sorted(tmp_dir.glob("*.json"))
    if not nii_files:
        raise RuntimeError(f"dcm2niix produced no NIfTI in {tmp_dir}")
    nii_files = sorted(nii_files, key=lambda path: path.stat().st_size, reverse=True)
    nii = nii_files[0]
    sidecar = nii.with_suffix(".json")
    if not sidecar.exists():
        if len(json_files) != 1:
            raise RuntimeError(f"Could not identify JSON sidecar in {tmp_dir}")
        sidecar = json_files[0]
    return nii, sidecar


def run_dcm2niix_collect(
    dcm2niix: Path,
    series_dir: Path,
    tmp_dir: Path,
    filename: str,
) -> tuple[list[Path], list[Path], subprocess.CompletedProcess[str]]:
    cmd = [
        str(dcm2niix),
        "-z",
        "n",
        "-b",
        "y",
        "-ba",
        "n",
        "-f",
        filename,
        "-o",
        str(tmp_dir),
        str(series_dir),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"dcm2niix failed for {series_dir}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return sorted(tmp_dir.glob("*.nii")), sorted(tmp_dir.glob("*.json")), proc


def write_stacked_niftis(nii_files: list[Path], final_nii: Path, final_json: Path, payload: dict[str, Any]) -> None:
    arrays = []
    for nii in nii_files:
        data, _ = read_nifti(nii)
        if data.ndim == 3:
            data = data[..., None]
        if data.ndim != 4:
            raise RuntimeError(f"Expected a 3D/4D NIfTI for stacking, got {data.shape} from {nii}")
        if arrays and data.shape[:3] != arrays[0].shape[:3]:
            raise RuntimeError(f"Split NIfTI shape mismatch: {data.shape[:3]} != {arrays[0].shape[:3]}")
        arrays.append(data)
    stacked = np.concatenate(arrays, axis=3)
    write_nifti_like(nii_files[0], final_nii, stacked)
    final_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def convert_dynamic(
    dcm2niix: Path,
    header: SeriesHeader,
    final_nii: Path,
    final_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if final_nii.exists() and final_json.exists() and not overwrite:
        return {"status": "exists"}
    final_nii.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ispy1_dynamic_", dir=final_nii.parent) as tmp:
        tmp_dir = Path(tmp)
        nii_files, json_files, proc = run_dcm2niix_collect(dcm2niix, header.path, tmp_dir, "dynamic")
        if not nii_files:
            raise RuntimeError(f"dcm2niix produced no NIfTI for {header.path}")
        if len(nii_files) == 1:
            sidecar = nii_files[0].with_suffix(".json")
            if not sidecar.exists():
                if len(json_files) != 1:
                    raise RuntimeError(f"Could not identify JSON sidecar for {header.path}")
                sidecar = json_files[0]
            shutil.move(str(nii_files[0]), str(final_nii))
            shutil.move(str(sidecar), str(final_json))
            return {"status": "converted", "stdout": proc.stdout, "stderr": proc.stderr}
        sizes = {path.stat().st_size for path in nii_files}
        if len(sizes) == 1:
            write_stacked_niftis(
                nii_files,
                final_nii,
                final_json,
                {
                    "Conversion": "dynamic_split_stack",
                    "SourceSeries": str(header.path),
                    "SourceNiftis": [str(path) for path in nii_files],
                    "SourceJsons": [str(path) for path in json_files],
                },
            )
            return {"status": f"converted_split_stack_{len(nii_files)}", "stdout": proc.stdout, "stderr": proc.stderr}
        largest = max(nii_files, key=lambda path: path.stat().st_size)
        sidecar = largest.with_suffix(".json")
        if not sidecar.exists() and len(json_files) == 1:
            sidecar = json_files[0]
        if not sidecar.exists():
            raise RuntimeError(f"Could not identify JSON sidecar for largest NIfTI from {header.path}")
        shutil.move(str(largest), str(final_nii))
        shutil.move(str(sidecar), str(final_json))
        return {"status": f"converted_largest_of_{len(nii_files)}", "stdout": proc.stdout, "stderr": proc.stderr}


def convert_phase_stack(
    dcm2niix: Path,
    headers: list[SeriesHeader],
    final_nii: Path,
    final_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if final_nii.exists() and final_json.exists() and not overwrite:
        return {"status": "exists"}
    final_nii.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ispy1_phase_", dir=final_nii.parent) as tmp:
        tmp_dir = Path(tmp)
        volumes = []
        first_nii: Path | None = None
        phase_jsons = []
        for idx, header in enumerate(headers):
            phase_dir = tmp_dir / f"phase_{idx:02d}"
            phase_dir.mkdir()
            run_dcm2niix(
                dcm2niix=dcm2niix,
                series_dir=header.path,
                final_nii=phase_dir / "phase.nii",
                final_json=phase_dir / "phase.json",
                overwrite=True,
            )
            nii = phase_dir / "phase.nii"
            sidecar = phase_dir / "phase.json"
            data, _ = read_nifti(nii)
            if data.ndim == 4 and data.shape[-1] == 1:
                data = data[..., 0]
            if data.ndim != 3:
                raise RuntimeError(f"Expected a 3D phase NIfTI from {header.path}, got {data.shape}")
            if volumes and data.shape != volumes[0].shape:
                raise RuntimeError(
                    f"Phase shape mismatch for {header.path}: {data.shape} != {volumes[0].shape}"
                )
            volumes.append(data)
            if first_nii is None:
                first_nii = nii
            phase_jsons.append(str(sidecar))
        assert first_nii is not None
        stacked = np.stack(volumes, axis=3)
        write_nifti_like(first_nii, final_nii, stacked)
        payload = {
            "Conversion": "phase_stack",
            "SourceSeries": [
                {
                    "path": str(header.path),
                    "description": header.description,
                    "n_dicoms": header.n_dicoms,
                    "series_number": header.series_number,
                    "acquisition_time": header.acquisition_time,
                }
                for header in headers
            ],
            "SourceJsons": phase_jsons,
        }
        final_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"status": "converted_phase_stack", "n_phases": len(headers)}


def convert_dce(
    dcm2niix: Path,
    mode: str,
    headers: list[SeriesHeader],
    final_nii: Path,
    final_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if mode == "dynamic":
        return convert_dynamic(dcm2niix, headers[0], final_nii, final_json, overwrite)
    if mode == "phase_stack":
        return convert_phase_stack(dcm2niix, headers, final_nii, final_json, overwrite)
    raise ValueError(f"Unknown DCE selection mode: {mode}")


def load_clinical_table(path: Path) -> pd.DataFrame:
    clinical = pd.read_excel(path, sheet_name="TCIA Patient Clinical Subset")
    outcome = pd.read_excel(path, sheet_name="TCIA Outcomes Subset")
    clinical["patient_id"] = "ISPY1_" + clinical["SUBJECTID"].astype(int).astype(str)
    outcome["patient_id"] = "ISPY1_" + outcome["SUBJECTID"].astype(int).astype(str)
    return clinical.merge(outcome, on="patient_id", suffixes=("_clinical", "_outcome"))


def label_row(row: pd.Series, output_root: Path) -> dict[str, Any]:
    patient_id = str(row["patient_id"])
    manifest_path = output_root / patient_id / "manifest.json"
    complete = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            complete = not manifest.get("missing") and not manifest.get("failed_visits") and len(manifest.get("visits", [])) == 4
        except (OSError, json.JSONDecodeError):
            complete = False
    return {
        "patient_id": patient_id,
        "cohort": "I-SPY1",
        "label_pcr": int(row["PCR"]) if pd.notna(row.get("PCR")) else "",
        "arm": "ISPY1_NACT",
        "label_hr": int(row["HR Pos"]) if pd.notna(row.get("HR Pos")) else 0,
        "label_her2": int(row["Her2MostPos"]) if pd.notna(row.get("Her2MostPos")) else 0,
        "label_mp": 0,
        "age_at_screening": float(row["age"]) if pd.notna(row.get("age")) else "",
        "complete_4visits": bool(complete and pd.notna(row.get("PCR"))),
        "hr_her2_status": row.get("HR_HER2_STATUS", ""),
        "mri_ld_baseline": row.get("MRI LD Baseline", ""),
        "mri_ld_1_3dac": row.get("MRI LD 1-3dAC", ""),
        "mri_ld_interreg": row.get("MRI LD InterReg", ""),
        "mri_ld_presurg": row.get("MRI LD PreSurg", ""),
        "rcb_class": row.get("RCBClass", ""),
    }


def complete_manifest(output_root: Path, patient_id: str) -> bool:
    manifest_path = output_root / patient_id / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        not manifest.get("missing")
        and not manifest.get("failed_visits")
        and len(manifest.get("visits", [])) == 4
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preprocess_patient(
    patient_id: str,
    raw_root: Path,
    output_root: Path,
    dcm2niix: Path,
    overwrite: bool,
) -> dict[str, Any]:
    patient_dir = raw_root / "ispy1" / patient_id
    patient_out = output_root / patient_id
    manifest_path = patient_out / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return {"patient_id": patient_id, "status": "exists", "manifest": str(manifest_path)}
    if not patient_dir.exists():
        return {"patient_id": patient_id, "status": "missing_patient_dir"}

    studies = sorted(read_patient_headers(patient_dir).items(), key=study_sort_key)
    if len(studies) < 4:
        return {"patient_id": patient_id, "status": "missing_studies", "n_studies": len(studies)}
    selected_studies = studies[:4]

    visits = []
    failed: list[str] = []
    for visit_name, (study_dir, headers) in zip(VISITS, selected_studies):
        visit_out = patient_out / visit_name
        final_nii = visit_out / f"{patient_id}_{visit_name}_original_DCE.nii"
        final_json = visit_out / f"{patient_id}_{visit_name}_original_DCE.json"
        try:
            mode, dce_headers = select_dce(headers)
            conversion = convert_dce(dcm2niix, mode, dce_headers, final_nii, final_json, overwrite)
            dce, meta = read_nifti(final_nii)
            if dce.ndim == 3:
                dce = dce[..., None]
                write_nifti_like(final_nii, final_nii, dce)
                dce, meta = read_nifti(final_nii)
            if dce.shape[-1] < 2:
                raise RuntimeError(f"DCE has fewer than 2 phases: {dce.shape}")
            visits.append(
                {
                    "visit": visit_name,
                    "study_dir": str(study_dir),
                    "study_date": min(header.study_date for header in headers if header.study_date),
                    "raw_dce_series": [str(header.path) for header in dce_headers],
                    "dce_selection_mode": mode,
                    "dce_series_description": " | ".join(header.description for header in dce_headers),
                    "dce_dicoms": int(sum(header.n_dicoms for header in dce_headers)),
                    "dce_nifti": str(final_nii),
                    "dce_nifti_dcm2niix": str(final_nii),
                    "mask_nifti": None,
                    "ftv_mask_nifti": None,
                    "dce_shape": meta["shape"],
                    "n_z": int(dce.shape[2]) if dce.ndim >= 3 else 0,
                    "n_times": int(dce.shape[3]) if dce.ndim >= 4 else 1,
                    "dce_pixdim": meta["pixdim"],
                    "mask_values": {},
                    "ftv_mask_values": {},
                    "ftv_voxels": 0,
                    "bbox_nii_xyz_inclusive": None,
                    "breastdcedl_like_bbox": None,
                    "conversion": {"dce": conversion.get("status", "converted"), "mask": "not_available"},
                    "patient_id": patient_id,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch preprocessing alive.
            failed.append(f"{visit_name}: {exc}")

    manifest = {
        "patient_id": patient_id,
        "cohort": "I-SPY1",
        "raw_root": str(raw_root),
        "raw_patient_dir": str(patient_dir),
        "output_dir": str(patient_out),
        "dcm2niix": str(dcm2niix),
        "missing": [] if len(visits) == 4 else ["incomplete_visits"],
        "failed_visits": failed,
        "visits": visits,
    }
    patient_out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "patient_id": patient_id,
        "status": "ok" if not manifest["missing"] and not failed else "failed",
        "n_visits": len(visits),
        "failed_visits": "; ".join(failed),
        "manifest": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    dcm2niix = resolve_dcm2niix(args.dcm2niix)
    clinical = load_clinical_table(args.clinical_xlsx)
    clinical = clinical[clinical["PCR"].notna()].copy()
    clinical = clinical.sort_values("patient_id")
    if args.patient_id:
        wanted = set(args.patient_id)
        clinical = clinical[clinical["patient_id"].isin(wanted)].copy()
    if args.failed_only:
        clinical = clinical[
            ~clinical["patient_id"].astype(str).map(lambda patient_id: complete_manifest(args.output_root, patient_id))
        ].copy()
    if args.max_patients is not None:
        clinical = clinical.head(args.max_patients)

    rows = []
    if not args.labels_only:
        for patient_id in clinical["patient_id"].astype(str).tolist():
            result = preprocess_patient(patient_id, args.raw_root, args.output_root, dcm2niix, args.overwrite)
            rows.append(result)
            print(json.dumps(result, ensure_ascii=False))
        write_rows(args.output_root / "_ispy1_preprocess_summary.csv", rows)

    label_rows = [label_row(row, args.output_root) for _, row in clinical.iterrows()]
    write_rows(args.output_root / "clinical_labels_complete4visits.csv", label_rows)
    n_complete = sum(bool(row["complete_4visits"]) for row in label_rows)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "labels_csv": str(args.output_root / "clinical_labels_complete4visits.csv"),
                "n_labels": len(label_rows),
                "n_complete_4visits": n_complete,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
