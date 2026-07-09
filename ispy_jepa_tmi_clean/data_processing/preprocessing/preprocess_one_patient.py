#!/usr/bin/env python3
"""Preprocess one I-SPY2 patient from DICOM to NIfTI.

This intentionally avoids pydicom for the first pass because the current
environment may not have it. DICOM visit/series discovery is based on the
ASCII metadata strings embedded in the files, while conversion is delegated
to dcm2niix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from path_defaults import dcm2niix_path, ispy2_raw_root


DEFAULT_RAW_ROOT = ispy2_raw_root()
DEFAULT_DCM2NIIX = dcm2niix_path()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "preprocessing" / "outputs"


@dataclass(frozen=True)
class SeriesInfo:
    path: Path
    kind: str
    description: str
    n_dicoms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", default="ISPY2-178649")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dcm2niix", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dcm2niix(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    found = shutil.which("dcm2niix")
    if found:
        return Path(found)
    return dcm2niix_path()


def find_patient_dir(raw_root: Path, patient_id: str) -> Path:
    candidates = [
        raw_root / "ispy2" / patient_id,
        raw_root / "acrin_6698" / patient_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find patient directory for {patient_id!r}")


def first_dicom(series_dir: Path) -> Path | None:
    for path in sorted(series_dir.glob("*.dcm")):
        return path
    return None


def dicom_count(series_dir: Path) -> int:
    return sum(1 for _ in series_dir.glob("*.dcm"))


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def printable_strings(data: bytes, min_len: int = 4) -> list[str]:
    return [
        match.group(0).decode("latin1", errors="ignore").strip()
        for match in re.finditer(rb"[ -~]{%d,}" % min_len, data)
    ]


def series_description(strings: list[str]) -> str:
    preferred = [
        s
        for s in strings
        if "ISPY2" in s and ("VOLSER" in s or "ACRIN_DYN" in s or "T2W" in s)
    ]
    if preferred:
        return preferred[0]
    for s in strings:
        if "VOLSER" in s or "Analysis Mask" in s or "original DCE" in s:
            return s
    return ""


def detect_visit_from_dicom(dicom_path: Path) -> str:
    data = read_bytes(dicom_path)
    patterns = [
        rb"ISPY2_MRI_(T[0-3])",
        rb"ISPY2MRI(T[0-3])",
        rb"ACRIN-6698ISPY2MRI(T[0-3])",
        rb"\d{5,6}_(T[0-3])",
    ]
    for pattern in patterns:
        match = re.search(pattern, data)
        if match:
            return match.group(1).decode("ascii")
    raise ValueError(f"Could not detect visit from {dicom_path}")


def largest_series_dicom(study_dir: Path) -> Path:
    series_dirs = [p for p in study_dir.iterdir() if p.is_dir()]
    if not series_dirs:
        raise FileNotFoundError(f"No series directories in {study_dir}")
    largest = max(
        series_dirs,
        key=lambda p: sum(f.stat().st_size for f in p.glob("*.dcm")),
    )
    dicom = first_dicom(largest)
    if dicom is None:
        raise FileNotFoundError(f"No DICOM files in {largest}")
    return dicom


def discover_visit_series(patient_dir: Path) -> dict[str, dict[str, SeriesInfo]]:
    visits: dict[str, dict[str, SeriesInfo]] = {}
    for study_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        visit = detect_visit_from_dicom(largest_series_dicom(study_dir))
        visit_series: dict[str, SeriesInfo] = {}

        for series_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
            dicom = first_dicom(series_dir)
            if dicom is None:
                continue
            strings = printable_strings(read_bytes(dicom))
            desc = series_description(strings)
            desc_lower = desc.lower()
            if "volser" not in desc_lower:
                continue

            kind = None
            if "original_dce" in desc_lower or "original dce" in desc_lower:
                kind = "original_dce"
            elif "analysis_mask" in desc_lower or "analysis mask" in desc_lower:
                kind = "analysis_mask"

            if kind is not None:
                visit_series[kind] = SeriesInfo(
                    path=series_dir,
                    kind=kind,
                    description=desc,
                    n_dicoms=dicom_count(series_dir),
                )

        visits[visit] = visit_series
    return dict(sorted(visits.items()))


def run_dcm2niix(
    dcm2niix: Path,
    series_dir: Path,
    final_nii: Path,
    final_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if final_nii.exists() and final_json.exists() and not overwrite:
        return {"status": "exists", "stdout": "", "stderr": ""}

    final_nii.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dcm2niix_", dir=final_nii.parent) as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            str(dcm2niix),
            "-z",
            "n",
            "-b",
            "y",
            "-ba",
            "n",
            "-f",
            "%i_%s_%p",
            "-o",
            str(tmp_dir),
            str(series_dir),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"dcm2niix failed for {series_dir}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        nii_files = sorted(tmp_dir.glob("*.nii"))
        json_files = sorted(tmp_dir.glob("*.json"))
        status = "converted"
        if len(nii_files) != 1:
            if not nii_files:
                raise RuntimeError(f"Expected one NIfTI from {series_dir}, found 0")
            by_size = sorted(nii_files, key=lambda path: path.stat().st_size, reverse=True)
            if len(by_size) > 1 and by_size[0].stat().st_size == by_size[1].stat().st_size:
                raise RuntimeError(f"Expected one NIfTI from {series_dir}, found {len(nii_files)}")
            nii_files = [by_size[0]]
            status = f"converted_largest_of_{len(by_size)}"

        expected_json = nii_files[0].with_suffix(".json")
        if expected_json.exists():
            json_files = [expected_json]
        elif len(json_files) != 1:
            raise RuntimeError(f"Expected one JSON sidecar from {series_dir}, found {len(json_files)}")

        if final_nii.exists():
            final_nii.unlink()
        if final_json.exists():
            final_json.unlink()
        nii_files[0].replace(final_nii)
        json_files[0].replace(final_json)

    return {"status": status, "stdout": proc.stdout, "stderr": proc.stderr}


def read_nifti(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with path.open("rb") as f:
        header = f.read(348)

    endian = "<"
    sizeof_hdr = struct.unpack("<i", header[:4])[0]
    if sizeof_hdr != 348:
        sizeof_hdr = struct.unpack(">i", header[:4])[0]
        endian = ">"
    if sizeof_hdr != 348:
        raise ValueError(f"{path} does not look like a NIfTI-1 file")

    dims = struct.unpack(endian + "8h", header[40:56])
    datatype = struct.unpack(endian + "h", header[70:72])[0]
    bitpix = struct.unpack(endian + "h", header[72:74])[0]
    pixdim = struct.unpack(endian + "8f", header[76:108])
    vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])

    dtype_by_code = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        512: np.uint16,
        768: np.uint32,
    }
    dtype = dtype_by_code.get(datatype)
    if dtype is None:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
    dtype = np.dtype(dtype).newbyteorder(endian)

    shape = tuple(int(x) for x in dims[1 : 1 + dims[0]])
    data = np.fromfile(path, dtype=dtype, offset=vox_offset).reshape(shape, order="F")
    meta = {
        "dim": [int(x) for x in dims],
        "shape": list(shape),
        "datatype": int(datatype),
        "bitpix": int(bitpix),
        "pixdim": [float(x) for x in pixdim],
        "vox_offset": int(vox_offset),
    }
    return data, meta


def nifti_dtype_code(dtype: np.dtype) -> tuple[int, int]:
    dtype = np.dtype(dtype)
    code_by_dtype = {
        np.dtype(np.uint8): (2, 8),
        np.dtype(np.int16): (4, 16),
        np.dtype(np.int32): (8, 32),
        np.dtype(np.float32): (16, 32),
        np.dtype(np.float64): (64, 64),
        np.dtype(np.uint16): (512, 16),
        np.dtype(np.uint32): (768, 32),
    }
    if dtype not in code_by_dtype:
        raise ValueError(f"Unsupported output dtype for NIfTI: {dtype}")
    return code_by_dtype[dtype]


def write_nifti_like(reference_path: Path, output_path: Path, array: np.ndarray) -> None:
    """Write a NIfTI using geometry/header bytes from a reference NIfTI."""
    with reference_path.open("rb") as f:
        header = bytearray(f.read(348))

    endian = "<"
    sizeof_hdr = struct.unpack("<i", header[:4])[0]
    if sizeof_hdr != 348:
        sizeof_hdr = struct.unpack(">i", header[:4])[0]
        endian = ">"
    if sizeof_hdr != 348:
        raise ValueError(f"{reference_path} does not look like a NIfTI-1 file")

    vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
    with reference_path.open("rb") as f:
        prefix = bytearray(f.read(vox_offset))

    dtype = np.dtype(array.dtype)
    datatype, bitpix = nifti_dtype_code(dtype)
    dims = [0] * 8
    dims[0] = array.ndim
    for idx, size in enumerate(array.shape, start=1):
        dims[idx] = int(size)
    struct.pack_into(endian + "8h", prefix, 40, *dims)
    struct.pack_into(endian + "h", prefix, 70, int(datatype))
    struct.pack_into(endian + "h", prefix, 72, int(bitpix))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(prefix)
        f.write(np.asfortranarray(array).tobytes(order="F"))


def write_uint8_nifti_like(reference_path: Path, output_path: Path, array: np.ndarray) -> None:
    write_nifti_like(reference_path, output_path, array.astype(np.uint8))


def dicom_ascii_tag_value(data: bytes, group: int, element: int, vr: bytes) -> str | None:
    tag = struct.pack("<HH", group, element) + vr
    offset = data.find(tag)
    if offset < 0:
        return None
    length_offset = offset + 6
    if length_offset + 2 > len(data):
        return None
    length = struct.unpack("<H", data[length_offset : length_offset + 2])[0]
    value_offset = length_offset + 2
    value = data[value_offset : value_offset + length]
    return value.decode("ascii", errors="ignore").strip("\x00 ").strip()


def unique_dicom_acquisition_times(series_dir: Path) -> list[str]:
    values: set[str] = set()
    for path in series_dir.iterdir():
        if not path.is_file():
            continue
        value = dicom_ascii_tag_value(path.read_bytes(), 0x0008, 0x0032, b"TM")
        if value:
            values.add(value)
    return sorted(values)


def align_split_dce_to_mask(
    dce: np.ndarray,
    mask: np.ndarray,
    dce_info: SeriesInfo,
    mask_nii: Path,
    aligned_dce_nii: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if dce.ndim != 4:
        raise ValueError("DCE/mask spatial shape mismatch and DCE is not 4D")
    if dce.shape[:2] != mask.shape[:2]:
        raise ValueError("DCE/mask in-plane shape mismatch")
    if mask.shape[2] % dce.shape[2] != 0:
        raise ValueError("DCE/mask z shape mismatch is not an integer slab split")

    slab_factor = mask.shape[2] // dce.shape[2]
    if slab_factor <= 1 or dce.shape[3] % slab_factor != 0:
        raise ValueError("DCE/mask z split does not match DCE time dimension")

    expected_times = dce.shape[3] // slab_factor
    acquisition_times = unique_dicom_acquisition_times(dce_info.path)
    if len(acquisition_times) != expected_times:
        raise ValueError(
            "DCE/mask spatial mismatch: DICOM acquisition-time count does not "
            f"support slab correction ({len(acquisition_times)} != {expected_times})"
        )

    aligned = np.empty(mask.shape + (expected_times,), dtype=dce.dtype)
    slab_z = dce.shape[2]
    for time_index in range(expected_times):
        for slab_index in range(slab_factor):
            src_time = time_index * slab_factor + slab_index
            z0 = slab_index * slab_z
            z1 = z0 + slab_z
            aligned[:, :, z0:z1, time_index] = dce[:, :, :, src_time]

    write_nifti_like(mask_nii, aligned_dce_nii, aligned)
    _, aligned_meta = read_nifti(aligned_dce_nii)
    adjustment = {
        "method": "concat_consecutive_dce_slabs_by_time",
        "reason": "dcm2niix represented one DCE acquisition as multiple z-slabs in the time dimension",
        "slab_factor": int(slab_factor),
        "dicom_acquisition_times": acquisition_times,
        "aligned_dce_nifti": str(aligned_dce_nii),
        "aligned_dce_shape": aligned_meta["shape"],
    }
    return aligned, adjustment


def value_counts(array: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(array, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def ftv_bbox_from_inverse_mask(mask: np.ndarray) -> dict[str, Any]:
    ftv = mask == 0
    coords = np.argwhere(ftv)
    if coords.size == 0:
        return {
            "ftv_voxels": 0,
            "bbox_nii_xyz_inclusive": None,
            "breastdcedl_like_bbox": None,
        }

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox_nii = {
        "x_min": int(mins[0]),
        "x_max": int(maxs[0]),
        "y_min": int(mins[1]),
        "y_max": int(maxs[1]),
        "z_min": int(mins[2]),
        "z_max": int(maxs[2]),
    }

    # dcm2niix flips the row direction relative to the BreastDCEDL metadata.
    # Keep both the raw NIfTI bbox and a BreastDCEDL-style convenience view.
    n_y = mask.shape[1]
    breastdcedl_like = {
        "mask_start": int(mins[2]),
        "mask_end_inclusive": int(maxs[2]),
        "sraw": int(n_y - 1 - maxs[1]),
        "eraw_exclusive": int(n_y - mins[1]),
        "scol": int(mins[0]),
        "ecol_exclusive": int(maxs[0] + 1),
    }

    return {
        "ftv_voxels": int(coords.shape[0]),
        "bbox_nii_xyz_inclusive": bbox_nii,
        "breastdcedl_like_bbox": breastdcedl_like,
    }


def summarize_visit(
    patient_id: str,
    visit: str,
    visit_out: Path,
    dce_info: SeriesInfo,
    mask_info: SeriesInfo,
    dcm2niix: Path,
    overwrite: bool,
) -> dict[str, Any]:
    dce_nii = visit_out / f"{patient_id}_{visit}_original_DCE.nii"
    dce_json = visit_out / f"{patient_id}_{visit}_original_DCE.json"
    mask_nii = visit_out / f"{patient_id}_{visit}_analysis_mask_raw.nii"
    mask_json = visit_out / f"{patient_id}_{visit}_analysis_mask_raw.json"
    aligned_dce_nii = visit_out / f"{patient_id}_{visit}_original_DCE_aligned.nii"
    ftv_mask_nii = visit_out / f"{patient_id}_{visit}_ftv_mask.nii"

    dce_convert = run_dcm2niix(dcm2niix, dce_info.path, dce_nii, dce_json, overwrite)
    mask_convert = run_dcm2niix(dcm2niix, mask_info.path, mask_nii, mask_json, overwrite)

    dce, dce_meta = read_nifti(dce_nii)
    mask, mask_meta = read_nifti(mask_nii)
    dce_nifti_for_manifest = dce_nii
    dce_layout_adjustment = None
    if dce.shape[:3] != mask.shape[:3]:
        dce, dce_layout_adjustment = align_split_dce_to_mask(
            dce=dce,
            mask=mask,
            dce_info=dce_info,
            mask_nii=mask_nii,
            aligned_dce_nii=aligned_dce_nii,
        )
        dce_nifti_for_manifest = aligned_dce_nii
        _, dce_meta = read_nifti(aligned_dce_nii)
        if dce.shape[:3] != mask.shape[:3]:
            raise ValueError(f"DCE/mask spatial shape mismatch for {patient_id} {visit}")

    ftv_mask = mask == 0
    write_uint8_nifti_like(mask_nii, ftv_mask_nii, ftv_mask)
    bbox = ftv_bbox_from_inverse_mask(mask)
    return {
        "visit": visit,
        "raw_dce_series": str(dce_info.path),
        "raw_mask_series": str(mask_info.path),
        "dce_series_description": dce_info.description,
        "mask_series_description": mask_info.description,
        "dce_dicoms": dce_info.n_dicoms,
        "mask_dicoms": mask_info.n_dicoms,
        "dce_nifti": str(dce_nifti_for_manifest),
        "dce_nifti_dcm2niix": str(dce_nii),
        "mask_nifti": str(mask_nii),
        "ftv_mask_nifti": str(ftv_mask_nii),
        "dce_shape": dce_meta["shape"],
        "mask_shape": mask_meta["shape"],
        "n_z": int(dce.shape[2]),
        "n_times": int(dce.shape[3]) if dce.ndim >= 4 else 1,
        "dce_pixdim": dce_meta["pixdim"],
        "mask_values": value_counts(mask),
        "ftv_mask_values": value_counts(ftv_mask.astype(np.uint8)),
        **bbox,
        "conversion": {
            "dce": dce_convert["status"],
            "mask": mask_convert["status"],
        },
        "dce_layout_adjustment": dce_layout_adjustment,
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "visit",
        "n_z",
        "n_times",
        "dce_shape",
        "mask_shape",
        "ftv_voxels",
        "mask_start",
        "mask_end_inclusive",
        "sraw",
        "eraw_exclusive",
        "scol",
        "ecol_exclusive",
        "dce_nifti",
        "mask_nifti",
        "ftv_mask_nifti",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            bbox = row.get("breastdcedl_like_bbox") or {}
            writer.writerow(
                {
                    "patient_id": row["patient_id"],
                    "visit": row["visit"],
                    "n_z": row["n_z"],
                    "n_times": row["n_times"],
                    "dce_shape": "x".join(map(str, row["dce_shape"])),
                    "mask_shape": "x".join(map(str, row["mask_shape"])),
                    "ftv_voxels": row["ftv_voxels"],
                    "mask_start": bbox.get("mask_start"),
                    "mask_end_inclusive": bbox.get("mask_end_inclusive"),
                    "sraw": bbox.get("sraw"),
                    "eraw_exclusive": bbox.get("eraw_exclusive"),
                    "scol": bbox.get("scol"),
                    "ecol_exclusive": bbox.get("ecol_exclusive"),
                    "dce_nifti": row["dce_nifti"],
                    "mask_nifti": row["mask_nifti"],
                    "ftv_mask_nifti": row["ftv_mask_nifti"],
                }
            )


def main() -> int:
    args = parse_args()
    dcm2niix = resolve_dcm2niix(args.dcm2niix)
    if not dcm2niix.exists() and shutil.which(str(dcm2niix)) is None:
        raise FileNotFoundError(f"dcm2niix not found at {dcm2niix}")

    patient_dir = find_patient_dir(args.raw_root, args.patient_id)
    patient_out = args.output_root / args.patient_id
    patient_out.mkdir(parents=True, exist_ok=True)

    visits = discover_visit_series(patient_dir)
    missing = []
    failed_visits = []
    summaries = []
    for visit in ["T0", "T1", "T2", "T3"]:
        series = visits.get(visit)
        if not series:
            missing.append(f"{visit}: no study")
            continue
        if "original_dce" not in series:
            missing.append(f"{visit}: missing original_DCE")
            continue
        if "analysis_mask" not in series:
            missing.append(f"{visit}: missing Analysis_Mask")
            continue

        print(f"Processing {args.patient_id} {visit}")
        try:
            summary = summarize_visit(
                patient_id=args.patient_id,
                visit=visit,
                visit_out=patient_out / visit,
                dce_info=series["original_dce"],
                mask_info=series["analysis_mask"],
                dcm2niix=dcm2niix,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failed_visits.append(
                {
                    "visit": visit,
                    "error": repr(exc),
                    "raw_dce_series": str(series["original_dce"].path),
                    "raw_mask_series": str(series["analysis_mask"].path),
                }
            )
            print(f"Failed {args.patient_id} {visit}: {exc}")
            continue
        summary["patient_id"] = args.patient_id
        summaries.append(summary)

    manifest = {
        "patient_id": args.patient_id,
        "raw_root": str(args.raw_root),
        "raw_patient_dir": str(patient_dir),
        "output_dir": str(patient_out),
        "dcm2niix": str(dcm2niix),
        "missing": missing,
        "failed_visits": failed_visits,
        "visits": summaries,
    }
    manifest_path = patient_out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_csv(summaries, patient_out / "visit_summary.csv")

    print(f"Wrote {manifest_path}")
    print(f"Wrote {patient_out / 'visit_summary.csv'}")
    if missing:
        print("Missing:", "; ".join(missing))
    if failed_visits:
        print("Failed visits:", "; ".join(v["visit"] for v in failed_visits))
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
