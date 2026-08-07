#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nifti import read_nifti

from path_defaults import breastdcedl_metadata_csv, ispy2_preprocessed_root


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    label_pcr: int
    manifest_path: Path


def load_complete4_records(
    labels_csv: Path,
    output_root: Path,
    max_patients: int | None = None,
) -> list[PatientRecord]:
    labels = pd.read_csv(labels_csv)
    labels = labels[labels["complete_4visits"].astype(str).str.lower() == "true"].copy()
    labels = labels.sort_values("patient_id")
    if max_patients is not None:
        labels = labels.head(max_patients)

    records: list[PatientRecord] = []
    for row in labels.itertuples(index=False):
        patient_id = str(row.patient_id)
        manifest_path = output_root / patient_id / "manifest.json"
        if not manifest_path.exists():
            continue
        records.append(
            PatientRecord(
                patient_id=patient_id,
                label_pcr=int(row.label_pcr),
                manifest_path=manifest_path,
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute BreastDCEDL MinCrop-style T0 RGB slice cache from I-SPY2 original_DCE NIfTI files."
    )
    parser.add_argument("--output-root", type=Path, default=ispy2_preprocessed_root())
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument(
        "--breastdcedl-metadata-csv",
        type=Path,
        default=breastdcedl_metadata_csv(),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ispy2_preprocessed_root() / "_breastdcedl_mincrop_t0_raw224_v1",
    )
    parser.add_argument("--visit", choices=("T0", "T1", "T2", "T3"), default="T0")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def minmax_global(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32, copy=False)
    hi = float(np.nanmax(rgb))
    lo = float(np.nanmin(rgb))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(rgb, dtype=np.float32)
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    pos = np.nonzero(mask)
    if len(pos[0]) == 0:
        return None
    return (
        int(pos[0].min()),
        int(pos[0].max()) + 1,
        int(pos[1].min()),
        int(pos[1].max()) + 1,
        int(pos[2].min()),
        int(pos[2].max()) + 1,
    )


def orient_slice(raw_xy: np.ndarray) -> np.ndarray:
    # Match BreastDCEDL's displayed image convention: x is column, y is image row
    # after flipping the NIfTI y-axis.
    return np.flip(raw_xy.T, axis=0)


def safe_crop(rgb: np.ndarray, center_x: float, center_y: float, crop_size: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    half = crop_size // 2
    left = int(round(center_x)) - half
    top = int(round(center_y)) - half
    right = left + crop_size
    bottom = top + crop_size

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    crop = rgb[top:bottom, left:right]
    if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
        return crop

    out = np.zeros((crop_size, crop_size, rgb.shape[2]), dtype=rgb.dtype)
    copy_h = min(crop.shape[0], crop_size)
    copy_w = min(crop.shape[1], crop_size)
    top_out = (crop_size - copy_h) // 2
    left_out = (crop_size - copy_w) // 2
    top_in = max((crop.shape[0] - copy_h) // 2, 0)
    left_in = max((crop.shape[1] - copy_w) // 2, 0)
    out[top_out : top_out + copy_h, left_out : left_out + copy_w] = crop[
        top_in : top_in + copy_h,
        left_in : left_in + copy_w,
    ]
    return out


def choose_phases(patient_id: str, n_times: int, meta_row: dict[str, Any] | None) -> tuple[int, int, int]:
    if meta_row is not None:
        phases = (
            int(meta_row["pre"]),
            int(meta_row["post_early"]),
            int(meta_row["post_late"]),
        )
    else:
        phases = (0, min(2, n_times - 1), min(max(n_times - 2, 0), n_times - 1))
    return tuple(int(np.clip(phase, 0, n_times - 1)) for phase in phases)


def process_one(task: dict[str, Any]) -> dict[str, Any]:
    patient_id = task["patient_id"]
    visit = task["visit"]
    output_root = Path(task["output_root"])
    cache_dir = Path(task["cache_dir"])
    crop_size = int(task["crop_size"])
    image_size = int(task["image_size"])
    overwrite = bool(task["overwrite"])
    meta_row = task.get("meta_row")
    label_pcr = int(task["label_pcr"])

    patient_dir = output_root / patient_id / visit
    dce_path = patient_dir / f"{patient_id}_{visit}_original_DCE.nii"
    ftv_path = patient_dir / f"{patient_id}_{visit}_ftv_mask.nii"
    analysis_path = patient_dir / f"{patient_id}_{visit}_analysis_mask_raw.nii"
    out_path = cache_dir / f"{patient_id}_{visit}_mincrop_raw_rgb224.npz"
    if out_path.exists() and not overwrite:
        return {"patient_id": patient_id, "status": "exists", "cache_path": str(out_path)}

    if not dce_path.exists():
        return {"patient_id": patient_id, "status": "missing_dce", "cache_path": str(out_path)}

    dce, dce_meta = read_nifti(dce_path)
    if dce.ndim == 3:
        dce = dce[..., None]
    if dce.ndim != 4:
        return {"patient_id": patient_id, "status": f"bad_dce_shape:{dce.shape}", "cache_path": str(out_path)}

    mask_source = None
    bbox = None
    if ftv_path.exists():
        ftv, _ = read_nifti(ftv_path)
        bbox = bbox_from_mask(ftv > 0)
        mask_source = "ftv_mask"
    if bbox is None and analysis_path.exists():
        analysis, _ = read_nifti(analysis_path)
        bbox = bbox_from_mask(analysis > 0)
        mask_source = "analysis_mask_raw"
    if bbox is None:
        shape_x, shape_y, shape_z = dce.shape[:3]
        bbox = (0, shape_x, 0, shape_y, 0, shape_z)
        mask_source = "image_center"

    x0, x1, y0, y1, z0, z1 = bbox
    center_x = 0.5 * (x0 + x1 - 1)
    center_y_raw = 0.5 * (y0 + y1 - 1)
    center_y = (dce.shape[1] - 1) - center_y_raw
    center_z = (z0 + z1) // 2
    first_z = max(center_z - 2, z0, 0)
    last_z = min(center_z + 2, z1, dce.shape[2])
    slice_indices = list(range(first_z, last_z))
    if not slice_indices:
        slice_indices = [int(np.clip(center_z, 0, dce.shape[2] - 1))]

    phases = choose_phases(patient_id, dce.shape[3], meta_row)
    images = []
    for z_idx in slice_indices:
        rgb = np.stack([orient_slice(dce[:, :, z_idx, phase]) for phase in phases], axis=-1)
        rgb = minmax_global(rgb)
        rgb = safe_crop(rgb, center_x=center_x, center_y=center_y, crop_size=crop_size)
        if image_size != crop_size:
            rgb = safe_crop(rgb, center_x=crop_size / 2, center_y=crop_size / 2, crop_size=image_size)
        images.append(np.rint(rgb * 255.0).clip(0, 255).astype(np.uint8))

    images_np = np.stack(images, axis=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("wb") as f:
            np.savez(
                f,
                images=images_np,
                label_pcr=np.asarray(label_pcr, dtype=np.int64),
                slice_indices=np.asarray(slice_indices, dtype=np.int16),
                phases=np.asarray(phases, dtype=np.int16),
                bbox_xyz=np.asarray(bbox, dtype=np.int16),
                center_xy=np.asarray([center_x, center_y], dtype=np.float32),
                dce_shape=np.asarray(dce.shape, dtype=np.int16),
                pixdim=np.asarray(dce_meta["pixdim"], dtype=np.float32),
            )
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "patient_id": patient_id,
        "status": "ok",
        "cache_path": str(out_path),
        "n_slices": len(slice_indices),
        "slice_indices": ";".join(str(x) for x in slice_indices),
        "phases": ";".join(str(x) for x in phases),
        "bbox_xyz": ";".join(str(x) for x in bbox),
        "mask_source": mask_source,
        "dce_shape": ";".join(str(x) for x in dce.shape),
        "label_pcr": label_pcr,
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels_csv = args.labels_csv or args.output_root / "clinical_labels_complete4visits.csv"
    records = load_complete4_records(labels_csv, args.output_root, args.max_patients)
    meta = pd.read_csv(args.breastdcedl_metadata_csv).set_index("pid")

    tasks = []
    for record in records:
        meta_row = meta.loc[record.patient_id].to_dict() if record.patient_id in meta.index else None
        tasks.append(
            {
                "patient_id": record.patient_id,
                "label_pcr": record.label_pcr,
                "visit": args.visit,
                "output_root": str(args.output_root),
                "cache_dir": str(args.cache_dir),
                "crop_size": args.crop_size,
                "image_size": args.image_size,
                "overwrite": args.overwrite,
                "meta_row": meta_row,
            }
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if args.num_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            row = process_one(task)
            rows.append(row)
            print(f"[{idx}/{len(tasks)}] {row['patient_id']} {row['status']}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(process_one, task): task for task in tasks}
            for idx, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(f"[{idx}/{len(tasks)}] {row['patient_id']} {row['status']}")

    rows = sorted(rows, key=lambda row: row["patient_id"])
    write_manifest(args.cache_dir / "manifest.csv", rows)
    summary = {
        "cache_dir": str(args.cache_dir),
        "visit": args.visit,
        "n_records": len(records),
        "status_counts": pd.Series([row["status"] for row in rows]).value_counts().to_dict(),
    }
    (args.cache_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
