#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT.parent / "local_response_state_multiseed_confirmation"
sys.path.insert(0, str(LOCAL_ROOT / "src"))

from lg_response_pilot.model import load_checkpoint_for_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export private final pre-pooling C1B spatial maps from a frozen LOCAL3 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="private CSV with patient_id,cache_path,split")
    parser.add_argument("--output", type=Path, required=True, help="must end in .private.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-patients", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output.name.endswith(".private.npz"):
        raise SystemExit("spatial maps are patient-level and must be private")
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    required = {"patient_id", "cache_path", "split"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"manifest must contain {sorted(required)}")
    rows = rows[: args.max_patients] if args.max_patients else rows
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_for_evaluation(args.checkpoint.resolve(), device)
    model.eval()
    maps = []
    patient_ids, splits = [], []
    with torch.no_grad():
        for row in rows:
            cache = np.load(Path(row["cache_path"]).resolve(), allow_pickle=False)
            image = torch.from_numpy(cache["image"].astype(np.float32, copy=False)).to(device)
            spatial = model.encoder(image)
            if spatial.ndim != 5 or spatial.shape[1] != 128:
                raise ValueError(f"unexpected final spatial map shape: {tuple(spatial.shape)}")
            maps.append(spatial.cpu().numpy().astype(np.float16))
            patient_ids.append(row["patient_id"])
            splits.append(row["split"])
    stacked = np.stack(maps, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, patient_id=np.asarray(patient_ids, dtype="U64"), split=np.asarray(splits, dtype="U16"), feature_map=stacked, feature_map_shape=np.asarray(stacked.shape[2:], dtype=np.int64), source_checkpoint=str(args.checkpoint), pcr_used=np.asarray(False), clinical_used=np.asarray(False))
    print({"status": "COMPLETE", "patients": len(rows), "visits": 4, "feature_map_shape": list(stacked.shape), "checkpoint_selected": bool(checkpoint.get("selected", False))})


if __name__ == "__main__":
    main()

