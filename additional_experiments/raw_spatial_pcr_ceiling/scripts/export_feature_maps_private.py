#!/usr/bin/env python3
"""Stream frozen LOCAL3 final maps to one private file per patient."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT.parent / "local_response_state_multiseed_confirmation"
CONDITIONAL_ROOT = ROOT.parent / "conditional_pcr_contrastive_ceiling"
sys.path.insert(0, str(LOCAL_ROOT / "src"))
sys.path.insert(0, str(CONDITIONAL_ROOT / "src"))

from lg_response_pilot.model import load_checkpoint_for_evaluation  # noqa: E402
from conditional_ceiling.data import CacheRecord, load_c1b_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    required = {"patient_id", "cache_path"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"manifest misses {sorted(required - set(frame.columns))}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame = frame.drop_duplicates("patient_id").sort_values("patient_id")
    if args.max_patients:
        frame = frame.head(args.max_patients)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_for_evaluation(args.checkpoint.resolve(), device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = 0
    with torch.inference_mode():
        for _, row in frame.iterrows():
            target = args.output_dir / f"{row['patient_id']}.private.npz"
            if args.resume and target.exists():
                complete += 1
                continue
            cache_path = Path(str(row["cache_path"])).expanduser().resolve(strict=True)
            record = CacheRecord(
                patient_id=str(row["patient_id"]),
                path=cache_path,
                sha256=str(row.get("cache_sha256", "")),
                size_bytes=int(row.get("cache_size_bytes", cache_path.stat().st_size)),
                mtime_ns=int(row.get("cache_mtime_ns", cache_path.stat().st_mtime_ns)),
            )
            image = torch.from_numpy(load_c1b_image(record)).to(device)
            spatial = model.encoder(image)
            if tuple(spatial.shape[1:]) != (128, 14, 22, 20):
                raise ValueError(f"unexpected feature map shape {tuple(spatial.shape)} for {row['patient_id']}")
            # Feature maps are already float16 and are high-entropy activations;
            # uncompressed NPZ avoids making the GPU wait on a CPU compressor.
            np.savez(
                target,
                feature_map=spatial.cpu().numpy().astype(np.float16),
                feature_map_shape=np.asarray(spatial.shape[1:], dtype=np.int64),
                source_checkpoint=str(args.checkpoint.resolve()),
                pcr_used=np.asarray(False),
                clinical_used=np.asarray(False),
            )
            complete += 1
            if complete % 25 == 0:
                print({"patients_written": complete}, flush=True)
    print({"status": "COMPLETE", "patients": complete, "feature_map_shape": [128, 14, 22, 20], "checkpoint_selected": bool(checkpoint.get("selected", False)), "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
