#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raw_spatial_pcr.contracts import ARMS, PRIMARY_TIMINGS, load_contract
from raw_spatial_pcr.metrics import classification_metrics
from raw_spatial_pcr.training import train_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Goal C private cell.")
    parser.add_argument("--input-npz", type=Path, required=True, help="private NPZ with inputs, labels, split, and optional patient_id")
    parser.add_argument("--arm", choices=ARMS[1:], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--timing", choices=PRIMARY_TIMINGS + ("T0_T3",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract()
    if args.seed not in contract.seeds or args.fold not in contract.folds:
        raise SystemExit("seed/fold is outside frozen contract")
    payload = np.load(args.input_npz, allow_pickle=False)
    required = {"inputs", "labels", "split"}
    if not required.issubset(payload.files):
        raise SystemExit(f"input NPZ must contain {sorted(required)}")
    inputs = payload["inputs"]
    labels = payload["labels"]
    split = payload["split"].astype(str)
    steps = {"T0": 1, "T0_T1": 2, "T0_T2": 3, "T0_T3": 4}[args.timing]
    if inputs.ndim == 6:
        inputs = inputs[:, :steps]
    result = train_cell(inputs, labels, split, args.arm, args.seed, device=args.device, max_epochs=args.max_epochs, patience=args.patience)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, indices, probabilities in (("train", split == "train", result.train_probability), ("validation", split == "validation", result.validation_probability), ("test", split == "test", result.test_probability)):
        metrics = classification_metrics(labels[indices], probabilities)
        metrics.update({"seed": args.seed, "fold": args.fold, "arm": args.arm, "timing": args.timing, "split": name})
        rows.append(metrics)
    with (args.output_dir / f"cell_{args.seed}_{args.fold}_{args.arm}_{args.timing}.metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"history": result.history, "selected_epoch": result.selected_epoch, "metrics": rows, "privacy": "patient-level predictions are not public"}, stream, indent=2, sort_keys=True)
    with (args.output_dir / f"cell_{args.seed}_{args.fold}_{args.arm}_{args.timing}.private.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_index", "split", "y_true", "probability", "seed", "fold", "arm", "timing"])
        writer.writeheader()
        for name, indices, probabilities in (("train", np.flatnonzero(split == "train"), result.train_probability), ("validation", np.flatnonzero(split == "validation"), result.validation_probability), ("test", np.flatnonzero(split == "test"), result.test_probability)):
            for row_index, truth, probability in zip(indices, labels[indices], probabilities):
                writer.writerow({"row_index": int(row_index), "split": name, "y_true": float(truth), "probability": float(probability), "seed": args.seed, "fold": args.fold, "arm": args.arm, "timing": args.timing})
    print(json.dumps({"status": "COMPLETE", "seed": args.seed, "fold": args.fold, "arm": args.arm, "timing": args.timing, "selected_epoch": result.selected_epoch}, sort_keys=True))


if __name__ == "__main__":
    main()

