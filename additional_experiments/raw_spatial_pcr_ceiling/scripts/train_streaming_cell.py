#!/usr/bin/env python3
"""Train one formal Goal C cell from the frozen private manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raw_spatial_pcr.contracts import ARMS, FOLDS, PRIMARY_TIMINGS, SEEDS, load_contract  # noqa: E402
from raw_spatial_pcr.metrics import classification_metrics  # noqa: E402
from raw_spatial_pcr.training import train_streaming  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS[1:], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--timing", choices=PRIMARY_TIMINGS + ("T0_T3",), required=True)
    parser.add_argument("--feature-dir", type=Path, help="private per-patient feature maps for C1-C4")
    parser.add_argument("--feature-array", type=Path, help="private fold-level .private.npy map array")
    parser.add_argument("--checkpoint", type=Path, help="selected prior encoder for C5 initialization")
    parser.add_argument("--raw-array", type=Path, help="private 808-patient raw .private.npy array for C5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract()
    if args.seed not in SEEDS or args.fold not in FOLDS:
        raise SystemExit("seed/fold is outside frozen contract")
    if args.arm == "C5" and args.checkpoint is None:
        raise SystemExit("C5 requires a selected prior encoder checkpoint")
    if args.arm == "C5" and args.raw_array is None:
        raise SystemExit("C5 requires the private raw array")
    if args.arm != "C5" and args.feature_dir is None and args.feature_array is None:
        raise SystemExit("C1-C4 require a private feature directory or array")
    manifest = pd.read_csv(args.manifest)
    if args.arm == "C5":
        if not args.raw_array.resolve().is_file():
            raise SystemExit(f"raw array missing: {args.raw_array}")
        manifest["raw_array_path"] = str(args.raw_array.resolve())
    if args.arm != "C5" and args.feature_array is None:
        feature_dir = args.feature_dir.resolve()
        manifest["feature_map_path"] = manifest["patient_id"].astype(str).map(lambda patient_id: str(feature_dir / f"{patient_id}.private.npz"))
        missing = [str(path) for path in manifest.loc[manifest["fold"].eq(args.fold), "feature_map_path"] if not Path(path).is_file()]
        if missing:
            raise SystemExit(f"feature maps missing ({len(missing)}), first={missing[0]}")
    if args.arm != "C5" and args.feature_array is not None:
        if not args.feature_array.resolve().is_file():
            raise SystemExit(f"feature array missing: {args.feature_array}")
        manifest["feature_array_path"] = str(args.feature_array.resolve())
    result = train_streaming(
        manifest,
        args.arm,
        args.seed,
        args.fold,
        args.timing,
        device=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        c5_checkpoint=args.checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = manifest.loc[manifest["fold"].eq(args.fold)].copy().set_index("row_index")
    rows: list[dict[str, object]] = []
    for split_name, probabilities, row_indices in (
        ("train", result.train_probability, result.train_row_index),
        ("validation", result.validation_probability, result.validation_row_index),
        ("test", result.test_probability, result.test_row_index),
    ):
        for row_index, probability in zip(row_indices, probabilities):
            row = fold_frame.loc[int(row_index)]
            rows.append({
                "row_index": int(row_index),
                "patient_id": str(row["patient_id"]),
                "split": split_name,
                "y_true": int(row["label_pcr"]),
                "probability": float(probability),
                "seed": args.seed,
                "fold": args.fold,
                "arm": args.arm,
                "timing": args.timing,
            })
    prediction_path = args.output_dir / f"cell_{args.seed}_{args.fold}_{args.arm}_{args.timing}.private.csv"
    pd.DataFrame(rows).sort_values(["split", "row_index"]).to_csv(prediction_path, index=False)
    metric_rows = []
    for split_name in ("train", "validation", "test"):
        part = pd.DataFrame(rows).loc[lambda value: value["split"].eq(split_name)]
        values = classification_metrics(part["y_true"].to_numpy(), part["probability"].to_numpy())
        values.update({"seed": args.seed, "fold": args.fold, "arm": args.arm, "timing": args.timing, "split": split_name})
        metric_rows.append(values)
    metadata = {
        "status": "COMPLETE",
        "seed": args.seed,
        "fold": args.fold,
        "arm": args.arm,
        "timing": args.timing,
        "selected_epoch": result.selected_epoch,
        "requested_batch_size": args.batch_size,
        "history": result.history,
        "metrics": metric_rows,
        "attention_diagnostics": result.attention_diagnostics,
        "input_contract": contract.lock["input_contract"],
        "privacy": "patient-level predictions remain private",
    }
    (args.output_dir / f"cell_{args.seed}_{args.fold}_{args.arm}_{args.timing}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "seed": args.seed, "fold": args.fold, "arm": args.arm, "timing": args.timing, "selected_epoch": result.selected_epoch}, sort_keys=True))


if __name__ == "__main__":
    main()
