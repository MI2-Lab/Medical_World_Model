#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raw_spatial_pcr.bootstrap import paired_patient_bootstrap
from raw_spatial_pcr.metrics import classification_metrics


REQUIRED = {"row_index", "split", "y_true", "probability", "seed", "fold", "arm", "timing"}


def _read_predictions(source: Path) -> pd.DataFrame:
    paths = [source] if source.is_file() else sorted(source.rglob("*.private.csv"))
    if not paths:
        raise FileNotFoundError(f"no private prediction files under {source}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"prediction file is missing required fields: {sorted(missing)}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["seed", "fold", "arm", "timing", "split", "row_index"]).any():
        raise ValueError("duplicate private prediction row")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate private OOF predictions into public Goal C metrics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument("--population", default="full_808")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _read_predictions(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for keys, group in frame.groupby(["seed", "fold", "arm", "timing", "split"], sort=True):
        metrics = classification_metrics(group["y_true"].to_numpy(), group["probability"].to_numpy())
        metrics.update(dict(zip(["seed", "fold", "arm", "timing", "split"], keys)))
        metrics["population"] = args.population
        metric_rows.append(metrics)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(args.output_dir / "mri_only_fold_metrics.csv", index=False)
    test = metrics_frame[metrics_frame["split"] == "test"].copy()
    test.to_csv(args.output_dir / "mri_only_metrics.csv", index=False)
    gap_rows = []
    for keys, group in metrics_frame.groupby(["seed", "fold", "arm", "timing"], sort=True):
        by_split = group.set_index("split")
        if not {"train", "validation", "test"}.issubset(by_split.index):
            continue
        gap_rows.append({"seed": keys[0], "fold": keys[1], "arm": keys[2], "timing": keys[3], "train_auroc": by_split.loc["train", "auroc"], "validation_auroc": by_split.loc["validation", "auroc"], "oof_auroc": by_split.loc["test", "auroc"], "train_minus_oof_auroc": by_split.loc["train", "auroc"] - by_split.loc["test", "auroc"]})
    pd.DataFrame(gap_rows).to_csv(args.output_dir / "generalization_gap.csv", index=False)
    bootstrap_rows = []
    test_frame = frame[frame["split"] == "test"].copy()
    for (seed, timing), group in test_frame.groupby(["seed", "timing"], sort=True):
        available_arms = set(group["arm"])
        comparisons = [("C0", arm) for arm in sorted(available_arms - {"C0"})]
        if {"C3", "C4"}.issubset(available_arms):
            comparisons.append(("C3", "C4"))
        for reference_arm, arm in comparisons:
            reference = group[group["arm"] == reference_arm]
            candidate = group[group["arm"] == arm]
            joined = reference.merge(candidate, on=["seed", "fold", "timing", "row_index"], suffixes=("_ref", "_cand"))
            if joined.empty:
                continue
            effect = paired_patient_bootstrap(joined["y_true_ref"].to_numpy(), joined["probability_ref"].to_numpy(), joined["probability_cand"].to_numpy(), joined["fold"].to_numpy(), draws=args.bootstrap_draws)
            bootstrap_rows.append({"seed": seed, "timing": timing, "reference_arm": reference_arm, "candidate_arm": arm, "population": args.population, **effect})
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    print(f"wrote aggregate metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
