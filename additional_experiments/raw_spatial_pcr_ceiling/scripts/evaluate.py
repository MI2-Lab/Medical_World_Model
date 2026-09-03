#!/usr/bin/env python3
"""Aggregate private OOF predictions into public Goal C metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raw_spatial_pcr.bootstrap import paired_patient_bootstrap  # noqa: E402
from raw_spatial_pcr.metrics import classification_metrics  # noqa: E402


REQUIRED = {"row_index", "patient_id", "split", "y_true", "probability", "seed", "fold", "arm", "timing"}


def _read_predictions(source: Path) -> pd.DataFrame:
    paths = [source] if source.is_file() else sorted(source.rglob("*.private.csv"))
    if not paths:
        raise FileNotFoundError(f"no private prediction files under {source}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"prediction file {path} misses {sorted(missing)}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["seed", "fold", "arm", "timing", "split", "row_index"]).any():
        raise ValueError("duplicate private prediction row")
    return result


def _summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    value_columns = ["auroc", "auprc", "brier", "calibration_slope", "ece10"]
    rows: list[dict[str, object]] = []
    for group_keys, group in frame.groupby(keys, sort=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(keys, group_keys))
        for column in value_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        row["n_cells"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _attention_table(output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "predictions" / "formal").rglob("*.json")):
        if path.name.startswith("cell_"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"seed": payload.get("seed"), "fold": payload.get("fold"), "arm": payload.get("arm"), "timing": payload.get("timing"), **payload.get("attention_diagnostics", {})})
    pd.DataFrame(rows).to_csv(output_dir / "attention_diagnostics.csv", index=False)


def _fusion_summaries(output_dir: Path) -> None:
    paths = sorted(path for path in output_dir.glob("fusion_*.csv") if path.name != "fusion_fold_metrics.csv")
    if not paths:
        return
    frames = [pd.read_csv(path) for path in paths]
    fusion = pd.concat(frames, ignore_index=True)
    fusion.to_csv(output_dir / "fusion_fold_metrics.csv", index=False)
    summary = _summarize(fusion, ["population", "arm", "timing", "model"])
    if "full_808" in set(fusion["population"]):
        summary.loc[summary["population"].eq("full_808")].to_csv(output_dir / "clinical_complementarity.csv", index=False)
    if "ftv_complete_375" in set(fusion["population"]):
        summary.loc[summary["population"].eq("ftv_complete_375")].to_csv(output_dir / "beyond_ftv.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument("--population", default="full_808")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _read_predictions(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["seed", "fold", "arm", "timing", "split"], sort=True):
        values = classification_metrics(group["y_true"].to_numpy(), group["probability"].to_numpy())
        values.update(dict(zip(["seed", "fold", "arm", "timing", "split"], keys)))
        values["population"] = args.population
        metric_rows.append(values)
    fold_metrics = pd.DataFrame(metric_rows)
    fold_metrics.to_csv(args.output_dir / "mri_only_fold_metrics.csv", index=False)
    test = fold_metrics.loc[fold_metrics["split"].eq("test")].copy()
    _summarize(test, ["arm", "timing"]).to_csv(args.output_dir / "mri_only_metrics.csv", index=False)
    gap_rows: list[dict[str, object]] = []
    for keys, group in fold_metrics.groupby(["seed", "fold", "arm", "timing"], sort=True):
        by_split = group.set_index("split")
        if not {"train", "validation", "test"}.issubset(by_split.index):
            continue
        gap_rows.append({"seed": keys[0], "fold": keys[1], "arm": keys[2], "timing": keys[3], "train_auroc": by_split.loc["train", "auroc"], "validation_auroc": by_split.loc["validation", "auroc"], "oof_auroc": by_split.loc["test", "auroc"], "train_minus_oof_auroc": by_split.loc["train", "auroc"] - by_split.loc["test", "auroc"], "validation_minus_oof_auroc": by_split.loc["validation", "auroc"] - by_split.loc["test", "auroc"]})
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(args.output_dir / "generalization_gap.csv", index=False)
    if not gaps.empty:
        gaps.groupby(["arm", "timing"], as_index=False).agg(
            train_auroc_mean=("train_auroc", "mean"),
            validation_auroc_mean=("validation_auroc", "mean"),
            oof_auroc_mean=("oof_auroc", "mean"),
            train_minus_oof_auroc_mean=("train_minus_oof_auroc", "mean"),
            validation_minus_oof_auroc_mean=("validation_minus_oof_auroc", "mean"),
        ).to_csv(args.output_dir / "generalization_gap_summary.csv", index=False)
    bootstrap_rows: list[dict[str, object]] = []
    test_frame = frame.loc[frame["split"].eq("test")].copy()
    for (seed, timing), group in test_frame.groupby(["seed", "timing"], sort=True):
        arms = set(group["arm"])
        comparisons = [("C0", arm) for arm in sorted(arms - {"C0"})]
        if {"C2", "C3"}.issubset(arms):
            comparisons.append(("C2", "C3"))
        if {"C3", "C4"}.issubset(arms):
            comparisons.append(("C3", "C4"))
        for reference_arm, candidate_arm in comparisons:
            reference = group.loc[group["arm"].eq(reference_arm)]
            candidate = group.loc[group["arm"].eq(candidate_arm)]
            joined = reference.merge(candidate, on=["seed", "fold", "timing", "row_index"], suffixes=("_ref", "_cand"), validate="one_to_one")
            if len(joined) != 808:
                raise ValueError(f"paired bootstrap needs exactly 808 test patients, got {len(joined)} for {seed}/{timing}/{reference_arm}/{candidate_arm}")
            effect = paired_patient_bootstrap(joined["y_true_ref"].to_numpy(), joined["probability_ref"].to_numpy(), joined["probability_cand"].to_numpy(), joined["fold"].to_numpy(), draws=args.bootstrap_draws, seed=260814 + int(seed) + sum(ord(char) for char in str(timing)) + sum(ord(char) for char in candidate_arm))
            bootstrap_rows.append({"seed": int(seed), "timing": timing, "reference_arm": reference_arm, "candidate_arm": candidate_arm, "population": args.population, **effect})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    if not bootstrap.empty:
        seed_summary = bootstrap.groupby(["timing", "reference_arm", "candidate_arm"], as_index=False).agg(delta_auroc_mean=("delta_auroc", "mean"), delta_auroc_std=("delta_auroc", "std"), seeds=("seed", "nunique"))
        seed_summary.to_csv(args.output_dir / "seed_consistency.csv", index=False)
        context = bootstrap.loc[bootstrap["reference_arm"].eq("C3") & bootstrap["candidate_arm"].eq("C4")].copy()
        context.to_csv(args.output_dir / "local_vs_full_context.csv", index=False)
    _attention_table(args.output_dir)
    _fusion_summaries(args.output_dir)
    print({"status": "COMPLETE", "prediction_rows": len(frame), "metric_rows": len(fold_metrics), "bootstrap_rows": len(bootstrap)})


if __name__ == "__main__":
    main()
