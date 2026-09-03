#!/usr/bin/env python3
"""Evaluate the locked V6 states on pCR as an explicitly exploratory readout."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dinov3_rg.contracts import FOLDS, atomic_json  # noqa: E402
from dinov3_rg.evaluation import (  # noqa: E402
    evaluate_fold_cell, fold_metrics, load_outcome_manifest_after_lock,
    pooled_metrics, stratified_early_macro_bootstrap, validate_oof_coverage,
)
from dinov3_rg.locking import verify_exploratory_lock  # noqa: E402


def main() -> None:
    lock = verify_exploratory_lock()
    manifest_path = ROOT.parent / "raw_spatial_pcr_ceiling/manifests/formal_input.private.csv"
    manifest = load_outcome_manifest_after_lock(manifest_path)
    state_root = ROOT.parent / "dinov3_spatial_parallel_radiomics_v6/features/private/states"
    specs = (
        ("C0_192", "S0_SUMMARY", "c0_state"),
        ("S0_256", "S0_SUMMARY", "parallel_state"),
        ("S1_256", "S1_SPATIAL", "parallel_state"),
    )
    predictions, diagnostics = [], []
    for arm, source_arm, state_key in specs:
        for fold in FOLDS:
            path = state_root / f"seed2026_fold{fold}_{source_arm}.npz"
            rows, diag = evaluate_fold_cell(
                manifest.loc[manifest["fold"].eq(fold)].copy(), path,
                seed=2026, fold=fold, arm=arm, population="primary_375",
                feature_source="state", state_key=state_key,
            )
            predictions.extend(rows); diagnostics.extend(diag)
            print({"arm": arm, "fold": fold, "status": "EVALUATED"}, flush=True)
    pred = pd.DataFrame(predictions)
    validate_oof_coverage(pred, 375)
    pred_path = ROOT / "predictions/exploratory_pcr.private.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True); pred.to_csv(pred_path, index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT / "metrics/exploratory_fusion_diagnostics.private.csv", index=False)
    pooled = pooled_metrics(pred); folds = fold_metrics(pred)
    pooled.to_csv(ROOT / "metrics/exploratory_pooled_metrics.csv", index=False)
    folds.to_csv(ROOT / "metrics/exploratory_fold_metrics.csv", index=False)
    s1 = stratified_early_macro_bootstrap(pred, arm="S1_256", draws=2000, seed=260817)
    s1_s0 = stratified_early_macro_bootstrap(pred, arm="S1_256", reference_arm="S0_256", draws=2000, seed=260818)
    bootstrap = pd.DataFrame([
        {"comparison": "S1_vs_C_FTV", **s1},
        {"comparison": "S1_vs_S0", **s1_s0},
    ])
    bootstrap.to_csv(ROOT / "metrics/exploratory_bootstrap_metrics.csv", index=False)
    atomic_json(ROOT / "metrics/exploratory_execution.json", {
        "status": "PASS", "lock_content_sha256": lock["lock_content_sha256"],
        "cells": 15, "arms": [x[0] for x in specs], "primary_patients_per_cell": 375,
        "outcome_fields_read_after_lock": ["label_pcr"], "clinical_fields_read_after_lock": [
            "label_hr", "label_her2", "label_mp", "age_at_screening", "race_simple",
            "menopausal_status_simple", "ethnicity", "arm"],
    })
    print({"status": "PASS", "cells": 15, "s1_vs_baseline": s1, "s1_vs_s0": s1_s0}, flush=True)


if __name__ == "__main__":
    main()
