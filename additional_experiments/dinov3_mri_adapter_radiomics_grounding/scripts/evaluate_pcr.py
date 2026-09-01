#!/usr/bin/env python3
"""Run locked conditional pCR fusion, mechanisms, bootstrap, and decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import ARMS, FOLDS, SEEDS, atomic_json  # noqa: E402
from dinov3_rg.decision import make_decision  # noqa: E402
from dinov3_rg.evaluation import (  # noqa: E402
    evaluate_fold_cell, fold_metrics, load_outcome_manifest_after_lock,
    mechanism_metrics, pooled_metrics, validate_oof_coverage,
)
from dinov3_rg.locking import verify_evaluation_lock  # noqa: E402
from dinov3_rg.security import public_artifact_privacy_scan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT.parent / "raw_spatial_pcr_ceiling/manifests/formal_input.private.csv",
    )
    parser.add_argument("--state-root", type=Path, default=ROOT / "features/private/states")
    parser.add_argument("--target-root", type=Path, default=ROOT / "features/private/fold_targets")
    parser.add_argument("--skip-secondary-808", action="store_true")
    parser.add_argument("--skip-radiomics-head-sensitivity", action="store_true")
    args = parser.parse_args()
    lock = verify_evaluation_lock()
    # This is deliberately the first outcome-bearing read in the process.
    manifest = load_outcome_manifest_after_lock(args.manifest)
    prediction_rows = []
    diagnostic_rows = []
    populations = ("primary_375",) if args.skip_secondary_808 else ("primary_375", "secondary_808")
    for seed in SEEDS:
        for fold in FOLDS:
            fold_frame = manifest.loc[manifest["fold"].eq(fold)].copy()
            for arm in ARMS:
                state_path = args.state_root / f"seed{seed}_fold{fold}_{arm}_states.private.npz"
                for population in populations:
                    predictions, diagnostics = evaluate_fold_cell(
                        fold_frame, state_path, seed=seed, fold=fold, arm=arm,
                        population=population, feature_source="state",
                    )
                    prediction_rows.extend(predictions)
                    diagnostic_rows.extend(diagnostics)
                if arm == "D3" and not args.skip_radiomics_head_sensitivity:
                    predictions, diagnostics = evaluate_fold_cell(
                        fold_frame, state_path, seed=seed, fold=fold, arm=arm,
                        population="primary_375", feature_source="radiomics_head",
                    )
                    prediction_rows.extend(predictions)
                    diagnostic_rows.extend(diagnostics)
                print({"seed": seed, "fold": fold, "arm": arm, "status": "EVALUATED"}, flush=True)
    predictions = pd.DataFrame(prediction_rows)
    primary_state = predictions.loc[
        predictions["population"].eq("primary_375") & predictions["feature_source"].eq("state")
    ]
    validate_oof_coverage(primary_state, 375)
    if not args.skip_secondary_808:
        secondary = predictions.loc[
            predictions["population"].eq("secondary_808") & predictions["feature_source"].eq("state")
        ]
        validate_oof_coverage(secondary, 808)
    if not args.skip_radiomics_head_sensitivity:
        sensitivity = predictions.loc[predictions["feature_source"].eq("radiomics_head")]
        validate_oof_coverage(sensitivity, 375)
    prediction_path = ROOT / "predictions/conditional_fusion.private.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics_path = ROOT / "metrics/fusion_diagnostics.private.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    pooled = pooled_metrics(predictions)
    folds = fold_metrics(predictions)
    pooled.to_csv(ROOT / "metrics/pooled_oof_metrics.csv", index=False)
    folds.to_csv(ROOT / "metrics/fold_metrics.csv", index=False)
    mechanism = mechanism_metrics(manifest, args.state_root, args.target_root)
    mechanism.to_csv(ROOT / "metrics/mechanism_metrics.csv", index=False)
    privacy = public_artifact_privacy_scan(ROOT)
    safety_pass = privacy["status"] == "PASS" and lock["cell_count"] == 75
    decision = make_decision(primary_state, mechanism, safety_pass=safety_pass)
    acceptance = {
        "schema_version": 1,
        "status": "PASS" if safety_pass else "FAIL",
        "evaluation_lock_verified": True,
        "representation_cells": 75,
        "primary_oof_patients_per_cell": 375,
        "secondary_oof_patients_per_cell": None if args.skip_secondary_808 else 808,
        "inner_oof_offset_fusion": True,
        "test_prediction_once_per_fold": True,
        "bootstrap_replicates": 2000,
        "privacy_gate": privacy,
        "decision_class": decision["decision_class"],
    }
    atomic_json(ROOT / "acceptance_check.json", acceptance)
    print({"acceptance": acceptance, "decision": decision})


if __name__ == "__main__":
    main()
