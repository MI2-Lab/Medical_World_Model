#!/usr/bin/env python3
"""Evaluate the outcome-blind V3 pilot and lock the smallest passing weight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import (  # noqa: E402
    FOLDS, PILOT_ARMS, PILOT_SEED, PILOT_WEIGHTS, V2_STATE_ROOT, V2_TARGET_DIR,
    atomic_json, canonical_sha256, load_protocol,
)
from dinov3_rg.locking import freeze_pilot_lock  # noqa: E402
from dinov3_rg.probes import evaluate_matched_probes  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel, public_artifact_privacy_scan  # noqa: E402


def _state_path(seed: int, fold: int, arm: str) -> Path:
    if seed != PILOT_SEED:
        raise ValueError("pilot state resolver only admits seed 2026")
    if arm == "C0":
        return V2_STATE_ROOT / f"seed2026_fold{fold}_D1_states.private.npz"
    return ROOT / f"features/private/pilot_states/seed2026_fold{fold}_{arm}_states.private.npz"


def _candidate_training_safety(arm: str) -> dict[str, object]:
    failures: list[str] = []
    failed_checkpoints: list[dict[str, object]] = []
    checked = 0
    state_sd: list[float] = []
    for fold in FOLDS:
        tag = f"seed2026_fold{fold}_{arm}"
        cell = ROOT / "checkpoints/pilot" / tag
        complete_path = cell / "cell_complete.private.json"
        failed_path = cell / "cell_failed.private.json"
        history_path = cell / "history.private.json"
        if not complete_path.is_file() or not history_path.is_file():
            if failed_path.is_file():
                failed = json.loads(failed_path.read_text(encoding="utf-8"))
                observed = float(failed["minimum_observed_jepa_loss"])
                allowed = float(failed["maximum_allowed_jepa_loss"])
                failed_checkpoints.append({
                    "fold": fold,
                    "epochs_completed": int(failed["epochs_completed"]),
                    "paired_c0_jepa_loss": float(failed["paired_c0_jepa_loss"]),
                    "maximum_allowed_jepa_loss": allowed,
                    "minimum_observed_jepa_loss": observed,
                    "excess_over_allowed_fraction": observed / allowed - 1.0,
                })
                failures.append(f"fold{fold}:no_feasible_checkpoint")
            else:
                failures.append(f"fold{fold}:not_run_after_fail_fast")
            continue
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
        if canonical_sha256(history) != complete.get("history_sha256"):
            failures.append(f"fold{fold}:history_hash")
            continue
        checked += 1
        validation = complete["selected_validation"]
        state_sd.append(float(validation["state_mean_sd"]))
        numeric = []
        for epoch in history:
            for split in ("train", "validation"):
                numeric.extend(float(value) for value in epoch[split].values())
            train = epoch["train"]
            if float(train["first_batch_radiomics_head_gradient_norm"]) <= 0:
                failures.append(f"fold{fold}:head_gradient")
            if float(train["first_batch_adapter_gradient_norm"]) <= 0:
                failures.append(f"fold{fold}:adapter_gradient")
            if float(train["first_batch_response_projection_gradient_norm"]) <= 0:
                failures.append(f"fold{fold}:response_projection_gradient")
        if not np.isfinite(numeric).all():
            failures.append(f"fold{fold}:nonfinite")
        if float(complete.get("ftv_weight", -1)) != 0.0:
            failures.append(f"fold{fold}:ftv_weight")
    minimum_sd = float(load_protocol()["pilot"]["minimum_state_sd"])
    if state_sd and min(state_sd) < minimum_sd:
        failures.append("state_noncollapse")
    return {
        "status": "PASS" if checked == 5 and not failures else "FAIL",
        "checked_cells": checked,
        "minimum_selected_state_mean_sd": min(state_sd) if state_sd else None,
        "failed_checkpoints": failed_checkpoints,
        "failures": sorted(set(failures)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-lock", action="store_true")
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    execution_path = ROOT / "metrics/pilot_execution.json"
    if not execution_path.is_file():
        raise SystemExit("pilot execution summary is absent")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    eligible = [
        arm for arm in PILOT_ARMS
        if all(execution["statuses"].get(f"seed2026_fold{fold}_{arm}") == "COMPLETE" for fold in FOLDS)
    ]
    arms = ["C0", *eligible]
    metrics, diagnostics = evaluate_matched_probes(
        seeds=[PILOT_SEED], arms=arms, state_path=_state_path, target_root=V2_TARGET_DIR
    )
    metrics_path = ROOT / "metrics/pilot_probe_metrics.csv"
    diagnostics_path = ROOT / "metrics/pilot_probe_diagnostics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    indexed = metrics.set_index("arm")
    baseline = indexed.loc["C0"]
    thresholds = load_protocol()["success"]
    candidate_gates: dict[str, dict[str, bool]] = {}
    candidate_values: dict[str, dict[str, object]] = {}
    for arm in PILOT_ARMS:
        safety = _candidate_training_safety(arm)
        if arm in indexed.index:
            row = indexed.loc[arm]
            rad_gain = float(
                row["matched_probe_radiomics_macro_spearman"]
                - baseline["matched_probe_radiomics_macro_spearman"]
            )
            static_change = float(
                row["matched_probe_static_ftv_macro_spearman"]
                - baseline["matched_probe_static_ftv_macro_spearman"]
            )
            delta_change = float(
                row["matched_probe_delta_ftv_macro_spearman"]
                - baseline["matched_probe_delta_ftv_macro_spearman"]
            )
            gates = {
                "five_checkpoint_safety": safety["status"] == "PASS",
                "direct_head_absolute": float(row["direct_head_radiomics_macro_spearman"]) >= float(thresholds["direct_head_absolute_spearman"]),
                "matched_probe_absolute": float(row["matched_probe_radiomics_macro_spearman"]) >= float(thresholds["matched_probe_absolute_spearman"]),
                "matched_probe_gain": rad_gain >= float(thresholds["matched_probe_gain"]),
                "static_ftv_retained": static_change >= -float(thresholds["maximum_ftv_spearman_drop"]),
                "delta_ftv_retained": delta_change >= -float(thresholds["maximum_ftv_spearman_drop"]),
                "state_noncollapse": float(row["state_mean_sd"]) >= float(load_protocol()["pilot"]["minimum_state_sd"]),
                "finite_and_gradient": safety["status"] == "PASS",
            }
            values = {
                "radiomics_weight": PILOT_WEIGHTS[arm],
                "direct_head_radiomics_macro_spearman": float(row["direct_head_radiomics_macro_spearman"]),
                "matched_probe_radiomics_macro_spearman": float(row["matched_probe_radiomics_macro_spearman"]),
                "matched_probe_gain_vs_c0": rad_gain,
                "static_ftv_change_vs_c0": static_change,
                "delta_ftv_change_vs_c0": delta_change,
                "training_safety": safety,
            }
        else:
            gates = {name: False for name in (
                "five_checkpoint_safety", "direct_head_absolute", "matched_probe_absolute",
                "matched_probe_gain", "static_ftv_retained", "delta_ftv_retained",
                "state_noncollapse", "finite_and_gradient",
            )}
            values = {"radiomics_weight": PILOT_WEIGHTS[arm], "training_safety": safety}
        candidate_gates[arm] = gates
        candidate_values[arm] = values
    privacy = public_artifact_privacy_scan(ROOT)
    for gates in candidate_gates.values():
        gates["privacy"] = privacy["status"] == "PASS"
    passing = [arm for arm in PILOT_ARMS if all(candidate_gates[arm].values())]
    selected = min(passing, key=lambda arm: PILOT_WEIGHTS[arm]) if passing else None
    gate = {
        "schema_version": 1,
        "status": "PASS" if selected else "NO_GO",
        "selected_arm": selected,
        "selected_radiomics_weight": None if selected is None else PILOT_WEIGHTS[selected],
        "selected_candidate_gates": {} if selected is None else candidate_gates[selected],
        "candidate_gates": candidate_gates,
        "candidate_evaluation_status": {
            arm: "EVALUATED" if arm in indexed.index else "NOT_EVALUATED_NO_FEASIBLE_CHECKPOINT"
            for arm in PILOT_ARMS
        },
        "candidate_metrics": candidate_values,
        "baseline_metrics": {key: float(value) for key, value in baseline.items() if key != "seed" and isinstance(value, (float, int, np.number))},
        "privacy": privacy,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    gate_path = ROOT / "pilot_gate.json"
    atomic_json(gate_path, gate)
    if selected is None:
        atomic_json(ROOT / "decision.json", {
            "schema_version": 1, "status": "TERMINATED_AT_PILOT_GATE",
            "decision_class": "DIRECT_RAD_WEIGHT_SCREEN_NO_GO",
            "termination_reason": "all registered weights failed paired JEPA checkpoint safety",
            "pcr_evaluation_started": False, "pcr_outcomes_read": False,
            "pilot_gate_status": "NO_GO",
        })
        attempted = sum(value in {"COMPLETE", "NO_FEASIBLE_CHECKPOINT"} for value in execution["statuses"].values())
        state_complete = sum(value == "COMPLETE" for value in execution["statuses"].values())
        skipped = sum(value == "SKIPPED_AFTER_ARM_FAILURE" for value in execution["statuses"].values())
        atomic_json(ROOT / "acceptance_check.json", {
            "schema_version": 1, "status": "FAIL", "failed_stage": "pilot_gate",
            "pilot_cells_requested": 15,
            "pilot_cells_fully_trained": attempted,
            "pilot_cells_with_feasible_checkpoint_and_state": state_complete,
            "pilot_cells_skipped_after_decisive_failure": skipped,
            "formal_cells_completed": 0, "pilot_lock_created": False,
            "evaluation_lock_created": False, "mechanism_lock_created": False,
            "pcr_outcomes_read": False, "privacy_gate": privacy["status"],
            "decision_class": "DIRECT_RAD_WEIGHT_SCREEN_NO_GO",
        })
        print(gate)
        raise SystemExit("pilot hard gate failed; formal matrix and pCR remain locked")
    lock = freeze_pilot_lock(gate_path, metrics_path, execution_path, overwrite=args.overwrite_lock)
    print({"gate": gate, "lock": lock})


if __name__ == "__main__":
    main()
