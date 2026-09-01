"""Outcome-blind matched-probe mechanism evaluation and V3 hard gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import EXPERIMENT_ROOT, FOLDS, SEEDS, V2_TARGET_DIR, canonical_sha256, load_protocol
from .probes import evaluate_matched_probes


def formal_state_path(seed: int, fold: int, arm: str) -> Path:
    return EXPERIMENT_ROOT / f"features/private/formal_states/seed{seed}_fold{fold}_{arm}_states.private.npz"


def mechanism_metrics(target_root: str | Path = V2_TARGET_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_matched_probes(
        seeds=SEEDS, arms=("C0", "RAD"), state_path=formal_state_path,
        target_root=target_root,
    )


def _formal_training_safety(checkpoint_root: str | Path) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for seed in SEEDS:
        for fold in FOLDS:
            tag = f"seed{seed}_fold{fold}_RAD"
            cell = Path(checkpoint_root) / tag
            history_path = cell / "history.private.json"
            complete_path = cell / "cell_complete.private.json"
            if not history_path.is_file() or not complete_path.is_file():
                failures.append(f"{tag}:incomplete")
                continue
            history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if canonical_sha256(history) != complete.get("history_sha256"):
                failures.append(f"{tag}:history_hash")
                continue
            checked += 1
            all_values = []
            for epoch in history:
                for split in ("train", "validation"):
                    all_values.extend(float(value) for value in epoch[split].values())
                train = epoch["train"]
                for name in (
                    "first_batch_adapter_gradient_norm",
                    "first_batch_response_projection_gradient_norm",
                    "first_batch_radiomics_head_gradient_norm",
                ):
                    if float(train[name]) <= 0:
                        failures.append(f"{tag}:{name}")
            if not np.isfinite(all_values).all():
                failures.append(f"{tag}:nonfinite")
            if float(complete.get("ftv_weight", -1)) != 0.0:
                failures.append(f"{tag}:ftv_weight")
    smoke_path = EXPERIMENT_ROOT / "metrics/smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {}
    if smoke.get("status") != "PASS":
        failures.append("isolated_radiomics_gradient_smoke")
    return {
        "status": "PASS" if checked == 25 and not failures else "FAIL",
        "checked_candidate_cells": checked,
        "isolated_radiomics_gradient_smoke": smoke.get("status") == "PASS",
        "failures": sorted(set(failures)),
    }


def mechanism_gate(metrics: pd.DataFrame, checkpoint_root: str | Path) -> dict[str, Any]:
    required = {"seed", "arm", "direct_head_radiomics_macro_spearman",
                "matched_probe_radiomics_macro_spearman",
                "matched_probe_static_ftv_macro_spearman",
                "matched_probe_delta_ftv_macro_spearman", "state_mean_sd"}
    if not required.issubset(metrics.columns) or len(metrics) != 10:
        raise ValueError("formal mechanism metrics must contain five seeds x two arms")
    pivot = metrics.pivot(index="seed", columns="arm")
    direct = pivot["direct_head_radiomics_macro_spearman"]["RAD"]
    matched = pivot["matched_probe_radiomics_macro_spearman"]["RAD"]
    gain = matched - pivot["matched_probe_radiomics_macro_spearman"]["C0"]
    static_change = (
        pivot["matched_probe_static_ftv_macro_spearman"]["RAD"]
        - pivot["matched_probe_static_ftv_macro_spearman"]["C0"]
    )
    delta_change = (
        pivot["matched_probe_delta_ftv_macro_spearman"]["RAD"]
        - pivot["matched_probe_delta_ftv_macro_spearman"]["C0"]
    )
    protocol = load_protocol(); thresholds = protocol["success"]
    safety = _formal_training_safety(checkpoint_root)
    gates = {
        "direct_head_absolute": float(direct.mean()) >= float(thresholds["direct_head_absolute_spearman"]),
        "matched_probe_absolute": float(matched.mean()) >= float(thresholds["matched_probe_absolute_spearman"]),
        "matched_probe_gain": float(gain.mean()) >= float(thresholds["matched_probe_gain"]),
        "four_of_five_seed_gains_positive": int((gain > 0).sum()) >= int(thresholds["matched_probe_positive_seeds"]),
        "static_ftv_retained": float(static_change.mean()) >= -float(thresholds["maximum_ftv_spearman_drop"]),
        "delta_ftv_retained": float(delta_change.mean()) >= -float(thresholds["maximum_ftv_spearman_drop"]),
        "state_noncollapse": bool((metrics["state_mean_sd"] >= float(protocol["pilot"]["minimum_state_sd"])).all()),
        "finite_gradient_and_ftv_zero": safety["status"] == "PASS",
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "NO_GO",
        "gates": gates,
        "direct_head_by_seed": {str(k): float(v) for k, v in direct.items()},
        "matched_probe_by_seed": {str(k): float(v) for k, v in matched.items()},
        "matched_probe_gain_by_seed": {str(k): float(v) for k, v in gain.items()},
        "static_ftv_change_by_seed": {str(k): float(v) for k, v in static_change.items()},
        "delta_ftv_change_by_seed": {str(k): float(v) for k, v in delta_change.items()},
        "formal_training_safety": safety,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }


__all__ = ["formal_state_path", "mechanism_gate", "mechanism_metrics"]
