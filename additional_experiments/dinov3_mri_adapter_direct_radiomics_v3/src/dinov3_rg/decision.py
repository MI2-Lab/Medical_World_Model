"""Preregistered V3 conditional-pCR decision mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import EXPERIMENT_ROOT, atomic_json, load_protocol
from .evaluation import classification_metrics, stratified_early_macro_bootstrap


EARLY = ("T0-T1", "T0-T2")


def _seed_averaged_effects(predictions: pd.DataFrame, arm: str) -> dict[str, dict[str, float]]:
    selected = predictions.loc[
        predictions["arm"].eq(arm)
        & predictions["population"].eq("primary_375")
        & predictions["feature_source"].eq("state")
    ]
    averaged = selected.groupby(["patient_id", "fold", "timing", "label_pcr"], as_index=False)[
        ["baseline_probability", "augmented_probability"]
    ].mean()
    result = {}
    for timing, group in averaged.groupby("timing", sort=True):
        labels = group["label_pcr"].to_numpy(int)
        baseline = classification_metrics(labels, group["baseline_probability"])
        augmented = classification_metrics(labels, group["augmented_probability"])
        result[str(timing)] = {
            "delta_auroc": augmented["auroc"] - baseline["auroc"],
            "delta_auprc": augmented["auprc"] - baseline["auprc"],
            "brier_improvement": baseline["brier"] - augmented["brier"],
        }
    return result


def _seed_macro(predictions: pd.DataFrame, arm: str) -> dict[int, float]:
    output = {}
    for seed, seed_frame in predictions.loc[predictions["arm"].eq(arm)].groupby("seed"):
        effects = []
        for timing in EARLY:
            frame = seed_frame.loc[seed_frame["timing"].eq(timing)]
            baseline = classification_metrics(frame["label_pcr"], frame["baseline_probability"])
            augmented = classification_metrics(frame["label_pcr"], frame["augmented_probability"])
            effects.append(augmented["auroc"] - baseline["auroc"])
        output[int(seed)] = float(np.mean(effects))
    return output


def _arm_comparison(predictions: pd.DataFrame) -> tuple[dict[str, float], dict[int, float]]:
    selected = predictions.loc[predictions["timing"].isin(EARLY)]
    timing_result = {}; seed_result = {}
    averaged = selected.groupby(
        ["patient_id", "fold", "timing", "label_pcr", "arm"], as_index=False
    )["augmented_probability"].mean()
    for timing, group in averaged.groupby("timing"):
        pivot = group.pivot(index=["patient_id", "fold", "label_pcr"], columns="arm", values="augmented_probability").reset_index()
        timing_result[str(timing)] = (
            classification_metrics(pivot["label_pcr"], pivot["RAD"])["auroc"]
            - classification_metrics(pivot["label_pcr"], pivot["C0"])["auroc"]
        )
    for seed, group in selected.groupby("seed"):
        effects = []
        for _, timing_group in group.groupby("timing"):
            pivot = timing_group.pivot(index=["patient_id", "fold", "label_pcr"], columns="arm", values="augmented_probability").reset_index()
            effects.append(
                classification_metrics(pivot["label_pcr"], pivot["RAD"])["auroc"]
                - classification_metrics(pivot["label_pcr"], pivot["C0"])["auroc"]
            )
        seed_result[int(seed)] = float(np.mean(effects))
    return timing_result, seed_result


def make_decision(
    predictions: pd.DataFrame,
    mechanism: pd.DataFrame,
    *,
    safety_pass: bool,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    del mechanism  # The valid MECHANISM_LOCK already binds the preregistered mechanism gate.
    thresholds = load_protocol()["success"]
    primary = predictions.loc[
        predictions["population"].eq("primary_375")
        & predictions["feature_source"].eq("state")
    ]
    rad_effects = _seed_averaged_effects(primary, "RAD")
    early = [rad_effects[timing] for timing in EARLY]
    macro_auroc = float(np.mean([value["delta_auroc"] for value in early]))
    macro_auprc = float(np.mean([value["delta_auprc"] for value in early]))
    macro_brier = float(np.mean([value["brier_improvement"] for value in early]))
    rad_seed = _seed_macro(primary, "RAD")
    c0_seed = _seed_macro(primary, "C0")
    bootstrap = stratified_early_macro_bootstrap(primary, arm="RAD", draws=2000)
    rad_c0_timing, rad_c0_seed = _arm_comparison(primary)
    rad_c0_macro = float(np.mean(list(rad_c0_timing.values())))
    pcr_gates = {
        "both_early_delta_auroc_positive": all(value["delta_auroc"] > 0 for value in early),
        "early_macro_delta_auroc": macro_auroc >= float(thresholds["candidate_cf_early_macro_delta_auroc"]),
        "four_of_five_seeds_positive": sum(value > 0 for value in rad_seed.values()) >= int(thresholds["candidate_positive_seeds"]),
        "bootstrap_ci_lower_positive": bootstrap["ci_low"] > 0,
        "macro_delta_auprc_nonnegative": macro_auprc >= float(thresholds["candidate_macro_delta_auprc"]),
        "macro_brier_improvement_nonnegative": macro_brier >= float(thresholds["candidate_macro_brier_improvement"]),
    }
    comparison_gates = {
        "early_macro_delta_auroc": rad_c0_macro >= float(thresholds["candidate_minus_c0_early_macro_delta_auroc"]),
        "four_of_five_seeds_positive": sum(value > 0 for value in rad_c0_seed.values()) >= int(thresholds["candidate_minus_c0_positive_seeds"]),
    }
    pcr_pass = all(pcr_gates.values()) and all(comparison_gates.values()) and bool(safety_pass)
    c0_signal = (
        float(np.mean(list(c0_seed.values()))) >= float(thresholds["candidate_cf_early_macro_delta_auroc"])
        and sum(value > 0 for value in c0_seed.values()) >= int(thresholds["candidate_positive_seeds"])
    )
    if pcr_pass:
        decision_class = "RAD_GROUNDED_CONDITIONAL_PCR_SUPPORTED"
    elif c0_signal and not all(comparison_gates.values()):
        decision_class = "DINO_ADAPTER_ONLY"
    else:
        decision_class = "REPRESENTATION_ONLY"
    payload = {
        "schema_version": 1, "status": "COMPLETE", "decision_class": decision_class,
        "mechanism_lock_verified": True, "pcr_gates": pcr_gates,
        "candidate_vs_c0_gates": comparison_gates,
        "candidate_seed_averaged_early": {timing: rad_effects[timing] for timing in EARLY},
        "candidate_early_macro_delta_auroc": macro_auroc,
        "candidate_early_macro_delta_auprc": macro_auprc,
        "candidate_early_macro_brier_improvement": macro_brier,
        "candidate_seed_early_macro_delta_auroc": rad_seed,
        "candidate_bootstrap": bootstrap,
        "candidate_minus_c0_timing_delta_auroc": rad_c0_timing,
        "candidate_minus_c0_seed_early_macro_delta_auroc": rad_c0_seed,
        "candidate_minus_c0_early_macro_delta_auroc": rad_c0_macro,
        "c0_seed_early_macro_delta_auroc": c0_seed,
        "reporting_boundary": "internal five-fold OOF; independent-cohort validation required",
    }
    atomic_json(output_path or EXPERIMENT_ROOT / "decision.json", payload)
    return payload


__all__ = ["make_decision"]
