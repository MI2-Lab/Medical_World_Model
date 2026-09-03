"""Preregistered hard-gate result mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import EXPERIMENT_ROOT, SEEDS, atomic_json, load_protocol
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
    output: dict[str, dict[str, float]] = {}
    for timing, group in averaged.groupby("timing", sort=True):
        labels = group["label_pcr"].to_numpy(int)
        baseline = classification_metrics(labels, group["baseline_probability"].to_numpy(float))
        augmented = classification_metrics(labels, group["augmented_probability"].to_numpy(float))
        output[str(timing)] = {
            "delta_auroc": augmented["auroc"] - baseline["auroc"],
            "delta_auprc": augmented["auprc"] - baseline["auprc"],
            "brier_improvement": baseline["brier"] - augmented["brier"],
            "log_loss_improvement": baseline["log_loss"] - augmented["log_loss"],
        }
    return output


def _arm_comparison(predictions: pd.DataFrame, candidate: str, reference: str) -> tuple[dict[str, float], dict[int, float]]:
    selected = predictions.loc[
        predictions["population"].eq("primary_375")
        & predictions["feature_source"].eq("state")
        & predictions["timing"].isin(EARLY)
        & predictions["arm"].isin((candidate, reference))
    ]
    timing_effects: dict[str, float] = {}
    seed_effects: dict[int, float] = {}
    seed_average = selected.groupby(
        ["patient_id", "fold", "timing", "label_pcr", "arm"], as_index=False
    )["augmented_probability"].mean()
    for timing, group in seed_average.groupby("timing"):
        pivot = group.pivot(index=["patient_id", "fold", "label_pcr"], columns="arm", values="augmented_probability").reset_index()
        timing_effects[str(timing)] = float(
            classification_metrics(pivot["label_pcr"], pivot[candidate])["auroc"]
            - classification_metrics(pivot["label_pcr"], pivot[reference])["auroc"]
        )
    for seed, seed_group in selected.groupby("seed"):
        effects: list[float] = []
        for _, timing_group in seed_group.groupby("timing"):
            pivot = timing_group.pivot(index=["patient_id", "fold", "label_pcr"], columns="arm", values="augmented_probability").reset_index()
            effects.append(
                classification_metrics(pivot["label_pcr"], pivot[candidate])["auroc"]
                - classification_metrics(pivot["label_pcr"], pivot[reference])["auroc"]
            )
        seed_effects[int(seed)] = float(np.mean(effects))
    return timing_effects, seed_effects


def make_decision(
    predictions: pd.DataFrame,
    mechanism: pd.DataFrame,
    *,
    safety_pass: bool,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    thresholds = load_protocol()["success"]
    primary = predictions.loc[
        predictions["population"].eq("primary_375")
        & predictions["feature_source"].eq("state")
    ]
    d3_effects = _seed_averaged_effects(primary, "D3")
    early_effects = [d3_effects[timing] for timing in EARLY]
    d3_macro_auroc = float(np.mean([value["delta_auroc"] for value in early_effects]))
    d3_macro_auprc = float(np.mean([value["delta_auprc"] for value in early_effects]))
    d3_macro_brier = float(np.mean([value["brier_improvement"] for value in early_effects]))
    d3_seed_macro: dict[int, float] = {}
    pooled = []
    for seed, seed_group in primary.loc[primary["arm"].eq("D3")].groupby("seed"):
        effects = []
        for timing in EARLY:
            group = seed_group.loc[seed_group["timing"].eq(timing)]
            baseline = classification_metrics(group["label_pcr"], group["baseline_probability"])
            augmented = classification_metrics(group["label_pcr"], group["augmented_probability"])
            effects.append(augmented["auroc"] - baseline["auroc"])
        d3_seed_macro[int(seed)] = float(np.mean(effects))
    bootstrap = stratified_early_macro_bootstrap(primary, arm="D3", draws=2000)

    d3_d2_timing, d3_d2_seed = _arm_comparison(primary, "D3", "D2")
    d3_d2_macro = float(np.mean(list(d3_d2_timing.values())))

    mechanism_seed = mechanism.groupby(["seed", "arm"], as_index=False)[
        ["radiomics_pc_macro_spearman", "static_ftv_macro_spearman", "delta_ftv_macro_spearman"]
    ].mean()
    pivot = mechanism_seed.pivot(index="seed", columns="arm")
    rad_gain = pivot["radiomics_pc_macro_spearman"]["D3"] - pivot["radiomics_pc_macro_spearman"]["D2"]
    static_change = pivot["static_ftv_macro_spearman"]["D3"] - pivot["static_ftv_macro_spearman"]["D2"]
    delta_change = pivot["delta_ftv_macro_spearman"]["D3"] - pivot["delta_ftv_macro_spearman"]["D2"]

    pcr_gates = {
        "both_early_delta_auroc_positive": all(value["delta_auroc"] > 0 for value in early_effects),
        "early_macro_delta_auroc": d3_macro_auroc >= float(thresholds["d3_cf_early_macro_delta_auroc"]),
        "four_of_five_seeds_positive": sum(value > 0 for value in d3_seed_macro.values()) >= int(thresholds["d3_positive_seeds"]),
        "bootstrap_ci_lower_positive": bootstrap["ci_low"] > 0,
        "macro_delta_auprc_nonnegative": d3_macro_auprc >= float(thresholds["d3_macro_delta_auprc"]),
        "macro_brier_improvement_nonnegative": d3_macro_brier >= float(thresholds["d3_macro_brier_improvement"]),
    }
    d3_d2_gates = {
        "early_macro_delta_auroc": d3_d2_macro >= float(thresholds["d3_minus_d2_early_macro_delta_auroc"]),
        "four_of_five_seeds_positive": sum(value > 0 for value in d3_d2_seed.values()) >= int(thresholds["d3_minus_d2_positive_seeds"]),
    }
    mechanism_gates = {
        "radiomics_spearman_gain": float(rad_gain.mean()) >= float(thresholds["radiomics_spearman_gain"]),
        "radiomics_four_of_five_seeds_positive": int((rad_gain > 0).sum()) >= int(thresholds["radiomics_positive_seeds"]),
        "static_ftv_not_degraded": float(static_change.mean()) >= -float(thresholds["maximum_ftv_spearman_drop"]),
        "delta_ftv_not_degraded": float(delta_change.mean()) >= -float(thresholds["maximum_ftv_spearman_drop"]),
        "optimization_noncollapse_privacy": bool(safety_pass),
    }
    pcr_pass = all(pcr_gates.values()) and all(d3_d2_gates.values())
    mechanism_pass = all(mechanism_gates.values())

    auxiliary_signal = False
    auxiliary_details: dict[str, float] = {}
    for arm in ("D1", "D2"):
        effects = _seed_averaged_effects(primary, arm)
        macro = float(np.mean([effects[timing]["delta_auroc"] for timing in EARLY]))
        auxiliary_details[arm] = macro
        auxiliary_signal = auxiliary_signal or macro >= float(thresholds["d3_cf_early_macro_delta_auroc"])

    if pcr_pass and mechanism_pass:
        decision = "RAD_GROUNDED_CONDITIONAL_PCR_SUPPORTED"
    elif auxiliary_signal and not all(d3_d2_gates.values()):
        decision = "DINO_ADAPTER_OR_FTV_ONLY"
    elif mechanism_pass and not pcr_pass:
        decision = "REPRESENTATION_ONLY"
    elif pcr_pass and not mechanism_pass:
        decision = "PCR_SIGNAL_MECHANISM_INCONCLUSIVE"
    else:
        decision = "NO_GO"
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "decision_class": decision,
        "pcr_gates": pcr_gates,
        "d3_vs_d2_gates": d3_d2_gates,
        "mechanism_and_safety_gates": mechanism_gates,
        "d3_seed_averaged_early": {timing: d3_effects[timing] for timing in EARLY},
        "d3_early_macro_delta_auroc": d3_macro_auroc,
        "d3_early_macro_delta_auprc": d3_macro_auprc,
        "d3_early_macro_brier_improvement": d3_macro_brier,
        "d3_seed_early_macro_delta_auroc": d3_seed_macro,
        "d3_bootstrap": bootstrap,
        "d3_minus_d2_timing_delta_auroc": d3_d2_timing,
        "d3_minus_d2_seed_early_macro_delta_auroc": d3_d2_seed,
        "d3_minus_d2_early_macro_delta_auroc": d3_d2_macro,
        "radiomics_spearman_gain_by_seed": {str(index): float(value) for index, value in rad_gain.items()},
        "static_ftv_spearman_change_by_seed": {str(index): float(value) for index, value in static_change.items()},
        "delta_ftv_spearman_change_by_seed": {str(index): float(value) for index, value in delta_change.items()},
        "auxiliary_early_macro_delta_auroc": auxiliary_details,
        "reporting_boundary": "internal five-fold OOF; independent-cohort validation required",
    }
    atomic_json(output_path or EXPERIMENT_ROOT / "decision.json", payload)
    return payload


__all__ = ["make_decision"]
