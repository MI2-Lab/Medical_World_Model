"""Outcome-blind held-out grounding-transfer evaluation and hard gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .contracts import EXPERIMENT_ROOT, FOLDS, SEEDS, canonical_sha256, load_protocol
from .data import FoldTargets, load_fold_frame, validate_state_archive


def _safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if int(valid.sum()) < 3 or np.ptp(truth[valid]) <= 0 or np.ptp(prediction[valid]) <= 0:
        return float("nan")
    return float(spearmanr(truth[valid], prediction[valid]).statistic)


def _state_heads(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    patient_ids, state = validate_state_archive(path)
    with np.load(path, allow_pickle=False) as payload:
        radiomics = np.asarray(payload["radiomics_prediction"], dtype=np.float32)
        ftv = np.asarray(payload["ftv_prediction"], dtype=np.float32)
    if radiomics.shape != (808, 4, 16) or ftv.shape != (808, 4):
        raise ValueError("grounding-head state archive contract failed")
    return {value: index for index, value in enumerate(patient_ids)}, state, radiomics, ftv


def mechanism_metrics(
    state_root: str | Path,
    target_root: str | Path,
    fold_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pool each seed's five held-out folds, then score the OOF heads."""
    folds = load_fold_frame() if fold_frame is None else fold_frame.copy()
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ("D2", "D3"):
            observed_ids: list[str] = []
            rad_targets: list[np.ndarray] = []
            rad_masks: list[np.ndarray] = []
            ftv_targets: list[np.ndarray] = []
            ftv_masks: list[np.ndarray] = []
            rad_predictions: list[np.ndarray] = []
            ftv_predictions: list[np.ndarray] = []
            held_out_states: list[np.ndarray] = []
            primary_ids: set[str] | None = None
            for fold in FOLDS:
                test_ids = tuple(
                    folds.loc[
                        folds["fold"].eq(fold) & folds["split"].eq("test"), "patient_id"
                    ].astype(str)
                )
                targets = FoldTargets.load(Path(target_root) / f"fold_{fold}_targets.private.npz")
                fold_primary = set(targets.patient_ids)
                if primary_ids is None:
                    primary_ids = fold_primary
                elif fold_primary != primary_ids:
                    raise ValueError("primary target cohort differs across folds")
                target_lookup = {value: index for index, value in enumerate(targets.patient_ids)}
                eligible_ids = tuple(value for value in test_ids if value in target_lookup)
                target_indices = np.asarray([target_lookup[value] for value in eligible_ids])
                fold_rad_mask = targets.radiomics_mask[target_indices]
                if fold_rad_mask[:, 3].any():
                    raise ValueError("T3 entered V2 mechanism targets")
                path = Path(state_root) / f"seed{seed}_fold{fold}_{arm}_states.private.npz"
                lookup, state, rad_prediction, ftv_prediction = _state_heads(path)
                indices = np.asarray([lookup[value] for value in eligible_ids])
                observed_ids.extend(eligible_ids)
                rad_targets.append(targets.radiomics[target_indices])
                rad_masks.append(fold_rad_mask)
                ftv_targets.append(targets.ftv[target_indices])
                ftv_masks.append(targets.ftv_mask[target_indices])
                rad_predictions.append(rad_prediction[indices])
                ftv_predictions.append(ftv_prediction[indices])
                held_out_states.append(state[indices])
            if primary_ids is None or len(primary_ids) != 375:
                raise RuntimeError("mechanism primary cohort contract failed")
            if len(observed_ids) != 375 or len(set(observed_ids)) != 375 or set(observed_ids) != primary_ids:
                raise RuntimeError("each primary patient must enter held-out OOF exactly once")
            rad_target = np.concatenate(rad_targets)
            rad_mask = np.concatenate(rad_masks)
            ftv_target = np.concatenate(ftv_targets)
            ftv_mask = np.concatenate(ftv_masks)
            rad_prediction = np.concatenate(rad_predictions)
            ftv_prediction = np.concatenate(ftv_predictions)
            state = np.concatenate(held_out_states)
            radiomics_rhos: list[float] = []
            for visit in range(3):
                for component in range(16):
                    valid = rad_mask[:, visit]
                    rho = _safe_spearman(
                        rad_target[:, visit, component][valid],
                        rad_prediction[:, visit, component][valid],
                    )
                    if np.isfinite(rho):
                        radiomics_rhos.append(rho)
            static_rhos = [
                _safe_spearman(
                    ftv_target[:, visit][ftv_mask[:, visit]],
                    ftv_prediction[:, visit][ftv_mask[:, visit]],
                )
                for visit in range(4)
            ]
            delta_rhos = []
            for visit in range(3):
                valid = ftv_mask[:, visit] & ftv_mask[:, visit + 1]
                delta_rhos.append(
                    _safe_spearman(
                        ftv_target[:, visit + 1][valid] - ftv_target[:, visit][valid],
                        ftv_prediction[:, visit + 1][valid] - ftv_prediction[:, visit][valid],
                    )
                )
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "held_out_patients": len(observed_ids),
                    "oof_folds": len(FOLDS),
                    "radiomics_pc_visit_correlations": len(radiomics_rhos),
                    "radiomics_pc_macro_spearman": float(np.mean(radiomics_rhos)),
                    "static_ftv_macro_spearman": float(np.nanmean(static_rhos)),
                    "delta_ftv_macro_spearman": float(np.nanmean(delta_rhos)),
                    "state_mean_sd": float(state.reshape(-1, 192).std(axis=0).mean()),
                }
            )
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["seed"])
    if len(frame) != 10 or not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("mechanism metrics are incomplete or non-finite")
    return frame


def _gradient_safety(checkpoint_root: str | Path) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for seed in SEEDS:
        for fold in FOLDS:
            cell = Path(checkpoint_root) / f"seed{seed}_fold{fold}_D3"
            history_path = cell / "history.private.json"
            complete_path = cell / "cell_complete.private.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if canonical_sha256(history) != complete["history_sha256"]:
                failures.append(f"seed{seed}_fold{fold}:history_hash")
                continue
            checked += 1
            gradients = [
                float(epoch["train"]["first_batch_radiomics_head_gradient_norm"])
                for epoch in history
            ]
            if not gradients or not np.isfinite(gradients).all() or not all(value > 0 for value in gradients):
                failures.append(f"seed{seed}_fold{fold}:radiomics_head_gradient")
    return {"status": "PASS" if not failures and checked == 25 else "FAIL", "checked": checked, "failures": failures}


def mechanism_gate(metrics: pd.DataFrame, checkpoint_root: str | Path) -> dict[str, Any]:
    thresholds = load_protocol()["success"]
    seed_metrics = metrics.groupby(["seed", "arm"], as_index=False)[
        [
            "radiomics_pc_macro_spearman",
            "static_ftv_macro_spearman",
            "delta_ftv_macro_spearman",
            "state_mean_sd",
        ]
    ].mean()
    pivot = seed_metrics.pivot(index="seed", columns="arm")
    d3 = pivot["radiomics_pc_macro_spearman"]["D3"]
    gain = d3 - pivot["radiomics_pc_macro_spearman"]["D2"]
    static_change = (
        pivot["static_ftv_macro_spearman"]["D3"]
        - pivot["static_ftv_macro_spearman"]["D2"]
    )
    delta_change = (
        pivot["delta_ftv_macro_spearman"]["D3"]
        - pivot["delta_ftv_macro_spearman"]["D2"]
    )
    gradient = _gradient_safety(checkpoint_root)
    noncollapse = bool(
        (metrics["state_mean_sd"] >= float(load_protocol()["training"]["minimum_state_sd"])).all()
    )
    gates = {
        "d3_absolute_radiomics_spearman": float(d3.mean()) >= float(thresholds["radiomics_absolute_spearman"]),
        "d3_minus_d2_radiomics_spearman": float(gain.mean()) >= float(thresholds["radiomics_spearman_gain"]),
        "radiomics_four_of_five_seeds_positive": int((gain > 0).sum()) >= int(thresholds["radiomics_positive_seeds"]),
        "static_ftv_not_degraded": float(static_change.mean()) >= -float(thresholds["maximum_ftv_spearman_drop"]),
        "delta_ftv_not_degraded": float(delta_change.mean()) >= -float(thresholds["maximum_ftv_spearman_drop"]),
        "state_noncollapse": noncollapse,
        "formal_d3_radiomics_gradients": gradient["status"] == "PASS",
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "NO_GO",
        "gates": gates,
        "d3_radiomics_spearman_by_seed": {str(k): float(v) for k, v in d3.items()},
        "d3_minus_d2_radiomics_spearman_by_seed": {str(k): float(v) for k, v in gain.items()},
        "static_ftv_change_by_seed": {str(k): float(v) for k, v in static_change.items()},
        "delta_ftv_change_by_seed": {str(k): float(v) for k, v in delta_change.items()},
        "gradient_safety": gradient,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }


__all__ = ["mechanism_gate", "mechanism_metrics"]
