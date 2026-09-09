#!/usr/bin/env python3
"""Outcome-blind V5 pilot mechanism gate."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from parallel_rg.contracts import ROOT as EXPERIMENT_ROOT, V2_TARGET_DIR, load_folds, FOLDS, atomic_json


ARMS = ("HEAD_ONLY", "P1_PARALLEL")
VISITS = (0, 1, 2)
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def target(fold: int) -> dict[str, np.ndarray]:
    with np.load(V2_TARGET_DIR / f"fold_{fold}_targets.private.npz", allow_pickle=False) as z:
        result = {key: np.asarray(z[key]) for key in ("patient_id", "radiomics", "ftv")}
        result["radiomics_mask"] = np.asarray(z["radiomics_mask"], dtype=bool)
        result["ftv_mask"] = np.asarray(z["ftv_mask"], dtype=bool)
        return result


def state(fold: int, arm: str) -> dict[str, np.ndarray]:
    path = EXPERIMENT_ROOT / "features/private/pilot_states" / f"seed2026_fold{fold}_{arm}_states.private.npz"
    with np.load(path, allow_pickle=False) as z:
        required = {"patient_id", "c0_state", "rad_initial_state", "rad_state", "parallel_initial_state", "parallel_state", "radiomics_prediction", "c0_checkpoint_sha256", "rad_checkpoint_sha256"}
        if not required.issubset(z.files): raise ValueError(f"state contract failed: {path}")
        result = {key: np.asarray(z[key]) for key in required}
        if result["c0_state"].shape != (808, 4, 192) or result["rad_initial_state"].shape != (808, 4, 64) or result["rad_state"].shape != (808, 4, 64) or result["parallel_state"].shape != (808, 4, 256): raise ValueError(f"state shape failed: {path}")
        return result


def splits(fold: int) -> dict[str, set[str]]:
    frame = load_folds(); current = frame.loc[frame["fold"].eq(fold)]
    return {name: set(current.loc[current["split"].eq(name), "patient_id"].astype(str)) for name in ("train", "val", "test")}


def x_with_visit(values: np.ndarray, visit: int) -> np.ndarray:
    one_hot = np.zeros((len(values), 3), dtype=np.float64); one_hot[:, visit] = 1.0
    return np.concatenate((values.astype(np.float64), one_hot), axis=1)


def corr(y: np.ndarray, prediction: np.ndarray) -> float:
    keep = np.isfinite(y) & np.isfinite(prediction)
    if int(keep.sum()) < 4 or np.unique(y[keep]).size < 2 or np.unique(prediction[keep]).size < 2: return float("nan")
    return float(spearmanr(y[keep], prediction[keep]).statistic)


def fit_predict(features: np.ndarray, values: np.ndarray, masks: np.ndarray, ids: np.ndarray, fold_splits: dict[str, set[str]], visit: int) -> tuple[np.ndarray, np.ndarray, float]:
    train = np.asarray([str(pid) in fold_splits["train"] for pid in ids]) & masks[:, visit]
    validation = np.asarray([str(pid) in fold_splits["val"] for pid in ids]) & masks[:, visit]
    test = np.asarray([str(pid) in fold_splits["test"] for pid in ids]) & masks[:, visit]
    if train.sum() < 20 or validation.sum() < 10 or test.sum() < 10: raise ValueError("insufficient outer split rows")
    x_train, x_validation = x_with_visit(features[train, visit], visit), x_with_visit(features[validation, visit], visit)
    scaler = StandardScaler().fit(x_train); x_train_s, x_validation_s = scaler.transform(x_train), scaler.transform(x_validation)
    best_alpha, best_mse = None, float("inf")
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(x_train_s, values[train, visit]); mse = float(np.mean((model.predict(x_validation_s) - values[validation, visit]) ** 2))
        if mse < best_mse - 1e-12 or (abs(mse - best_mse) <= 1e-12 and (best_alpha is None or alpha > best_alpha)): best_alpha, best_mse = alpha, mse
    train_val = np.asarray([str(pid) in (fold_splits["train"] | fold_splits["val"]) for pid in ids]) & masks[:, visit]
    scaler = StandardScaler().fit(x_with_visit(features[train_val, visit], visit)); model = Ridge(alpha=float(best_alpha)).fit(scaler.transform(x_with_visit(features[train_val, visit], visit)), values[train_val, visit])
    return model.predict(scaler.transform(x_with_visit(features[test, visit], visit))), values[test, visit], float(best_alpha)


def probe_radiomics(features: np.ndarray, target_payload: dict[str, np.ndarray], fold_splits: dict[str, set[str]], ids: np.ndarray) -> tuple[float, dict[str, float]]:
    correlations: dict[str, float] = {}
    for pc in range(16):
        values = target_payload["radiomics"][:, :, pc]
        for visit in VISITS:
            prediction, truth, _ = fit_predict(features, values, target_payload["radiomics_mask"], ids, fold_splits, visit)
            correlations[f"pc{pc:02d}_t{visit}"] = corr(truth, prediction)
    return float(np.nanmean(list(correlations.values()))), correlations


def direct_head(prediction: np.ndarray, target_payload: dict[str, np.ndarray], fold_splits: dict[str, set[str]], ids: np.ndarray) -> float:
    keep_patient = np.asarray([str(pid) in fold_splits["test"] for pid in target_payload["patient_id"]])
    state_position = {str(pid): i for i, pid in enumerate(ids)}
    values = []
    for pc in range(16):
        for visit in VISITS:
            truth, pred = [], []
            for i, pid in enumerate(target_payload["patient_id"]):
                if keep_patient[i] and target_payload["radiomics_mask"][i, visit] and str(pid) in state_position:
                    truth.append(target_payload["radiomics"][i, visit, pc]); pred.append(prediction[state_position[str(pid)], visit, pc])
            values.append(corr(np.asarray(truth), np.asarray(pred)))
    return float(np.nanmean(values))


def ftv_probe(features: np.ndarray, target_payload: dict[str, np.ndarray], fold_splits: dict[str, set[str]], ids: np.ndarray) -> tuple[float, float]:
    static = []; predictions: dict[int, np.ndarray] = {}; truths: dict[int, np.ndarray] = {}
    for visit in VISITS:
        pred, truth, _ = fit_predict(features, target_payload["ftv"], target_payload["ftv_mask"], ids, fold_splits, visit)
        predictions[visit], truths[visit] = pred, truth; static.append(corr(truth, pred))
    deltas = [corr(truths[visit] - truths[visit - 1], predictions[visit] - predictions[visit - 1]) for visit in (1, 2)]
    return float(np.nanmean(static)), float(np.nanmean(deltas))


def main() -> None:
    rows = []; fold_details = []
    for fold in FOLDS:
        target_payload = target(fold); fold_splits = splits(fold); candidate = state(fold, "P1_PARALLEL"); head = state(fold, "HEAD_ONLY"); state_position = {str(pid): i for i, pid in enumerate(candidate["patient_id"].astype(str))}; ids = target_payload["patient_id"].astype(str)
        aligned_candidate = {key: np.stack([candidate[key][state_position[str(pid)]] for pid in ids]) for key in ("rad_initial_state", "rad_state", "parallel_initial_state", "parallel_state", "radiomics_prediction")}
        aligned_head = {key: np.stack([head[key][state_position[str(pid)]] for pid in ids]) for key in ("rad_state", "radiomics_prediction")}
        if not np.array_equal(candidate["c0_state"], head["c0_state"]): raise AssertionError(f"C0 state drift fold {fold}")
        if not np.array_equal(candidate["parallel_initial_state"], head["parallel_initial_state"]): raise AssertionError(f"initial parallel state drift fold {fold}")
        initial_probe, _ = probe_radiomics(aligned_candidate["rad_initial_state"], target_payload, fold_splits, ids)
        head_probe, _ = probe_radiomics(aligned_head["rad_state"], target_payload, fold_splits, ids)
        candidate_probe, _ = probe_radiomics(aligned_candidate["rad_state"], target_payload, fold_splits, ids)
        direct = direct_head(aligned_candidate["radiomics_prediction"], target_payload, fold_splits, ids); head_direct = direct_head(aligned_head["radiomics_prediction"], target_payload, fold_splits, ids)
        initial_ftv = ftv_probe(aligned_candidate["parallel_initial_state"], target_payload, fold_splits, ids); candidate_ftv = ftv_probe(aligned_candidate["parallel_state"], target_payload, fold_splits, ids)
        history = json.loads((EXPERIMENT_ROOT / "checkpoints/pilot" / f"seed2026_fold{fold}_P1_PARALLEL/history.private.json").read_text())["history"]
        warmup = [item for item in history if item["stage"] == "head_warmup"]
        row = {"fold": fold, "head_only_validation_initial_loss": warmup[0]["validation"]["loss"], "head_only_validation_final_loss": warmup[-1]["validation"]["loss"], "head_only_validation_loss_improvement": warmup[0]["validation"]["loss"] - warmup[-1]["validation"]["loss"], "rad_initial_probe": initial_probe, "head_only_probe": head_probe, "candidate_probe": candidate_probe, "candidate_probe_gain": candidate_probe - initial_probe, "candidate_direct_head": direct, "head_only_direct_head": head_direct, "initial_ftv_static": initial_ftv[0], "candidate_ftv_static": candidate_ftv[0], "initial_ftv_delta": initial_ftv[1], "candidate_ftv_delta": candidate_ftv[1], "ftv_static_change": candidate_ftv[0] - initial_ftv[0], "ftv_delta_change": candidate_ftv[1] - initial_ftv[1]}
        rows.append(row); fold_details.append(row)
    mean = {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0] if key != "fold"}; positive = int(sum(row["candidate_probe_gain"] > 0 for row in rows)); head_positive = int(sum(row["head_only_validation_loss_improvement"] > 0 for row in rows))
    checks = {"head_only_learnable": head_positive == 5, "candidate_direct_head_ge_0_10": mean["candidate_direct_head"] >= 0.10, "candidate_probe_ge_0_10": mean["candidate_probe"] >= 0.10, "candidate_gain_ge_0_05": mean["candidate_probe_gain"] >= 0.05, "positive_fold_count_ge_4": positive >= 4, "ftv_static_retained": mean["ftv_static_change"] >= -0.02, "ftv_delta_retained": mean["ftv_delta_change"] >= -0.02, "c0_identity_all_folds": True, "finite": bool(np.isfinite(np.asarray([[value for key, value in row.items() if key != "fold"] for row in rows], dtype=float)).all())}
    passed = bool(all(checks.values()))
    result = {"status": "PASS" if passed else "FAIL", "pilot_cells": 5, "folds": fold_details, "mean": mean, "positive_probe_gain_folds": positive, "checks": checks, "outcome_fields_read": [], "clinical_fields_read": []}
    out = EXPERIMENT_ROOT / "metrics"; out.mkdir(exist_ok=True); atomic_json(out / "pilot_gate.json", result)
    with (out / "pilot_probe_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    decision = {"decision": "PILOT_LOCKED" if passed else ("RADIOMICS_TARGET_HEAD_NOT_LEARNABLE" if not checks["head_only_learnable"] else "PARALLEL_ADAPTER_NOT_TRANSFERRED"), "pilot_gate": result["status"], "formal_matrix": "LOCKED" if not passed else "AUTHORIZED", "pCR_evaluation": "LOCKED", "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(EXPERIMENT_ROOT / "decision.json", decision)
    if passed: atomic_json(EXPERIMENT_ROOT / "PILOT_LOCK.json", {"status": "PASS", "selected_arm": "P1_PARALLEL", "pilot_gate_sha256": "recorded_in_private_manifest", "outcome_fields_read": [], "clinical_fields_read": []})
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
