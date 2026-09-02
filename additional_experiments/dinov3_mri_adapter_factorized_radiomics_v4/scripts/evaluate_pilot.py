#!/usr/bin/env python3
"""Outcome-blind pilot mechanism evaluation.

This script deliberately reads only fold residual-radiomics targets and the
exported representation states.  It never opens an outcome or clinical table.
All probe fitting is outer-fold safe: alpha selection uses the fold's train and
validation patients, while metrics are computed on that fold's test patients.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorized_rg.contracts import EXPERIMENT_ROOT, FOLDS, PILOT_ARMS, load_folds, atomic_json


IMPLEMENTATION_VERSION = "v4-factorized-20260902-r2"
PRIMARY_ARMS = ("F0", "F005", "F010", "F025")
VISIT_INDEX = (0, 1, 2)


def _finite_corr(y: np.ndarray, pred: np.ndarray) -> float:
    keep = np.isfinite(y) & np.isfinite(pred)
    if int(keep.sum()) < 4 or np.unique(y[keep]).size < 2 or np.unique(pred[keep]).size < 2:
        return float("nan")
    return float(spearmanr(y[keep], pred[keep]).statistic)


def _load_target(fold: int) -> dict[str, Any]:
    path = EXPERIMENT_ROOT.parent / "dinov3_mri_adapter_radiomics_grounding_v2" / "features/private/fold_targets" / f"fold_{fold}_targets.private.npz"
    with np.load(path, allow_pickle=False) as z:
        return {
            "patient_id": z["patient_id"].astype(str),
            "radiomics": np.asarray(z["radiomics"], dtype=np.float64),
            "radiomics_mask": np.asarray(z["radiomics_mask"], dtype=bool),
            "ftv": np.asarray(z["ftv"], dtype=np.float64),
            "ftv_mask": np.asarray(z["ftv_mask"], dtype=bool),
        }


def _load_state(fold: int, arm: str) -> dict[str, np.ndarray]:
    path = EXPERIMENT_ROOT / "features/private/pilot_states" / f"seed2026_fold{fold}_{arm}_states.private.npz"
    with np.load(path, allow_pickle=False) as z:
        if set(("patient_id", "phenotype_state", "full_state", "radiomics_prediction", "checkpoint_sha256")).difference(z.files):
            raise ValueError(f"state archive contract failed: {path}")
        state = {key: np.asarray(z[key]) for key in ("patient_id", "phenotype_state", "full_state", "radiomics_prediction")}
        if state["phenotype_state"].shape != (808, 4, 64) or state["full_state"].shape != (808, 4, 192):
            raise ValueError(f"state shape contract failed: {path}")
        if not all(np.isfinite(value).all() for value in state.values() if value.dtype.kind in "fc"):
            raise ValueError(f"non-finite state archive: {path}")
        return state


def _split_sets(fold: int) -> dict[str, set[str]]:
    frame = load_folds()
    result: dict[str, set[str]] = {}
    current = frame.loc[frame["fold"].eq(int(fold))]
    for split in ("train", "val", "test"):
        result[split] = set(current.loc[current["split"].eq(split), "patient_id"].astype(str))
    return result


def _make_x(state: np.ndarray, visit: int) -> np.ndarray:
    one_hot = np.zeros((state.shape[0], 3), dtype=np.float64)
    one_hot[:, visit] = 1.0
    return np.concatenate((state.astype(np.float64), one_hot), axis=1)


def _fit_probe(state: np.ndarray, target: np.ndarray, mask: np.ndarray, ids: np.ndarray, splits: dict[str, set[str]], alphas: tuple[float, ...], visits: tuple[int, ...], multioutput: bool) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    positions = {str(pid): i for i, pid in enumerate(ids)}
    predictions: dict[int, np.ndarray] = {}
    truths: dict[int, np.ndarray] = {}
    selected: dict[int, float] = {}
    train_val_ids = splits["train"] | splits["val"]
    for visit in visits:
        valid = np.asarray([str(pid) in positions for pid in ids], dtype=bool) & mask[:, visit]
        train = valid & np.asarray([str(pid) in splits["train"] for pid in ids])
        val = valid & np.asarray([str(pid) in splits["val"] for pid in ids])
        test = valid & np.asarray([str(pid) in splits["test"] for pid in ids])
        if train.sum() < 20 or val.sum() < 10 or test.sum() < 10:
            raise ValueError(f"insufficient probe rows fold={len(splits['test'])} visit={visit}: {train.sum()}/{val.sum()}/{test.sum()}")
        x_train = _make_x(state[train, visit], visit)
        x_val = _make_x(state[val, visit], visit)
        x_test = _make_x(state[test, visit], visit)
        y_train = target[train, visit]
        y_val = target[val, visit]
        scaler = StandardScaler().fit(x_train)
        x_train_s, x_val_s = scaler.transform(x_train), scaler.transform(x_val)
        best_alpha, best_mse = None, float("inf")
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(x_train_s, y_train)
            mse = float(np.mean((model.predict(x_val_s) - y_val) ** 2))
            if mse < best_mse - 1e-12 or (abs(mse - best_mse) <= 1e-12 and (best_alpha is None or alpha > best_alpha)):
                best_alpha, best_mse = alpha, mse
        assert best_alpha is not None
        train_val = valid & np.asarray([str(pid) in train_val_ids for pid in ids])
        scaler = StandardScaler().fit(_make_x(state[train_val, visit], visit))
        model = Ridge(alpha=best_alpha).fit(scaler.transform(_make_x(state[train_val, visit], visit)), target[train_val, visit])
        predictions[visit] = model.predict(scaler.transform(x_test))
        truths[visit] = target[test, visit]
        selected[visit] = float(best_alpha)
    return predictions, truths, selected


def _direct_metrics(state: dict[str, np.ndarray], target: dict[str, Any], splits: dict[str, set[str]]) -> tuple[float, dict[str, float]]:
    pos = {str(pid): i for i, pid in enumerate(state["patient_id"])}
    correlations = []
    by_key: dict[str, float] = {}
    for pc in range(16):
        for visit in VISIT_INDEX:
            y, pred = [], []
            for i, pid in enumerate(target["patient_id"]):
                if str(pid) in splits["test"] and target["radiomics_mask"][i, visit] and str(pid) in pos:
                    y.append(target["radiomics"][i, visit, pc]); pred.append(state["radiomics_prediction"][pos[str(pid)], visit, pc])
            corr = _finite_corr(np.asarray(y), np.asarray(pred)); correlations.append(corr); by_key[f"pc{pc:02d}_{visit}"] = corr
    return float(np.nanmean(correlations)), by_key


def _probe_metrics(state: dict[str, np.ndarray], target: dict[str, Any], splits: dict[str, set[str]]) -> tuple[float, dict[str, float], dict[int, float]]:
    pos = {str(pid): i for i, pid in enumerate(state["patient_id"])}
    keep = np.asarray([str(pid) in pos for pid in target["patient_id"]])
    ids = target["patient_id"][keep]
    phenotype = np.stack([state["phenotype_state"][pos[str(pid)]] for pid in ids])
    values = target["radiomics"][keep]
    masks = target["radiomics_mask"][keep]
    preds: dict[tuple[int, int], np.ndarray] = {}
    ys: dict[tuple[int, int], np.ndarray] = {}
    alphas: dict[int, float] = {}
    for pc in range(16):
        p, y, selected = _fit_probe(phenotype, values[:, :, pc], masks, ids, splits, (0.1, 1.0, 10.0, 100.0), VISIT_INDEX, multioutput=False)
        alphas.update({pc * 10 + visit: alpha for visit, alpha in selected.items()})
        for visit in VISIT_INDEX:
            preds[(pc, visit)] = p[visit]; ys[(pc, visit)] = y[visit]
    correlations = {_key: _finite_corr(y, preds[_key]) for _key, y in ys.items()}
    return float(np.nanmean(list(correlations.values()))), {f"pc{pc:02d}_{visit}": value for (pc, visit), value in correlations.items()}, alphas


def _ftv_metrics(state: dict[str, np.ndarray], target: dict[str, Any], splits: dict[str, set[str]]) -> tuple[float, float]:
    pos = {str(pid): i for i, pid in enumerate(state["patient_id"])}
    keep = np.asarray([str(pid) in pos for pid in target["patient_id"]])
    ids = target["patient_id"][keep]
    full = np.stack([state["full_state"][pos[str(pid)]] for pid in ids])
    values = target["ftv"][keep]
    masks = target["ftv_mask"][keep]
    pred, truth, _ = _fit_probe(full, values, masks, ids, splits, (0.1, 1.0, 10.0, 100.0), VISIT_INDEX, multioutput=False)
    static = [_finite_corr(truth[v], pred[v]) for v in VISIT_INDEX]
    # The same visit-specific predictions yield a leakage-safe change diagnostic.
    deltas = [_finite_corr(truth[v] - truth[v - 1], pred[v] - pred[v - 1]) for v in (1, 2)]
    return float(np.nanmean(static)), float(np.nanmean(deltas))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--write-decision", action="store_true"); args = parser.parse_args()
    metrics_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, list[float]]] = {arm: {"direct": [], "probe": [], "ftv_static": [], "ftv_delta": []} for arm in PRIMARY_ARMS}
    fold_probe: dict[str, list[float]] = {arm: [] for arm in PRIMARY_ARMS}
    for fold in FOLDS:
        target = _load_target(fold); splits = _split_sets(fold)
        for arm in PRIMARY_ARMS:
            state = _load_state(fold, arm)
            direct, _ = _direct_metrics(state, target, splits)
            probe, _, alphas = _probe_metrics(state, target, splits)
            ftv_static, ftv_delta = _ftv_metrics(state, target, splits)
            row = {"seed": 2026, "fold": fold, "arm": arm, "direct_radiomics_macro_spearman": direct, "matched_probe_macro_spearman": probe, "ftv_static_spearman": ftv_static, "ftv_delta_spearman": ftv_delta, "selected_alpha_count": len(alphas), "implementation_version": IMPLEMENTATION_VERSION}
            metrics_rows.append(row); fold_probe[arm].append(probe)
            for key, value in (("direct", direct), ("probe", probe), ("ftv_static", ftv_static), ("ftv_delta", ftv_delta)): by_arm[arm][key].append(value)
            diagnostic_rows.append(row)
    summary: dict[str, dict[str, float]] = {}
    for arm in PRIMARY_ARMS:
        summary[arm] = {key: float(np.nanmean(values)) for key, values in by_arm[arm].items()}
        summary[arm]["positive_probe_folds_vs_F0"] = float(sum(a > b for a, b in zip(by_arm[arm]["probe"], by_arm["F0"]["probe"]))) if arm != "F0" else 0.0
        summary[arm]["probe_gain_vs_F0"] = summary[arm]["probe"] - summary["F0"]["probe"]
        summary[arm]["ftv_static_drop_vs_F0"] = summary["F0"]["ftv_static"] - summary[arm]["ftv_static"]
        summary[arm]["ftv_delta_drop_vs_F0"] = summary["F0"]["ftv_delta"] - summary[arm]["ftv_delta"]
    gate_rows = {}
    gate_rows["F0"] = {"checks": {}, "pass": True, "summary": summary["F0"]}
    for arm in ("F005", "F010", "F025"):
        s = summary[arm]
        checks = {
            "direct_abs_ge_0_10": s["direct"] >= 0.10,
            "matched_probe_abs_ge_0_10": s["probe"] >= 0.10,
            "matched_probe_gain_ge_0_05": s["probe_gain_vs_F0"] >= 0.05,
            "positive_fold_count_ge_4": s["positive_probe_folds_vs_F0"] >= 4,
            "ftv_static_drop_le_0_02": s["ftv_static_drop_vs_F0"] <= 0.02,
            "ftv_delta_drop_le_0_02": s["ftv_delta_drop_vs_F0"] <= 0.02,
        }
        gate_rows[arm] = {"checks": checks, "pass": bool(all(checks.values())), "summary": s}
    passing = [arm for arm in ("F005", "F010", "F025") if gate_rows[arm]["pass"]]
    selected = passing[0] if passing else None
    out = EXPERIMENT_ROOT / "metrics"; out.mkdir(exist_ok=True)
    with (out / "pilot_probe_metrics.csv").open("w", newline="") as stream:
        fields = list(metrics_rows[0]); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(metrics_rows)
    with (out / "pilot_probe_diagnostics.csv").open("w", newline="") as stream:
        fields = list(diagnostic_rows[0]); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(diagnostic_rows)
    gate = {"status": "PASS" if selected else "FAIL", "pilot_cells": 20, "arms": gate_rows, "selected_arm": selected, "outcome_fields_read": [], "clinical_fields_read": [], "representation_only": True}
    atomic_json(out / "pilot_gate.json", gate)
    decision = {"decision": "PILOT_LOCKED" if selected else "FACTORIZED_RADIOMICS_PILOT_NO_GO", "selected_arm": selected, "reason": "all outcome-blind mechanism checks passed" if selected else "no radiomics weight passed the prespecified outcome-blind mechanism gate", "pcr_evaluation": "LOCKED", "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(EXPERIMENT_ROOT / "decision.json", decision)
    print(json.dumps({"pilot_gate": gate["status"], "selected_arm": selected, "summary": summary}, indent=2))


if __name__ == "__main__": main()
