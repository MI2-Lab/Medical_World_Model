from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def paired_patient_bootstrap(
    y_true: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    outer_fold: np.ndarray,
    draws: int = 5000,
    seed: int = 260814,
) -> dict[str, float]:
    """Paired bootstrap within outer fold; reference and candidate share rows."""
    y = np.asarray(y_true)
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    fold = np.asarray(outer_fold)
    if not (y.shape == ref.shape == cand.shape == fold.shape):
        raise ValueError("paired bootstrap arrays must have identical shape")
    rng = np.random.default_rng(seed)
    fold_values = np.unique(fold)
    effects = {"delta_auroc": [], "delta_auprc": [], "delta_brier": []}
    target_draws = int(draws)
    if target_draws <= 0:
        raise ValueError("draws must be positive")
    while len(effects["delta_auroc"]) < target_draws:
        selected = []
        for value in fold_values:
            members = np.flatnonzero(fold == value)
            selected.append(rng.choice(members, size=members.size, replace=True))
        indices = np.concatenate(selected)
        yy, rr, cc = y[indices], ref[indices], cand[indices]
        if np.unique(yy).size != 2:
            continue
        effects["delta_auroc"].append(float(roc_auc_score(yy, cc) - roc_auc_score(yy, rr)))
        effects["delta_auprc"].append(float(average_precision_score(yy, cc) - average_precision_score(yy, rr)))
        effects["delta_brier"].append(float(brier_score_loss(yy, cc) - brier_score_loss(yy, rr)))
    if not effects["delta_auroc"]:
        raise ValueError("no valid bootstrap draws")
    result: dict[str, float] = {"draws": float(len(effects["delta_auroc"]))}
    for name, values in effects.items():
        array = np.asarray(values)
        result[name] = float(array.mean())
        result[f"{name}_ci_low"] = float(np.quantile(array, 0.025))
        result[f"{name}_ci_high"] = float(np.quantile(array, 0.975))
    return result
