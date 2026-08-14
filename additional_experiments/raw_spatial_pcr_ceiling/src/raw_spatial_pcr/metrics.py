from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression


def _finite_binary(y_true: Iterable[float], probability: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(probability), dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], np.clip(p[keep], 1e-6, 1 - 1e-6)
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("binary metrics require both classes")
    return y, p


def calibration_slope(y_true: Iterable[float], probability: Iterable[float]) -> float:
    y, p = _finite_binary(y_true, probability)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, y.astype(int))
    return float(model.coef_[0, 0])


def ece10(y_true: Iterable[float], probability: Iterable[float]) -> float:
    y, p = _finite_binary(y_true, probability)
    bins = np.linspace(0.0, 1.0, 11)
    total = float(y.size)
    result = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        selected = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if selected.any():
            result += float(selected.sum()) / total * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return result


def classification_metrics(y_true: Iterable[float], probability: Iterable[float]) -> dict[str, float]:
    y, p = _finite_binary(y_true, probability)
    return {
        "n": float(y.size),
        "n_positive": float(y.sum()),
        "n_negative": float(y.size - y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": calibration_slope(y, p),
        "ece10": ece10(y, p),
    }


def attention_diagnostics(attention: np.ndarray, local_indices: np.ndarray | None = None) -> dict[str, float]:
    """Descriptive attention diagnostics; never used for model selection."""
    values = np.asarray(attention, dtype=float)
    if values.size == 0:
        return {"attention_entropy": math.nan, "attention_concentration_top10": math.nan, "center_mass": math.nan, "outer_mass": math.nan}
    values = values.reshape(-1, values.shape[-1])
    values = np.maximum(values, 0.0)
    values /= np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)
    entropy = -(values * np.log(np.maximum(values, 1e-12))).sum(axis=-1) / np.log(max(values.shape[-1], 2))
    order = np.sort(values, axis=-1)[:, ::-1]
    top = order[:, : max(1, int(np.ceil(values.shape[-1] * 0.10)))].sum(axis=-1)
    result = {"attention_entropy": float(entropy.mean()), "attention_concentration_top10": float(top.mean()), "center_mass": math.nan, "outer_mass": math.nan}
    if local_indices is not None:
        mask = np.asarray(local_indices, dtype=bool).reshape(-1)
        if mask.size == values.shape[-1]:
            result["center_mass"] = float(values[:, mask].sum(axis=-1).mean())
            result["outer_mass"] = float(values[:, ~mask].sum(axis=-1).mean())
    return result

