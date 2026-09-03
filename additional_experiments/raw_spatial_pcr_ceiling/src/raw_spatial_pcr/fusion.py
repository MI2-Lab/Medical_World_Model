from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_fold_safe_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit fusion on train, select C on validation, and predict test once."""
    best_c = None
    best_score = -np.inf
    from sklearn.metrics import roc_auc_score

    for c_value in c_grid:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=10000))
        model.fit(train_x, train_y)
        score = roc_auc_score(validation_y, model.predict_proba(validation_x)[:, 1])
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (best_c is None or c_value < best_c)):
            best_score = score
            best_c = c_value
    if best_c is None:
        raise RuntimeError("no validation model selected")
    selected = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, penalty="l2", solver="liblinear", max_iter=10000))
    selected.fit(train_x, train_y)
    return selected.predict_proba(test_x)[:, 1], {"selected_c": float(best_c), "validation_auroc": float(best_score)}

