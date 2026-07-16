from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ReadoutConfig


LANDMARK_NAMES = ("T0", "T0+T1", "T0+T1+T2")


def landmark_features(states: np.ndarray, landmark: int) -> np.ndarray:
    """Construct the primary FLR representation.

    Input:
        ``states [N,3,Ds]`` containing s-hat1/s-hat2/s-hat3.
    Output:
        ``x [N,20*Ds+3]``. For ``Ds=64``, this is ``[N,1283]``.
    """

    if states.ndim != 3 or states.shape[1] != 3 or landmark not in (0, 1, 2):
        raise ValueError(f"Expected states [N,3,Ds] and landmark 0/1/2, got {states.shape}, {landmark}")
    first = states[:, 0]
    current = states[:, landmark]
    mean = states[:, : landmark + 1].mean(axis=1)
    recent_revision = np.zeros_like(first) if landmark == 0 else states[:, landmark] - states[:, landmark - 1]
    displacement = current - first
    base = np.concatenate((first, current, mean, recent_revision, displacement), axis=1)
    landmark_code = np.zeros((len(states), 3), dtype=np.float32)
    landmark_code[:, landmark] = 1.0
    interactions = (base[:, None, :] * landmark_code[:, :, None]).reshape(len(states), -1)
    return np.concatenate((base, interactions, landmark_code), axis=1).astype(np.float32)


def _stack_landmarks(states: np.ndarray, labels: np.ndarray, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    matrices, outcomes = [], []
    for landmark in range(3):
        matrices.append(landmark_features(states[indices], landmark))
        outcomes.append(labels[indices])
    return np.concatenate(matrices), np.concatenate(outcomes)


def _model(penalty: str, c_value: float, max_iter: int) -> Any:
    estimator = LogisticRegression(
        penalty=penalty,
        C=c_value,
        solver="liblinear",
        class_weight="balanced",
        max_iter=max_iter,
        random_state=0,
    )
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _auc_by_landmark(model: Any, states: np.ndarray, labels: np.ndarray, indices: list[int]) -> list[float]:
    output = []
    for landmark in range(3):
        probability = model.predict_proba(landmark_features(states[indices], landmark))[:, 1]
        output.append(float(roc_auc_score(labels[indices], probability)))
    return output


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    positive = labels == 1
    negative = labels == 0
    sensitivity = float(((prediction == 1) & positive).sum() / max(int(positive.sum()), 1))
    specificity = float(((prediction == 0) & negative).sum() / max(int(negative.sum()), 1))
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def fit_frozen_landmark_readout(
    states_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    config: ReadoutConfig,
) -> dict[str, Any]:
    """Fit one shared class-balanced logistic FLR and evaluate all landmarks."""

    cache = np.load(states_path, allow_pickle=False)
    states = cache["future_response_state"].astype(np.float32)
    labels = cache["pcr"].astype(np.int64)
    n_primary = int(cache["n_primary"])
    if np.any(labels[:n_primary] < 0):
        raise RuntimeError("Primary records must have pCR labels for FLR")
    splits = json.loads(Path(splits_path).read_text())
    train_indices = [int(index) for index in splits["primary_train"]]
    validation_indices = [int(index) for index in splits["validation"]]
    test_indices = [int(index) for index in splits["test"]]
    train_x, train_y = _stack_landmarks(states, labels, train_indices)
    selection_weights = np.asarray(config.landmark_weights, dtype=np.float64)
    best: dict[str, Any] | None = None
    best_model: Any = None
    for penalty in config.penalties:
        for c_value in config.c_grid:
            model = _model(penalty, c_value, config.max_iter)
            model.fit(train_x, train_y)
            validation_auc = _auc_by_landmark(model, states, labels, validation_indices)
            selection = float(np.dot(validation_auc, selection_weights) / selection_weights.sum())
            candidate = {
                "penalty": penalty,
                "C": float(c_value),
                "validation_selection_auroc": selection,
                "validation_landmark_auroc": validation_auc,
            }
            if best is None or (selection, -np.log10(c_value), penalty == "l2") > (
                best["validation_selection_auroc"],
                -np.log10(best["C"]),
                best["penalty"] == "l2",
            ):
                best, best_model = candidate, model
    if best is None or best_model is None:
        raise RuntimeError("FLR hyperparameter grid is empty")
    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    patient_ids = cache["patient_ids"].astype(str)
    for split_name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        for landmark, name in enumerate(LANDMARK_NAMES):
            probability = best_model.predict_proba(landmark_features(states[indices], landmark))[:, 1]
            rows.append({"split": split_name, "landmark": name, "n": len(indices), **_metrics(labels[indices], probability)})
            score_rows.extend(
                {
                    "patient_id": patient_ids[index],
                    "split": split_name,
                    "landmark": name,
                    "pcr": int(labels[index]),
                    "probability": float(probability[row]),
                }
                for row, index in enumerate(indices)
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "flr.pkl").open("wb") as stream:
        pickle.dump(best_model, stream)
    with (output_dir / "flr_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "flr_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    summary = {"selection": best, "feature_dim": int(20 * states.shape[-1] + 3), "metrics": rows}
    (output_dir / "flr_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
