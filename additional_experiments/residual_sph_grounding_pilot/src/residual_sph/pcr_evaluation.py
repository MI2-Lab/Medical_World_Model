"""Post-freeze fold-safe pCR modeling primitives.

This module contains no paths and performs no I/O.  The executable that reads
clinical labels imports it only after the representation-freeze verifier passes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


NUMERIC_FIELDS = ("label_hr", "label_her2", "label_mp", "age_at_screening")
CATEGORICAL_FIELDS = ("race_simple", "menopausal_status_simple", "ethnicity", "arm")
CLINICAL_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TIMINGS = ("T0", "T0-T1", "T0-T2", "T0-T3")
MODEL_NAMES = ("M", "C", "C+M", "C+F", "C+F+M")


class TrainOnlyClinicalEncoder:
    def __init__(self) -> None:
        self.numeric_medians: dict[str, float] = {}
        self.categories: dict[str, tuple[str, ...]] = {}
        self.fitted = False

    @staticmethod
    def _category(series: pd.Series) -> np.ndarray:
        return np.asarray(
            ["__MISSING__" if pd.isna(value) or not str(value).strip() else str(value) for value in series],
            dtype=object,
        )

    def fit(self, train: pd.DataFrame) -> "TrainOnlyClinicalEncoder":
        if self.fitted or train.empty:
            raise ValueError("clinical encoder must fit once on nonempty train rows")
        if missing := set(CLINICAL_FIELDS).difference(train.columns):
            raise ValueError(f"clinical fields are missing: {sorted(missing)}")
        for field in NUMERIC_FIELDS:
            values = pd.to_numeric(train[field], errors="raise").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not finite.size:
                raise ValueError(f"no train value for numeric clinical field {field}")
            self.numeric_medians[field] = float(np.median(finite))
        for field in CATEGORICAL_FIELDS:
            values = self._category(train[field])
            self.categories[field] = tuple(sorted(set(values.tolist()) | {"__MISSING__"}))
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("clinical encoder is not fitted")
        blocks: list[np.ndarray] = []
        for field in NUMERIC_FIELDS:
            values = pd.to_numeric(frame[field], errors="raise").to_numpy(dtype=np.float64)
            blocks.append(np.where(np.isnan(values), self.numeric_medians[field], values)[:, None])
        for field in CATEGORICAL_FIELDS:
            values = self._category(frame[field])
            levels = self.categories[field]
            lookup = {level: index for index, level in enumerate(levels)}
            block = np.zeros((len(frame), len(levels)), dtype=np.float64)
            for row, value in enumerate(values):
                index = lookup.get(str(value))
                if index is not None:
                    block[row, index] = 1.0
            blocks.append(block)
        output = np.concatenate(blocks, axis=1)
        if not np.isfinite(output).all():
            raise ValueError("clinical encoding contains non-finite values")
        return output


@dataclass(frozen=True)
class LogisticFit:
    scaler: StandardScaler
    model: LogisticRegression
    selected_c: float
    validation_auroc: float
    grid_scores: tuple[tuple[float, float], ...]
    feature_dim: int

    def predict_probability(self, matrix: Any) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError("post-freeze prediction feature shape drifted")
        probability = self.model.predict_proba(self.scaler.transform(values))[:, 1]
        if not np.isfinite(probability).all():
            raise FloatingPointError("post-freeze probability is non-finite")
        return np.asarray(probability, dtype=np.float64)


def fit_logistic(
    train_x: Any,
    train_y: Any,
    val_x: Any,
    val_y: Any,
) -> LogisticFit:
    train_x = np.asarray(train_x, dtype=np.float64)
    val_x = np.asarray(val_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.int64).reshape(-1)
    val_y = np.asarray(val_y, dtype=np.int64).reshape(-1)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("logistic feature shapes differ")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("logistic row counts differ")
    if set(np.unique(train_y)) != {0, 1} or set(np.unique(val_y)) != {0, 1}:
        raise ValueError("train and validation must contain both pCR classes")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise ValueError("logistic features contain non-finite values")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    candidates: list[tuple[float, float, LogisticRegression]] = []
    for c_value in C_GRID:
        model = LogisticRegression(
            penalty="l2",
            C=c_value,
            solver="liblinear",
            class_weight=None,
            max_iter=10_000,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_scaled, train_y)
        score = float(roc_auc_score(val_y, model.predict_proba(val_scaled)[:, 1]))
        if not math.isfinite(score):
            raise FloatingPointError("validation AUROC is non-finite")
        candidates.append((c_value, score, model))
    best = max(score for _, score, _ in candidates)
    c_value, score, model = min(
        (item for item in candidates if item[1] >= best - 1e-12),
        key=lambda item: item[0],
    )
    return LogisticFit(
        scaler=scaler,
        model=model,
        selected_c=float(c_value),
        validation_auroc=float(score),
        grid_scores=tuple((float(c), float(value)) for c, value, _ in candidates),
        feature_dim=int(train_x.shape[1]),
    )


def timing_prefix(response: Any, ftv: Any, timing: str) -> tuple[np.ndarray, np.ndarray]:
    response = np.asarray(response, dtype=np.float64)
    ftv = np.asarray(ftv, dtype=np.float64)
    if response.ndim != 3 or response.shape[1:] != (4, 192):
        raise ValueError("response state must be [N,4,192]")
    if ftv.shape != (len(response), 4) or np.any(ftv < 0):
        raise ValueError("FTV must be nonnegative [N,4]")
    if timing not in TIMINGS:
        raise ValueError("unknown information timing")
    end = TIMINGS.index(timing) + 1
    return response[:, :end].reshape(len(response), end * 192), np.log1p(ftv[:, :end])


def feature_sets(
    clinical: np.ndarray,
    mri: np.ndarray,
    ftv: np.ndarray,
) -> Mapping[str, np.ndarray]:
    return {
        "M": mri,
        "C": clinical,
        "C+M": np.concatenate((clinical, mri), axis=1),
        "C+F": np.concatenate((clinical, ftv), axis=1),
        "C+F+M": np.concatenate((clinical, ftv, mri), axis=1),
    }


__all__ = [
    "C_GRID",
    "CLINICAL_FIELDS",
    "MODEL_NAMES",
    "TIMINGS",
    "LogisticFit",
    "TrainOnlyClinicalEncoder",
    "feature_sets",
    "fit_logistic",
    "timing_prefix",
]
