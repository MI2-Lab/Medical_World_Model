from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raw_spatial_pcr.bootstrap import paired_patient_bootstrap
from raw_spatial_pcr.metrics import classification_metrics, ece10


def test_classification_metrics_are_finite() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.6])
    result = classification_metrics(y, p)
    assert result["auroc"] > 0.9
    assert np.isfinite(list(result.values())).all()
    assert 0.0 <= ece10(y, p) <= 1.0


def test_bootstrap_is_paired_and_fold_stratified() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    reference = np.array([0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.8])
    candidate = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9])
    folds = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    result = paired_patient_bootstrap(y, reference, candidate, folds, draws=100, seed=1)
    assert result["draws"] == 100
    assert result["delta_auroc_ci_low"] <= result["delta_auroc_ci_high"]
    assert result["delta_brier"] < 0

