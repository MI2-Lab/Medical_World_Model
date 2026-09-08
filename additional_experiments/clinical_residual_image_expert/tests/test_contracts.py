"""Small, data-free invariants for the residual experiment implementation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("clinical_residual_expert", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sigmoid_is_finite_and_bounded():
    values = MODULE.sigmoid(np.asarray([-1_000.0, 0.0, 1_000.0]))
    assert np.isfinite(values).all()
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_required_arms_include_recalibration_and_residual():
    assert "clinical_recalibration" in MODULE.ARMS
    assert "clinical_image_residual" in MODULE.ARMS
