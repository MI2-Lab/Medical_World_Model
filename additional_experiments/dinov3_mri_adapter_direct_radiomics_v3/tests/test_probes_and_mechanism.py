import numpy as np
import pandas as pd

from dinov3_rg.mechanism import mechanism_gate
from dinov3_rg.probes import _fit_matched_ridge, safe_spearman


def test_matched_ridge_uses_registered_split_and_larger_alpha_tie_break():
    rng = np.random.default_rng(2)
    patient_ids = np.asarray([f"p{i}" for i in range(30)])
    x = rng.normal(size=(30, 6))
    y = np.zeros((30, 2))
    split = {
        patient_id: ("train" if i < 15 else "val" if i < 23 else "test")
        for i, patient_id in enumerate(patient_ids)
    }
    prediction, test, alpha = _fit_matched_ridge(x, y, patient_ids, split, (0.1, 1.0, 10.0, 100.0))
    assert prediction.shape == (7, 2)
    assert test.sum() == 7
    assert alpha == 100.0


def test_safe_spearman_handles_constant_inputs():
    assert np.isnan(safe_spearman(np.ones(8), np.arange(8)))
    assert safe_spearman(np.arange(8), np.arange(8)) == 1.0


def test_formal_mechanism_gate_requires_matched_transfer(tmp_path, monkeypatch):
    rows = []
    for seed in (7026, 8026, 9026, 10026, 11026):
        rows.extend([
            {
                "seed": seed, "arm": "C0",
                "direct_head_radiomics_macro_spearman": 0.0,
                "matched_probe_radiomics_macro_spearman": 0.04,
                "matched_probe_static_ftv_macro_spearman": 0.20,
                "matched_probe_delta_ftv_macro_spearman": 0.15,
                "state_mean_sd": 0.20,
            },
            {
                "seed": seed, "arm": "RAD",
                "direct_head_radiomics_macro_spearman": 0.15,
                "matched_probe_radiomics_macro_spearman": 0.11,
                "matched_probe_static_ftv_macro_spearman": 0.19,
                "matched_probe_delta_ftv_macro_spearman": 0.14,
                "state_mean_sd": 0.20,
            },
        ])
    monkeypatch.setattr(
        "dinov3_rg.mechanism._formal_training_safety",
        lambda _: {"status": "PASS", "checked_candidate_cells": 25, "failures": []},
    )
    gate = mechanism_gate(pd.DataFrame(rows), tmp_path)
    assert gate["status"] == "PASS"
    assert all(gate["gates"].values())
