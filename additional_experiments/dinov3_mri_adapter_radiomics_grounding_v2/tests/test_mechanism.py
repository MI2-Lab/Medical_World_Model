import numpy as np
import pandas as pd

import dinov3_rg.mechanism as mechanism_module
from dinov3_rg.mechanism import mechanism_gate


def test_mechanism_gate_requires_absolute_and_relative_signal(monkeypatch, tmp_path):
    rows = []
    for seed in (2026, 3026, 4026, 5026, 6026):
        for arm, radiomics in (("D2", 0.10), ("D3", 0.20)):
            rows.append(
                {
                    "seed": seed,
                    "fold": 0,
                    "arm": arm,
                    "radiomics_pc_macro_spearman": radiomics,
                    "static_ftv_macro_spearman": 0.30,
                    "delta_ftv_macro_spearman": 0.20,
                    "state_mean_sd": 0.20,
                }
            )
    monkeypatch.setattr(
        mechanism_module,
        "_gradient_safety",
        lambda _: {"status": "PASS", "checked": 25, "failures": []},
    )
    gate = mechanism_gate(pd.DataFrame(rows), tmp_path)
    assert gate["status"] == "PASS"
    assert all(gate["gates"].values())


def test_mechanism_gate_rejects_relative_only_signal(monkeypatch, tmp_path):
    rows = []
    for seed in (2026, 3026, 4026, 5026, 6026):
        for arm, radiomics in (("D2", -0.10), ("D3", 0.00)):
            rows.append(
                {
                    "seed": seed,
                    "fold": 0,
                    "arm": arm,
                    "radiomics_pc_macro_spearman": radiomics,
                    "static_ftv_macro_spearman": 0.30,
                    "delta_ftv_macro_spearman": 0.20,
                    "state_mean_sd": 0.20,
                }
            )
    monkeypatch.setattr(
        mechanism_module,
        "_gradient_safety",
        lambda _: {"status": "PASS", "checked": 25, "failures": []},
    )
    gate = mechanism_gate(pd.DataFrame(rows), tmp_path)
    assert gate["status"] == "NO_GO"
    assert not gate["gates"]["d3_absolute_radiomics_spearman"]
