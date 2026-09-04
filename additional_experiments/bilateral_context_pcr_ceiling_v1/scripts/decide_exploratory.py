#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics = pd.read_csv(ROOT / "metrics/exploratory_pcr_metrics.csv")
    bootstrap = pd.read_csv(ROOT / "metrics/exploratory_bootstrap_metrics.csv")
    early = ["T0_T1", "T0_T2"]
    s1 = metrics.loc[(metrics.arm == "S1_256") & metrics.timing.isin(early)]
    s0 = metrics.loc[(metrics.arm == "S0_256") & metrics.timing.isin(early)]
    if len(s1) != 2 or len(s0) != 2:
        raise SystemExit("exploratory metrics are incomplete")
    by_timing = {row.timing: row for row in s1.itertuples()}
    b1_boot = bootstrap.loc[(bootstrap.arm == "S1_256") & bootstrap.timing.isin(early)]
    checks = {
        "both_early_delta_auroc_positive": bool((s1["delta_auroc"] > 0).all()),
        "early_macro_delta_auroc": float(s1["delta_auroc"].mean()),
        "early_macro_threshold": bool(s1["delta_auroc"].mean() >= 0.01),
        "spatial_extra_delta_auroc": float((s1.set_index("timing")["auroc"] - s0.set_index("timing")["auroc"]).mean()),
        "spatial_extra_positive": bool((s1.set_index("timing")["auroc"] - s0.set_index("timing")["auroc"]).mean() > 0),
        "auprc_not_decreased": bool((s1["delta_auprc"] >= 0).all()),
        "brier_not_degraded": bool((s1["delta_brier"] <= 0).all()),
        "bootstrap_upper_supports_positive": bool((b1_boot["ci_high"] > 0).any()) if not b1_boot.empty else False,
    }
    # Explicitly report the paired metrics; the rough screening gate is deliberately conservative.
    passed = all((checks["both_early_delta_auroc_positive"], checks["early_macro_threshold"], checks["spatial_extra_positive"], checks["bootstrap_upper_supports_positive"], checks["auprc_not_decreased"], checks["brier_not_degraded"]))
    result = {"status": "PASS" if passed else "FAIL", "decision": "CONTINUE_STAGE_B" if passed else "DINO_SPATIAL_NO_CONDITIONAL_PCR_SIGNAL", "checks": checks, "seed": 2026, "pcr_evaluation_stage": "exploratory_only", "formal_training": "UNLOCKED" if passed else "LOCKED", "outcome_fields_read": ["label_pcr"], "clinical_fields_read": ["clinical+FTV for fusion only"]}
    (ROOT / "metrics/exploratory_decision.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "decision.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
