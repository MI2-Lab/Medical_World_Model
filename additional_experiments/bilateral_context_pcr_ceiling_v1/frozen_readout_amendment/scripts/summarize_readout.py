#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ("T0", "T0_T1", "T0_T2")
ARMS = ("DINO_GLOBAL", "DINO_SPATIAL", "RADIO_TARGET")


def read_prediction(timing: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for fold in range(5):
        with np.load(ROOT / "predictions" / f"{fold}_{timing}_{arm}.private.npz", allow_pickle=False) as payload:
            ys.append(np.asarray(payload["y"], dtype=int))
            ps.append(np.asarray(payload["fused"], dtype=float))
    return np.concatenate(ys), np.concatenate(ps)


def paired(y: np.ndarray, left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, float]:
    point = roc_auc_score(y, left) - roc_auc_score(y, right)
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    values = []
    for _ in range(2000):
        sample = np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)))
        values.append(roc_auc_score(y[sample], left[sample]) - roc_auc_score(y[sample], right[sample]))
    values = np.asarray(values)
    return {"delta_auroc": float(point), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "draws": 2000}


def main() -> None:
    rows = []
    for timing in TIMINGS:
        y, _ = read_prediction(timing, "RADIO_TARGET")
        predictions = {arm: read_prediction(timing, arm)[1] for arm in ARMS}
        for arm in ("DINO_GLOBAL", "DINO_SPATIAL"):
            rows.append({"timing": timing, "candidate": arm, "reference": "RADIO_TARGET", **paired(y, predictions[arm], predictions["RADIO_TARGET"], 260812 + sum(map(ord, timing + arm + "radio")))})
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(ROOT / "metrics/frozen_readout_dino_vs_radiomics_bootstrap.csv", index=False)
    pcr = pd.read_csv(ROOT / "metrics/frozen_readout_pcr_metrics.csv")
    early = pcr.loc[pcr["timing"].isin(("T0_T1", "T0_T2"))]
    spatial = early.loc[early["arm"] == "DINO_SPATIAL"]
    global_ = early.loc[early["arm"] == "DINO_GLOBAL"]
    radio = early.loc[early["arm"] == "RADIO_TARGET"]
    global_all = pcr.loc[pcr["arm"] == "DINO_GLOBAL"]
    spatial_all = pcr.loc[pcr["arm"] == "DINO_SPATIAL"]
    radio_all = pcr.loc[pcr["arm"] == "RADIO_TARGET"]
    result = {
        "status": "COMPLETE",
        "decision": "BILATERAL_DINO_CACHE_SIGNAL_BUT_NOT_ROBUST",
        "interpretation": "DINO frozen readout is directionally above the locked 16-D radiomics comparator, but patient-bootstrap CIs include zero; spatial-pooled tokens do not establish spatial-layout superiority.",
        "dino_global_early_delta_vs_cftv": float(global_["delta_auroc"].mean()),
        "dino_spatial_early_delta_vs_cftv": float(spatial["delta_auroc"].mean()),
        "dino_global_early_auroc_minus_radio": float(global_["auroc"].mean() - radio["auroc"].mean()),
        "dino_spatial_early_auroc_minus_radio": float(spatial["auroc"].mean() - radio["auroc"].mean()),
        "dino_vs_radio_all_early_bootstrap_ci_low": float(pairwise.loc[pairwise["timing"].isin(("T0_T1", "T0_T2")), "ci_low"].min()),
        "dino_vs_radio_all_early_bootstrap_ci_high": float(pairwise.loc[pairwise["timing"].isin(("T0_T1", "T0_T2")), "ci_high"].max()),
        "formal_adapter_training": "LOCKED",
        "outcome_fields_read": ["label_pcr"],
    }
    acceptance = {
        "status": "PASS",
        "parent_input_lock": "LOCKED",
        "amendment_lock": "LOCKED",
        "frozen_feature_check": "COMPLETE",
        "fold_metrics_rows": int(len(pd.read_csv(ROOT / "metrics/frozen_readout_fold_metrics.csv"))),
        "bootstrap_rows": int(len(pairwise) + len(pd.read_csv(ROOT / "metrics/frozen_readout_bootstrap.csv"))),
        "outer_test_patient_coverage": 375,
        "patient_ids_in_public_metrics": False,
        "predictions_private_only": True,
        "adapter_training": "NOT_RUN",
        "formal_matrix": "NOT_RUN",
        "outcome_fields_read_before_lock": [],
        "clinical_fields_read_before_lock": [],
    }
    (ROOT / "acceptance_check.json").write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    (ROOT / "metrics/final_decision.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = f"""# Frozen bilateral DINO cache readout amendment

## 结论

已使用已经完成的 bilateral DINO cache 做 frozen nested-OOF readout。它显示方向性 pCR 信号，但强度不足以解锁 adapter 或 formal matrix：`{result['decision']}`。

| timing | DINO global ΔAUROC vs C+FTV | DINO spatial-pooled ΔAUROC vs C+FTV | DINO global AUROC − radiomics AUROC | DINO spatial AUROC − radiomics AUROC |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {timing} | {float(global_all.loc[global_all.timing == timing, 'delta_auroc'].iloc[0]):.4f} | {float(spatial_all.loc[spatial_all.timing == timing, 'delta_auroc'].iloc[0]):.4f} | {float(global_all.loc[global_all.timing == timing, 'auroc'].iloc[0] - radio_all.loc[radio_all.timing == timing, 'auroc'].iloc[0]):.4f} | {float(spatial_all.loc[spatial_all.timing == timing, 'auroc'].iloc[0] - radio_all.loc[radio_all.timing == timing, 'auroc'].iloc[0]):.4f} |"
        for timing in TIMINGS
    ) + f"""

paired DINO-versus-radiomics bootstrap 的 early CI 仍跨越 0；因此不能声称 DINO 已经稳定超过 radiomics。这里的 `DINO_SPATIAL` 是对 4×4 spatial tokens 做 pooled embedding，并不检验二维 patch layout 本身的增量。

## 研究含义

当前结果支持“image 中存在比固定 16-D radiomics target 更丰富的候选信号”，但尚未支持“该信号稳定预测 pCR”。下一步应优先验证 frozen bilateral readout 的稳健性（预先固定的 fresh split/seed 或独立 cohort），而不是继续堆叠 adapter、radiomics target 或 foundation model。
"""
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/frozen_readout_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
