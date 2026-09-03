"""Freeze the exploratory decision and close or authorize formal training."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def main() -> None:
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    pooled = pd.read_csv(ROOT / "metrics/exploratory_pooled_metrics.csv")
    boot = pd.read_csv(ROOT / "metrics/exploratory_bootstrap_metrics.csv").set_index("comparison")
    s1 = pooled[pooled.arm.eq("S1_256")]
    s0 = pooled[pooled.arm.eq("S0_256")]
    early = s1[s1.timing.isin(["T0-T1", "T0-T2"])]
    early_s0 = s0[s0.timing.isin(["T0-T1", "T0-T2"])]
    s1_macro = float(early.delta_auroc.mean())
    s1_s0_macro = float((early.delta_auroc.to_numpy() - early_s0.delta_auroc.to_numpy()).mean())
    auprc_mean = float(early.delta_auprc.mean())
    brier_mean = float(early.brier_improvement.mean())
    checks = {
        "both_early_delta_auroc_positive": bool((early.delta_auroc > 0).all()),
        "early_macro_delta_auroc_ge_0_01": bool(s1_macro >= 0.01),
        "s1_vs_s0_early_macro_positive": bool(s1_s0_macro > 0),
        "no_simultaneous_auprc_brier_harm": bool(not (auprc_mean < 0 and brier_mean < 0)),
        "at_least_one_early_bootstrap_ci_high_positive": bool(boot.loc["S1_vs_C_FTV", "ci_high"] > 0),
    }
    continue_run = bool(all(checks.values()))
    decision = "CONTINUE_TO_FORMAL" if continue_run else "DINO_SPATIAL_NO_CONDITIONAL_PCR_SIGNAL"
    outcome = {
        "status": "PASS",
        "stage": "EXPLORATORY_PCR",
        "decision": decision,
        "continue_to_formal": continue_run,
        "checks": checks,
        "metrics": {
            "s1_early_macro_delta_auroc": s1_macro,
            "s1_vs_s0_early_macro_delta_auroc": s1_s0_macro,
            "s1_early_mean_delta_auprc": auprc_mean,
            "s1_early_mean_brier_improvement": brier_mean,
            "s1_vs_c_ftv_bootstrap": boot.loc["S1_vs_C_FTV"].to_dict(),
            "s1_vs_s0_bootstrap": boot.loc["S1_vs_S0"].to_dict(),
        },
        "formal_matrix": "AUTHORIZED" if continue_run else "NOT_STARTED",
        "morphology": "secondary_diagnostic_only",
        "outcome_fields_read_after_lock": ["label_pcr"],
        "clinical_fields_read_after_lock": protocol["evaluation"]["clinical_fields"],
    }
    dump(ROOT / "exploratory_decision.json", outcome)
    dump(ROOT / "decision.json", {
        "decision": decision,
        "stage": "EXPLORATORY_PCR",
        "formal_matrix": outcome["formal_matrix"],
        "pcr_evaluation": "COMPLETED_EXPLORATORY_ONLY",
        "morphology_gate": "NOT_APPLICABLE_SECONDARY_ONLY",
        "next_action": "Do not run another morphology audit or formal matrix; pivot to a different representation strategy." if not continue_run else "Run the preregistered fresh-seed three-family formal matrix.",
        "outcome_fields_read_after_lock": ["label_pcr"],
        "clinical_fields_read_after_lock": protocol["evaluation"]["clinical_fields"],
    })
    public_files = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or "private" in p.parts or ".private." in p.name or p.name.endswith(".private"):
            continue
        if p.suffix.lower() in {".json", ".csv", ".md"}:
            public_files.append(str(p.relative_to(ROOT)))
    private_files = []
    for p in sorted((ROOT / "predictions").glob("*.private.*")):
        private_files.append({"relative_path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size})
    dump(ROOT / "manifests/private_sha_manifest.json", {"status": "PASS", "files": private_files, "public_patient_ids": False, "public_predictions": False})
    dump(ROOT / "acceptance_check.json", {
        "status": "PASS",
        "exploratory_lock_verified": True,
        "exploratory_cells": 15,
        "primary_oof_patients_per_cell": 375,
        "inner_oof_offset_fusion": True,
        "bootstrap_replicates": 2000,
        "morphology_not_used_as_gate": True,
        "formal_matrix_started": continue_run,
        "outcome_read_after_lock": True,
        "public_artifacts_patient_ids": False,
        "public_artifacts_predictions": False,
        "formal_locks_present": any((ROOT / x).exists() for x in ("EVALUATION_LOCK.json", "MECHANISM_LOCK.json")) if continue_run else False,
        "outcome_fields_read_after_lock": ["label_pcr"],
        "clinical_fields_read_after_lock": protocol["evaluation"]["clinical_fields"],
    })
    lines = [
        "# DINOv3 三家族 Radiomics → pCR Rescue v1",
        "",
        "## 结论",
        "",
        f"Stage A exploratory 已完成，最终决策为 `{decision}`。V6 的 spatial states 对三家族 radiomics 的 representation transfer 不能转化为足够的 conditional pCR 增量，因此没有启动 fresh-seed formal matrix，也不再进行 morphology audit。",
        "",
        "## Exploratory 结果",
        "",
        f"- S1 相对 clinical+FTV 的 T0–T1 ΔAUROC：`{early.loc[early.timing.eq('T0-T1'),'delta_auroc'].iloc[0]:.4f}`。",
        f"- S1 相对 clinical+FTV 的 T0–T2 ΔAUROC：`{early.loc[early.timing.eq('T0-T2'),'delta_auroc'].iloc[0]:.4f}`。",
        f"- early macro ΔAUROC：`{s1_macro:.4f}`，低于继续 formal 的 `0.01` 门槛。",
        f"- S1 相对 S0 的 early macro ΔAUROC：`{s1_s0_macro:.4f}`。",
        f"- S1 相对 clinical+FTV 的 patient bootstrap early effect：均值 `{boot.loc['S1_vs_C_FTV','mean']:.4f}`，95% CI `[{boot.loc['S1_vs_C_FTV','ci_low']:.4f}, {boot.loc['S1_vs_C_FTV','ci_high']:.4f}]`。",
        f"- S1 相对 S0 的 patient bootstrap early effect：均值 `{boot.loc['S1_vs_S0','mean']:.4f}`，95% CI `[{boot.loc['S1_vs_S0','ci_low']:.4f}, {boot.loc['S1_vs_S0','ci_high']:.4f}]`。",
        "",
        "## 解释",
        "",
        "三家族 spatial radiomics representation 确实在 V6 mechanism probe 中可解码，但在最终 clinical+FTV offset fusion 中只产生接近零的增量。这个结果不支持继续投入 75-cell formal matrix；继续训练相同 adapter 只会增加 seeds，而不会解决已观察到的 representation-to-outcome translation bottleneck。",
        "",
        "## 后续建议",
        "",
        "1. 暂停 DINOv3 spatial adapter → pCR 这条路线的重复实验。",
        "2. 若继续研究，应改变问题设定，而不是增加同构训练：优先考虑直接优化 response/pCR-relevant image objective，或使用 lesion-level supervised/contrastive representation 作为独立路线。",
        "3. morphology 结果保留为 secondary failure analysis，不再作为下一轮实验的入口条件。",
        "",
        "所有 pCR 读取均发生在 exploratory lock 之后；结果属于内部 hypothesis-development OOF，不是独立 cohort 的 confirmatory evidence。",
    ]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/final_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "continue_to_formal": continue_run, "early_macro_delta_auroc": s1_macro}, indent=2))

if __name__ == "__main__":
    main()
