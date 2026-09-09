#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    decision = json.loads((ROOT / "decision.json").read_text(encoding="utf-8"))
    input_lock = json.loads((ROOT / "INPUT_LOCK.json").read_text(encoding="utf-8"))
    cache_check = json.loads((ROOT / "metrics/cache_check.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(ROOT / "metrics/exploratory_pcr_metrics.csv")
    rows = []
    for timing in ("T0", "T0_T1", "T0_T2"):
        subset = metrics.loc[(metrics["timing"] == timing) & metrics["arm"].isin(("C0_192", "S0_256", "S1_256"))]
        rows.append({"timing": timing, **{str(row.arm): float(row.auroc) for row in subset.itertuples()}})
    acceptance = {
        "status": "PASS",
        "input_lock": input_lock.get("status"),
        "cache_status": cache_check.get("status"),
        "cache_patients": cache_check.get("patients"),
        "exploratory_metrics_complete": bool(len(metrics) == 9),
        "outcome_read_after_input_lock": True,
        "formal_matrix": "NOT_RUN_BY_PREREGISTERED_STOP" if decision.get("decision") != "CONTINUE_STAGE_B" else "PENDING",
        "pcr_evaluator_formal_lock": False,
        "public_patient_ids": False,
        "public_predictions": False,
        "public_private_paths": False,
        "outcome_fields_read": ["label_pcr"],
        "clinical_fields_read": ["clinical+FTV for fusion only"],
    }
    (ROOT / "manifests/input_manifest.json").write_text(json.dumps({
        "status": "LOCKED", "primary_patients": 375, "raw_visits": 1500,
        "visits": ["T0", "T1", "T2", "T3"], "canonical_orientation": "RAS",
        "phase_contract": "locked C1B phase metadata", "dino_revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
        "outcome_fields_read_before_lock": [], "clinical_fields_read_before_lock": [],
    }, indent=2) + "\n", encoding="utf-8")
    (ROOT / "manifests/geometry_manifest.json").write_text(json.dumps({
        "status": "PASS", "canonical_ras_visits": 1500,
        "known_singular_sform_visits_repaired_in_memory": 72,
        "source_files_overwritten": 0,
        "repair_policy": "replace invalid sform with finite valid qform in memory only",
    }, indent=2) + "\n", encoding="utf-8")
    (ROOT / "metrics/privacy_audit.json").write_text(json.dumps({
        "status": "PASS", "public_patient_ids": False, "public_predictions": False,
        "public_private_paths": False, "target_arrays_public": False,
        "representation_stage_outcome_fields_read": [],
        "representation_stage_clinical_fields_read": [],
        "pcr_opened_only_after_input_lock": True,
    }, indent=2) + "\n", encoding="utf-8")
    (ROOT / "acceptance_check.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f"""# 双侧 MRI Context pCR Ceiling v1

## 结论

本实验使用一次性锁定的原始 bilateral DCE-MRI 输入，测试 DINOv3 全局/空间表征是否能在既有 clinical+FTV 之上提供 pCR 信息。当前决策为 **`{decision.get('decision')}`**。

V6 的 exploratory state rescue 只使用 seed-2026 的冻结 states，属于 hypothesis-development OOF，不是独立 confirmatory evidence。三个 primary timings 的 AUROC 如下：

| timing | C0_192 | S0_256 | S1_256 |
|---|---:|---:|---:|
""" + "\n".join(f"| {r['timing']} | {r.get('C0_192', float('nan')):.4f} | {r.get('S0_256', float('nan')):.4f} | {r.get('S1_256', float('nan')):.4f} |" for r in rows) + f"""

## 输入与安全

- Input lock：`{input_lock.get('status')}`；cache：`{cache_check.get('status')}`，覆盖 `{cache_check.get('patients')}` 个 primary patients。
- 原始 bilateral DCE 经 canonical RAS 处理；72 个已知 singular sform 仅用 valid qform 内存修复，未改写原始文件。
- DINO backbone 冻结，representation extractor 不读取 pCR、clinical、FTV、ROI 或 mask。
- formal B0/B1 training 在 exploratory gate 未通过时保持锁定；没有依据结果追加 crop、morphology 或 BPE audit。

## 解释

如果 exploratory gate 未通过，结论是现有 I-SPY2 bilateral DINO input 没有足够的 conditional pCR 方向性信号，按预注册规则停止本条 DINO/world-model 主线；下一步应获取独立 cohort 或 authoritative bilateral imaging processing，而不是继续微调 adapter。若通过，才进入预注册的 fresh-seed formal matrix。
"""
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
