#!/usr/bin/env python3
"""Generate the public Chinese V3 report from machine-readable gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def status(payload) -> str:
    return "NOT_RUN" if payload is None else str(payload.get("status", "PRESENT"))


def main() -> None:
    inheritance = read("inheritance_check.json")
    preflight = read("metrics/preflight.json")
    smoke = read("metrics/smoke.json")
    pilot_execution = read("metrics/pilot_execution.json")
    pilot = read("pilot_gate.json")
    pilot_lock = read("PILOT_LOCK.json")
    matrix = read("metrics/representation_matrix_complete.json")
    evaluation_lock = read("EVALUATION_LOCK.json")
    mechanism = read("mechanism_gate.json")
    mechanism_lock = read("MECHANISM_LOCK.json")
    decision = read("decision.json")
    acceptance = read("acceptance_check.json")
    private_sha = read("manifests/private_artifact_sha_summary.json")
    decision_class = None if decision is None else decision.get("decision_class")

    lines = [
        "# DINOv3 MRI Adapter + Direct Radiomics Grounding V3 报告",
        "",
        "## 当前结论",
        "",
    ]
    if decision_class:
        lines.append(f"当前预注册决策为 `{decision_class}`。")
    else:
        lines.append("实验仍处于 outcome-blind representation 阶段，尚无最终决策。")
    if decision_class == "DIRECT_RAD_WEIGHT_SCREEN_NO_GO":
        lines.extend([
            "",
            "三个 direct-radiomics 权重都未产生满足 paired JEPA safety 的可冻结 checkpoint，因此 downstream candidate mechanism 指标不可评估。",
            "fresh-seed 50-cell 矩阵未启动，pCR outcome 未读取；该结论是当前 fine-tuning objective 的 optimization/safety failure，不是 radiomics target feasibility 或 pCR 模型负结果。",
        ])
    elif decision_class == "GROUNDING_OPTIMIZATION_CONFLICT":
        lines.extend(["", "正式 paired JEPA/优化安全失败，pCR 保持锁定。"])
    elif decision_class == "RADIOMICS_NOT_TRANSFERRED":
        lines.extend(["", "正式 fresh-seed matched-probe mechanism gate 未通过，pCR 保持锁定。"])
    elif decision_class == "REPRESENTATION_ONLY":
        lines.extend(["", "Radiomics 表型迁移通过，但未形成预注册的 conditional pCR 增量。"])

    stages = [
        ("V2 inheritance", inheritance), ("preflight", preflight),
        ("one-batch gradient smoke", smoke), ("pilot orchestration (6 trained; 9 fail-fast skips)", pilot_execution),
        ("pilot mechanism gate", pilot), ("pilot weight lock", pilot_lock),
        ("50-cell fresh-seed matrix", matrix), ("evaluation lock", evaluation_lock),
        ("formal mechanism gate", mechanism), ("mechanism lock", mechanism_lock),
        ("pCR evaluation / acceptance", acceptance), ("private SHA summary", private_sha),
    ]
    lines.extend(["", "## 阶段状态", "", "| 阶段 | 状态 |", "|---|---|"])
    for name, payload in stages:
        lines.append(f"| {name} | `{status(payload)}` |")

    if pilot:
        lines.extend(["", "## Pilot outcome-blind mechanism gate", ""])
        lines.append("| Arm | λ | Direct head ρ | Matched probe ρ | 相对 C0 gain | Static FTV Δρ | ΔFTV Δρ | 结果 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for arm in ("R025", "R050", "R100"):
            values = pilot.get("candidate_metrics", {}).get(arm, {})
            gates = pilot.get("candidate_gates", {}).get(arm, {})
            def value(name):
                observed = values.get(name)
                return "—" if observed is None else f"{float(observed):.4f}"
            lines.append(
                f"| {arm} | {float(values.get('radiomics_weight', {'R025':.25,'R050':.5,'R100':1}[arm])):.2f} | "
                f"{value('direct_head_radiomics_macro_spearman')} | {value('matched_probe_radiomics_macro_spearman')} | "
                f"{value('matched_probe_gain_vs_c0')} | {value('static_ftv_change_vs_c0')} | "
                f"{value('delta_ftv_change_vs_c0')} | {'PASS' if gates and all(gates.values()) else 'FAIL'} |"
            )
        baseline = pilot.get("baseline_metrics", {})
        if "matched_probe_radiomics_macro_spearman" in baseline:
            lines.extend(["", f"固定 V2 C0 matched-probe radiomics macro Spearman 为 {baseline['matched_probe_radiomics_macro_spearman']:.4f}。"])
        lines.extend(["", "Gate 解释："])
        for arm, gates in pilot.get("candidate_gates", {}).items():
            evaluation_status = pilot.get("candidate_evaluation_status", {}).get(arm)
            if evaluation_status == "NOT_EVALUATED_NO_FEASIBLE_CHECKPOINT":
                lines.append(f"- {arm}: checkpoint safety FAIL；direct head、matched probe 与 FTV retention 均未评估。")
            else:
                failed = [name for name, passed in gates.items() if not passed]
                lines.append(f"- {arm}: {', '.join(failed) if failed else 'none'}")
        safety_rows = []
        for arm, values in pilot.get("candidate_metrics", {}).items():
            for item in values.get("training_safety", {}).get("failed_checkpoints", []):
                safety_rows.append((arm, item))
        if safety_rows:
            lines.extend([
                "", "完整运行 cells 的 paired JEPA safety：", "",
                "| Arm | Fold | C0 JEPA | 允许上限 | 最低 observed JEPA | 超上限 |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for arm, item in safety_rows:
                lines.append(
                    f"| {arm} | {item['fold']} | {item['paired_c0_jepa_loss']:.5f} | "
                    f"{item['maximum_allowed_jepa_loss']:.5f} | {item['minimum_observed_jepa_loss']:.5f} | "
                    f"{item['excess_over_allowed_fraction']:.1%} |"
                )

    if mechanism:
        gains = list(mechanism.get("matched_probe_gain_by_seed", {}).values())
        direct = list(mechanism.get("direct_head_by_seed", {}).values())
        matched = list(mechanism.get("matched_probe_by_seed", {}).values())
        lines.extend(["", "## Fresh-seed mechanism", ""])
        if gains:
            lines.append(
                f"RAD direct-head ρ={np.mean(direct):.4f}，matched-probe ρ={np.mean(matched):.4f}，"
                f"相对 C0 gain={np.mean(gains):.4f}，正向 seeds={sum(x > 0 for x in gains)}/5。"
            )
        for name, passed in mechanism.get("gates", {}).items():
            lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")

    if decision and decision.get("status") == "COMPLETE":
        lines.extend([
            "", "## Conditional pCR", "",
            f"RAD early macro ΔAUROC={decision['candidate_early_macro_delta_auroc']:.4f}；"
            f"RAD−C0={decision['candidate_minus_c0_early_macro_delta_auroc']:.4f}。",
            f"RAD early macro ΔAUPRC={decision['candidate_early_macro_delta_auprc']:.4f}；"
            f"Brier improvement={decision['candidate_early_macro_brier_improvement']:.4f}。",
        ])

    lines.extend(["", "## 研究解释与下一步", ""])
    if decision_class == "DIRECT_RAD_WEIGHT_SCREEN_NO_GO":
        lines.extend([
            "- 不扩展到 50-cell confirmatory matrix，也不读取 pCR；这正是快速 Pilot gate 的资源保护作用。",
            "- C0 matched linear probe 已达到 0.2861，说明 residual-PC 信息在未 grounding 的 state 中已经可解码；先做 fold/PC/visit-wise audit，确认该结果不是 pooled-fold 口径造成的假象。",
            "- 在独立协议中先训练 frozen-C0 的 head-only calibrator，区分 head/target learnability 与 representation update；它只能作机制校准，不能宣称产生了新 representation。",
            "- 若 head-only 可行，再预注册 JEPA-preserving 更新（head warm-up、adapter 极低 LR、C0 state anchoring 或 gradient surgery），仍以 matched-probe gain 和 JEPA safety 选型，不得用未读取的 pCR 调参。",
        ])
    elif decision_class in {"GROUNDING_OPTIMIZATION_CONFLICT", "RADIOMICS_NOT_TRANSFERRED"}:
        lines.extend([
            "- 保持 pCR 锁定，先做 PC-wise、visit-wise transfer 和 gradient/objective conflict 审计。",
            "- 新假设应独立预注册，不因 outcome 调整 λ、fresh seeds 或 backbone 冻结策略。",
        ])
    elif decision_class == "RAD_GROUNDED_CONDITIONAL_PCR_SUPPORTED":
        lines.append("- 冻结模型并在独立 cohort 做同合同外部验证；内部 OOF 不足以支持临床声明。")
    else:
        lines.append("- 按硬门控继续；只有 EVALUATION_LOCK 与 MECHANISM_LOCK 同时有效才允许首次读取 pCR。")

    lines.extend([
        "", "## 边界", "",
        "- DINOv3 backbone 永久冻结；模型 forward 仅接收冻结 DINO summaries。",
        "- FTV objective 权重为零，FTV 仅用于 retention diagnostic 与最终 clinical+FTV baseline。",
        "- V2 DINO cache、五折 PCA16 targets、cohort 和 folds 均 hash-bound 复用，V2 结果不被修改。",
        "- 公开报告不包含患者标识、预测、private path 或 radiomics targets。",
    ])
    report = ROOT / "reports/final_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "WRITTEN", "report": str(report)})


if __name__ == "__main__":
    main()
