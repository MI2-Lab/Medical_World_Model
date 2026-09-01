#!/usr/bin/env python3
"""Generate V2 machine-readable status and Chinese report without private data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def stopped_decision(roi, target):
    if roi and roi.get("status") == "NO_GO":
        return {
            "schema_version": 1,
            "status": "TERMINATED_AT_V2_ROI_FEASIBILITY",
            "decision_class": "V2_ROI_FEASIBILITY_NO_GO",
            "failed_gates": [name for name, passed in roi["gates"].items() if not passed],
            "pcr_evaluation_started": False,
            "pcr_outcomes_read": False,
            "interpretation": "V2 target feasibility failure; no DINO adapter efficacy inference",
        }
    if target and target.get("status") == "NO_GO":
        return {
            "schema_version": 1,
            "status": "TERMINATED_AT_V2_TARGET_FEASIBILITY",
            "decision_class": "V2_RADIOMICS_TARGET_NO_GO",
            "completed_folds": int(target.get("completed_folds", 0)),
            "failure_reason": target.get("failure_reason"),
            "pcr_evaluation_started": False,
            "pcr_outcomes_read": False,
            "interpretation": "stable early residual-radiomics PCA target could not be constructed",
        }
    return None


def main() -> None:
    preflight = read(ROOT / "metrics/preflight.json")
    roi = read(ROOT / "metrics/roi_feasibility.json")
    raw = read(ROOT / "manifests/radiomics_raw_complete.json")
    target = read(ROOT / "target_feasibility.json")
    dino = read(ROOT / "manifests/dinov3_cache_complete.json")
    private_sha = read(ROOT / "manifests/private_artifact_sha_summary.json")
    smoke = read(ROOT / "metrics/smoke_cell.json")
    matrix = read(ROOT / "metrics/representation_matrix_complete.json")
    if matrix is None:
        matrix = read(ROOT / "metrics/representation_matrix_gate.json")
    representation_lock = read(ROOT / "EVALUATION_LOCK.json")
    mechanism = read(ROOT / "mechanism_gate.json")
    mechanism_lock = read(ROOT / "MECHANISM_LOCK.json")
    decision = read(ROOT / "decision.json")
    acceptance = read(ROOT / "acceptance_check.json")
    if decision is None:
        decision = stopped_decision(roi, target)
        if decision is not None:
            atomic_json(ROOT / "decision.json", decision)
            acceptance = {
                "schema_version": 1,
                "status": "FAIL",
                "failed_stage": decision["status"],
                "decision_class": decision["decision_class"],
                "representation_cells_completed": 0,
                "mechanism_lock_created": False,
                "pcr_outcomes_read": False,
            }
            atomic_json(ROOT / "acceptance_check.json", acceptance)

    pcr_stage = acceptance
    if decision and decision.get("pcr_evaluation_started") is False:
        pcr_stage = {"status": "NOT_RUN"}

    stages = (
        ("实现与 preflight", preflight),
        ("V2 ROI feasibility", roi),
        ("PyRadiomics extraction", raw),
        ("五折 target feasibility", target),
        ("DINOv3 frozen cache", dino),
        ("private SHA manifest", private_sha),
        ("paired smoke cell", smoke),
        ("75-cell representation matrix", matrix),
        ("representation lock", representation_lock),
        ("outcome-blind mechanism gate", mechanism),
        ("mechanism lock", mechanism_lock),
        ("pCR evaluation", pcr_stage),
    )
    lines = [
        "# DINOv3 MRI Adapter + Direct Radiomics Grounding V2 报告",
        "",
        "## 当前结论",
        "",
    ]
    if decision:
        interpretation = decision.get("interpretation", "结论仅适用于冻结的内部 OOF 协议。")
        if decision.get("decision_class") == "GROUNDING_OPTIMIZATION_CONFLICT":
            interpretation = "完整训练预算后 paired checkpoint safety 仍失败；这不是 target feasibility failure。"
        lines.extend(
            [
                f"当前决策为 `{decision['decision_class']}`。",
                "",
                interpretation,
            ]
        )
    elif mechanism_lock:
        lines.append("Mechanism gate 已通过并完成双锁；pCR evaluation 尚未完成。")
    else:
        lines.append("V2 正在 outcome-blind 阶段，尚无 pCR 结论。")
    lines.extend(["", "## 阶段状态", "", "| 阶段 | 状态 |", "|---|---|"])
    for name, payload in stages:
        status = "NOT_RUN" if payload is None else str(payload.get("status", "PRESENT"))
        lines.append(f"| {name} | `{status}` |")

    if roi:
        lines.extend(
            [
                "",
                "## ROI 与 erosion feasibility",
                "",
                "| Gate | Coverage | 阈值 | 结果 |",
                "|---|---:|---:|---|",
            ]
        )
        for visit in ("T0", "T1", "T2", "T3", "overall"):
            lines.append(
                f"| {visit} | {roi['coverage'][visit]:.1%} | {roi['thresholds'][visit]:.1%} | "
                f"{'PASS' if roi['coverage_gates'][visit] else 'FAIL'} |"
            )
        lines.extend(
            [
                "",
                "Erosion 只用于可提取子集的 symmetric stability audit；原始 target ROI 仍保持 64 voxels/3 slices。",
            ]
        )
    if target:
        lines.extend(
            [
                "",
                "## Early radiomics target",
                "",
                f"五折 target gate：`{target['status']}`；T0–T2 用于 residualizer/PCA16/grounding，T3 radiomics mask 永久为 false。",
            ]
        )
        if target.get("status") == "PASS":
            lines.append(
                f"每折保留 feature 数范围为 {target['minimum_selected_features']}–{target['maximum_selected_features']}。"
            )
        elif target.get("failure_reason"):
            lines.append(f"停止原因：`{target['failure_reason']}`。")
    if matrix and matrix.get("status") == "NO_GO":
        failure = matrix.get("failure", {})
        observed = failure.get("minimum_observed_validation_jepa_loss")
        maximum = failure.get("maximum_allowed_validation_jepa_loss")
        lines.extend(
            [
                "",
                "## Representation checkpoint safety",
                "",
                f"失败 cell：`{matrix.get('failed_cell')}`；停止前完成 "
                f"{matrix.get('completed_cells_before_stop', 0)}/75 个完整 state cells。",
            ]
        )
        if observed is not None and maximum is not None:
            lines.append(
                f"D2 在完整 {failure.get('epochs_completed')} epochs 内的最低 validation JEPA loss "
                f"为 {observed:.5f}，高于 paired D1 允许上限 {maximum:.5f} "
                f"（超出 {(observed / maximum - 1):.1%}）。"
            )
        lines.append("State non-collapse 未失败；停止原因是 paired objective safety constraint。")
    if mechanism:
        lines.extend(["", "## Outcome-blind mechanism gate", ""])
        d3_values = list(mechanism.get("d3_radiomics_spearman_by_seed", {}).values())
        gain_values = list(mechanism.get("d3_minus_d2_radiomics_spearman_by_seed", {}).values())
        if d3_values and gain_values:
            lines.extend(
                [
                    f"D3 pooled-OOF residual-PC macro Spearman：{np.mean(d3_values):.4f}。",
                    "",
                    f"D3−D2 pooled-OOF macro Spearman：{np.mean(gain_values):.4f}；"
                    f"正向 seeds：{sum(value > 0 for value in gain_values)}/5。",
                    "",
                ]
            )
        for name, passed in mechanism["gates"].items():
            lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")

    if decision and decision.get("status") == "COMPLETE":
        lines.extend(
            [
                "",
                "## pCR conditional fusion",
                "",
                f"D3 early macro ΔAUROC：{decision['d3_early_macro_delta_auroc']:.4f}。",
                f"D3−D2 early macro ΔAUROC：{decision['d3_minus_d2_early_macro_delta_auroc']:.4f}。",
                f"D3 early macro ΔAUPRC：{decision['d3_early_macro_delta_auprc']:.4f}；"
                f"Brier improvement：{decision['d3_early_macro_brier_improvement']:.4f}。",
            ]
        )

    lines.extend(["", "## 下一步", ""])
    decision_class = None if decision is None else decision.get("decision_class")
    if decision_class == "RAD_GROUNDED_CONDITIONAL_PCR_SUPPORTED":
        lines.append("冻结本轮模型与分析，在独立 cohort 做同合同外部验证；不以内部 OOF 结果作临床声明。")
    elif decision_class == "REPRESENTATION_ONLY":
        lines.append("保留 D3 表型迁移结论，停止本轮 pCR 调参；下一步应检验独立 cohort 的 radiomics transfer 与更匹配的 downstream task。")
    elif decision_class in {"RADIOMICS_NOT_TRANSFERRED", "GROUNDING_OPTIMIZATION_CONFLICT"}:
        lines.append("pCR 保持锁定。先诊断 PC-wise/visit-wise transfer 与 objective 冲突，不根据未读取的 pCR 解冻 DINO backbone。")
    elif decision_class:
        lines.append("本轮不支持既定 headline claim；保留冻结结果，新增假设必须在独立 protocol 中预注册。")
    else:
        lines.append("完成当前 outcome-blind gate；只有双锁均通过才进入 pCR evaluation。")

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- V2 不修改 DINOv3 backbone、C1B-H geometry、cohort 或 folds。",
            "- Representation 与 mechanism 阶段不读取 pCR 或 clinical outcome。",
            "- 只有 representation lock 与 mechanism lock 同时有效，pCR evaluator 才能打开 outcome manifest。",
            "- 即使最终通过，也只是内部 OOF 证据，仍需独立 cohort 验证。",
        ]
    )
    report = ROOT / "reports/final_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "WRITTEN", "report": str(report)})


if __name__ == "__main__":
    main()
