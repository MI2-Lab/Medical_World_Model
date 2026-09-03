#!/usr/bin/env python3
"""Generate the Chinese status/final report from machine-readable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    preflight = read(ROOT / "metrics/preflight.json")
    cache = read(ROOT / "manifests/dinov3_cache_complete.json")
    roi = read(ROOT / "metrics/radiomics_stage_a_gate.json")
    matrix = read(ROOT / "metrics/representation_matrix_complete.json")
    lock = read(ROOT / "EVALUATION_LOCK.json")
    acceptance = read(ROOT / "acceptance_check.json")
    decision = read(ROOT / "decision.json")
    sensitivity = read(ROOT / "metrics/roi_threshold_sensitivity.json")
    smoke = read(ROOT / "metrics/smoke_validation.json")
    privacy = read(ROOT / "metrics/public_artifact_privacy_gate.json")
    if roi and roi.get("status") == "NO_GO" and decision is None:
        decision = {
            "schema_version": 1,
            "status": "TERMINATED_AT_RADIOMICS_STAGE_A",
            "decision_class": "NO_GO",
            "reason": "preregistered_radiomics_coverage_gate_failed",
            "coverage": roi["coverage"],
            "failed_gates": [name for name, passed in roi["gates"].items() if not passed],
            "representation_matrix_started": False,
            "evaluation_lock_created": False,
            "pcr_evaluation_started": False,
            "pcr_outcomes_read": False,
            "interpretation": "feasibility NO-GO; not evidence that DINOv3 or radiomics grounding lacks value",
        }
        (ROOT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        acceptance = {
            "schema_version": 1,
            "status": "FAIL",
            "failed_stage": "radiomics_stage_a_coverage",
            "coverage_gates": roi["gates"],
            "representation_cells_completed": 0,
            "evaluation_lock_verified": False,
            "pcr_outcomes_read": False,
            "decision_class": "NO_GO",
        }
        (ROOT / "acceptance_check.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if roi and roi.get("status") == "NO_GO":
        acceptance = {
            "schema_version": 1,
            "status": "FAIL",
            "failed_stage": "radiomics_stage_a_coverage",
            "coverage_gates": roi["gates"],
            "representation_cells_completed": 0,
            "evaluation_lock_verified": False,
            "pcr_outcomes_read": False,
            "privacy_gate": None if privacy is None else privacy.get("status"),
            "decision_class": "NO_GO",
        }
        (ROOT / "acceptance_check.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stages = [
        ("实现与 preflight", preflight),
        ("DINOv3 frozen cache", cache),
        ("radiomics Stage-A coverage", roi),
        ("75-cell representation matrix", matrix),
        ("evaluation lock", lock),
        ("pCR evaluation", acceptance),
    ]
    lines = [
        "# DINOv3 MRI Adapter + Direct Radiomics Grounding 报告",
        "",
        "## 结论",
        "",
    ]
    if decision:
        lines.extend(
            [
                f"当前协议决策为 `{decision['decision_class']}`。",
                "",
                ("该结果来自 radiomics Stage-A feasibility gate；representation matrix 和 pCR evaluation 均未启动，因此不能解释为 DINOv3 或 grounding 无效。"
                 if decision.get("pcr_evaluation_started") is False else
                 "该结论仅适用于当前 C1B-H、seed-2026 五折和内部 375 人 primary cohort；独立 cohort 验证前不作临床或外部泛化声明。"),
            ]
        )
    else:
        lines.extend(
            [
                "实验实现已版本化，但尚未形成正式 pCR 结论。评估器保持锁定，直到 DINO/radiomics targets、75 个 representation cells 和 75 份 `[808,4,192]` states 全部完成并冻结。",
                "",
                "因此当前状态是 `IMPLEMENTED_NOT_EVALUATED`，不是 `NO_GO`，也不是 positive finding。",
            ]
        )
    lines.extend(["", "## 阶段状态", "", "| 阶段 | 状态 |", "|---|---|"])
    for name, payload in stages:
        status = "NOT_RUN" if payload is None else str(payload.get("status", "PRESENT"))
        if name == "pCR evaluation" and payload and not payload.get("evaluation_lock_verified", False):
            status = "NOT_RUN_STAGE_A_STOP"
        lines.append(f"| {name} | `{status}` |")
    if roi:
        lines.extend(
            [
                "",
                "## Radiomics Stage-A 实测",
                "",
                "| Gate | 实测 coverage | 阈值 | 结果 |",
                "|---|---:|---:|---|",
            ]
        )
        for name in ("T0", "T1", "T2", "T3", "overall"):
            lines.append(
                f"| {name} | {roi['coverage'][name]:.1%} | {roi['thresholds'][name]:.1%} | "
                f"{'PASS' if roi['gates'][name] else 'FAIL'} |"
            )
        lines.extend(
            [
                "",
                "T2 为唯一失败 gate（87.2% < 90.0%，还差 11 个有效 visits）。按预注册规则必须停止训练；不得查看 pCR 后再放宽 ROI。",
            ]
        )
    if sensitivity:
        current = sensitivity["coverage_grid"]
        lines.extend(
            [
                "",
                "## Outcome-blind threshold sensitivity",
                "",
                f"把 axial slice 下限从 3 改成 2，T2 coverage 仅从 {current['voxels_64_slices_3']['T2']:.1%} 变为 {current['voxels_64_slices_2']['T2']:.1%}，不能解决 gate。",
                f"保持 3 slices 时，voxel 下限必须降至 ≤{sensitivity['maximum_voxel_threshold_passing_with_three_slices']['T2']} 才能达到 90%；32 voxels 时 T2 coverage 为 {current['voxels_32_slices_3']['T2']:.1%}。这可能降低 texture radiomics 的可靠性，只能作为新协议候选，不能补救当前结果。",
            ]
        )
    lines.extend(
        [
            "",
            "## 已实现的关键约束",
            "",
            "- DINOv3 checkpoint revision 和三个 artifact SHA-256 均固定；每个 channel/slice 保存 CLS、patch mean、patch population SD，并排除 4 个 register tokens。",
            "- 模型 forward 仅接受 `[B,4,7,32,2304]` frozen summaries；clinical、pCR、FTV、radiomics、ROI mask 和 geometry 均不在推理接口中。",
            "- PyRadiomics target 使用 Original-only、force2D axial、binWidth 0.25，以及 first-order/GLCM/GLRLM/GLSZM/GLDM/NGTDM；shape、wavelet 和 LoG 不启用。",
            "- radiomics feature filtering、稳定性审计、FTV/局部 mask volume/visit residualization 和 PCA16 均按 outer-train 单独拟合。",
            "- D1–D3 结构相同并共享初始化；D2/D3 checkpoint 选择必须满足相对 paired arm 的 JEPA/FTV 5% 安全约束。",
            "- pCR evaluator 在 `EVALUATION_LOCK.json` 生成前 fail closed；最终 fusion 使用 inner-OOF clinical+FTV logits 和 image logits，只拟合 offset alpha/beta。",
            "",
            "## 下一执行顺序",
            "",
            "1. 不启动 75-cell matrix，也不读取 pCR。先由 PI 决定是否发起一个新的、独立预注册的 ROI feasibility revision。",
            "2. 当前 outcome-blind sensitivity 已表明 slice gate 不是瓶颈。PI 可在新协议中选择：保留 64-voxel 可靠性并把 T2 coverage gate 预注册为 85%，或采用 ≥32 voxels 后重新做完整 morphology stability audit；前者较保守，后者风险是 texture 不稳定。",
            "3. 只有新协议在看 pCR 前锁定且重新通过 coverage/stability gates，才可复用已经实现的 DINO cache、D1–D3 training、evaluation lock 和 fusion 代码。",
        ]
    )
    if smoke:
        index = lines.index("## 下一执行顺序")
        lines[index:index] = [
            "## 真实数据 smoke",
            "",
            f"已验证 DINO cache `{smoke['dinov3_summary_shape']}`/`{smoke['dinov3_summary_dtype']}`，以及 radiomics `{smoke['radiomics_shape']}`（{smoke['radiomics_feature_count']} features）。",
            "",
        ]
    report = ROOT / "reports/final_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "WRITTEN", "report": str(report)})


if __name__ == "__main__":
    main()
