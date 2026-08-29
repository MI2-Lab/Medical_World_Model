#!/usr/bin/env python3
"""Generate the aggregate-only Chinese Goal C report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def value_or_pending(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "尚未确定"
    return f"{float(value):.{digits}f}"


def metric_sentence(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "尚未确定（正式矩阵未完成）"
    best = metrics.loc[metrics["auroc_mean"].idxmax()]
    return f"{best['arm']} / {best['timing']}，AUROC={value_or_pending(best['auroc_mean'])}（fold/seed 均值）"


def gate_status(decision: dict, name: str) -> str:
    return str(decision.get("gates", {}).get(name, {}).get("status", "NOT_RUN"))


def render_table(frame: pd.DataFrame, columns: list[str], *, sort_by: str | None = None) -> str:
    if frame.empty:
        return "尚无正式指标。"
    show = frame.copy()
    if sort_by and sort_by in show.columns:
        show = show.sort_values(sort_by, ascending=False)
    return show[columns].to_markdown(index=False, floatfmt=".4f")


def fusion_comparison_table(frame: pd.DataFrame, population: str, base_model: str, augmented_model: str) -> pd.DataFrame:
    subset = frame.loc[
        frame["population"].eq(population) & frame["model"].isin({base_model, augmented_model}),
        ["arm", "timing", "model", "auroc_mean", "auroc_std", "n_cells"],
    ]
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.pivot_table(index=["arm", "timing"], columns="model", values=["auroc_mean", "auroc_std", "n_cells"], aggfunc="first")
    pivot.columns = ["_".join(str(part) for part in column if part) for column in pivot.columns]
    pivot = pivot.reset_index()
    base_mean = f"auroc_mean_{base_model}"
    augmented_mean = f"auroc_mean_{augmented_model}"
    if base_mean not in pivot or augmented_mean not in pivot:
        return pd.DataFrame()
    pivot["delta_auroc"] = pivot[augmented_mean] - pivot[base_mean]
    return pivot.rename(columns={base_mean: "base_auroc", augmented_mean: "augmented_auroc"})[["arm", "timing", "base_auroc", "augmented_auroc", "delta_auroc"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "final_report.md")
    args = parser.parse_args()
    decision_path = args.metrics_dir / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {"status": "NOT_RUN", "decision_class": "NOT_RUN", "gates": {}}
    provenance_path = args.metrics_dir.parent / "reports" / "delivery_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    metrics = read_csv(args.metrics_dir / "mri_only_metrics.csv")
    gaps = read_csv(args.metrics_dir / "generalization_gap.csv")
    bootstrap = read_csv(args.metrics_dir / "paired_bootstrap.csv")
    fusion = read_csv(args.metrics_dir / "fusion_fold_metrics.csv")
    clinical_metrics = read_csv(args.metrics_dir / "clinical_complementarity.csv")
    ftv_metrics = read_csv(args.metrics_dir / "beyond_ftv.csv")
    status = str(decision.get("status", "NOT_RUN"))
    best_sentence = metric_sentence(metrics)
    if not gaps.empty:
        gap_best = gaps.loc[gaps["train_minus_oof_auroc"].idxmax()]
        gap_sentence = f"最大 train−OOF AUROC gap={value_or_pending(gap_best['train_minus_oof_auroc'])}（{gap_best['arm']}/{gap_best['timing']}）"
    else:
        gap_sentence = "尚未确定"
    def delta_text(reference: str, candidate: str) -> str:
        rows = bootstrap.loc[bootstrap["reference_arm"].eq(reference) & bootstrap["candidate_arm"].eq(candidate)] if not bootstrap.empty else pd.DataFrame()
        if rows.empty:
            return "尚未确定"
        best = rows.loc[rows["delta_auroc"].idxmax()]
        return f"最大 paired ΔAUROC={value_or_pending(best['delta_auroc'])}（{best['candidate_arm']} vs {best['reference_arm']}, {best['timing']}, seed {best['seed']}）"
    def fusion_delta(population: str, base: str, augmented: str) -> str:
        if fusion.empty:
            return "尚未确定"
        subset = fusion.loc[fusion["population"].eq(population)]
        rows = []
        for (arm, timing), group in subset.groupby(["arm", "timing"]):
            b = group.loc[group["model"].eq(base)].groupby("seed")["auroc"].mean()
            a = group.loc[group["model"].eq(augmented)].groupby("seed")["auroc"].mean()
            common = sorted(set(a.index) & set(b.index))
            if common:
                rows.append({"arm": arm, "timing": timing, "delta": float((a.loc[common] - b.loc[common]).mean())})
        if not rows:
            return "尚未确定"
        row = max(rows, key=lambda item: item["delta"])
        return f"最大两 seed 均值 ΔAUROC={value_or_pending(row['delta'])}（{row['arm']}/{row['timing']}）"
    gate_a = gate_status(decision, "A_spatial_vs_C0")
    gate_b = gate_status(decision, "B_raw_vs_C0")
    gate_c = gate_status(decision, "C_full_context_vs_LOCAL_patch")
    gate_d = gate_status(decision, "D_clinical_increment")
    gate_e = gate_status(decision, "E_ftv_increment")
    if status == "COMPLETE":
        q1 = f"正式结果为 {gate_a}；{delta_text('C0', 'C3')}。"
        q2 = f"C2 相对 C0：{delta_text('C0', 'C2')}。"
        q3 = f"C3 相对 C2：{delta_text('C2', 'C3')}。"
        q4 = f"C4 相对 C3：{delta_text('C3', 'C4')}；Gate C={gate_c}。"
        q5 = f"C5 相对 C0：{delta_text('C0', 'C5')}；Gate B={gate_b}。"
        q6 = best_sentence
        q7 = gap_sentence
        q8 = f"{fusion_delta('full_808', 'C', 'C_PLUS_M')}；Gate D={gate_d}。"
        q9 = f"{fusion_delta('ftv_complete_375', 'C_PLUS_F', 'C_PLUS_F_PLUS_M')}；Gate E={gate_e}。"
        q10 = f"预注册决策类：`{decision.get('decision_class', 'UNKNOWN')}`。"
        q11 = "继续 Patch-Token World Model 的支持度由 Gate A/C 决定：" + ("有支持，但仍需外部/因果验证。" if gate_a == "PASS" else "当前 Gate A 未支持把 attention 图解释为继续理由。")
        q12 = "Gate C 通过则倾向 broader context；否则保留 LOCAL-only：" + ("broad context。" if gate_c == "PASS" else "LOCAL-only。")
    else:
        q1 = q2 = q3 = q4 = q5 = q6 = q7 = q8 = q9 = q10 = q11 = q12 = "尚未确定；正式 C1–C5 矩阵尚未完成。"
    metric_table = "尚无正式指标。"
    if not metrics.empty:
        metric_table = render_table(metrics, ["arm", "timing", "auroc_mean", "auprc_mean", "brier_mean", "ece10_mean"], sort_by="auroc_mean")
    if not gaps.empty:
        gap_summary = gaps.groupby(["arm", "timing"], as_index=False).agg(
            train_auroc=("train_auroc", "mean"),
            validation_auroc=("validation_auroc", "mean"),
            oof_auroc=("oof_auroc", "mean"),
            train_minus_oof_auroc=("train_minus_oof_auroc", "mean"),
            validation_minus_oof_auroc=("validation_minus_oof_auroc", "mean"),
            n_cells=("seed", "size"),
        )
        gap_table = render_table(gap_summary, ["arm", "timing", "train_auroc", "validation_auroc", "oof_auroc", "train_minus_oof_auroc", "validation_minus_oof_auroc", "n_cells"])
    else:
        gap_table = "尚无正式指标。"
    clinical_table_frame = fusion_comparison_table(clinical_metrics, "full_808", "C", "C_PLUS_M")
    clinical_table = render_table(clinical_table_frame, ["arm", "timing", "base_auroc", "augmented_auroc", "delta_auroc"])
    ftv_table_frame = fusion_comparison_table(ftv_metrics, "ftv_complete_375", "C_PLUS_F", "C_PLUS_F_PLUS_M")
    ftv_table = render_table(ftv_table_frame, ["arm", "timing", "base_auroc", "augmented_auroc", "delta_auroc"])
    attention = pd.DataFrame()
    attention_path = args.metrics_dir / "attention_diagnostics.csv"
    if attention_path.exists():
        attention = read_csv(attention_path).dropna(subset=["attention_entropy"])
    attention_summary = attention.groupby(["arm", "timing"], as_index=False).agg(
        attention_entropy=("attention_entropy", "mean"),
        attention_concentration=("attention_concentration", "mean"),
        attention_concentration_top10=("attention_concentration_top10", "mean"),
        center_mass=("center_mass", "mean"),
        outer_mass=("outer_mass", "mean"),
        longitudinal_embedding_cosine=("longitudinal_embedding_cosine", "mean"),
        n_cells=("seed", "size"),
    ) if not attention.empty else pd.DataFrame()
    attention_table = render_table(attention_summary, ["arm", "timing", "attention_entropy", "attention_concentration", "attention_concentration_top10", "center_mass", "outer_mass", "longitudinal_embedding_cosine", "n_cells"])
    primary_bootstrap = bootstrap.loc[bootstrap["timing"].isin({"T0", "T0_T1", "T0_T2"})].copy() if not bootstrap.empty else pd.DataFrame()
    bootstrap_table = render_table(primary_bootstrap, ["seed", "timing", "reference_arm", "candidate_arm", "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high", "delta_auprc", "delta_brier"])
    seed_consistency = read_csv(args.metrics_dir / "seed_consistency.csv")
    seed_table = render_table(seed_consistency.loc[seed_consistency["timing"].isin({"T0", "T0_T1", "T0_T2"})] if not seed_consistency.empty else seed_consistency, ["timing", "reference_arm", "candidate_arm", "delta_auroc_mean", "delta_auroc_std", "seeds"])
    local_context = read_csv(args.metrics_dir / "local_vs_full_context.csv")
    local_context_table = render_table(local_context, ["seed", "timing", "reference_arm", "candidate_arm", "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high", "delta_auprc", "delta_brier"])
    gate_table = "\n".join(f"- {name}: `{gate_status(decision, name)}`" for name in ["A_spatial_vs_C0", "B_raw_vs_C0", "C_full_context_vs_LOCAL_patch", "D_clinical_increment", "E_ftv_increment", "overfit_generalization_audit"])
    text = f"""# Goal C：Raw-Image / Spatial pCR Ceiling Audit

## 结论边界

本报告只估计当前 C1B-H DCE7 输入合同下的**经验性有监督上限**，不是信息论上限、pCR-free World Model、治疗因果证据、外部验证或生产模型。MRI 分支没有临床变量、治疗匹配、contrastive/triplet loss、BPE proxy、pCR geometry 或 lesion mask；C2/C3 只使用 outcome-free 固定中心 64-mm LOCAL feature-cell support。

## 运行状态

- 分支：`{provenance.get('branch', '尚未记录')}`
- parent SHA：`{provenance.get('parent_commit', '尚未记录')}`
- experiment commit SHA：`{provenance.get('experiment_commit', '尚未记录')}`
- push status：`{provenance.get('push_status', '尚未记录')}`
- 正式状态：`{status}`
- 决策类：`{decision.get('decision_class', 'NOT_RUN')}`
- 人群：`full_808`（MRI-only/clinical）与 `ftv_complete_375`（FTV 增量）
- 外层：5 folds；seeds=2026/3026；主 timing=T0、T0_T1、T0_T2；T0_T3 为 late/pre-surgery supplementary
- 补充说明：C2/C3/C4 导出描述性 attention diagnostics；C5 保持相同小型空间 readout 的预测路径，但不保留原始特征图 attention tensor，以避免测试期显存物化，且不影响其预测结果。
- 机器判定：[`../metrics/decision.json`](../metrics/decision.json)

## Gate 结果

{gate_table}

## 主要 MRI-only 表（aggregate-only）

{metric_table}

## Train/validation/OOF 泛化 gap 表

{gap_table}

## Clinical complementarity 表（full_808）

{clinical_table}

## Beyond-FTV 表（ftv_complete_375）

{ftv_table}

## Attention concentration diagnostics（描述性）

{attention_table}

## LOCAL vs full-context 比较

{local_context_table}

## Paired patient bootstrap（primary timings，5000 draws）

{bootstrap_table}

## Seed consistency

{seed_table}

患者级预测、原始 MRI、pCR 标签、临床表、特征和 checkpoint 均保持在私有 gitignored 路径；公开 CSV 只包含 aggregate/fold-level metrics。

## 必须回答的 12 个问题

1. 空间监督学习是否超过 C0 pooled ceiling？{q1}
2. LOCAL attention 是否优于 LOCAL mean/pool reference？{q2}
3. Patch-token Transformer 是否优于 attention pooling？{q3}
4. Full C1B context 是否超过 64-mm LOCAL？{q4}
5. Direct raw-image supervised training 是否超过 representation training？{q5}
6. 最佳 MRI-only AUROC 多高？{q6}
7. train–OOF 泛化差距多大？{q7}
8. MRI 是否增加 clinical？{q8}
9. MRI 是否增加 clinical+FTV？{q9}
10. 主要瓶颈是 pooling、representation、input 还是 generalization？{q10}
11. 是否支持继续 Patch-Token World Model？{q11}
12. 后续应 LOCAL-only 还是加入 broader context？{q12}

## 产物索引

- `metrics/mri_only_metrics.csv`：fold/seed test AUROC/AUPRC/Brier/calibration slope/ECE 的 aggregate
- `metrics/generalization_gap.csv`：train/validation/test(O0F) gap
- `metrics/clinical_complementarity.csv` 与 `metrics/beyond_ftv.csv`：fold-safe fusion
- `metrics/attention_diagnostics.csv`：entropy/concentration/center-vs-outer/longitudinal descriptive diagnostics
- `metrics/paired_bootstrap.csv`：within-fold paired patient bootstrap，5000 draws
- `metrics/seed_consistency.csv` 与 `metrics/local_vs_full_context.csv`
- `figures/`：只由 aggregate metrics 生成的静态图

## 前置证据

此前实验已确认 LOCAL3 是稳定 pooled response baseline；conditional pCR supervised ceiling 约 0.548 且未显示稳定的 clinical/FTV 之外增量；spatial/mask-free audits 提示局部表征可存在但未稳定超越既有目标；foundation audits 未证明替换基础编码器。因此本文只把这些作为先验背景，不冒充本矩阵结果。
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
