"""Generate the aggregate-only Chinese Goal-F final report."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd

from .evaluation_contracts import EXPERIMENT_ROOT, load_evaluation_config
from .evaluation_lock import require_before_outcome_access


class ReportDataError(ValueError):
    """Raised when required aggregate evaluation artifacts are incomplete."""


def _read_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ReportDataError(f"{path.name} lacks columns: {sorted(missing)}")
    forbidden = {column for column in frame.columns if column.lower() in {"patient_id", "label_pcr", "probability"}}
    if forbidden:
        raise ReportDataError(f"public aggregate {path.name} contains patient-level columns")
    return frame


def _gate(decision: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = decision.get("gates", {}).get(key)
    if not isinstance(value, Mapping) or not isinstance(value.get("pass"), bool):
        raise ReportDataError(f"decision gate is missing: {key}")
    return value


def _yes(value: bool) -> str:
    return "是" if value else "否"


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _markdown(frame: pd.DataFrame, columns: list[str], labels: Mapping[str, str] | None = None) -> str:
    selected = frame.loc[:, columns].copy()
    if labels:
        selected = selected.rename(columns=dict(labels))
    for column in selected.select_dtypes(include="number").columns:
        selected[column] = selected[column].map(lambda value: _number(value))
    return selected.to_markdown(index=False)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_report(
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    require_before_outcome_access()
    config = load_evaluation_config(require_representation_lock=False)
    metrics = EXPERIMENT_ROOT / "metrics"
    decision_path = metrics / "decision_summary.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    diagnostics = _read_csv(
        metrics / "state_diagnostics.csv",
        ("arm", "seed_base", "fold", "z_P_mean_std", "z_P_effective_rank", "standardized_crosscov_rms"),
    )
    nearest = _read_csv(
        metrics / "nearest_neighbor_stability.csv", ("arm", "state", "timing", "mean_jaccard")
    )
    response = _read_csv(
        metrics / "response_metrics.csv", ("seed_base", "arm", "state", "task", "endpoint", "spearman")
    )
    profile = _read_csv(
        metrics / "phenotype_probes.csv", ("seed_base", "arm", "state", "endpoint", "target", "auroc")
    )
    pcr = _read_csv(
        metrics / "pcr_metrics.csv", ("seed_base", "arm", "population", "timing", "model", "auroc", "auprc", "brier")
    )
    effects = _read_csv(
        metrics / "paired_bootstrap_effects.csv",
        (
            "arm", "seed_base", "population", "timing", "comparison",
            "delta_auroc", "delta_auroc_ci_lower", "delta_auroc_ci_upper", "n_bootstrap",
        ),
    )
    if effects.n_bootstrap.min() < int(config["bootstrap"]["replicates"]):
        raise ReportDataError("paired bootstrap count is below the preregistered value")

    gate_a = _gate(decision, "A_RESPONSE_PRESERVED")
    gate_b = _gate(decision, "B_FACTORIZATION_WORKS")
    gate_c = _gate(decision, "C_CLINICAL_REDUNDANCY_REDUCED")
    gate_d = _gate(decision, "D_PHENOTYPE_COMPLEMENTARITY")

    diag_summary = (
        diagnostics.loc[diagnostics.arm.isin(("F1", "F2"))]
        .groupby("arm", as_index=False)
        .agg(
            zP_mean_std=("z_P_mean_std", "mean"),
            zP_effective_rank=("z_P_effective_rank", "mean"),
            crosscov_rms=("standardized_crosscov_rms", "mean"),
            augmentation_cosine=("augmentation_mean_cosine", "mean"),
            future_mse=("future_phenotype_mse", "mean"),
            future_gain_over_persistence=(
                "future_mse_improvement_over_persistence",
                "mean",
            ),
        )
    )
    response_macro = response.loc[
        response.endpoint.eq("macro")
        & response.state.isin(("F0", "z_R"))
        & response.arm.isin(("F0", "F1", "F2"))
    ].sort_values(["task", "seed_base", "arm"])
    profile_macro = profile.loc[
        profile.endpoint.eq("T0_T1_T2_macro")
        & profile.state.eq("z_P")
        & profile.arm.isin(("F1", "F2"))
        & profile.target.isin(("label_hr", "label_her2", "subtype"))
    ].sort_values(["target", "seed_base", "arm"])
    critical = effects.loc[
        effects.comparison.isin(
            (
                "MRI_full_vs_zR",
                "beyond_C_full_vs_zR",
                "beyond_C_zP_vs_C",
                "beyond_C_F_full_vs_zR",
                "beyond_C_F_zP_vs_C_F",
                "adversarial_F2_vs_F1_full",
            )
        )
        & effects.timing.isin(("T0-T1", "T0-T2"))
    ].sort_values(["comparison", "timing", "arm", "seed_base"])

    f1_profile = profile_macro.loc[profile_macro.arm.eq("F1") & profile_macro.target.isin(("label_hr", "label_her2"))]
    f1_profile_text = ", ".join(
        f"{row.target.replace('label_', '').upper()} seed {int(row.seed_base)} AUROC={_number(row.auroc)}"
        for row in f1_profile.itertuples(index=False)
    )
    nn_macro = nearest.loc[nearest.fold.eq(-1)] if "fold" in nearest else nearest
    nn_text = ", ".join(
        f"{row.arm}={_number(row.mean_jaccard)}"
        for row in nn_macro.itertuples(index=False)
    )
    mri_effects = critical.loc[critical.comparison.eq("MRI_full_vs_zR")]
    clinical_effects = critical.loc[critical.comparison.eq("beyond_C_full_vs_zR")]
    clinical_zp_effects = critical.loc[critical.comparison.eq("beyond_C_zP_vs_C")]
    ftv_effects = critical.loc[critical.comparison.eq("beyond_C_F_full_vs_zR")]
    adversarial = critical.loc[critical.comparison.eq("adversarial_F2_vs_F1_full")]

    def effects_text(frame: pd.DataFrame) -> str:
        return "; ".join(
            f"{row.arm}/seed {int(row.seed_base)}/{row.timing}: ΔAUROC={_number(row.delta_auroc)} "
            f"(95% CI {_number(row.delta_auroc_ci_lower)}, {_number(row.delta_auroc_ci_upper)})"
            for row in frame.itertuples(index=False)
        ) or "无可用 cell"

    classification = decision["classification"]["label"]
    all_gates_pass = all(
        gate["pass"] for gate in (gate_a, gate_b, gate_c, gate_d)
    )
    if all_gates_pass:
        q12 = "在至少一个纵向时点，结果支持 HR/HER2 已知 profile 之外仍存在可由 outcome-free MRI state 捕获的增量信息；仍需更多种子及外部验证。"
    elif gate_d["pass"]:
        q12 = "虽然关键条件增量在两个 seed 中方向一致，但 response/factorization/residualization gate 未全部成立，因此不能把该增量解释为已验证的 HR/HER2-complementary phenotype state。"
    elif gate_c["pass"]:
        q12 = "F2 能降低 HR/HER2 可解码性，但没有证明这种残余 MRI 信息对 pCR 有条件增量；不能把去冗余本身解释为 phenotype 成功。"
    else:
        q12 = "当前目标没有证明稳定的 HR/HER2-complementary MRI 信息；这既可能是目标函数未恢复信号，也可能反映条件 MRI ceiling 较低。"
    final_state = (
        "保留 factorized state，并把 F2 作为候选" if classification.startswith("CLINICAL-RESIDUAL")
        else "保留 F1 factorization、停止 adversarial residualization" if classification.startswith("FACTORIZATION HELPS")
        else "仅把 factorization 作为表征诊断，不宣称 phenotype pCR value" if classification.startswith("CLINICAL RESIDUALIZATION")
        else "回到单一 LOCAL response state；后续优先 patch-token/world-model 目标"
    )

    report = f"""# Goal F：临床残余 phenotype state 最终报告

## 结论

最终分类：**{classification}**。

四个预注册 gate：A={_status(gate_a['pass'])}，B={_status(gate_b['pass'])}，C={_status(gate_c['pass'])}，D={_status(gate_d['pass'])}。本结果是两个训练种子、五个 outer folds 的 internal OOF pilot，不是外部验证。

## 十二个明确问题

1. **response performance 是否保留？** {_yes(gate_a['pass'])}。z_R 相对 F0 的 static degradation floor 为 {config['gates']['response_static_ftv_spearman_degradation_floor']:+.2f}；逐 seed 结果见 response table。ΔFTV 以“同一 arm 两个 seed 均下降”定义 systematic degradation。
2. **z_P 是否 noncollapsed？** F1={_yes(bool(gate_b['F1_noncollapsed']))}，F2={_yes(bool(gate_c['F2_noncollapsed']))}。阈值为 mean per-dimension std ≥ {_number(config['diagnostics']['phenotype_mean_std_floor'], 2)}、effective rank ≥ {_number(config['diagnostics']['effective_rank_floor'], 1)}。
3. **factorization 是否降低 z_R/z_P redundancy？** {_yes(bool(gate_b['crosscov_lower_than_F0_unseparated_control']))}。比较的是每个 seed 内五个 fold outer-test standardized cross-covariance RMS 的非加权均值，并要求两个 seed 均严格低于 F0；F0 control 是未分离 192-D state 的预注册前/后 96 维描述性切分，不赋予其生物学含义。
4. **residualization 前 z_P 能否解码 HR/HER2？** F1 的结果为：{f1_profile_text}。线性关联先在每个 static endpoint 转为 flip-invariant `0.5+|AUROC−0.5|`，再平均 T0/T1/T2；AUROC 在 0.5 任一侧偏离都可能表示可测 profile correlate，但不等同于因果 phenotype。
5. **F2 是否减少 HR/HER2 redundancy？** {_yes(gate_c['pass'])}。两个 seed 的 mean endpoint flip-invariant decodability 均下降的 targets：{', '.join(gate_c['targets_decreased_in_both_seeds']) or '无'}；此定义既不会把 0.45→0.40 错判为去冗余，也不会让 0.4/0.6 endpoint 在 raw macro 中互相抵消，并且同时要求 F2 不 collapse。
6. **z_P 是否保留纵向 image information？** F1={_yes(bool(gate_b['image_information_retained']))}，F2={_yes(bool(gate_c['F2_image_information_retained']))}。要求 frozen weak-view cosine 达标、future predictor 在每 fold 的 T1→T2/T2→T3 上优于同一 EMA-target-projector 空间的 persistence baseline、且 z_P noncollapsed；T0→T1 因 export 未含 EMA-target T0 context 而不进入 baseline 比较。nearest-neighbor 只在同一 fold coordinate system 内跨 seed 比较，T0–T2 aggregate Jaccard 为 {nn_text}。
7. **z_P 是否改善 MRI-only pCR？** 以 `[z_R,z_P] − z_R` 衡量：{effects_text(mri_effects)}。
8. **z_P 是否增加 clinical 之外的信息？** `C+z_R+z_P − C+z_R` 为：{effects_text(clinical_effects)}。直接的 `C+z_P − C` 为：{effects_text(clinical_zp_effects)}。
9. **z_P 是否增加 clinical+FTV 之外的信息？** Gate D={_status(gate_d['pass'])}；关键 `C+F+z_R+z_P − C+F+z_R` 为：{effects_text(ftv_effects)}。positive-both-seed timings={gate_d['positive_both_seed_timings'] or '无'}；strong mean ≥ +0.03 timings={gate_d['strong_mean_ge_0_03_timings'] or '无'}。
10. **adversarial clinical residualization 有帮助还是伤害？** F2 相对 F1 的 full clinical+FTV model：{effects_text(adversarial)}。必须与 Gate C 的去冗余结果一起解读；降低 HR/HER2 AUROC 单独不算成功。
11. **最终 state 应是 single-state 还是 factorized？** {final_state}。
12. **这对 HR/HER2-complementary MRI information 意味着什么？** {q12}

## State diagnostics（outer-test、outcome-free aggregate）

{_markdown(diag_summary, ['arm', 'zP_mean_std', 'zP_effective_rank', 'crosscov_rms', 'augmentation_cosine', 'future_mse', 'future_gain_over_persistence'], {'arm': 'Arm'})}

逐维 variance/std 与完整 covariance eigenspectrum 分别规范化存于 `metrics/state_dimension_diagnostics.csv` 与 `metrics/state_covariance_eigenspectra.csv`；CCA 为 ridge-regularized、supplied rows 内描述统计。没有用 t-SNE/UMAP 作主要证据。

## Response metrics

{_markdown(response_macro, ['seed_base', 'arm', 'state', 'task', 'spearman', 'rmse', 'r2'])}

Static probe 严格复现 confirmed LOCAL3 contract：outer-train winsor/median-IQR FTV transform、lsqr Ridge、自然尺度 inverse；ΔFTV 使用 literal `FTV(t+1)-FTV(t)`、state difference 与 outer-train target standardization。Ridge alpha 仅由 outer-validation analysis-space MSE 选择。

## Phenotype profile probes

{_markdown(profile_macro, ['seed_base', 'arm', 'state', 'target', 'auroc', 'flip_invariant_decodability'])}

HR/HER2 使用 balanced linear logistic probe；subtype 使用 multiclass macro one-vs-rest AUROC。所有 C/alpha 只在 outer validation 选择。

## pCR complementarity 与 paired bootstrap

{_markdown(critical, ['arm', 'seed_base', 'population', 'timing', 'comparison', 'delta_auroc', 'delta_auroc_ci_lower', 'delta_auroc_ci_upper', 'n_bootstrap'])}

每个比较使用 {int(config['bootstrap']['replicates']):,} 次 paired patient bootstrap，并在 outer-fold × pCR outcome 精确 strata 内重采样；AUROC/AUPRC 为 augmented−baseline，Brier improvement 为 baseline−augmented。patient-level OOF probabilities 与 bootstrap draws 仅写入 gitignored `predictions/`。

## 模型与 cohort contract

- F0：既有 LOCAL3 192-D response state；F1/F2：96-D z_R + 96-D z_P，总维数固定 192。
- Confirmed LOCAL3 的 z_R transition 保持 canonical image-only；新增 z_P future predictor 保留 causal treatment/HR/HER2/MP condition。phenotype query 本身不接收 clinical/treatment，treatment 也不作 adversarial removal。
- Representation training 明确禁止读取 pCR；本阶段只接受完整且 hash/provenance 绑定的 frozen exports 后才加载 pCR。
- Clinical C：HR、HER2、MP、screening age、treatment arm；类别词表与缺失值处理只在 outer train 拟合。
- FTV complete cohort 固定 n=375；全 cohort 固定 n=808。时点为 T0、T0–T1、T0–T2。
- F3 在本次 primary run 中关闭；它仅是可选 downstream control，不是 World Model arm，且不得参与主分类。

## Goal B ceiling 的条件解释

本报告不伪造尚未接入的 Goal B 结果。若 supervised conditional ceiling 强而 Goal F 失败，较合理的解释是 MRI signal 存在但 outcome-free residualization objective 未恢复；若 Goal B 也失败，有限 conditional MRI ceiling 更可信；若二者均成功，则 outcome-supervised ceiling 与 outcome-free recovery 形成最强相互支持。

## 局限

1. 两个 seed 只够 primary pilot，不能替代五 seed confirmation。
2. Internal OOF 与 paired bootstrap 不解决 dataset shift、label noise 或 external transportability。
3. F0 前/后 96-D 切分仅是 redundancy control；不能解释为天然 response/phenotype decomposition。
4. Adversarial independence 只针对线性 HR/HER2 adversary，不证明统计独立或去除所有 clinical correlate。
5. Gate D 使用 point ΔAUROC 的方向一致性；CI 同时报告但不是预注册 pass 条件。
"""
    output = Path(output_path).resolve() if output_path else EXPERIMENT_ROOT / "reports" / "final_report.md"
    if output.exists() and not overwrite:
        raise FileExistsError(f"report exists: {output}")
    _atomic_text(report, output)
    return output


__all__ = ["ReportDataError", "generate_report"]
