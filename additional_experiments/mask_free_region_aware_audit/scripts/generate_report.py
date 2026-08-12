#!/usr/bin/env python3
"""Generate the Chinese diagnostic report from public aggregate results.

This module intentionally imports only the aggregate-table reader used by the
figure renderer.  It never opens private OOF predictions, clinical labels,
feature arrays, masks, or checkpoints.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_figures import (  # noqa: E402
    FIGURES,
    PRIMARY_REGIONS,
    PUBLIC_TABLES,
    file_sha256,
    load_public_tables,
    numeric_column,
    region_delta_frame,
    resolve_column,
)


REPORT_PATH = Path("reports/final_report.md")
REPORT_MANIFEST_PATH = Path("reports/report_manifest.json")
GATES_PATH = Path("metrics/gates.json")
RUN_SUMMARY_PATH = Path("metrics/run_summary.json")

REQUIRED_REPORT_MARKERS: tuple[str, ...] = (
    *(f"### Q{number} —" for number in range(1, 13)),
    "mask-free-at-readout-only",
    "T0 localization-centered C1B",
    "Oracle denominator representation mismatch",
    "diagnostic/non-biological boundary",
    "T3 late/pre-surgery",
    "Push error：",
)

CLASSIFICATION_LABELS = {
    "A": "DEPLOYABLE_REGION_AWARE_SIGNAL_SUPPORTED",
    "B": "REGION_SIGNAL_EXISTS_BUT_NOT_BEYOND_FTV",
    "C": "ORACLE_REQUIRES_LESION_RELATIVE_LOCALIZATION",
    "D": "MASK_FREE_REGIONALIZATION_NOT_SUPPORTED",
    "INDETERMINATE": "INDETERMINATE_DIAGNOSTIC",
}

GATE_ALIASES = {
    "A": ("A", "gate_a", "MASK_FREE_REGIONAL_SIGNAL_SUPPORTED"),
    "B": ("B", "gate_b", "MASK_FREE_BEYOND_FTV_SUPPORTED"),
    "C": ("C", "gate_c", "ORACLE_SIGNAL_PARTIALLY_RECOVERED"),
    "D": ("D", "gate_d", "PROFILE_ASSOCIATED_REGIONAL_SIGNAL_SUPPORTED"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only final_report.md and report_manifest.json.",
    )
    return parser.parse_args(argv)


def _normal(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = file_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if file_sha256(path) != before:
        raise RuntimeError(f"JSON input changed while being read: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return payload


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = _normal(value)
        if normalized in {"pass", "passed", "true", "supported", "yes", "1"}:
            return True
        if normalized in {"fail", "failed", "false", "unsupported", "no", "0"}:
            return False
    if isinstance(value, Mapping):
        for key in ("passed", "pass", "status", "result", "supported", "value"):
            if key in value:
                parsed = _bool_value(value[key])
                if parsed is not None:
                    return parsed
    return None


def extract_gate_results(payload: Mapping[str, Any]) -> dict[str, bool]:
    """Extract A-D booleans from the documented nested or flat gate schema."""

    containers: list[Mapping[str, Any]] = [payload]
    for key in ("gates", "primary_gates", "gate_results", "decisions"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    results: dict[str, bool] = {}
    for letter, aliases in GATE_ALIASES.items():
        wanted = {_normal(value) for value in aliases}
        matches: list[bool] = []
        for container in containers:
            for key, value in container.items():
                normalized = _normal(key)
                if normalized in wanted or normalized.startswith(f"gate{letter.lower()}"):
                    parsed = _bool_value(value)
                    if parsed is not None:
                        matches.append(parsed)
        if not matches:
            raise ValueError(f"gates.json does not expose a parseable Gate {letter}")
        if len(set(matches)) != 1:
            raise ValueError(f"gates.json contains conflicting Gate {letter} results")
        results[letter] = matches[0]
    return results


def derive_classification(
    gates: Mapping[str, bool],
    tables: Mapping[str, pd.DataFrame],
    *,
    any_two_seed_positive: bool | None = None,
) -> tuple[str, str]:
    """Apply the preregistered classification precedence without retuning."""

    if gates["A"] and gates["C"]:
        return "A", CLASSIFICATION_LABELS["A"]
    if gates["A"] and not gates["B"]:
        return "B", CLASSIFICATION_LABELS["B"]

    has_consistent_positive = bool(any_two_seed_positive)
    try:
        if any_two_seed_positive is not None:
            raise StopIteration
        seed_table = tables["seed_consistency"].copy()
        context = resolve_column(seed_table, ("context", "probe_context"), required=False)
        if context is not None:
            selected = seed_table[context].astype(str).str.upper().eq("MRI_ONLY")
            if selected.any():
                seed_table = seed_table.loc[selected].copy()
        _, first = numeric_column(
            seed_table,
            ("seed_2026_delta_auroc", "delta_seed_2026", "gain_seed_2026"),
            required=False,
        )
        _, second = numeric_column(
            seed_table,
            ("seed_3026_delta_auroc", "delta_seed_3026", "gain_seed_3026"),
            required=False,
        )
        if first is not None and second is not None:
            has_consistent_positive = bool(((first > 0) & (second > 0)).any())
        else:
            deltas = region_delta_frame(seed_table)
            seed_column = resolve_column(deltas, ("seed", "seed_base"), required=False)
            if seed_column is None:
                raise ValueError("long seed table lacks seed identity")
            grouped = deltas.groupby(["_region", "_timing", seed_column])["_delta"].mean()
            pivot = grouped.unstack(seed_column)
            if 2026 in pivot.columns and 3026 in pivot.columns:
                has_consistent_positive = bool(((pivot[2026] > 0) & (pivot[3026] > 0)).any())
            elif "2026" in pivot.columns and "3026" in pivot.columns:
                has_consistent_positive = bool(
                    ((pivot["2026"] > 0) & (pivot["3026"] > 0)).any()
                )
    except StopIteration:
        pass
    except (KeyError, ValueError):
        # The gate object remains authoritative.  This fallback merely selects
        # C versus D when a compact seed table does not repeat long-form rows.
        has_consistent_positive = gates["A"]

    if not gates["C"] and has_consistent_positive:
        return "C", CLASSIFICATION_LABELS["C"]
    if not gates["A"] and not has_consistent_positive:
        return "D", CLASSIFICATION_LABELS["D"]
    return "INDETERMINATE", CLASSIFICATION_LABELS["INDETERMINATE"]


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _delta_summary(frame: pd.DataFrame) -> pd.DataFrame:
    deltas = region_delta_frame(frame)
    summary = (
        deltas.loc[deltas["_region"].isin(PRIMARY_REGIONS)]
        .groupby("_region", sort=False)["_delta"]
        .agg(mean="mean", minimum="min", maximum="max", cells="count")
    )
    return summary.reindex([region for region in PRIMARY_REGIONS if region in summary.index])


def _best_delta(frame: pd.DataFrame, candidates: Iterable[str]) -> tuple[str, float]:
    summary = _delta_summary(frame)
    selected = summary.loc[summary.index.intersection(list(candidates)), "mean"].dropna()
    if selected.empty:
        return "无法判定", float("nan")
    best = str(selected.idxmax())
    return best, float(selected.loc[best])


def _oracle_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_region"] = work[
        resolve_column(work, ("variant", "region", "candidate", "feature_variant"))
    ].astype(str).str.upper().str.extract(r"\b(R[0-5])\b", expand=False)
    _, ratio = numeric_column(
        work,
        ("recovery_ratio", "oracle_recovery_ratio", "mean_recovery_ratio"),
        required=False,
    )
    if ratio is None:
        _, numerator = numeric_column(
            work, ("numerator", "mask_free_uplift", "delta_auroc_vs_r0")
        )
        _, denominator = numeric_column(
            work, ("denominator", "oracle_uplift", "peri20_uplift")
        )
        assert numerator is not None and denominator is not None
        valid = denominator > 0
        ratio = pd.Series(np.nan, index=work.index, dtype=float)
        ratio.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    work["_ratio"] = ratio
    return work.groupby("_region", sort=False)["_ratio"].agg(
        mean="mean", minimum="min", maximum="max", cells="count"
    )


def _ftv_summary(frame: pd.DataFrame) -> tuple[str, str, float]:
    work = frame.copy()
    scope = resolve_column(work, ("analysis_scope", "scope"), required=False)
    if scope is not None:
        primary = work[scope].astype(str).eq("primary_measurement_valid")
        if primary.any():
            work = work.loc[primary].copy()
    region = resolve_column(work, ("variant", "region", "feature_variant", "candidate"))
    target = resolve_column(work, ("target", "endpoint", "outcome"), required=False)
    metric, values = numeric_column(
        work,
        ("r2", "test_r2", "spearman", "test_spearman", "pearson", "rmse", "mae"),
    )
    assert metric is not None and values is not None
    work["_region"] = work[region].astype(str).str.upper().str.extract(r"\b(R[0-5])\b", expand=False)
    work["_target"] = work[target].astype(str) if target else "FTV"
    work["_value"] = values
    grouped = work.groupby(["_target", "_region"], sort=False)["_value"].mean()
    lower_better = _normal(metric) in {"rmse", "mae", "testrmse", "testmae"}
    index = grouped.idxmin() if lower_better else grouped.idxmax()
    return str(index[0]), str(index[1]), float(grouped.loc[index])


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 12, max_columns: int = 10) -> str:
    shown = frame.head(max_rows).iloc[:, :max_columns].copy()
    columns = [str(column) for column in shown.columns]

    def clean(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.5g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")[:60]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    if len(frame) > max_rows or len(frame.columns) > max_columns:
        lines.append(
            f"\n（预览 {min(len(frame), max_rows)}/{len(frame)} 行、"
            f"{min(len(frame.columns), max_columns)}/{len(frame.columns)} 列；完整表见链接。）"
        )
    return "\n".join(lines)


def _status_text(passed: bool) -> str:
    return "通过" if passed else "未通过"


def _safe_scalar(payload: Mapping[str, Any], aliases: Iterable[str], default: str = "未记录") -> str:
    wanted = {_normal(value) for value in aliases}
    for key, value in payload.items():
        if _normal(key) in wanted and isinstance(value, (str, int, float, bool)):
            text = str(value)
            # Do not replicate environment paths or patient-like tokens into a
            # public report even if an upstream summary is malformed.
            if "/data/" not in text and "/home/" not in text and not re.search(
                r"\bACRIN[-_ ]?\d+\b", text, flags=re.IGNORECASE
            ):
                return text
    return default


def _validate_run_summary(payload: Mapping[str, Any]) -> None:
    status = _safe_scalar(payload, ("status", "run_status"), default="")
    if _normal(status) not in {"complete", "completed", "pass", "passed", "success"}:
        raise ValueError(f"formal run_summary status is not complete: {status!r}")
    if "push_error" not in payload:
        raise ValueError("run_summary lacks required push_error")
    for key in (
        "public_outputs_contain_patient_level_data",
        "contains_patient_level_data",
        "public_contains_patient_level_data",
    ):
        if key in payload and _bool_value(payload[key]) is not False:
            raise ValueError("run_summary does not attest patient-free public outputs")


def build_report(
    tables: Mapping[str, pd.DataFrame],
    gates_payload: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    gates = extract_gate_results(gates_payload)
    any_positive = gates_payload.get("any_primary_candidate_two_seed_positive")
    if not isinstance(any_positive, bool):
        raise ValueError(
            "gates.json lacks boolean any_primary_candidate_two_seed_positive"
        )
    classification_code, classification = derive_classification(
        gates, tables, any_two_seed_positive=any_positive
    )
    _validate_run_summary(run_summary)
    recorded_classification = str(gates_payload.get("scientific_classification", ""))
    if recorded_classification != classification:
        raise ValueError(
            "gates scientific_classification disagrees with preregistered precedence: "
            f"recorded={recorded_classification!r}, expected={classification!r}"
        )
    if str(run_summary.get("scientific_classification", "")) != classification:
        raise ValueError("run_summary scientific classification differs from gates")

    best_region, best_gain = _best_delta(
        tables["mri_only_pcr"], ("R1", "R2", "R3", "R4", "R5")
    )
    best_shell, shell_gain = _best_delta(tables["mri_only_pcr"], ("R1", "R2", "R3"))
    mri_summary = _delta_summary(tables["mri_only_pcr"])
    r5_gain = float(mri_summary.loc["R5", "mean"]) if "R5" in mri_summary.index else float("nan")
    incremental_region, incremental_gain = _best_delta(
        tables["clinical_ftv_incremental"], ("R1", "R2", "R3", "R4", "R5")
    )
    phenotype_region, phenotype_gain = _best_delta(
        tables["phenotype"], ("R1", "R2", "R3", "R4", "R5")
    )
    ftv_target, ftv_region, ftv_value = _ftv_summary(tables["ftv"])
    oracle = _oracle_summary(tables["oracle_recovery"])
    oracle_candidates = oracle.loc[
        oracle.index.intersection(["R1", "R2", "R3", "R5"]), "mean"
    ].dropna()
    oracle_region = str(oracle_candidates.idxmax()) if not oracle_candidates.empty else "无法判定"
    oracle_ratio = float(oracle_candidates.max()) if not oracle_candidates.empty else float("nan")

    if classification_code == "A":
        next_choice = "fixed region-aware"
        train_answer = "值得进入正式 Region-Aware Response State 的独立预注册训练阶段，但本审计本身不启动训练。"
    elif classification_code == "C":
        next_choice = "lesion-relative region"
        train_answer = "当前证据不足以训练正式 fixed Region-Aware Response State；应先做独立的 lesion-relative/soft-localization 可部署性研究。"
    else:
        next_choice = "保持 Full Local"
        train_answer = "不值得据此启动正式 Region-Aware Response State 训练；保留 Full Local，等待独立证据。"

    if gates["C"]:
        geometry_answer = (
            "不支持“必须依赖”的断言：固定坐标区域已达到预注册的部分恢复门槛；"
            "但这不能说明 lesion-relative geometry 没有额外价值。"
        )
    else:
        geometry_answer = (
            "固定坐标 readout 未达到部分恢复门槛，结果与 lesion-relative localization 可能重要一致；"
            "但失败不能证明它是必要条件，learned localization 仍只是待检验假设。"
        )

    branch = _safe_scalar(run_summary, ("branch", "experiment_branch"), str(config.get("branch", "未记录")))
    commit = _safe_scalar(run_summary, ("commit_sha", "commit", "git_commit"))
    push = _safe_scalar(run_summary, ("push_status", "github_push_status"))
    push_error = (
        "null"
        if run_summary.get("push_error") is None
        else _safe_scalar(run_summary, ("push_error",), default="未记录")
    )
    push_error = " ".join(push_error.split()).replace("`", "'")
    elapsed = _safe_scalar(run_summary, ("elapsed_seconds", "runtime_seconds"))

    gate_lines = "\n".join(
        f"- Gate {letter}：{_status_text(gates[letter])}" for letter in ("A", "B", "C", "D")
    )
    artifact_lines = []
    required_attachments = (
        ("occupancy", FIGURES[1], "Region occupancy statistics"),
        ("mri_only_pcr", FIGURES[2], "MRI-only pCR"),
        ("phenotype", FIGURES[3], "Phenotype probes"),
        ("clinical_ftv_incremental", FIGURES[4], "C+F incremental"),
        ("ftv", FIGURES[5], "FTV / delta-FTV"),
        ("oracle_recovery", FIGURES[6], "Oracle recovery"),
        ("bootstrap", FIGURES[7], "Patient-level bootstrap"),
        ("seed_consistency", FIGURES[8], "Seed consistency"),
        ("timing_sensitivity", FIGURES[9], "Timing sensitivity"),
    )
    for logical_name, figure, title in required_attachments:
        filename = PUBLIC_TABLES[logical_name]
        artifact_lines.append(
            f"### {title}\n\n"
            f"[完整 CSV](../metrics/{filename}) · [图](../figures/{figure})\n\n"
            f"{_markdown_table(tables[logical_name])}"
        )
    attachment_text = "\n\n".join(artifact_lines)

    report = f"""# Mask-Free Region-Aware Representation Audit 最终报告

## 结论先行

本 frozen-feature diagnostic 的唯一科学分类是 **{classification}**。四个预注册门控如下：

{gate_lines}

本报告中的 “mask-free” 是严格限定语：**mask-free-at-readout-only**。新增区域 readout 不读取 lesion mask、bbox、FTV、clinical、pCR、phenotype 或 future visit；但上游输入是 **T0 localization-centered C1B** crops，因此它不是 acquisition-centered、端到端 mask-independent deployment 验证。

## 逐项回答 12 个问题

### Q1 — 哪个 mask-free region 最好？

按注册 MRI-only pCR cells 的描述性平均 R0 增量，最好的是 **{best_region}**（平均 ΔAUROC `{_fmt(best_gain)}`）。这是内部 OOF 汇总，不是事后选择新 primary，也不改变预注册 gates。

### Q2 — Central / Inner / Outer 哪个更有用？

三者的描述性最佳项是 **{best_shell}**（平均 ΔAUROC `{_fmt(shell_gain)}`；R1=Central、R2=Inner、R3=Outer）。该排序只描述冻结 probe，不把 shell 解码性解释为组织来源。

### Q3 — Three-region representation 是否优于 Full Local？

R5 相对 R0 的注册 cells 平均 ΔAUROC 为 `{_fmt(r5_gain)}`；因此描述性答案为 **{'是' if math.isfinite(r5_gain) and r5_gain > 0 else '否'}**。正式支持仍由 Gate A（当前{_status_text(gates['A'])}）决定，而非单个平均值。

### Q4 — pCR 是否改善？

最佳 mask-free candidate 的 MRI-only pCR 平均增量为 `{_fmt(best_gain)}`；Gate A **{_status_text(gates['A'])}**。这仅说明 frozen representation 中的可解码信息变化，不等于治疗反应机制或外部临床效用。

### Q5 — Clinical+FTV 后是否仍有增量？

C+F+Rk 相对 C+F+R0 的最佳描述性项为 **{incremental_region}**（平均 ΔAUROC `{_fmt(incremental_gain)}`）；Gate B **{_status_text(gates['B'])}**。Gate B 同时约束了相对 C+F baseline 不系统性为负，不能用单个正 cell 替代该判断。

### Q6 — HR/HER2/subtype 是否改善？

相对 R0 的 phenotype 汇总中最佳项为 **{phenotype_region}**（平均 ΔAUROC `{_fmt(phenotype_gain)}`）；Gate D **{_status_text(gates['D'])}**。Gate D 未通过时，pCR regional signal 不得称为 molecular phenotype；即使通过，也只能称 profile-associated decoding，不能称分子机制。

### Q7 — FTV / ΔFTV 是否改善？

FTV response-control 表中所报指标的最佳描述性 cell 为 **{ftv_target} / {ftv_region}**（值 `{_fmt(ftv_value)}`）。这些 Ridge probes 用于判断信号是否主要反映 burden/response；它们不是影像测量替代品，也不证明生物学过程。

### Q8 — 是否部分恢复 Goal 5 PERI20 Oracle gain？

最佳注册 candidate 是 **{oracle_region}**，平均 recovery ratio 为 `{_fmt(oracle_ratio)}`；Gate C **{_status_text(gates['C'])}**。ratio 只在 matched、Oracle uplift 为正的 LOCAL0/T0-T1 cells 定义。

### Q9 — Oracle gain 是否必须依赖 lesion-relative geometry？

{geometry_answer}

必须突出 **Oracle denominator representation mismatch**：新 numerator 是 `Rk(raw regional means) - R0(raw full-local mean)`，冻结 denominator 是 Goal 5 `PERI20(mean+std) - FIXED_P3(full-local mean+std)`。因此 recovery ratio 是预注册诊断桥接量，不是同构表示之间的因果分解。

### Q10 — 下一步应保持 Full Local、fixed region-aware、learned localization 还是 lesion-relative region？

当前唯一选择是 **{next_choice}**。该选择由预注册 classification 映射产生，不根据 test set 重选半径、region 或模型容量。

### Q11 — 是否值得训练正式 Region-Aware Response State？

{train_answer} 禁止从本 Goal 自动训练 region-token JEPA、attention、MIL、segmentation-guided branch 或任何主 encoder/JEPA。

### Q12 — 哪些结论必须保持 diagnostic，不能写成 biological claim？

这是 **diagnostic/non-biological boundary**：AUROC、FTV/ΔFTV 可解码性、central/shell 排序和 Oracle recovery 都只是 frozen、内部 OOF representation diagnostics。不得写成统计显著性、外部泛化、因果疗效、peritumoral biology、molecular mechanism 或端到端 mask-independent deployment。限制包括粗 Z spacing、大 receptive field、complete-four-visit selection，以及上游 T0-centered localization。`T0-T3` 必须始终解释为 **T3 late/pre-surgery**，不能与 early/mid timing 合并宣传。

## 时间、几何与比较口径

- Primary physical partition 固定为 32/48/64 mm；secondary 24/40/64 mm 不参与 primary classification。
- 所有区域使用 fractional feature-cell occupancy weighted pooling，而非 nearest-voxel binary assignment。
- pCR causal prefixes 为 T0、T0-T1、T0-T2、T0-T3；最后一项永久标记 late/pre-surgery。
- patient-level bootstrap 是 outer-fold 内 paired resampling；公开文件只保留 aggregate。
- Goal 5 absolute AUROC 只在同一 matched population 内比较，不跨 variant-specific population。

## 必附图表

### Region schematic

[区域示意图](../figures/{FIGURES[0]})

{attachment_text}

### Clinical pCR 辅助表

[完整 CSV](../metrics/{PUBLIC_TABLES['clinical_pcr']})

{_markdown_table(tables['clinical_pcr'])}

## 执行与 Git 记录

- Branch：`{branch}`
- Commit SHA：`{commit}`
- Push status：`{push}`
- Push error：`{push_error}`
- Formal elapsed seconds：`{elapsed}`
- Run status：`{_safe_scalar(run_summary, ('status', 'run_status'))}`

如果 push 失败，正式 run summary 必须把状态写为 `GITHUB_PUSH_FAILED` 并保留真实错误；禁止 force push。报告与图只读取公开 aggregate metrics、gates 和 run summary，没有读取 private predictions 或 labels。
"""
    for marker in REQUIRED_REPORT_MARKERS:
        if marker not in report:
            raise RuntimeError(f"generated report is missing required marker: {marker}")
    if re.search(r"/(?:data|home)/", report):
        raise RuntimeError("generated public report contains an absolute environment path")
    if re.search(r"\bACRIN[-_ ]?\d+\b", report, flags=re.IGNORECASE):
        raise RuntimeError("generated public report contains a patient-like token")
    return report


def _atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    _atomic_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        destination,
    )


def generate_report(root: Path = ROOT, *, overwrite: bool = False) -> Path:
    root = root.resolve()
    destination = root / REPORT_PATH
    manifest_path = root / REPORT_MANIFEST_PATH
    existing = [path for path in (destination, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "report outputs already exist; pass --overwrite to replace the declared set"
        )
    config_path = root / "configs" / "audit.json"
    config = _read_json(config_path)
    gates_path = root / GATES_PATH
    summary_path = root / RUN_SUMMARY_PATH
    gates = _read_json(gates_path)
    run_summary = _read_json(summary_path)
    tables = load_public_tables(root)
    for figure in FIGURES:
        if not (root / "figures" / figure).is_file():
            raise FileNotFoundError(root / "figures" / figure)

    report = build_report(tables, gates, run_summary, config)
    _atomic_text(report, destination)
    source_paths = [
        config_path,
        gates_path,
        summary_path,
        *(root / "metrics" / filename for filename in PUBLIC_TABLES.values()),
        *(root / "figures" / filename for filename in FIGURES),
    ]
    manifest = {
        "schema_version": 1,
        "experiment": "mask_free_region_aware_audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report": REPORT_PATH.as_posix(),
        "report_sha256": file_sha256(destination),
        "source_artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "patient_level_private": False,
            }
            for path in source_paths
        ],
        "public_outputs_contain_patient_level_data": False,
    }
    _atomic_json(manifest, manifest_path)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    os.umask(0o022)
    destination = generate_report(args.root, overwrite=args.overwrite)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
