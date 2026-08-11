#!/usr/bin/env python3
"""Generate the required Chinese report from frozen aggregate outputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    atomic_json,
    file_sha256,
    load_config,
    require_preregistration_lock,
)
from generate_figures import pair_matched_oracle_deltas  # noqa: E402


TABLES = (
    "table1_pooling_contract.csv",
    "table2_phenotype_probes.csv",
    "table3_mri_only_pcr.csv",
    "table4_clinical_ftv_incremental.csv",
    "table5_residualized_mri.csv",
    "table6_longitudinal_heterogeneity.csv",
    "table7_oracle_regions.csv",
    "table8_stage_b.csv",
)
FIGURES = (
    "01_pooling_schematic.png",
    "02_phenotype_auroc_by_pooling.png",
    "03_pcr_auroc.png",
    "04_beyond_ftv_delta.png",
    "05_mean_vs_std.png",
    "06_core_peritumoral_comparison.png",
    "07_longitudinal_heterogeneity.png",
    "08_representative_spatial_activation_statistics.png",
)
REQUIRED_COMMIT_SUBJECT = "Add spatial heterogeneity phenotype audit"


def preregistration_commit_sha() -> str:
    """Return the committed lock anchor and require byte identity with disk."""

    repo = ROOT.parents[1]
    relative = (ROOT / "PREREGISTRATION_LOCK.json").relative_to(repo).as_posix()
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=repo,
        text=True,
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("preregistration lock has no committed Git anchor")
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=repo
    )
    if hashlib.sha256(committed).hexdigest() != file_sha256(
        ROOT / "PREREGISTRATION_LOCK.json"
    ):
        raise ValueError("committed preregistration lock differs from the current lock")
    return commit


def validate_final_git_provenance(
    commit: str,
    push_status: str,
    branch: str,
    preregistration_commit: str | None = None,
) -> None:
    """Authenticate the non-self-referential experiment commit and push claim."""

    repo = ROOT.parents[1]
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, check=True
    )
    subject = subprocess.check_output(
        ["git", "show", "-s", "--format=%s", commit], cwd=repo, text=True
    ).strip()
    if subject != REQUIRED_COMMIT_SUBJECT:
        raise ValueError("experiment commit subject differs from the required message")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repo,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("experiment commit is not an ancestor of current branch HEAD")
    observed_preregistration_commit = preregistration_commit_sha()
    if (
        preregistration_commit is not None
        and str(preregistration_commit) != observed_preregistration_commit
    ):
        raise ValueError("reported preregistration commit differs from the lock anchor")
    if (
        observed_preregistration_commit == commit
        or subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                observed_preregistration_commit,
                commit,
            ],
            cwd=repo,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError(
            "preregistration lock commit must be a strict ancestor of experiment commit"
        )
    relative_root = ROOT.relative_to(repo)
    for required in (
        relative_root / "PREREGISTRATION_LOCK.json",
        relative_root / "metrics" / "gates.json",
        relative_root / "reports" / "final_report.md",
    ):
        if (
            subprocess.run(
                ["git", "cat-file", "-e", f"{commit}:{required.as_posix()}"],
                cwd=repo,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError(
                f"experiment commit does not contain required audit artifact: {required}"
            )
    if push_status == "PUSHED":
        reference = f"refs/heads/{branch}"
        output = subprocess.check_output(
            ["git", "ls-remote", "--heads", "origin", reference],
            cwd=repo,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        rows = [line.split() for line in output.splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != reference:
            raise ValueError("origin does not expose the configured experiment branch")
        remote_tip = rows[0][0]
        subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-write-fetch-head",
                "origin",
                reference,
            ],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "cat-file", "-e", f"{remote_tip}^{{commit}}"],
            cwd=repo,
            check=True,
        )
        if (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, remote_tip],
                cwd=repo,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError(
                "reported experiment commit is not contained in origin branch"
            )


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        temporary.replace(path)
        path.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _load_table(name: str) -> pd.DataFrame:
    path = ROOT / "metrics" / name
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"required table is empty: {path}")
    forbidden = {"patient_id", "clinical_patient_id", "raw_Patient_ID"}
    if forbidden & set(frame.columns):
        raise ValueError(f"public table exposes patient identity: {path}")
    return frame


def _paired_delta(
    frame: pd.DataFrame,
    *,
    column: str,
    comparison: str,
    reference: str,
    mask: pd.Series | None = None,
) -> dict[str, Any]:
    selected = frame if mask is None else frame.loc[mask]
    identity = [
        name
        for name in ("seed", "arm", "view", "target", "population")
        if name in selected.columns
    ]
    paired = selected.loc[selected[column].isin((reference, comparison))]
    pivot = paired.pivot_table(
        index=identity, columns=column, values="auroc", aggfunc="first"
    )
    if reference not in pivot or comparison not in pivot:
        return {
            "count": 0,
            "mean": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "best": "无可比记录",
        }
    delta = (pivot[comparison] - pivot[reference]).dropna()
    if delta.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "best": "无可比记录",
        }
    best_index = delta.idxmax()
    values = best_index if isinstance(best_index, tuple) else (best_index,)
    best = ", ".join(
        f"{name}={value}" for name, value in zip(identity, values, strict=True)
    )
    return {
        "count": int(len(delta)),
        "mean": float(delta.mean()),
        "minimum": float(delta.min()),
        "maximum": float(delta.max()),
        "best": best,
    }


def _delta_text(summary: Mapping[str, Any]) -> str:
    if int(summary["count"]) == 0:
        return "无成对记录"
    return (
        f"平均 {float(summary['mean']):+.3f}，范围 "
        f"[{float(summary['minimum']):+.3f}, {float(summary['maximum']):+.3f}]；"
        f"最大值位于 {summary['best']}"
    )


def two_seed_endpoint_support(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    population: str,
    expected_seeds: tuple[int, ...],
    threshold: float,
) -> dict[str, list[str]]:
    """Derive endpoint-specific P3-vs-P1 support under the Gate-A rule."""

    required = {
        "seed",
        "arm",
        "view",
        "target",
        "variant",
        "population",
        "auroc",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"endpoint support table lacks columns: {sorted(required - set(frame))}"
        )
    selected = frame.loc[
        frame["target"].isin(targets)
        & frame["population"].eq(population)
        & frame["variant"].isin(("P1", "P3"))
    ].copy()
    identity = ["seed", "arm", "view", "target", "population", "variant"]
    if selected.empty or selected.duplicated(identity).any():
        raise ValueError("endpoint support rows are absent or non-unique")
    pivot = selected.pivot(
        index=["seed", "arm", "view", "target", "population"],
        columns="variant",
        values="auroc",
    )
    if set(pivot.columns) != {"P1", "P3"} or pivot.isna().any().any():
        raise ValueError("endpoint support lacks exact P1/P3 pairs")
    values = (pivot["P3"] - pivot["P1"]).rename("delta").reset_index()
    if set(values["target"].astype(str)) != set(targets):
        raise ValueError("endpoint support target coverage drifted")
    if not np.isfinite(values["delta"].to_numpy(dtype=float)).all():
        raise ValueError("endpoint support contains a non-finite AUROC delta")
    expected = set(int(seed) for seed in expected_seeds)
    output = {target: [] for target in targets}
    group_columns = ["arm", "view", "target", "population"]
    for key, group in values.groupby(group_columns, sort=True):
        by_seed = dict(
            zip(
                group["seed"].astype(int),
                group["delta"].astype(float),
                strict=True,
            )
        )
        if set(by_seed) != expected:
            raise ValueError("endpoint support lacks the exact two-seed comparison")
        arm, view, target, current_population = key
        if all(value >= threshold for value in by_seed.values()):
            output[str(target)].append(f"{arm}/{view}/{current_population}")
    return output


def _endpoint_support_text(support: Mapping[str, list[str]]) -> str:
    return "；".join(
        f"{target}={'SUPPORTED' if rows else 'NOT_SUPPORTED'}"
        f"（{len(rows)} 个 arm/view 比较）"
        for target, rows in support.items()
    )


def _best_oracle(table: pd.DataFrame) -> str:
    paired = pair_matched_oracle_deltas(table)
    paired = paired.loc[paired["target"].isin(("HR", "HER2", "subtype_4class"))]
    if paired.empty:
        return "无有效 phenotype Oracle 记录"
    summary = (
        paired.groupby("variant", sort=True)["delta_auroc"]
        .mean()
        .sort_values(ascending=False)
    )
    return ", ".join(f"{name}: ΔAUROC={value:+.3f}" for name, value in summary.items())


def _classification_code(name: str) -> str:
    mapping = {
        "PHENOTYPE INFORMATION PRESENT BUT MEAN-POOLED AWAY": "A",
        "PHENOTYPE SPATIALLY LOCALIZED": "B",
        "CURRENT ENCODER LACKS PHENOTYPE INFORMATION": "C",
        "MIXED": "D",
    }
    if name not in mapping:
        raise ValueError(f"unknown scientific classification: {name}")
    return mapping[name]


def _stage_b_text(table: pd.DataFrame, authorization: Mapping[str, Any]) -> str:
    authorized = bool(authorization["authorized"])
    status_values = sorted(set(table.get("status", pd.Series(dtype=str)).astype(str)))
    status = ", ".join(status_values) if status_values else "未提供状态列"
    if not authorized:
        if "NOT_RUN_NOT_AUTHORIZED" not in status_values:
            raise ValueError("unauthorized Stage B table lacks NOT_RUN_NOT_AUTHORIZED")
        return "未授权且未运行（NOT_RUN_NOT_AUTHORIZED），符合 Gate A OR Gate C 规则。"
    required = {"target", "delta_auroc", "brier_improvement"}
    if not required.issubset(table.columns):
        raise ValueError("authorized Stage B table lacks paired delta columns")
    phenotype = pd.to_numeric(
        table.loc[table["target"].ne("pCR"), "delta_auroc"], errors="coerce"
    )
    pcr = pd.to_numeric(
        table.loc[table["target"].eq("pCR"), "delta_auroc"], errors="coerce"
    )
    brier = pd.to_numeric(
        table.loc[table["target"].eq("pCR"), "brier_improvement"],
        errors="coerce",
    )
    if (
        phenotype.empty
        or pcr.empty
        or brier.empty
        or not np.isfinite(phenotype.to_numpy(dtype=float)).all()
        or not np.isfinite(pcr.to_numpy(dtype=float)).all()
        or not np.isfinite(brier.to_numpy(dtype=float)).all()
    ):
        raise ValueError("authorized Stage B paired deltas are incomplete/non-finite")
    return (
        f"已授权；表 8 状态为 {status}。相对同 seed/view/target/population 的 "
        f"Stage-A LOCAL3/P1 基线（Stage-B 为 dual-state arm）：phenotype ΔAUROC 平均 {phenotype.mean():+.3f}，"
        f"范围 [{phenotype.min():+.3f}, {phenotype.max():+.3f}]；"
        f"pCR ΔAUROC 平均 {pcr.mean():+.3f}，范围 [{pcr.min():+.3f}, "
        f"{pcr.max():+.3f}]；pCR Brier 改善平均 {brier.mean():+.3f}。"
    )


def render(*, experiment_commit: str, push_status: str) -> tuple[str, dict[str, Any]]:
    config = load_config(ROOT / "configs" / "audit.json", verify_inputs=True)
    lock = require_preregistration_lock(config)
    gates_path = ROOT / "metrics" / "gates.json"
    authorization_path = ROOT / "metrics" / "stage_b_authorization.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("stage_a_gates_sha256") != file_sha256(gates_path):
        raise ValueError("Stage-B authorization is not bound to current gates")
    tables = {name: _load_table(name) for name in TABLES}
    for name in FIGURES:
        if not (ROOT / "figures" / name).is_file():
            raise FileNotFoundError(ROOT / "figures" / name)

    table2 = tables[TABLES[1]]
    table3 = tables[TABLES[2]]
    table4 = tables[TABLES[3]]
    table7 = tables[TABLES[6]]
    phenotype_p3 = _paired_delta(
        table2, column="variant", comparison="P3", reference="P1"
    )
    hr_p3 = _paired_delta(
        table2,
        column="variant",
        comparison="P3",
        reference="P1",
        mask=table2["target"].eq("HR"),
    )
    std_vs_mean = _paired_delta(
        table2, column="variant", comparison="P2", reference="P1"
    )
    pcr_p3 = _paired_delta(
        table3,
        column="variant",
        comparison="P3",
        reference="P1",
        mask=table3["population"].eq("ftv_complete_375"),
    )
    beyond = _paired_delta(
        table4,
        column="model",
        comparison="C+F+P3",
        reference="C+F",
    )
    gate_values = gates["gates"]
    gate_a = bool(gate_values["A"]["passed"])
    gate_b = bool(gate_values["B"]["passed"])
    gate_c = bool(gate_values["C"]["passed"])
    gate_d = bool(gate_values["D"]["passed"])
    classification = str(gates["scientific_classification"])
    code = _classification_code(classification)
    stage_b = _stage_b_text(tables[TABLES[7]], authorization)
    seeds = tuple(int(value) for value in config["frozen_cells"]["seed_bases"])
    threshold_a = float(config["gates"]["A"]["minimum_auroc_gain_each_seed"])
    phenotype_support = two_seed_endpoint_support(
        table2,
        targets=("HR", "HER2", "subtype_4class"),
        population="full_808",
        expected_seeds=seeds,
        threshold=threshold_a,
    )
    pcr_support = two_seed_endpoint_support(
        table3,
        targets=("pCR",),
        population=str(config["analysis"]["primary_pcr_population"]),
        expected_seeds=seeds,
        threshold=threshold_a,
    )
    phenotype_gate_a_disjunct = bool(
        phenotype_support["HER2"] or phenotype_support["subtype_4class"]
    )
    pcr_gate_a_disjunct = bool(pcr_support["pCR"])
    if gate_a != (phenotype_gate_a_disjunct or pcr_gate_a_disjunct):
        raise ValueError("Gate A differs from endpoint-specific two-seed support")
    direct_or_complementary_support = gate_a or gate_b
    evidence_sources = (
        ", ".join(
            name for name, passed in (("Gate A", gate_a), ("Gate B", gate_b)) if passed
        )
        or "none"
    )
    preregistration_commit = preregistration_commit_sha()

    answers = [
        f"1. **Mean pooling 是否丢失 heterogeneity information？** {'有预注册阈值证据' if direct_or_complementary_support else '未获得预注册阈值证据'}（支持来源：{evidence_sources}）。Gate A 检验 P3−P1 的 phenotype/matched-pCR 信号；Gate B 独立检验 P3 在 C+F 之外且优于 C+F+P1 的互补性。P3−P1 phenotype AUROC：{_delta_text(phenotype_p3)}。",
        f"2. **Std 是否有独立价值？** P2−P1 phenotype AUROC 的描述性结果为：{_delta_text(std_vs_mean)}。这回答可解码性，不把单次正增益解释为因果或泛化证明。",
        f"3. **Mean+Std 是否优于 Mean？** {'存在稳定的直接或互补证据' if direct_or_complementary_support else '未满足直接 Gate-A 或互补 Gate-B 的双-seed标准'}（{evidence_sources}）；Gate B 通过时即表示 C+F+P3 同时优于 C+F 与 C+F+P1，不能因 Gate A 失败而忽略。",
        f"4. **HR/HER2/subtype 是否改善？** 以两个 seed 均满足 P3−P1 AUROC ≥ +{threshold_a:.2f} 分别判定：{_endpoint_support_text(phenotype_support)}。HR 不进入 Gate A；HER2/subtype disjunct={'PASS' if phenotype_gate_a_disjunct else 'FAIL'}。HR 全部成对结果：{_delta_text(hr_p3)}。",
        f"5. **pCR 是否改善？** matched 375 的独立双-seed pCR disjunct={'PASS' if pcr_gate_a_disjunct else 'FAIL'}（{_endpoint_support_text(pcr_support)}）；全部 P3−P1 成对结果：{_delta_text(pcr_p3)}。该结论不由 phenotype disjunct 代替。",
        f"6. **是否有 FTV 之外的信息？** Gate B={'PASS' if gate_b else 'FAIL'}。C+F+P3−C+F：{_delta_text(beyond)}。",
        f"7. **Core 还是 peritumoral 含更多 phenotype signal？** 仅在 HR/HER2/subtype 内，各区域与其同一 oracle_pair complete-case 人群的 FIXED_P3 比较；最大 matched uplift 及平均配对增量排序为：{_best_oracle(table7)}。区域间使用不同 variant-specific cohorts，因此 absolute core-vs-peri AUROC 排名不可识别，也不作该排序。",
        f"8. **Oracle 是否强于 mask-free fixed local？** Gate C={'PASS' if gate_c else 'FAIL'}，支持比较数={len(gate_values['C']['supporting_comparisons'])}。每个比较都在相同、variant-specific complete-case 人群并限制于同一 64-mm LOCAL support。",
        f"9. **当前瓶颈是什么？** Stage-A 唯一分类为 {code}. `{classification}`；Gate D={'PASS' if gate_d else 'FAIL'}。",
        f"10. **是否需要 Response–Phenotype factorized state？** {'值得进入后续确认，因为 Gate A 或 Gate B 支持统计保留/互补。' if gate_a or gate_b else '当前证据不足以把它列为首选；应先处理定位或表征问题。'}",
        f"11. **是否需要 foundation encoder？** {'是，当前 encoder-deficit 分类使预训练/foundation encoder 成为首要下一步。' if code == 'C' else '本审计不要求立即更换；应先按当前分类完成更大样本/多 seed 确认。'}",
        f"12. **Stage B 是否授权、结果如何？** {stage_b}",
    ]

    table_links = "\n".join(
        f"- [表 {index}：{name}](../metrics/{name})"
        for index, name in enumerate(TABLES, start=1)
    )
    figure_links = "\n".join(
        f"- [图 {index}：{name}](../figures/{name})"
        for index, name in enumerate(FIGURES, start=1)
    )
    text = f"""# Spatial Heterogeneity / Phenotype Pooling Audit 最终报告

## 结论先行

本实验的 Stage-A 科学分类为 **{code}. `{classification}`**。四个预注册 Gate：A={'PASS' if gate_a else 'FAIL'}，B={'PASS' if gate_b else 'FAIL'}，C={'PASS' if gate_c else 'FAIL'}，D={'PASS' if gate_d else 'FAIL'}。该分类只由冻结 Stage A 决定，Conditional Stage B 不回写诊断结论。

## 十二个问题的明确回答

{chr(10).join(answers)}

## 方法与解释边界

- Stage A 使用已选择且 test-blind 的 LOCAL0/LOCAL3 checkpoint，共 2 seeds × 5 folds；encoder 不重训。
- 实际 pre-pooling tensor 为 128×14×22×20；P1/P2/P3/P4/P5 均在同一固定 64-mm fractional LOCAL support 上计算。
- 所有 scaler、临床编码、FTV residualization、C 与阈值选择只在 outer train/validation 内完成；outer test 只预测一次。
- Oracle 只做机制定位，所有区域在 feature grid 上再限制到同一 500-cell LOCAL support；大 receptive field 与粗 Z 方向分辨率仍限制精细边界解释。
- 本报告中的“改善”是冻结 OOF 诊断结果，不等于独立外部验证或统计显著性。

## 表格

{table_links}

## 图

{figure_links}

## 可复现性与交付

- Branch：`{config['branch']}`
- Preregistration lock：`{file_sha256(ROOT / 'PREREGISTRATION_LOCK.json')}`
- Preregistration commit SHA：`{preregistration_commit}`
- Preregistration base HEAD：`{lock['git_provenance_before_freeze']['base_head']}`
- Experiment commit SHA：`{experiment_commit}`
- GitHub push status：`{push_status}`
- 原始 MRI、patient-level feature/prediction 与 checkpoint 均未纳入版本控制；公开表不含 patient identifier。
"""
    input_paths = [
        ROOT / "PREREGISTRATION_LOCK.json",
        gates_path,
        authorization_path,
        *(ROOT / "metrics" / name for name in TABLES),
        *(ROOT / "figures" / name for name in FIGURES),
    ]
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "language": "zh-CN",
        "scientific_classification": classification,
        "preregistration_commit": preregistration_commit,
        "experiment_commit": experiment_commit,
        "push_status": push_status,
        "input_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in input_paths
        },
        "contains_patient_level_data": False,
    }
    return text, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument(
        "--push-status",
        required=True,
        choices=("PENDING", "PUSHED", "GITHUB_PUSH_FAILED"),
    )
    args = parser.parse_args()
    os.umask(0o077)
    commit = str(args.experiment_commit)
    pending_commit = commit == "PENDING"
    pending_push = args.push_status == "PENDING"
    if pending_commit != pending_push:
        raise ValueError(
            "report provenance must be either (PENDING,PENDING) or fully final"
        )
    if not pending_commit and re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("experiment commit must be PENDING or a full lowercase SHA")
    repo = ROOT.parents[1]
    config = load_config(ROOT / "configs" / "audit.json", verify_inputs=True)
    require_preregistration_lock(config)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    if branch != config["branch"]:
        raise ValueError("report provenance is being generated on another branch")
    if not pending_commit:
        validate_final_git_provenance(
            commit,
            str(args.push_status),
            str(config["branch"]),
            preregistration_commit_sha(),
        )
    report_path = ROOT / "reports" / "final_report.md"
    manifest_path = ROOT / "reports" / "report_manifest.json"
    text, manifest = render(experiment_commit=commit, push_status=args.push_status)
    if report_path.exists() != manifest_path.exists():
        # Neither member alone is a completed/authenticated report. Recover only
        # this exact output pair after all frozen inputs have been validated.
        report_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if report_path.exists() or manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("report_sha256") != file_sha256(report_path):
            raise ValueError("existing report pair fails its recorded report hash")
        if previous.get("input_sha256") != manifest["input_sha256"]:
            raise ValueError("scientific report inputs changed after first rendering")
        if previous.get("preregistration_commit") != manifest["preregistration_commit"]:
            raise ValueError("preregistration Git anchor changed after first rendering")
        previous_pending = previous.get("experiment_commit") == "PENDING"
        previous_push_pending = previous.get("push_status") == "PENDING"
        if previous_pending != previous_push_pending:
            raise ValueError("existing report has an invalid mixed provenance state")
        if not previous_pending:
            raise FileExistsError("completed report provenance is immutable")
        if pending_commit:
            raise ValueError(
                "report provenance update must supply final commit and push status"
            )
    _atomic_text(text, report_path)
    manifest["report_sha256"] = file_sha256(report_path)
    atomic_json(manifest, manifest_path)
    print(
        json.dumps({"status": "COMPLETE", "report": str(report_path)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
