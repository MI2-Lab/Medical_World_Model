"""Grounding–JEPA conflict audit 的中文图表、最终报告与严格验收。

交付器只消费已经由正式分析 marker 封口的数据。它不会计算新的诊断规则，
也不会在缺文件、hash 漂移或数据质量不合格时生成占位图或伪造数字。
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

from .aggregation import (  # noqa: E402
    COVERAGE_COLUMNS,
    EXISTING_METRIC_COLUMNS,
    LAYER_LEVEL_COLUMNS,
    PRIVATE_MANIFEST_COLUMNS,
    PUBLIC_MANIFEST_COLUMNS,
    REPRESENTATIVE_COLUMNS,
    RUN_LEVEL_COLUMNS,
    TRAJECTORY_CHANGE_COLUMNS,
    TRAJECTORY_COLUMNS,
)
from .analysis import (  # noqa: E402
    ANALYSIS_INPUT_MANIFEST_COLUMNS,
    CORRELATION_COLUMNS,
    COVERAGE_CORRELATION_COLUMNS,
    DYNAMICS_COLUMNS,
    FOLD_SIGNATURE_COLUMNS,
    LOCALIZATION_COLUMNS,
    PASS_FAIL_COLUMNS,
)
from .assets import ASSET_MANIFEST_COLUMNS  # noqa: E402
from .contracts import (  # noqa: E402
    AUDIT_ROOT,
    FOLDS,
    GROUPS,
    REPO_ROOT,
    SEED_BASES,
    SPLITS,
    atomic_json,
    canonical_json_sha256,
    ensure_no_patient_columns,
    file_sha256,
)
from .gradients import BATCH_GRADIENT_COLUMNS  # noqa: E402
from .phase_a import PHASE_A_COLUMNS  # noqa: E402
from .source_contract import (  # noqa: E402
    PRIVATE_CACHE_MANIFEST,
    SOURCE_CONTRACT,
    SOURCE_MANIFEST_COLUMNS,
    assert_source_contract,
)


FINAL_ANALYSIS_MANIFEST = AUDIT_ROOT / "metrics" / "final_analysis_manifest.json"
FINAL_ANALYSIS_MARKER = AUDIT_ROOT / "metrics" / "FINAL_ANALYSIS_COMPLETE.json"
FIGURE_MANIFEST = AUDIT_ROOT / "metrics" / "figure_manifest.csv"
FINAL_REPORT = AUDIT_ROOT / "reports" / "final_report.md"
ACCEPTANCE_REPORT = AUDIT_ROOT / "reports" / "acceptance.json"
ACCEPTANCE_COMPAT = AUDIT_ROOT / "metrics" / "acceptance_check.json"

FIGURE_DPI = 240
MIN_FIGURE_WIDTH = 2_400
MIN_FIGURE_HEIGHT = 1_400
PASS_COLOR = "#2A9D8F"
FAIL_COLOR = "#E76F51"
TRAIN_COLOR = "#457B9D"
VALIDATION_COLOR = "#9B5DE5"

FIGURE_MANIFEST_COLUMNS = (
    "schema_version",
    "figure_id",
    "filename",
    "title",
    "analysis_unit",
    "uncertainty_note",
    "input_files_json",
    "input_sha256",
    "image_sha256",
    "width_pixels",
    "height_pixels",
    "dpi",
    "font_family",
    "contains_patient_ids",
)

TRAINING_HISTORY_COLUMNS = (
    "seed_base",
    "fold",
    "base_gate",
    "base_degradation",
    "epoch",
    "is_selected_checkpoint",
    "train_total_loss",
    "train_base_loss",
    "train_state_loss",
    "train_sigreg_loss",
    "train_ftv_loss",
    "train_weighted_ftv_loss",
    "val_state_loss",
    "val_base_objective",
    "val_ftv_loss",
    "representation_std",
    "grounded_exposure",
)

HYPOTHESIS_DECISION_COLUMNS = (
    "schema_version",
    "hypothesis",
    "hierarchy_reached",
    "rule_eligible",
    "rule_satisfied",
    "selected",
    "decision_status",
    "selected_hypothesis",
    "first_recommendation",
    "second_recommendation",
    "condition_details_json",
    "contains_patient_ids",
)

REQUIRED_ANALYSIS_ARTIFACTS = frozenset(
    {
        "metrics/analysis_input_manifest.csv",
        "metrics/pass_fail_comparison.csv",
        "metrics/gradient_correlation_metrics.csv",
        "metrics/dynamics_correlations.csv",
        "metrics/coverage_correlations.csv",
        "metrics/layer_localization_metrics.csv",
        "metrics/fold_signature_metrics.csv",
        "metrics/hypothesis_decision.csv",
        "metrics/diagnosis.json",
    }
)


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    filename: str
    title: str
    analysis_unit: str
    uncertainty_note: str
    inputs: tuple[str, ...]
    renderer: str


FIGURE_SPECS = (
    FigureSpec(
        "F01",
        "01_base_degradation_vs_static_ftv_gain.png",
        "基础损失退化与静态 FTV 表征增益",
        "seed×fold run（n=25）",
        "相关系数误差为20,000次 crossed seed×fold percentile bootstrap 95% CI；散点无测量误差条。",
        (
            "metrics/run_level_existing_metrics.csv",
            "metrics/phase_a_gain_correlations.csv",
        ),
        "phase_static",
    ),
    FigureSpec(
        "F02",
        "02_base_degradation_vs_observed_delta_ftv_gain.png",
        "基础损失退化与 observed ΔFTV 增益",
        "seed×fold run（n=25）",
        "相关系数误差为20,000次 crossed seed×fold percentile bootstrap 95% CI；散点无测量误差条。",
        (
            "metrics/run_level_existing_metrics.csv",
            "metrics/phase_a_gain_correlations.csv",
        ),
        "phase_dynamic",
    ),
    FigureSpec(
        "F03",
        "03_base_degradation_vs_all_shared_cosine.png",
        "基础损失退化与 all-shared 梯度 cosine",
        "seed×fold run（每点先聚合8个固定 batch，n=25）",
        "相关系数误差为20,000次 crossed seed×fold percentile bootstrap 95% CI；train/validation 分开。",
        (
            "metrics/run_level_conflict_metrics.csv",
            "metrics/gradient_correlation_metrics.csv",
        ),
        "gradient_scatter_cosine",
    ),
    FigureSpec(
        "F04",
        "04_base_degradation_vs_mbase.png",
        "基础损失退化与 base-descent margin",
        "seed×fold run（每点先聚合8个固定 batch，n=25）",
        "相关系数误差为20,000次 crossed seed×fold percentile bootstrap 95% CI；M_base=0/1 为解释参考线。",
        (
            "metrics/run_level_conflict_metrics.csv",
            "metrics/gradient_correlation_metrics.csv",
        ),
        "gradient_scatter_mbase",
    ),
    FigureSpec(
        "F05",
        "05_pass_fail_cosine.png",
        "PASS/FAIL 的 all-shared 梯度 cosine",
        "seed×fold run（PASS=17，FAIL=8）",
        "箱体为run值IQR、须为1.5×IQR；标注为crossed seed×fold bootstrap的FAIL−PASS median difference 95% CI。",
        ("metrics/run_level_conflict_metrics.csv", "metrics/pass_fail_comparison.csv"),
        "pass_fail_cosine",
    ),
    FigureSpec(
        "F06",
        "06_pass_fail_mbase.png",
        "PASS/FAIL 的 all-shared base-descent margin",
        "seed×fold run（PASS=17，FAIL=8）",
        "箱体为run值IQR、须为1.5×IQR；标注为crossed seed×fold bootstrap的FAIL−PASS median difference 95% CI。",
        ("metrics/run_level_conflict_metrics.csv", "metrics/pass_fail_comparison.csv"),
        "pass_fail_mbase",
    ),
    FigureSpec(
        "F07",
        "07_pass_fail_norm_ratio.png",
        "PASS/FAIL 的 all-shared weighted gradient norm ratio",
        "seed×fold run（PASS=17，FAIL=8）",
        "箱体为run值IQR、须为1.5×IQR；标注为crossed seed×fold bootstrap的FAIL−PASS median difference及ratio 95% CI。",
        ("metrics/run_level_conflict_metrics.csv", "metrics/pass_fail_comparison.csv"),
        "pass_fail_ratio",
    ),
    FigureSpec(
        "F08",
        "08_layerwise_cosine_heatmap.png",
        "Layer-wise 梯度 cosine 热图",
        "seed×fold run×非重叠共享层（每格为8 batch中位数）",
        "描述性点估计，无额外误差条；正式定位只使用stage1–4与response projection五个非重叠组。",
        ("metrics/layer_level_conflict_metrics.csv",),
        "layer_cosine",
    ),
    FigureSpec(
        "F09",
        "09_layerwise_mbase_heatmap.png",
        "Layer-wise base-descent margin 热图",
        "seed×fold run×非重叠共享层（每格为8 batch中位数）",
        "描述性点估计，无额外误差条；M_base=1表示FTV对base一阶下降既不帮助也不削弱。",
        ("metrics/layer_level_conflict_metrics.csv",),
        "layer_mbase",
    ),
    FigureSpec(
        "F10",
        "10_seed_fold_conflict_heatmap.png",
        "Validation all-shared 的 seed×fold conflict",
        "seed×fold run（5×5完整网格）",
        "描述性点估计，无额外误差条；左图为8 batch median cosine，右图为8 batch median M_base。",
        ("metrics/run_level_conflict_metrics.csv",),
        "seed_fold",
    ),
    FigureSpec(
        "F11",
        "11_representative_loss_trajectories.png",
        "3个PASS与3个FAIL代表run的既有训练轨迹",
        "epoch（六个预注册代表run）",
        "完整已保存history的描述性轨迹，无误差条；竖线是selected epoch，不是重新训练结果。",
        ("metrics/training_history_audit.csv", "metrics/representative_runs.csv"),
        "history",
    ),
    FigureSpec(
        "F12",
        "12_conflict_vs_selected_epoch.png",
        "Validation conflict 与 selected epoch",
        "seed×fold run（n=25）",
        "散点为8 batch聚合run值；图内Spearman仅描述selected epoch与conflict的关系，不作为预注册H3 gate。",
        ("metrics/run_level_conflict_metrics.csv",),
        "selected_epoch",
    ),
    FigureSpec(
        "F13",
        "13_ftv_coverage_exposure_vs_degradation.png",
        "FTV coverage/exposure 与基础损失退化",
        "coverage为fold（n=5）；累计grounded exposure为seed×fold run（n=25）",
        "coverage相关复用crossed fold draws的percentile CI；固定配额导致常量SD时如实标为constant_input。",
        (
            "metrics/ftv_coverage_metrics.csv",
            "metrics/coverage_correlations.csv",
            "metrics/run_level_existing_metrics.csv",
            "metrics/dynamics_correlations.csv",
        ),
        "coverage",
    ),
    FigureSpec(
        "F14",
        "14_selected_to_last_conflict_change.png",
        "代表run selected→last 的 conflict变化",
        "checkpoint pair（6个代表run×train/validation）",
        "仅有selected与last两个post-hoc时间点，无误差条；不代表early/逐epoch gradient trajectory。",
        (
            "metrics/trajectory_conflict_metrics.csv",
            "metrics/trajectory_change_metrics.csv",
            "metrics/representative_runs.csv",
        ),
        "trajectory",
    ),
)


TABLE_CONTRACTS: dict[str, tuple[Sequence[str], int, tuple[str, ...]]] = {
    "configs/audit_batch_manifest.csv": (
        PUBLIC_MANIFEST_COLUMNS,
        80,
        ("fold", "split", "batch_index"),
    ),
    "metrics/source_manifest.csv": (SOURCE_MANIFEST_COLUMNS, 126, ("artifact_id",)),
    "metrics/asset_manifest.csv": (
        ASSET_MANIFEST_COLUMNS,
        31,
        ("seed_base", "fold", "checkpoint_kind"),
    ),
    "metrics/run_level_existing_metrics.csv": (
        EXISTING_METRIC_COLUMNS,
        25,
        ("seed_base", "fold"),
    ),
    "metrics/training_history_audit.csv": (
        TRAINING_HISTORY_COLUMNS,
        161,
        ("seed_base", "fold", "epoch"),
    ),
    "metrics/representative_runs.csv": (
        REPRESENTATIVE_COLUMNS,
        6,
        ("seed_base", "fold"),
    ),
    "metrics/batch_gradient_metrics.csv": (
        BATCH_GRADIENT_COLUMNS,
        2_800,
        ("seed_base", "fold", "checkpoint_kind", "split", "batch_id", "group"),
    ),
    "metrics/trajectory_batch_gradient_metrics.csv": (
        BATCH_GRADIENT_COLUMNS,
        672,
        ("seed_base", "fold", "checkpoint_kind", "split", "batch_id", "group"),
    ),
    "metrics/layer_level_conflict_metrics.csv": (
        LAYER_LEVEL_COLUMNS,
        350,
        ("seed_base", "fold", "split", "group"),
    ),
    "metrics/run_level_conflict_metrics.csv": (
        RUN_LEVEL_COLUMNS,
        50,
        ("seed_base", "fold", "split"),
    ),
    "metrics/pass_fail_comparison.csv": (
        PASS_FAIL_COLUMNS,
        84,
        ("split", "group", "metric"),
    ),
    "metrics/gradient_correlation_metrics.csv": (
        CORRELATION_COLUMNS,
        56,
        ("split", "group", "endpoint"),
    ),
    "metrics/phase_a_gain_correlations.csv": (PHASE_A_COLUMNS, 3, ("endpoint",)),
    "metrics/dynamics_correlations.csv": (DYNAMICS_COLUMNS, 8, ("endpoint",)),
    "metrics/ftv_coverage_metrics.csv": (COVERAGE_COLUMNS, 10, ("fold", "split")),
    "metrics/coverage_correlations.csv": (
        COVERAGE_CORRELATION_COLUMNS,
        4,
        ("endpoint",),
    ),
    "metrics/layer_localization_metrics.csv": (LOCALIZATION_COLUMNS, 5, ("group",)),
    "metrics/fold_signature_metrics.csv": (
        FOLD_SIGNATURE_COLUMNS,
        10,
        ("fold", "split"),
    ),
    "metrics/trajectory_conflict_metrics.csv": (
        TRAJECTORY_COLUMNS,
        168,
        ("seed_base", "fold", "checkpoint_kind", "split", "group"),
    ),
    "metrics/trajectory_change_metrics.csv": (
        TRAJECTORY_CHANGE_COLUMNS,
        84,
        ("seed_base", "fold", "split", "group"),
    ),
    "metrics/hypothesis_decision.csv": (
        HYPOTHESIS_DECISION_COLUMNS,
        4,
        ("hypothesis",),
    ),
    "metrics/figure_manifest.csv": (FIGURE_MANIFEST_COLUMNS, 14, ("figure_id",)),
}


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 非严格boolean: {value!r}")


def _root(root: str | Path) -> Path:
    resolved = Path(root).resolve()
    if resolved != AUDIT_ROOT.resolve():
        raise ValueError("交付/验收root必须是冻结audit root")
    return resolved


def _relative_file(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"manifest含非法路径: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"路径越出audit root: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(f"正式交付输入缺失: {relative}")
    return path


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root不是object: {path.name}")
    return payload


def _sha64(value: Any, name: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name}不是lowercase SHA-256")
    return text


def _analysis_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证analysis manifest与最后写入marker的三重hash绑定。"""

    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    manifest_path = _relative_file(root, "metrics/final_analysis_manifest.json")
    marker_path = _relative_file(root, "metrics/FINAL_ANALYSIS_COMPLETE.json")
    diagnosis_path = _relative_file(root, "metrics/diagnosis.json")
    source_sha = file_sha256(SOURCE_CONTRACT)
    diagnosis_sha = file_sha256(diagnosis_path)
    manifest = _json(manifest_path)
    marker = _json(marker_path)
    if (
        int(manifest.get("schema_version", -1)) != 1
        or manifest.get("status") != "complete"
        or manifest.get("contains_patient_ids") is not False
        or _sha64(
            manifest.get("source_contract_sha256"), "analysis manifest source SHA"
        )
        != source_sha
    ):
        raise ValueError("final analysis manifest header contract失败")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("final analysis manifest artifacts必须为非空list")
    registered: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("analysis artifact entry不是mapping")
        relative = str(item.get("path", ""))
        if not relative or relative in registered:
            raise ValueError("analysis artifact path缺失/重复")
        path = _relative_file(root, relative)
        if _sha64(item.get("sha256"), f"{relative}.sha256") != file_sha256(path):
            raise ValueError(f"analysis artifact SHA漂移: {relative}")
        if int(item.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"analysis artifact bytes漂移: {relative}")
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            if int(item.get("rows", -1)) != len(frame) or item.get("columns") != list(
                frame.columns
            ):
                raise ValueError(f"analysis artifact CSV rows/columns漂移: {relative}")
        registered[relative] = item
    if not REQUIRED_ANALYSIS_ARTIFACTS.issubset(registered):
        missing = sorted(REQUIRED_ANALYSIS_ARTIFACTS - set(registered))
        raise ValueError(f"analysis manifest缺正式分析artifact: {missing}")
    input_manifest_path = _relative_file(root, "metrics/analysis_input_manifest.csv")
    input_manifest = pd.read_csv(input_manifest_path)
    if (
        tuple(input_manifest.columns) != ANALYSIS_INPUT_MANIFEST_COLUMNS
        or len(input_manifest) != 9
        or input_manifest["artifact_role"].duplicated().any()
        or file_sha256(input_manifest_path)
        != str(manifest.get("analysis_input_manifest_sha256", ""))
    ):
        raise ValueError("analysis input manifest schema/hash失败")
    expected_input_paths = {
        "metrics/batch_gradient_metrics.csv",
        "metrics/trajectory_batch_gradient_metrics.csv",
        "metrics/layer_level_conflict_metrics.csv",
        "metrics/run_level_conflict_metrics.csv",
        "metrics/trajectory_conflict_metrics.csv",
        "metrics/trajectory_change_metrics.csv",
        "metrics/ftv_coverage_metrics.csv",
        "metrics/run_level_existing_metrics.csv",
        "metrics/phase_a_gain_correlations.csv",
    }
    if set(input_manifest["path"].astype(str)) != expected_input_paths:
        raise ValueError("analysis input manifest artifact set漂移")
    for row in input_manifest.itertuples(index=False):
        path = _relative_file(root, str(row.path))
        frame = pd.read_csv(path)
        if (
            file_sha256(path) != _sha64(row.sha256, f"analysis input {row.path}.sha256")
            or path.stat().st_size != int(row.bytes)
            or len(frame) != int(row.rows)
            or len(frame.columns) != int(row.column_count)
            or canonical_json_sha256(list(frame.columns))
            != _sha64(row.columns_sha256, f"analysis input {row.path}.columns_sha256")
            or _strict_bool(
                row.contains_patient_ids, f"analysis input {row.path}.privacy"
            )
        ):
            raise ValueError(f"analysis input artifact不闭环: {row.path}")
    if "payload_sha256" in manifest:
        unsigned = dict(manifest)
        digest = _sha64(unsigned.pop("payload_sha256"), "analysis manifest payload SHA")
        if canonical_json_sha256(unsigned) != digest:
            raise ValueError("analysis manifest payload digest不闭环")
    if (
        int(marker.get("schema_version", -1)) != 1
        or marker.get("status") != "complete"
        or marker.get("contains_patient_ids") is not False
        or _sha64(marker.get("final_analysis_manifest_sha256"), "marker manifest SHA")
        != file_sha256(manifest_path)
        or _sha64(marker.get("source_contract_sha256"), "marker source SHA")
        != source_sha
        or _sha64(marker.get("diagnosis_sha256"), "marker diagnosis SHA")
        != diagnosis_sha
        or re.fullmatch(r"[0-9a-f]{40}", str(marker.get("analysis_commit", ""))) is None
    ):
        raise ValueError("FINAL_ANALYSIS_COMPLETE marker绑定失败")
    if "payload_sha256" in marker:
        unsigned_marker = dict(marker)
        digest = _sha64(
            unsigned_marker.pop("payload_sha256"), "analysis marker payload SHA"
        )
        if canonical_json_sha256(unsigned_marker) != digest:
            raise ValueError("analysis marker payload digest不闭环")
    marker_mtime = marker_path.stat().st_mtime_ns
    if marker_mtime < max(
        [manifest_path.stat().st_mtime_ns, diagnosis_path.stat().st_mtime_ns]
        + [_relative_file(root, relative).stat().st_mtime_ns for relative in registered]
    ):
        raise ValueError("analysis completion marker不是最后创建的bundle成员")
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{marker['analysis_commit']}^{{commit}}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("analysis_commit不是当前repository可解析commit")
    return manifest, marker


def _read_inputs(root: Path) -> dict[str, Any]:
    manifest, marker = _analysis_gate(root)
    required = {relative for spec in FIGURE_SPECS for relative in spec.inputs} | {
        "metrics/run_level_existing_metrics.csv",
        "metrics/run_level_conflict_metrics.csv",
        "metrics/layer_level_conflict_metrics.csv",
        "metrics/pass_fail_comparison.csv",
        "metrics/gradient_correlation_metrics.csv",
        "metrics/phase_a_gain_correlations.csv",
        "metrics/dynamics_correlations.csv",
        "metrics/ftv_coverage_metrics.csv",
        "metrics/coverage_correlations.csv",
        "metrics/layer_localization_metrics.csv",
        "metrics/fold_signature_metrics.csv",
        "metrics/trajectory_conflict_metrics.csv",
        "metrics/trajectory_change_metrics.csv",
        "metrics/training_history_audit.csv",
        "metrics/representative_runs.csv",
        "metrics/hypothesis_decision.csv",
    }
    frames = {
        relative: pd.read_csv(_relative_file(root, relative))
        for relative in sorted(required)
    }
    diagnosis = _json(_relative_file(root, "metrics/diagnosis.json"))
    return {
        "frames": frames,
        "diagnosis": diagnosis,
        "analysis_manifest": manifest,
        "analysis_marker": marker,
    }


def _font() -> str:
    candidates = font_manager.findSystemFonts()
    priority = (
        "notosanscjk-regular",
        "notosanscjk",
        "sourcehansans",
        "droidsansfallback",
        "wqy",
        "simhei",
    )
    chosen: str | None = None
    for token in priority:
        matched = sorted(
            path
            for path in candidates
            if token in Path(path).stem.lower().replace("_", "")
        )
        if matched:
            chosen = matched[0]
            break
    if chosen is None:
        raise RuntimeError("未找到可渲染中文的CJK字体；拒绝生成缺字图")
    font_manager.fontManager.addfont(chosen)
    family = font_manager.FontProperties(fname=chosen).get_name()
    plt.rcParams.update(
        {
            "font.family": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFC",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return family


def _fmt(value: Any, digits: int = 3) -> str:
    if (
        value is None
        or (isinstance(value, float) and not math.isfinite(value))
        or pd.isna(value)
    ):
        return "NA"
    return f"{float(value):.{digits}f}"


def _ci(row: pd.Series) -> str:
    return f"ρ={_fmt(row.get('spearman_rho'))}，95% CI [{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}]，p_Holm={_fmt(row.get('p_holm'))}"


def _gate_colors(frame: pd.DataFrame) -> list[str]:
    return [
        PASS_COLOR if str(value) == "PASS" else FAIL_COLOR
        for value in frame["base_gate"]
    ]


def _finish(fig: plt.Figure, spec: FigureSpec, destination: Path) -> None:
    fig.suptitle(spec.title, fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.01,
        0.008,
        f"统计单位：{spec.analysis_unit}　|　误差说明：{spec.uncertainty_note}",
        fontsize=8.5,
        color="#333333",
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(destination, dpi=FIGURE_DPI, bbox_inches=None, facecolor="white")
    plt.close(fig)


def _scatter(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str) -> None:
    ax.scatter(
        frame[x].to_numpy(dtype=float) * 100,
        frame[y].to_numpy(dtype=float),
        c=_gate_colors(frame),
        s=55,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.92,
    )
    ax.axvline(
        5.0, color="#555555", linestyle=":", linewidth=1.2, label="5% safety阈值"
    )
    ax.set_xlabel("基础损失退化（%）")


def _phase_figure(
    data: Mapping[str, Any], spec: FigureSpec, destination: Path, dynamic: bool
) -> None:
    frames = data["frames"]
    existing = frames["metrics/run_level_existing_metrics.csv"]
    phase = frames["metrics/phase_a_gain_correlations.csv"]
    endpoint = (
        "degradation_vs_observed_delta_spearman"
        if dynamic
        else "degradation_vs_static_delta_spearman"
    )
    column = "delta_ftv_delta_spearman" if dynamic else "static_ftv_delta_spearman"
    row = phase.loc[phase["endpoint"].eq(endpoint)]
    if len(row) != 1:
        raise ValueError(f"Phase-A endpoint缺失: {endpoint}")
    fig, ax = plt.subplots(figsize=(12, 7.5))
    _scatter(ax, existing, "base_degradation", column)
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_ylabel(
        "observed ΔFTV Spearman增益" if dynamic else "static FTV Spearman增益"
    )
    item = row.iloc[0]
    ax.text(
        0.02,
        0.97,
        f"ρ={_fmt(item.spearman_rho)}\n95% CI [{_fmt(item.ci_low)}, {_fmt(item.ci_high)}]\nraw p={_fmt(item.p_raw_two_sided)}；Holm p={_fmt(item.p_holm)}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.legend(loc="lower left")
    _finish(fig, spec, destination)


def _gradient_scatter(
    data: Mapping[str, Any], spec: FigureSpec, destination: Path, endpoint: str
) -> None:
    run = data["frames"]["metrics/run_level_conflict_metrics.csv"]
    corr = data["frames"]["metrics/gradient_correlation_metrics.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
    for ax, split in zip(axes, SPLITS, strict=True):
        current = run.loc[run["split"].eq(split)].copy()
        _scatter(ax, current, "base_degradation", endpoint)
        match = corr.loc[
            corr["split"].eq(split)
            & corr["group"].eq("all_shared")
            & corr["endpoint"].eq(endpoint)
        ]
        if len(match) != 1:
            raise ValueError(f"gradient correlation endpoint缺失: {split}/{endpoint}")
        item = match.iloc[0]
        if endpoint == "base_descent_margin":
            ax.axhline(
                0, color=FAIL_COLOR, linestyle="--", linewidth=1, label="M_base=0"
            )
            ax.axhline(1, color="#555555", linestyle=":", linewidth=1, label="M_base=1")
        else:
            ax.axhline(0, color="#555555", linestyle=":", linewidth=1)
        ax.set_title("训练池" if split == "train" else "验证池（primary）")
        ax.text(
            0.02,
            0.97,
            _ci(item),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel(
        "8-batch median M_base"
        if endpoint == "base_descent_margin"
        else "8-batch median cosine"
    )
    _finish(fig, spec, destination)


def _box_figure(
    data: Mapping[str, Any], spec: FigureSpec, destination: Path, metric: str
) -> None:
    run = data["frames"]["metrics/run_level_conflict_metrics.csv"]
    comparison = data["frames"]["metrics/pass_fail_comparison.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
    for ax, split in zip(axes, SPLITS, strict=True):
        current = run.loc[run["split"].eq(split)]
        values = [
            current.loc[current["base_gate"].eq(gate), metric].to_numpy(dtype=float)
            for gate in ("PASS", "FAIL")
        ]
        box = ax.boxplot(
            values,
            tick_labels=["PASS (17)", "FAIL (8)"],
            patch_artist=True,
            widths=0.55,
        )
        for patch, color in zip(box["boxes"], (PASS_COLOR, FAIL_COLOR), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for position, (group_values, color) in enumerate(
            zip(values, (PASS_COLOR, FAIL_COLOR), strict=True), start=1
        ):
            jitter = np.linspace(-0.10, 0.10, len(group_values))
            ax.scatter(
                position + jitter,
                group_values,
                color=color,
                edgecolors="white",
                s=38,
                zorder=3,
            )
        match = comparison.loc[
            comparison["split"].eq(split)
            & comparison["group"].eq("all_shared")
            & comparison["metric"].eq(metric)
        ]
        if len(match) != 1:
            raise ValueError(f"PASS/FAIL endpoint缺失: {split}/{metric}")
        item = match.iloc[0]
        note = (
            f"FAIL−PASS median={_fmt(item.median_difference_fail_minus_pass)}\n"
            f"95% CI [{_fmt(item.median_ci_low)}, {_fmt(item.median_ci_high)}]\n"
            f"permutation p={_fmt(item.permutation_p_raw_two_sided)}"
        )
        if metric == "weighted_gradient_norm_ratio":
            note += f"\nFAIL/PASS ratio={_fmt(item.fail_over_pass_median_ratio)}"
        ax.text(
            0.02,
            0.97,
            note,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.set_title("训练池" if split == "train" else "验证池（primary）")
        if metric == "base_descent_margin":
            ax.axhline(0, color=FAIL_COLOR, linestyle="--", linewidth=1)
            ax.axhline(1, color="#555555", linestyle=":", linewidth=1)
        elif metric == "gradient_cosine":
            ax.axhline(0, color="#555555", linestyle=":", linewidth=1)
    labels = {
        "gradient_cosine": "8-batch median cosine",
        "base_descent_margin": "8-batch median M_base",
        "weighted_gradient_norm_ratio": "8-batch median ||0.25g_FTV||/||g_base||",
    }
    axes[0].set_ylabel(labels[metric])
    _finish(fig, spec, destination)


_NONOVERLAP_GROUPS = (
    "encoder_stage_1",
    "encoder_stage_2",
    "encoder_stage_3",
    "encoder_stage_4",
    "response_projection",
)
_GROUP_LABELS = ("Stage 1", "Stage 2", "Stage 3", "Stage 4", "Response projection")


def _heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    rows: Sequence[str],
    columns: Sequence[str],
    *,
    center: float,
    label: str,
) -> None:
    span = max(
        abs(float(np.nanmin(matrix)) - center),
        abs(float(np.nanmax(matrix)) - center),
        1e-6,
    )
    image = ax.imshow(
        matrix, aspect="auto", cmap="coolwarm_r", vmin=center - span, vmax=center + span
    )
    ax.set_xticks(range(len(columns)), columns, rotation=28, ha="right")
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label=label)


def _layer_heatmap(
    data: Mapping[str, Any], spec: FigureSpec, destination: Path, metric: str
) -> None:
    layer = data["frames"]["metrics/layer_level_conflict_metrics.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    ordered_runs = [(seed, fold) for seed in SEED_BASES for fold in FOLDS]
    row_labels = [f"S{seed}/F{fold}" for seed, fold in ordered_runs]
    for ax, split in zip(axes, SPLITS, strict=True):
        current = layer.loc[
            layer["split"].eq(split) & layer["group"].isin(_NONOVERLAP_GROUPS)
        ]
        pivot = current.pivot(
            index=["seed_base", "fold"], columns="group", values=metric
        ).reindex(
            index=pd.MultiIndex.from_tuples(ordered_runs, names=["seed_base", "fold"]),
            columns=_NONOVERLAP_GROUPS,
        )
        matrix = pivot.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("layer heatmap grid nonfinite")
        _heatmap(
            ax,
            matrix,
            row_labels,
            _GROUP_LABELS,
            center=0 if metric == "gradient_cosine" else 1,
            label=metric,
        )
        ax.set_title("训练池" if split == "train" else "验证池（primary）")
    _finish(fig, spec, destination)


def _seed_fold(data: Mapping[str, Any], spec: FigureSpec, destination: Path) -> None:
    run = data["frames"]["metrics/run_level_conflict_metrics.csv"]
    current = run.loc[run["split"].eq("validation")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    for ax, metric, center, title in (
        (axes[0], "gradient_cosine", 0.0, "8-batch median cosine"),
        (axes[1], "base_descent_margin", 1.0, "8-batch median M_base"),
    ):
        pivot = current.pivot(index="seed_base", columns="fold", values=metric).reindex(
            index=SEED_BASES, columns=FOLDS
        )
        matrix = pivot.to_numpy(dtype=float)
        _heatmap(
            ax,
            matrix,
            [str(seed) for seed in SEED_BASES],
            [f"Fold {fold}" for fold in FOLDS],
            center=center,
            label=metric,
        )
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("Training seed base")
        ax.set_title(title)
    _finish(fig, spec, destination)


def _history(data: Mapping[str, Any], spec: FigureSpec, destination: Path) -> None:
    history = data["frames"]["metrics/training_history_audit.csv"]
    representatives = data["frames"]["metrics/representative_runs.csv"]
    reps = representatives.sort_values(
        ["base_gate", "base_degradation"], ascending=[False, True]
    ).reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False, sharey=False)
    for ax, rep in zip(axes.ravel(), reps.itertuples(index=False), strict=True):
        current = history.loc[
            history["seed_base"].eq(rep.seed_base) & history["fold"].eq(rep.fold)
        ].sort_values("epoch")
        if current.empty:
            raise ValueError("representative history缺失")
        ax.plot(
            current["epoch"],
            current["train_total_loss"],
            marker="o",
            label="train total",
        )
        ax.plot(
            current["epoch"],
            current["train_base_loss"],
            marker="s",
            label="train full base",
        )
        ax.plot(
            current["epoch"],
            current["train_ftv_loss"],
            marker="^",
            label="train raw FTV",
        )
        ax.axvline(
            int(rep.selected_epoch),
            color="#333333",
            linestyle="--",
            linewidth=1,
            label="selected epoch",
        )
        color = PASS_COLOR if str(rep.base_gate) == "PASS" else FAIL_COLOR
        ax.set_title(
            f"{rep.base_gate} | seed={int(rep.seed_base)}, fold={int(rep.fold)}",
            color=color,
            fontweight="bold",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
    _finish(fig, spec, destination)


def _selected_epoch(
    data: Mapping[str, Any], spec: FigureSpec, destination: Path
) -> None:
    run = data["frames"]["metrics/run_level_conflict_metrics.csv"]
    current = run.loc[run["split"].eq("validation")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    for ax, metric, ylabel in (
        (axes[0], "gradient_cosine", "8-batch median cosine"),
        (axes[1], "base_descent_margin", "8-batch median M_base"),
    ):
        ax.scatter(
            current["selected_epoch"],
            current[metric],
            c=_gate_colors(current),
            s=58,
            edgecolors="white",
        )
        point = pd.Series(current["selected_epoch"]).corr(
            pd.Series(current[metric]), method="spearman"
        )
        ax.text(
            0.02,
            0.97,
            f"描述性Spearman ρ={_fmt(point)}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.set_xlabel("Selected epoch")
        ax.set_ylabel(ylabel)
    _finish(fig, spec, destination)


def _coverage(data: Mapping[str, Any], spec: FigureSpec, destination: Path) -> None:
    coverage = data["frames"]["metrics/ftv_coverage_metrics.csv"]
    existing = data["frames"]["metrics/run_level_existing_metrics.csv"]
    corr = data["frames"]["metrics/coverage_correlations.csv"]
    dynamics = data["frames"]["metrics/dynamics_correlations.csv"]
    fold_degradation = (
        existing.groupby("fold")["base_degradation"].median().reindex(FOLDS) * 100
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, split, endpoint in (
        (axes[0, 0], "train", "train_pool_ftv_proportion"),
        (axes[0, 1], "validation", "validation_pool_ftv_proportion"),
    ):
        current = (
            coverage.loc[coverage["split"].eq(split)].set_index("fold").reindex(FOLDS)
        )
        ax.scatter(
            current["pool_ftv_proportion"] * 100,
            fold_degradation,
            color=TRAIN_COLOR if split == "train" else VALIDATION_COLOR,
            s=70,
        )
        match = corr.loc[corr["endpoint"].eq(endpoint)]
        if len(match) != 1:
            raise ValueError(f"coverage endpoint缺失: {endpoint}")
        row = match.iloc[0]
        ax.text(
            0.02,
            0.97,
            f"fold-level ρ={_fmt(row.spearman_rho)}\n95% CI [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}]\nstatus={row.status}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.set_xlabel(
            f"{'Train' if split == 'train' else 'Validation'} pool FTV available（%）"
        )
        ax.set_ylabel("Fold median基础损失退化（%）")
    axes[1, 0].scatter(
        existing["cumulative_grounded_exposure_to_selected"],
        existing["base_degradation"] * 100,
        c=_gate_colors(existing),
        s=50,
        edgecolors="white",
    )
    dyn = dynamics.loc[dynamics["endpoint"].eq("cumulative_grounded_exposure")]
    if len(dyn) != 1:
        raise ValueError("cumulative grounded dynamics endpoint缺失")
    item = dyn.iloc[0]
    axes[1, 0].text(
        0.02,
        0.97,
        f"oriented ρ={_fmt(item.spearman_rho_oriented)}\nHolm p={_fmt(item.p_holm)}",
        transform=axes[1, 0].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axes[1, 0].set_xlabel("Selected前累计grounded exposure")
    axes[1, 0].set_ylabel("基础损失退化（%）")
    width = 0.34
    train_sd = (
        coverage.loc[coverage["split"].eq("train")]
        .set_index("fold")
        .reindex(FOLDS)["batch_ftv_sd"]
    )
    val_sd = (
        coverage.loc[coverage["split"].eq("validation")]
        .set_index("fold")
        .reindex(FOLDS)["batch_ftv_sd"]
    )
    x = np.arange(5)
    axes[1, 1].bar(x - width / 2, train_sd, width, label="Train", color=TRAIN_COLOR)
    axes[1, 1].bar(
        x + width / 2, val_sd, width, label="Validation", color=VALIDATION_COLOR
    )
    axes[1, 1].set_xticks(x, [f"F{fold}" for fold in FOLDS])
    axes[1, 1].set_ylabel("8 batches FTV-count sample SD")
    axes[1, 1].legend()
    _finish(fig, spec, destination)


def _trajectory(data: Mapping[str, Any], spec: FigureSpec, destination: Path) -> None:
    trajectory = data["frames"]["metrics/trajectory_conflict_metrics.csv"]
    representatives = data["frames"]["metrics/representative_runs.csv"]
    rep_gate = representatives.set_index(["seed_base", "fold"])["base_gate"].to_dict()
    current = trajectory.loc[trajectory["group"].eq("all_shared")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    for ax, metric, ylabel in (
        (axes[0], "gradient_cosine", "8-batch median cosine"),
        (axes[1], "base_descent_margin", "8-batch median M_base"),
    ):
        for index, ((seed, fold, split), pair) in enumerate(
            current.groupby(["seed_base", "fold", "split"], sort=True)
        ):
            pair = pair.set_index("checkpoint_kind").reindex(["selected", "last"])
            if pair[metric].isna().any():
                raise ValueError("trajectory selected/last pair缺失")
            gate = str(rep_gate[(seed, fold)])
            color = PASS_COLOR if gate == "PASS" else FAIL_COLOR
            linestyle = "-" if split == "validation" else "--"
            offset = (-0.03 if split == "train" else 0.03) + (index % 3 - 1) * 0.006
            ax.plot(
                [0 + offset, 1 + offset],
                pair[metric],
                color=color,
                linestyle=linestyle,
                marker="o",
                alpha=0.8,
            )
        ax.set_xticks([0, 1], ["Selected", "Last"])
        ax.set_ylabel(ylabel)
        if metric == "base_descent_margin":
            ax.axhline(0, color=FAIL_COLOR, linestyle=":", linewidth=1)
            ax.axhline(1, color="#555555", linestyle=":", linewidth=1)
        else:
            ax.axhline(0, color="#555555", linestyle=":", linewidth=1)
    axes[0].text(
        0.02,
        0.03,
        "颜色：PASS/FAIL；实线：validation；虚线：train",
        transform=axes[0].transAxes,
        fontsize=8,
    )
    _finish(fig, spec, destination)


_RENDERERS: dict[str, Callable[[Mapping[str, Any], FigureSpec, Path], None]] = {
    "phase_static": lambda data, spec, path: _phase_figure(data, spec, path, False),
    "phase_dynamic": lambda data, spec, path: _phase_figure(data, spec, path, True),
    "gradient_scatter_cosine": lambda data, spec, path: _gradient_scatter(
        data, spec, path, "gradient_cosine"
    ),
    "gradient_scatter_mbase": lambda data, spec, path: _gradient_scatter(
        data, spec, path, "base_descent_margin"
    ),
    "pass_fail_cosine": lambda data, spec, path: _box_figure(
        data, spec, path, "gradient_cosine"
    ),
    "pass_fail_mbase": lambda data, spec, path: _box_figure(
        data, spec, path, "base_descent_margin"
    ),
    "pass_fail_ratio": lambda data, spec, path: _box_figure(
        data, spec, path, "weighted_gradient_norm_ratio"
    ),
    "layer_cosine": lambda data, spec, path: _layer_heatmap(
        data, spec, path, "gradient_cosine"
    ),
    "layer_mbase": lambda data, spec, path: _layer_heatmap(
        data, spec, path, "base_descent_margin"
    ),
    "seed_fold": _seed_fold,
    "history": _history,
    "selected_epoch": _selected_epoch,
    "coverage": _coverage,
    "trajectory": _trajectory,
}


def _input_hashes(root: Path, inputs: Sequence[str]) -> dict[str, str]:
    return {
        relative: file_sha256(_relative_file(root, relative)) for relative in inputs
    }


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if tuple(row) != tuple(columns):
                raise ValueError("temporary CSV row schema漂移")
            writer.writerow(row)


def _validate_diagnosis(data: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    diagnosis = data["diagnosis"]
    decision = data["frames"]["metrics/hypothesis_decision.csv"]
    selected = str(diagnosis.get("selected_hypothesis", ""))
    first = diagnosis.get("first_recommendation")
    second = diagnosis.get("second_recommendation")
    if (
        int(diagnosis.get("schema_version", -1)) != 1
        or diagnosis.get("core_eligible") is not True
        or selected not in {"H1", "H2", "H3", "H4"}
        or not isinstance(first, str)
        or not first
        or not isinstance(second, str)
        or not second
        or first == second
        or diagnosis.get("fix_executed") is not False
        or diagnosis.get("contains_patient_ids") is not False
    ):
        raise ValueError("diagnosis唯一性/数据质量/停止条件失败")
    selected_flags = decision["selected"].map(
        lambda value: _strict_bool(value, "hypothesis.selected")
    )
    if (
        tuple(decision.columns) != HYPOTHESIS_DECISION_COLUMNS
        or len(decision) != 4
        or set(decision["hypothesis"]) != {"H1", "H2", "H3", "H4"}
        or int(selected_flags.sum()) != 1
        or str(decision.loc[selected_flags, "hypothesis"].iloc[0]) != selected
        or set(decision["selected_hypothesis"].astype(str)) != {selected}
    ):
        raise ValueError("hypothesis decision CSV与diagnosis不闭环")
    return diagnosis, decision


def _match(frame: pd.DataFrame, **conditions: Any) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    matched = frame.loc[mask]
    if len(matched) != 1:
        raise ValueError(f"结果行不唯一: {conditions}")
    return matched.iloc[0]


def _report_text(root: Path, data: Mapping[str, Any]) -> str:
    frames = data["frames"]
    diagnosis, decision = _validate_diagnosis(data)
    existing = frames["metrics/run_level_existing_metrics.csv"].sort_values(
        ["seed_base", "fold"]
    )
    run = frames["metrics/run_level_conflict_metrics.csv"]
    phase = frames["metrics/phase_a_gain_correlations.csv"]
    comparison = frames["metrics/pass_fail_comparison.csv"]
    correlations = frames["metrics/gradient_correlation_metrics.csv"]
    dynamics = frames["metrics/dynamics_correlations.csv"]
    coverage = frames["metrics/ftv_coverage_metrics.csv"]
    coverage_corr = frames["metrics/coverage_correlations.csv"]
    localization = frames["metrics/layer_localization_metrics.csv"].sort_values(
        "localization_rank"
    )
    folds = frames["metrics/fold_signature_metrics.csv"]
    trajectory_change = frames["metrics/trajectory_change_metrics.csv"]
    representatives = frames["metrics/representative_runs.csv"]

    validation = run.loc[run["split"].eq("validation")].copy()
    train = run.loc[run["split"].eq("train")].copy()
    if len(validation) != 25 or len(train) != 25:
        raise ValueError("报告输入run split不是25/25")
    pass_count = int(existing["base_gate"].eq("PASS").sum())
    fail_count = int(existing["base_gate"].eq("FAIL").sum())
    if (pass_count, fail_count) != (17, 8):
        raise ValueError("报告输入PASS/FAIL不是17/8")

    def pf(metric: str, split: str = "validation") -> pd.Series:
        return _match(comparison, split=split, group="all_shared", metric=metric)

    def corr(endpoint: str, split: str = "validation") -> pd.Series:
        return _match(correlations, split=split, group="all_shared", endpoint=endpoint)

    cosine_pf = pf("gradient_cosine")
    neg_pf = pf("negative_fraction")
    ratio_pf = pf("weighted_gradient_norm_ratio")
    mbase_pf = pf("base_descent_margin")
    mfail_pf = pf("base_descent_failure_fraction")
    cosine_corr = corr("gradient_cosine")
    mbase_corr = corr("base_descent_margin")
    ratio_corr = corr("weighted_gradient_norm_ratio")
    static_phase = _match(phase, endpoint="degradation_vs_static_delta_spearman")
    observed_phase = _match(phase, endpoint="degradation_vs_observed_delta_spearman")
    observed_r2_phase = _match(phase, endpoint="degradation_vs_observed_delta_r2")
    fold3 = _match(folds, fold=3, split="validation")
    selected_hypothesis = str(diagnosis["selected_hypothesis"])
    first_recommendation = str(diagnosis["first_recommendation"])
    second_recommendation = str(diagnosis["second_recommendation"])

    validation_negative = float(validation["negative_fraction"].mean())
    train_negative = float(train["negative_fraction"].mean())
    validation_mfail = float(validation["base_descent_failure_fraction"].mean())
    train_mfail = float(train["base_descent_failure_fraction"].mean())
    top = localization.iloc[0]
    second_layer = localization.iloc[1]
    widespread = _strict_bool(top["widespread_conflict"], "widespread_conflict")
    layer_statement = (
        f"五个非重叠候选层中有{int(top.widespread_layer_count)}层越过方向阈值，因此属于广泛冲突；"
        if widespread
        else "未达到至少四层同时越阈的广泛冲突定义；"
    )
    layer_statement += (
        f"定位rank前两位为 `{top.group}`（score={_fmt(top.localization_score)}）与"
        f" `{second_layer.group}`（score={_fmt(second_layer.localization_score)}）。"
    )
    direction_count = sum(
        (
            float(cosine_pf.median_difference_fail_minus_pass) < 0,
            float(neg_pf.mean_difference_fail_minus_pass) > 0,
            float(mbase_pf.median_difference_fail_minus_pass) < 0,
            float(mfail_pf.mean_difference_fail_minus_pass) > 0,
        )
    )
    phase_tradeoff = (
        f"static gain: ρ={_fmt(static_phase.spearman_rho)}, Holm p={_fmt(static_phase.p_holm)}；"
        f"observed ΔFTV Spearman gain: ρ={_fmt(observed_phase.spearman_rho)}, Holm p={_fmt(observed_phase.p_holm)}；"
        f"observed ΔFTV R² gain: ρ={_fmt(observed_r2_phase.spearman_rho)}, Holm p={_fmt(observed_r2_phase.p_holm)}。"
    )
    trajectory_val = trajectory_change.loc[
        trajectory_change["split"].eq("validation")
        & trajectory_change["group"].eq("all_shared")
    ]
    if len(trajectory_val) != 6:
        raise ValueError("报告trajectory代表run不是6")
    trajectory_note = (
        f"六个代表run从selected到last的validation cosine变化中位数为"
        f"{_fmt(trajectory_val['last_minus_selected_gradient_cosine'].median())}，"
        f"M_base变化中位数为{_fmt(trajectory_val['last_minus_selected_base_descent_margin'].median())}。"
    )
    strongest_dynamic = (
        dynamics.assign(
            magnitude=pd.to_numeric(
                dynamics["spearman_rho_oriented"], errors="coerce"
            ).abs()
        )
        .sort_values("magnitude", ascending=False)
        .iloc[0]
    )
    constant_coverage = coverage_corr.loc[
        coverage_corr["status"].eq("constant_input"), "endpoint"
    ].tolist()
    coverage_note = (
        "；固定配额使下列端点为constant_input："
        + "、".join(map(str, constant_coverage))
        if constant_coverage
        else "；四个coverage端点均有可计算变异"
    )

    q1 = (
        f"Validation固定batch中，run-level负cosine比例的均值为{_fmt(validation_negative)}，"
        f"run median cosine的总体中位数为{_fmt(validation['gradient_cosine'].median())}；"
        "这量化了冲突频率，不把接近零的方向自动称为因果不兼容。"
    )
    q2 = (
        f"FAIL相对PASS的四个方向性端点有{direction_count}/4朝更严重冲突方向："
        f"D_cos={_fmt(cosine_pf.median_difference_fail_minus_pass)}，"
        f"D_neg={_fmt(neg_pf.mean_difference_fail_minus_pass)}，"
        f"D_mbase={_fmt(mbase_pf.median_difference_fail_minus_pass)}，"
        f"D_mfail={_fmt(mfail_pf.mean_difference_fail_minus_pass)}。"
    )
    q3 = (
        f"Validation degradation相关为 cosine ρ={_fmt(cosine_corr.spearman_rho)}、"
        f"M_base ρ={_fmt(mbase_corr.spearman_rho)}；对应Holm p分别为"
        f"{_fmt(cosine_corr.p_holm)}与{_fmt(mbase_corr.p_holm)}。"
    )
    q4 = (
        "既有grounding benefit与degradation的post-hoc关联为："
        + phase_tradeoff
        + " 这些是trade-off的间接关联，不是优化干预。"
    )
    q5 = layer_statement
    q6 = (
        f"Validation FAIL−PASS norm-ratio差为{_fmt(ratio_pf.median_difference_fail_minus_pass)}，"
        f"FAIL/PASS median ratio={_fmt(ratio_pf.fail_over_pass_median_ratio)}，"
        f"degradation相关ρ={_fmt(ratio_corr.spearman_rho)}（Holm p={_fmt(ratio_corr.p_holm)}）。"
    )
    q7 = (
        f"M_base<0确实发生的比例：validation全部固定batch={_fmt(validation_mfail)}，"
        f"train={_fmt(train_mfail)}；validation PASS均值={_fmt(mfail_pf.pass_mean)}，"
        f"FAIL均值={_fmt(mfail_pf.fail_mean)}。"
    )
    q8 = (
        f"Fold3 validation cosine是否严格最差={fold3.fold3_strictly_worst_cosine}，"
        f"M_base是否严格最差={fold3.fold3_strictly_worst_mbase}，"
        f"最终special signature={fold3.fold3_special_signature}。"
    )
    q9 = f"预注册层级判定唯一选择 **{selected_hypothesis}**；该标签来自机械H1→H2→H3→H4规则。"
    q10 = f"唯一第一优先级：**{first_recommendation}**；第二优先级：**{second_recommendation}**。本轮未执行任何修复。"

    figure_links = {
        spec.figure_id: f"[图{int(spec.figure_id[1:])}](../figures/{spec.filename})"
        for spec in FIGURE_SPECS
    }
    lines = [
        "# Grounding–JEPA Conflict Audit 最终报告",
        "",
        "> 本报告由冻结机器表和 `diagnosis.json` 机械生成；数值未手工转录。所有相关与检验的正式推断单位均为seed×fold run，而非batch。",
        "",
        "## 1. 科学问题",
        "",
        "本审计只回答：G3的base-loss instability是否与Direct FTV grounding和JEPA完整base objective在shared image representation上的梯度关系一致。它是诊断实验，不是方法改进实验。",
        "",
        "## 2. 现有 multi-seed evidence",
        "",
        f"固定5个training seed×5 folds共25个run；state-loss safety gate得到{pass_count} PASS/{fail_count} FAIL。base degradation中位数={_fmt(existing['base_degradation'].median())}，范围[{_fmt(existing['base_degradation'].min())}, {_fmt(existing['base_degradation'].max())}]。既有static FTV gain中位数={_fmt(existing['static_ftv_delta_spearman'].median())}，observed ΔFTV Spearman gain中位数={_fmt(existing['delta_ftv_delta_spearman'].median())}。",
        "",
        f"关联图：{figure_links['F01']}、{figure_links['F02']}。",
        "",
        "## 3. 为什么怀疑 gradient conflict",
        "",
        "Grounding representation gain已在现有multi-seed结果中重复出现，但同一机制的25格仍有8格越过5% state-loss degradation阈值。因此需区分：shared-gradient方向冲突、scale imbalance、checkpoint/training dynamics，或未被简单端点解释的seed×fold stochastic interaction。",
        "",
        "## 4. Shared computational graph",
        "",
        "DCE7经四stage 3-D encoder、GAP和response projection得到observed response state。JEPA分支使用projector/causal transition并形成完整 `L_base = state_loss + 0.09×SIGReg`；FTV分支使用线性head形成raw patient-mean SmoothL1。真实shared参数仅为encoder（892,672）与response projection（25,152），all-shared共917,824。projector/transition、FTV head与EMA target均不进入cosine。",
        "",
        "**安全指标与梯度目标不可混同：** 17/8 gate及base degradation使用validation `state_loss`；本审计的 `g_base` 则是训练定义的完整base objective（state loss加SIGReg）。",
        "",
        "## 5. Audit batch protocol",
        "",
        f"每fold分别固定8个train与8个validation batch，batch size=32，每批至少8名FTV available患者；同fold五个training seed复用完全相同composition。公开manifest共80行，不含患者标识。Validation池跨batch均衡复用是预注册设计，batch从未当作独立推断样本。",
        "",
        "每个checkpoint/batch分别执行full base与raw FTV的独立forward/backward，固定随机流，不创建optimizer且不step。controlled `model.train()`图保留transition dropout，但并不重放原训练RNG。",
        "",
        "## 6. Overall gradient cosine",
        "",
        q1,
        "",
        f"Train负cosine比例均值={_fmt(train_negative)}。degradation关联见{figure_links['F03']}，PASS/FAIL分布见{figure_links['F05']}。Train与validation是同25个模型上的composition sensitivity，不是两个独立样本。",
        "",
        "## 7. Gradient norm ratio",
        "",
        q6,
        "",
        f"分布见{figure_links['F07']}。R是 `||0.25g_FTV||/||g_base||`，它描述一阶梯度scale；即使R较大，也不能单独证明高阶curvature机制。",
        "",
        "## 8. Base-descent margin",
        "",
        q7,
        "",
        f"`M_base<0`表示联合梯度的一阶方向会增加完整base objective；`0<M_base<1`表示仍下降但被削弱。关联与分布见{figure_links['F04']}、{figure_links['F06']}。该完整objective证据不同于state-loss safety gate。",
        "",
        "## 9. PASS vs FAIL",
        "",
        q2,
        "",
        f"Validation cosine permutation p={_fmt(cosine_pf.permutation_p_raw_two_sided)}，M_base permutation p={_fmt(mbase_pf.permutation_p_raw_two_sided)}，norm-ratio permutation p={_fmt(ratio_pf.permutation_p_raw_two_sided)}。PASS/FAIL差值CI来自metric与gate同步抽cell的crossed seed×fold bootstrap，而不是把17/8两组独立重抽。这些p值属于预注册小样本诊断，不称confirmatory significance。",
        "",
        "## 10. Layer-wise localization",
        "",
        q5,
        "",
        f"Layer热图见{figure_links['F08']}、{figure_links['F09']}；seed×fold all-shared对照见{figure_links['F10']}。`encoder_overall`与`all_shared`因重叠只作描述，不作为独立定位候选。",
        "",
        "## 11. Base degradation vs grounding benefit",
        "",
        q4,
        "",
        "关联来自同一批既有25个run，不能据此推断“更强grounding导致base损伤”。",
        "",
        "## 12. Checkpoint/training dynamics",
        "",
        f"八个预注册dynamics端点中，|oriented ρ|最大的端点为 `{strongest_dynamic.endpoint}`：ρ={_fmt(strongest_dynamic.spearman_rho_oriented)}，Holm p={_fmt(strongest_dynamic.p_holm)}。代表history见{figure_links['F11']}，selected epoch与conflict的描述性关系见{figure_links['F12']}。",
        "",
        trajectory_note
        + f" 配对图见{figure_links['F14']}。当前资产只有六个代表run的selected→last两点；fallback与selected同state，不是独立时间点。没有early/逐epoch checkpoint，因此不支持epoch-level gradient trajectory。",
        "",
        "原训练没有保存足以bit-exact恢复minibatch顺序、dropout mask、SIGReg direction与DataLoader RNG的状态；不得把本controlled post-hoc图解释成原训练轨迹重放。",
        "",
        "## 13. FTV sample exposure",
        "",
        f"Train pool FTV比例跨fold范围[{_fmt(coverage.loc[coverage['split'].eq('train'), 'pool_ftv_proportion'].min())}, {_fmt(coverage.loc[coverage['split'].eq('train'), 'pool_ftv_proportion'].max())}]；validation范围[{_fmt(coverage.loc[coverage['split'].eq('validation'), 'pool_ftv_proportion'].min())}, {_fmt(coverage.loc[coverage['split'].eq('validation'), 'pool_ftv_proportion'].max())}]{coverage_note}。见{figure_links['F13']}。Coverage关联单位是fold（n=5），不把五个seed复制成25个独立coverage观察。",
        "",
        "## 14. Competing hypotheses 判断",
        "",
        f"预注册层级唯一选择 **{selected_hypothesis}**。H4若被选中只表示未达到H1–H3预注册证据，不证明stochastic interaction的因果机制。",
        "",
        "### 十个必须回答的问题",
        "",
        "| 编号 | 机械回答 |",
        "|---|---|",
        f"| Q1 Shared gradient是否经常冲突？ | {q1} |",
        f"| Q2 FAIL是否更严重？ | {q2} |",
        f"| Q3 degradation是否定量相关？ | {q3} |",
        f"| Q4 grounding越强是否损伤越重？ | {q4} |",
        f"| Q5 冲突定位在哪里？ | {q5} |",
        f"| Q6 FTV gradient是否过强？ | {q6} |",
        f"| Q7 M_base<0是否发生？ | {q7} |",
        f"| Q8 Fold3是否特殊？ | {q8} |",
        f"| Q9 最符合哪个hypothesis？ | {q9} |",
        f"| Q10 下一步测试什么？ | {q10} |",
        "",
        "## 15. 下一阶段推荐",
        "",
        q10,
        "",
        "推荐严格来自所选H1–H4的冻结一对一映射。本轮没有执行PCGrad、gradient normalization、grounding warm-up、two-stage、checkpoint averaging、adaptive lambda或任何其他修复。",
        "",
        "### 证据边界",
        "",
        "本审计同时包含：(1) 既有grounding gain与degradation的post-hoc association；(2) 固定checkpoint上full base与raw FTV的objective-level gradient geometry；(3) selected→last有限trajectory。三者都不是随机化优化干预。因而报告定位的是与哪类机制一致，而不是证明FTV grounding对JEPA损伤的因果效应。",
        "",
    ]
    text = "\n".join(lines)
    headings = re.findall(r"^## (\d+)\.", text, flags=re.MULTILINE)
    if headings != [str(index) for index in range(1, 16)]:
        raise AssertionError("最终报告不是固定15节")
    if any(f"Q{index} " not in text for index in range(1, 11)):
        raise AssertionError("最终报告未逐条回答十问")
    for spec in FIGURE_SPECS:
        if f"../figures/{spec.filename}" not in text:
            raise AssertionError(f"最终报告未引用图: {spec.figure_id}")
    return text


def _prepare_deliverables(
    root: Path, temporary_root: Path
) -> tuple[list[dict[str, Any]], str]:
    data = _read_inputs(root)
    family = _font()
    figure_dir = temporary_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in FIGURE_SPECS:
        renderer = _RENDERERS.get(spec.renderer)
        if renderer is None:
            raise KeyError(f"figure renderer未注册: {spec.renderer}")
        destination = figure_dir / spec.filename
        renderer(data, spec, destination)
        if not destination.is_file():
            raise RuntimeError(f"figure未生成: {spec.filename}")
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            width, height = image.size
            dpi_info = image.info.get("dpi", (FIGURE_DPI, FIGURE_DPI))
            observed_dpi = int(round(min(float(dpi_info[0]), float(dpi_info[1]))))
        if width < MIN_FIGURE_WIDTH or height < MIN_FIGURE_HEIGHT or observed_dpi < 200:
            raise ValueError(
                f"figure分辨率不足: {spec.filename}: {width}×{height}@{observed_dpi}"
            )
        hashes = _input_hashes(root, spec.inputs)
        row = {
            "schema_version": 1,
            "figure_id": spec.figure_id,
            "filename": f"figures/{spec.filename}",
            "title": spec.title,
            "analysis_unit": spec.analysis_unit,
            "uncertainty_note": spec.uncertainty_note,
            "input_files_json": json.dumps(
                hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "input_sha256": canonical_json_sha256(hashes),
            "image_sha256": file_sha256(destination),
            "width_pixels": width,
            "height_pixels": height,
            "dpi": observed_dpi,
            "font_family": family,
            "contains_patient_ids": False,
        }
        if tuple(row) != FIGURE_MANIFEST_COLUMNS:
            raise AssertionError("figure manifest row schema漂移")
        rows.append(row)
    report = _report_text(root, data)
    return rows, report


def make_deliverables(root: str | Path = AUDIT_ROOT) -> dict[str, Any]:
    """在完整analysis marker之后一次性生成14图、figure manifest与15节中文报告。"""

    root = _root(root)
    destinations = [root / "figures" / spec.filename for spec in FIGURE_SPECS] + [
        FIGURE_MANIFEST,
        FINAL_REPORT,
    ]
    present = [str(path.relative_to(root)) for path in destinations if path.exists()]
    if present:
        raise FileExistsError(f"拒绝覆盖已有交付物: {present}")
    temporary = Path(tempfile.mkdtemp(prefix=".delivery.", dir=root))
    try:
        rows, report = _prepare_deliverables(root, temporary)
        manifest_temp = temporary / "metrics" / "figure_manifest.csv"
        report_temp = temporary / "reports" / "final_report.md"
        _write_csv(manifest_temp, rows, FIGURE_MANIFEST_COLUMNS)
        report_temp.parent.mkdir(parents=True, exist_ok=True)
        report_temp.write_text(report, encoding="utf-8")
        # 所有临时产物已完成后才移动到正式位置；marker本身永不由交付器修改。
        for spec in FIGURE_SPECS:
            destination = root / "figures" / spec.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(temporary / "figures" / spec.filename, destination)
        FIGURE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
        os.link(manifest_temp, FIGURE_MANIFEST)
        os.link(report_temp, FINAL_REPORT)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "status": "ok",
        "figures": len(FIGURE_SPECS),
        "figure_manifest_sha256": file_sha256(FIGURE_MANIFEST),
        "final_report_sha256": file_sha256(FINAL_REPORT),
    }


def _read_contract_tables(root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for relative, (columns, rows, keys) in TABLE_CONTRACTS.items():
        path = _relative_file(root, relative)
        frame = pd.read_csv(
            path,
            keep_default_na=(
                False if relative == "metrics/source_manifest.csv" else True
            ),
        )
        if tuple(frame.columns) != tuple(columns):
            raise ValueError(f"验收CSV exact schema漂移: {relative}")
        if len(frame) != rows or frame.duplicated(list(keys)).any():
            raise ValueError(f"验收CSV row/key失败: {relative}")
        frames[relative] = frame
    return frames


def _expected_grid_checks(frames: Mapping[str, pd.DataFrame]) -> None:
    run_keys = {(seed, fold) for seed in SEED_BASES for fold in FOLDS}
    public = frames["configs/audit_batch_manifest.csv"]
    if set(
        public[["fold", "split", "batch_index"]].itertuples(index=False, name=None)
    ) != {
        (fold, split, batch) for fold in FOLDS for split in SPLITS for batch in range(8)
    }:
        raise ValueError("public batch manifest Cartesian grid失败")
    existing = frames["metrics/run_level_existing_metrics.csv"]
    if (
        set(existing[["seed_base", "fold"]].itertuples(index=False, name=None))
        != run_keys
    ):
        raise ValueError("existing 5×5 grid失败")
    if (
        int(existing["base_gate"].eq("PASS").sum()),
        int(existing["base_gate"].eq("FAIL").sum()),
    ) != (17, 8):
        raise ValueError("existing 17/8 contract失败")
    if not (
        existing["base_gate"].astype(str)
        == np.where(existing["base_degradation"].astype(float) <= 0.05, "PASS", "FAIL")
    ).all():
        raise ValueError("base gate没有由未四舍五入degradation唯一分类")

    representatives = frames["metrics/representative_runs.csv"]
    representative_keys = set(
        representatives[["seed_base", "fold"]].itertuples(index=False, name=None)
    )
    if (
        len(representative_keys) != 6
        or set(representatives["base_gate"]) != {"PASS", "FAIL"}
        or any(
            int(representatives["base_gate"].eq(gate).sum()) != 3
            for gate in ("PASS", "FAIL")
        )
    ):
        raise ValueError("representative runs不是3 PASS/3 FAIL")
    assets = frames["metrics/asset_manifest.csv"]
    expected_assets = {(seed, fold, "selected") for seed, fold in run_keys} | {
        (seed, fold, "last") for seed, fold in representative_keys
    }
    if (
        set(
            assets[["seed_base", "fold", "checkpoint_kind"]].itertuples(
                index=False, name=None
            )
        )
        != expected_assets
    ):
        raise ValueError("asset 25 selected+6 last grid失败")

    batch = frames["metrics/batch_gradient_metrics.csv"]
    expected_batch = {
        (
            seed,
            fold,
            "selected",
            split,
            f"f{fold}_{'tr' if split == 'train' else 'va'}_{index:02d}",
            group,
        )
        for seed, fold in run_keys
        for split in SPLITS
        for index in range(8)
        for group in GROUPS
    }
    if (
        set(
            batch[
                ["seed_base", "fold", "checkpoint_kind", "split", "batch_id", "group"]
            ].itertuples(index=False, name=None)
        )
        != expected_batch
    ):
        raise ValueError("selected batch gradient Cartesian grid失败")
    trajectory_batch = frames["metrics/trajectory_batch_gradient_metrics.csv"]
    expected_trajectory_batch = {
        (
            seed,
            fold,
            "last",
            split,
            f"f{fold}_{'tr' if split == 'train' else 'va'}_{index:02d}",
            group,
        )
        for seed, fold in representative_keys
        for split in SPLITS
        for index in range(8)
        for group in GROUPS
    }
    if (
        set(
            trajectory_batch[
                ["seed_base", "fold", "checkpoint_kind", "split", "batch_id", "group"]
            ].itertuples(index=False, name=None)
        )
        != expected_trajectory_batch
    ):
        raise ValueError("last batch gradient Cartesian grid失败")
    for frame, label in ((batch, "selected"), (trajectory_batch, "trajectory-last")):
        for column in (
            "optimizer_created",
            "optimizer_step",
            "pcr_signal_used",
            "contains_patient_ids",
        ):
            if (
                frame[column]
                .map(lambda value: _strict_bool(value, f"{label}.{column}"))
                .any()
            ):
                raise ValueError(f"{label} gradient safety/privacy flag失败: {column}")
        for column in ("deterministic_algorithms", "paired_forward_outputs_exact"):
            if (
                not frame[column]
                .map(lambda value: _strict_bool(value, f"{label}.{column}"))
                .all()
            ):
                raise ValueError(
                    f"{label} gradient deterministic/paired flag失败: {column}"
                )
        if not (
            frame["model_state_sha256_before"].astype(str)
            == frame["model_state_sha256_after"].astype(str)
        ).all():
            raise ValueError(f"{label} gradient extraction修改了model state")

    layer = frames["metrics/layer_level_conflict_metrics.csv"]
    expected_layer = {
        (seed, fold, split, group)
        for seed, fold in run_keys
        for split in SPLITS
        for group in GROUPS
    }
    if (
        set(
            layer[["seed_base", "fold", "split", "group"]].itertuples(
                index=False, name=None
            )
        )
        != expected_layer
    ):
        raise ValueError("layer aggregate Cartesian grid失败")
    run = frames["metrics/run_level_conflict_metrics.csv"]
    if set(run[["seed_base", "fold", "split"]].itertuples(index=False, name=None)) != {
        (seed, fold, split) for seed, fold in run_keys for split in SPLITS
    } or set(run["group"]) != {"all_shared"}:
        raise ValueError("run aggregate Cartesian/all-shared失败")
    for frame, name in ((layer, "layer"), (run, "run")):
        if set(frame["n_batches"].astype(int)) != {8} or set(
            frame["n_undefined"].astype(int)
        ) != {0}:
            raise ValueError(f"{name} aggregate batch/undefined失败")
        core = (
            "gradient_cosine",
            "negative_fraction",
            "weighted_gradient_norm_ratio",
            "base_descent_margin",
            "base_descent_failure_fraction",
            "ftv_descent_margin",
        )
        if not np.isfinite(frame[list(core)].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} aggregate core nonfinite")

    trajectory = frames["metrics/trajectory_conflict_metrics.csv"]
    if set(
        trajectory[
            ["seed_base", "fold", "checkpoint_kind", "split", "group"]
        ].itertuples(index=False, name=None)
    ) != {
        (seed, fold, kind, split, group)
        for seed, fold in representative_keys
        for kind in ("selected", "last")
        for split in SPLITS
        for group in GROUPS
    }:
        raise ValueError("trajectory 168-row Cartesian grid失败")
    change = frames["metrics/trajectory_change_metrics.csv"]
    if set(
        change[["seed_base", "fold", "split", "group"]].itertuples(
            index=False, name=None
        )
    ) != {
        (seed, fold, split, group)
        for seed, fold in representative_keys
        for split in SPLITS
        for group in GROUPS
    }:
        raise ValueError("trajectory change 84-row Cartesian grid失败")

    pass_fail = frames["metrics/pass_fail_comparison.csv"]
    if set(
        pass_fail[["split", "group", "metric"]].itertuples(index=False, name=None)
    ) != {
        (split, group, metric)
        for split in SPLITS
        for group in GROUPS
        for metric in (
            "gradient_cosine",
            "negative_fraction",
            "strong_negative_fraction",
            "weighted_gradient_norm_ratio",
            "base_descent_margin",
            "base_descent_failure_fraction",
        )
    }:
        raise ValueError("PASS/FAIL 84-row Cartesian grid失败")
    gradient_corr = frames["metrics/gradient_correlation_metrics.csv"]
    if set(
        gradient_corr[["split", "group", "endpoint"]].itertuples(index=False, name=None)
    ) != {
        (split, group, endpoint)
        for split in SPLITS
        for group in GROUPS
        for endpoint in (
            "gradient_cosine",
            "negative_fraction",
            "weighted_gradient_norm_ratio",
            "base_descent_margin",
        )
    }:
        raise ValueError("gradient correlation 56-row Cartesian grid失败")
    coverage = frames["metrics/ftv_coverage_metrics.csv"]
    if set(coverage[["fold", "split"]].itertuples(index=False, name=None)) != {
        (fold, split) for fold in FOLDS for split in SPLITS
    }:
        raise ValueError("coverage 10-row grid失败")
    if set(frames["metrics/layer_localization_metrics.csv"]["group"]) != set(
        _NONOVERLAP_GROUPS
    ):
        raise ValueError("localization五个非重叠组失败")
    if set(
        frames["metrics/fold_signature_metrics.csv"][["fold", "split"]].itertuples(
            index=False, name=None
        )
    ) != {(fold, split) for fold in FOLDS for split in SPLITS}:
        raise ValueError("fold signature 10-row grid失败")


def _public_privacy_checks(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    private_ids: set[str] = set()
    private_membership = root / "configs" / "private" / "audit_batch_membership.csv"
    if not private_membership.is_file() or not PRIVATE_CACHE_MANIFEST.is_file():
        raise FileNotFoundError("隐私验收需要ignored private membership/cache manifest")
    private_ids.update(
        pd.read_csv(private_membership, dtype={"patient_id": str})["patient_id"]
        .dropna()
        .astype(str)
    )
    private_ids.update(
        pd.read_csv(PRIVATE_CACHE_MANIFEST, dtype={"patient_id": str})["patient_id"]
        .dropna()
        .astype(str)
    )
    if not private_ids:
        raise ValueError("private identifier set为空")
    forbidden_exact = {
        "patient_id",
        "trial_id",
        "pcr",
        "label_pcr",
        "treatment",
        "subtype",
        "clinical",
    }
    for relative, frame in frames.items():
        ensure_no_patient_columns(frame.columns)
        lowered = {str(column).strip().lower() for column in frame.columns}
        if forbidden_exact & lowered:
            raise ValueError(f"公开CSV含禁止列: {relative}")
        if "contains_patient_ids" in frame:
            if (
                frame["contains_patient_ids"]
                .map(
                    lambda value: _strict_bool(
                        value, f"{relative}.contains_patient_ids"
                    )
                )
                .any()
            ):
                raise ValueError(f"公开CSV privacy flag失败: {relative}")
        if "manifest_contains_patient_ids" in frame:
            if (
                frame["manifest_contains_patient_ids"]
                .map(
                    lambda value: _strict_bool(
                        value, f"{relative}.manifest_contains_patient_ids"
                    )
                )
                .any()
            ):
                raise ValueError(f"source manifest privacy flag失败: {relative}")
        object_values: set[str] = set()
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            values = frame[column].dropna().astype(str)
            if values.map(
                lambda value: Path(value).is_absolute() or value.startswith("file://")
            ).any():
                raise ValueError(f"公开CSV含absolute path: {relative}/{column}")
            object_values.update(values)
        leaked = object_values & private_ids
        if leaked:
            raise ValueError(f"公开CSV泄漏private identifier: {relative}")
    report = _relative_file(root, "reports/final_report.md").read_text(encoding="utf-8")
    if any(
        token in report
        for token in (
            "/home/",
            "/data/",
            "/tmp/",
            "/mnt/",
            "/root/",
            "/usr/",
            "file://",
        )
    ):
        raise ValueError("最终报告含absolute path")
    searchable_ids = [
        identifier
        for identifier in private_ids
        if len(identifier) >= 6 and not identifier.isdigit()
    ]
    if searchable_ids:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_-])(?:"
            + "|".join(
                re.escape(value)
                for value in sorted(searchable_ids, key=len, reverse=True)
            )
            + r")(?![A-Za-z0-9_-])"
        )
        if pattern.search(report):
            raise ValueError("最终报告泄漏private identifier")


def _validate_figures(root: Path, manifest: pd.DataFrame) -> None:
    expected = {spec.figure_id: spec for spec in FIGURE_SPECS}
    if set(manifest["figure_id"]) != set(expected):
        raise ValueError("figure manifest不是固定F01–F14")
    for row in manifest.itertuples(index=False):
        spec = expected[str(row.figure_id)]
        if (
            str(row.filename) != f"figures/{spec.filename}"
            or str(row.title) != spec.title
            or str(row.analysis_unit) != spec.analysis_unit
            or str(row.uncertainty_note) != spec.uncertainty_note
            or _strict_bool(row.contains_patient_ids, "figure.contains_patient_ids")
        ):
            raise ValueError(f"figure manifest预注册内容漂移: {row.figure_id}")
        image_path = _relative_file(root, str(row.filename))
        if file_sha256(image_path) != _sha64(
            row.image_sha256, f"{row.figure_id}.image SHA"
        ):
            raise ValueError(f"figure image SHA漂移: {row.figure_id}")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            dpi_info = image.info.get("dpi", (0, 0))
            dpi = int(round(min(float(dpi_info[0]), float(dpi_info[1]))))
        if (
            width != int(row.width_pixels)
            or height != int(row.height_pixels)
            or dpi != int(row.dpi)
            or width < MIN_FIGURE_WIDTH
            or height < MIN_FIGURE_HEIGHT
            or dpi < 200
        ):
            raise ValueError(f"figure decode/resolution contract失败: {row.figure_id}")
        inputs = json.loads(str(row.input_files_json))
        if not isinstance(inputs, dict) or set(inputs) != set(spec.inputs):
            raise ValueError(f"figure input file set漂移: {row.figure_id}")
        observed = {
            relative: file_sha256(_relative_file(root, relative))
            for relative in sorted(inputs)
        }
        if inputs != observed or _sha64(
            row.input_sha256, f"{row.figure_id}.input SHA"
        ) != canonical_json_sha256(observed):
            raise ValueError(f"figure input SHA不闭环: {row.figure_id}")
        if not str(row.font_family).strip():
            raise ValueError(f"figure font family缺失: {row.figure_id}")


def _validate_report(root: Path) -> None:
    text = _relative_file(root, "reports/final_report.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (\d+)\.", text, flags=re.MULTILINE)
    if headings != [str(index) for index in range(1, 16)]:
        raise ValueError("final report不是固定15节")
    if any(f"Q{index} " not in text for index in range(1, 11)):
        raise ValueError("final report没有逐条十问")
    if sum("\u4e00" <= character <= "\u9fff" for character in text) < 500:
        raise ValueError("final report中文内容不足")
    required_phrases = (
        "state-loss safety gate",
        "完整base objective",
        "post-hoc association",
        "objective-level gradient geometry",
        "不支持epoch-level gradient trajectory",
        "bit-exact",
        "selected→last",
        "唯一第一优先级",
        "第二优先级",
        "本轮没有执行",
    )
    if any(phrase not in text for phrase in required_phrases):
        missing = [phrase for phrase in required_phrases if phrase not in text]
        raise ValueError(f"final report缺关键边界/回答: {missing}")
    for spec in FIGURE_SPECS:
        if text.count(f"../figures/{spec.filename}") < 1:
            raise ValueError(f"final report缺figure引用: {spec.figure_id}")


def _protected_directories_clean() -> tuple[str, str]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--short",
            "--",
            "additional_experiments/direct_grounded_response_state",
            "additional_experiments/g3_multiseed_generalization",
        ],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise ValueError(f"protected source directories不clean: {status}")
    if (
        branch != "feature/ispy-clean-corejepa"
        or re.fullmatch(r"[0-9a-f]{40}", head) is None
    ):
        raise ValueError("git branch/HEAD contract失败")
    return branch, head


def validate_acceptance(
    root: str | Path = AUDIT_ROOT, *, write: bool = True
) -> dict[str, Any]:
    """严格验收所有正式表、hash、grid、隐私、图、报告与唯一诊断。"""

    root = _root(root)
    manifest, marker = _analysis_gate(root)
    frames = _read_contract_tables(root)
    _expected_grid_checks(frames)
    diagnosis_data = {
        "diagnosis": _json(_relative_file(root, "metrics/diagnosis.json")),
        "frames": {
            "metrics/hypothesis_decision.csv": frames["metrics/hypothesis_decision.csv"]
        },
    }
    diagnosis, _ = _validate_diagnosis(diagnosis_data)
    _validate_figures(root, frames["metrics/figure_manifest.csv"])
    _validate_report(root)
    _public_privacy_checks(root, frames)
    branch, head = _protected_directories_clean()
    required_non_table = (
        "PLAN_FREEZE.json",
        "SOURCE_CONTRACT.json",
        "metrics/cache_input_contract.json",
        "metrics/resampling_indices.npz",
        "metrics/resampling_manifest.json",
        "metrics/diagnosis.json",
        "metrics/final_analysis_manifest.json",
        "metrics/FINAL_ANALYSIS_COMPLETE.json",
        "reports/asset_and_graph_inspection.md",
        "reports/final_report.md",
    )
    non_table_hashes = {
        relative: file_sha256(_relative_file(root, relative))
        for relative in required_non_table
    }
    selected = str(diagnosis["selected_hypothesis"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checks": {
            "source_contract_closed": True,
            "final_analysis_manifest_closed": True,
            "final_analysis_marker_last_and_closed": True,
            "all_csv_exact_schema_rows_keys": True,
            "seed_fold_grid_5x5": True,
            "base_gate_pass_fail_17_8": True,
            "batch_aggregation_eight_no_undefined": True,
            "diagnosis_unique_and_core_eligible": True,
            "figures_14_decodable_high_resolution": True,
            "final_report_15_sections_and_10_answers": True,
            "public_patient_identifiers_absent": True,
            "public_absolute_paths_absent": True,
            "protected_source_directories_clean": True,
            "fix_executed": False,
        },
        "row_counts": {relative: len(frame) for relative, frame in frames.items()},
        "selected_hypothesis": selected,
        "first_recommendation": diagnosis["first_recommendation"],
        "second_recommendation": diagnosis["second_recommendation"],
        "source_contract_sha256": file_sha256(SOURCE_CONTRACT),
        "final_analysis_manifest_sha256": file_sha256(FINAL_ANALYSIS_MANIFEST),
        "final_analysis_marker_sha256": file_sha256(FINAL_ANALYSIS_MARKER),
        "diagnosis_sha256": file_sha256(root / "metrics" / "diagnosis.json"),
        "figure_manifest_sha256": file_sha256(FIGURE_MANIFEST),
        "final_report_sha256": file_sha256(FINAL_REPORT),
        "non_table_file_sha256": non_table_hashes,
        "analysis_manifest_artifact_count": len(manifest["artifacts"]),
        "analysis_commit": marker["analysis_commit"],
        "validation_git_branch": branch,
        "validation_git_head": head,
        "contains_patient_ids": False,
    }
    if write:
        present = [
            path for path in (ACCEPTANCE_REPORT, ACCEPTANCE_COMPAT) if path.exists()
        ]
        if present:
            raise FileExistsError(
                f"拒绝覆盖已有验收文件: {[path.name for path in present]}"
            )
        atomic_json(ACCEPTANCE_REPORT, payload)
        atomic_json(ACCEPTANCE_COMPAT, payload)
    return payload


__all__ = [
    "ACCEPTANCE_COMPAT",
    "ACCEPTANCE_REPORT",
    "FIGURE_MANIFEST",
    "FIGURE_MANIFEST_COLUMNS",
    "FIGURE_SPECS",
    "FINAL_ANALYSIS_MANIFEST",
    "FINAL_ANALYSIS_MARKER",
    "FINAL_REPORT",
    "make_deliverables",
    "validate_acceptance",
]
