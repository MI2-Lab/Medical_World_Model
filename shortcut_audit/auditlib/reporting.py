"""CoRe-WM shortcut audit 的跨折汇总、对齐校验与图表生成。

本模块只消费已保存的 prediction/latent metric 明细，不会访问
checkpoint，也不会重新训练模型。汇总之前必须先证明：

* 每个 OOF patient 只属于一折，pCR label 在所有决策点与 audit 中恒定；
* 每条 perturbation/baseline 记录都能对齐唯一 native 记录；
* donor repetition 不会被当作独立患者来放大样本量；
* fold error bar 是五折指标的样本标准差（``ddof=1``），pooled OOF 另报；
* 主要差值始终是 ``comparison - native``，bootstrap 以 patient 为配对块。

所有写入都先在同一目录中生成临时文件，再原子提交。默认拒绝
覆盖任何已有文件。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DECISION_POINTS, PREDICTION_COLUMNS, validate_prediction_frame
from .metrics import (
    BINARY_METRICS,
    TRANSITION_METRICS,
    binary_classification_metrics,
    difference_from_native,
    fold_mean_sample_std,
    paired_patient_bootstrap_difference,
    summarize_transition_folds,
)


COPY_TRANSITIONS = ("T0->T1", "T1->T2", "T2->T3")
_CJK_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
)
LATENT_CHANGE_METRICS = (
    "native_layer_norm_mse",
    "comparison_layer_norm_mse",
    "latent_error_change",
    "native_cosine_similarity",
    "comparison_cosine_similarity",
    "response_state_mean_abs_change",
    "response_state_l2_change",
    "response_state_cosine_similarity",
)
REPORTING_INPUT_CONTRACTS: Mapping[str, tuple[str, ...]] = {
    "predictions": tuple(PREDICTION_COLUMNS),
    "copy_latent_metrics": (
        "patient_id",
        "fold",
        "transition",
        "learned_layer_norm_mse",
        "copy_layer_norm_mse",
        "normalized_transition_gain",
    ),
    "paired_perturbation_metrics": (
        "patient_id",
        "fold",
        "transition",
        "audit_condition",
        "native_layer_norm_mse",
        "perturbed_layer_norm_mse",
        "latent_error_change",
    ),
    "donor_metrics": (
        "recipient_patient_id",
        "donor_patient_id",
        "fold",
        "audit_repetition",
        "matching_distance",
        "transition",
        "native_layer_norm_mse",
        "donor_layer_norm_mse",
        "latent_error_change",
    ),
}
REQUIRED_FIGURE_FILENAMES = (
    "01_native_perturbation_auroc.png",
    "02_fold_auroc_change.png",
    "03_learned_vs_copy_error.png",
    "04_transition_gain_distribution.png",
    "05_repeated_t0_probability_change.png",
    "06_temporal_swap_probability_change.png",
    "07_followup_swap_probability_change.png",
    "08_f1_f5_native_comparison.png",
)


@dataclass(frozen=True)
class ValidatedPredictionSet:
    """已通过 OOF/native 对齐的 prediction 集合。"""

    frame: pd.DataFrame
    coverage: pd.DataFrame
    native_condition: str
    expected_folds: tuple[int, ...]


@dataclass(frozen=True)
class PredictionReport:
    """Prediction 级、patient 级、fold 级与 pooled 级的完整汇总。"""

    predictions: pd.DataFrame
    coverage: pd.DataFrame
    patient_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_summary: pd.DataFrame
    pooled_oof: pd.DataFrame
    native_differences: pd.DataFrame
    fold_changes: pd.DataFrame
    paired_bootstrap: pd.DataFrame
    bootstrap_samples: pd.DataFrame
    probability_changes: pd.DataFrame
    patient_probability_changes: pd.DataFrame
    repetition_metrics: pd.DataFrame
    repetition_summary: pd.DataFrame
    native_condition: str
    expected_folds: tuple[int, ...]


@dataclass(frozen=True)
class LatentMetricReport:
    """Copy-current latent audit 的 equal-patient 汇总。"""

    records: pd.DataFrame
    patient_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_summary: pd.DataFrame
    pooled_metrics: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    bootstrap_samples: pd.DataFrame


@dataclass(frozen=True)
class PerturbationMetricReport:
    """C/D/E paired latent/response-state 指标汇总。"""

    records: pd.DataFrame
    patient_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_summary: pd.DataFrame
    pooled_patient_metrics: pd.DataFrame


@dataclass(frozen=True)
class AuditReportingBundle:
    """统一 reporting 输入及已验证的汇总产物。"""

    prediction: PredictionReport
    copy_latent: LatentMetricReport | None
    perturbation_latent: PerturbationMetricReport | None


@dataclass(frozen=True)
class FigureConditionRoles:
    """必需图表中各类 condition 的显式映射。"""

    perturbations: tuple[str, ...]
    repeated_t0: tuple[str, ...]
    temporal_order: tuple[str, ...]
    followup_swap: tuple[str, ...]
    simplified_baselines: tuple[str, ...]


@dataclass(frozen=True)
class FigureGenerationResult:
    """八类图的路径、定义与 hash manifest。"""

    artifacts: pd.DataFrame
    manifest_path: Path


@dataclass(frozen=True)
class ReportingPipelineResult:
    """一键汇总的内存 bundle、table manifest 和 figure manifest。"""

    bundle: AuditReportingBundle
    table_manifest_path: Path
    figures: FigureGenerationResult


def _normalize_expected_folds(expected_folds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in expected_folds)
    if not values or len(values) != len(set(values)):
        raise ValueError("expected_folds 必须是非空且无重复的 fold 序列")
    return values


def _nonempty_text(value: Any, name: str) -> str:
    output = str(value).strip()
    if not output:
        raise ValueError(f"{name} 不得为空")
    return output


def _donor_presence(frame: pd.DataFrame) -> pd.Series:
    donor = frame["donor_patient_id"].astype("string")
    return donor.notna() & donor.str.strip().ne("")


def _is_simplified_baseline_name(value: str) -> bool:
    normalized = str(value).strip().lower().replace("-", "_")
    return normalized.startswith(
        (
            "simplified_baseline_",
            "f1_",
            "f2_",
            "f3_",
            "f4_",
            "f5_",
        )
    )


def validate_oof_predictions(
    predictions: pd.DataFrame,
    *,
    native_condition: str = "native",
    expected_folds: Iterable[int] = range(5),
    expected_decisions: Sequence[str] = DECISION_POINTS,
    allow_incomplete_conditions: Iterable[str] = (),
) -> ValidatedPredictionSet:
    """验证统一 prediction frame 的 OOF、label 和 native 对齐。

    普通 perturbation/baseline 在其存在的每个 decision point 必须完整覆盖
    native patient 集合。Donor condition 允许因匹配失败而只覆盖子集，
    但每条记录仍必须能对齐 native。其他有意的不完整 condition 必须
    通过 ``allow_incomplete_conditions`` 显式列出。
    """

    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions 必须是非空 pandas DataFrame")
    folds = _normalize_expected_folds(expected_folds)
    native_condition = _nonempty_text(native_condition, "native_condition")
    decisions = tuple(str(value) for value in expected_decisions)
    if not decisions or len(decisions) != len(set(decisions)):
        raise ValueError("expected_decisions 必须非空且无重复")
    incomplete = {_nonempty_text(value, "allow_incomplete_conditions") for value in allow_incomplete_conditions}

    incoming = predictions.reset_index(drop=True).copy()
    normalized_contract = validate_prediction_frame(incoming, expected_folds=folds)
    # 保留 donor matching 的 subtype/treatment/volume 等额外列，但以契约校验后的
    # dtype/value 覆盖核心列。
    for column in PREDICTION_COLUMNS:
        incoming[column] = normalized_contract[column]
    frame = incoming

    if frame["patient_id"].isna().any():
        raise ValueError("patient_id 不得缺失")
    patient_fold_count = frame.groupby("patient_id", observed=True)["fold"].nunique()
    if patient_fold_count.gt(1).any():
        bad = patient_fold_count[patient_fold_count.gt(1)].index.astype(str).tolist()
        raise ValueError(f"OOF patient 出现在多折：{bad[:5]}")
    patient_label_count = frame.groupby("patient_id", observed=True)["y_true"].nunique()
    if patient_label_count.ne(1).any():
        bad = patient_label_count[patient_label_count.ne(1)].index.astype(str).tolist()
        raise ValueError(f"patient 的 pCR label 在记录间不一致：{bad[:5]}")

    has_donor = _donor_presence(frame)
    repetition_numeric = pd.to_numeric(frame["repetition_id"], errors="coerce")
    distance_numeric = pd.to_numeric(frame["matching_distance"], errors="coerce")
    has_repetition = repetition_numeric.notna()
    has_distance = distance_numeric.notna()
    if not (
        np.array_equal(has_donor.to_numpy(dtype=bool), has_repetition.to_numpy(dtype=bool))
        and np.array_equal(has_donor.to_numpy(dtype=bool), has_distance.to_numpy(dtype=bool))
    ):
        raise ValueError("donor_patient_id/repetition_id/matching_distance 必须同时出现")
    if has_donor.any():
        donor_rows = frame.loc[has_donor]
        repetition = repetition_numeric.loc[has_donor].to_numpy(dtype=float)
        distance = distance_numeric.loc[has_donor].to_numpy(dtype=float)
        if (
            not np.isfinite(repetition).all()
            or not np.equal(repetition, np.floor(repetition)).all()
            or np.any(repetition <= 0)
        ):
            raise ValueError("donor repetition_id 必须是正整数")
        if not np.isfinite(distance).all() or np.any(distance < 0):
            raise ValueError("donor matching_distance 必须是有限非负数")
        if donor_rows["donor_patient_id"].astype(str).eq(donor_rows["patient_id"]).any():
            raise ValueError("donor 不得等于 recipient")
        frame.loc[has_donor, "repetition_id"] = repetition.astype(np.int64)
        frame.loc[has_donor, "matching_distance"] = distance

    donor_kind_count = (
        pd.DataFrame({"audit_condition": frame["audit_condition"], "has_donor": has_donor})
        .groupby("audit_condition", observed=True)["has_donor"]
        .nunique()
    )
    if donor_kind_count.gt(1).any():
        bad = donor_kind_count[donor_kind_count.gt(1)].index.tolist()
        raise ValueError(f"同一 audit_condition 不得混合 donor/非 donor 行：{bad}")

    native = frame.loc[frame["audit_condition"].eq(native_condition)].copy()
    if native.empty:
        raise ValueError(f"缺少 native condition：{native_condition}")
    if _donor_presence(native).any():
        raise ValueError("native 记录不得含 donor provenance")
    observed_decisions = set(native["decision_point"])
    if observed_decisions != set(decisions):
        raise ValueError(
            f"native decision points 不完整：observed={sorted(observed_decisions)}, "
            f"expected={list(decisions)}"
        )
    native_key = ["patient_id", "fold", "decision_point"]
    if native.duplicated(native_key, keep=False).any():
        raise ValueError("native patient/fold/decision 必须唯一")
    counts = native.groupby("patient_id", observed=True)["decision_point"].nunique()
    if not counts.eq(len(decisions)).all():
        bad = counts[counts.ne(len(decisions))].index.astype(str).tolist()
        raise ValueError(f"native patient 未完整覆盖所有 decision points：{bad[:5]}")
    for decision in decisions:
        decision_folds = set(native.loc[native["decision_point"].eq(decision), "fold"])
        if decision_folds != set(folds):
            raise ValueError(
                f"native {decision} 折集不完整：observed={sorted(decision_folds)}, "
                f"expected={list(folds)}"
            )

    reference = native.loc[:, [*native_key, "y_true", "threshold", "checkpoint"]].rename(
        columns={
            "y_true": "native_y_true",
            "threshold": "native_threshold",
            "checkpoint": "native_checkpoint",
        }
    )
    aligned = frame.merge(reference, on=native_key, how="left", validate="many_to_one", indicator=True)
    missing_reference = aligned["_merge"].ne("both")
    if missing_reference.any():
        examples = aligned.loc[missing_reference, native_key].head(5).to_dict("records")
        raise ValueError(f"audit 记录无法对齐 native：{examples}")
    if not aligned["y_true"].eq(aligned["native_y_true"]).all():
        examples = aligned.loc[
            ~aligned["y_true"].eq(aligned["native_y_true"]),
            [*native_key, "audit_condition", "y_true", "native_y_true"],
        ].head(5)
        raise ValueError(f"audit/native label 不一致：{examples.to_dict('records')}")
    shared_readout = (
        ~aligned["audit_condition"].eq(native_condition)
        & ~aligned["audit_condition"].map(_is_simplified_baseline_name)
    )
    threshold_aligned = np.isclose(
        aligned.loc[shared_readout, "threshold"].to_numpy(float),
        aligned.loc[shared_readout, "native_threshold"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )
    if not threshold_aligned.all():
        raise ValueError("C/D/E perturbation 必须复用对齐 native 的 validation-selected threshold")
    checkpoint_aligned = aligned.loc[shared_readout, "checkpoint"].eq(
        aligned.loc[shared_readout, "native_checkpoint"]
    )
    if not checkpoint_aligned.all():
        raise ValueError("C/D/E perturbation 必须复用对齐 native 的 fold checkpoint/readout")

    native_patients_by_decision = {
        decision: set(native.loc[native["decision_point"].eq(decision), "patient_id"])
        for decision in decisions
    }
    coverage_rows: list[dict[str, Any]] = []
    for (condition, decision), group in frame.groupby(
        ["audit_condition", "decision_point"], sort=True, observed=True
    ):
        patients = set(group["patient_id"])
        native_patients = native_patients_by_decision[str(decision)]
        condition_has_donor = bool(_donor_presence(group).any())
        complete = patients == native_patients
        if (
            condition != native_condition
            and not condition_has_donor
            and condition not in incomplete
            and not complete
        ):
            missing = sorted(native_patients - patients)
            raise ValueError(
                f"condition={condition}, decision={decision} 未完整覆盖 native；"
                f"缺少 {len(missing)} 名，示例={missing[:5]}"
            )
        coverage_rows.append(
            {
                "audit_condition": condition,
                "decision_point": decision,
                "n_records": int(len(group)),
                "n_patients": int(len(patients)),
                "n_native_patients": int(len(native_patients)),
                "n_unmatched_patients": int(len(native_patients - patients)),
                "coverage_fraction": float(len(patients) / len(native_patients)),
                "complete_native_coverage": bool(complete),
                "is_donor_condition": condition_has_donor,
                "n_repetitions": int(group["repetition_id"].nunique(dropna=True)),
            }
        )

    if has_donor.any():
        donor = frame.loc[has_donor].copy()
        native_fold_by_patient = (
            native.loc[:, ["patient_id", "fold"]]
            .drop_duplicates()
            .set_index("patient_id")["fold"]
            .to_dict()
        )
        unknown_donors = sorted(
            set(donor["donor_patient_id"].astype(str)).difference(native_fold_by_patient)
        )
        if unknown_donors:
            raise ValueError(f"donor patient 不在 native OOF 集合：{unknown_donors[:5]}")
        wrong_fold = donor.apply(
            lambda row: native_fold_by_patient[str(row["donor_patient_id"])]
            != int(row["fold"]),
            axis=1,
        )
        if wrong_fold.any():
            examples = donor.loc[
                wrong_fold,
                ["patient_id", "donor_patient_id", "fold", "audit_condition"],
            ].head(5)
            raise ValueError(f"donor 必须来自 recipient 同一 held-out fold：{examples.to_dict('records')}")
        donor["_donor"] = donor["donor_patient_id"].astype(str)
        donor["_rep"] = pd.to_numeric(donor["repetition_id"]).astype(int)
        donor["_distance"] = pd.to_numeric(donor["matching_distance"]).astype(float)
        pair_group = donor.groupby(
            ["audit_condition", "patient_id", "_rep"], observed=True, dropna=False
        )
        for column in ("fold", "_donor", "_distance"):
            if pair_group[column].nunique(dropna=False).gt(1).any():
                raise ValueError(f"donor pair 的 {column} 在 decision points 间不一致")
        duplicate_decision = pair_group["decision_point"].apply(lambda values: values.duplicated().any())
        if duplicate_decision.any():
            raise ValueError("donor patient/repetition/decision 存在重复")

    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["audit_condition", "decision_point"], kind="stable"
    ).reset_index(drop=True)
    return ValidatedPredictionSet(
        frame=frame.reset_index(drop=True),
        coverage=coverage,
        native_condition=native_condition,
        expected_folds=folds,
    )


def _patient_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["audit_condition", "decision_point", "patient_id", "fold", "y_true"]
    grouped = frame.groupby(keys, sort=True, observed=True, dropna=False)
    threshold_count = grouped["threshold"].nunique(dropna=False)
    if threshold_count.gt(1).any():
        examples = threshold_count[threshold_count.gt(1)].head(5).index.tolist()
        raise ValueError(f"同一 patient/condition/decision 的 threshold 不一致：{examples}")
    output = grouped.agg(
        predicted_probability=("predicted_probability", "mean"),
        threshold=("threshold", "first"),
        n_records=("predicted_probability", "size"),
        n_donors=("donor_patient_id", lambda values: values.nunique(dropna=True)),
        n_repetitions=("repetition_id", lambda values: values.nunique(dropna=True)),
    ).reset_index()
    output["predicted_label"] = (
        output["predicted_probability"] >= output["threshold"]
    ).astype(np.int64)
    return output


def _metric_record(frame: pd.DataFrame) -> dict[str, Any]:
    return binary_classification_metrics(
        frame["y_true"].to_numpy(),
        frame["predicted_probability"].to_numpy(),
        threshold=frame["threshold"].to_numpy(),
    )


def _binary_summary_tables(
    patients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for (condition, decision, fold), group in patients.groupby(
        ["audit_condition", "decision_point", "fold"], sort=True, observed=True
    ):
        fold_rows.append(
            {
                "audit_condition": condition,
                "decision_point": decision,
                "fold": int(fold),
                **_metric_record(group),
            }
        )
    for (condition, decision), group in patients.groupby(
        ["audit_condition", "decision_point"], sort=True, observed=True
    ):
        pooled_rows.append(
            {
                "audit_condition": condition,
                "decision_point": decision,
                **_metric_record(group),
                "n_folds": int(group["fold"].nunique()),
                "aggregation": "pooled_oof_equal_patient",
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    pooled = pd.DataFrame(pooled_rows)

    summary_rows: list[dict[str, Any]] = []
    for (condition, decision), group in fold_metrics.groupby(
        ["audit_condition", "decision_point"], sort=True, observed=True
    ):
        summary = fold_mean_sample_std(group, metric_columns=BINARY_METRICS)
        summary.insert(0, "decision_point", decision)
        summary.insert(0, "audit_condition", condition)
        summary["error_bar_definition"] = "fold metric sample SD (ddof=1)"
        summary_rows.extend(summary.to_dict("records"))
    fold_summary = pd.DataFrame(summary_rows)
    return fold_metrics, fold_summary, pooled


def _probability_change_tables(
    frame: pd.DataFrame, native_condition: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    native = frame.loc[
        frame["audit_condition"].eq(native_condition),
        ["patient_id", "fold", "decision_point", "y_true", "predicted_probability"],
    ].rename(columns={"predicted_probability": "native_probability"})
    comparison = frame.loc[~frame["audit_condition"].eq(native_condition)].copy()
    changes = comparison.merge(
        native,
        on=["patient_id", "fold", "decision_point", "y_true"],
        how="left",
        validate="many_to_one",
    )
    if changes["native_probability"].isna().any():
        raise RuntimeError("内部错误：probability change 未对齐 native")
    changes.rename(columns={"predicted_probability": "comparison_probability"}, inplace=True)
    changes["probability_change"] = (
        changes["comparison_probability"] - changes["native_probability"]
    )
    changes["absolute_probability_change"] = changes["probability_change"].abs()
    selected = [
        "patient_id",
        "fold",
        "decision_point",
        "audit_condition",
        "y_true",
        "native_probability",
        "comparison_probability",
        "probability_change",
        "absolute_probability_change",
        "donor_patient_id",
        "repetition_id",
        "matching_distance",
    ]
    changes = changes.loc[:, selected]
    patient = (
        changes.groupby(
            ["patient_id", "fold", "decision_point", "audit_condition", "y_true"],
            sort=True,
            observed=True,
            dropna=False,
        )
        .agg(
            native_probability=("native_probability", "first"),
            comparison_probability=("comparison_probability", "mean"),
            probability_change=("probability_change", "mean"),
            mean_absolute_probability_change=("absolute_probability_change", "mean"),
            n_records=("probability_change", "size"),
            n_donors=("donor_patient_id", lambda values: values.nunique(dropna=True)),
            n_repetitions=("repetition_id", lambda values: values.nunique(dropna=True)),
        )
        .reset_index()
    )
    patient["absolute_probability_change"] = patient["probability_change"].abs()
    return changes, patient


def _paired_comparison_tables(
    patients: pd.DataFrame,
    *,
    native_condition: str,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    native = patients.loc[patients["audit_condition"].eq(native_condition)].copy()
    difference_rows: list[dict[str, Any]] = []
    fold_change_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_samples: list[pd.DataFrame] = []

    comparisons = patients.loc[~patients["audit_condition"].eq(native_condition)]
    group_index = 0
    for (condition, decision), comparison in comparisons.groupby(
        ["audit_condition", "decision_point"], sort=True, observed=True
    ):
        reference = native.loc[
            native["decision_point"].eq(decision)
            & native["patient_id"].isin(comparison["patient_id"])
        ].copy()
        if set(reference["patient_id"]) != set(comparison["patient_id"]):
            raise RuntimeError("内部错误：paired native patient set 不一致")
        native_metric = _metric_record(reference)
        comparison_metric = _metric_record(comparison)
        differences = difference_from_native(
            native_metric, comparison_metric, metrics=BINARY_METRICS
        )
        differences.insert(0, "decision_point", decision)
        differences.insert(0, "audit_condition", condition)
        differences["n_paired_patients"] = int(len(comparison))
        differences["native_reference_scope"] = "same paired patients"
        difference_rows.extend(differences.to_dict("records"))

        for fold, fold_comparison in comparison.groupby("fold", sort=True, observed=True):
            fold_reference = reference.loc[
                reference["fold"].eq(fold)
                & reference["patient_id"].isin(fold_comparison["patient_id"])
            ]
            fold_native_metric = _metric_record(fold_reference)
            fold_comparison_metric = _metric_record(fold_comparison)
            fold_diff = difference_from_native(
                fold_native_metric, fold_comparison_metric, metrics=BINARY_METRICS
            )
            fold_diff.insert(0, "fold", int(fold))
            fold_diff.insert(0, "decision_point", decision)
            fold_diff.insert(0, "audit_condition", condition)
            fold_diff["n_paired_patients"] = int(len(fold_comparison))
            fold_change_rows.extend(fold_diff.to_dict("records"))

        bootstrap = paired_patient_bootstrap_difference(
            reference,
            comparison,
            pair_columns=["fold"],
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed + group_index,
        )
        summary = bootstrap["summary"].copy()
        summary.insert(0, "decision_point", decision)
        summary.insert(0, "audit_condition", condition)
        bootstrap_rows.extend(summary.to_dict("records"))
        samples = bootstrap["bootstrap_samples"].copy()
        samples.insert(0, "decision_point", decision)
        samples.insert(0, "audit_condition", condition)
        bootstrap_samples.append(samples)
        group_index += 1

    return (
        pd.DataFrame(difference_rows),
        pd.DataFrame(fold_change_rows),
        pd.DataFrame(bootstrap_rows),
        pd.concat(bootstrap_samples, ignore_index=True)
        if bootstrap_samples
        else pd.DataFrame(
            columns=[
                "audit_condition",
                "decision_point",
                "bootstrap_index",
                "auroc_difference",
                "auprc_difference",
            ]
        ),
    )


def _repetition_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor = frame.loc[_donor_presence(frame)].copy()
    metric_rows: list[dict[str, Any]] = []
    if donor.empty:
        return pd.DataFrame(), pd.DataFrame()
    donor["repetition_id"] = pd.to_numeric(donor["repetition_id"]).astype(int)
    for (condition, decision, repetition, fold), group in donor.groupby(
        ["audit_condition", "decision_point", "repetition_id", "fold"],
        sort=True,
        observed=True,
    ):
        metric_rows.append(
            {
                "audit_condition": condition,
                "decision_point": decision,
                "repetition_id": int(repetition),
                "aggregation_scope": "fold",
                "fold": int(fold),
                **_metric_record(group),
            }
        )
    for (condition, decision, repetition), group in donor.groupby(
        ["audit_condition", "decision_point", "repetition_id"],
        sort=True,
        observed=True,
    ):
        metric_rows.append(
            {
                "audit_condition": condition,
                "decision_point": decision,
                "repetition_id": int(repetition),
                "aggregation_scope": "pooled_oof",
                "fold": pd.NA,
                **_metric_record(group),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    pooled = metrics.loc[metrics["aggregation_scope"].eq("pooled_oof")]
    summary_rows: list[dict[str, Any]] = []
    for (condition, decision), group in pooled.groupby(
        ["audit_condition", "decision_point"], sort=True, observed=True
    ):
        summary = fold_mean_sample_std(group, metric_columns=BINARY_METRICS).rename(
            columns={
                "sample_std": "repetition_sample_std",
                "n_valid_folds": "n_valid_repetitions",
            }
        )
        summary.insert(0, "decision_point", decision)
        summary.insert(0, "audit_condition", condition)
        summary["error_bar_definition"] = "pooled OOF metric sample SD across donor repetitions (ddof=1)"
        summary_rows.extend(summary.to_dict("records"))
    return metrics, pd.DataFrame(summary_rows)


def build_prediction_report(
    predictions: pd.DataFrame,
    *,
    native_condition: str = "native",
    expected_folds: Iterable[int] = range(5),
    allow_incomplete_conditions: Iterable[str] = (),
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> PredictionReport:
    """生成 fold mean±sample SD、pooled OOF、native 差与 paired bootstrap。"""

    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap <= 0:
        raise ValueError("n_bootstrap 必须是正整数")
    validated = validate_oof_predictions(
        predictions,
        native_condition=native_condition,
        expected_folds=expected_folds,
        allow_incomplete_conditions=allow_incomplete_conditions,
    )
    patient = _patient_prediction_frame(validated.frame)
    fold_metrics, fold_summary, pooled = _binary_summary_tables(patient)
    changes, patient_changes = _probability_change_tables(
        validated.frame, validated.native_condition
    )
    differences, fold_changes, bootstrap, samples = _paired_comparison_tables(
        patient,
        native_condition=validated.native_condition,
        n_bootstrap=int(n_bootstrap),
        confidence_level=confidence_level,
        seed=int(seed),
    )
    repetition_metrics, repetition_summary = _repetition_tables(validated.frame)
    return PredictionReport(
        predictions=validated.frame,
        coverage=validated.coverage,
        patient_predictions=patient,
        fold_metrics=fold_metrics,
        fold_summary=fold_summary,
        pooled_oof=pooled,
        native_differences=differences,
        fold_changes=fold_changes,
        paired_bootstrap=bootstrap,
        bootstrap_samples=samples,
        probability_changes=changes,
        patient_probability_changes=patient_changes,
        repetition_metrics=repetition_metrics,
        repetition_summary=repetition_summary,
        native_condition=validated.native_condition,
        expected_folds=validated.expected_folds,
    )


def validate_copy_latent_metrics(
    records: pd.DataFrame,
    *,
    expected_folds: Iterable[int] = range(5),
    expected_transitions: Sequence[str] = COPY_TRANSITIONS,
    gain_epsilon: float = 1e-12,
) -> pd.DataFrame:
    """验证 copy-current transition 明细与 G 公式。"""

    required = {
        "patient_id",
        "fold",
        "transition",
        "learned_layer_norm_mse",
        "copy_layer_norm_mse",
        "normalized_transition_gain",
    }
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("copy latent metrics 必须是非空 DataFrame")
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"copy latent metrics 缺少列：{missing}")
    folds = _normalize_expected_folds(expected_folds)
    output = records.reset_index(drop=True).copy()
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    output["transition"] = output["transition"].astype(str)
    if (output["patient_id"] == "").any() or not output["fold"].isin(folds).all():
        raise ValueError("copy latent patient_id/fold 无效")
    transitions = tuple(str(value) for value in expected_transitions)
    if set(output["transition"]) != set(transitions):
        raise ValueError(
            f"copy transitions 不完整：observed={sorted(output['transition'].unique())}, "
            f"expected={list(transitions)}"
        )
    for transition in transitions:
        observed = set(output.loc[output["transition"].eq(transition), "fold"])
        if observed != set(folds):
            raise ValueError(f"copy {transition} 未覆盖 expected folds")
    if output.groupby("patient_id", observed=True)["fold"].nunique().gt(1).any():
        raise ValueError("copy latent patient 出现在多折")
    if output.duplicated(["patient_id", "fold", "transition"], keep=False).any():
        raise ValueError("copy latent patient/fold/transition 不唯一")

    numeric_columns = [column for column in TRANSITION_METRICS if column in output]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
        values = output[column].to_numpy()
        if np.isinf(values).any():
            raise ValueError(f"copy latent {column} 含 infinity")
    for column in (
        "learned_layer_norm_mse",
        "copy_layer_norm_mse",
        "normalized_transition_gain",
    ):
        if not np.isfinite(output[column]).all():
            raise ValueError(f"copy latent {column} 必须全部有限")
    if (output[["learned_layer_norm_mse", "copy_layer_norm_mse"]] < 0).any().any():
        raise ValueError("copy/learned latent error 不得为负")
    expected_gain = (
        output["copy_layer_norm_mse"] - output["learned_layer_norm_mse"]
    ) / (output["copy_layer_norm_mse"] + gain_epsilon)
    if not np.allclose(
        output["normalized_transition_gain"], expected_gain, rtol=1e-7, atol=1e-10
    ):
        raise ValueError("normalized_transition_gain 与指定 G 公式不一致")
    return output


def summarize_copy_latent_metrics(
    records: pd.DataFrame,
    *,
    expected_folds: Iterable[int] = range(5),
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> LatentMetricReport:
    """先在 patient 内聚合，再计算 copy 汇总与患者块 bootstrap CI。"""

    if (
        isinstance(n_bootstrap, bool)
        or int(n_bootstrap) != n_bootstrap
        or n_bootstrap <= 0
    ):
        raise ValueError("n_bootstrap 必须是正整数")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level 必须严格位于 (0, 1)")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed 必须是整数")

    output = validate_copy_latent_metrics(records, expected_folds=expected_folds)
    metric_columns = [column for column in TRANSITION_METRICS if column in output]
    patient_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    pooled_rows: list[dict[str, Any]] = []
    bootstrap_summary_frames: list[pd.DataFrame] = []
    bootstrap_sample_frames: list[pd.DataFrame] = []
    scopes: list[tuple[str, pd.DataFrame]] = [
        (transition, output.loc[output["transition"].eq(transition)])
        for transition in COPY_TRANSITIONS
    ]
    scopes.append(("ALL", output))
    for transition, subset in scopes:
        summary = summarize_transition_folds(subset, metric_columns=metric_columns)
        patient = summary["patient_metrics"].copy()
        patient.insert(0, "transition_scope", transition)
        patient_frames.append(patient)
        folds = summary["fold_metrics"].copy()
        folds.insert(0, "transition_scope", transition)
        fold_frames.append(folds)
        fold_summary = summary["fold_summary"].copy()
        fold_summary.insert(0, "transition_scope", transition)
        fold_summary["error_bar_definition"] = "fold metric sample SD (ddof=1)"
        summary_frames.append(fold_summary)
        for aggregation, values in (
            ("pooled_transition", summary["pooled_transition_metrics"]),
            ("pooled_equal_patient", summary["pooled_patient_metrics"]),
        ):
            row = {"transition_scope": transition, "aggregation": aggregation}
            row.update(values)
            pooled_rows.append(row)

        bootstrap_metrics = patient.loc[:, metric_columns].copy()
        # 保留 paired learned-vs-copy 差；正值表示 learned 误差更低。
        bootstrap_metrics["copy_minus_learned_layer_norm_mse"] = (
            bootstrap_metrics["copy_layer_norm_mse"]
            - bootstrap_metrics["learned_layer_norm_mse"]
        )
        rng = np.random.default_rng(int(seed))
        sample_indices = rng.integers(
            0,
            len(bootstrap_metrics),
            size=(int(n_bootstrap), len(bootstrap_metrics)),
        )
        alpha = (1.0 - float(confidence_level)) / 2.0
        scope_summary_rows: list[dict[str, Any]] = []
        scope_sample_frames: list[pd.DataFrame] = []
        for metric in bootstrap_metrics.columns:
            values = bootstrap_metrics[metric].to_numpy(dtype=np.float64)
            sampled = values[sample_indices]
            finite_count = np.isfinite(sampled).sum(axis=1)
            bootstrap_values = np.divide(
                np.nansum(sampled, axis=1),
                finite_count,
                out=np.full(int(n_bootstrap), np.nan, dtype=np.float64),
                where=finite_count > 0,
            )
            finite_bootstrap = bootstrap_values[np.isfinite(bootstrap_values)]
            finite_values = values[np.isfinite(values)]
            lower, upper = (
                np.quantile(finite_bootstrap, [alpha, 1.0 - alpha])
                if finite_bootstrap.size
                else (np.nan, np.nan)
            )
            scope_summary_rows.append(
                {
                    "transition_scope": transition,
                    "metric": metric,
                    "estimate": (
                        float(finite_values.mean()) if finite_values.size else np.nan
                    ),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "confidence_level": float(confidence_level),
                    "n_bootstrap": int(n_bootstrap),
                    "n_valid_bootstrap": int(finite_bootstrap.size),
                    "n_patients": int(len(bootstrap_metrics)),
                    "bootstrap_unit": "patient",
                    "aggregation": "pooled_equal_patient",
                }
            )
            scope_sample_frames.append(
                pd.DataFrame(
                    {
                        "transition_scope": transition,
                        "bootstrap_index": np.arange(int(n_bootstrap)),
                        "metric": metric,
                        "value": bootstrap_values,
                    }
                )
            )
        bootstrap_summary_frames.append(pd.DataFrame(scope_summary_rows))
        bootstrap_sample_frames.append(
            pd.concat(scope_sample_frames, ignore_index=True)
        )
    return LatentMetricReport(
        records=output,
        patient_metrics=pd.concat(patient_frames, ignore_index=True),
        fold_metrics=pd.concat(fold_frames, ignore_index=True),
        fold_summary=pd.concat(summary_frames, ignore_index=True),
        pooled_metrics=pd.DataFrame(pooled_rows),
        bootstrap_summary=pd.concat(bootstrap_summary_frames, ignore_index=True),
        bootstrap_samples=pd.concat(bootstrap_sample_frames, ignore_index=True),
    )


def _normalize_paired_latent_records(
    paired_metrics: pd.DataFrame | None,
    donor_metrics: pd.DataFrame | None,
    *,
    expected_folds: tuple[int, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if paired_metrics is not None:
        if not isinstance(paired_metrics, pd.DataFrame) or paired_metrics.empty:
            raise ValueError("paired_perturbation_metrics 如果提供必须为非空 DataFrame")
        paired = paired_metrics.reset_index(drop=True).copy()
        required = {
            "patient_id",
            "fold",
            "transition",
            "audit_condition",
            "native_layer_norm_mse",
            "perturbed_layer_norm_mse",
            "latent_error_change",
        }
        missing = sorted(required.difference(paired.columns))
        if missing:
            raise ValueError(f"paired latent metrics 缺少列：{missing}")
        paired.rename(
            columns={
                "perturbed_layer_norm_mse": "comparison_layer_norm_mse",
                "perturbed_cosine_similarity": "comparison_cosine_similarity",
            },
            inplace=True,
        )
        paired["donor_patient_id"] = pd.NA
        paired["repetition_id"] = pd.NA
        paired["matching_distance"] = np.nan
        frames.append(paired)

    if donor_metrics is not None:
        if not isinstance(donor_metrics, pd.DataFrame) or donor_metrics.empty:
            raise ValueError("donor_metrics 如果提供必须为非空 DataFrame")
        donor = donor_metrics.reset_index(drop=True).copy()
        required = {
            "recipient_patient_id",
            "donor_patient_id",
            "fold",
            "audit_repetition",
            "matching_distance",
            "transition",
            "native_layer_norm_mse",
            "donor_layer_norm_mse",
            "latent_error_change",
        }
        missing = sorted(required.difference(donor.columns))
        if missing:
            raise ValueError(f"donor latent metrics 缺少列：{missing}")
        donor.rename(
            columns={
                "recipient_patient_id": "patient_id",
                "audit_repetition": "repetition_id",
                "donor_layer_norm_mse": "comparison_layer_norm_mse",
                "donor_cosine_similarity": "comparison_cosine_similarity",
            },
            inplace=True,
        )
        if "audit_condition" not in donor:
            donor["audit_condition"] = "matched_followup_swap"
        frames.append(donor)

    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False)
    required_common = [
        "patient_id",
        "fold",
        "transition",
        "audit_condition",
        "native_layer_norm_mse",
        "comparison_layer_norm_mse",
        "latent_error_change",
        "donor_patient_id",
        "repetition_id",
        "matching_distance",
    ]
    output = output.loc[:, [*required_common, *[c for c in LATENT_CHANGE_METRICS if c in output and c not in required_common]]]
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    output["audit_condition"] = output["audit_condition"].astype(str).str.strip()
    output["transition"] = output["transition"].astype(str)
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    if (
        (output["patient_id"] == "").any()
        or (output["audit_condition"] == "").any()
        or not output["fold"].isin(expected_folds).all()
    ):
        raise ValueError("paired/donor latent patient/condition/fold 无效")
    if output.groupby("patient_id", observed=True)["fold"].nunique().gt(1).any():
        raise ValueError("paired/donor latent patient 出现在多折")
    numeric = [column for column in LATENT_CHANGE_METRICS if column in output]
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
        if np.isinf(output[column].to_numpy()).any():
            raise ValueError(f"paired/donor latent {column} 含 infinity")
    for column in ("native_layer_norm_mse", "comparison_layer_norm_mse", "latent_error_change"):
        if not np.isfinite(output[column]).all():
            raise ValueError(f"paired/donor latent {column} 必须全部有限")
    if (output[["native_layer_norm_mse", "comparison_layer_norm_mse"]] < 0).any().any():
        raise ValueError("paired/donor latent error 不得为负")
    expected_change = output["comparison_layer_norm_mse"] - output["native_layer_norm_mse"]
    if not np.allclose(output["latent_error_change"], expected_change, rtol=1e-7, atol=1e-10):
        raise ValueError("latent_error_change 不等于 comparison-native")

    donor_present = output["donor_patient_id"].astype("string").notna()
    repetition_present = pd.to_numeric(output["repetition_id"], errors="coerce").notna()
    distance_present = pd.to_numeric(output["matching_distance"], errors="coerce").notna()
    if not (
        np.array_equal(donor_present.to_numpy(dtype=bool), repetition_present.to_numpy(dtype=bool))
        and np.array_equal(donor_present.to_numpy(dtype=bool), distance_present.to_numpy(dtype=bool))
    ):
        raise ValueError("donor latent provenance 必须成套出现")
    if donor_present.any():
        donor_rows = output.loc[donor_present]
        if donor_rows["donor_patient_id"].astype(str).eq(donor_rows["patient_id"]).any():
            raise ValueError("donor latent 不得 self-match")
        repetitions = pd.to_numeric(donor_rows["repetition_id"]).to_numpy(float)
        distances = pd.to_numeric(donor_rows["matching_distance"]).to_numpy(float)
        if (
            not np.isfinite(repetitions).all()
            or not np.equal(repetitions, np.floor(repetitions)).all()
            or np.any(repetitions <= 0)
            or not np.isfinite(distances).all()
            or np.any(distances < 0)
        ):
            raise ValueError("donor latent repetition/distance 无效")
    key = ["audit_condition", "patient_id", "fold", "transition", "donor_patient_id", "repetition_id"]
    if output.duplicated(key, keep=False).any():
        raise ValueError("paired/donor latent 主键重复")
    return output


def summarize_perturbation_latent_metrics(
    paired_metrics: pd.DataFrame | None,
    donor_metrics: pd.DataFrame | None = None,
    *,
    expected_folds: Iterable[int] = range(5),
) -> PerturbationMetricReport | None:
    """将 paired C/D 与 donor E latent 指标统一后按 patient 等权汇总。"""

    folds = _normalize_expected_folds(expected_folds)
    records = _normalize_paired_latent_records(
        paired_metrics, donor_metrics, expected_folds=folds
    )
    if records.empty:
        return None
    metrics = [column for column in LATENT_CHANGE_METRICS if column in records]
    patient_keys = ["audit_condition", "transition", "fold", "patient_id"]
    patient = (
        records.groupby(patient_keys, sort=True, observed=True, dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    count = (
        records.groupby(patient_keys, sort=True, observed=True, dropna=False)
        .size()
        .rename("n_records")
        .reset_index()
    )
    patient = count.merge(patient, on=patient_keys, validate="one_to_one")
    fold = (
        patient.groupby(["audit_condition", "transition", "fold"], sort=True, observed=True)[metrics]
        .mean()
        .reset_index()
    )
    fold_counts = (
        patient.groupby(["audit_condition", "transition", "fold"], sort=True, observed=True)
        .agg(n_patients=("patient_id", "size"), n_records=("n_records", "sum"))
        .reset_index()
    )
    fold = fold_counts.merge(
        fold, on=["audit_condition", "transition", "fold"], validate="one_to_one"
    )
    summary_frames: list[pd.DataFrame] = []
    pooled_rows: list[dict[str, Any]] = []
    for (condition, transition), group in fold.groupby(
        ["audit_condition", "transition"], sort=True, observed=True
    ):
        summary = fold_mean_sample_std(group, metric_columns=metrics)
        summary.insert(0, "transition", transition)
        summary.insert(0, "audit_condition", condition)
        summary["error_bar_definition"] = "fold metric sample SD (ddof=1)"
        summary_frames.append(summary)
        patient_group = patient.loc[
            patient["audit_condition"].eq(condition) & patient["transition"].eq(transition)
        ]
        row: dict[str, Any] = {
            "audit_condition": condition,
            "transition": transition,
            "n_patients": int(len(patient_group)),
            "n_records": int(patient_group["n_records"].sum()),
            "aggregation": "pooled_equal_patient",
        }
        row.update({metric: float(patient_group[metric].mean()) for metric in metrics})
        pooled_rows.append(row)
    return PerturbationMetricReport(
        records=records,
        patient_metrics=patient,
        fold_metrics=fold,
        fold_summary=pd.concat(summary_frames, ignore_index=True),
        pooled_patient_metrics=pd.DataFrame(pooled_rows),
    )


def _validate_latent_patient_alignment(
    records: pd.DataFrame,
    prediction_report: PredictionReport,
) -> None:
    native = prediction_report.patient_predictions.loc[
        prediction_report.patient_predictions["audit_condition"].eq(
            prediction_report.native_condition
        ),
        ["patient_id", "fold"],
    ].drop_duplicates()
    aligned = records.loc[:, ["patient_id", "fold"]].drop_duplicates().merge(
        native,
        on=["patient_id", "fold"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    if aligned["_merge"].ne("both").any():
        bad = aligned.loc[aligned["_merge"].ne("both"), ["patient_id", "fold"]].head(5)
        raise ValueError(f"latent metrics 无法对齐 native OOF：{bad.to_dict('records')}")


def build_audit_reporting_bundle(
    predictions: pd.DataFrame,
    *,
    copy_latent_metrics: pd.DataFrame | None = None,
    paired_perturbation_metrics: pd.DataFrame | None = None,
    donor_metrics: pd.DataFrame | None = None,
    native_condition: str = "native",
    expected_folds: Iterable[int] = range(5),
    allow_incomplete_conditions: Iterable[str] = (),
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> AuditReportingBundle:
    """统一验证 prediction、copy latent、paired perturbation 和 donor metric。"""

    folds = _normalize_expected_folds(expected_folds)
    prediction = build_prediction_report(
        predictions,
        native_condition=native_condition,
        expected_folds=folds,
        allow_incomplete_conditions=allow_incomplete_conditions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    copy_report = (
        summarize_copy_latent_metrics(
            copy_latent_metrics,
            expected_folds=folds,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
        if copy_latent_metrics is not None
        else None
    )
    perturbation_report = summarize_perturbation_latent_metrics(
        paired_perturbation_metrics,
        donor_metrics,
        expected_folds=folds,
    )
    if copy_report is not None:
        _validate_latent_patient_alignment(copy_report.records, prediction)
    if perturbation_report is not None:
        _validate_latent_patient_alignment(perturbation_report.records, prediction)
    return AuditReportingBundle(
        prediction=prediction,
        copy_latent=copy_report,
        perturbation_latent=perturbation_report,
    )


def plotting_backend_status() -> dict[str, Any]:
    """返回绘图依赖可用性；不导入 pyplot，因而可用于 preflight。"""

    status: dict[str, Any] = {}
    for package in ("matplotlib", "seaborn"):
        specification = importlib.util.find_spec(package)
        status[package] = {
            "available": specification is not None,
            "origin": specification.origin if specification is not None else None,
        }
    font_path = next((path for path in _CJK_FONT_CANDIDATES if path.is_file()), None)
    status["chinese_font"] = {
        "available": font_path is not None,
        "path": str(font_path) if font_path is not None else None,
    }
    status["ready"] = bool(
        status["matplotlib"]["available"]
        and status["seaborn"]["available"]
        and status["chinese_font"]["available"]
    )
    return status


def _plot_modules() -> tuple[Any, Any]:
    status = plotting_backend_status()
    if not status["ready"]:
        missing = [
            name
            for name in ("matplotlib", "seaborn", "chinese_font")
            if not status[name]["available"]
        ]
        raise RuntimeError(f"无法生成 audit figures，缺少依赖：{missing}")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import font_manager

    sns.set_theme(style="whitegrid", context="notebook")
    font_path = Path(status["chinese_font"]["path"])
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams["font.family"] = [font_name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt, sns


def infer_figure_condition_roles(
    conditions: Iterable[str], *, native_condition: str = "native"
) -> FigureConditionRoles:
    """按明确的 condition 命名规则推断八类图的分组。"""

    names = tuple(sorted({_nonempty_text(value, "audit condition") for value in conditions}))
    native_condition = _nonempty_text(native_condition, "native_condition")
    non_native = tuple(value for value in names if value != native_condition)

    def select(predicate: Any) -> tuple[str, ...]:
        return tuple(value for value in non_native if predicate(value.lower().replace("-", "_")))

    repeated = select(lambda value: "repeated_t0" in value or "repeat_t0" in value)
    temporal = select(
        lambda value: "temporal" in value and ("swap" in value or "order" in value)
    )
    followup = select(
        lambda value: (
            "followup_swap" in value
            or "follow_up_swap" in value
            or ("matched" in value and ("donor" in value or "swap" in value))
        )
    )
    baselines = select(
        lambda value: (
            "simplified_baseline" in value
            or value.startswith(("f1_", "f2_", "f3_", "f4_", "f5_"))
        )
    )
    perturbations = tuple(dict.fromkeys((*repeated, *temporal, *followup)))
    return FigureConditionRoles(
        perturbations=perturbations,
        repeated_t0=repeated,
        temporal_order=temporal,
        followup_swap=followup,
        simplified_baselines=baselines,
    )


def _validate_figure_roles(
    roles: FigureConditionRoles,
    conditions: set[str],
    native_condition: str,
) -> None:
    if not isinstance(roles, FigureConditionRoles):
        raise TypeError("roles 必须为 FigureConditionRoles")
    for field in fields(roles):
        values = getattr(roles, field.name)
        if not values:
            raise ValueError(f"必需图表分组 {field.name} 为空")
        unknown = sorted(set(values).difference(conditions))
        if unknown:
            raise ValueError(f"图表分组 {field.name} 含未知 condition：{unknown}")
        if native_condition in values:
            raise ValueError(f"图表分组 {field.name} 不得再包含 native")
    if len(set(roles.simplified_baselines)) < 5:
        raise ValueError("F1–F5/native 对比图要求至少五个不同 simplified baseline")


def _condition_label(condition: str, native_condition: str) -> str:
    if condition == native_condition:
        return "原生 CoRe-WM"
    normalized = condition.lower()
    labels = {
        "f1": "F1 仅临床信息",
        "f2": "F2 仅几何信息",
        "f3": "F3 临床+几何",
        "f4": "F4 仅时间点",
        "f5": "F5 静态 T0 影像",
    }
    for marker, label in labels.items():
        if marker in normalized.split("_") or normalized.startswith(marker):
            return label
    return condition.replace("_", " ")


def _decision_positions(values: Sequence[str]) -> tuple[list[str], np.ndarray]:
    observed = set(values)
    ordered = [decision for decision in DECISION_POINTS if decision in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered, np.arange(len(ordered), dtype=float)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_commit(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"拒绝覆盖已有文件：{destination}") from error
    temporary.unlink()


def _atomic_save_figure(figure: Any, path: Path, *, dpi: int, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有图：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _exclusive_commit(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_write_bytes(
    content: bytes, path: Path, *, overwrite: bool = False
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _exclusive_commit(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _plot_auroc_comparison(
    report: PredictionReport,
    conditions: Sequence[str],
    path: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    plt, _ = _plot_modules()
    selected = [report.native_condition, *conditions]
    summary = report.fold_summary.loc[
        report.fold_summary["audit_condition"].isin(selected)
        & report.fold_summary["metric"].eq("auroc")
    ]
    pooled = report.pooled_oof.loc[report.pooled_oof["audit_condition"].isin(selected)]
    decisions, x = _decision_positions(summary["decision_point"].tolist())
    figure, axis = plt.subplots(figsize=(max(9, 1.7 * len(decisions)), 6))
    palette = plt.get_cmap("tab10")
    for index, condition in enumerate(selected):
        group = summary.loc[summary["audit_condition"].eq(condition)].set_index("decision_point")
        y = np.array([group.loc[d, "mean"] if d in group.index else np.nan for d in decisions])
        error = np.array(
            [group.loc[d, "sample_std"] if d in group.index else np.nan for d in decisions]
        )
        color = "black" if condition == report.native_condition else palette(index % 10)
        axis.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            linewidth=2.4 if condition == report.native_condition else 1.4,
            capsize=4,
            color=color,
            label=f"{_condition_label(condition, report.native_condition)}（折均值±SD）",
        )
        pooled_group = pooled.loc[pooled["audit_condition"].eq(condition)].set_index("decision_point")
        pooled_y = [
            pooled_group.loc[d, "auroc"] if d in pooled_group.index else np.nan
            for d in decisions
        ]
        axis.scatter(x, pooled_y, marker="D", s=35, color=color, alpha=0.85)
    axis.set_xticks(x, decisions)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("AUROC")
    axis.set_xlabel("决策点（菱形 = pooled OOF）")
    axis.set_title("原生序列与各扰动条件的 AUROC 对比")
    axis.legend(fontsize=8, ncol=2, loc="best")
    axis.text(
        0.01,
        -0.19,
        "Error bar：各 fold AUROC 的样本标准差（ddof=1）；黑色线：Native reference。",
        transform=axis.transAxes,
        fontsize=9,
    )
    _atomic_save_figure(figure, path, dpi=dpi, overwrite=overwrite)
    plt.close(figure)
    return {
        "figure_id": "01_auroc_comparison",
        "title": "原生序列与各扰动条件的 AUROC 对比",
        "metric": "AUROC",
        "decision_points": decisions,
        "aggregation": "折均值；菱形为 pooled OOF",
        "error_bar": "各折 AUROC 的样本标准差（ddof=1）",
        "native_reference": "黑色原生序列折均值线",
    }


def _plot_fold_change(
    report: PredictionReport,
    conditions: Sequence[str],
    path: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    plt, _ = _plot_modules()
    data = report.fold_changes.loc[
        report.fold_changes["audit_condition"].isin(conditions)
        & report.fold_changes["metric"].eq("auroc")
    ].copy()
    figure, axis = plt.subplots(figsize=(10, 6))
    markers = {"T0": "o", "T0-T1": "s", "T0-T2": "^"}
    palette = plt.get_cmap("tab10")
    for condition_index, condition in enumerate(conditions):
        for decision in DECISION_POINTS:
            group = data.loc[
                data["audit_condition"].eq(condition)
                & data["decision_point"].eq(decision)
            ].sort_values("fold")
            if group.empty:
                continue
            axis.plot(
                group["fold"],
                group["absolute_difference"],
                marker=markers.get(decision, "o"),
                color=palette(condition_index % 10),
                linestyle="-" if decision == "T0-T2" else "--",
                alpha=0.85,
                label=f"{_condition_label(condition, report.native_condition)} / {decision}",
            )
    axis.axhline(0.0, color="black", linewidth=1.5, label="原生序列参照（差=0）")
    axis.set_xticks(list(report.expected_folds))
    axis.set_xlabel("患者级留出折")
    axis.set_ylabel("配对 AUROC 差（扰动条件 - 原生序列）")
    axis.set_title("各留出折的 AUROC 变化")
    axis.legend(fontsize=7, ncol=2, loc="best")
    axis.text(
        0.01,
        -0.16,
        "每个点为标注 fold 的 held-out patient；无 error bar；水平零线为 Native reference。",
        transform=axis.transAxes,
        fontsize=9,
    )
    _atomic_save_figure(figure, path, dpi=dpi, overwrite=overwrite)
    plt.close(figure)
    return {
        "figure_id": "02_fold_auroc_change",
        "title": "各留出折的 AUROC 变化",
        "metric": "AUROC 差（扰动条件 - 原生序列）",
        "decision_points": list(DECISION_POINTS),
        "aggregation": "按留出折、相同配对患者计算",
        "error_bar": "无（每个点即一个折）",
        "native_reference": "水平零线",
    }


def _plot_copy_error(
    report: LatentMetricReport,
    path: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    plt, _ = _plot_modules()
    transitions = [value for value in COPY_TRANSITIONS if value in set(report.records["transition"])]
    x = np.arange(len(transitions), dtype=float)
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 6))
    definitions = [
        ("learned_layer_norm_mse", "学习到的转移", -width / 2, "#4C72B0"),
        ("copy_layer_norm_mse", "复制当前状态", width / 2, "#DD8452"),
    ]
    for metric, label, offset, color in definitions:
        means: list[float] = []
        errors: list[float] = []
        pooled_values: list[float] = []
        for transition in transitions:
            values = report.fold_metrics.loc[
                report.fold_metrics["transition_scope"].eq(transition), metric
            ].to_numpy(float)
            means.append(float(np.nanmean(values)))
            errors.append(float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() >= 2 else np.nan)
            pooled = report.pooled_metrics.loc[
                report.pooled_metrics["transition_scope"].eq(transition)
                & report.pooled_metrics["aggregation"].eq("pooled_equal_patient"),
                metric,
            ]
            pooled_values.append(float(pooled.iloc[0]))
        axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            color=color,
            alpha=0.82,
            label=f"{label}（折均值±SD）",
        )
        axis.scatter(x + offset, pooled_values, color="black", marker="D", s=30, zorder=4)
    axis.set_xticks(x, transitions)
    axis.set_ylabel("JEPA 特征维 LayerNorm 均方误差")
    axis.set_xlabel("转移阶段（黑色菱形 = pooled 患者等权结果）")
    axis.set_title("学习到的转移与复制当前状态的 latent 误差")
    axis.legend()
    axis.text(
        0.01,
        -0.17,
        "Error bar：各 fold 的 patient-mean latent error 样本标准差（ddof=1）；Copy current 为参照。",
        transform=axis.transAxes,
        fontsize=9,
    )
    _atomic_save_figure(figure, path, dpi=dpi, overwrite=overwrite)
    plt.close(figure)
    return {
        "figure_id": "03_learned_vs_copy_error",
        "title": "学习到的转移与复制当前状态的 latent 误差",
        "metric": "特征维 LayerNorm 均方误差",
        "decision_points": transitions,
        "aggregation": "患者等权的折均值；菱形为 pooled 结果",
        "error_bar": "各折指标的样本标准差（ddof=1）",
        "native_reference": "复制当前状态柱",
    }


def _plot_gain_distribution(
    report: LatentMetricReport,
    path: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    plt, sns = _plot_modules()
    data = report.patient_metrics.loc[
        report.patient_metrics["transition_scope"].isin(COPY_TRANSITIONS)
    ].copy()
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=data,
        x="transition_scope",
        y="normalized_transition_gain",
        order=list(COPY_TRANSITIONS),
        color="#55A868",
        showfliers=False,
        ax=axis,
    )
    sns.stripplot(
        data=data,
        x="transition_scope",
        y="normalized_transition_gain",
        order=list(COPY_TRANSITIONS),
        color="black",
        alpha=0.35,
        size=3,
        ax=axis,
    )
    axis.axhline(0.0, color="#C44E52", linewidth=1.6, linestyle="--")
    axis.set_xlabel("转移阶段")
    axis.set_ylabel("归一化转移增益 G")
    axis.set_title("归一化转移增益 G 的患者级分布")
    axis.text(
        0.01,
        -0.15,
        "每点为一名 patient；无 error bar；G=0 虚线表示 learned 与 copy 等误差。",
        transform=axis.transAxes,
        fontsize=9,
    )
    _atomic_save_figure(figure, path, dpi=dpi, overwrite=overwrite)
    plt.close(figure)
    return {
        "figure_id": "04_transition_gain_distribution",
        "title": "归一化转移增益 G 分布",
        "metric": "G=(copy error - learned error)/(copy error + epsilon)",
        "decision_points": list(COPY_TRANSITIONS),
        "aggregation": "患者级",
        "error_bar": "无（箱线图+单个患者点）",
        "native_reference": "G=0 线（学习转移与复制当前状态等误差）",
    }


def _plot_probability_distribution(
    report: PredictionReport,
    conditions: Sequence[str],
    path: Path,
    *,
    figure_id: str,
    title: str,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    plt, sns = _plot_modules()
    data = report.patient_probability_changes.loc[
        report.patient_probability_changes["audit_condition"].isin(conditions)
    ].copy()
    if data.empty:
        raise ValueError(f"{figure_id} 没有可绘制的 patient probability change")
    data["condition_label"] = data["audit_condition"].map(
        lambda value: _condition_label(value, report.native_condition)
    )
    decisions, _ = _decision_positions(data["decision_point"].tolist())
    figure, axis = plt.subplots(figsize=(max(9, len(decisions) * 2.2), 6))
    sns.boxplot(
        data=data,
        x="decision_point",
        y="probability_change",
        hue="condition_label",
        order=decisions,
        showfliers=False,
        ax=axis,
    )
    axis.axhline(0.0, color="black", linewidth=1.5, linestyle="--")
    axis.set_xlabel("决策点")
    axis.set_ylabel("患者级概率变化（扰动条件 - 原生序列）")
    axis.set_title(title)
    axis.legend(title="Audit condition", fontsize=8, loc="best")
    axis.text(
        0.01,
        -0.16,
        "每个分布以 patient 等权（donor repetitions 先在 patient 内取均值）；无 error bar；零线为 Native reference。",
        transform=axis.transAxes,
        fontsize=9,
    )
    _atomic_save_figure(figure, path, dpi=dpi, overwrite=overwrite)
    plt.close(figure)
    return {
        "figure_id": figure_id,
        "title": title,
        "metric": "患者级概率变化（扰动条件 - 原生序列）",
        "decision_points": decisions,
        "aggregation": "患者等权；donor repetitions 先在患者内取均值",
        "error_bar": "无（箱线分布）",
        "native_reference": "水平零线",
    }


def _plot_baseline_comparison(
    report: PredictionReport,
    conditions: Sequence[str],
    path: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    plt, _ = _plot_modules()
    selected = [*conditions, report.native_condition]
    summary = report.fold_summary.loc[
        report.fold_summary["audit_condition"].isin(selected)
        & report.fold_summary["metric"].eq("auroc")
    ]
    decisions, x = _decision_positions(summary["decision_point"].tolist())
    width = min(0.13, 0.8 / max(1, len(selected)))
    figure, axis = plt.subplots(figsize=(max(11, 2.2 * len(decisions)), 6.5))
    palette = plt.get_cmap("tab10")
    offsets = (np.arange(len(selected)) - (len(selected) - 1) / 2.0) * width
    for index, (condition, offset) in enumerate(zip(selected, offsets)):
        group = summary.loc[summary["audit_condition"].eq(condition)].set_index("decision_point")
        means = [group.loc[d, "mean"] if d in group.index else np.nan for d in decisions]
        errors = [group.loc[d, "sample_std"] if d in group.index else np.nan for d in decisions]
        color = "black" if condition == report.native_condition else palette(index % 10)
        axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=2,
            color=color,
            alpha=0.83,
            label=_condition_label(condition, report.native_condition),
        )
    axis.set_xticks(x, decisions)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("决策点")
    axis.set_ylabel("AUROC（折均值±样本 SD）")
    axis.set_title("F1–F5 简化输入基线与原生 CoRe-WM 性能")
    axis.legend(fontsize=8, ncol=2, loc="best")
    axis.text(
        0.01,
        -0.16,
        "Error bar：各 fold AUROC 样本标准差（ddof=1）；黑色柱为 Native reference；横轴明确标注 decision point。",
        transform=axis.transAxes,
        fontsize=9,
    )
    _atomic_save_figure(figure, path, dpi=dpi, overwrite=overwrite)
    plt.close(figure)
    return {
        "figure_id": "08_f1_f5_native_comparison",
        "title": "F1–F5 与原生 CoRe-WM 性能对比",
        "metric": "AUROC",
        "decision_points": decisions,
        "aggregation": "折均值",
        "error_bar": "各折 AUROC 的样本标准差（ddof=1）",
        "native_reference": "黑色原生序列柱",
    }


def generate_required_figures(
    bundle: AuditReportingBundle,
    output_dir: str | Path,
    *,
    roles: FigureConditionRoles | None = None,
    dpi: int = 180,
    overwrite: bool = False,
) -> FigureGenerationResult:
    """生成规格要求的八类 PNG，并写入中文 JSON manifest。

    默认 ``overwrite=False``。函数会在开始绘图前检查所有目标；如果中途
    失败，会只回滚本次新创建的图，不触及任何先前存在的文件。
    """

    if not isinstance(bundle, AuditReportingBundle):
        raise TypeError("bundle 必须为 AuditReportingBundle")
    if bundle.copy_latent is None:
        raise ValueError("绘制必需图 3/4 需要 copy_latent_metrics")
    if isinstance(dpi, bool) or int(dpi) != dpi or dpi <= 0:
        raise ValueError("dpi 必须是正整数")
    status = plotting_backend_status()
    if not status["ready"]:
        raise RuntimeError(f"绘图环境未就绪：{status}")
    report = bundle.prediction
    conditions = set(report.predictions["audit_condition"])
    roles = roles or infer_figure_condition_roles(
        conditions, native_condition=report.native_condition
    )
    _validate_figure_roles(roles, conditions, report.native_condition)

    destination = Path(output_dir)
    manifest_path = destination / "required_figures_manifest.json"
    targets = [destination / name for name in REQUIRED_FIGURE_FILENAMES]
    if not overwrite:
        existing = [str(path) for path in [*targets, manifest_path] if path.exists()]
        if existing:
            raise FileExistsError(f"拒绝覆盖已有 figure 产物：{existing[:3]}")
    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    metadata: list[dict[str, Any]] = []
    try:
        metadata.append(
            _plot_auroc_comparison(
                report, roles.perturbations, targets[0], dpi=int(dpi), overwrite=overwrite
            )
        )
        created.append(targets[0])
        metadata.append(
            _plot_fold_change(
                report, roles.perturbations, targets[1], dpi=int(dpi), overwrite=overwrite
            )
        )
        created.append(targets[1])
        metadata.append(
            _plot_copy_error(
                bundle.copy_latent, targets[2], dpi=int(dpi), overwrite=overwrite
            )
        )
        created.append(targets[2])
        metadata.append(
            _plot_gain_distribution(
                bundle.copy_latent, targets[3], dpi=int(dpi), overwrite=overwrite
            )
        )
        created.append(targets[3])
        metadata.append(
            _plot_probability_distribution(
                report,
                roles.repeated_t0,
                targets[4],
                figure_id="05_repeated_t0_probability_change",
                title="重复 T0 条件下的患者级概率变化",
                dpi=int(dpi),
                overwrite=overwrite,
            )
        )
        created.append(targets[4])
        metadata.append(
            _plot_probability_distribution(
                report,
                roles.temporal_order,
                targets[5],
                figure_id="06_temporal_swap_probability_change",
                title="时间顺序交换条件下的患者级概率变化",
                dpi=int(dpi),
                overwrite=overwrite,
            )
        )
        created.append(targets[5])
        metadata.append(
            _plot_probability_distribution(
                report,
                roles.followup_swap,
                targets[6],
                figure_id="07_followup_swap_probability_change",
                title="匹配随访交换条件下的患者级概率变化",
                dpi=int(dpi),
                overwrite=overwrite,
            )
        )
        created.append(targets[6])
        metadata.append(
            _plot_baseline_comparison(
                report,
                roles.simplified_baselines,
                targets[7],
                dpi=int(dpi),
                overwrite=overwrite,
            )
        )
        created.append(targets[7])

        for row, path in zip(metadata, targets):
            row.update(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "dpi": int(dpi),
                }
            )
        manifest = {
            "schema_version": "shortcut_audit.required_figures.v1",
            "说明": "图表只展示经 OOF/native 对齐校验的记录。",
            "native_condition": report.native_condition,
            "expected_folds": list(report.expected_folds),
            "error_bar_policy": "fold/sample SD 一律使用 ddof=1；无 error bar 的分布图在图注明确标注。",
            "difference_direction": "comparison - native",
            "artifacts": metadata,
        }
        _atomic_write_bytes(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
            manifest_path,
            overwrite=overwrite,
        )
        created.append(manifest_path)
    except Exception:
        if not overwrite:
            for path in created:
                path.unlink(missing_ok=True)
        raise
    return FigureGenerationResult(
        artifacts=pd.DataFrame(metadata), manifest_path=manifest_path
    )


def _bundle_tables(bundle: AuditReportingBundle) -> dict[str, pd.DataFrame]:
    prediction = bundle.prediction
    output = {
        "prediction_coverage": prediction.coverage,
        "patient_predictions": prediction.patient_predictions,
        "fold_metrics": prediction.fold_metrics,
        "fold_summary": prediction.fold_summary,
        "pooled_oof": prediction.pooled_oof,
        "native_differences": prediction.native_differences,
        "fold_changes": prediction.fold_changes,
        "paired_bootstrap": prediction.paired_bootstrap,
        "bootstrap_samples": prediction.bootstrap_samples,
        "probability_changes": prediction.probability_changes,
        "patient_probability_changes": prediction.patient_probability_changes,
        "repetition_metrics": prediction.repetition_metrics,
        "repetition_summary": prediction.repetition_summary,
    }
    if bundle.copy_latent is not None:
        output.update(
            {
                "copy_patient_metrics": bundle.copy_latent.patient_metrics,
                "copy_fold_metrics": bundle.copy_latent.fold_metrics,
                "copy_fold_summary": bundle.copy_latent.fold_summary,
                "copy_pooled_metrics": bundle.copy_latent.pooled_metrics,
                "copy_bootstrap": bundle.copy_latent.bootstrap_summary,
                "copy_bootstrap_samples": bundle.copy_latent.bootstrap_samples,
            }
        )
    if bundle.perturbation_latent is not None:
        output.update(
            {
                "perturbation_latent_patient_metrics": bundle.perturbation_latent.patient_metrics,
                "perturbation_latent_fold_metrics": bundle.perturbation_latent.fold_metrics,
                "perturbation_latent_fold_summary": bundle.perturbation_latent.fold_summary,
                "perturbation_latent_pooled": bundle.perturbation_latent.pooled_patient_metrics,
            }
        )
    return output


def save_reporting_tables(
    bundle: AuditReportingBundle,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """原子保存全部汇总 CSV 及 hash manifest，默认不覆盖。"""

    if not isinstance(bundle, AuditReportingBundle):
        raise TypeError("bundle 必须为 AuditReportingBundle")
    destination = Path(output_dir)
    tables = _bundle_tables(bundle)
    targets = {name: destination / f"{name}.csv" for name in tables}
    manifest_path = destination / "reporting_tables_manifest.json"
    if not overwrite:
        existing = [str(path) for path in [*targets.values(), manifest_path] if path.exists()]
        if existing:
            raise FileExistsError(f"拒绝覆盖已有 reporting table：{existing[:3]}")
    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    artifacts: list[dict[str, Any]] = []
    try:
        for name, frame in tables.items():
            path = targets[name]
            content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            _atomic_write_bytes(content, path, overwrite=overwrite)
            created.append(path)
            artifacts.append(
                {
                    "table": name,
                    "path": str(path.resolve()),
                    "n_rows": int(len(frame)),
                    "columns": list(frame.columns),
                    "sha256": _sha256(path),
                }
            )
        manifest = {
            "schema_version": "shortcut_audit.reporting_tables.v1",
            "说明": "所有差值为 comparison - native；fold SD 为样本标准差（ddof=1）。",
            "native_condition": bundle.prediction.native_condition,
            "expected_folds": list(bundle.prediction.expected_folds),
            "artifacts": artifacts,
        }
        _atomic_write_bytes(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
            manifest_path,
            overwrite=overwrite,
        )
        created.append(manifest_path)
    except Exception:
        if not overwrite:
            for path in created:
                path.unlink(missing_ok=True)
        raise
    return manifest_path


def load_reporting_frame(
    source: pd.DataFrame | str | Path | Sequence[pd.DataFrame | str | Path],
    *,
    name: str,
) -> pd.DataFrame:
    """从单个/多个 DataFrame 或 CSV 加载并合并正式 reporting 输入。"""

    name = _nonempty_text(name, "name")
    if isinstance(source, pd.DataFrame):
        if source.empty:
            raise ValueError(f"{name} DataFrame 不得为空")
        return source.reset_index(drop=True).copy()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"{name} CSV 不存在：{path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"{name} CSV 为空：{path}")
        return frame
    if isinstance(source, Sequence):
        if not source:
            raise ValueError(f"{name} 输入列表不得为空")
        frames = [load_reporting_frame(value, name=f"{name}[{index}]") for index, value in enumerate(source)]
        return pd.concat(frames, ignore_index=True, sort=False)
    raise TypeError(f"{name} 必须是 DataFrame、CSV path 或它们的序列")


def run_reporting_pipeline(
    predictions: pd.DataFrame | str | Path | Sequence[pd.DataFrame | str | Path],
    output_root: str | Path,
    *,
    copy_latent_metrics: pd.DataFrame
    | str
    | Path
    | Sequence[pd.DataFrame | str | Path],
    paired_perturbation_metrics: pd.DataFrame
    | str
    | Path
    | Sequence[pd.DataFrame | str | Path]
    | None = None,
    donor_metrics: pd.DataFrame
    | str
    | Path
    | Sequence[pd.DataFrame | str | Path]
    | None = None,
    native_condition: str = "native",
    expected_folds: Iterable[int] = range(5),
    allow_incomplete_conditions: Iterable[str] = (),
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
    roles: FigureConditionRoles | None = None,
    dpi: int = 180,
    overwrite: bool = False,
) -> ReportingPipelineResult:
    """一键执行输入加载、五折汇总、bootstrap、CSV 与八类图。

    ``output_root`` 下仅写入 ``metrics/`` 和 ``figures/``。默认拒绝
    覆盖；当默认模式下图表生成失败时，本次新建的汇总 CSV 也会
    被回滚，不会留下看似完整的半套产物。
    """

    prediction_frame = load_reporting_frame(predictions, name="predictions")
    copy_frame = load_reporting_frame(copy_latent_metrics, name="copy_latent_metrics")
    paired_frame = (
        load_reporting_frame(
            paired_perturbation_metrics, name="paired_perturbation_metrics"
        )
        if paired_perturbation_metrics is not None
        else None
    )
    donor_frame = (
        load_reporting_frame(donor_metrics, name="donor_metrics")
        if donor_metrics is not None
        else None
    )
    bundle = build_audit_reporting_bundle(
        prediction_frame,
        copy_latent_metrics=copy_frame,
        paired_perturbation_metrics=paired_frame,
        donor_metrics=donor_frame,
        native_condition=native_condition,
        expected_folds=expected_folds,
        allow_incomplete_conditions=allow_incomplete_conditions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    # 写入前先完成图表分组与环境 preflight。
    inferred_roles = roles or infer_figure_condition_roles(
        bundle.prediction.predictions["audit_condition"].unique(),
        native_condition=bundle.prediction.native_condition,
    )
    _validate_figure_roles(
        inferred_roles,
        set(bundle.prediction.predictions["audit_condition"]),
        bundle.prediction.native_condition,
    )
    if not plotting_backend_status()["ready"]:
        raise RuntimeError(f"绘图环境未就绪：{plotting_backend_status()}")

    root = Path(output_root)
    table_manifest = save_reporting_tables(
        bundle, root / "metrics", overwrite=overwrite
    )
    try:
        figures = generate_required_figures(
            bundle,
            root / "figures",
            roles=inferred_roles,
            dpi=dpi,
            overwrite=overwrite,
        )
    except Exception:
        if not overwrite and table_manifest.is_file():
            with table_manifest.open(encoding="utf-8") as stream:
                table_metadata = json.load(stream)
            for artifact in table_metadata.get("artifacts", []):
                Path(artifact["path"]).unlink(missing_ok=True)
            table_manifest.unlink(missing_ok=True)
        raise
    return ReportingPipelineResult(
        bundle=bundle,
        table_manifest_path=table_manifest,
        figures=figures,
    )


__all__ = [
    "AuditReportingBundle",
    "COPY_TRANSITIONS",
    "FigureConditionRoles",
    "FigureGenerationResult",
    "LatentMetricReport",
    "PerturbationMetricReport",
    "PredictionReport",
    "REQUIRED_FIGURE_FILENAMES",
    "REPORTING_INPUT_CONTRACTS",
    "ReportingPipelineResult",
    "ValidatedPredictionSet",
    "build_audit_reporting_bundle",
    "build_prediction_report",
    "generate_required_figures",
    "infer_figure_condition_roles",
    "load_reporting_frame",
    "plotting_backend_status",
    "save_reporting_tables",
    "run_reporting_pipeline",
    "summarize_copy_latent_metrics",
    "summarize_perturbation_latent_metrics",
    "validate_copy_latent_metrics",
    "validate_oof_predictions",
]
