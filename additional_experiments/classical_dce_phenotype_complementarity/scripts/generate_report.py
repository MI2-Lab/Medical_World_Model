#!/usr/bin/env python3
"""Build the Chinese final report from aggregate metrics and figures.

The report is deliberately evidence-constrained: it reads the preregistration,
aggregate CSVs, and the seven generated figures, but never patient-level OOF
predictions or source workbooks.  Missing required evidence is an error rather
than a cue to fill prose with placeholders.  Git delivery fields are the sole
exception and are explicitly rendered as ``PENDING`` until a delivery
provenance file is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FILTERS = {
    "protocol": "primary_stratified_384",
    "population": "clinical_radiomics_complete_384",
    "scenario": "complete_case",
}
TIMINGS = ("T0", "T1", "T2", "T3")
VIEWS = ("static", "longitudinal")
TIMING_LABELS = {
    "T0": "T0",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3 (late/pre-surgery)",
}
COMPARISONS = (
    "C+FULL_vs_C+F",
    "C+N_vs_C",
    "C+F+N_res_vs_C+F",
)
FIGURES = (
    ("timing_auroc.png", "各 timing 的 pCR AUROC"),
    ("c_f_vs_c_full_auroc.png", "C+F 与 C+FULL"),
    ("delta_auroc_forest.png", "配对 bootstrap ΔAUROC forest"),
    ("phenotype_family_comparison.png", "phenotype family ablation"),
    ("hr_her2_heatmap.png", "HR/HER2 phenotype probe heatmap"),
    ("residualized_results.png", "FTV residualization"),
    ("feature_correlation_matrix.png", "feature correlation matrix"),
)

_FORBIDDEN_PATIENT_LEVEL_COLUMNS = {
    "patient_id",
    "trial_id",
    "subject_id",
    "clinical-trial-subject-id",
    "y_true",
    "y_score",
    "predicted_probability",
    "predicted_label",
    "prediction_probability",
}


class ReportDataError(ValueError):
    """Raised when required report evidence is missing or inconsistent."""


def _read_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required aggregate {label}: {path}. Complete aggregation "
            "before generating the report."
        )
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise ReportDataError(f"Could not read {label} at {path}: {exc}") from exc
    if frame.empty:
        raise ReportDataError(f"Required aggregate {label} is empty: {path}")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ReportDataError(
            f"{label} is missing required columns {missing}; observed {list(frame.columns)}"
        )
    forbidden = sorted(
        _FORBIDDEN_PATIENT_LEVEL_COLUMNS.intersection(
            str(column).strip().lower() for column in frame.columns
        )
    )
    if forbidden:
        raise ReportDataError(
            f"{label} contains patient-level columns {forbidden}; final reporting "
            "accepts aggregate artifacts only"
        )
    return frame


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReportDataError(f"Could not parse {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportDataError(f"{label} must be a JSON object: {path}")
    return value


def _filter_value(frame: pd.DataFrame, column: str, value: str, label: str) -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    mask = frame[column].astype(str).str.strip().str.casefold().eq(value.casefold())
    selected = frame.loc[mask].copy()
    if selected.empty:
        observed = sorted(frame[column].dropna().astype(str).unique().tolist())
        raise ReportDataError(
            f"{label} has no {column}={value!r} rows; observed {observed}"
        )
    return selected


def _primary(frame: pd.DataFrame, label: str, *, logistic: bool = True) -> pd.DataFrame:
    selected = frame.copy()
    for column, value in PRIMARY_FILTERS.items():
        selected = _filter_value(selected, column, value, label)
    if logistic:
        selected = _filter_value(selected, "model_type", "logistic", label)
    return selected


def _numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        except Exception as exc:
            raise ReportDataError(f"{label}.{column} must be numeric") from exc
        if not np.isfinite(values).all():
            raise ReportDataError(f"{label}.{column} contains non-finite values")
        if lower is not None and np.any(values < lower):
            raise ReportDataError(f"{label}.{column} contains values below {lower}")
        if upper is not None and np.any(values > upper):
            raise ReportDataError(f"{label}.{column} contains values above {upper}")


def _require_values(
    frame: pd.DataFrame, column: str, values: Iterable[str], label: str
) -> None:
    observed = set(frame[column].dropna().astype(str))
    missing = [value for value in values if value not in observed]
    if missing:
        raise ReportDataError(
            f"{label} lacks required {column} values {missing}; observed {sorted(observed)}"
        )


def _resolve(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ReportDataError(
        f"{label} requires one of columns {list(candidates)}; observed {list(frame.columns)}"
    )


def _effects(frame: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, str] = {}
    for metric in ("auroc", "auprc", "brier"):
        estimate_candidates = (
            ("brier_improvement", "delta_brier", "delta_brier_estimate", "brier_delta")
            if metric == "brier"
            else (f"delta_{metric}", f"delta_{metric}_estimate", f"{metric}_delta")
        )
        low_candidates = (
            (
                "brier_improvement_ci_low",
                "brier_improvement_ci_lower",
                "brier_improvement_ci95_low",
                "delta_brier_ci_low",
                "delta_brier_ci_lower",
                "delta_brier_ci95_low",
                "delta_brier_low",
                "brier_ci_low",
            )
            if metric == "brier"
            else (
                f"delta_{metric}_ci_low",
                f"delta_{metric}_ci_lower",
                f"delta_{metric}_ci95_low",
                f"delta_{metric}_low",
                f"{metric}_ci_low",
            )
        )
        high_candidates = (
            (
                "brier_improvement_ci_high",
                "brier_improvement_ci_upper",
                "brier_improvement_ci95_high",
                "delta_brier_ci_high",
                "delta_brier_ci_upper",
                "delta_brier_ci95_high",
                "delta_brier_high",
                "brier_ci_high",
            )
            if metric == "brier"
            else (
                f"delta_{metric}_ci_high",
                f"delta_{metric}_ci_upper",
                f"delta_{metric}_ci95_high",
                f"delta_{metric}_high",
                f"{metric}_ci_high",
            )
        )
        columns[f"_{metric}"] = _resolve(
            frame,
            estimate_candidates,
            f"incremental {metric} estimate",
        )
        columns[f"_{metric}_low"] = _resolve(
            frame,
            low_candidates,
            f"incremental {metric} lower CI",
        )
        columns[f"_{metric}_high"] = _resolve(
            frame,
            high_candidates,
            f"incremental {metric} upper CI",
        )
    result = frame.copy()
    for canonical, source in columns.items():
        result[canonical] = pd.to_numeric(result[source], errors="raise")
    _numeric(result, columns, "incremental effects")
    for metric in ("auroc", "auprc", "brier"):
        if (result[f"_{metric}_low"] > result[f"_{metric}_high"]).any():
            raise ReportDataError(f"incremental {metric} lower CI exceeds upper CI")
        if (
            (result[f"_{metric}"] < result[f"_{metric}_low"] - 1e-12)
            | (result[f"_{metric}"] > result[f"_{metric}_high"] + 1e-12)
        ).any():
            raise ReportDataError(
                f"incremental {metric} point estimate lies outside its reported CI"
            )
    return result


def _link(label: str, target: Path, report_path: Path) -> str:
    relative = os.path.relpath(target, report_path.parent).replace(os.sep, "/")
    return f"[{label}]({relative})"


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _cell(row: pd.Series) -> str:
    return f"{row.get('view', 'NA')}/{TIMING_LABELS.get(str(row.get('timing')), str(row.get('timing')))}"


def _best(frame: pd.DataFrame, metric: str = "auroc") -> pd.Series:
    if frame.empty:
        raise ReportDataError(f"Cannot summarize empty evidence for {metric}")
    values = pd.to_numeric(frame[metric], errors="raise")
    return frame.loc[values.idxmax()]


def _effect_rows(frame: pd.DataFrame, comparison: str) -> pd.DataFrame:
    rows = frame.loc[frame["comparison"].astype(str).eq(comparison)].copy()
    if rows.empty:
        raise ReportDataError(f"incremental effects lack comparison {comparison}")
    return rows


def _effect_summary(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values(["view", "timing"])
    strong = ordered.loc[ordered["_auroc_low"] > 0]
    positive_cells = [f"{row.view}/{TIMING_LABELS.get(str(row.timing), row.timing)}" for row in strong.itertuples()]
    estimates = pd.to_numeric(ordered["_auroc"])
    return (
        f"ΔAUROC 范围 {_fmt(estimates.min())} 至 {_fmt(estimates.max())}；"
        f"95% CI 全高于 0 的 cell 为 "
        + ("、".join(positive_cells) if positive_cells else "无")
        + "。"
    )


def _inventory_summary(inventory: pd.DataFrame) -> tuple[str, str, str]:
    independent = inventory.loc[
        inventory["independent_measurement"].astype(str).str.casefold().isin(("true", "1", "yes"))
    ].copy()
    families = ("FTV", "LD", "SPH", "BPE")
    rows = []
    definitions = {
        "FTV": "functional tumor volume（F）",
        "LD": "longest diameter（D）",
        "SPH": "sphericity / shape（S）",
        "BPE": "contralateral 5-slice mean enhancement（B）",
    }
    for family in families:
        subset = independent.loc[independent["family"].astype(str).eq(family)]
        if subset.empty:
            raise ReportDataError(f"feature inventory has no independent {family} measurements")
        visits = ", ".join(sorted(subset["visit"].astype(str).unique(), key=lambda x: (len(x), x)))
        rows.append((family, definitions[family], len(subset), visits))
    derived = inventory.loc[
        inventory["role"].astype(str).eq("derived_baseline_percent_change")
    ]
    family_table = _markdown_table(
        ("Family", "真实 measurement", "独立绝对值列数", "访视"), rows
    )
    actual = (
        f"工作簿 inventory 识别出 {len(independent)} 个独立 imaging measurement："
        "FTV、LD、SPH、BPE 各覆盖 T0–T3；另有 "
        f"{len(derived)} 个相对 T0 的派生 percent-change 列，它们不算新的独立观测。"
    )
    grouping = (
        "F=FTV；NONFTV=N=D+S+B（LD、SPH、BPE）；FULL=F+NONFTV。"
        "Patient ID 只用于 join/split，绝不进入 predictor。"
    )
    return actual, grouping, family_table


def _family_summary(family: pd.DataFrame) -> tuple[str, str]:
    primary = _primary(family, "family ablation")
    _require_values(primary, "model", ("C+F", "C+F+D", "C+F+S", "C+F+B"), "family ablation")
    reference = primary.loc[primary["model"].eq("C+F"), ["view", "timing", "auroc"]].rename(
        columns={"auroc": "reference_auroc"}
    )
    candidates = primary.loc[primary["model"].isin(("C+F+D", "C+F+S", "C+F+B"))]
    merged = candidates.merge(reference, on=["view", "timing"], how="inner", validate="many_to_one")
    if len(merged) != len(candidates):
        raise ReportDataError("family ablation cannot be paired to C+F for every view/timing")
    merged["delta"] = pd.to_numeric(merged["auroc"]) - pd.to_numeric(merged["reference_auroc"])
    summary = (
        merged.groupby("model", sort=False)["delta"]
        .agg(["mean", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    best_model = str(summary["mean"].idxmax())
    rows = [
        (model.replace("C+F+", ""), _fmt(values["mean"]), _fmt(values["min"]), _fmt(values["max"]))
        for model, values in summary.iterrows()
    ]
    text = (
        f"按所有预注册 view/timing cell 的配对 AUROC 差均值，{best_model.replace('C+F+', '')} "
        "排名最高；这是 family 定位描述，不是根据 test 表现重新选择主模型。"
    )
    return text, _markdown_table(("Family", "mean ΔAUROC", "min", "max"), rows)


def _lr_svm_summary(pcr: pd.DataFrame, lr_svm: pd.DataFrame) -> tuple[str, str]:
    all_primary = _primary(pcr, "pCR metrics", logistic=False)
    _require_values(all_primary, "model_type", ("logistic", "rbf_svm"), "pCR metrics")
    increments: dict[str, float] = {}
    for model_type in ("logistic", "rbf_svm"):
        subset = all_primary.loc[all_primary["model_type"].eq(model_type)]
        baseline = subset.loc[subset["model"].eq("C+F"), ["view", "timing", "auroc"]].rename(
            columns={"auroc": "baseline"}
        )
        augmented = subset.loc[
            subset["model"].eq("C+FULL"), ["view", "timing", "auroc"]
        ].rename(columns={"auroc": "augmented"})
        paired = augmented.merge(baseline, on=["view", "timing"], validate="one_to_one")
        if paired.empty:
            raise ReportDataError(f"No paired C+FULL/C+F cells for {model_type}")
        increments[model_type] = float(
            (pd.to_numeric(paired["augmented"]) - pd.to_numeric(paired["baseline"])).mean()
        )
    logistic_column = _resolve(
        lr_svm,
        ("logistic_auroc", "auroc__logistic"),
        "LR AUROC",
    )
    svm_column = _resolve(
        lr_svm,
        ("svm_auroc", "auroc__rbf_svm"),
        "SVM AUROC",
    )
    delta_column = _resolve(
        lr_svm,
        ("delta_svm_minus_lr", "svm_minus_logistic_auroc", "delta_auroc"),
        "LR vs SVM delta",
    )
    # Family-ablation rows are LR-only by design and can appear in the wide
    # table with empty SVM cells.  The sensitivity conclusion is restricted to
    # the seven models that were actually run with both algorithms.
    comparable_models = ("C", "F", "N", "FULL", "C+F", "C+N", "C+FULL")
    comparable = lr_svm.loc[lr_svm["model"].astype(str).isin(comparable_models)].copy()
    _require_values(comparable, "model", comparable_models, "LR vs SVM")
    _numeric(comparable, (logistic_column, svm_column, delta_column), "LR vs SVM")
    median_model_delta = float(pd.to_numeric(comparable[delta_column]).median())
    same_direction = np.sign(increments["logistic"]) == np.sign(increments["rbf_svm"])
    verdict = "一致" if same_direction else "不一致"
    text = (
        f"C+FULL−C+F 的跨 cell 平均差：LR {_fmt(increments['logistic'])}，"
        f"RBF-SVM {_fmt(increments['rbf_svm'])}，因此增量方向{verdict}。"
        f"逐模型 SVM−LR AUROC 的中位数为 {_fmt(median_model_delta)}；"
        "SVM 是 secondary sensitivity，不改变 LR primary classification。"
    )
    table = _markdown_table(
        ("模型", "mean C+FULL−C+F AUROC"),
        (("Logistic regression", _fmt(increments["logistic"])), ("RBF SVM", _fmt(increments["rbf_svm"]))),
    )
    return text, table


def _metric_column(frame: pd.DataFrame) -> str | None:
    for column in ("auroc", "auroc_mean"):
        if column in frame.columns:
            return column
    return None


def _n_column(frame: pd.DataFrame) -> str | None:
    for column in ("n", "n_patients", "n_patients_per_cell"):
        if column in frame.columns:
            return column
    return None


def _assess_mri(
    mri_frames: Mapping[Path, pd.DataFrame], pcr_metrics: pd.DataFrame
) -> tuple[str, str, str, str]:
    """Return status, evidence prose, and a conservative Q10 answer."""

    differences: list[float] = []
    cell_labels: list[str] = []
    evidence: list[str] = []
    for path, frame in mri_frames.items():
        if {"traditional_auroc", "mri_auroc"}.issubset(frame.columns):
            if "population" not in frame.columns or not frame["population"].astype(str).eq("mri_matched_375").all():
                raise ReportDataError(
                    f"{path.name} direct MRI comparison is not labeled mri_matched_375"
                )
            if "n" not in frame.columns or not pd.to_numeric(
                frame["n"], errors="raise"
            ).eq(375).all():
                raise ReportDataError(
                    f"{path.name} direct MRI comparison is not matched at n=375"
                )
            comparable = frame
            if {"traditional_model", "mri_model"}.issubset(frame.columns):
                n_vs_m = frame.loc[
                    frame["traditional_model"].astype(str).eq("N")
                    & frame["mri_model"].astype(str).eq("M")
                ]
                if not n_vs_m.empty:
                    comparable = n_vs_m
            traditional = pd.to_numeric(comparable["traditional_auroc"], errors="raise")
            mri = pd.to_numeric(comparable["mri_auroc"], errors="raise")
            if not (np.isfinite(traditional).all() and np.isfinite(mri).all()):
                raise ReportDataError(f"{path.name} has non-finite MRI comparison metrics")
            if "difference_mri_minus_traditional" in comparable.columns:
                recorded = pd.to_numeric(
                    comparable["difference_mri_minus_traditional"], errors="raise"
                )
                if not np.allclose(
                    recorded.to_numpy(float),
                    (mri - traditional).to_numpy(float),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ReportDataError(
                        f"{path.name} MRI-minus-traditional differences are inconsistent"
                    )
            differences.extend((mri - traditional).tolist())
            for _, row in comparable.iterrows():
                parts = [
                    str(row[column])
                    for column in ("task", "target", "view", "timing")
                    if column in comparable.columns and pd.notna(row[column])
                ]
                parts = [value for index, value in enumerate(parts) if index == 0 or value != parts[index - 1]]
                cell_labels.append("/".join(parts) or path.stem)
            evidence.append(
                f"{path.name} 的 N-vs-M direct matched descriptive comparison"
            )
            continue
        metric = _metric_column(frame)
        if metric is None or "model" not in frame.columns:
            continue
        n_column = _n_column(frame)
        key_candidates = ("population", "target", "view", "timing")
        keys = [column for column in key_candidates if column in frame.columns]
        if not keys:
            continue
        pairs = (("M", "N"), ("C+M", "C+N"), ("C+F+M", "C+FULL"))
        for mri_model, traditional_model in pairs:
            left = frame.loc[frame["model"].astype(str).eq(mri_model)]
            right = frame.loc[frame["model"].astype(str).eq(traditional_model)]
            if left.empty or right.empty:
                continue
            keep_left = keys + [metric] + ([n_column] if n_column else [])
            keep_right = keys + [metric] + ([n_column] if n_column else [])
            paired = left[keep_left].merge(
                right[keep_right], on=keys, suffixes=("_mri", "_traditional")
            )
            if n_column:
                paired = paired.loc[
                    pd.to_numeric(paired[f"{n_column}_mri"])
                    .eq(pd.to_numeric(paired[f"{n_column}_traditional"]))
                ]
            if not paired.empty:
                differences.extend(
                    (
                        pd.to_numeric(paired[f"{metric}_mri"])
                        - pd.to_numeric(paired[f"{metric}_traditional"])
                    ).tolist()
                )
                for _, row in paired.iterrows():
                    parts = [str(row[column]) for column in keys if pd.notna(row[column])]
                    cell_labels.append("/".join(parts) or f"{mri_model}-vs-{traditional_model}")
                evidence.append(f"{path.name}: {mri_model} vs {traditional_model}")
    # The frozen LOCAL reference is summarized across four seed x arm cells,
    # while the experiment runner fits traditional radiomics on the exact
    # locked 375-person manifest.  Their population labels intentionally differ
    # by provenance, so pair them explicitly only after n and timing match.
    if not differences:
        traditional = pcr_metrics.copy()
        for column, value in (
            ("protocol", "locked_mri_manifest_375"),
            ("population", "mri_matched_375"),
            ("scenario", "complete_case"),
            ("view", "longitudinal"),
            ("model_type", "logistic"),
        ):
            if column in traditional.columns:
                traditional = traditional.loc[
                    traditional[column].astype(str).str.casefold().eq(value.casefold())
                ]
        for path, frame in mri_frames.items():
            metric = _metric_column(frame)
            n_column = _n_column(frame)
            if metric is None or n_column is None or "model" not in frame.columns or "timing" not in frame.columns:
                continue
            if "target" in frame.columns:
                frame = frame.loc[frame["target"].astype(str).str.casefold().eq("pcr")]
            for mri_model, traditional_model in (("M", "N"),):
                left = frame.loc[
                    frame["model"].astype(str).eq(mri_model),
                    ["timing", n_column, metric],
                ].rename(columns={n_column: "n_mri", metric: "auroc_mri"})
                right = traditional.loc[
                    traditional["model"].astype(str).eq(traditional_model),
                    ["timing", "n", "auroc"],
                ].rename(columns={"n": "n_traditional", "auroc": "auroc_traditional"})
                paired = left.merge(right, on="timing", validate="one_to_one")
                paired = paired.loc[
                    pd.to_numeric(paired["n_mri"]).eq(
                        pd.to_numeric(paired["n_traditional"])
                    )
                ]
                if paired.empty:
                    continue
                differences.extend(
                    (
                        pd.to_numeric(paired["auroc_mri"])
                        - pd.to_numeric(paired["auroc_traditional"])
                    ).tolist()
                )
                cell_labels.extend(
                    f"pCR/{timing}/{mri_model}-vs-{traditional_model}"
                    for timing in paired["timing"].astype(str)
                )
                evidence.append(
                    f"{path.name}: {mri_model} vs {traditional_model} on locked n=375"
                )
    if not differences:
        return (
            "not_evaluable",
            "现有 MRI aggregate reference 没有可配对的同一人群 N-vs-M cell；不跨 384 与 375 人作数值排名。",
            "不能严谨判定。",
            "没有可配对的 matched N-vs-M cell。",
        )
    values = np.asarray(differences, dtype=float)
    if np.all(values >= 0):
        status = "at_least"
        verdict = "是"
    elif np.all(values < 0):
        status = "below"
        verdict = "否"
    else:
        status = "mixed"
        verdict = "不同 matched cell 结论混合"
    if len(cell_labels) != len(values):
        cell_labels = [f"cell_{index + 1}" for index in range(len(values))]
    traditional_better = [
        (label, value) for label, value in zip(cell_labels, values) if value < 0
    ]
    strongest = sorted(traditional_better, key=lambda item: item[1])[:3]
    strongest_text = "、".join(
        f"{label} ({value:+.3f})" for label, value in strongest
    )
    gap_summary = (
        f"traditional N 高于 M 的 matched cell 为 {len(traditional_better)}/{len(values)}"
        + (f"，最大三个差距为 {strongest_text}" if strongest_text else "")
    )
    detail = (
        f"可配对 MRI−traditional AUROC 共 {len(values)} 个 cell，范围 "
        f"{_fmt(values.min())} 至 {_fmt(values.max())}；{gap_summary}；"
        f"证据：{'；'.join(evidence)}。"
    )
    return status, detail, f"{verdict}。", gap_summary


def _delivery(path: Path) -> Mapping[str, str]:
    if not path.is_file():
        return {"branch": "PENDING", "commit_sha": "PENDING", "push_status": "PENDING", "push_error": "PENDING"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReportDataError(f"Could not parse delivery provenance {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportDataError("delivery_provenance.json must be a JSON object")
    push = value.get("push") if isinstance(value.get("push"), dict) else {}

    def first(*keys: str) -> str:
        for key in keys:
            candidate = value.get(key)
            if candidate not in (None, ""):
                return str(candidate)
        return "PENDING"

    push_status = (
        first("push_status", "status")
        if not push
        else str(push.get("status") or first("push_status", "status"))
    )
    raw_push_error = (
        value.get("push_error", value.get("error"))
        if not push
        else push.get("error", value.get("push_error", value.get("error")))
    )
    push_error = str(raw_push_error) if raw_push_error not in (None, "") else (
        "NONE"
        if push_status.upper() in {"PUSHED", "SUCCESS", "PUSH_SUCCESS", "GITHUB_PUSH_SUCCESS"}
        else "PENDING"
    )
    return {
        "branch": first("branch", "git_branch"),
        "commit_sha": first("commit_sha", "sha", "commit", "git_commit"),
        "push_status": push_status,
        "push_error": push_error,
    }


def load_report_inputs(root: Path, figures_dir: Path) -> Mapping[str, Any]:
    metrics = root / "metrics"
    features = root / "features"
    config_path = root / "configs" / "experiment.json"
    config = _read_json(config_path, "experiment config")
    rules = config.get("classification_rule")
    required_rules = {
        "supported_increment",
        "any_increment_signal",
        "residual_signal",
        "standalone_signal",
        "profile_signal",
        "b_ftv_redundant",
        "c_profile_only",
        "d_weak",
        "mixed_rule",
    }
    if not isinstance(rules, dict) or not required_rules.issubset(rules):
        raise ReportDataError(
            f"experiment config classification_rule must contain {sorted(required_rules)}"
        )
    if int(config.get("bootstrap_draws", 0)) < 2000:
        raise ReportDataError("experiment config must preregister at least 2,000 bootstrap draws")

    inventory = _read_csv(
        features / "radiomics_feature_inventory.csv",
        ("column", "family", "visit", "role", "independent_measurement", "missingness_pct"),
        "feature inventory",
    )
    missingness_raw = _read_csv(
        metrics / "missingness.csv",
        (
            "scope",
            "column",
            "family",
            "visit",
            "n_patients",
            "n_valid_patients",
            "n_missing_patients",
        ),
        "missingness table",
    )
    # Normalize the inventory-oriented names once; downstream prose uses the
    # population/count vocabulary of the analysis tables.
    missingness = missingness_raw.rename(
        columns={
            "scope": "population",
            "n_patients": "n_population",
            "n_valid_patients": "n_observed",
            "n_missing_patients": "n_missing",
        }
    )
    matched = _read_csv(
        metrics / "matched_population_manifest.csv",
        ("comparison", "n"),
        "matched-population manifest",
    )
    root_matched = _read_csv(
        root / "matched_population_manifest.csv",
        ("comparison", "n"),
        "root matched-population manifest mirror",
    )
    if not root_matched.equals(matched):
        raise ReportDataError(
            "root and metrics matched_population_manifest.csv copies differ"
        )
    pcr = _read_csv(
        metrics / "pcr_oof_metrics.csv",
        (
            "protocol", "population", "scenario", "view", "timing", "timing_label",
            "model_type", "model", "n", "n_positive", "auroc", "auprc",
            "balanced_accuracy", "brier",
        ),
        "pCR metrics",
    )
    pcr_table_columns = (
        "protocol",
        "population",
        "scenario",
        "view",
        "timing",
        "timing_label",
        "model_type",
        "model",
        "n",
        "n_positive",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "brier",
    )
    _read_csv(metrics / "static_radiomics.csv", pcr_table_columns, "static radiomics table")
    _read_csv(
        metrics / "longitudinal_radiomics.csv",
        pcr_table_columns,
        "longitudinal radiomics table",
    )
    incremental_raw = _read_csv(
        metrics / "incremental_effects.csv",
        ("comparison", "view", "timing", "n", "n_positive", "n_bootstrap"),
        "incremental effects",
    )
    profile = _read_csv(
        metrics / "profile_oof_metrics.csv",
        ("view", "timing", "feature_set", "model_type", "target", "n", "auroc", "auprc", "balanced_accuracy"),
        "profile metrics",
    )
    redundancy = _read_csv(
        metrics / "redundancy_metrics.csv", ("view", "timing", "n", "r2", "spearman"), "redundancy metrics"
    )
    residual = _read_csv(
        metrics / "residualization_metrics.csv",
        ("view", "timing", "model_type", "model", "n", "auroc", "auprc", "balanced_accuracy", "brier"),
        "residualization metrics",
    )
    family = _read_csv(
        metrics / "family_ablation_metrics.csv",
        ("view", "timing", "model_type", "model", "n", "auroc", "auprc", "balanced_accuracy", "brier"),
        "family ablation metrics",
    )
    lr_svm = _read_csv(
        metrics / "lr_vs_svm.csv",
        ("model", "view", "timing", "logistic_auroc", "svm_auroc", "delta_svm_minus_lr"),
        "LR vs SVM metrics",
    )
    correlation = _read_csv(metrics / "feature_correlation_matrix.csv", (), "feature correlation matrix")
    timing = _read_csv(
        root / "information_timing_contract.csv", ("timing", "allowed_visits", "label"), "timing contract"
    )
    t3 = timing.loc[timing["timing"].astype(str).eq("T3"), "label"].astype(str)
    if t3.empty or not t3.str.casefold().str.contains("late/pre-surgery", regex=False).all():
        raise ReportDataError("timing contract must label T3 as late/pre-surgery")

    primary_pcr = _primary(pcr, "pCR metrics")
    _numeric(primary_pcr, ("n", "n_positive"), "pCR metrics", lower=0)
    _numeric(primary_pcr, ("auroc", "auprc", "balanced_accuracy", "brier"), "pCR metrics", lower=0, upper=1)
    _require_values(primary_pcr, "model", ("C", "F", "N", "FULL", "C+F", "C+N", "C+FULL"), "pCR metrics")
    _require_values(primary_pcr, "view", VIEWS, "pCR metrics")
    _require_values(primary_pcr, "timing", TIMINGS, "pCR metrics")
    # Matched comparisons must have identical n within every primary view/timing.
    for (view, timing_value), cell in primary_pcr.groupby(["view", "timing"], sort=False):
        for pair in (("C+F", "C+FULL"), ("C", "C+N")):
            rows = cell.loc[cell["model"].isin(pair)]
            if len(rows) != 2 or rows["n"].nunique() != 1:
                raise ReportDataError(
                    f"primary matched-n contract failed for {pair} at {view}/{timing_value}"
                )

    incremental = _primary(_effects(incremental_raw), "incremental effects")
    _numeric(incremental, ("n", "n_positive", "n_bootstrap"), "incremental effects", lower=0)
    if (pd.to_numeric(incremental["n_bootstrap"]) < 2000).any():
        raise ReportDataError("incremental effects must use at least 2,000 bootstrap draws")
    _require_values(incremental, "comparison", COMPARISONS, "incremental effects")
    _require_values(incremental, "view", VIEWS, "incremental effects")
    _require_values(incremental, "timing", TIMINGS, "incremental effects")
    incremental = incremental.loc[
        incremental["comparison"].isin(COMPARISONS)
        & incremental["view"].isin(VIEWS)
        & incremental["timing"].isin(TIMINGS)
    ].copy()
    primary_profile = _primary(profile, "profile metrics")
    _numeric(primary_profile, ("n",), "profile metrics", lower=0)
    _numeric(primary_profile, ("auroc", "auprc", "balanced_accuracy"), "profile metrics", lower=0, upper=1)
    _require_values(primary_profile, "feature_set", ("N", "FULL"), "profile metrics")
    _require_values(primary_profile, "target", ("HR", "HER2"), "profile metrics")
    primary_residual = _primary(residual, "residualization metrics")
    _require_values(primary_residual, "model", ("N_res", "C+F+N_res"), "residualization metrics")
    _numeric(redundancy, ("n", "r2", "spearman"), "redundancy metrics")
    _numeric(primary_residual, ("auroc", "auprc", "balanced_accuracy", "brier"), "residualization metrics", lower=0, upper=1)

    figure_paths = [figures_dir / filename for filename, _ in FIGURES]
    absent_figures = [path for path in figure_paths if not path.is_file() or path.stat().st_size == 0]
    if absent_figures:
        raise FileNotFoundError(
            "Missing required generated figures: " + ", ".join(str(path) for path in absent_figures)
        )

    mri_paths = sorted(metrics.glob("mri_reference*.csv"))
    mri_frames = {path: _read_csv(path, (), f"MRI reference {path.name}") for path in mri_paths}
    all_metric_tables = sorted(metrics.glob("*.csv"))
    # Link every aggregate table produced by the runner, not only those used in
    # the A-D mapping.  Re-reading here is deliberate: it privacy-gates any new
    # future metric table before the report links it.
    for aggregate_path in all_metric_tables:
        _read_csv(aggregate_path, (), f"aggregate table {aggregate_path.name}")
    table_paths = [
        features / "radiomics_feature_inventory.csv",
        root / "information_timing_contract.csv",
        root / "matched_population_manifest.csv",
        *all_metric_tables,
    ]
    return {
        "config": config,
        "config_path": config_path,
        "inventory": inventory,
        "inventory_md": root / "reports" / "radiomics_feature_inventory.md",
        "missingness": missingness,
        "matched": matched,
        "pcr": pcr,
        "primary_pcr": primary_pcr,
        "effects": incremental,
        "profile": primary_profile,
        "redundancy": redundancy,
        "residual": primary_residual,
        "family": family,
        "lr_svm": lr_svm,
        "correlation": correlation,
        "timing_path": root / "information_timing_contract.csv",
        "figures": figure_paths,
        "mri_frames": mri_frames,
        "table_paths": table_paths,
    }


def _classification(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    effects: pd.DataFrame = inputs["effects"]
    pcr: pd.DataFrame = inputs["primary_pcr"]
    profile: pd.DataFrame = inputs["profile"]
    increment = _effect_rows(effects, "C+FULL_vs_C+F")
    residual = _effect_rows(effects, "C+F+N_res_vs_C+F")
    positive_increment_timings = set(
        increment.loc[increment["_auroc_low"] > 0, "timing"].astype(str)
    )
    any_increment_signal = bool(positive_increment_timings)
    supported_increment = len(positive_increment_timings) >= 2 and bool(
        positive_increment_timings.intersection(("T0", "T1", "T2"))
    )
    positive_residual_timings = set(
        residual.loc[residual["_auroc_low"] > 0, "timing"].astype(str)
    )
    residual_signal = bool(positive_residual_timings)
    standalone_rows = pcr.loc[pcr["model"].astype(str).eq("N")]
    if standalone_rows.empty:
        raise ReportDataError("primary pCR metrics lack standalone N")
    standalone_signal = bool((pd.to_numeric(standalone_rows["auroc"]) >= 0.60).any())
    profile_rows = profile.loc[
        profile["feature_set"].astype(str).eq("N")
        & profile["target"].astype(str).isin(("HR", "HER2", "subtype", "subtype_4class"))
    ]
    profile_signal = bool((pd.to_numeric(profile_rows["auroc"]) >= 0.60).any())
    if supported_increment and residual_signal:
        code = "A"
        title = "CLASSICAL PHENOTYPE COMPLEMENTARITY SUPPORTED"
        meaning = "传统 DCE phenotype 在 FTV 之外提供稳定且 residualization 后仍存在的 pCR 信息。"
    elif standalone_signal and not any_increment_signal and not residual_signal:
        code = "B"
        title = "NONFTV SIGNAL EXISTS BUT IS FTV-REDUNDANT"
        meaning = "NONFTV 有 standalone signal，但没有任何 CI>0 的 increment，也没有 residualized signal。"
    elif profile_signal and not standalone_signal and not any_increment_signal and not residual_signal:
        code = "C"
        title = "PROFILE-RELATED BUT NOT pCR-COMPLEMENTARY"
        meaning = "传统 phenotype 可解码分子 profile correlate，但未满足 pCR complementarity。"
    elif not standalone_signal and not profile_signal and not any_increment_signal and not residual_signal:
        code = "D"
        title = "CLASSICAL DCE PHENOTYPE WEAK"
        meaning = "NONFTV 对 pCR 与 HR/HER2 的预注册 signal 标准均未达到。"
    else:
        code = "MIXED"
        title = "MIXED/INCONCLUSIVE — NO EXACT A–D CATEGORY"
        if residual_signal and not supported_increment:
            meaning = (
                "稳定 C+FULL 增量不足，故 A 不成立；但至少一个 residualized "
                "effect 的 CI>0，直接反驳 B 的“residual signal 消失”前提。"
                + (
                    "另有孤立 incremental CI>0；它不满足 A 的稳定性，但也不能当作零增量。"
                    if any_increment_signal
                    else ""
                )
            )
        elif supported_increment and not residual_signal:
            meaning = (
                "C+FULL 有稳定增量，但 residualized criterion 未满足；A 不成立，"
                "其余 B/C/D 也不能诚实概括该证据。"
            )
        else:
            meaning = "观测同时跨越多个预注册判据，无法无矛盾地映射到 A–D，故不强制分类。"
    return {
        "code": code,
        "title": title,
        "meaning": meaning,
        "supported_increment": supported_increment,
        "any_increment_signal": any_increment_signal,
        "increment_timings": sorted(positive_increment_timings, key=TIMINGS.index),
        "residual_signal": residual_signal,
        "residual_timings": sorted(positive_residual_timings, key=TIMINGS.index),
        "standalone_signal": standalone_signal,
        "profile_signal": profile_signal,
    }


def build_report(root: Path, figures_dir: Path, output_path: Path) -> str:
    inputs = load_report_inputs(root, figures_dir)
    classification = _classification(inputs)
    config: Mapping[str, Any] = inputs["config"]
    inventory: pd.DataFrame = inputs["inventory"]
    missingness: pd.DataFrame = inputs["missingness"]
    matched: pd.DataFrame = inputs["matched"]
    pcr: pd.DataFrame = inputs["primary_pcr"]
    effects: pd.DataFrame = inputs["effects"]
    profile: pd.DataFrame = inputs["profile"]
    redundancy: pd.DataFrame = inputs["redundancy"]
    residual: pd.DataFrame = inputs["residual"]
    family: pd.DataFrame = inputs["family"]
    lr_svm: pd.DataFrame = inputs["lr_svm"]

    actual_features, feature_grouping, family_inventory_table = _inventory_summary(inventory)
    n_rows = pcr.loc[pcr["model"].eq("C+F"), "n"]
    primary_n = int(pd.to_numeric(n_rows).mode().iloc[0])
    n_positive_rows = pcr.loc[pcr["model"].eq("C+F"), "n_positive"]
    primary_positive = int(pd.to_numeric(n_positive_rows).mode().iloc[0])
    n_best = _best(pcr.loc[pcr["model"].eq("N")])
    cn_effect = _effect_rows(effects, "C+N_vs_C")
    cn_ci_supported = bool((cn_effect["_auroc_low"] > 0).any())
    full_effect = _effect_rows(effects, "C+FULL_vs_C+F")
    residual_effect = _effect_rows(effects, "C+F+N_res_vs_C+F")
    n_res_best = _best(residual.loc[residual["model"].eq("N_res")])
    profile_best = _best(
        profile.loc[
            profile["feature_set"].isin(("N", "FULL"))
            & profile["target"].isin(("HR", "HER2", "subtype", "subtype_4class"))
        ]
    )
    profile_n_best = _best(
        profile.loc[
            profile["feature_set"].eq("N")
            & profile["target"].isin(("HR", "HER2", "subtype", "subtype_4class"))
        ]
    )
    family_text, family_table = _family_summary(family)
    lr_svm_text, lr_svm_table = _lr_svm_summary(inputs["pcr"], lr_svm)
    mri_status, mri_detail, mri_answer, mri_gap_summary = _assess_mri(
        inputs["mri_frames"], inputs["pcr"]
    )
    delivery = _delivery(root / "reports" / "delivery_provenance.json")

    source_missing = missingness.loc[
        missingness["population"].astype(str).eq("source_workbook")
        & missingness["family"].astype(str).isin(("FTV", "LD", "SPH", "BPE"))
        & missingness.get("role", pd.Series("", index=missingness.index)).astype(str).eq("absolute_measurement")
    ]
    if source_missing.empty:
        source_missing = missingness.loc[
            missingness["population"].astype(str).eq("source_workbook")
            & missingness["family"].astype(str).isin(("FTV", "LD", "SPH", "BPE"))
        ]
    max_missing = int(pd.to_numeric(source_missing["n_missing"]).max()) if not source_missing.empty else -1
    redundancy_best = redundancy.loc[pd.to_numeric(redundancy["r2"]).idxmax()]

    if classification["code"] == "A":
        q12 = "加入 phenotype-aware state/auxiliary targets（显式 FTV、LD、shape、BPE），并用 FTV-residual objective 检验 latent 是否学到非体积信息。"
    elif classification["code"] == "B":
        q12 = "优先显式建模 tumor burden 与其动态，并用 residual/orthogonality 约束避免把 LD、shape 当作新的独立 biology。"
    elif classification["code"] == "C":
        q12 = "尝试 HR/HER2-aware multi-task phenotype supervision，但不要预设它必然提高 pCR；先做外部验证与 calibration。"
    elif classification["code"] == "D":
        q12 = "不要围绕这组 handcrafted measurement 扩大模型；优先检验 foundation/更广视野 signal、标签噪声与 imaging-complementarity 上限。"
    else:
        q12 = (
            "先把孤立 residual effect 当作需复现的 hypothesis：优先验证 T0–T2、"
            "外部 cohort 与不同 split 的稳定性，并检查其是否仅由 late/pre-surgery T3 驱动；"
            "在复现前既不宣称 complementarity，也不宣称 FTV redundancy。"
        )

    if classification["code"] == "A" and mri_status == "below":
        q11 = "是：传统 phenotype 的 complementarity 已满足 A，且 matched current MRI reference 低于 traditional comparison，支持 representation-learning gap。"
    elif classification["code"] == "A" and mri_status == "at_least":
        q11 = "存在 FTV 外 phenotype，但 current MRI 已至少达到 traditional matched level；不能据此说 World Model 没学到它。"
    elif classification["code"] == "A" and mri_status == "mixed":
        q11 = (
            "A 类结果支持在本 cohort 中传统 DCE 存在 FTV 外、对 pCR 有增量的 phenotype；"
            "同一 375 人的 N-vs-M 对照也确实存在，但方向随 target/timing 改变。"
            f"{mri_gap_summary}。这些 traditional>N cell 支持 target/timing-specific representation gap "
            "作为候选解释，却不足以宣称 World Model 全局遗漏，更不是因果证据。"
        )
    elif classification["code"] == "A":
        q11 = "传统 DCE 中存在 FTV 外 phenotype；但缺少可配对的 matched MRI-vs-traditional aggregate，尚不能把它归因于 World Model 遗漏。"
    elif classification["code"] == "C":
        q11 = "存在 profile-related correlate，但没有 pCR complementarity 证据；是否被 World Model 遗漏仍需 matched latent comparison。"
    elif classification["code"] == "MIXED":
        q11 = (
            "存在值得追踪的 classical phenotype 线索，但没有稳定 primary complementarity。"
            + (
                "Matched current MRI 低于 traditional comparison，提示可能的 representation gap；"
                "由于证据仅落在 mixed/isolated pattern，这仍是待复现假设。"
                if mri_status == "below"
                else "现有证据不足以断言 World Model 已明确遗漏该 phenotype。"
            )
        )
    else:
        q11 = "本实验没有提供 World Model 遗漏强 classical phenotype 的充分证据。"

    table_descriptions = {
        "radiomics_feature_inventory.csv": "Table 1：真实 feature inventory",
        "information_timing_contract.csv": "方法表：information timing contract",
        "missingness.csv": "Table 2：missingness 与 coverage",
        "matched_population_manifest.csv": "Table 3：paired matched populations",
        "static_radiomics.csv": "Table 4：static radiomics pCR metrics",
        "longitudinal_radiomics.csv": "Table 5：longitudinal radiomics pCR metrics",
        "pcr_oof_metrics.csv": "Table 6：C/F/N/FULL comparison（全部 OOF aggregate）",
        "incremental_effects.csv": "Table 7：paired incremental effects 与 95% CI",
        "profile_oof_metrics.csv": "Table 8：HR/HER2/subtype probes",
        "redundancy_metrics.csv": "Table 9a：NONFTV→FTV redundancy",
        "residualization_metrics.csv": "Table 9b：FTV-residualized pCR metrics",
        "family_ablation_metrics.csv": "Table 10：D/S/B family ablation",
        "lr_vs_svm.csv": "Table 11：LR vs RBF-SVM",
        "feature_correlation_matrix.csv": "补充表：feature correlation matrix",
        "mri_reference_metrics.csv": "补充表：matched current MRI pCR reference",
        "mri_reference_profile_metrics.csv": "补充表：current MRI profile reference",
        "mri_reference_traditional_pcr_comparison.csv": "补充表：n=375 matched MRI-vs-traditional pCR comparison",
        "mri_reference_traditional_profile_comparison.csv": "补充表：n=375 matched MRI-vs-traditional profile comparison",
        "split_summary.csv": "方法表：outer train/validation/test split summary",
        "hyperparameter_selections.csv": "方法表：outer-fold hyperparameter selections",
        "preprocessing_audit.csv": "方法表：outer-train preprocessing audit",
        "pcr_fold_metrics.csv": "补充表：pCR fold-level aggregate metrics",
        "redundancy_fold_metrics.csv": "补充表：redundancy fold-level aggregate metrics",
        "residualization_fold_metrics.csv": "补充表：residualization fold-level aggregate metrics",
    }
    table_links = []
    for path in inputs["table_paths"]:
        description = table_descriptions.get(path.name, f"补充表：{path.name}")
        if path == root / "matched_population_manifest.csv":
            description += "（root mirror）"
        table_links.append(f"- {_link(description, path, output_path)}")
    figure_links = [
        f"- {_link(f'Figure {index}：{description}', path, output_path)}"
        for index, ((_, description), path) in enumerate(zip(FIGURES, inputs["figures"]), start=1)
    ]

    classification_table = _markdown_table(
        ("预注册判据", "观测", "是否满足"),
        (
            (
                "C+FULL−C+F 的 ΔAUROC CI>0：≥2 timings，且至少一个 T0–T2",
                "、".join(classification["increment_timings"]) or "无",
                "是" if classification["supported_increment"] else "否",
            ),
            (
                "是否存在任一孤立 C+FULL−C+F ΔAUROC CI>0（B/C/D 均要求无）",
                "、".join(classification["increment_timings"]) or "无",
                "是" if classification["any_increment_signal"] else "否",
            ),
            (
                "C+F+N_res−C+F 的 ΔAUROC CI>0：≥1 timing",
                "、".join(classification["residual_timings"]) or "无",
                "是" if classification["residual_signal"] else "否",
            ),
            (
                "N standalone AUROC≥0.60：≥1 timing",
                f"最佳描述值 {_fmt(n_best['auroc'])} @ {_cell(n_best)}",
                "是" if classification["standalone_signal"] else "否",
            ),
            (
                "N→HR/HER2/subtype AUROC≥0.60：≥1 timing",
                f"N 最佳描述值 {_fmt(profile_n_best['auroc'])} (N→{profile_n_best['target']}, {_cell(profile_n_best)})",
                "是" if classification["profile_signal"] else "否",
            ),
        ),
    )

    matched_primary = matched.loc[
        matched["protocol"].astype(str).eq(PRIMARY_FILTERS["protocol"])
        & matched["population"].astype(str).eq(PRIMARY_FILTERS["population"])
        & matched["scenario"].astype(str).eq(PRIMARY_FILTERS["scenario"])
    ].copy()
    if matched_primary.empty:
        raise ReportDataError("matched manifest has no primary 384 rows for report preview")
    matched_preview_columns = [
        column
        for column in (
            "protocol",
            "population",
            "scenario",
            "comparison",
            "view",
            "timing",
            "n",
            "n_positive",
            "pCR_positive",
            "missingness_exclusions",
        )
        if column in matched_primary.columns
    ]
    matched_preview = _markdown_table(
        tuple(matched_preview_columns),
        matched_primary.loc[:, matched_preview_columns]
        .head(12)
        .itertuples(index=False, name=None),
    )

    report = f"""# Classical DCE Phenotype Complementarity Baseline：最终报告

## 结论先行

主分析是 `{PRIMARY_FILTERS['protocol']}` / `{PRIMARY_FILTERS['population']}` / `complete_case` / logistic regression，共 **n={primary_n}（pCR+={primary_positive}）**。按预注册判据作严格、不强制的科学映射，结果为：

> **{classification['code']}. {classification['title']}**
>
> {classification['meaning']}

该映射只由 primary radiomics 的预注册规则产生；current MRI reference、SVM 和 family ablation 均不参与类别选择。只有完整满足定义才赋 A/B/C/D；若残差信号等证据与某类前提冲突，则报告 `MIXED/INCONCLUSIVE`，而不强塞进“否则 D”或错误声称 FTV redundancy。T3 在全文与图中均标为 **late/pre-surgery**，不能与较早、可行动的 timing 等量解释。

{classification_table}

## 12 个必答问题

### 1. 实际有哪些 radiomics features？

{actual_features}

{family_inventory_table}

这里的 “radiomics” 是低维、纵向 DCE measurement，不是高维 texture feature。完整字段、单位、coverage 与 leakage concern 见 {_link('feature inventory CSV', inputs['table_paths'][0], output_path)} 和 {_link('可读 inventory', inputs['inventory_md'], output_path)}。

### 2. 哪些是 FTV，哪些属于 non-FTV phenotype？

{feature_grouping} 原始工作簿另有 percent-change 派生列，但 pipeline 从 timing-safe absolute observations 按预注册公式重建 change，避免重复计数。

### 3. non-FTV 是否单独预测 pCR？

N standalone 的最高**描述性** OOF AUROC 为 **{_fmt(n_best['auroc'])}**（{_cell(n_best)}，AUPRC={_fmt(n_best['auprc'])}）。按固定 AUROC≥0.60 综合阈值，答案为 **{'是' if classification['standalone_signal'] else '否'}**。这个“最高值”只用于描述，未用于选择 feature、timing 或 primary model。

### 4. 是否增加 Clinical-only performance？

**{'至少一个预注册 cell 有明确正增量' if cn_ci_supported else '没有预注册 cell 的 95% CI 全高于 0'}**。{_effect_summary(cn_effect)} 因而应结合所有 timing 的 effect size/CI 阅读，不能只挑最大的 test cell；完整 ΔAUROC、ΔAUPRC 与 Brier improvement 见 incremental-effects 表。

### 5. 是否增加 Clinical+FTV performance？

{_effect_summary(full_effect)} 预注册的“稳定增量”判据 **{'满足' if classification['supported_increment'] else '不满足'}**；判据要求至少两个 distinct timings 的 paired ΔAUROC 95% CI 全高于 0，且至少一个来自 T0–T2。

### 6. residualized non-FTV 是否仍有 signal？

N_res 最高描述性 AUROC 为 **{_fmt(n_res_best['auroc'])}**（{_cell(n_res_best)}）。{_effect_summary(residual_effect)} 因而 residual-signal 判据 **{'满足' if classification['residual_signal'] else '不满足'}**。NONFTV→FTV redundancy 的最高 R² 为 {_fmt(redundancy_best['r2'])}（{_cell(redundancy_best)}，Spearman={_fmt(redundancy_best['spearman'])}）。Residualizer 与 redundancy regression 均只在 outer train 拟合。

### 7. 哪类 feature 最有价值？

{family_text}

{family_table}

### 8. HR/HER2 是否可从传统 DCE phenotype 预测？

全部传统 feature sets 中的最佳描述性 probe 是 **{profile_best['feature_set']}→{profile_best['target']}**，AUROC={_fmt(profile_best['auroc'])}、AUPRC={_fmt(profile_best['auprc'])}、balanced accuracy={_fmt(profile_best['balanced_accuracy'])}（{_cell(profile_best)}）。用于 C 类判定的 NONFTV-only 最佳值为 **N→{profile_n_best['target']} AUROC={_fmt(profile_n_best['auroc'])}**（{_cell(profile_n_best)}）；按固定 N-only AUROC≥0.60 规则，profile signal **{'存在' if classification['profile_signal'] else '未达到'}**。FULL 仍单独作描述，但 FULL-only crossing 不归因于 non-FTV，因为它可能由 FTV 驱动。Probe 是 cohort-level correlate，不是因果机制或可直接临床部署的 biomarker。

### 9. LR 和 SVM 结论是否一致？

{lr_svm_text}

{lr_svm_table}

### 10. 当前 MRI latent 是否至少达到传统 radiomics 水平？

{mri_answer}

MRI reference 细节：{mri_detail} 它是 supplementary sensitivity，绝不改变 384 人 primary radiomics classification。

这里的等级判断只使用同一 375 人上的 **N vs M 描述性对照**。Frozen MRI audit 的 C/F preprocessing 与 prediction head 不同于本实验：旧 F 主要是 log1p absolute prefix，新 F 包含 outer-train winsorized absolute/delta/relative features，clinical encoding 也不同。因此 `C+N vs C+M` 与 `C+FULL vs C+F+M` 有 baseline confounding，只展示在 direct comparison 表中，不能解释为 causal 或 paired incremental effect。

### 11. MRI 里是否存在 World Model 还没学到的 phenotype？

{q11}

### 12. 对下一版 World Model 最直接的建议是什么？

{q12}

## 方法、matching 与 leakage control

- Primary estimand 是 strict matched complete-case；每个 paired comparison 在相同 view/timing/fold 使用完全相同患者。Manifest 的 primary-384 子集前 12 行如下，完整 cross-protocol 表见链接。

{matched_preview}

- Static `Tk` 只读当前 `Tk`；longitudinal `Tk` 只读 T0…Tk，并仅从已观察 prefix 构建 absolute/relative change。{_link('机器可读 timing contract', inputs['timing_path'], output_path)}明确 T3 为 late/pre-surgery。
- Log transform、1%/99% winsorization、median/missingness indicator、categorical vocabulary、scaling、FTV residualization 与 redundancy regression 均只在 outer train 拟合；validation 只选超参数/threshold；outer test 不参与 selection。
- Primary model 是 L2 logistic regression；RBF SVM 只作 sensitivity。Feature families、change formula 和模型空间均在 outcome modeling 前固定，禁止按 test pCR 表现筛 family。
- 关键 comparisons 使用 {int(config['bootstrap_draws']):,} 次 paired patient-level bootstrap；AUROC/AUPRC 为 augmented−baseline，Brier improvement 为 baseline−augmented。报告与绘图阶段只读取 aggregate metrics，不读取 patient IDs、OOF probabilities 或原始 workbook。
- Source-workbook 的独立 FTV/LD/SPH/BPE measurement 最大 cell missing count 为 {max_missing}；secondary train-median+indicator scenario 是稳健性分析，不替代 primary complete-case estimand。

## 选择与解释限制

1. 这是单 cohort 的 OOF internal validation，不是 external validation；bootstrap CI 不消除 dataset shift、标签误差或治疗方案差异。
2. 多个 timing、view、target 与 metric 会产生 multiplicity。A–D 规则是预注册的保守综合，不等同于每个 cell 的正式多重检验校正，也不应把 AUROC=0.60 当临床阈值。
   此外没有预注册 equivalence margin；“未满足稳定增量”不等于效果为零或两模型约等，任何孤立 CI>0 都会阻止 B/C/D 并触发 mixed mapping。
3. Family ablation 只做 D/S/B 三个预注册单-family add-on，不穷举组合；报告的 family 排名是定位性描述，不能作为 post-test feature selection。
4. SPH 来自 FTV mask geometry、LD 是 burden measure，二者可能与 FTV 强相关；BPE 则可能来自 lesion-centered crop 之外。Residualization 缓解线性 FTV redundancy，但不能证明生物学独立或因果性。
5. T3 接近术前，预测值即便较高也可能缺乏早期决策价值。HR/HER2 probe 只说明可解码关联；profile class imbalance、calibration 与外部泛化仍需单独验证。
6. Current MRI reference 只有在同一 matched population、相同 target/timing 且 aggregate 定义可配对时才排名；不跨 n=384 primary 与 n=375 reference 作伪配对。Frozen audit 与新 classical pipeline 的 F/clinical preprocessing 和 head 不同，所以仅 N-vs-M 是同人群描述性 benchmark；clinical/FTV-augmented rows baseline-confounded，不能当作 paired incremental effect，且 primary radiomics 结论不依赖 MRI reference。

## 与 Goal 3 / Goal 5 的条件解释框架

本实验不等待其他 Goal 完成；下表只定义将来的条件解释，不把未知结果填成事实。

| 条件 | 解释 / 下一步 |
| --- | --- |
| Goal 6 classical phenotype 强，而 current MRI 弱 | current representation learning failure；优先 phenotype-aware representation learning |
| Goal 6 强，且 Goal 5 heterogeneity 强 | 优先设计显式 phenotype state，并分层检查稳定性 |
| Goal 6 弱，但 Foundation strong | foundation 捕获了 handcrafted radiomics 之外的信息 |
| Goal 6 弱，且 Foundation 也弱 | dataset 的 imaging-complementarity 可能本身有限 |

## 表格索引

{chr(10).join(table_links)}

`static_radiomics.csv` 与 `longitudinal_radiomics.csv` 是 `pcr_oof_metrics.csv` 的预定义 view 投影；Table 6 仍保留完整 `model` comparison，没有按结果挑选行。配置与规则见 {_link('experiment.json', inputs['config_path'], output_path)}。

## 图索引

{chr(10).join(figure_links)}

## Git / GitHub delivery provenance

| 字段 | 值 |
| --- | --- |
| branch | `{_escape(delivery['branch'])}` |
| commit SHA | `{_escape(delivery['commit_sha'])}` |
| push status | `{_escape(delivery['push_status'])}` |
| push error | `{_escape(delivery['push_error'])}` |

若 `reports/delivery_provenance.json` 尚不存在或字段缺失，上述值按要求显示 `PENDING`；报告生成器不会猜测 branch、SHA 或 push 状态。
"""
    return report


def write_report(root: Path, figures_dir: Path, output_path: Path) -> Path:
    report = build_report(root, figures_dir, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary.write_text(report, encoding="utf-8")
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Classical DCE experiment directory.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Figure directory (default: <experiment-root>/figures).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: <experiment-root>/reports/final_report.md).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.experiment_root.resolve()
    figures_dir = (args.figures_dir or root / "figures").resolve()
    output = (args.output or root / "reports" / "final_report.md").resolve()
    print(write_report(root, figures_dir, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
