"""Complete-matrix aggregation, paired effects, DiD, and Stage B figures."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .contracts import ARMS, FOLDS, SEED_BASES, file_sha256
from .gate import StageAAuthorization


METRICS = (
    "spearman",
    "r2",
    "rmse",
    "mae",
    "b0_rmse",
    "rmse_gain_over_b0",
    "prediction_target_variance_ratio",
)

FIGURE_NAMES = (
    "07_static_ftv_r2_comparison.png",
    "08_static_ftv_spearman_comparison.png",
    "09_literal_delta_ftv_r2_comparison.png",
    "10_literal_delta_ftv_spearman_comparison.png",
    "11_base_degradation_heatmap.png",
    "12_grounding_difference_in_differences.png",
    "13_representative_training_curves.png",
    "14_natural_ftv_predicted_vs_true.png",
)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _cell_path(root: Path, seed: int, arm: str, fold: int) -> Path:
    return root / f"seed_{seed}" / arm / f"fold_{fold}"


def collect_complete_matrix(
    checkpoint_root: str | Path,
    probe_root: str | Path,
    authorization: StageAAuthorization,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_root = Path(checkpoint_root).resolve()
    probe_root = Path(probe_root).resolve()
    selections: list[dict[str, Any]] = []
    metrics: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    missing: list[str] = []
    for seed in SEED_BASES:
        for arm in ARMS:
            for fold in FOLDS:
                run = _cell_path(checkpoint_root, seed, arm, fold)
                probe = _cell_path(probe_root, seed, arm, fold)
                required = {
                    "selection": run / "selection.json",
                    "history": run / "history.csv",
                    "metrics": probe / "probe_metrics.csv",
                    "predictions": probe / "ridge_predictions.private.csv",
                }
                absent = [str(path) for path in required.values() if not path.is_file()]
                if absent:
                    missing.extend(absent)
                    continue
                selection = json.loads(required["selection"].read_text(encoding="utf-8"))
                expected = {"arm": arm, "seed_base": seed, "fold": fold}
                if any(selection.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"selection identity mismatch: {required['selection']}")
                if selection.get("test_data_used") is not False:
                    raise ValueError("checkpoint selection used test data")
                if selection.get("stage_a_sentinel_sha256") != authorization.sha256:
                    raise ValueError("Stage A authorization differs across the result matrix")
                selections.append(selection)
                for name, container in (
                    ("metrics", metrics), ("history", histories), ("predictions", predictions)
                ):
                    frame = pd.read_csv(required[name])
                    for identity, expected_value in expected.items():
                        if identity not in frame or not frame[identity].eq(expected_value).all():
                            raise ValueError(
                                f"{name} identity mismatch for {required[name]}: {identity}"
                            )
                    frame["arm"] = arm
                    frame["seed_base"] = seed
                    frame["fold"] = fold
                    container.append(frame)
    if missing:
        preview = "\n".join(missing[:12])
        raise RuntimeError(
            f"Stage B matrix is incomplete ({len(missing)} missing artifacts); no aggregation or conclusion is allowed:\n{preview}"
        )
    if len(selections) != len(SEED_BASES) * len(ARMS) * len(FOLDS):
        raise AssertionError("complete matrix cardinality mismatch")
    selection_frame = pd.DataFrame(selections)
    metric_frame = pd.concat(metrics, ignore_index=True)
    history_frame = pd.concat(histories, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    _audit_four_arm_matrix_contract(selection_frame, metric_frame, history_frame)
    _audit_oof_prediction_contract(metric_frame, prediction_frame)
    return selection_frame, metric_frame, history_frame, prediction_frame


def _audit_oof_prediction_contract(
    metrics: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Require exact five-fold OOF uniqueness and paired four-arm targets."""

    required = {
        "patient_id", "arm", "seed_base", "fold", "task", "endpoint",
        "analysis_scope", "target_semantics", "split", "y_true", "y_pred",
        "b0_prediction", "test_predict_call_count",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"Stage B predictions miss required columns: {missing}")
    if not predictions["split"].eq("test").all():
        raise ValueError("pooled OOF assets may contain only outer-test predictions")
    if not predictions["test_predict_call_count"].eq(1).all():
        raise ValueError("one or more probe cells evaluated outer test more than once")
    identity = [
        "seed_base", "arm", "task", "endpoint", "analysis_scope", "patient_id"
    ]
    if predictions.duplicated(identity).any():
        raise ValueError("a patient appears in more than one outer-test fold for one OOF endpoint")

    natural = metrics.loc[
        metrics["analysis_scope"].isin(predictions["analysis_scope"].unique())
        & metrics["scale"].eq("natural")
        & metrics["endpoint"].ne("macro")
    ]
    count_keys = ["seed_base", "arm", "fold", "task", "endpoint", "analysis_scope"]
    expected_counts = natural.set_index(count_keys)["n_test"]
    if expected_counts.index.duplicated().any():
        raise ValueError("natural probe metrics contain duplicate fold endpoint rows")
    observed_counts = predictions.groupby(count_keys, sort=False).size()
    expected_counts = expected_counts.astype(int).sort_index()
    observed_counts = observed_counts.astype(int).sort_index()
    if not expected_counts.index.equals(observed_counts.index) or not expected_counts.equals(
        observed_counts
    ):
        raise ValueError("OOF prediction counts disagree with the fold probe metrics")

    paired_keys = [
        "seed_base", "task", "endpoint", "analysis_scope", "target_semantics", "patient_id"
    ]
    for keys, rows in predictions.groupby(paired_keys, sort=False):
        if set(rows["arm"]) != set(ARMS) or len(rows) != len(ARMS):
            raise ValueError(f"OOF patient coverage differs across arms for {keys}")
        for column in ("fold", "y_true", "b0_prediction"):
            values = rows[column].to_numpy()
            if column == "fold":
                aligned = np.all(values == values[0])
            else:
                aligned = np.allclose(values.astype(float), float(values[0]), rtol=0.0, atol=1e-12)
            if not aligned:
                raise ValueError(f"paired OOF {column} differs across arms for {keys}")


def _pooled_natural_metrics(rows: pd.DataFrame) -> dict[str, float]:
    truth = rows["y_true"].to_numpy(dtype=float)
    prediction = rows["y_pred"].to_numpy(dtype=float)
    baseline = rows["b0_prediction"].to_numpy(dtype=float)
    if not len(truth) or not all(np.isfinite(value).all() for value in (truth, prediction, baseline)):
        raise FloatingPointError("pooled OOF prediction rows are empty or non-finite")
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    b0_rmse = float(math.sqrt(mean_squared_error(truth, baseline)))
    target_variance = float(np.var(truth, ddof=0))
    correlation = float(spearmanr(truth, prediction).statistic)
    return {
        "spearman": correlation if math.isfinite(correlation) else math.nan,
        "r2": float(r2_score(truth, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, prediction)),
        "b0_rmse": b0_rmse,
        "rmse_gain_over_b0": (b0_rmse - rmse) / b0_rmse if b0_rmse > 0 else math.nan,
        "prediction_target_variance_ratio": (
            float(np.var(prediction, ddof=0)) / target_variance
            if target_variance > 0 else math.nan
        ),
    }


def pooled_oof_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recompute nonlinear natural-scale endpoints after pooling five test folds."""

    group_keys = [
        "seed_base", "arm", "task", "endpoint", "analysis_scope", "target_semantics"
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_keys, sort=False):
        if group["patient_id"].astype(str).duplicated().any():
            raise ValueError(f"pooled OOF group contains a repeated patient: {keys}")
        rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "scale": "natural",
                "aggregation": "pooled_5fold_oof",
                "n_test": len(group),
                **_pooled_natural_metrics(group),
            }
        )
    endpoint_frame = pd.DataFrame(rows)
    expected = {
        "static": set(("T0", "T1", "T2", "T3")),
        "delta": set(("T0→T1", "T1→T2", "T2→T3")),
    }
    macros: list[dict[str, Any]] = []
    macro_keys = ["seed_base", "arm", "task", "analysis_scope", "target_semantics"]
    for keys, group in endpoint_frame.groupby(macro_keys, sort=False):
        task = str(keys[2])
        if set(group["endpoint"]) != expected[task]:
            raise ValueError(f"pooled OOF endpoint coverage drifted for {keys}")
        macros.append(
            {
                **dict(zip(macro_keys, keys, strict=True)),
                "endpoint": "macro",
                "scale": "natural",
                "aggregation": "mean_of_pooled_endpoint_metrics",
                "n_test": int(group["n_test"].sum()),
                **{metric: float(group[metric].mean()) for metric in METRICS},
            }
        )
    return pd.concat([endpoint_frame, pd.DataFrame(macros)], ignore_index=True)


def _audit_four_arm_matrix_contract(
    selections: pd.DataFrame, metrics: pd.DataFrame, histories: pd.DataFrame
) -> None:
    """Refuse aggregation if any paired run contract or epoch order drifted."""

    for seed in SEED_BASES:
        for fold in FOLDS:
            current = selections.loc[
                selections["seed_base"].eq(seed) & selections["fold"].eq(fold)
            ]
            if set(current["arm"]) != set(ARMS) or len(current) != len(ARMS):
                raise ValueError(f"seed {seed}/fold {fold} is not an exact four-arm cell")
            for column in (
                "paired_initialization_sha256",
                "train_patient_sha256",
                "val_patient_sha256",
                "data_provenance_sha256",
            ):
                if current[column].nunique(dropna=False) != 1:
                    raise ValueError(f"seed {seed}/fold {fold} paired {column} drifted")
            hyperparameters = current["hyperparameters"].map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
            )
            if hyperparameters.nunique(dropna=False) != 1:
                raise ValueError(f"seed {seed}/fold {fold} hyperparameters differ across arms")
            resolved = current.iloc[0]["hyperparameters"]
            pair = (int(resolved["physical_batch_size"]), int(resolved["accumulation_steps"]))
            if pair not in {(4, 8), (2, 16)}:
                raise ValueError("matrix contains an unregistered physical/accumulation contract")
            fallback_flags = set(bool(value) for value in current["global_fallback_restart"])
            if fallback_flags != {pair == (2, 16)}:
                raise ValueError("global 2/16 restart provenance is missing or arm-specific")
            history = histories.loc[
                histories["seed_base"].eq(seed) & histories["fold"].eq(fold)
            ]
            epoch_one = history.loc[history["epoch"].eq(1)]
            if set(epoch_one["arm"]) != set(ARMS):
                raise ValueError(f"seed {seed}/fold {fold} lacks epoch-one evidence for all arms")
            for epoch, epoch_rows in history.groupby("epoch", sort=False):
                for column in (
                    "patient_order_sha256",
                    "dropped_logical_tail_patients",
                    "train_optimizer_steps",
                ):
                    if epoch_rows[column].nunique(dropna=False) != 1:
                        raise ValueError(
                            f"seed {seed}/fold {fold}/epoch {epoch} {column} differs across arms"
                        )
    primary = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["scale"].eq("natural")
    ]
    expected = {
        "static": {"T0", "T1", "T2", "T3", "macro"},
        "delta": {"T0→T1", "T1→T2", "T2→T3", "macro"},
    }
    for (seed, fold, arm, task), rows in primary.groupby(
        ["seed_base", "fold", "arm", "task"], sort=False
    ):
        if set(rows["endpoint"]) != expected[str(task)]:
            raise ValueError(f"probe endpoint coverage drifted for {seed}/{fold}/{arm}/{task}")
        if task == "delta" and set(rows["target_semantics"]) != {
            "literal_ftv_end_minus_ftv_start"
        }:
            raise ValueError("delta probe is not literal natural FTV subtraction")


def paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["scale"].eq("natural")
    ].copy()
    keys = ["seed_base"]
    if "fold" in primary.columns:
        keys.append("fold")
    keys.extend(["task", "endpoint", "target_semantics"])
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        wide = primary.pivot(index=keys, columns="arm", values=metric).reset_index()
        if set(ARMS).difference(wide.columns):
            raise ValueError(f"probe metric {metric} lacks one or more Stage B arms")
        for row in wide.itertuples(index=False):
            values = row._asdict()
            l_effect = float(values["L3"] - values["L1"])
            n_effect = float(values["N3"] - values["N1"])
            rows.append(
                {
                    **{key: values[key] for key in keys},
                    "metric": metric,
                    "L3_minus_L1": l_effect,
                    "N3_minus_N1": n_effect,
                    "N1_minus_L1": float(values["N1"] - values["L1"]),
                    "difference_in_differences": n_effect - l_effect,
                }
            )
    return pd.DataFrame(rows)


def optimization_table(selections: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (str(row.arm), int(row.seed_base), int(row.fold)): row
        for row in selections.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for seed in SEED_BASES:
        for fold in FOLDS:
            for arm in ARMS:
                row = lookup[(arm, seed, fold)]
                if arm in {"L3", "N3"}:
                    baseline_arm = "L1" if arm == "L3" else "N1"
                    baseline = float(lookup[(baseline_arm, seed, fold)].selected_validation_state_loss)
                    degradation = float(row.selected_validation_state_loss) / baseline - 1.0
                    safety_pass = degradation <= 0.05 and bool(row.experiment_pass)
                else:
                    baseline_arm = arm
                    baseline = float(row.selected_validation_state_loss)
                    degradation = 0.0
                    safety_pass = bool(row.experiment_pass)
                rows.append(
                    {
                        "arm": arm,
                        "seed_base": seed,
                        "fold": fold,
                        "selected_epoch": int(row.selected_epoch),
                        "selected_validation_state_loss": float(row.selected_validation_state_loss),
                        "paired_baseline_arm": baseline_arm,
                        "paired_baseline_state_loss": baseline,
                        "base_degradation_fraction": degradation,
                        "base_degradation_gt_5pct": degradation > 0.05,
                        "selected_validation_ftv_loss": float(row.selected_validation_ftv_loss),
                        "selected_representation_std": float(row.selected_representation_std),
                        "selection_mode": str(row.selection_mode),
                        "finite": all(
                            math.isfinite(float(value))
                            for value in (
                                row.selected_validation_state_loss,
                                row.selected_representation_std,
                            )
                        ),
                        "optimization_safety_pass": safety_pass,
                    }
                )
    return pd.DataFrame(rows)


def _comparison_figure(metrics: pd.DataFrame, task: str, metric: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = metrics.loc[
        metrics["analysis_scope"].eq("primary_measurement_valid")
        & metrics["scale"].eq("natural")
        & metrics["task"].eq(task)
        & metrics["endpoint"].eq("macro")
    ]
    values = [subset.loc[subset["arm"].eq(arm), metric].to_numpy(float) for arm in ARMS]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(values, tick_labels=ARMS, showmeans=True)
    for index, array in enumerate(values, start=1):
        axis.scatter(np.full(len(array), index), array, alpha=0.65, s=20)
    axis.set_ylabel(metric)
    axis.set_title(f"Stage B {task} macro {metric}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_stage_b_figures(
    metrics: pd.DataFrame,
    effects: pd.DataFrame,
    optimization: pd.DataFrame,
    histories: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: str | Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output = Path(figure_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in FIGURE_NAMES]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite Stage B figures")
    _comparison_figure(metrics, "static", "r2", paths[0])
    _comparison_figure(metrics, "static", "spearman", paths[1])
    _comparison_figure(metrics, "delta", "r2", paths[2])
    _comparison_figure(metrics, "delta", "spearman", paths[3])

    grounded = optimization.loc[optimization["arm"].isin(["L3", "N3"])].copy()
    grounded["cell"] = grounded["seed_base"].astype(str) + "/f" + grounded["fold"].astype(str)
    heat = grounded.pivot(index="arm", columns="cell", values="base_degradation_fraction")
    figure, axis = plt.subplots(figsize=(10, 2.8))
    image = axis.imshow(heat.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-0.05, vmax=0.10)
    axis.set_yticks(range(len(heat.index)), labels=heat.index)
    axis.set_xticks(range(len(heat.columns)), labels=heat.columns, rotation=45, ha="right")
    figure.colorbar(image, ax=axis, label="validation state-loss degradation")
    figure.tight_layout()
    figure.savefig(paths[4], dpi=180)
    plt.close(figure)

    interaction = effects.loc[
        effects["metric"].eq("r2") & effects["endpoint"].eq("macro")
    ]
    figure, axis = plt.subplots(figsize=(5, 5))
    for task, marker in (("static", "o"), ("delta", "s")):
        part = interaction.loc[interaction["task"].eq(task)]
        axis.scatter(part["L3_minus_L1"], part["N3_minus_N1"], label=task, marker=marker)
    limits = axis.get_xlim()
    lower, upper = min(limits[0], axis.get_ylim()[0]), max(limits[1], axis.get_ylim()[1])
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("L3 - L1 macro natural R²")
    axis.set_ylabel("N3 - N1 macro natural R²")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[5], dpi=180)
    plt.close(figure)

    representative = histories.loc[
        histories["seed_base"].eq(SEED_BASES[0]) & histories["fold"].eq(0)
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for arm in ARMS:
        part = representative.loc[representative["arm"].eq(arm)].sort_values("epoch")
        axis.plot(part["epoch"], part["val_state_loss"], marker="o", label=arm)
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation state loss")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[6], dpi=180)
    plt.close(figure)

    static = predictions.loc[
        predictions["task"].eq("static")
        & predictions["analysis_scope"].eq("primary_measurement_valid")
        & predictions["arm"].eq("N3")
    ]
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(static["y_true"], static["y_pred"], alpha=0.35, s=12)
    low = float(min(static["y_true"].min(), static["y_pred"].min()))
    high = float(max(static["y_true"].max(), static["y_pred"].max()))
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("observed natural FTV")
    axis.set_ylabel("predicted natural FTV")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[7], dpi=180)
    plt.close(figure)
    return paths


def aggregate_stage_b(
    *,
    checkpoint_root: str | Path,
    probe_root: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    authorization: StageAAuthorization,
) -> dict[str, Any]:
    selections, metrics, histories, predictions = collect_complete_matrix(
        checkpoint_root, probe_root, authorization
    )
    pooled_metrics = pooled_oof_metrics(predictions)
    effects = paired_effects(pooled_metrics)
    fold_effects = paired_effects(metrics)
    optimization = optimization_table(selections)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "static": output / "table2_static_ftv.csv",
        "delta": output / "table3_literal_observed_delta_ftv.csv",
        "optimization": output / "table4_optimization_safety.csv",
        "effects": output / "table5_difference_in_differences.csv",
        "fold_effects": output / "table5_fold_level_sensitivity.csv",
        "summary": output / "stage_b_aggregation_summary.json",
    }
    figure_paths_expected = [Path(figure_dir).resolve() / name for name in FIGURE_NAMES]
    if any(path.exists() for path in (*outputs.values(), *figure_paths_expected)):
        raise FileExistsError("refusing to overwrite Stage B aggregate outputs")
    primary_natural = pooled_metrics.loc[
        pooled_metrics["analysis_scope"].eq("primary_measurement_valid")
        & pooled_metrics["scale"].eq("natural")
    ]
    _atomic_csv(outputs["static"], primary_natural.loc[primary_natural["task"].eq("static")])
    _atomic_csv(outputs["delta"], primary_natural.loc[primary_natural["task"].eq("delta")])
    _atomic_csv(outputs["optimization"], optimization)
    _atomic_csv(outputs["effects"], effects)
    _atomic_csv(outputs["fold_effects"], fold_effects)
    figure_paths = make_stage_b_figures(
        pooled_metrics, effects, optimization, histories, predictions, figure_dir
    )
    summary = {
        "schema_version": 1,
        "matrix_complete": True,
        "run_count": len(selections),
        "seeds": list(SEED_BASES),
        "folds": list(FOLDS),
        "arms": list(ARMS),
        "stage_a_sentinel_sha256": authorization.sha256,
        "comparisons": ["L3-L1", "N3-N1", "N1-L1", "(N3-N1)-(L3-L1)"],
        "primary_representation_aggregation": "five-fold pooled OOF recomputed per seed",
        "fold_level_effects_retained_as_sensitivity": True,
        "literal_delta_enforced": True,
        "test_used_for_checkpoint_selection": False,
        "figure_sha256": {path.name: file_sha256(path) for path in figure_paths},
        "interpretation": "not auto-classified; apply only the three preregistered scientific categories to these complete paired tables",
    }
    _atomic_json(outputs["summary"], summary)
    return summary


__all__ = [
    "aggregate_stage_b",
    "collect_complete_matrix",
    "make_stage_b_figures",
    "optimization_table",
    "paired_effects",
    "pooled_oof_metrics",
]
