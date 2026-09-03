"""Locked timing-safe conditional pCR evaluation and mechanism analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import (
    ARMS,
    EXPERIMENT_ROOT,
    FOLDS,
    PROJECTION_SEEDS,
    SEEDS,
    load_protocol,
)
from .data import FoldTargets, validate_state_archive
from .locking import verify_evaluation_lock


CLINICAL_FIELDS = (
    "label_hr", "label_her2", "label_mp", "age_at_screening", "race_simple",
    "menopausal_status_simple", "ethnicity", "arm",
)
NUMERIC_FIELDS = frozenset({"label_hr", "label_her2", "label_mp", "age_at_screening"})
MISSING = "__MISSING__"
TIMINGS = ("T0", "T0-T1", "T0-T2", "T0-T3")


class ClinicalEncoder:
    def __init__(self) -> None:
        self.medians: dict[str, float] = {}
        self.categories: dict[str, tuple[str, ...]] = {}
        self.fitted = False

    @staticmethod
    def _categorical(series: pd.Series) -> np.ndarray:
        return np.asarray(
            [MISSING if pd.isna(value) or not str(value).strip() else str(value).strip() for value in series],
            dtype=object,
        )

    def fit(self, frame: pd.DataFrame) -> "ClinicalEncoder":
        if self.fitted or frame.empty:
            raise ValueError("clinical encoder fit contract failed")
        for field in CLINICAL_FIELDS:
            if field in NUMERIC_FIELDS:
                values = pd.to_numeric(frame[field], errors="raise").to_numpy(float)
                finite = values[np.isfinite(values)]
                if not len(finite):
                    raise ValueError(f"clinical numeric field has no finite train value: {field}")
                self.medians[field] = float(np.median(finite))
            else:
                values = self._categorical(frame[field])
                self.categories[field] = tuple(sorted(set(values.tolist()) | {MISSING}))
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("clinical encoder is not fitted")
        blocks: list[np.ndarray] = []
        for field in CLINICAL_FIELDS:
            if field in NUMERIC_FIELDS:
                values = pd.to_numeric(frame[field], errors="raise").to_numpy(float)
                blocks.append(np.where(np.isfinite(values), values, self.medians[field])[:, None])
            else:
                values = self._categorical(frame[field])
                levels = self.categories[field]
                lookup = {value: index for index, value in enumerate(levels)}
                block = np.zeros((len(frame), len(levels)), dtype=np.float64)
                for row, value in enumerate(values):
                    if value in lookup:
                        block[row, lookup[value]] = 1.0
                blocks.append(block)
        result = np.concatenate(blocks, axis=1)
        if not np.isfinite(result).all():
            raise ValueError("clinical transform is non-finite")
        return result


def ftv_prefix(frame: pd.DataFrame, timing: str) -> np.ndarray:
    end = {"T0": 1, "T0-T1": 2, "T0-T2": 3, "T0-T3": 4}[timing]
    values = frame[[f"FTV_T{i}" for i in range(end)]].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("primary FTV prefix is invalid")
    return np.log1p(values)


def image_prefix(state: np.ndarray, timing: str) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("image state must be finite [N,4,D]")
    if timing == "T0":
        blocks = [values[:, 0]]
    elif timing == "T0-T1":
        blocks = [values[:, 0], values[:, 1], values[:, 1] - values[:, 0]]
    elif timing == "T0-T2":
        blocks = [
            values[:, 0], values[:, 1], values[:, 2],
            values[:, 1] - values[:, 0], values[:, 2] - values[:, 1], values[:, 2] - values[:, 0],
        ]
    elif timing == "T0-T3":
        blocks = [values[:, index] for index in range(4)]
        blocks.extend(values[:, right] - values[:, left] for left in range(4) for right in range(left + 1, 4))
    else:
        raise ValueError(f"unregistered timing: {timing}")
    return np.concatenate(blocks, axis=1)


def gaussian_projection(values: np.ndarray, seed: int, output_dim: int = 32) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    matrix = generator.normal(
        0.0, 1.0 / math.sqrt(values.shape[1]), size=(values.shape[1], int(output_dim))
    )
    return values @ matrix


def _logistic(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), penalty="l2", solver="liblinear", max_iter=10000),
    )


def _select_c(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    grid: Sequence[float],
) -> float:
    best: tuple[float, float] | None = None
    for c_value in grid:
        model = _logistic(c_value)
        model.fit(train_x, train_y)
        score = float(roc_auc_score(validation_y, model.predict_proba(validation_x)[:, 1]))
        candidate = (-score, float(c_value))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("no L2 logistic candidate selected")
    return best[1]


def _inner_oof_numeric(
    values: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    output = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    for train_index, holdout_index in splitter.split(values, labels):
        model = _logistic(c_value)
        model.fit(values[train_index], labels[train_index])
        output[holdout_index] = model.decision_function(values[holdout_index])
    if not np.isfinite(output).all():
        raise RuntimeError("inner OOF numeric logits are incomplete")
    return output


def _cf_matrix(train_fit: pd.DataFrame, frame: pd.DataFrame, timing: str, include_ftv: bool) -> np.ndarray:
    clinical = ClinicalEncoder().fit(train_fit)
    values = clinical.transform(frame)
    if include_ftv:
        values = np.column_stack((values, ftv_prefix(frame, timing)))
    return values


def clinical_offset_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    timing: str,
    *,
    include_ftv: bool,
    c_grid: Sequence[float],
    inner_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    encoder = ClinicalEncoder().fit(train)
    train_x = encoder.transform(train)
    validation_x = encoder.transform(validation)
    test_x = encoder.transform(test)
    if include_ftv:
        train_x = np.column_stack((train_x, ftv_prefix(train, timing)))
        validation_x = np.column_stack((validation_x, ftv_prefix(validation, timing)))
        test_x = np.column_stack((test_x, ftv_prefix(test, timing)))
    train_y = train["label_pcr"].to_numpy(int)
    c_value = _select_c(
        train_x, train_y, validation_x, validation["label_pcr"].to_numpy(int), c_grid
    )
    final = _logistic(c_value).fit(train_x, train_y)
    test_logit = np.asarray(final.decision_function(test_x), dtype=np.float64)
    oof = np.full(len(train), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(inner_seed))
    for inner_train, holdout in splitter.split(train, train_y):
        inner_fit = train.iloc[inner_train]
        inner_holdout = train.iloc[holdout]
        inner_encoder = ClinicalEncoder().fit(inner_fit)
        fit_x = inner_encoder.transform(inner_fit)
        holdout_x = inner_encoder.transform(inner_holdout)
        if include_ftv:
            fit_x = np.column_stack((fit_x, ftv_prefix(inner_fit, timing)))
            holdout_x = np.column_stack((holdout_x, ftv_prefix(inner_holdout, timing)))
        model = _logistic(c_value).fit(fit_x, train_y[inner_train])
        oof[holdout] = model.decision_function(holdout_x)
    if not np.isfinite(oof).all() or not np.isfinite(test_logit).all():
        raise RuntimeError("clinical offset logits are incomplete")
    return oof, test_logit, c_value


def image_offset_inputs(
    train_state: np.ndarray,
    validation_state: np.ndarray,
    test_state: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    timing: str,
    *,
    c_grid: Sequence[float],
    inner_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train_prefix = image_prefix(train_state, timing)
    validation_prefix = image_prefix(validation_state, timing)
    test_prefix = image_prefix(test_state, timing)
    train_logits: list[np.ndarray] = []
    test_logits: list[np.ndarray] = []
    selected: list[float] = []
    for offset, projection_seed in enumerate(PROJECTION_SEEDS):
        train_x = gaussian_projection(train_prefix, projection_seed)
        validation_x = gaussian_projection(validation_prefix, projection_seed)
        test_x = gaussian_projection(test_prefix, projection_seed)
        c_value = _select_c(train_x, train_y, validation_x, validation_y, c_grid)
        selected.append(c_value)
        train_logits.append(
            _inner_oof_numeric(train_x, train_y, c_value, inner_seed + offset)
        )
        final = _logistic(c_value).fit(train_x, train_y)
        test_logits.append(np.asarray(final.decision_function(test_x), dtype=np.float64))
    return np.mean(train_logits, axis=0), np.mean(test_logits, axis=0), selected


def fit_offset(baseline_oof_logit: np.ndarray, image_oof_score: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    baseline = np.asarray(baseline_oof_logit, dtype=np.float64)
    image = np.asarray(image_oof_score, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = baseline + parameters[0] + parameters[1] * image
        probability = expit(linear)
        value = float(np.logaddexp(0.0, linear).sum() - (labels * linear).sum())
        residual = probability - labels
        gradient = np.asarray([residual.sum(), (residual * image).sum()])
        return value, gradient

    result = minimize(
        lambda value: objective(value)[0],
        np.zeros(2),
        jac=lambda value: objective(value)[1],
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    if not result.success and np.linalg.norm(result.jac) > 1e-5:
        raise RuntimeError(f"offset fit failed: {result.message}")
    if not np.isfinite(result.x).all():
        raise RuntimeError("offset parameters are non-finite")
    return float(result.x[0]), float(result.x[1])


def classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise ValueError("classification metrics require both classes")
    return {
        "n": float(len(labels)),
        "n_positive": float(labels.sum()),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def _state_lookup(state_path: str | Path) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    patient_ids, state = validate_state_archive(state_path)
    with np.load(state_path, allow_pickle=False) as payload:
        radiomics = np.asarray(payload["radiomics_prediction"], dtype=np.float32)
        ftv = np.asarray(payload["ftv_prediction"], dtype=np.float32)
    if radiomics.shape != (808, 4, 16) or ftv.shape != (808, 4):
        raise ValueError("grounding head export shape failed")
    return {patient_id: index for index, patient_id in enumerate(patient_ids)}, state, radiomics, ftv


def evaluate_fold_cell(
    manifest_fold: pd.DataFrame,
    state_path: str | Path,
    *,
    seed: int,
    fold: int,
    arm: str,
    population: str = "primary_375",
    feature_source: str = "state",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup, state, radiomics_head, _ = _state_lookup(state_path)
    frame = manifest_fold.copy()
    frame["split"] = frame["split"].replace({"val": "validation"})
    if population == "primary_375":
        frame = frame.loc[frame["ftv_complete"].eq(1)].copy()
        include_ftv = True
    elif population == "secondary_808":
        include_ftv = False
    else:
        raise ValueError("population must be primary_375 or secondary_808")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("outer fold lacks train/validation/test")
    frame["state_index"] = frame["patient_id"].map(lookup)
    if frame["state_index"].isna().any():
        raise ValueError("state archive misses evaluation patient")
    by_split = {name: group.reset_index(drop=True) for name, group in frame.groupby("split", sort=False)}
    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    c_grid = tuple(map(float, load_protocol()["evaluation"]["l2_grid"]))
    source = state if feature_source == "state" else radiomics_head
    for timing_index, timing in enumerate(TIMINGS):
        train, validation, test = by_split["train"], by_split["validation"], by_split["test"]
        train_y = train["label_pcr"].to_numpy(int)
        validation_y = validation["label_pcr"].to_numpy(int)
        baseline_oof, baseline_test, baseline_c = clinical_offset_inputs(
            train, validation, test, timing, include_ftv=include_ftv, c_grid=c_grid,
            inner_seed=7919 + fold * 101 + timing_index,
        )
        image_oof, image_test, image_c = image_offset_inputs(
            source[train["state_index"].to_numpy(int)],
            source[validation["state_index"].to_numpy(int)],
            source[test["state_index"].to_numpy(int)],
            train_y, validation_y, timing, c_grid=c_grid,
            inner_seed=104729 + fold * 101 + timing_index,
        )
        alpha, beta = fit_offset(baseline_oof, image_oof, train_y)
        baseline_probability = expit(baseline_test)
        augmented_probability = expit(baseline_test + alpha + beta * image_test)
        for row, baseline, augmented, image_score in zip(
            test.itertuples(index=False), baseline_probability, augmented_probability, image_test
        ):
            predictions.append(
                {
                    "patient_id": str(row.patient_id),
                    "fold": int(fold),
                    "seed": int(seed),
                    "arm": str(arm),
                    "timing": timing,
                    "population": population,
                    "feature_source": feature_source,
                    "label_pcr": int(row.label_pcr),
                    "baseline_probability": float(baseline),
                    "augmented_probability": float(augmented),
                    "image_score": float(image_score),
                }
            )
        diagnostics.append(
            {
                "fold": fold, "seed": seed, "arm": arm, "timing": timing,
                "population": population, "feature_source": feature_source,
                "baseline_c": baseline_c, "image_projection_c": image_c,
                "alpha": alpha, "beta": beta,
                "inner_oof_rows": len(train), "test_rows": len(test),
            }
        )
    return predictions, diagnostics


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["seed", "arm", "timing", "population", "feature_source"]
    for values, group in predictions.groupby(keys, sort=True):
        labels = group["label_pcr"].to_numpy(int)
        baseline = classification_metrics(labels, group["baseline_probability"].to_numpy(float))
        augmented = classification_metrics(labels, group["augmented_probability"].to_numpy(float))
        row = dict(zip(keys, values))
        row.update({f"baseline_{name}": value for name, value in baseline.items()})
        row.update({f"augmented_{name}": value for name, value in augmented.items()})
        row.update(
            {
                "delta_auroc": augmented["auroc"] - baseline["auroc"],
                "delta_auprc": augmented["auprc"] - baseline["auprc"],
                "brier_improvement": baseline["brier"] - augmented["brier"],
                "log_loss_improvement": baseline["log_loss"] - augmented["log_loss"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["seed", "fold", "arm", "timing", "population", "feature_source"]
    for values, group in predictions.groupby(keys, sort=True):
        labels = group["label_pcr"].to_numpy(int)
        baseline = classification_metrics(labels, group["baseline_probability"].to_numpy(float))
        augmented = classification_metrics(labels, group["augmented_probability"].to_numpy(float))
        row = dict(zip(keys, values))
        row.update({f"baseline_{name}": value for name, value in baseline.items()})
        row.update({f"augmented_{name}": value for name, value in augmented.items()})
        row["delta_auroc"] = augmented["auroc"] - baseline["auroc"]
        row["delta_auprc"] = augmented["auprc"] - baseline["auprc"]
        row["brier_improvement"] = baseline["brier"] - augmented["brier"]
        rows.append(row)
    return pd.DataFrame(rows)


def mechanism_metrics(
    manifest: pd.DataFrame,
    state_root: str | Path,
    target_root: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            fold_frame = manifest.loc[
                manifest["fold"].eq(fold) & manifest["split"].eq("test") & manifest["ftv_complete"].eq(1)
            ]
            targets = FoldTargets.load(Path(target_root) / f"fold_{fold}_targets.private.npz")
            target_lookup = {patient_id: index for index, patient_id in enumerate(targets.patient_ids)}
            target_indices = np.asarray([target_lookup[value] for value in fold_frame["patient_id"].astype(str)])
            rad_target = targets.radiomics[target_indices]
            rad_mask = targets.radiomics_mask[target_indices]
            ftv_target = targets.ftv[target_indices]
            ftv_mask = targets.ftv_mask[target_indices]
            for arm in ("D2", "D3"):
                path = Path(state_root) / f"seed{seed}_fold{fold}_{arm}_states.private.npz"
                lookup, _, rad_prediction, ftv_prediction = _state_lookup(path)
                indices = np.asarray([lookup[value] for value in fold_frame["patient_id"].astype(str)])
                rad_prediction = rad_prediction[indices]
                ftv_prediction = ftv_prediction[indices]
                rad_rhos: list[float] = []
                for component in range(16):
                    valid = rad_mask & np.isfinite(rad_target[..., component]) & np.isfinite(rad_prediction[..., component])
                    if valid.sum() >= 3:
                        rho = float(spearmanr(rad_target[..., component][valid], rad_prediction[..., component][valid]).statistic)
                        if np.isfinite(rho):
                            rad_rhos.append(rho)
                static_rhos: list[float] = []
                for visit in range(4):
                    valid = ftv_mask[:, visit]
                    if valid.sum() >= 3:
                        rho = float(spearmanr(ftv_target[:, visit][valid], ftv_prediction[:, visit][valid]).statistic)
                        if np.isfinite(rho):
                            static_rhos.append(rho)
                delta_rhos: list[float] = []
                for visit in range(3):
                    valid = ftv_mask[:, visit] & ftv_mask[:, visit + 1]
                    if valid.sum() >= 3:
                        truth = ftv_target[:, visit + 1] - ftv_target[:, visit]
                        predicted = ftv_prediction[:, visit + 1] - ftv_prediction[:, visit]
                        rho = float(spearmanr(truth[valid], predicted[valid]).statistic)
                        if np.isfinite(rho):
                            delta_rhos.append(rho)
                rows.append(
                    {
                        "seed": seed, "fold": fold, "arm": arm,
                        "radiomics_pc_macro_spearman": float(np.mean(rad_rhos)) if rad_rhos else np.nan,
                        "static_ftv_macro_spearman": float(np.mean(static_rhos)) if static_rhos else np.nan,
                        "delta_ftv_macro_spearman": float(np.mean(delta_rhos)) if delta_rhos else np.nan,
                        "radiomics_components": len(rad_rhos),
                    }
                )
    return pd.DataFrame(rows)


def stratified_early_macro_bootstrap(
    predictions: pd.DataFrame,
    *,
    arm: str,
    reference_arm: str | None = None,
    draws: int = 2000,
    seed: int = 260817,
) -> dict[str, float]:
    frame = predictions.loc[
        predictions["timing"].isin(("T0-T1", "T0-T2")) & predictions["arm"].eq(arm)
    ].copy()
    averaged = frame.groupby(["patient_id", "fold", "timing", "label_pcr"], as_index=False)[
        ["baseline_probability", "augmented_probability"]
    ].mean()
    if reference_arm is None:
        averaged["reference_probability"] = averaged["baseline_probability"]
    else:
        reference = predictions.loc[
            predictions["timing"].isin(("T0-T1", "T0-T2")) & predictions["arm"].eq(reference_arm)
        ].groupby(["patient_id", "fold", "timing", "label_pcr"], as_index=False)["augmented_probability"].mean()
        averaged = averaged.merge(
            reference.rename(columns={"augmented_probability": "reference_probability"}),
            on=["patient_id", "fold", "timing", "label_pcr"], validate="one_to_one",
        )
    patients = averaged[["patient_id", "fold", "label_pcr"]].drop_duplicates()
    generator = np.random.default_rng(int(seed))
    effects: list[float] = []
    for _ in range(int(draws)):
        sampled: list[str] = []
        for _, stratum in patients.groupby(["fold", "label_pcr"], sort=True):
            values = stratum["patient_id"].astype(str).to_numpy()
            sampled.extend(generator.choice(values, size=len(values), replace=True).tolist())
        draw_effects: list[float] = []
        # Repeated patient IDs intentionally preserve bootstrap multiplicity.
        for timing in ("T0-T1", "T0-T2"):
            part = averaged.loc[averaged["timing"].eq(timing)].set_index("patient_id")
            sample = part.loc[sampled]
            labels = sample["label_pcr"].to_numpy(int)
            draw_effects.append(
                float(roc_auc_score(labels, sample["augmented_probability"]) - roc_auc_score(labels, sample["reference_probability"]))
            )
        effects.append(float(np.mean(draw_effects)))
    array = np.asarray(effects)
    return {
        "draws": int(draws),
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
    }


def validate_oof_coverage(predictions: pd.DataFrame, expected_patients: int) -> None:
    keys = ["seed", "arm", "timing", "population", "feature_source"]
    for _, group in predictions.groupby(keys, sort=False):
        if len(group) != expected_patients or group["patient_id"].duplicated().any():
            raise ValueError("every evaluation cell must contain each OOF test patient exactly once")


def load_outcome_manifest_after_lock(path: str | Path) -> pd.DataFrame:
    verify_evaluation_lock()
    required = {
        "patient_id", "fold", "split", "label_pcr", "ftv_complete", *CLINICAL_FIELDS,
        "FTV_T0", "FTV_T1", "FTV_T2", "FTV_T3",
    }
    frame = pd.read_csv(path, usecols=lambda column: column in required, dtype={"patient_id": str})
    if set(frame.columns) != required:
        raise ValueError(f"outcome manifest misses {sorted(required - set(frame.columns))}")
    if len(frame) != 808 * 5 or frame.duplicated(["patient_id", "fold"]).any():
        raise ValueError("outcome manifest must contain 808 unique patients in each fold")
    frame["label_pcr"] = pd.to_numeric(frame["label_pcr"], errors="raise").astype(int)
    if not frame["label_pcr"].isin((0, 1)).all():
        raise ValueError("pCR outcome must be binary")
    return frame


__all__ = [
    "ClinicalEncoder", "classification_metrics", "clinical_offset_inputs", "evaluate_fold_cell",
    "fit_offset", "fold_metrics", "gaussian_projection", "image_offset_inputs", "image_prefix",
    "load_outcome_manifest_after_lock", "mechanism_metrics", "pooled_metrics",
    "stratified_early_macro_bootstrap", "validate_oof_coverage"
]
