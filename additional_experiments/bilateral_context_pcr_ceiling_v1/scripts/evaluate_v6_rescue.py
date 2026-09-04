#!/usr/bin/env python3
"""Outcome-gated exploratory rescue evaluation of the already frozen V6 states."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "conditional_pcr_contrastive_ceiling" / "src"))
from bilateral_context.contracts import (  # noqa: E402
    FOLDS, PRIMARY_N, SEEDS, V6, atomic_json, canonical_sha256, fold_frame, primary_ids,
)
from conditional_ceiling.clinical import CLINICAL_FIELDS, TrainOnlyClinicalEncoder  # noqa: E402

VISITS = ("T0", "T0_T1", "T0_T2")
TIMING_END = {"T0": 1, "T0_T1": 2, "T0_T2": 3}
MANIFEST = ROOT.parents[1] / "additional_experiments/raw_spatial_pcr_ceiling/manifests/formal_input.private.csv"


def read_outcome_table() -> pd.DataFrame:
    columns = ["patient_id", "label_pcr", *CLINICAL_FIELDS, "FTV_T0", "FTV_T1", "FTV_T2", "FTV_T3"]
    frame = pd.read_csv(MANIFEST, usecols=columns, dtype={"patient_id": str})
    frame = frame.loc[frame["patient_id"].isin(primary_ids())].sort_values("patient_id").drop_duplicates("patient_id").copy()
    if len(frame) != PRIMARY_N or frame["patient_id"].duplicated().any():
        raise ValueError("primary outcome table is not exactly 375 patients")
    frame["label_pcr"] = pd.to_numeric(frame["label_pcr"], errors="raise").astype(int)
    if not frame["label_pcr"].isin((0, 1)).all():
        raise ValueError("pCR must be binary")
    for col in ("FTV_T0", "FTV_T1", "FTV_T2", "FTV_T3"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.set_index("patient_id").sort_index()


def state_table(arm: str) -> tuple[np.ndarray, tuple[str, ...]]:
    paths = sorted((V6 / "features/private/states").glob(f"seed2026_fold*_{arm}.npz"))
    if len(paths) != 5:
        raise FileNotFoundError(f"V6 state matrix incomplete for {arm}")
    records = []
    ids = None
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            current_ids = tuple(map(str, payload["patient_id"].tolist()))
            current = np.asarray(payload["parallel_state" if arm != "C0" else "c0_state"], dtype=np.float64)
        if ids is None:
            ids = current_ids
        if current_ids != ids:
            raise ValueError(f"V6 state patient order differs: {path}")
        records.append(current)
    # Each fold file is a different OOF archive; the evaluator selects the matching fold.
    return np.asarray(records), ids  # [fold, patient, visit, dim]


def load_states() -> tuple[dict[str, dict[int, np.ndarray]], tuple[str, ...]]:
    result: dict[str, dict[int, np.ndarray]] = {"C0_192": {}, "S0_256": {}, "S1_256": {}}
    ids = None
    for fold in FOLDS:
        for name, archive in (("C0_192", "S0_SUMMARY"), ("S0_256", "S0_SUMMARY"), ("S1_256", "S1_SPATIAL")):
            path = V6 / "features/private/states" / f"seed2026_fold{fold}_{archive}.npz"
            with np.load(path, allow_pickle=False) as payload:
                current_ids = tuple(map(str, payload["patient_id"].tolist()))
                key = "c0_state" if name == "C0_192" else "parallel_state"
                current = np.asarray(payload[key], dtype=np.float64)
            if ids is None:
                ids = current_ids
            if current_ids != ids:
                raise ValueError("V6 state archives do not share patient order")
            result[name][fold] = current
    assert ids is not None
    return result, ids


def prefix(state: np.ndarray, timing: str) -> np.ndarray:
    return state[:, : TIMING_END[timing]].reshape(len(state), -1)


def base_features(encoder: TrainOnlyClinicalEncoder, frame: pd.DataFrame, timing: str) -> np.ndarray:
    end = TIMING_END[timing]
    values = np.log1p(frame[[f"FTV_{v}" for v in ("T0", "T1", "T2")[:end]]].to_numpy(float))
    return np.column_stack((encoder.transform(frame), values))


def fit_probability(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="liblinear", max_iter=2000))
    model.fit(train_x, train_y)
    return np.clip(model.predict_proba(test_x)[:, 1], 1e-6, 1 - 1e-6)


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def offset_fusion(base_logit: np.ndarray, image_logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    def objective(value: np.ndarray) -> float:
        score = base_logit + value[0] + value[1] * image_logit
        return float(np.mean(np.logaddexp(0.0, score) - y * score))
    fitted = minimize(objective, np.zeros(2), method="BFGS")
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"offset fusion failed: {fitted.message}")
    return float(fitted.x[0]), float(fitted.x[1])


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=(0, 1))),
    }


def paired_bootstrap(y: np.ndarray, reference: np.ndarray, candidate: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    effects = []
    for _ in range(draws):
        sample = np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)))
        effects.append(roc_auc_score(y[sample], candidate[sample]) - roc_auc_score(y[sample], reference[sample]))
    values = np.asarray(effects)
    return {"delta_auroc": float(np.mean(values)), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "draws": int(draws)}


def main() -> None:
    lock_path = ROOT / "EXPLORATORY_EVALUATION_LOCK.json"
    if not lock_path.is_file():
        raise SystemExit("pCR evaluator is locked: create EXPLORATORY_EVALUATION_LOCK.json after cache/input checks")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("pcr_open") is not True:
        raise SystemExit("exploratory pCR lock is not open")
    outcome = read_outcome_table()
    states, state_ids = load_states()
    folds = fold_frame().set_index(["patient_id", "fold"])
    ids = tuple(sorted(set(primary_ids()) & set(state_ids)))
    y_all = outcome.loc[list(ids), "label_pcr"].to_numpy(int)
    seed_rows: list[dict[str, object]] = []
    pooled: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    for seed in SEEDS[:1]:
        for fold in FOLDS:
            labels = outcome.loc[list(ids), "label_pcr"].to_numpy(int)
            split = np.asarray([folds.loc[(pid, fold), "split"] for pid in ids])
            train_ids = np.flatnonzero(split == "train")
            test_ids = np.flatnonzero(split == "test")
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=260812 + seed + fold)
            for timing in VISITS:
                base_oof = np.full(len(ids), np.nan)
                base_test_parts: list[np.ndarray] = []
                image_oof = {"C0_192": np.full(len(ids), np.nan), "S0_256": np.full(len(ids), np.nan), "S1_256": np.full(len(ids), np.nan)}
                image_test_parts = {key: [] for key in image_oof}
                for inner_train_rel, inner_val_rel in inner.split(train_ids, labels[train_ids]):
                    inner_train = train_ids[inner_train_rel]
                    inner_val = train_ids[inner_val_rel]
                    train_frame = outcome.loc[list(np.asarray(ids)[inner_train])].copy()
                    val_frame = outcome.loc[list(np.asarray(ids)[inner_val])].copy()
                    test_frame = outcome.loc[list(np.asarray(ids)[test_ids])].copy()
                    encoder = TrainOnlyClinicalEncoder(CLINICAL_FIELDS).fit(train_frame)
                    base_train = base_features(encoder, train_frame, timing)
                    base_val = base_features(encoder, val_frame, timing)
                    base_test_features = base_features(encoder, test_frame, timing)
                    base_oof[inner_val] = fit_probability(base_train, labels[inner_train], base_val)
                    base_test_parts.append(fit_probability(base_train, labels[inner_train], base_test_features))
                    for arm in image_oof:
                        fold_state = states[arm][fold]
                        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="liblinear", max_iter=2000))
                        model.fit(prefix(fold_state[inner_train], timing), labels[inner_train])
                        image_oof[arm][inner_val] = model.predict_proba(prefix(fold_state[inner_val], timing))[:, 1]
                        image_test_parts[arm].append(model.predict_proba(prefix(fold_state[test_ids], timing))[:, 1])
                base_test = np.mean(base_test_parts, axis=0)
                valid = np.isfinite(base_oof)
                row_base = metrics(labels[test_ids], base_test)
                for arm in image_oof:
                    image_test = np.mean(image_test_parts[arm], axis=0)
                    alpha, beta = offset_fusion(logit(base_oof[valid]), logit(image_oof[arm][valid]), labels[valid])
                    fused = 1.0 / (1.0 + np.exp(-(logit(base_test) + alpha + beta * logit(image_test))))
                    values = metrics(labels[test_ids], fused)
                    seed_rows.append({"seed": seed, "fold": fold, "timing": timing, "arm": arm, **values, "base_auroc": row_base["auroc"], "base_auprc": row_base["auprc"], "base_brier": row_base["brier"], "delta_auroc": values["auroc"] - row_base["auroc"], "delta_auprc": values["auprc"] - row_base["auprc"], "delta_brier": values["brier"] - row_base["brier"], "alpha": alpha, "beta": beta})
                    pooled[(seed, timing, arm)] = {"y": labels[test_ids].copy(), "base": base_test.copy(), "candidate": fused.copy()}
    fold_metrics = pd.DataFrame(seed_rows)
    fold_metrics.to_csv(ROOT / "metrics/exploratory_fold_metrics.csv", index=False)
    summary = fold_metrics.groupby(["timing", "arm"], as_index=False).mean(numeric_only=True)
    summary.to_csv(ROOT / "metrics/exploratory_pcr_metrics.csv", index=False)
    bootstrap_rows = []
    for (seed, timing, arm), values in pooled.items():
        bootstrap_rows.append({"seed": seed, "timing": timing, "arm": arm, **paired_bootstrap(values["y"], values["base"], values["candidate"], 2000, 260814 + seed + sum(map(ord, timing + arm)))})
    pd.DataFrame(bootstrap_rows).to_csv(ROOT / "metrics/exploratory_bootstrap_metrics.csv", index=False)
    atomic_json(ROOT / "metrics/exploratory_evaluation_check.json", {"status": "COMPLETE", "rows": len(fold_metrics), "seeds": [2026], "outcome_fields_read": ["label_pcr"], "clinical_fields_read": list(CLINICAL_FIELDS), "primary_n": PRIMARY_N})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
