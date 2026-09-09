#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT / "src"))
sys.path.insert(0, str(PARENT.parent / "conditional_pcr_contrastive_ceiling" / "src"))
from bilateral_context.contracts import FOLDS, PRIMARY_N, V6, fold_frame, primary_ids, private_token  # noqa: E402
from conditional_ceiling.clinical import CLINICAL_FIELDS, TrainOnlyClinicalEncoder  # noqa: E402

MANIFEST = PARENT.parents[1] / "additional_experiments/raw_spatial_pcr_ceiling/manifests/formal_input.private.csv"
TIMINGS = ("T0", "T0_T1", "T0_T2")
END = {"T0": 1, "T0_T1": 2, "T0_T2": 3}


def outcome_table() -> pd.DataFrame:
    columns = ["patient_id", "label_pcr", *CLINICAL_FIELDS, "FTV_T0", "FTV_T1", "FTV_T2", "FTV_T3"]
    frame = pd.read_csv(MANIFEST, usecols=columns, dtype={"patient_id": str})
    frame = frame.drop_duplicates("patient_id").set_index("patient_id").loc[list(primary_ids())]
    frame["label_pcr"] = pd.to_numeric(frame["label_pcr"], errors="raise").astype(int)
    for column in ("FTV_T0", "FTV_T1", "FTV_T2", "FTV_T3"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if len(frame) != PRIMARY_N or not frame["label_pcr"].isin((0, 1)).all():
        raise ValueError("primary outcome contract failed")
    return frame


def load_features() -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "features/frozen_bilateral_features.private.npz"
    with np.load(path, allow_pickle=False) as payload:
        tokens = tuple(map(str, payload["patient_token"].tolist()))
        expected = tuple(private_token(pid) for pid in primary_ids())
        if tokens != expected:
            raise ValueError("frozen feature patient order does not match primary contract")
        global_features = np.asarray(payload["global_features"], dtype=np.float64)
        spatial_features = np.asarray(payload["spatial_features"], dtype=np.float64)
    if global_features.shape != (PRIMARY_N, 4, 4608) or spatial_features.shape != (PRIMARY_N, 4, 1536):
        raise ValueError("frozen feature shapes do not match protocol")
    return global_features, spatial_features


def prefix(values: np.ndarray, timing: str) -> np.ndarray:
    end = END[timing]
    observed = values[:, :end]
    if end == 1:
        return observed[:, 0]
    differences = observed[:, 1:] - observed[:, :-1]
    return np.concatenate((observed.reshape(len(values), -1), differences.reshape(len(values), -1)), axis=1)


class ImagePreprocessor:
    def __init__(self) -> None:
        self.scalers: list[StandardScaler] = []
        self.pcas: list[PCA] = []

    def fit(self, values: np.ndarray, train: np.ndarray) -> "ImagePreprocessor":
        self.scalers = []
        self.pcas = []
        for visit in range(4):
            scaler = StandardScaler().fit(values[train, visit])
            scaled = scaler.transform(values[train, visit])
            pca = PCA(n_components=32, svd_solver="randomized", random_state=260812 + visit).fit(scaled)
            self.scalers.append(scaler)
            self.pcas.append(pca)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.stack([pca.transform(scaler.transform(values[:, visit])) for visit, (scaler, pca) in enumerate(zip(self.scalers, self.pcas))], axis=1)


class RadioPreprocessor:
    def __init__(self) -> None:
        self.medians: np.ndarray | None = None

    def fit(self, target: np.ndarray, mask: np.ndarray, train: np.ndarray) -> "RadioPreprocessor":
        values = target[train, :3].copy()
        valid = mask[train, :3, None] & np.isfinite(values)
        medians = np.zeros(16, dtype=np.float64)
        for pc in range(16):
            finite = values[:, :, pc][valid[:, :, pc]]
            medians[pc] = float(np.median(finite)) if finite.size else 0.0
        self.medians = medians
        return self

    def transform(self, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("radiomics preprocessor is not fitted")
        values = target[:, :3].astype(np.float64).copy()
        valid = mask[:, :3, None] & np.isfinite(values)
        for pc in range(16):
            values[:, :, pc] = np.where(valid[:, :, pc], values[:, :, pc], self.medians[pc])
        # Keep missingness visible to the comparator, but never feed it to DINO branches.
        observed_fraction = mask[:, :3].astype(np.float64)
        return np.concatenate((values, observed_fraction[:, :, None]), axis=-1)


def clinical_ftv_features(encoder: TrainOnlyClinicalEncoder, frame: pd.DataFrame, timing: str) -> np.ndarray:
    end = END[timing]
    ftv = np.log1p(frame[[f"FTV_{visit}" for visit in ("T0", "T1", "T2")[:end]]].to_numpy(float))
    return np.column_stack((encoder.transform(frame), ftv))


def fit_probability(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="liblinear", max_iter=3000))
    model.fit(train_x, train_y)
    return np.clip(model.predict_proba(test_x)[:, 1], 1e-6, 1 - 1e-6)


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def fuse(base: np.ndarray, image: np.ndarray, y: np.ndarray, test_base: np.ndarray, test_image: np.ndarray) -> np.ndarray:
    base_logit = logit(base)
    image_logit = logit(image)
    def objective(value: np.ndarray) -> float:
        score = base_logit + value[0] + value[1] * image_logit
        return float(np.mean(np.logaddexp(0.0, score) - y * score))
    fitted = minimize(objective, np.zeros(2), method="BFGS")
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError("offset fusion failed")
    score = logit(test_base) + fitted.x[0] + fitted.x[1] * logit(test_image)
    return 1.0 / (1.0 + np.exp(-score))


def classification(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=(0, 1)))}


def bootstrap(y: np.ndarray, ref: np.ndarray, cand: np.ndarray, seed: int = 260812) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    values = []
    for _ in range(2000):
        sample = np.concatenate((rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)))
        values.append(roc_auc_score(y[sample], cand[sample]) - roc_auc_score(y[sample], ref[sample]))
    values = np.asarray(values)
    return {"delta_auroc": float(values.mean()), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "draws": 2000}


def radiomics_probe(image_values: dict[str, np.ndarray], target: np.ndarray, target_mask: np.ndarray, splits: pd.DataFrame, ids: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    split_index = splits.set_index(["patient_id", "fold"])
    for fold in FOLDS:
        labels = np.asarray([str(split_index.loc[(pid, fold), "split"]) for pid in ids])
        train = labels == "train"
        validation = labels == "val"
        test = labels == "test"
        for arm, values in image_values.items():
            prep = ImagePreprocessor().fit(values, train | validation)
            transformed = prep.transform(values)
            for pc in range(16):
                for visit in range(3):
                    valid_train = train & target_mask[:, visit]
                    valid_val = validation & target_mask[:, visit]
                    valid_test = test & target_mask[:, visit]
                    if valid_train.sum() < 20 or valid_val.sum() < 10 or valid_test.sum() < 10:
                        continue
                    selected_alpha = 1.0
                    best = float("inf")
                    for alpha in (0.1, 1.0, 10.0, 100.0):
                        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                        model.fit(transformed[valid_train, visit], target[valid_train, visit, pc])
                        error = mean_squared_error(target[valid_val, visit, pc], model.predict(transformed[valid_val, visit]))
                        if error < best - 1e-12 or abs(error - best) <= 1e-12 and alpha > selected_alpha:
                            best, selected_alpha = error, alpha
                    fit_mask = (train | validation) & target_mask[:, visit]
                    model = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha))
                    model.fit(transformed[fit_mask, visit], target[fit_mask, visit, pc])
                    prediction = model.predict(transformed[valid_test, visit])
                    actual = target[valid_test, visit, pc]
                    rows.append({"fold": fold, "arm": arm, "pc": pc, "visit": f"T{visit}", "spearman": float(spearmanr(actual, prediction).statistic) if np.unique(actual).size > 1 and np.unique(prediction).size > 1 else np.nan, "r2": float(r2_score(actual, prediction)), "n_test": int(valid_test.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    lock = json.loads((ROOT / "FROZEN_READOUT_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("pcr_open") is not True:
        raise SystemExit("frozen readout lock is not open")
    frame = outcome_table()
    ids = tuple(primary_ids())
    y = frame.loc[list(ids), "label_pcr"].to_numpy(int)
    split_frame = fold_frame()
    split_index = split_frame.set_index(["patient_id", "fold"])
    global_values, spatial_values = load_features()
    image_values = {"DINO_GLOBAL": global_values, "DINO_SPATIAL": spatial_values}
    radio_target = []
    radio_mask = []
    for fold in FOLDS:
        with np.load(V6 / "features/private/fold_targets" / f"fold_{fold}_targets.private.npz", allow_pickle=False) as payload:
            target_ids = tuple(map(str, payload["patient_id"].tolist()))
            pos = {pid: i for i, pid in enumerate(target_ids)}
            ix = [pos[pid] for pid in ids]
            radio_target.append(np.asarray(payload["target"])[ix])
            radio_mask.append(np.asarray(payload["target_mask"])[ix])
    fold_rows = []
    probe_inputs = {"DINO_GLOBAL": global_values, "DINO_SPATIAL": spatial_values}
    probe_rows = []
    (ROOT / "predictions").mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        split = np.asarray([str(split_index.loc[(pid, fold), "split"]) for pid in ids])
        pool = np.flatnonzero(np.isin(split, ("train", "val")))
        test = np.flatnonzero(split == "test")
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=260812 + fold)
        for timing in TIMINGS:
            base_oof = np.full(len(ids), np.nan)
            base_tests = []
            candidate_oofs = {"DINO_GLOBAL": np.full(len(ids), np.nan), "DINO_SPATIAL": np.full(len(ids), np.nan), "RADIO_TARGET": np.full(len(ids), np.nan)}
            candidate_tests = {key: [] for key in candidate_oofs}
            target = radio_target[fold]
            mask = radio_mask[fold]
            for train_rel, val_rel in inner.split(pool, y[pool]):
                train = pool[train_rel]
                val = pool[val_rel]
                train_frame = frame.loc[list(np.asarray(ids)[train])]
                val_frame = frame.loc[list(np.asarray(ids)[val])]
                test_frame = frame.loc[list(np.asarray(ids)[test])]
                encoder = TrainOnlyClinicalEncoder(CLINICAL_FIELDS).fit(train_frame)
                base_train = clinical_ftv_features(encoder, train_frame, timing)
                base_val = clinical_ftv_features(encoder, val_frame, timing)
                base_test = clinical_ftv_features(encoder, test_frame, timing)
                base_oof[val] = fit_probability(base_train, y[train], base_val)
                base_tests.append(fit_probability(base_train, y[train], base_test))
                for arm, values in image_values.items():
                    prep = ImagePreprocessor().fit(values, train)
                    projected = prep.transform(values)
                    candidate_oofs[arm][val] = fit_probability(prefix(projected[train], timing), y[train], prefix(projected[val], timing))
                    candidate_tests[arm].append(fit_probability(prefix(projected[train], timing), y[train], prefix(projected[test], timing)))
                radio_prep = RadioPreprocessor().fit(target, mask, train)
                radio_values = radio_prep.transform(target, mask)
                candidate_oofs["RADIO_TARGET"][val] = fit_probability(prefix(radio_values[train], timing), y[train], prefix(radio_values[val], timing))
                candidate_tests["RADIO_TARGET"].append(fit_probability(prefix(radio_values[train], timing), y[train], prefix(radio_values[test], timing)))
            valid = np.isfinite(base_oof)
            base_test_probability = np.mean(base_tests, axis=0)
            for arm in candidate_oofs:
                candidate_test_probability = np.mean(candidate_tests[arm], axis=0)
                fused = fuse(base_oof[valid], candidate_oofs[arm][valid], y[valid], base_test_probability, candidate_test_probability)
                values = classification(y[test], fused)
                base_values = classification(y[test], base_test_probability)
                fold_rows.append({"fold": fold, "timing": timing, "arm": arm, **values, "base_auroc": base_values["auroc"], "base_auprc": base_values["auprc"], "base_brier": base_values["brier"], "delta_auroc": values["auroc"] - base_values["auroc"], "delta_auprc": values["auprc"] - base_values["auprc"], "delta_brier": values["brier"] - base_values["brier"]})
                # Store only aggregate-ready arrays; no patient identifiers enter public metrics.
                key = (fold, timing, arm)
                np.savez_compressed(ROOT / "predictions" / f"{fold}_{timing}_{arm}.private.npz", y=y[test], base=base_test_probability, fused=fused)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(ROOT / "metrics/frozen_readout_fold_metrics.csv", index=False)
    summary = fold_metrics.groupby(["timing", "arm"], as_index=False).mean(numeric_only=True)
    summary.to_csv(ROOT / "metrics/frozen_readout_pcr_metrics.csv", index=False)
    bootstrap_rows = []
    prediction_paths = sorted((ROOT / "predictions").glob("*.private.npz"))
    for arm in ("DINO_GLOBAL", "DINO_SPATIAL", "RADIO_TARGET"):
        for timing in TIMINGS:
            ys, bases, fused = [], [], []
            for fold in FOLDS:
                path = ROOT / "predictions" / f"{fold}_{timing}_{arm}.private.npz"
                with np.load(path, allow_pickle=False) as payload:
                    ys.append(payload["y"]); bases.append(payload["base"]); fused.append(payload["fused"])
            y_pool, base_pool, fused_pool = np.concatenate(ys), np.concatenate(bases), np.concatenate(fused)
            bootstrap_rows.append({"timing": timing, "arm": arm, **bootstrap(y_pool, base_pool, fused_pool, 260812 + sum(map(ord, timing + arm)))})
    pd.DataFrame(bootstrap_rows).to_csv(ROOT / "metrics/frozen_readout_bootstrap.csv", index=False)
    for fold in FOLDS:
        with np.load(V6 / "features/private/fold_targets" / f"fold_{fold}_targets.private.npz", allow_pickle=False) as payload:
            pos = {str(pid): i for i, pid in enumerate(payload["patient_id"].tolist())}
            ix = [pos[pid] for pid in ids]
            probe_target = np.asarray(payload["target"])[ix]
            probe_mask = np.asarray(payload["target_mask"])[ix]
        probe_rows.append(radiomics_probe(probe_inputs, probe_target, probe_mask, split_frame, ids))
    probe = pd.concat(probe_rows, ignore_index=True)
    probe.to_csv(ROOT / "metrics/frozen_readout_radiomics_probe.csv", index=False)
    early = summary.loc[summary["timing"].isin(("T0_T1", "T0_T2"))]
    spatial = early.loc[early["arm"] == "DINO_SPATIAL"]
    radio = early.loc[early["arm"] == "RADIO_TARGET"]
    checks = {
        "spatial_both_early_positive": bool((spatial["delta_auroc"] > 0).all()),
        "spatial_early_macro_delta": float(spatial["delta_auroc"].mean()),
        "spatial_vs_radio_early_macro_auroc": float(spatial["auroc"].mean() - radio["auroc"].mean()),
        "spatial_vs_radio_positive": bool(spatial["auroc"].mean() > radio["auroc"].mean()),
        "spatial_bootstrap_ci_low_all": float(pd.read_csv(ROOT / "metrics/frozen_readout_bootstrap.csv").query("arm == 'DINO_SPATIAL' and timing in ['T0_T1','T0_T2']")["ci_low"].min()),
        "probe_global_mean_spearman": float(probe.loc[probe.arm == "DINO_GLOBAL", "spearman"].mean()),
        "probe_spatial_mean_spearman": float(probe.loc[probe.arm == "DINO_SPATIAL", "spearman"].mean()),
    }
    if checks["spatial_early_macro_delta"] >= 0.01 and checks["spatial_both_early_positive"]:
        decision = "BILATERAL_DINO_CACHE_CONDITIONAL_SIGNAL"
    elif checks["spatial_vs_radio_positive"] and checks["spatial_early_macro_delta"] > 0:
        decision = "BILATERAL_DINO_CACHE_SIGNAL_BUT_NOT_ROBUST"
    else:
        decision = "BILATERAL_DINO_CACHE_NO_CONDITIONAL_SIGNAL"
    result = {"status": "COMPLETE", "decision": decision, "checks": checks, "cohort": PRIMARY_N, "outcome_fields_read": ["label_pcr"], "clinical_fields_read": list(CLINICAL_FIELDS), "formal_adapter_training": "LOCKED"}
    (ROOT / "metrics/decision.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
