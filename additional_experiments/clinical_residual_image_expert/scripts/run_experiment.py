#!/usr/bin/env python3
"""Leakage-guarded Clinical Residual Image Expert on frozen T0 DCE caches.

The script deliberately owns no DICOM/NIfTI preprocessing: each image is read
from the formally validated MAMA-MIA three-phase T0 cache.  Patient-level
outputs, checkpoints, logs and bootstrap indices are private/ignored; only
aggregate metrics and audits are public.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPO = HERE.parents[3]
STATIC_ROOT = REPO / "additional_experiments" / "mama_mia_static_dce_baseline"
FOUNDATION_SRC = REPO / "additional_experiments" / "foundation_mri_baselines" / "src"
if str(FOUNDATION_SRC) not in sys.path:
    sys.path.insert(0, str(FOUNDATION_SRC))

from foundation_mri.data import FOLDS, ClinicalTable, load_clinical_labels, load_fold_manifest
from foundation_mri.evaluation import ClinicalEncoder, select_logistic


SEED = 2026
EXPECTED_FOLD_SHA = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
SHAPE = (32, 96, 96)
ARMS = ("clinical_only", "t0_image_only", "clinical_recalibration", "naive_clinical_image", "clinical_image_residual")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_root() -> Path:
    value = os.environ.get("ISPY2_PREPROCESSED_ROOT")
    if not value:
        raise RuntimeError("ISPY2_PREPROCESSED_ROOT is required")
    return Path(value).expanduser().resolve(strict=True)


def paths() -> dict[str, Path]:
    root = data_root()
    return {
        "folds": root / "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026" / "matched_patient_cv_splits_seed2026.csv",
        "clinical": root / "clinical_labels_complete4visits.csv",
        "cache": STATIC_ROOT / "cache",
        "static_predictions": STATIC_ROOT / "predictions" / "formal",
        "static_bootstrap": STATIC_ROOT / "manifests" / "bootstrap_indices_seed2026.private.npz",
    }


def torch_imports():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.video import R3D_18_Weights, r3d_18
    return torch, nn, DataLoader, Dataset, R3D_18_Weights, r3d_18


def set_seed(seed: int) -> None:
    torch, _, _, _, _, _ = torch_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigmoid(logit: np.ndarray) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def youden(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr - fpr)[valid])])


def metric_dict(y: np.ndarray, probability: np.ndarray, threshold: np.ndarray | float) -> dict[str, float]:
    hard = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "f1": float(f1_score(y, hard, zero_division=0)),
        "prevalence": float(np.mean(y)),
    }


def validate_static_source() -> None:
    source = STATIC_ROOT / "metrics" / "formal_validation.json"
    payload = json.loads(source.read_text())
    if payload.get("status") != "PASS" or payload.get("patients") != 808:
        raise RuntimeError("MAMA-MIA static source is not formally validated")
    cache = paths()["cache"]
    files = sorted(cache.glob("*_T0_static_dce.npz"))
    if len(files) != 808:
        raise RuntimeError("expected exactly 808 reused T0 cache files")


@dataclass(frozen=True)
class ClinicalFit:
    encoder: ClinicalEncoder
    scaler: StandardScaler
    model: LogisticRegression
    penalty: str
    c_value: float
    threshold: float

    def logits(self, clinical: ClinicalTable) -> np.ndarray:
        matrix = self.encoder.transform(clinical)
        return np.asarray(self.model.decision_function(self.scaler.transform(matrix)), dtype=np.float64)


def fit_fixed_clinical(clinical: ClinicalTable, indices: np.ndarray, *, penalty: str, c_value: float, seed: int) -> ClinicalFit:
    encoder = ClinicalEncoder.fit(clinical, indices)
    matrix = encoder.transform(clinical)
    scaler = StandardScaler().fit(matrix[indices])
    kwargs: dict[str, Any] = {"penalty": penalty}
    if penalty == "l1":
        kwargs["dual"] = False
    model = LogisticRegression(
        C=float(c_value), solver="liblinear", class_weight="balanced", max_iter=20_000,
        tol=1e-7, random_state=int(seed), **kwargs,
    )
    model.fit(scaler.transform(matrix[indices]), clinical.pcr[indices])
    return ClinicalFit(encoder, scaler, model, str(penalty), float(c_value), math.nan)


def select_outer_clinical(clinical: ClinicalTable, train: np.ndarray, val: np.ndarray, fold: int) -> ClinicalFit:
    encoder = ClinicalEncoder.fit(clinical, train)
    matrix = encoder.transform(clinical)
    selected = select_logistic(matrix[train], clinical.pcr[train], matrix[val], clinical.pcr[val], random_state=SEED + fold)
    return ClinicalFit(encoder, selected.scaler, selected.model, selected.penalty, selected.c_value, selected.threshold)


def inner_cross_fitted_logits(clinical: ClinicalTable, outer_train: np.ndarray, outer_fit: ClinicalFit, fold: int) -> np.ndarray:
    """Return one strictly out-of-fit clinical logit for every outer-train patient.

    The outer validation set selected the locked clinical hyperparameters. Each
    inner predictor then fits only its own inner-training population; no held
    out patient's outcome or features contribute to its imputer/scaler/model.
    """
    y = clinical.pcr[outer_train]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
    result = np.full(len(outer_train), np.nan, dtype=np.float64)
    for inner_fold, (relative_train, relative_holdout) in enumerate(splitter.split(outer_train, y)):
        train_indices = outer_train[relative_train]
        holdout_indices = outer_train[relative_holdout]
        inner = fit_fixed_clinical(clinical, train_indices, penalty=outer_fit.penalty, c_value=outer_fit.c_value, seed=SEED + fold * 10 + inner_fold)
        result[relative_holdout] = inner.logits(clinical)[holdout_indices]
    if not np.isfinite(result).all():
        raise RuntimeError("inner cross-fitted clinical logits do not cover outer training")
    return result


def verify_split(folds: Any, clinical: ClinicalTable) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if folds.sha256 != EXPECTED_FOLD_SHA or len(clinical.patient_ids) != 808 or int(clinical.pcr.sum()) != 275:
        raise RuntimeError("locked 808-patient clinical cohort drifted")
    for fold in FOLDS:
        roles = folds.roles(fold, clinical.patient_ids)
        groups = {name: set(clinical.patient_ids[roles == name].tolist()) for name in ("train", "val", "test")}
        if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
            raise RuntimeError(f"patient overlap in fold {fold}")
        for split, group in groups.items():
            labels = clinical.pcr[[i for i, pid in enumerate(clinical.patient_ids) if pid in group]]
            rows.append({"fold": fold, "split": split, "n": len(group), "pcr_positive": int(labels.sum()), "pcr_prevalence": float(labels.mean())})
    return rows


def load_image_only(clinical: ClinicalTable, folds: Any) -> pd.DataFrame:
    frames = [pd.read_csv(paths()["static_predictions"] / f"three_phase_fold_{fold}.private.csv", dtype={"patient_id": str}) for fold in FOLDS]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["split"].eq("test")].copy()
    if len(frame) != 808 or frame.patient_id.duplicated().any() or set(frame.patient_id) != set(clinical.patient_ids):
        raise RuntimeError("reused T0 image-only OOF coverage drifted")
    canonical = pd.DataFrame({"patient_id": clinical.patient_ids, "ground_truth_pCR": clinical.pcr})
    merged = canonical.merge(frame, on=["patient_id", "ground_truth_pCR"], how="inner", validate="one_to_one")
    expected_fold = {pid: fold for fold in FOLDS for pid in clinical.patient_ids[folds.roles(fold, clinical.patient_ids) == "test"]}
    if any(int(row.fold) != expected_fold[row.patient_id] for row in merged.itertuples()):
        raise RuntimeError("reused image-only fold assignment drifted")
    return merged.loc[:, ["patient_id", "fold", "ground_truth_pCR", "predicted_probability", "threshold"]].rename(columns={"predicted_probability": "probability"})


def make_dataset_class():
    torch, _, _, Dataset, _, _ = torch_imports()

    class CachedDataset(Dataset):
        def __init__(self, table: pd.DataFrame, *, mode: str, augment: bool, shuffled_paths: list[str] | None = None, zero_image: bool = False):
            self.table = table.reset_index(drop=True)
            self.mode, self.augment = mode, augment
            self.shuffled_paths, self.zero_image = shuffled_paths, zero_image

        def __len__(self) -> int:
            return len(self.table)

        def __getitem__(self, index: int):
            row = self.table.iloc[index]
            if self.zero_image:
                image = np.zeros((3, *SHAPE), dtype=np.float32)
            else:
                patient_id = self.shuffled_paths[index] if self.shuffled_paths is not None else str(row.patient_id)
                with np.load(paths()["cache"] / f"{patient_id}_T0_static_dce.npz", allow_pickle=False) as archive:
                    image = archive["phases"].astype(np.float32)
                if image.shape != (3, *SHAPE):
                    raise RuntimeError("reused cache image shape drifted")
            if self.augment:
                if random.random() < 0.5:
                    image = image[:, :, :, ::-1].copy()
                if random.random() < 0.5:
                    image = image[:, :, ::-1, :].copy()
            scalar = float(row.clinical_logit)
            features = np.asarray(json.loads(row.clinical_features), dtype=np.float32)
            return torch.from_numpy(image), torch.tensor(float(row.ground_truth_pCR)), torch.tensor(scalar), torch.from_numpy(features), str(row.patient_id)

    return CachedDataset


def build_models(clinical_dim: int):
    torch, nn, _, _, Weights, r3d_18 = torch_imports()

    class ResidualExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = r3d_18(weights=Weights.KINETICS400_V1)
            width = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            self.residual_head = nn.Linear(width, 1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

        def forward(self, image, clinical_logit, clinical_features):
            correction = self.residual_head(self.encoder(image)).squeeze(1)
            return clinical_logit + correction, correction

    class NaiveFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = r3d_18(weights=Weights.KINETICS400_V1)
            width = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            self.fusion_head = nn.Linear(width + clinical_dim, 1)

        def forward(self, image, clinical_logit, clinical_features):
            logit = self.fusion_head(torch.cat((self.encoder(image), clinical_features), dim=1)).squeeze(1)
            return logit, torch.zeros_like(logit)

    return ResidualExpert, NaiveFusion


@dataclass
class ImageRun:
    name: str
    test: pd.DataFrame
    val_threshold: float
    history: list[dict[str, float]]


def run_image_arm(*, name: str, fold: int, train_table: pd.DataFrame, val_table: pd.DataFrame, test_table: pd.DataFrame, device: str) -> ImageRun:
    if name not in {"naive_clinical_image", "clinical_image_residual"}:
        raise ValueError("unknown image arm")
    torch, nn, DataLoader, _, _, _ = torch_imports()
    set_seed(SEED + fold + (100 if name == "naive_clinical_image" else 0))
    Dataset = make_dataset_class()
    train_loader = DataLoader(Dataset(train_table, mode=name, augment=True), batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(Dataset(val_table, mode=name, augment=False), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(Dataset(test_table, mode=name, augment=False), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    clinical_dim = len(json.loads(train_table.clinical_features.iloc[0]))
    ResidualExpert, NaiveFusion = build_models(clinical_dim)
    model = (ResidualExpert() if name == "clinical_image_residual" else NaiveFusion()).to(device)
    n_pos = int(train_table.ground_truth_pCR.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train_table) - n_pos) / n_pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    def infer(loader, training: bool = False, *, zero: bool = False, shuffled: list[str] | None = None):
        if zero or shuffled is not None:
            source = loader.dataset.table
            loader = DataLoader(Dataset(source, mode=name, augment=False, shuffled_paths=shuffled, zero_image=zero), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        model.train(training)
        ys: list[float] = []; logits: list[float] = []; corrections: list[float] = []; ids: list[str] = []; loss_sum = 0.0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for image, target, clinical_logit, features, patient_ids in loader:
                image = image.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
                clinical_logit = clinical_logit.to(device, non_blocking=True); features = features.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    final_logit, correction = model(image, clinical_logit, features)
                    loss = criterion(final_logit, target)
                    if name == "clinical_image_residual":
                        loss = loss + 1e-3 * correction.square().mean()
                if training:
                    loss.backward(); optimizer.step()
                loss_sum += float(loss.detach()) * len(target)
                ys.extend(target.detach().cpu().numpy().tolist()); logits.extend(final_logit.detach().float().cpu().numpy().tolist())
                corrections.extend(correction.detach().float().cpu().numpy().tolist()); ids.extend(patient_ids)
        return np.asarray(ys, dtype=int), np.asarray(logits, dtype=float), np.asarray(corrections, dtype=float), ids, loss_sum / len(loader.dataset)

    checkpoint_dir = ROOT / "checkpoints" / "formal" / name; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / "formal" / name; log_dir.mkdir(parents=True, exist_ok=True)
    best: tuple[tuple[float, float], int, dict[str, Any]] | None = None
    patience = 0; history: list[dict[str, float]] = []
    for epoch in range(1, 21):
        _, _, _, _, train_loss = infer(train_loader, True)
        y_val, val_logit, _, _, val_loss = infer(val_loader)
        val_probability = sigmoid(val_logit)
        score = (float(roc_auc_score(y_val, val_probability)), float(average_precision_score(y_val, val_probability)))
        scheduler.step(score[0])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auroc": score[0], "val_auprc": score[1], "lr": float(optimizer.param_groups[0]["lr"])})
        if best is None or score > best[0]:
            best = (score, epoch, {key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    if best is None:
        raise RuntimeError("image training failed to produce a checkpoint")
    torch.save({"state_dict": best[2], "fold": fold, "arm": name, "epoch": best[1], "validation_auroc": best[0][0], "validation_auprc": best[0][1], "residual_head_zero_initialized": name == "clinical_image_residual"}, checkpoint_dir / f"fold_{fold}.pt")
    pd.DataFrame(history).to_csv(log_dir / f"fold_{fold}.csv", index=False)
    model.load_state_dict(best[2]); model.eval()
    y_val, val_logit, _, _, _ = infer(val_loader)
    threshold = youden(y_val, sigmoid(val_logit))
    y_test, test_logit, correction, ids, _ = infer(test_loader)
    result = test_table.loc[:, ["patient_id", "fold", "ground_truth_pCR", "clinical_logit"]].copy()
    if ids != result.patient_id.tolist() or not np.array_equal(y_test, result.ground_truth_pCR.to_numpy()):
        raise RuntimeError("test inference ordering drifted")
    result["final_logit"] = test_logit; result["probability"] = sigmoid(test_logit)
    result["image_residual_logit"] = correction; result["threshold"] = threshold
    if name == "clinical_image_residual":
        _, zero_logit, zero_correction, _, _ = infer(test_loader, zero=True)
        shuffle_ids = result.patient_id.to_numpy().copy()
        rng = np.random.default_rng(SEED + fold)
        rng.shuffle(shuffle_ids)
        _, shuffle_logit, shuffle_correction, _, _ = infer(test_loader, shuffled=shuffle_ids.tolist())
        result["zero_final_logit"] = zero_logit; result["zero_image_residual_logit"] = zero_correction
        result["shuffled_final_logit"] = shuffle_logit; result["shuffled_image_residual_logit"] = shuffle_correction
    return ImageRun(name, result, threshold, history)


def calibration_fit(oof_logit: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=20_000, tol=1e-9, random_state=seed)
    model.fit(oof_logit.reshape(-1, 1), y)
    return model


def aggregate(frame: pd.DataFrame, arm: str, probability_column: str = "probability") -> tuple[dict[str, float], list[dict[str, Any]]]:
    result = metric_dict(frame.ground_truth_pCR.to_numpy(), frame[probability_column].to_numpy(), frame.threshold.to_numpy())
    per_fold = []
    for fold, group in frame.groupby("fold", sort=True):
        row = {"model": arm, "fold": int(fold), **metric_dict(group.ground_truth_pCR.to_numpy(), group[probability_column].to_numpy(), group.threshold.to_numpy())}
        per_fold.append(row)
    return result, per_fold


def bootstrap(rows: dict[str, pd.DataFrame]) -> dict[str, Any]:
    archive = np.load(paths()["static_bootstrap"], allow_pickle=False)
    indices = archive["indices"]
    ids = archive["patient_ids"].astype(str)
    if indices.shape != (5000, 808) or len(ids) != 808 or len(set(ids.tolist())) != 808:
        raise RuntimeError("reused MAMA bootstrap index contract drifted")
    ordered: dict[str, pd.DataFrame] = {}
    for arm, frame in rows.items():
        indexed = frame.set_index("patient_id")
        if set(indexed.index) != set(ids):
            raise RuntimeError(f"bootstrap patient coverage drifted for {arm}")
        ordered[arm] = indexed.loc[ids]
    y = ordered["clinical_only"].ground_truth_pCR.to_numpy(dtype=int)
    payload: dict[str, Any] = {"bootstrap_replicates": 5000, "seed": SEED, "baseline": "clinical_only", "comparisons": {}}
    base = ordered["clinical_only"]
    for arm, frame in ordered.items():
        if arm == "clinical_only":
            continue
        auc_delta: list[float] = []; ap_delta: list[float] = []
        for sample in indices:
            yy = y[sample]
            if len(np.unique(yy)) < 2:
                continue
            auc_delta.append(float(roc_auc_score(yy, frame.probability.to_numpy()[sample]) - roc_auc_score(yy, base.probability.to_numpy()[sample])))
            ap_delta.append(float(average_precision_score(yy, frame.probability.to_numpy()[sample]) - average_precision_score(yy, base.probability.to_numpy()[sample])))
        payload["comparisons"][arm] = {
            "delta_auroc": float(roc_auc_score(y, frame.probability) - roc_auc_score(y, base.probability)),
            "delta_auprc": float(average_precision_score(y, frame.probability) - average_precision_score(y, base.probability)),
            "delta_auroc_ci95": [float(value) for value in np.percentile(auc_delta, [2.5, 97.5])],
            "delta_auprc_ci95": [float(value) for value in np.percentile(ap_delta, [2.5, 97.5])],
            "valid_bootstrap_replicates": len(auc_delta),
        }
    return payload


def correction_analysis(residual: pd.DataFrame) -> dict[str, Any]:
    probability = residual.clinical_probability.to_numpy()
    truth = residual.ground_truth_pCR.to_numpy(dtype=int)
    predicted = residual.clinical_predicted_label.to_numpy(dtype=int)
    groups = {
        "clinical_confident_correct": (np.maximum(probability, 1.0 - probability) >= 0.7) & (predicted == truth),
        "clinical_uncertain": (probability > 0.3) & (probability < 0.7),
        "clinical_confident_wrong": (np.maximum(probability, 1.0 - probability) >= 0.7) & (predicted != truth),
    }
    output: dict[str, Any] = {}
    before = np.abs(probability - truth)
    after = np.abs(residual.probability.to_numpy() - truth)
    for name, chosen in groups.items():
        output[name] = {
            "n": int(chosen.sum()),
            "mean_abs_image_residual_logit": float(np.abs(residual.image_residual_logit.to_numpy()[chosen]).mean()) if chosen.any() else math.nan,
            "beneficial_corrections": int(np.count_nonzero(after[chosen] < before[chosen])),
            "worsened_corrections": int(np.count_nonzero(after[chosen] > before[chosen])),
            "unchanged_corrections": int(np.count_nonzero(np.isclose(after[chosen], before[chosen]))),
        }
    return output


def write_report(summary: dict[str, Any], paired: dict[str, Any], reliance: dict[str, Any], correction: dict[str, Any], clinical_audit: dict[str, Any]) -> None:
    rows = []
    labels = {"clinical_only": "Clinical-only", "t0_image_only": "T0 image-only（既有）", "clinical_recalibration": "Clinical recalibration", "naive_clinical_image": "Naive Clinical + Image", "clinical_image_residual": "Clinical + Image Residual"}
    for arm in ARMS:
        m = summary[arm]
        delta = paired["comparisons"].get(arm, {})
        delta_auc = "–" if arm == "clinical_only" else f"{delta['delta_auroc']:+.4f}"
        delta_ap = "–" if arm == "clinical_only" else f"{delta['delta_auprc']:+.4f}"
        rows.append(f"| {labels[arm]} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['balanced_accuracy']:.4f} | {delta_auc} | {delta_ap} |")
    residual = paired["comparisons"]["clinical_image_residual"]
    recal = paired["comparisons"]["clinical_recalibration"]
    real = reliance["real_image"]
    zero = reliance["zero_image"]
    shuffled = reliance["shuffled_image"]
    valid = real["auroc"] > zero["auroc"] and real["auroc"] > shuffled["auroc"]
    support = residual["delta_auroc"] > 0 and residual["delta_auprc"] > 0 and residual["delta_auroc"] > recal["delta_auroc"] and valid
    conclusion = "SUCCESS" if support and residual["delta_auroc_ci95"][0] > 0 else ("PARTIAL SUPPORT" if support else "NO COMPLEMENTARY T0 SIGNAL")
    text = f"""# Clinical Residual Image Expert（T0 DCE）

## 1. 一句话结论

本次严格外层 OOF 实验的判定为 **{conclusion}**：T0 MRI 在已使用 clinical 信息后{'显示出' if conclusion != 'NO COMPLEMENTARY T0 SIGNAL' else '没有显示出'}可信的患者特异性 pCR 增量；解释必须同时参考重校准和影像依赖审计，不能只看 AUROC 的单点差异。

## 2. 实验动机

Clinical-only 已是强预测器，普通拼接融合可能被 clinical dominance 主导。主模型固定 clinical 预 sigmoid logit，仅令 Kinetics 预训练 R3D-18 从冻结的 T0 三期 DCE 中学习校正项：`l = l_c + Δl_I`。残差头权重与 bias 均以零初始化，故第 0 步严格等于 clinical-only。

## 3. 数据与 split

808 名患者、275 名 pCR，使用锁定的 seed-2026 五折 patient-level outer CV（manifest SHA-256: `{EXPECTED_FOLD_SHA}`）。T0 输入直接读取既有 MAMA-MIA 缓存：pre/post1/post2、32×96×96、FTV bbox+25% 和冻结的 per-patient T0-pre 归一化；没有读取 T1/T2/T3 或重新做影像预处理。

## 4. Clinical baseline

特征为 HR、HER2、MammaPrint、年龄和 exact treatment arm。年龄只用 outer-train 均值填补，arm 只以 outer-train support one-hot，随后 StandardScaler 与 balanced liblinear Logistic Regression；C/penalty 仅由 validation AUROC→AUPRC 选择。复现实验 AUROC={clinical_audit['observed_auroc']:.6f}、AUPRC={clinical_audit['observed_auprc']:.6f}，相对锁定值的 AUROC 差={clinical_audit['auroc_difference_from_locked']:+.6f}。

## 5. Image residual architecture

`l = l_c + Δl_I`，训练为 `BCEWithLogitsLoss(l, y) + 1e-3·mean(Δl_I²)`。clinical 分支不在影像训练中更新。每个 outer train 患者的 `l_c` 来自五折 inner cross-fitting，因此并非 clinical in-sample prediction；validation/test 只使用 outer-train 拟合的 clinical model。

## 6. 主要结果

| Model | AUROC | AUPRC | Balanced Acc. | ΔAUROC vs Clinical | ΔAUPRC vs Clinical |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 7. Paired bootstrap

所有比较共享 MAMA-MIA 的同一组 patient-level seed-2026、5,000 次 bootstrap indices。Residual vs Clinical：ΔAUROC={residual['delta_auroc']:+.6f}，95% CI [{residual['delta_auroc_ci95'][0]:+.6f}, {residual['delta_auroc_ci95'][1]:+.6f}]；ΔAUPRC={residual['delta_auprc']:+.6f}，95% CI [{residual['delta_auprc_ci95'][0]:+.6f}, {residual['delta_auprc_ci95'][1]:+.6f}]。Naive fusion 的 ΔAUROC={paired['comparisons']['naive_clinical_image']['delta_auroc']:+.6f}。

## 8. Recalibration control

不读取 MRI 的 logistic recalibration（`l'=a·l_c+b`）ΔAUROC={recal['delta_auroc']:+.6f}。Residual 的 ΔAUROC={residual['delta_auroc']:+.6f}，{'未超过' if residual['delta_auroc'] <= recal['delta_auroc'] else '超过'}该 control；必须与上述 paired CI 一起解释。

## 9. Image reliance audit

Residual 模型 real-image AUROC={real['auroc']:.4f}；zero-image={zero['auroc']:.4f}；固定 patient-shuffled image={shuffled['auroc']:.4f}。real-image {'高于两项反事实' if valid else '未同时高于两项反事实'}。

## 10. Correction analysis

clinical confident-but-wrong 组 n={correction['clinical_confident_wrong']['n']}，平均 |Δl_I|={correction['clinical_confident_wrong']['mean_abs_image_residual_logit']:.4f}，有益/恶化校正={correction['clinical_confident_wrong']['beneficial_corrections']}/{correction['clinical_confident_wrong']['worsened_corrections']}。该分析是 post-hoc，不用于调参。

## 11. 结论

该诊断实验只检验 `I(T0 MRI; pCR | Clinical)>0?`。结论：**{conclusion}**。若未同时优于 clinical recalibration 且不能通过 zero/shuffled image 审计，则不应把单点性能变化称为 MRI 的补充信息。
"""
    destination = ROOT / "reports" / "clinical_residual_image_expert_zh.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def formal(device: str) -> None:
    validate_static_source()
    source = paths()
    folds = load_fold_manifest(source["folds"])
    clinical = load_clinical_labels(source["clinical"], expected_patient_ids=folds.patient_ids)
    split_rows = verify_split(folds, clinical)
    (ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(ROOT / "metrics" / "split_summary.csv", index=False)
    image_only = load_image_only(clinical, folds)
    clinical_rows: list[pd.DataFrame] = []; recal_rows: list[pd.DataFrame] = []; residual_rows: list[pd.DataFrame] = []; fusion_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        roles = folds.roles(fold, clinical.patient_ids)
        train, val, test = (np.flatnonzero(roles == role) for role in ("train", "val", "test"))
        outer = select_outer_clinical(clinical, train, val, fold)
        all_logits = outer.logits(clinical)
        oof_train = inner_cross_fitted_logits(clinical, train, outer, fold)
        clinical_test_probability = sigmoid(all_logits[test])
        clinical_frame = pd.DataFrame({"patient_id": clinical.patient_ids[test], "fold": fold, "ground_truth_pCR": clinical.pcr[test], "clinical_logit": all_logits[test], "probability": clinical_test_probability, "threshold": outer.threshold})
        clinical_rows.append(clinical_frame)
        calibration = calibration_fit(oof_train, clinical.pcr[train], SEED + fold)
        val_cal_prob = calibration.predict_proba(all_logits[val].reshape(-1, 1))[:, 1]
        cal_threshold = youden(clinical.pcr[val], val_cal_prob)
        recal_rows.append(pd.DataFrame({"patient_id": clinical.patient_ids[test], "fold": fold, "ground_truth_pCR": clinical.pcr[test], "clinical_logit": all_logits[test], "probability": calibration.predict_proba(all_logits[test].reshape(-1, 1))[:, 1], "threshold": cal_threshold}))
        feature_matrix = outer.scaler.transform(outer.encoder.transform(clinical)).astype(np.float32)
        def table(indices: np.ndarray, logits: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame({"patient_id": clinical.patient_ids[indices], "fold": fold, "ground_truth_pCR": clinical.pcr[indices], "clinical_logit": logits, "clinical_features": [json.dumps(row.tolist(), separators=(",", ":")) for row in feature_matrix[indices]]})
        train_table = table(train, oof_train)
        val_table = table(val, all_logits[val])
        test_table = table(test, all_logits[test])
        residual = run_image_arm(name="clinical_image_residual", fold=fold, train_table=train_table, val_table=val_table, test_table=test_table, device=device)
        fusion = run_image_arm(name="naive_clinical_image", fold=fold, train_table=train_table, val_table=val_table, test_table=test_table, device=device)
        residual_rows.append(residual.test); fusion_rows.append(fusion.test)
        audit_rows.append({"fold": fold, "n_train": len(train), "n_val": len(val), "n_test": len(test), "clinical_selected_penalty": outer.penalty, "clinical_selected_C": outer.c_value, "clinical_threshold": outer.threshold, "inner_crossfit_folds": 5, "inner_train_logit_is_out_of_fit": True, "test_used_for_clinical_selection": False, "test_used_for_image_checkpoint_selection": False, "test_used_for_threshold_selection": False, "residual_zero_head_initialization": True})
    clinical_oof = pd.concat(clinical_rows, ignore_index=True)
    recal_oof = pd.concat(recal_rows, ignore_index=True)
    residual_oof = pd.concat(residual_rows, ignore_index=True)
    fusion_oof = pd.concat(fusion_rows, ignore_index=True)
    for frame, name in ((clinical_oof, "clinical"), (recal_oof, "recal"), (residual_oof, "residual"), (fusion_oof, "fusion")):
        if len(frame) != 808 or frame.patient_id.duplicated().any() or set(frame.patient_id) != set(clinical.patient_ids):
            raise RuntimeError(f"{name} OOF coverage drifted")
    image_only = image_only.rename(columns={"probability": "probability"})
    arm_frames = {"clinical_only": clinical_oof, "t0_image_only": image_only, "clinical_recalibration": recal_oof, "naive_clinical_image": fusion_oof, "clinical_image_residual": residual_oof}
    summary: dict[str, Any] = {}; per_fold: list[dict[str, Any]] = []
    for arm, frame in arm_frames.items():
        summary[arm], rows = aggregate(frame, arm)
        per_fold.extend(rows)
    paired = bootstrap(arm_frames)
    zero = residual_oof.copy(); zero["probability"] = sigmoid(zero.zero_final_logit.to_numpy())
    shuffled = residual_oof.copy(); shuffled["probability"] = sigmoid(shuffled.shuffled_final_logit.to_numpy())
    reliance = {"real_image": summary["clinical_image_residual"], "zero_image": aggregate(zero, "zero_image")[0], "shuffled_image": aggregate(shuffled, "shuffled_image")[0], "shuffle_rule": "per_outer_test_fold_fixed_rng_seed_2026_plus_fold"}
    residual_oof["clinical_probability"] = sigmoid(residual_oof.clinical_logit.to_numpy())
    clinical_threshold_map = clinical_oof.set_index("patient_id").threshold
    residual_oof["clinical_threshold"] = residual_oof.patient_id.map(clinical_threshold_map)
    residual_oof["clinical_predicted_label"] = (residual_oof.clinical_probability >= residual_oof.clinical_threshold).astype(int)
    residual_oof["clinical_correct"] = residual_oof.clinical_predicted_label.eq(residual_oof.ground_truth_pCR)
    residual_oof["clinical_confidence"] = np.maximum(residual_oof.clinical_probability, 1.0 - residual_oof.clinical_probability)
    correction = correction_analysis(residual_oof)
    # The locked values are fixed by the public foundation metrics table; keep
    # literal values here to audit reproduction without retaining IDs.
    clinical_audit = {"locked_auroc": 0.709155722326454, "locked_auprc": 0.5575141168308745, "observed_auroc": summary["clinical_only"]["auroc"], "observed_auprc": summary["clinical_only"]["auprc"], "auroc_difference_from_locked": summary["clinical_only"]["auroc"] - 0.709155722326454, "auprc_difference_from_locked": summary["clinical_only"]["auprc"] - 0.5575141168308745, "pass_abs_auroc_difference_le_0p01": abs(summary["clinical_only"]["auroc"] - 0.709155722326454) <= 0.01}
    if not clinical_audit["pass_abs_auroc_difference_le_0p01"]:
        raise RuntimeError("clinical-only reproduction differs from locked AUROC by >0.01")
    # Patient-bearing primary artifact stays ignored; it contains every required
    # correction field, plus explicit counterfactual columns for audit.
    clinical_private = clinical_oof.rename(columns={"probability": "clinical_probability"}).copy(); clinical_private["arm"] = "clinical_only"
    recal_private = recal_oof.copy(); recal_private["arm"] = "clinical_recalibration"
    fusion_private = fusion_oof.copy(); fusion_private["arm"] = "naive_clinical_image"
    residual_private = residual_oof.copy(); residual_private["arm"] = "clinical_image_residual"; residual_private["final_probability"] = residual_private.pop("probability")
    private = pd.concat([clinical_private, recal_private, fusion_private, residual_private], ignore_index=True, sort=False)
    private_dir = ROOT / "predictions" / "formal"; private_dir.mkdir(parents=True, exist_ok=True)
    private.to_csv(private_dir / "oof_predictions.private.csv", index=False)
    pd.DataFrame(per_fold).to_csv(ROOT / "metrics" / "per_fold_metrics.csv", index=False)
    (ROOT / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "metrics" / "paired_bootstrap.json").write_text(json.dumps(paired, indent=2) + "\n")
    (ROOT / "metrics" / "image_reliance_audit.json").write_text(json.dumps(reliance, indent=2) + "\n")
    (ROOT / "metrics" / "correction_analysis.json").write_text(json.dumps(correction, indent=2) + "\n")
    (ROOT / "metrics" / "clinical_baseline_audit.json").write_text(json.dumps(clinical_audit, indent=2) + "\n")
    (ROOT / "metrics" / "leakage_audit.json").write_text(json.dumps({"status": "PASS", "fold_manifest_sha256": folds.sha256, "outer_patient_intersections_empty": True, "clinical_train_logits": "five_fold_inner_cross_fitted", "clinical_test_fit_population": "outer_train_only", "test_used_for_any_selection": False, "T0_cache_directly_reused": True, "only_T0_images_loaded": True}, indent=2) + "\n")
    (ROOT / "metrics" / "run_provenance.json").write_text(json.dumps({"static_source_formal_validation": "PASS", "static_source_summary_sha256": sha256(STATIC_ROOT / "metrics" / "summary.json"), "static_bootstrap_reused": True, "fold_manifest_sha256": folds.sha256, "clinical_labels_sha256": clinical.sha256}, indent=2) + "\n")
    write_report(summary, paired, reliance, correction, clinical_audit)
    validate()


def validate() -> None:
    required = [ROOT / "configs" / "formal.yaml", ROOT / "metrics" / "summary.json", ROOT / "metrics" / "paired_bootstrap.json", ROOT / "metrics" / "image_reliance_audit.json", ROOT / "metrics" / "correction_analysis.json", ROOT / "metrics" / "clinical_baseline_audit.json", ROOT / "metrics" / "leakage_audit.json", ROOT / "metrics" / "split_summary.csv", ROOT / "reports" / "clinical_residual_image_expert_zh.md"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing formal artifacts: {missing}")
    summary = json.loads((ROOT / "metrics" / "summary.json").read_text())
    paired = json.loads((ROOT / "metrics" / "paired_bootstrap.json").read_text())
    if set(summary) != set(ARMS) or paired.get("bootstrap_replicates") != 5000:
        raise RuntimeError("aggregate artifact schema drift")
    private = ROOT / "predictions" / "formal" / "oof_predictions.private.csv"
    checkpoints = list((ROOT / "checkpoints" / "formal").glob("*/*.pt"))
    logs = list((ROOT / "logs" / "formal").glob("*/*.csv"))
    if not private.exists() or len(checkpoints) != 10 or len(logs) != 10:
        raise RuntimeError("formal private artifact coverage failure")
    if subprocess.run(["git", "check-ignore", "--quiet", str(private)], cwd=REPO, check=False).returncode != 0:
        raise RuntimeError("patient-level output is not ignored")
    output = {"status": "PASS", "patients": 808, "pcr_positive": 275, "models": list(ARMS), "formal_checkpoints": len(checkpoints), "training_logs": len(logs), "bootstrap_replicates": 5000, "patient_level_predictions_gitignored": True, "only_T0_reused_cache": True}
    (ROOT / "metrics" / "formal_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("formal"); run.add_argument("--device", default="cuda:0")
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "formal":
        formal(args.device)
    else:
        validate()


if __name__ == "__main__":
    main()
