#!/usr/bin/env python3
"""Auditable T0-only lesion-centred static DCE pCR baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPO = HERE.parents[3]
EXPECTED_FOLD_SHA = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
SEED = 2026
SHAPE = (32, 96, 96)
VARIANTS = ("subtraction", "three_phase")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data_root() -> Path:
    value = os.environ.get("ISPY2_PREPROCESSED_ROOT")
    if not value:
        raise RuntimeError("ISPY2_PREPROCESSED_ROOT is required")
    return Path(value).expanduser().resolve(strict=True)


def paths() -> dict[str, Path]:
    root = data_root()
    return {
        "data": root,
        "folds": root / "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026" / "matched_patient_cv_splits_seed2026.csv",
        "clinical": root / "clinical_labels_complete4visits.csv",
        "phase": REPO / "ispy_jepa_tmi_clean" / "data_processing" / "metadata" / "BreastDCEDL_metadata_min_crop.csv",
        "cache": ROOT / "cache",
        "manifest": ROOT / "manifests" / "t0_phase_audit.private.csv",
    }


def load_folds() -> pd.DataFrame:
    source = paths()["folds"]
    if sha256(source) != EXPECTED_FOLD_SHA:
        raise RuntimeError("locked fold manifest SHA-256 mismatch")
    frame = pd.read_csv(source, dtype={"patient_id": str})
    if set(frame.columns) != {"patient_id", "fold", "split", "label_pcr"}:
        raise RuntimeError("fold schema drift")
    if len(frame) != 4040 or frame.patient_id.nunique() != 808:
        raise RuntimeError("fold population drift")
    for fold in range(5):
        current = frame[frame.fold.eq(fold)]
        groups = {s: set(current.loc[current.split.eq(s), "patient_id"]) for s in ("train", "val", "test")}
        if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError(f"patient leakage in fold {fold}")
        if set.union(*groups.values()) != set(frame.patient_id.unique()):
            raise RuntimeError(f"incomplete fold {fold}")
    return frame


def _t0_record(patient_id: str) -> dict:
    manifest_path = paths()["data"] / patient_id / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    visits = [v for v in manifest.get("visits", []) if v.get("visit") == "T0"]
    if len(visits) != 1:
        return {}
    return visits[0]


def _actual_t0_path(patient_id: str, suffix: str) -> Path:
    path = paths()["data"] / patient_id / "T0" / f"{patient_id}_T0_{suffix}"
    # This is the only image path constructor in the experiment. Fail closed.
    if any(token in str(path) for token in ("/T1/", "/T2/", "/T3/")):
        raise RuntimeError("longitudinal path rejected")
    return path


def _t0_dce_path(patient_id: str, record: dict | None = None) -> Path:
    """Resolve the manifest-selected T0 DCE, including repaired aligned files."""
    record = _t0_record(patient_id) if record is None else record
    declared = record.get("dce_nifti") if record else None
    name = Path(str(declared)).name if declared else f"{patient_id}_T0_original_DCE.nii"
    path = paths()["data"] / patient_id / "T0" / name
    if any(token in str(path) for token in ("/T1/", "/T2/", "/T3/")):
        raise RuntimeError("longitudinal path rejected")
    return path


def audit() -> None:
    p = paths()
    folds = load_folds()
    phase = pd.read_csv(p["phase"]).set_index("pid")
    label_map = folds.drop_duplicates("patient_id").set_index("patient_id").label_pcr
    rows = []
    for patient_id in sorted(label_map.index):
        rec = _t0_record(patient_id)
        dce = _t0_dce_path(patient_id, rec)
        roi = _actual_t0_path(patient_id, "ftv_mask.nii")
        bbox = rec.get("bbox_nii_xyz_inclusive") if rec else None
        n_times = int(rec.get("n_times", 0)) if rec else 0
        meta_pre = None
        if patient_id in phase.index:
            try:
                meta_pre = int(round(float(phase.loc[patient_id, "pre"])))
            except (TypeError, ValueError):
                pass
        pre = meta_pre
        post1 = None if pre is None else pre + 1
        post2 = None if pre is None else pre + 2
        complete = bool(dce.exists() and n_times >= 3 and pre is not None and pre >= 0 and post2 < n_times)
        roi_ok = bool(roi.exists() or bbox)
        rows.append({
            "patient_id": patient_id,
            "label_pcr": int(label_map[patient_id]),
            "dce_exists": dce.exists(),
            "roi_exists": roi_ok,
            "n_times": n_times,
            "pre_index": pre,
            "post1_index": post1,
            "post2_index": post2,
            "complete_three_phase": complete,
            "eligible": complete and roi_ok,
        })
    out = pd.DataFrame(rows)
    p["manifest"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p["manifest"], index=False)
    summary = {
        "total_patients": int(len(out)),
        "complete_pre_post1_post2": int(out.complete_three_phase.sum()),
        "missing_phase_patients": int((~out.complete_three_phase).sum()),
        "missing_pcr_labels": int(out.label_pcr.isna().sum()),
        "missing_roi": int((~out.roi_exists).sum()),
        "eligible_patients": int(out.eligible.sum()),
        "manifest_selected_repaired_dce": int(sum(_t0_dce_path(pid).name.endswith("_aligned.nii") for pid in out.patient_id)),
        "pcr_positive": int(out.loc[out.eligible, "label_pcr"].sum()),
        "pcr_prevalence": float(out.loc[out.eligible, "label_pcr"].mean()),
        "all_metadata_pre_indices_zero": bool(out.pre_index.dropna().eq(0).all()),
        "fold_manifest_sha256": sha256(p["folds"]),
        "longitudinal_paths_accessed": False,
    }
    (ROOT / "metrics").mkdir(exist_ok=True)
    (ROOT / "metrics" / "phase_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    split_rows = []
    for (fold, split), group in folds.groupby(["fold", "split"]):
        split_rows.append({"fold": fold, "split": split, "n": len(group), "pcr_positive": int(group.label_pcr.sum()), "pcr_prevalence": group.label_pcr.mean()})
    pd.DataFrame(split_rows).to_csv(ROOT / "metrics" / "split_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


def _bbox_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    pos = np.nonzero(mask > 0)
    if not pos[0].size:
        return None
    lo = np.asarray([v.min() for v in pos], dtype=int)
    hi = np.asarray([v.max() + 1 for v in pos], dtype=int)
    return lo, hi


def _expanded_crop(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray, order: int) -> np.ndarray:
    size = hi - lo
    margin = np.maximum(np.ceil(size * 0.25).astype(int), 2)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, np.asarray(arr.shape[:3]))
    cropped = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], ...]
    if 0 in cropped.shape[:3]:
        raise RuntimeError("empty lesion crop")
    factors = (SHAPE[2] / cropped.shape[0], SHAPE[1] / cropped.shape[1], SHAPE[0] / cropped.shape[2])
    if cropped.ndim == 4:
        factors = (*factors, 1.0)
    resized = zoom(cropped, factors, order=order, mode="nearest", prefilter=order > 1)
    if cropped.ndim == 4:
        return np.transpose(resized, (3, 2, 1, 0)).astype(np.float32)
    return np.transpose(resized, (2, 1, 0)).astype(np.float32)


def build_one(row: pd.Series, overwrite: bool = False) -> dict:
    patient_id = str(row.patient_id)
    out_path = paths()["cache"] / f"{patient_id}_T0_static_dce.npz"
    if out_path.exists() and not overwrite:
        return {"patient_id": patient_id, "status": "exists"}
    rec = _t0_record(patient_id)
    dce_path = _t0_dce_path(patient_id, rec)
    roi_path = _actual_t0_path(patient_id, "ftv_mask.nii")
    dce_img = nib.load(dce_path, mmap=True)
    indices = [int(row.pre_index), int(row.post1_index), int(row.post2_index)]
    # ArrayProxy deliberately avoids loading the other DCE frames.
    phases = np.stack(
        [np.asarray(dce_img.dataobj[..., index], dtype=np.float32) for index in indices],
        axis=-1,
    )
    if roi_path.exists():
        roi = np.asarray(nib.load(roi_path, mmap=True).dataobj, dtype=np.float32) > 0
        if roi.shape != phases.shape[:3]:
            raise RuntimeError(f"T0 DCE/ROI grid mismatch for {patient_id}")
        mask_bbox = _bbox_from_mask(roi)
    else:
        mask_bbox = None
        roi = np.zeros(phases.shape[:3], dtype=bool)
    # The preprocessing manifest records the released ROI bbox in the selected
    # DCE voxel grid, including the 15 repaired partial-z acquisitions.
    b = rec.get("bbox_nii_xyz_inclusive")
    bbox = (np.array([b["x_min"], b["y_min"], b["z_min"]]), np.array([b["x_max"] + 1, b["y_max"] + 1, b["z_max"] + 1])) if b else mask_bbox
    if bbox is None:
        raise RuntimeError(f"missing T0 ROI for {patient_id}")
    if np.any(bbox[0] < 0) or np.any(bbox[1] > np.asarray(phases.shape[:3])):
        raise RuntimeError(f"T0 ROI bbox lies outside selected DCE grid for {patient_id}")
    if not roi.any():
        roi[tuple(slice(int(a), int(b)) for a, b in zip(*bbox))] = True
    crop = _expanded_crop(phases, *bbox, order=1)
    mask = _expanded_crop(roi.astype(np.float32), *bbox, order=0) > 0.5
    pre_values = crop[0][np.isfinite(crop[0]) & (crop[0] > 0)]
    if pre_values.size < 128:
        raise RuntimeError(f"insufficient foreground for {patient_id}")
    low, high = np.percentile(pre_values, (1, 99))
    ref = np.clip(pre_values, low, high)
    median = float(np.median(ref))
    q1, q3 = np.percentile(ref, (25, 75))
    scale = float((q3 - q1) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(ref) + 1e-6)
    crop = np.clip((crop - median) / scale, -5, 5).astype(np.float32)
    paths()["cache"].mkdir(exist_ok=True)
    np.savez_compressed(out_path, phases=crop.astype(np.float16), mask=mask.astype(np.uint8), phase_indices=np.asarray(indices, dtype=np.int16), label_pcr=np.int8(row.label_pcr), normalization=np.asarray([median, scale], dtype=np.float32), source_name=np.asarray(dce_path.name))
    return {"patient_id": patient_id, "status": "built"}


def build_cache(overwrite: bool = False) -> None:
    audit_path = paths()["manifest"]
    if not audit_path.exists():
        audit()
    frame = pd.read_csv(audit_path, dtype={"patient_id": str})
    if not frame.eligible.all() or len(frame) != 808:
        raise RuntimeError("formal cohort must have 808 eligible T0 studies")
    start = time.time()
    statuses = []
    for i, row in frame.iterrows():
        statuses.append(build_one(row, overwrite=overwrite))
        if (i + 1) % 25 == 0:
            print(f"cached {i + 1}/{len(frame)} elapsed={time.time()-start:.1f}s", flush=True)
    pd.DataFrame(statuses).to_csv(ROOT / "manifests" / "cache_build_summary.private.csv", index=False)
    make_qc(frame)


def make_qc(frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    chosen = frame.iloc[rng.choice(len(frame), size=6, replace=False)]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(15, 3 * len(chosen)))
    for row_axis, (_, row) in zip(axes, chosen.iterrows()):
        z = np.load(paths()["cache"] / f"{row.patient_id}_T0_static_dce.npz")
        phases, mask = z["phases"].astype(np.float32), z["mask"].astype(bool)
        zi = int(np.argmax(mask.sum(axis=(1, 2))))
        images = [phases[0, zi], phases[1, zi], phases[1, zi] - phases[0, zi], phases[2, zi], phases[1, zi]]
        titles = ["pre", "post1", "post1-pre", "post2", "ROI overlay"]
        for ax, image, title in zip(row_axis, images, titles):
            ax.imshow(image, cmap="gray")
            if title == "ROI overlay":
                ax.contour(mask[zi], levels=[0.5], colors="r", linewidths=0.8)
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "t0_crop_qc.png", dpi=140)
    plt.close(fig)


def _torch_imports():
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models.video import r3d_18, R3D_18_Weights
    return torch, nn, Dataset, DataLoader, r3d_18, R3D_18_Weights


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "prevalence": float(np.mean(y)),
    }


def youden(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds)
    idx = np.argmax((tpr - fpr)[valid])
    return float(thresholds[valid][idx])


def train(variant: str, fold: int, device: str, shuffled_labels: bool = False, tag: str = "formal") -> None:
    if variant not in VARIANTS or fold not in range(5):
        raise ValueError("invalid variant/fold")
    torch, nn, Dataset, DataLoader, r3d_18, Weights = _torch_imports()
    random.seed(SEED + fold); np.random.seed(SEED + fold); torch.manual_seed(SEED + fold)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    folds = load_folds(); current = folds[folds.fold.eq(fold)].copy()
    cache_root = paths()["cache"]

    class StaticDataset(Dataset):
        def __init__(self, table: pd.DataFrame, augment: bool):
            self.table = table.reset_index(drop=True); self.augment = augment
        def __len__(self): return len(self.table)
        def __getitem__(self, index):
            row = self.table.iloc[index]
            z = np.load(cache_root / f"{row.patient_id}_T0_static_dce.npz")
            x = z["phases"].astype(np.float32)
            if variant == "subtraction":
                sub = x[1] - x[0]; x = np.stack([sub, sub, sub])
            if self.augment:
                if random.random() < 0.5: x = x[:, :, :, ::-1].copy()
                if random.random() < 0.5: x = x[:, :, ::-1, :].copy()
            return torch.from_numpy(x), torch.tensor(float(row.train_label)), str(row.patient_id)

    labels = dict(zip(current.patient_id, current.label_pcr))
    current["train_label"] = current.patient_id.map(labels).astype(int)
    if shuffled_labels:
        train_mask = current.split.eq("train")
        shuffled = current.loc[train_mask, "train_label"].to_numpy().copy()
        np.random.default_rng(SEED + fold).shuffle(shuffled)
        current.loc[train_mask, "train_label"] = shuffled
    tables = {s: current[current.split.eq(s)].copy() for s in ("train", "val", "test")}
    loaders = {s: DataLoader(StaticDataset(t, s == "train"), batch_size=16, shuffle=s == "train", num_workers=4, pin_memory=True, persistent_workers=s == "train") for s, t in tables.items()}
    try:
        model = r3d_18(weights=Weights.KINETICS400_V1)
    except Exception as exc:
        print(f"pretrained weights unavailable ({exc}); refusing unrecorded fallback", file=sys.stderr)
        raise
    model.fc = nn.Linear(model.fc.in_features, 1); model.to(device)
    n_pos = int(tables["train"].train_label.sum()); n_neg = len(tables["train"]) - n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    amp_dtype = torch.bfloat16

    def infer(loader, training=False):
        model.train(training); ys=[]; ps=[]; ids=[]; loss_sum=0.0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for x, y, patient_id in loader:
                x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
                if training: optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits=model(x).squeeze(1); loss=criterion(logits, y)
                if training:
                    loss.backward(); optimizer.step()
                loss_sum += float(loss.detach()) * len(y)
                ys.extend(y.detach().cpu().numpy()); ps.extend(torch.sigmoid(logits).detach().float().cpu().numpy()); ids.extend(patient_id)
        return np.asarray(ys, int), np.asarray(ps, float), ids, loss_sum / len(loader.dataset)

    out_dir = ROOT / "checkpoints" / tag / variant; out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / tag / variant; log_dir.mkdir(parents=True, exist_ok=True)
    best=None; patience=0; history=[]
    for epoch in range(1, 21):
        _, _, _, train_loss = infer(loaders["train"], True)
        yv, pv, _, val_loss = infer(loaders["val"])
        score=(roc_auc_score(yv,pv), average_precision_score(yv,pv))
        scheduler.step(score[0])
        history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"val_auroc":score[0],"val_auprc":score[1],"lr":optimizer.param_groups[0]["lr"]})
        print(variant, fold, epoch, history[-1], flush=True)
        if best is None or score > best[0]:
            best=(score,epoch,{k:v.detach().cpu() for k,v in model.state_dict().items()}); patience=0
            torch.save({"state_dict":best[2],"variant":variant,"fold":fold,"epoch":epoch,"validation_auroc":score[0],"validation_auprc":score[1],"shuffled_labels":shuffled_labels}, out_dir / f"fold_{fold}.pt")
        else: patience += 1
        if patience >= 5: break
    pd.DataFrame(history).to_csv(log_dir / f"fold_{fold}.csv", index=False)
    checkpoint=torch.load(out_dir / f"fold_{fold}.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["state_dict"])
    yv,pv,_,_=infer(loaders["val"]); threshold=youden(yv,pv)
    rows=[]
    for split in ("val","test"):
        y,pred,ids,_=infer(loaders[split])
        for pid,yy,pp in zip(ids,y,pred): rows.append({"patient_id":pid,"ground_truth_pCR":int(labels[pid]),"predicted_probability":pp,"fold":fold,"split":split,"model_variant":variant,"threshold":threshold})
    pred_dir=ROOT/"predictions"/tag; pred_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(pred_dir/f"{variant}_fold_{fold}.private.csv",index=False)


def _load_oof(tag="formal") -> pd.DataFrame:
    frames=[]
    for variant in VARIANTS:
        for fold in range(5):
            path=ROOT/"predictions"/tag/f"{variant}_fold_{fold}.private.csv"
            frames.append(pd.read_csv(path,dtype={"patient_id":str}))
    return pd.concat(frames,ignore_index=True)


def evaluate() -> None:
    pred=_load_oof(); test=pred[pred.split.eq("test")].copy()
    if test.groupby("model_variant").patient_id.nunique().to_dict() != {"subtraction":808,"three_phase":808}:
        raise RuntimeError("OOF patient coverage failure")
    metric_rows=[]
    for (variant,fold),g in pred.groupby(["model_variant","fold"]):
        gg=g[g.split.eq("test")]; m=metrics(gg.ground_truth_pCR.to_numpy(),gg.predicted_probability.to_numpy(),float(gg.threshold.iloc[0])); metric_rows.append({"model_variant":variant,"fold":fold,**m})
    pd.DataFrame(metric_rows).to_csv(ROOT/"metrics"/"per_fold_metrics.csv",index=False)
    summary={}
    for variant,g in test.groupby("model_variant"):
        # Each patient's threshold comes exclusively from that fold's validation set.
        y=g.ground_truth_pCR.to_numpy(); p=g.predicted_probability.to_numpy(); hard=p>=g.threshold.to_numpy()
        tn,fp,fn,tp=confusion_matrix(y,hard,labels=[0,1]).ravel()
        summary[variant]={"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,hard)),"sensitivity":float(tp/(tp+fn)),"specificity":float(tn/(tn+fp)),"f1":float(f1_score(y,hard)),"prevalence":float(y.mean())}
    wide={v:g.set_index("patient_id") for v,g in test.groupby("model_variant")}
    ids=sorted(set(wide[VARIANTS[0]].index)&set(wide[VARIANTS[1]].index)); y=wide["subtraction"].loc[ids].ground_truth_pCR.to_numpy(); pa=wide["subtraction"].loc[ids].predicted_probability.to_numpy(); pb=wide["three_phase"].loc[ids].predicted_probability.to_numpy()
    rng=np.random.default_rng(SEED); indices=rng.integers(0,len(ids),size=(5000,len(ids)),dtype=np.int32)
    np.savez_compressed(ROOT/"manifests"/"bootstrap_indices_seed2026.private.npz",patient_ids=np.asarray(ids),indices=indices)
    ta=wide["subtraction"].loc[ids].threshold.to_numpy()
    tb=wide["three_phase"].loc[ids].threshold.to_numpy()
    boots={v:{k:[] for k in ("auroc","auprc","balanced_accuracy")} for v in VARIANTS}; deltas={k:[] for k in ("auroc","auprc","balanced_accuracy")}
    for ix in indices:
        yy=y[ix]
        if len(np.unique(yy))<2: continue
        va={"auroc":roc_auc_score(yy,pa[ix]),"auprc":average_precision_score(yy,pa[ix]),"balanced_accuracy":balanced_accuracy_score(yy,pa[ix]>=ta[ix])}; vb={"auroc":roc_auc_score(yy,pb[ix]),"auprc":average_precision_score(yy,pb[ix]),"balanced_accuracy":balanced_accuracy_score(yy,pb[ix]>=tb[ix])}
        for k in va: boots["subtraction"][k].append(va[k]); boots["three_phase"][k].append(vb[k]); deltas[k].append(vb[k]-va[k])
    for v in VARIANTS:
        summary[v]["ci"]={k:[float(x) for x in np.percentile(boots[v][k],[2.5,97.5])] for k in boots[v]}
    summary["delta_dce"]={k:float(summary["three_phase"][k]-summary["subtraction"][k]) for k in ("auroc","auprc","balanced_accuracy")}
    summary["delta_dce"]["ci"]={k:[float(x) for x in np.percentile(deltas[k],[2.5,97.5])] for k in deltas}
    # Existing radiomics is only valid on its locked 375-patient complete-case
    # population. Report static-image scores on precisely that population.
    radiomics_source = paths()["data"] / "mri_nact_features_complete4visits_wide.csv"
    radiomics_ids = set(pd.read_csv(radiomics_source, usecols=["patient_id"], dtype={"patient_id": str}).patient_id)
    summary["radiomics_complete_case_375"] = {
        "existing_radiomics_only": {"auroc": 0.5637735849056604, "auprc": 0.3497278071074894, "balanced_accuracy": None},
        "note": "Existing locked pooled OOF result; balanced accuracy was not published by that protocol.",
    }
    for variant, g in test.groupby("model_variant"):
        gg = g[g.patient_id.isin(radiomics_ids)]
        if len(gg) != 375:
            raise RuntimeError(f"{variant} radiomics-paired population is not 375")
        y375=gg.ground_truth_pCR.to_numpy(); p375=gg.predicted_probability.to_numpy(); h375=p375>=gg.threshold.to_numpy()
        summary["radiomics_complete_case_375"][variant]={"n":375,"positive":int(y375.sum()),"auroc":float(roc_auc_score(y375,p375)),"auprc":float(average_precision_score(y375,p375)),"balanced_accuracy":float(balanced_accuracy_score(y375,h375))}
    (ROOT/"metrics"/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def zero_image_test(device: str) -> None:
    torch, nn, _, _, r3d_18, _ = _torch_imports(); folds=load_folds(); rows=[]
    for variant in VARIANTS:
        for fold in range(5):
            ck=torch.load(ROOT/"checkpoints"/"formal"/variant/f"fold_{fold}.pt",map_location=device,weights_only=False)
            model=r3d_18(weights=None); model.fc=nn.Linear(model.fc.in_features,1); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
            g=folds[(folds.fold.eq(fold))&(folds.split.eq("test"))]
            x=torch.zeros((len(g),3,*SHAPE),device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16): p=torch.sigmoid(model(x).squeeze(1)).float().cpu().numpy()
            rows.extend({"variant":variant,"fold":fold,"patient_id":pid,"label":int(y),"probability":float(pp)} for pid,y,pp in zip(g.patient_id,g.label_pcr,p))
    z=pd.DataFrame(rows); result={}
    for variant,g in z.groupby("variant"):
        result[variant]={"auroc":float(roc_auc_score(g.label,g.probability)),"auprc":float(average_precision_score(g.label,g.probability)),"prediction_std":float(g.probability.std())}
    (ROOT/"metrics"/"zero_image_test.json").write_text(json.dumps(result,indent=2)+"\n"); print(result)


def validate() -> None:
    """Fail closed on the complete formal artifact and leakage contract."""
    folds = load_folds()
    canonical = set(folds.patient_id.unique())
    cache_paths = sorted(paths()["cache"].glob("*_T0_static_dce.npz"))
    if len(cache_paths) != 808:
        raise RuntimeError("expected exactly 808 T0 cache files")
    cached_ids, repaired = set(), 0
    for path in cache_paths:
        patient_id = path.name.split("_T0_static_dce.npz")[0]
        if patient_id in cached_ids:
            raise RuntimeError("duplicate cache patient")
        cached_ids.add(patient_id)
        with np.load(path, allow_pickle=False) as z:
            required = {"phases", "mask", "phase_indices", "label_pcr", "normalization", "source_name"}
            if set(z.files) != required:
                raise RuntimeError(f"cache schema drift: {path.name}")
            image = z["phases"]
            if image.shape != (3, *SHAPE) or not np.isfinite(image).all():
                raise RuntimeError(f"invalid image cache: {path.name}")
            if z["mask"].shape != SHAPE or not np.any(z["mask"]):
                raise RuntimeError(f"invalid mask cache: {path.name}")
            if tuple(z["phase_indices"].tolist()) != (0, 1, 2):
                raise RuntimeError(f"phase contract drift: {path.name}")
            repaired += str(z["source_name"]) .endswith("_aligned.nii")
            source_name = str(z["source_name"])
            if "_T0_" not in source_name or any(token in source_name for token in ("_T1_", "_T2_", "_T3_")):
                raise RuntimeError(f"non-T0 cache provenance: {path.name}")
    if cached_ids != canonical or repaired != 15:
        raise RuntimeError("cache population/source provenance drift")
    predictions = _load_oof()
    formal_summary = json.loads((ROOT/"metrics/summary.json").read_text())
    for variant in VARIANTS:
        current = predictions[predictions.model_variant.eq(variant)]
        test = current[current.split.eq("test")]
        if len(test) != 808 or set(test.patient_id) != canonical or test.patient_id.duplicated().any():
            raise RuntimeError(f"{variant} OOF coverage failure")
        observed_auc = float(roc_auc_score(test.ground_truth_pCR, test.predicted_probability))
        observed_ap = float(average_precision_score(test.ground_truth_pCR, test.predicted_probability))
        if not np.isclose(observed_auc, formal_summary[variant]["auroc"]) or not np.isclose(observed_ap, formal_summary[variant]["auprc"]):
            raise RuntimeError(f"{variant} aggregate metric drift")
        truth = folds.drop_duplicates("patient_id").set_index("patient_id").label_pcr
        if not np.array_equal(test.set_index("patient_id").loc[truth.index].ground_truth_pCR, truth):
            raise RuntimeError(f"{variant} OOF label mismatch")
        for fold in range(5):
            subset = current[(current.fold.eq(fold)) & (current.split.eq("test"))]
            expected = set(folds[(folds.fold.eq(fold)) & (folds.split.eq("test"))].patient_id)
            if set(subset.patient_id) != expected or subset.threshold.nunique() != 1:
                raise RuntimeError(f"{variant}/fold-{fold} split or threshold drift")
    bootstrap = np.load(ROOT/"manifests"/"bootstrap_indices_seed2026.private.npz", allow_pickle=False)
    if bootstrap["indices"].shape != (5000, 808) or set(bootstrap["patient_ids"].astype(str)) != canonical:
        raise RuntimeError("bootstrap manifest drift")
    checkpoint_count = len(list((ROOT/"checkpoints/formal").glob("*/*.pt")))
    log_count = len(list((ROOT/"logs/formal").glob("*/*.csv")))
    if checkpoint_count != 10 or log_count != 10:
        raise RuntimeError("checkpoint/log coverage failure")
    required_public = [
        ROOT/"configs/formal.yaml", ROOT/"metrics/summary.json",
        ROOT/"metrics/per_fold_metrics.csv", ROOT/"metrics/phase_audit_summary.json",
        ROOT/"metrics/geometry_alignment_audit.json", ROOT/"metrics/leakage_audit.json",
        ROOT/"metrics/random_label_test.json", ROOT/"metrics/zero_image_test.json",
        ROOT/"figures/t0_crop_qc.png", ROOT/"reports/mama_mia_static_baseline_zh.md",
    ]
    missing = [str(path) for path in required_public if not path.exists()]
    if missing:
        raise RuntimeError(f"missing formal artifacts: {missing}")
    privacy_examples = [
        ROOT/"cache"/cache_paths[0].name,
        ROOT/"predictions/formal/subtraction_fold_0.private.csv",
        ROOT/"manifests/bootstrap_indices_seed2026.private.npz",
        ROOT/"checkpoints/formal/subtraction/fold_0.pt",
        ROOT/"logs/formal/subtraction/fold_0.csv",
    ]
    ignored = [
        subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)], cwd=REPO, check=False
        ).returncode == 0
        for path in privacy_examples
    ]
    if not all(ignored):
        raise RuntimeError("private artifact gitignore contract failure")
    result = {
        "status": "PASS", "patients": 808, "cache_files": 808,
        "manifest_selected_repaired_dce": repaired,
        "formal_oof_predictions_per_variant": 808,
        "checkpoints": checkpoint_count, "training_logs": log_count,
        "bootstrap_replicates": 5000,
        "fold_intersections_empty": True, "only_T0_images_loaded": True,
        "patient_level_artifacts_gitignored": True,
    }
    (ROOT/"metrics"/"formal_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("audit"); b=sub.add_parser("build-cache"); b.add_argument("--overwrite",action="store_true")
    t=sub.add_parser("train"); t.add_argument("--variant",choices=VARIANTS,required=True); t.add_argument("--fold",type=int,required=True); t.add_argument("--device",default="cuda:0"); t.add_argument("--shuffled-labels",action="store_true"); t.add_argument("--tag",default="formal")
    sub.add_parser("evaluate"); sub.add_parser("validate"); z=sub.add_parser("zero-image"); z.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.command=="audit": audit()
    elif args.command=="build-cache": build_cache(args.overwrite)
    elif args.command=="train": train(args.variant,args.fold,args.device,args.shuffled_labels,args.tag)
    elif args.command=="evaluate": evaluate()
    elif args.command=="validate": validate()
    elif args.command=="zero-image": zero_image_test(args.device)


if __name__ == "__main__": main()
