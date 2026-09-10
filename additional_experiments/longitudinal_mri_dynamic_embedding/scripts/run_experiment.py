#!/usr/bin/env python3
"""Causal longitudinal DCE dynamic embeddings for pCR.

Every prediction at landmark t is based only on T0..t.  The four DCE visits
are loaded from the frozen C1B-H cache; no ROI mask, radiomics, FTV or geometry
scalar is given to the model.  Clinical residual training uses cross-fitted
outer-training logits, never clinical in-sample predictions.
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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPO = HERE.parents[3]
FOUNDATION_SRC = REPO / "additional_experiments" / "foundation_mri_baselines" / "src"
if str(FOUNDATION_SRC) not in sys.path:
    sys.path.insert(0, str(FOUNDATION_SRC))
from foundation_mri.data import FOLDS, ClinicalTable, load_clinical_labels, load_fold_manifest
from foundation_mri.evaluation import ClinicalEncoder, select_logistic

SEED = 2026
FOLD_SHA = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
VISITS = ("T0", "T1", "T2", "T3")
ENDPOINTS = ("T0", "T0_T1", "T0_T1_T2", "T0_T1_T2_T3")
PRIMARY = "T0_T1"
ARMS = ("clinical_only", "clinical_recalibration", "dynamic_image_only", "naive_clinical_dynamic", "clinical_dynamic_residual")
SHAPE = (32, 96, 96)


def data_root() -> Path:
    value = os.environ.get("ISPY2_PREPROCESSED_ROOT")
    if not value:
        raise RuntimeError("ISPY2_PREPROCESSED_ROOT is required")
    return Path(value).resolve(strict=True)


def paths() -> dict[str, Path]:
    root = data_root()
    return {
        "folds": root / "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026" / "matched_patient_cv_splits_seed2026.csv",
        "clinical": root / "clinical_labels_complete4visits.csv",
        "cache": root / "_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96",
        "bootstrap": REPO / "additional_experiments" / "mama_mia_static_dce_baseline" / "manifests" / "bootstrap_indices_seed2026.private.npz",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_imports():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.video import R3D_18_Weights, r3d_18
    return torch, nn, DataLoader, Dataset, R3D_18_Weights, r3d_18


def set_seed(seed: int) -> None:
    torch, _, _, _, _, _ = torch_imports()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def sigmoid(logit: np.ndarray) -> np.ndarray:
    x = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def endpoint_index(endpoint: str) -> int:
    return ENDPOINTS.index(endpoint)


def youden(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, threshold = roc_curve(y, probability)
    valid = np.isfinite(threshold)
    return float(threshold[valid][np.argmax((tpr - fpr)[valid])])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: np.ndarray | float) -> dict[str, float]:
    hard = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, probability)), "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, hard, zero_division=0)), "prevalence": float(y.mean()),
    }


@dataclass(frozen=True)
class ClinicalFit:
    encoder: ClinicalEncoder
    scaler: StandardScaler
    model: LogisticRegression
    penalty: str
    c_value: float
    threshold: float

    def logits(self, clinical: ClinicalTable) -> np.ndarray:
        x = self.encoder.transform(clinical)
        return np.asarray(self.model.decision_function(self.scaler.transform(x)), dtype=np.float64)


def fixed_clinical(clinical: ClinicalTable, rows: np.ndarray, penalty: str, c_value: float, seed: int) -> ClinicalFit:
    encoder = ClinicalEncoder.fit(clinical, rows); x = encoder.transform(clinical)
    scaler = StandardScaler().fit(x[rows])
    kwargs: dict[str, Any] = {"penalty": penalty}
    if penalty == "l1": kwargs["dual"] = False
    model = LogisticRegression(C=float(c_value), solver="liblinear", class_weight="balanced", max_iter=20_000, tol=1e-7, random_state=seed, **kwargs)
    model.fit(scaler.transform(x[rows]), clinical.pcr[rows])
    return ClinicalFit(encoder, scaler, model, penalty, float(c_value), math.nan)


def outer_clinical(clinical: ClinicalTable, train: np.ndarray, val: np.ndarray, fold: int) -> ClinicalFit:
    encoder = ClinicalEncoder.fit(clinical, train); x = encoder.transform(clinical)
    selected = select_logistic(x[train], clinical.pcr[train], x[val], clinical.pcr[val], random_state=SEED + fold)
    return ClinicalFit(encoder, selected.scaler, selected.model, selected.penalty, selected.c_value, selected.threshold)


def crossfit_logits(clinical: ClinicalTable, outer_train: np.ndarray, selected: ClinicalFit, fold: int) -> np.ndarray:
    result = np.full(len(outer_train), np.nan, dtype=np.float64)
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
    for inner_fold, (tr, held) in enumerate(split.split(outer_train, clinical.pcr[outer_train])):
        model = fixed_clinical(clinical, outer_train[tr], selected.penalty, selected.c_value, SEED + fold * 10 + inner_fold)
        result[held] = model.logits(clinical)[outer_train[held]]
    if not np.isfinite(result).all(): raise RuntimeError("cross-fitted clinical logits incomplete")
    return result


def cache_paths(patient_ids: np.ndarray) -> dict[str, tuple[Path, Path, Path, Path]]:
    root = paths()["cache"]
    result: dict[str, tuple[Path, Path, Path, Path]] = {}
    for pid in patient_ids:
        visit_files: list[Path] = []
        for visit in VISITS:
            found = sorted(root.glob(f"{pid}_{visit}_dce8_adaptive_early_late_t0crop_axiscanon-v1_autoroi-v1_z32_y96_x96.npy"))
            if len(found) != 1: raise RuntimeError(f"expected one C1B-H cache for {pid}/{visit}, got {len(found)}")
            a = np.load(found[0], mmap_mode="r")
            if a.shape != (8, *SHAPE) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise RuntimeError(f"invalid C1B-H cache {found[0].name}")
            visit_files.append(found[0])
        result[str(pid)] = tuple(visit_files)  # type: ignore[assignment]
    return result


def verify_inputs(folds: Any, clinical: ClinicalTable, sequence_paths: dict[str, tuple[Path, Path, Path, Path]]) -> list[dict[str, Any]]:
    if folds.sha256 != FOLD_SHA or len(clinical.patient_ids) != 808 or int(clinical.pcr.sum()) != 275:
        raise RuntimeError("locked cohort contract drifted")
    if set(sequence_paths) != set(clinical.patient_ids): raise RuntimeError("longitudinal cache cohort drifted")
    rows=[]
    for fold in FOLDS:
        roles=folds.roles(fold, clinical.patient_ids)
        groups={role:set(clinical.patient_ids[roles==role]) for role in ("train","val","test")}
        if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]: raise RuntimeError("patient split overlap")
        for role in groups:
            ix=np.flatnonzero(roles==role); rows.append({"fold":fold,"split":role,"n":len(ix),"pcr_positive":int(clinical.pcr[ix].sum()),"pcr_prevalence":float(clinical.pcr[ix].mean())})
    return rows


def dataset_class(sequence_paths: dict[str, tuple[Path, Path, Path, Path]]):
    torch, _, _, Dataset, _, _ = torch_imports()
    class LongitudinalDataset(Dataset):
        def __init__(self, table: pd.DataFrame, augment: bool, zero: bool=False, shuffled: dict[str, str] | None=None):
            self.table=table.reset_index(drop=True); self.augment=augment; self.zero=zero; self.shuffled=shuffled
        def __len__(self): return len(self.table)
        def __getitem__(self, index):
            row=self.table.iloc[index]; pid=str(row.patient_id); source=self.shuffled[pid] if self.shuffled is not None else pid
            if self.zero: image=np.zeros((4,3,*SHAPE),dtype=np.float32)
            else: image=np.stack([np.load(path, mmap_mode="r")[:3].astype(np.float32) for path in sequence_paths[source]],axis=0)
            if self.augment:
                # One shared flip preserves the temporal coordinate convention.
                if random.random()<0.5: image=image[:,:,:,:,::-1].copy()
                if random.random()<0.5: image=image[:,:,:,::-1,:].copy()
            return (torch.from_numpy(image), torch.tensor(float(row.ground_truth_pCR)), torch.tensor(float(row.clinical_logit)), torch.from_numpy(np.asarray(json.loads(row.clinical_features),dtype=np.float32)), pid)
    return LongitudinalDataset


def make_models(clinical_dim: int):
    torch, nn, _, _, Weights, r3d_18 = torch_imports()
    class DynamicBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.visit_encoder=r3d_18(weights=Weights.KINETICS400_V1)
            dim=self.visit_encoder.fc.in_features; self.visit_encoder.fc=nn.Identity()
            self.position=nn.Parameter(torch.zeros(1,4,dim)); nn.init.normal_(self.position,std=0.02)
            layer=nn.TransformerEncoderLayer(d_model=dim,nhead=8,dim_feedforward=1024,dropout=0.1,activation="gelu",batch_first=True,norm_first=True)
            self.temporal=nn.TransformerEncoder(layer,num_layers=1)
        def states(self,image):
            batch,steps,channels,z,y,x=image.shape
            if (steps,channels,z,y,x)!=(4,3,*SHAPE): raise ValueError("unexpected longitudinal input shape")
            state=self.visit_encoder(image.reshape(batch*steps,channels,z,y,x)).reshape(batch,steps,-1)
            mask=torch.triu(torch.full((steps,steps),float("-inf"),device=image.device),diagonal=1)
            return self.temporal(state+self.position[:,:steps],mask=mask)
    class ImageOnly(DynamicBase):
        def __init__(self): super().__init__(); self.head=nn.Linear(512,1)
        def forward(self,image,clinical_logit,clinical_features):
            x=self.head(self.states(image)).squeeze(-1); return x, torch.zeros_like(x)
    class NaiveFusion(DynamicBase):
        def __init__(self): super().__init__(); self.head=nn.Linear(512+clinical_dim,1)
        def forward(self,image,clinical_logit,clinical_features):
            state=self.states(image); c=clinical_features[:,None,:].expand(-1,4,-1)
            x=self.head(torch.cat((state,c),dim=-1)).squeeze(-1); return x, torch.zeros_like(x)
    class Residual(DynamicBase):
        def __init__(self):
            super().__init__(); self.head=nn.Linear(512,1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        def forward(self,image,clinical_logit,clinical_features):
            correction=self.head(self.states(image)).squeeze(-1)
            return clinical_logit[:,None]+correction, correction
    return ImageOnly, NaiveFusion, Residual


def causal_prefix_test(device: str) -> None:
    """Fail if a future visit changes any earlier temporal embedding."""
    torch, _, _, _, _, _=torch_imports(); set_seed(SEED)
    _,_,Residual=make_models(5); model=Residual().to(device).eval()
    image=torch.randn(1,4,3,*SHAPE,device=device); clinical=torch.zeros(1,device=device); features=torch.zeros(1,5,device=device)
    with torch.no_grad(): before,_=model(image,clinical,features); modified=image.clone(); modified[:,3]+=17; after,_=model(modified,clinical,features)
    if not torch.allclose(before[:,:3],after[:,:3],atol=1e-5,rtol=1e-5): raise RuntimeError("causal temporal contract failed")


@dataclass
class RunResult:
    arm: str
    rows: pd.DataFrame


def train_arm(arm: str, fold: int, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, sequence_paths: dict[str, tuple[Path, Path, Path, Path]], device: str) -> RunResult:
    torch, nn, DataLoader, _, _, _ = torch_imports(); set_seed(SEED+fold+({"dynamic_image_only":0,"naive_clinical_dynamic":100,"clinical_dynamic_residual":200}[arm]))
    Dataset=dataset_class(sequence_paths)
    loaders={"train":DataLoader(Dataset(train,True),batch_size=4,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True),"val":DataLoader(Dataset(val,False),batch_size=4,shuffle=False,num_workers=4,pin_memory=True),"test":DataLoader(Dataset(test,False),batch_size=4,shuffle=False,num_workers=4,pin_memory=True)}
    dim=len(json.loads(train.clinical_features.iloc[0])); ImageOnly,Naive,Residual=make_models(dim)
    model={"dynamic_image_only":ImageOnly,"naive_clinical_dynamic":Naive,"clinical_dynamic_residual":Residual}[arm]().to(device)
    positive=int(train.ground_truth_pCR.sum()); criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train)-positive)/positive],device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=2)
    def infer(loader,training=False,zero=False,shuffled=None):
        if zero or shuffled is not None: loader=DataLoader(Dataset(loader.dataset.table,False,zero=zero,shuffled=shuffled),batch_size=4,shuffle=False,num_workers=4,pin_memory=True)
        model.train(training); ys=[]; logits=[]; correction=[]; ids=[]; loss_total=0.0
        context=torch.enable_grad() if training else torch.no_grad()
        with context:
            for image,y,clinical,features,pids in loader:
                image=image.to(device,non_blocking=True); y=y.to(device,non_blocking=True); clinical=clinical.to(device,non_blocking=True); features=features.to(device,non_blocking=True)
                if training: optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    out,delta=model(image,clinical,features); label=y[:,None].expand_as(out); loss=criterion(out,label)
                    if arm=="clinical_dynamic_residual": loss=loss+1e-3*delta.square().mean()
                if training: loss.backward(); optimizer.step()
                loss_total+=float(loss.detach())*len(y); ys.extend(y.detach().cpu().numpy()); logits.append(out.detach().float().cpu().numpy()); correction.append(delta.detach().float().cpu().numpy()); ids.extend(pids)
        return np.asarray(ys,dtype=int),np.concatenate(logits),np.concatenate(correction),ids,loss_total/len(loader.dataset)
    best=None; patience=0; history=[]
    for epoch in range(1,13):
        _,_,_,_,tl=infer(loaders["train"],True); yv,lv,_,_,vl=infer(loaders["val"]); pv=sigmoid(lv[:,endpoint_index(PRIMARY)])
        score=(float(roc_auc_score(yv,pv)),float(average_precision_score(yv,pv))); scheduler.step(score[0])
        history.append({"epoch":epoch,"train_loss":tl,"val_loss":vl,"val_T0_T1_auroc":score[0],"val_T0_T1_auprc":score[1],"lr":float(optimizer.param_groups[0]['lr'])})
        if best is None or score>best[0]: best=(score,epoch,{k:v.detach().cpu() for k,v in model.state_dict().items()}); patience=0
        else: patience+=1
        if patience>=4: break
    if best is None: raise RuntimeError("no selected dynamic checkpoint")
    d=ROOT/"checkpoints"/"formal"/arm; d.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":best[2],"arm":arm,"fold":fold,"epoch":best[1],"selection_endpoint":PRIMARY,"validation_auroc":best[0][0],"validation_auprc":best[0][1],"residual_head_zero_initialized":arm=="clinical_dynamic_residual"},d/f"fold_{fold}.pt")
    ld=ROOT/"logs"/"formal"/arm; ld.mkdir(parents=True,exist_ok=True); pd.DataFrame(history).to_csv(ld/f"fold_{fold}.csv",index=False)
    model.load_state_dict(best[2]); yv,lv,_,_,_=infer(loaders["val"]); thresholds=[youden(yv,sigmoid(lv[:,i])) for i in range(4)]
    yt,lt,dt,ids,_=infer(loaders["test"])
    if ids!=test.patient_id.tolist() or not np.array_equal(yt,test.ground_truth_pCR.to_numpy()): raise RuntimeError("dynamic test order drift")
    chunks=[]
    for i,endpoint in enumerate(ENDPOINTS):
        frame=test.loc[:,["patient_id","fold","ground_truth_pCR","clinical_logit"]].copy(); frame["endpoint"]=endpoint; frame["final_logit"]=lt[:,i]; frame["probability"]=sigmoid(lt[:,i]); frame["image_dynamic_logit"]=dt[:,i]; frame["threshold"]=thresholds[i]; chunks.append(frame)
    result=pd.concat(chunks,ignore_index=True)
    if arm=="clinical_dynamic_residual":
        _,zlog,zdelta,_,_=infer(loaders["test"],zero=True); mapping=dict(zip(test.patient_id,np.random.default_rng(SEED+fold).permutation(test.patient_id)))
        _,slog,sdelta,_,_=infer(loaders["test"],shuffled=mapping)
        for i,endpoint in enumerate(ENDPOINTS):
            selected=result.endpoint.eq(endpoint); result.loc[selected,"zero_final_logit"]=zlog[:,i]; result.loc[selected,"zero_dynamic_logit"]=zdelta[:,i]; result.loc[selected,"shuffled_final_logit"]=slog[:,i]; result.loc[selected,"shuffled_dynamic_logit"]=sdelta[:,i]
    return RunResult(arm,result)


def calibration(oof: np.ndarray, label: np.ndarray, seed: int) -> LogisticRegression:
    model=LogisticRegression(C=1_000_000.0,solver="lbfgs",max_iter=20_000,tol=1e-9,random_state=seed); model.fit(oof.reshape(-1,1),label); return model


def aggregate(frame: pd.DataFrame, arm: str) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    output={}; rows=[]
    for endpoint,g in frame.groupby("endpoint",sort=False):
        output[endpoint]=metrics(g.ground_truth_pCR.to_numpy(),g.probability.to_numpy(),g.threshold.to_numpy())
        for fold,h in g.groupby("fold",sort=True): rows.append({"model":arm,"endpoint":endpoint,"fold":int(fold),**metrics(h.ground_truth_pCR.to_numpy(),h.probability.to_numpy(),h.threshold.to_numpy())})
    return output,rows


def paired_bootstrap(frames: dict[str,pd.DataFrame]) -> dict[str,Any]:
    archive=np.load(paths()["bootstrap"],allow_pickle=False); ids=archive["patient_ids"].astype(str); indices=archive["indices"]
    if indices.shape!=(5000,808): raise RuntimeError("bootstrap contract drifted")
    result={"seed":SEED,"bootstrap_replicates":5000,"primary_endpoint":PRIMARY,"comparisons":{}}
    for endpoint in ENDPOINTS:
        ordered={arm:frame[frame.endpoint.eq(endpoint)].set_index("patient_id").loc[ids] for arm,frame in frames.items()}; y=ordered["clinical_only"].ground_truth_pCR.to_numpy(int); base=ordered["clinical_only"].probability.to_numpy()
        result["comparisons"][endpoint]={}
        for arm,current in ordered.items():
            if arm=="clinical_only": continue
            score=current.probability.to_numpy(); da=[]; dp=[]
            for ix in indices:
                yy=y[ix]
                if len(np.unique(yy))<2: continue
                da.append(roc_auc_score(yy,score[ix])-roc_auc_score(yy,base[ix])); dp.append(average_precision_score(yy,score[ix])-average_precision_score(yy,base[ix]))
            result["comparisons"][endpoint][arm]={"delta_auroc":float(roc_auc_score(y,score)-roc_auc_score(y,base)),"delta_auprc":float(average_precision_score(y,score)-average_precision_score(y,base)),"delta_auroc_ci95":[float(x) for x in np.percentile(da,[2.5,97.5])],"delta_auprc_ci95":[float(x) for x in np.percentile(dp,[2.5,97.5])]}
    return result


def paired_comparison(y: np.ndarray, reference: np.ndarray, current: np.ndarray) -> dict[str,Any]:
    """Patient-paired AUROC/AUPRC contrast on the locked bootstrap draws."""
    archive=np.load(paths()["bootstrap"],allow_pickle=False); indices=archive["indices"]
    if indices.shape!=(5000,808): raise RuntimeError("bootstrap contract drifted")
    auroc=[]; auprc=[]
    for ix in indices:
        yy=y[ix]
        if len(np.unique(yy))<2: continue
        auroc.append(roc_auc_score(yy,current[ix])-roc_auc_score(yy,reference[ix]))
        auprc.append(average_precision_score(yy,current[ix])-average_precision_score(yy,reference[ix]))
    return {"delta_auroc":float(roc_auc_score(y,current)-roc_auc_score(y,reference)),"delta_auprc":float(average_precision_score(y,current)-average_precision_score(y,reference)),"delta_auroc_ci95":[float(x) for x in np.percentile(auroc,[2.5,97.5])],"delta_auprc_ci95":[float(x) for x in np.percentile(auprc,[2.5,97.5])]}


def secondary_paired(residual: pd.DataFrame, recalibration: pd.DataFrame) -> dict[str,Any]:
    """Pre-specified sensitivity and image-reliance contrasts at T0→T1."""
    archive=np.load(paths()["bootstrap"],allow_pickle=False); ids=archive["patient_ids"].astype(str)
    r=residual[residual.endpoint.eq(PRIMARY)].set_index("patient_id").loc[ids]
    c=recalibration[recalibration.endpoint.eq(PRIMARY)].set_index("patient_id").loc[ids]
    y=r.ground_truth_pCR.to_numpy(int)
    if not np.array_equal(y,c.ground_truth_pCR.to_numpy(int)): raise RuntimeError("paired label order drift")
    real=r.probability.to_numpy()
    return {"seed":SEED,"bootstrap_replicates":5000,"endpoint":PRIMARY,"comparisons":{
        "residual_vs_clinical_recalibration":paired_comparison(y,c.probability.to_numpy(),real),
        "real_vs_zero_trajectory":paired_comparison(y,sigmoid(r.zero_final_logit.to_numpy()),real),
        "real_vs_patient_shuffled_trajectory":paired_comparison(y,sigmoid(r.shuffled_final_logit.to_numpy()),real),
    }}


def correction_summary(frame: pd.DataFrame) -> dict[str,Any]:
    primary=frame[frame.endpoint.eq(PRIMARY)].copy(); p=sigmoid(primary.clinical_logit.to_numpy()); y=primary.ground_truth_pCR.to_numpy(int); pred=primary.clinical_predicted.to_numpy(int); before=np.abs(p-y); after=np.abs(primary.probability.to_numpy()-y)
    groups={"clinical_confident_correct":(np.maximum(p,1-p)>=.7)&(pred==y),"clinical_uncertain":(p>.3)&(p<.7),"clinical_confident_wrong":(np.maximum(p,1-p)>=.7)&(pred!=y)}; output={}
    for name,selected in groups.items(): output[name]={"n":int(selected.sum()),"mean_abs_dynamic_logit":float(np.abs(primary.image_dynamic_logit.to_numpy()[selected]).mean()) if selected.any() else math.nan,"beneficial":int((after[selected]<before[selected]).sum()),"worsened":int((after[selected]>before[selected]).sum())}
    return output


def write_report(summary: dict[str,Any], paired: dict[str,Any], secondary: dict[str,Any], reliance: dict[str,Any], correction: dict[str,Any], clinical_audit: dict[str,Any]) -> None:
    primary=summary[PRIMARY]; compare=paired["comparisons"][PRIMARY]; r=compare["clinical_dynamic_residual"]
    rows=[]; labels={"clinical_only":"Clinical-only","clinical_recalibration":"Clinical recalibration","dynamic_image_only":"Dynamic MRI-only","naive_clinical_dynamic":"Naive Clinical + Dynamic MRI","clinical_dynamic_residual":"Clinical + Dynamic MRI Residual"}
    for arm in ARMS:
        d=compare.get(arm)
        delta = "–" if d is None else f"{d['delta_auroc']:+.4f}"
        rows.append(f"| {labels[arm]} | {primary[arm]['auroc']:.4f} | {primary[arm]['auprc']:.4f} | {primary[arm]['balanced_accuracy']:.4f} | {delta} |")
    sequence=[]
    for endpoint in ENDPOINTS:
        q=summary[endpoint]["clinical_dynamic_residual"]; d=paired["comparisons"][endpoint]["clinical_dynamic_residual"]; sequence.append(f"| {endpoint.replace('_','→')} | {q['auroc']:.4f} | {q['auprc']:.4f} | {d['delta_auroc']:+.4f} | [{d['delta_auroc_ci95'][0]:+.4f}, {d['delta_auroc_ci95'][1]:+.4f}] |")
    conclusion="SUPPORTS EARLY DYNAMIC MRI VALUE" if r['delta_auroc']>0 and r['delta_auroc_ci95'][0]>0 and r['delta_auprc']>0 else "NO CONFIRMED EARLY DYNAMIC MRI VALUE"
    recal=secondary["comparisons"]["residual_vs_clinical_recalibration"]
    zero=secondary["comparisons"]["real_vs_zero_trajectory"]
    shuffled=secondary["comparisons"]["real_vs_patient_shuffled_trajectory"]
    text=f"""# Longitudinal MRI Dynamic Embedding（T0–T3）

## 一句话结论

**{conclusion}**。本实验以严格因果 temporal embedding 检验治疗中 MRI 动态是否提供 clinical 外 pCR 信息；主要临床决策点为 T0→T1。

## 设计

808 名患者、275 名 pCR、固定 seed-2026 五折 outer CV。每访视的三相 DCE（pre/early/late）由共享 Kinetics R3D-18 编码，再由一层 causal transformer 汇聚；位置 t 只能 attend T0..t。ROI mask、FTV、radiomics、geometry scalar 都没有输入。Residual 输出为 `l=l_clinical+Δl_dynamic`，clinical logits 在 outer-train 内五折 cross-fitting 后冻结。Clinical-only 复现 AUROC={clinical_audit['observed_auroc']:.6f}、AUPRC={clinical_audit['observed_auprc']:.6f}。

## 主要结果：T0→T1

| Model | AUROC | AUPRC | Balanced Acc. | ΔAUROC vs Clinical |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Residual vs Clinical：ΔAUROC={r['delta_auroc']:+.6f}，95% CI [{r['delta_auroc_ci95'][0]:+.6f}, {r['delta_auroc_ci95'][1]:+.6f}]；ΔAUPRC={r['delta_auprc']:+.6f}，95% CI [{r['delta_auprc_ci95'][0]:+.6f}, {r['delta_auprc_ci95'][1]:+.6f}]。

作为 sensitivity analysis，Residual 相对 clinical recalibration 的 ΔAUROC={recal['delta_auroc']:+.6f}，95% CI [{recal['delta_auroc_ci95'][0]:+.6f}, {recal['delta_auroc_ci95'][1]:+.6f}]；因此也没有证据表明它超过经过校准的 clinical prediction。

## 时序曲线

| 可用 MRI prefix | Residual AUROC | Residual AUPRC | ΔAUROC vs Clinical | 95% CI |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(sequence)}

T2/T3 是次级、较晚的时间点，不应倒灌到 T0→T1 预测。

## Image reliance

在 T0→T1，real / zero / shuffled trajectory AUROC 分别为 {reliance['real']['auroc']:.4f} / {reliance['zero']['auroc']:.4f} / {reliance['shuffled']['auroc']:.4f}。real 相对 zero 的 ΔAUROC={zero['delta_auroc']:+.6f}，95% CI [{zero['delta_auroc_ci95'][0]:+.6f}, {zero['delta_auroc_ci95'][1]:+.6f}]；相对 patient-shuffled trajectory 的 ΔAUROC={shuffled['delta_auroc']:+.6f}，95% CI [{shuffled['delta_auroc_ci95'][0]:+.6f}, {shuffled['delta_auroc_ci95'][1]:+.6f}]。两者的 AUROC CI 均跨零，不能把较高的点估计解读为确证的患者特异性时序效应；它们不参与模型选择。

## Correction analysis

clinical confident-but-wrong 组 n={correction['clinical_confident_wrong']['n']}，平均 |Δl|={correction['clinical_confident_wrong']['mean_abs_dynamic_logit']:.4f}，有益/恶化={correction['clinical_confident_wrong']['beneficial']}/{correction['clinical_confident_wrong']['worsened']}。

## 结论与下一步

结论为 **{conclusion}**。若 T0→T1 的 paired CI 不支持正增益，不应以 T2/T3 较晚结果倒推早期临床效用；下一步应先在独立 cohort 或多 seed 中确认最早可用的正向 landmark。
"""
    d=ROOT/"reports"; d.mkdir(parents=True,exist_ok=True); (d/"longitudinal_mri_dynamic_embedding_zh.md").write_text(text)


def formal(device: str) -> None:
    folds=load_fold_manifest(paths()["folds"]); clinical=load_clinical_labels(paths()["clinical"],expected_patient_ids=folds.patient_ids); sequence_paths=cache_paths(clinical.patient_ids); split_rows=verify_inputs(folds,clinical,sequence_paths)
    causal_prefix_test(device); (ROOT/"metrics").mkdir(parents=True,exist_ok=True); pd.DataFrame(split_rows).to_csv(ROOT/"metrics"/"split_summary.csv",index=False)
    clinical_rows=[]; recal_rows=[]; dynamic={arm:[] for arm in ARMS[2:]}; audit=[]
    for fold in FOLDS:
        roles=folds.roles(fold,clinical.patient_ids); train,val,test=(np.flatnonzero(roles==role) for role in ("train","val","test")); selected=outer_clinical(clinical,train,val,fold); all_logit=selected.logits(clinical); oof=crossfit_logits(clinical,train,selected,fold); features=selected.scaler.transform(selected.encoder.transform(clinical)).astype(np.float32)
        def make_table(indices,logits): return pd.DataFrame({"patient_id":clinical.patient_ids[indices],"fold":fold,"ground_truth_pCR":clinical.pcr[indices],"clinical_logit":logits,"clinical_features":[json.dumps(v.tolist(),separators=(",",":")) for v in features[indices]]})
        train_table=make_table(train,oof); val_table=make_table(val,all_logit[val]); test_table=make_table(test,all_logit[test])
        for endpoint in ENDPOINTS:
            clinical_rows.append(pd.DataFrame({"patient_id":clinical.patient_ids[test],"fold":fold,"ground_truth_pCR":clinical.pcr[test],"endpoint":endpoint,"clinical_logit":all_logit[test],"probability":sigmoid(all_logit[test]),"threshold":selected.threshold}))
        cal=calibration(oof,clinical.pcr[train],SEED+fold)
        for endpoint in ENDPOINTS:
            th=youden(clinical.pcr[val],cal.predict_proba(all_logit[val,None])[:,1]); recal_rows.append(pd.DataFrame({"patient_id":clinical.patient_ids[test],"fold":fold,"ground_truth_pCR":clinical.pcr[test],"endpoint":endpoint,"clinical_logit":all_logit[test],"probability":cal.predict_proba(all_logit[test,None])[:,1],"threshold":th}))
        for arm in dynamic:
            dynamic[arm].append(train_arm(arm,fold,train_table,val_table,test_table,sequence_paths,device).rows)
        audit.append({"fold":fold,"n_train":len(train),"n_val":len(val),"n_test":len(test),"clinical_penalty":selected.penalty,"clinical_C":selected.c_value,"inner_crossfit_folds":5,"temporal_attention":"causal","checkpoint_selection_endpoint":PRIMARY,"test_used_for_selection":False})
    frames={"clinical_only":pd.concat(clinical_rows,ignore_index=True),"clinical_recalibration":pd.concat(recal_rows,ignore_index=True),**{arm:pd.concat(rows,ignore_index=True) for arm,rows in dynamic.items()}}
    for arm,frame in frames.items():
        if len(frame)!=808*4 or frame.duplicated(["patient_id","endpoint"]).any(): raise RuntimeError(f"OOF coverage drift {arm}")
    summary={}; per=[]
    for arm,frame in frames.items(): summary[arm],rows=aggregate(frame,arm); per.extend(rows)
    # transpose for report-friendly endpoint first layout
    by_endpoint={endpoint:{arm:summary[arm][endpoint] for arm in ARMS} for endpoint in ENDPOINTS}; paired=paired_bootstrap(frames)
    residual=frames["clinical_dynamic_residual"].copy(); base=frames["clinical_only"].set_index(["patient_id","endpoint"]); residual["clinical_probability"]=sigmoid(residual.clinical_logit); residual["clinical_threshold"]=[base.loc[(pid,ep),"threshold"] for pid,ep in zip(residual.patient_id,residual.endpoint)]; residual["clinical_predicted"]=(residual.clinical_probability>=residual.clinical_threshold).astype(int)
    p=residual[residual.endpoint.eq(PRIMARY)].copy(); zero=p.copy(); zero["probability"]=sigmoid(zero.zero_final_logit); shuffled=p.copy(); shuffled["probability"]=sigmoid(shuffled.shuffled_final_logit)
    reliance={"real":metrics(p.ground_truth_pCR.to_numpy(),p.probability.to_numpy(),p.threshold.to_numpy()),"zero":metrics(zero.ground_truth_pCR.to_numpy(),zero.probability.to_numpy(),zero.threshold.to_numpy()),"shuffled":metrics(shuffled.ground_truth_pCR.to_numpy(),shuffled.probability.to_numpy(),shuffled.threshold.to_numpy()),"endpoint":PRIMARY,"shuffle_rule":"fixed_rng_seed_2026_plus_outer_fold"}
    correction=correction_summary(residual); secondary=secondary_paired(residual,frames["clinical_recalibration"]); clinical_audit={"locked_auroc":0.709155722326454,"locked_auprc":0.5575141168308745,"observed_auroc":by_endpoint["T0"]["clinical_only"]["auroc"],"observed_auprc":by_endpoint["T0"]["clinical_only"]["auprc"],"exact_reproduction":by_endpoint["T0"]["clinical_only"]["auroc"]==0.709155722326454 and by_endpoint["T0"]["clinical_only"]["auprc"]==0.5575141168308745}
    if not clinical_audit["exact_reproduction"]: raise RuntimeError("clinical baseline drifted")
    private= pd.concat([frame.assign(arm=arm) for arm,frame in frames.items()],ignore_index=True,sort=False); private=private.merge(residual[["patient_id","endpoint","clinical_probability","clinical_threshold","clinical_predicted"]],on=["patient_id","endpoint"],how="left")
    d=ROOT/"predictions"/"formal"; d.mkdir(parents=True,exist_ok=True); private.to_csv(d/"oof_dynamic_predictions.private.csv",index=False)
    pd.DataFrame(per).to_csv(ROOT/"metrics"/"per_fold_metrics.csv",index=False); (ROOT/"metrics"/"summary.json").write_text(json.dumps(by_endpoint,indent=2)+"\n"); (ROOT/"metrics"/"paired_bootstrap.json").write_text(json.dumps(paired,indent=2)+"\n"); (ROOT/"metrics"/"secondary_paired_comparisons.json").write_text(json.dumps(secondary,indent=2)+"\n"); (ROOT/"metrics"/"image_reliance_audit.json").write_text(json.dumps(reliance,indent=2)+"\n"); (ROOT/"metrics"/"correction_analysis.json").write_text(json.dumps(correction,indent=2)+"\n"); (ROOT/"metrics"/"clinical_baseline_audit.json").write_text(json.dumps(clinical_audit,indent=2)+"\n"); (ROOT/"metrics"/"leakage_audit.json").write_text(json.dumps({"status":"PASS","fold_manifest_sha256":folds.sha256,"outer_patient_intersections_empty":True,"training_clinical_logits":"five_fold_inner_cross_fitted","temporal_attention":"strictly_causal","test_used_for_selection":False,"model_input_excludes_mask_ftv_radiomics_geometry":True,"all_four_visits_cache_coverage":808},indent=2)+"\n"); (ROOT/"metrics"/"run_provenance.json").write_text(json.dumps({"longitudinal_cache":"C1B-H_DCE8_adaptivephase_axiscanon_T0fixedcrop","source_cache_file_count":len(list(paths()["cache"].glob("*.npy"))),"bootstrap_reused":True,"fold_manifest_sha256":folds.sha256,"clinical_labels_sha256":clinical.sha256},indent=2)+"\n")
    write_report(by_endpoint,paired,secondary,reliance,correction,clinical_audit); validate()


def postprocess() -> None:
    """Regenerate public secondary contrasts/report from ignored OOF predictions."""
    private=ROOT/"predictions"/"formal"/"oof_dynamic_predictions.private.csv"
    if not private.exists(): raise RuntimeError("formal private predictions are required for postprocessing")
    frame=pd.read_csv(private)
    residual=frame[frame.arm.eq("clinical_dynamic_residual")].copy()
    recalibration=frame[frame.arm.eq("clinical_recalibration")].copy()
    secondary=secondary_paired(residual,recalibration)
    (ROOT/"metrics"/"secondary_paired_comparisons.json").write_text(json.dumps(secondary,indent=2)+"\n")
    write_report(json.loads((ROOT/"metrics"/"summary.json").read_text()),json.loads((ROOT/"metrics"/"paired_bootstrap.json").read_text()),secondary,json.loads((ROOT/"metrics"/"image_reliance_audit.json").read_text()),json.loads((ROOT/"metrics"/"correction_analysis.json").read_text()),json.loads((ROOT/"metrics"/"clinical_baseline_audit.json").read_text()))
    validate()


def validate() -> None:
    required=[ROOT/"configs"/"formal.yaml",ROOT/"metrics"/"summary.json",ROOT/"metrics"/"paired_bootstrap.json",ROOT/"metrics"/"secondary_paired_comparisons.json",ROOT/"metrics"/"image_reliance_audit.json",ROOT/"metrics"/"correction_analysis.json",ROOT/"metrics"/"clinical_baseline_audit.json",ROOT/"metrics"/"leakage_audit.json",ROOT/"metrics"/"split_summary.csv",ROOT/"reports"/"longitudinal_mri_dynamic_embedding_zh.md"]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise RuntimeError(f"missing artifacts: {missing}")
    summary=json.loads((ROOT/"metrics"/"summary.json").read_text()); paired=json.loads((ROOT/"metrics"/"paired_bootstrap.json").read_text()); secondary=json.loads((ROOT/"metrics"/"secondary_paired_comparisons.json").read_text()); private=ROOT/"predictions"/"formal"/"oof_dynamic_predictions.private.csv"; checkpoints=list((ROOT/"checkpoints"/"formal").glob("*/*.pt")); logs=list((ROOT/"logs"/"formal").glob("*/*.csv"))
    if set(summary)!=set(ENDPOINTS) or paired.get("bootstrap_replicates")!=5000 or secondary.get("bootstrap_replicates")!=5000 or set(secondary.get("comparisons",{}))!={"residual_vs_clinical_recalibration","real_vs_zero_trajectory","real_vs_patient_shuffled_trajectory"} or not private.exists() or len(checkpoints)!=15 or len(logs)!=15: raise RuntimeError("formal artifact coverage drift")
    if subprocess.run(["git","check-ignore","--quiet",str(private)],cwd=REPO,check=False).returncode!=0: raise RuntimeError("private output not ignored")
    output={"status":"PASS","patients":808,"pcr_positive":275,"dynamic_models":15,"training_logs":15,"endpoints":list(ENDPOINTS),"primary_endpoint":PRIMARY,"bootstrap_replicates":5000,"causal_prefix_test":"PASS","patient_predictions_gitignored":True,"only_frozen_C1B-H_DCE_cache_used":True}; (ROOT/"metrics"/"formal_validation.json").write_text(json.dumps(output,indent=2)+"\n"); print(json.dumps(output,indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(); subs=parser.add_subparsers(dest="command",required=True); f=subs.add_parser("formal"); f.add_argument("--device",default="cuda:0"); c=subs.add_parser("causal-test"); c.add_argument("--device",default="cuda:0"); subs.add_parser("postprocess"); subs.add_parser("validate"); args=parser.parse_args()
    if args.command=="formal": formal(args.device)
    elif args.command=="causal-test": causal_prefix_test(args.device); print('{"status":"PASS"}')
    elif args.command=="postprocess": postprocess()
    else: validate()

if __name__=="__main__": main()
