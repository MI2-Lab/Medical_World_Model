"""Fold-local, outcome-blind family-balanced target construction."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from .contracts import RAW_DIR, ROI_DIR, TARGET_DIR, V5_STATES, atomic_json, folds, token

VISITS=(0,1,2)
def corr(a,b):
    keep=np.isfinite(a)&np.isfinite(b)
    if keep.sum()<20 or np.unique(a[keep]).size<2 or np.unique(b[keep]).size<2:return np.nan
    return float(spearmanr(a[keep],b[keep]).statistic)

def load_population():
    paths=sorted(RAW_DIR.glob("*.private.npz")); rows=[]; roi_rows=[]; names=None
    for p in paths:
        with np.load(p,allow_pickle=False) as z:
            pid=str(z["patient_id"].item()); n=tuple(z["feature_name"].astype(str)); val=np.asarray(z["value"],np.float64); vv=np.asarray(z["variant_valid"],bool)
        rp=ROI_DIR/f"{token(pid)}.private.npz"
        with np.load(rp,allow_pickle=False) as z:
            roi=np.asarray(z["roi_mask"],bool); rm=np.asarray(z["radiomics_mask"],bool); ftv=np.asarray(z["ftv"],np.float64); tv=np.asarray(z["local_mask_volume_mm3"],np.float64)
        if names is None:names=n
        if n!=names or val.shape!=(4,3,len(names)) or roi.shape!=(4,32,72,72):raise ValueError("V2 raw contract failed")
        rows.append((pid,val,vv)); roi_rows.append((roi,rm,ftv,tv))
    if len(rows)!=375:raise ValueError(f"expected 375 V2 radiomics patients, got {len(rows)}")
    return tuple(x[0] for x in rows),tuple(names),np.stack([x[1] for x in rows]),np.stack([x[2] for x in rows]),np.stack([x[0] for x in roi_rows]),np.stack([x[1] for x in roi_rows]),np.stack([x[2] for x in roi_rows]),np.stack([x[3] for x in roi_rows])

def morphology(ids, roi, rm):
    book=pd.read_excel("/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx",sheet_name="datawith4visits")
    book={str(int(x)).zfill(6):r for x,r in zip(book["CLINICAL-TRIAL-SUBJECT-ID"],book.to_dict("records"))}
    out=np.full((len(ids),4,4),np.nan,np.float64)
    for i,pid in enumerate(ids):
        key=pid.split("-")[-1]; r=book.get(key)
        if r is None:raise ValueError(f"workbook morphology join missing {pid}")
        for v in range(4):
            m=roi[i,v]
            if not rm[i,v] or m.sum()<64:continue
            xyz=np.argwhere(m).astype(float); xyz[:,0]*=2.0; xyz[:,1:]*=.9
            cov=np.cov(xyz,rowvar=False) if len(xyz)>3 else np.zeros((3,3)); eig=np.sort(np.maximum(np.linalg.eigvalsh(cov),0))[::-1]
            if eig[0]>0:
                out[i,v,2]=np.sqrt(eig[1]/eig[0]); out[i,v,3]=np.sqrt(eig[2]/eig[0])
            out[i,v,0]=float(r[f"LD_T{v}"]) if np.isfinite(r[f"LD_T{v}"]) else np.nan
            out[i,v,1]=float(r[f"SPHERICITY_T{v}"]) if np.isfinite(r[f"SPHERICITY_T{v}"]) else np.nan
    return out

def family_indices(names):
    prefixes={"INTENSITY":("pre::","early::","late::"),"KINETIC":("early_minus_pre::","late_minus_pre::","peak_relative_enhancement::","late_minus_peak_relative_enhancement::")}
    result={}
    for fam,prefs in prefixes.items():
        result[fam]=[i for i,n in enumerate(names) if n.startswith(prefs) and "original_firstorder_" in n]
    result["TEXTURE"]=[i for i,n in enumerate(names) if "original_firstorder_" not in n]
    return result

def stable_selection(values, valid, variant_valid, names, train, family, maxn=12):
    idx=[]; audit=[]; train=np.asarray(train); fi=family_indices(names)[family]
    for j in fi:
        st=[]; ok=True; pooled=[]
        for v in VISITS:
            eligible=valid[train,v]; den=int(eligible.sum()); a=values[train,v,0,j]; d=values[train,v,2,j]; e=values[train,v,1,j]
            of=eligible&np.isfinite(a); df=eligible&np.isfinite(d); outward=corr(a[of&df],d[of&df]); er=eligible&variant_valid[train,v,1]; sym=er&np.isfinite(a)&np.isfinite(e)&np.isfinite(d); comparable=float(sym.sum()/max(1,er.sum())); pairs=[corr(a[sym],e[sym]),corr(a[sym],d[sym]),corr(e[sym],d[sym])]; symmetric=float(np.nanmedian(pairs)) if np.isfinite(pairs).sum()==3 else np.nan
            coverage=float(of.sum()/max(1,den))>=.98 and float(df.sum()/max(1,den))>=.98 and comparable>=.95
            good=coverage and np.isfinite(outward) and np.isfinite(symmetric) and outward>=.80 and symmetric>=.80
            ok &= good; st.extend([outward,symmetric]); pooled.append(a[of])
        allv=np.concatenate(pooled) if pooled else np.array([]); iqr=float(np.quantile(allv,.75)-np.quantile(allv,.25)) if len(allv) else 0
        minimum=float(np.nanmin(st)) if np.isfinite(st).all() else np.nan; passed=bool(ok and iqr>0 and np.isfinite(minimum)); audit.append({"feature":names[j],"minimum_stability":minimum,"iqr":iqr,"passed":passed})
        if passed:idx.append((j,minimum,names[j]))
    idx.sort(key=lambda x:(-x[1],x[2])); return [x[0] for x in idx[:maxn]],audit

def design(ftv,volume,train):
    x=np.column_stack([np.log1p(np.maximum(ftv.reshape(-1),0)),np.log1p(np.maximum(volume.reshape(-1),0))]); mu=x[train].mean(0); sd=np.where(x[train].std(0)>0,x[train].std(0),1); z=(x-mu)/sd; one=np.tile(np.eye(3), (len(ftv),1)); return np.column_stack([np.ones(len(z)),z,one,z[:,[0]]*one]),mu,sd

def residualize(base, c0, train, groups):
    scaler=StandardScaler().fit(c0[train]); xc=scaler.transform(c0); alphas=[1.,10.,100.,1000.]; g=np.asarray(groups)[train]; splitter=GroupKFold(5); scores=[]
    for alpha in alphas:
        score=[]
        for tr,va in splitter.split(xc[train],base[train],g):
            model=Ridge(alpha=alpha).fit(xc[train][tr],base[train][tr]); score.append(np.mean((model.predict(xc[train][va])-base[train][va])**2))
        scores.append(np.mean(score))
    alpha=alphas[int(np.argmin(scores))]; model=Ridge(alpha=alpha).fit(xc[train],base[train]); return base-model.predict(xc),alpha

def build_fold(fold, ids,names,values,variant,valid,rm,ftv,volume,c0):
    frame=folds(); fids=set(frame.loc[(frame.fold==fold)&(frame.split=="train"),"patient_id"].astype(str)); train=np.asarray([p in fids for p in ids]); row_train=np.repeat(train,3); groups=np.repeat(np.asarray(ids),3)
    morph=morphology(ids, np.asarray([np.load(ROI_DIR/f"{token(p)}.private.npz",allow_pickle=False)["roi_mask"] for p in ids]), rm)
    fam_raw={"MORPHOLOGY":morph[:,:3],"INTENSITY":values[:,:3,0,:][:,:,family_indices(names)["INTENSITY"]],"KINETIC":values[:,:3,0,:][:,:,family_indices(names)["KINETIC"]],"TEXTURE":values[:,:3,0,:][:,:,family_indices(names)["TEXTURE"]]}
    selected={}; audits={}
    for fam in ("INTENSITY","KINETIC","TEXTURE"):
        selected[fam],audits[fam]=stable_selection(values,valid,variant,names,train,fam)
        if len(selected[fam])<8:raise RuntimeError(f"fold {fold} {fam} has fewer than 8 stable features")
    common_mask=valid[:,:3].copy();
    for fam in ("INTENSITY","KINETIC","TEXTURE"):
        common_mask &= np.isfinite(values[:,:3,0,selected[fam]]).all(-1)
    common_mask &= np.isfinite(morph[:,:3]).all(-1)
    coverage={f"T{v}":float(common_mask[:,v].mean()) for v in VISITS}
    if coverage["T0"]<.90 or coverage["T1"]<.90 or coverage["T2"]<.85:raise RuntimeError(f"fold {fold} target coverage failed: {coverage}")
    y=np.zeros((len(ids),3,16),np.float64); transform={"fold":fold,"selected":{k:[names[i] for i in selected[k]] for k in selected},"coverage":coverage,"stability_audit":audits,"c0_alpha":{}}
    # Base residualization is fit on outer-train rows only. Each family then
    # receives a separate PCA4 so no high-variance intensity family dominates.
    for fi,(fam,raw) in enumerate((("MORPHOLOGY",fam_raw["MORPHOLOGY"]),("INTENSITY",fam_raw["INTENSITY"]),("KINETIC",fam_raw["KINETIC"]),("TEXTURE",fam_raw["TEXTURE"]))):
        arr=raw.reshape(len(ids)*3,-1); mask=common_mask.reshape(-1); design_x,mu,sd=design(ftv[:,:3],volume[:,:3],row_train); usable=mask&np.isfinite(arr).all(1); fit=usable&row_train
        base=np.where(np.isfinite(arr),arr,0); coef=np.linalg.lstsq(design_x[fit],base[fit],rcond=None)[0]; base_res=base-design_x@coef
        c0flat=np.concatenate([c0[:,:3],np.tile(np.eye(3),(len(ids),1,1))],-1).reshape(len(ids)*3,-1); residual,alpha=residualize(base_res,c0flat,fit,groups)
        scaler=StandardScaler().fit(residual[fit]); zs=scaler.transform(residual); pca=PCA(4,whiten=False,svd_solver="full").fit(zs[fit]); pc=pca.transform(zs); pc=pc/np.where(pc[fit].std(0)>0,pc[fit].std(0),1)
        pc[~usable]=0; y[:,:,fi*4:(fi+1)*4]=pc.reshape(len(ids),3,4); transform["c0_alpha"][fam]=alpha; transform[f"{fam.lower()}_pca_variance"]=pca.explained_variance_.tolist()
    target_mask=np.zeros((len(ids),4),bool); target_mask[:,:3]=common_mask; out=np.zeros((len(ids),4,16),np.float32); out[:,:3]=y.astype(np.float32); out[~target_mask]=0
    ftv_out=np.zeros((len(ids),4),np.float32); fm=np.asarray(rm,bool)&np.isfinite(ftv); ftv_out[fm]=ftv[fm].astype(np.float32)
    path=TARGET_DIR/f"fold_{fold}_targets.private.npz"; path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,patient_id=np.asarray(ids),target=out,target_mask=target_mask,ftv=ftv_out,ftv_mask=fm)
    transform["target_sha256"]=__import__("spatial_rg.contracts",fromlist=["sha256_file"]).sha256_file(path); transform["outcome_fields_read"]=[]; transform["clinical_fields_read"]=[]; atomic_json(TARGET_DIR/f"fold_{fold}_transform.private.json",transform)
    return {"fold":fold,"status":"PASS","patients":len(ids),"coverage":coverage,"selected_features":{k:len(v) for k,v in selected.items()},"target_sha256":transform["target_sha256"],"t3_mask_false":True}

def build_all():
    ids,names,values,variant,roi,rm,ftv,volume=load_population(); frame=folds();
    # C0 is used only for fold-train target residualization and is loaded from
    # the frozen V5 pilot archive; no outcome field is touched.
    c0_by_fold=[]
    for fold in range(5):
        with np.load(V5_STATES/f"seed2026_fold{fold}_P1_PARALLEL_states.private.npz",allow_pickle=False) as z:
            pos={str(p):i for i,p in enumerate(z["patient_id"].astype(str))}; c0_by_fold.append(np.stack([z["c0_state"][pos[p]] for p in ids]))
    results=[build_fold(f,ids,names,values,variant,rm,rm,ftv,volume,c0_by_fold[f]) for f in range(5)]
    payload={"status":"PASS","folds":results,"patients":len(ids),"target_shape":[4,16],"grounding_visits":["T0","T1","T2"],"t3_mask_false":True,"outcome_fields_read":[],"clinical_fields_read":[]}; atomic_json(Path(__file__).resolve().parents[2]/"target_feasibility.json",payload); return payload
