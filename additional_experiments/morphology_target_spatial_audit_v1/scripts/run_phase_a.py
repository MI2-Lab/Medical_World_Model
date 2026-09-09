#!/usr/bin/env python3
"""Outcome-blind morphology source, mask, residual and frozen-probe audit."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from morph_audit.contracts import *
from morph_audit.data import load_population, workbook, load_input_features, current_masks, load_c0

def corr(a,b):
    k=np.isfinite(a)&np.isfinite(b)
    return float(spearmanr(a[k],b[k]).statistic) if k.sum()>10 and np.unique(a[k]).size>1 and np.unique(b[k]).size>1 else float("nan")

def stats(x):
    x=np.asarray(x,float); z=x[np.isfinite(x)]
    q=np.percentile(z,[25,50,75]) if len(z) else [np.nan]*3
    return {"finite":float(np.isfinite(x).mean()),"n":int(len(z)),"zero":int((z==0).sum()),"negative":int((z<0).sum()),"unique":int(np.unique(z).size),"q25":float(q[0]),"median":float(q[1]),"q75":float(q[2]),"iqr":float(q[2]-q[0]),"mad":float(np.median(np.abs(z-np.median(z)))) if len(z) else float("nan")}

def design(ftv,vol,train):
    x=np.c_[np.log1p(np.maximum(ftv,0)),np.log1p(np.maximum(vol,0))]
    mu=x[train].mean(0); sd=np.where(x[train].std(0)>0,x[train].std(0),1); z=(x-mu)/sd
    oh=np.eye(3)[np.arange(len(x))%3]
    return np.c_[np.ones(len(x)),z,oh,z[:,0,None]*oh]

def residual_target(raw,mask,ftv,vol,c0,ids,train):
    n,t,d=raw.shape; rows=n*t; flat=raw.reshape(rows,d).astype(float); ok=mask.reshape(-1)&np.isfinite(flat).all(1); tr=np.repeat(train,t)&ok
    dx=design(ftv.reshape(-1),vol.reshape(-1),tr); base=np.full_like(flat,np.nan)
    for j in range(d):
        fit=tr&np.isfinite(flat[:,j]); coef=np.linalg.lstsq(dx[fit],flat[fit,j],rcond=None)[0]; base[:,j]=flat[:,j]-dx@coef
    c=np.concatenate([c0,np.tile(np.eye(3),(n,1,1))],axis=-1).reshape(rows,-1); out=np.full_like(base,np.nan); alphas=[]
    for j in range(d):
        fit=tr&np.isfinite(base[:,j]); xx=c[fit]; yy=base[fit,j]; g=np.asarray(ids).repeat(t)[fit]; scores=[]
        for a in (1.,10.,100.,1000.):
            ss=[]
            for ti,vi in GroupKFold(5).split(xx,yy,g):
                m=Ridge(alpha=a).fit(xx[ti],yy[ti]); ss.append(float(np.mean((m.predict(xx[vi])-yy[vi])**2)))
            scores.append(np.mean(ss))
        alpha=(1.,10.,100.,1000.)[int(np.argmin(scores))]; m=Ridge(alpha=alpha).fit(xx,yy); out[:,j]=base[:,j]-m.predict(c); alphas.append(alpha)
    out=np.where(ok[:,None],out,0.).reshape(n,t,d); fitmask=mask&np.isfinite(out).all(-1); z=out.reshape(rows,d); trflat=np.repeat(train,t)&fitmask.reshape(-1)
    sc=StandardScaler().fit(z[trflat]); zz=sc.transform(z); pca=PCA(d,svd_solver="full").fit(zz[trflat]); pc=pca.transform(zz); pc=pc/np.where(pc[trflat].std(0)>0,pc[trflat].std(0),1); pc[~fitmask.reshape(-1)]=0
    return pc.reshape(n,t,d).astype(np.float32),alphas,pca.explained_variance_ratio_.tolist()

def split_masks(frame,fold,ids):
    cur=frame[frame.fold==fold]
    sets={s:set(cur.loc[cur.split==s,"patient_id"].astype(str)) for s in ("train","val","test")}
    return tuple(np.asarray([str(p) in sets[s] for p in ids]) for s in ("train","val","test"))

def preprocess(x,train,seed):
    sc=StandardScaler().fit(x[train]); z=sc.transform(x); meta={"pca":False}
    if x.shape[1]>512:
        n=min(32,z.shape[0]-1,z.shape[1]); p=PCA(n,svd_solver="randomized",random_state=seed).fit(z[train]); z=p.transform(z); meta={"pca":True,"components":int(n)}
    return z,meta

def probe_visit(x,y,mask,frame,fold,ids,visit,seed):
    tr,va,te=split_masks(frame,fold,ids); valid=mask[:,visit]&np.isfinite(y[:,visit]).all(-1); tr&=valid; va&=valid; te&=valid
    if tr.sum()<20 or va.sum()<10 or te.sum()<10:return {"spearman":np.nan,"r2":np.nan,"nrmse":np.nan,"alpha":np.nan,"n_test":int(te.sum())}
    z,meta=preprocess(x[:,visit],tr,seed)
    oh=np.eye(3)[np.full(len(ids),visit)]
    z_aug=np.c_[z,oh]
    sc=StandardScaler().fit(z_aug[tr]); z=sc.transform(z_aug)
    best=(None,1e99)
    for a in (.1,1.,10.,100.,1000.):
        m=Ridge(alpha=a).fit(z[tr],y[tr,visit]); mse=float(np.mean((m.predict(z[va])-y[va,visit])**2))
        if mse<best[1]-1e-12 or (abs(mse-best[1])<=1e-12 and (best[0] is None or a>best[0])):best=(a,mse)
    trva=tr|va
    sc2=StandardScaler().fit(z_aug[trva]); z2=sc2.transform(z_aug)
    m=Ridge(alpha=best[0]).fit(z2[trva],y[trva,visit]); pred=m.predict(z2[te]); truth=y[te,visit]
    return {"spearman":corr(truth,pred),"r2":float(r2_score(truth,pred)),"nrmse":float(np.sqrt(np.mean((truth-pred)**2))/(np.std(truth)+1e-12)),"alpha":float(best[0]),"n_test":int(te.sum()),"preprocess":meta}

def probe_all(x,y,mask,frame,fold,ids):
    return [probe_visit(x,y,mask,frame,fold,ids,v,202600+fold) for v in range(3)]

def main():
    pop=load_population(); ids=pop["ids"]; wb=workbook(ids); frame=folds(); c0=load_c0(ids); inputs=load_input_features(ids); shared=current_masks(ids)
    raw_base=np.concatenate((wb[:,:3,:],pop["morph"][:,:3,0,:2]),axis=-1)
    raw_local=pop["morph"][:,:3,0,:4]
    candidates={
        "CURRENT_SHARED":(raw_base,shared[:,:,:3]),
        "M1_MORPH_ONLY":(raw_base,np.broadcast_to((pop["rm"][:,:3]&np.isfinite(raw_base).all(-1))[None],(5,len(ids),3)).copy()),
        "M2_LOCAL_SHAPE":(raw_local,np.broadcast_to((pop["rm"][:,:3]&np.isfinite(raw_local).all(-1))[None],(5,len(ids),3)).copy()),
        "M3_ZERO_AWARE":(raw_base,np.broadcast_to(((pop["rm"][:,:3]&np.isfinite(raw_base).all(-1))&(wb[:,:3,0]>0))[None],(5,len(ids),3)).copy())
    }
    quality={}; masks_report={}; residuals={}; source_order={"CURRENT_SHARED":["WB_LD","WB_SPH","MASK_E21","MASK_F31"],"M1_MORPH_ONLY":["WB_LD","WB_SPH","MASK_E21","MASK_F31"],"M2_LOCAL_SHAPE":["MASK_E21","MASK_F31","MASK_BBOX_FILL","MASK_SLICE_PROFILE"],"M3_ZERO_AWARE":["WB_LD","WB_SPH","MASK_E21","MASK_F31"]}
    for name,(raw,ms) in candidates.items():
        rows=[]; eligible=True
        for f in range(5):
            m=ms[f]; cov=m.mean(0); ok=bool(cov[0]>=.90 and cov[1]>=.90 and cov[2]>=.85 and np.isfinite(raw[m]).all())
            rows.append({"fold":f,"coverage":{"T0":float(cov[0]),"T1":float(cov[1]),"T2":float(cov[2])},"mask_rows":[int(x) for x in m.sum(0)],"features":[stats(raw[:,:,j]) for j in range(4)],"eligible":ok}); eligible &= ok
        quality[name]={"candidate":name,"sources":source_order[name],"eligible":bool(eligible),"folds":rows}
        masks_report[name]={"coverage_by_fold":[r["coverage"] for r in rows],"row_counts_by_fold":[int(x.sum()) for x in ms],"shared_mask_dependency":name=="CURRENT_SHARED","t3_mask_false":True}
        if name!="CURRENT_SHARED":
            for f in range(5):
                train=np.asarray([p in set(frame.loc[(frame.fold==f)&(frame.split=="train"),"patient_id"].astype(str)) for p in ids])
                pc,alpha,var=residual_target(raw,ms[f],pop["ftv"][:,:3],pop["volume"][:,:3],c0[f,:,:3],ids,train)
                residuals[(name,f)]=(pc,ms[f])
                q=quality[name]["folds"][f];q["c0_alpha"]=alpha;q["pca_explained_variance_ratio"]=var
                path=ROOT/f"features/private/{name}_fold{f}.private.npz";path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,patient_id=ids,target=pc,target_mask=ms[f],raw=raw,ftv=pop["ftv"][:,:3],ftv_mask=pop["ftvm"][:,:3])
    stability={}
    for j,nm in enumerate(("MASK_E21","MASK_F31","MASK_BBOX_FILL","MASK_SLICE_PROFILE")):
        rr=[]
        for v in range(3):
            a=pop["morph"][:,v,0,j];e=pop["morph"][:,v,1,j];d=pop["morph"][:,v,2,j]; eligible=pop["rm"][:,v]; sym=eligible&np.isfinite(a)&np.isfinite(e)&np.isfinite(d); od=eligible&np.isfinite(a)&np.isfinite(d)
            rr.append({"visit":VISIT_NAMES[v],"original_dilation":corr(a[od],d[od]),"original_erosion":corr(a[sym],e[sym]),"erosion_dilation":corr(e[sym],d[sym]),"comparable_coverage":float(sym.sum()/max(1,eligible.sum())),"absolute_change_dilation":float(np.nanmedian(np.abs(d[sym]-a[sym]))) if sym.any() else np.nan,"relative_change_dilation":float(np.nanmedian(np.abs(d[sym]-a[sym])/(np.abs(a[sym])+1e-8))) if sym.any() else np.nan})
        stability[nm]=rr
    required_local={"M1_MORPH_ONLY":("MASK_E21","MASK_F31"),"M2_LOCAL_SHAPE":("MASK_E21","MASK_F31","MASK_BBOX_FILL","MASK_SLICE_PROFILE"),"M3_ZERO_AWARE":("MASK_E21","MASK_F31")}
    stability_gate={}
    for name,needed in required_local.items():
        checks=[]
        for nm in needed:
            for rec in stability[nm]:
                pairwise=float(np.nanmedian([rec["original_dilation"],rec["original_erosion"],rec["erosion_dilation"]]))
                checks.append({"feature":nm,"visit":rec["visit"],"outward":rec["original_dilation"],"symmetric_median":pairwise,"comparable_coverage":rec["comparable_coverage"],"pass":bool(rec["original_dilation"]>=.80 and pairwise>=.80 and rec["comparable_coverage"]>=.95)})
        stability_gate[name]={"checks":checks,"pass":bool(all(x["pass"] for x in checks))}
        quality[name]["stability_gate"]=stability_gate[name]
        quality[name]["eligible"]=bool(quality[name]["eligible"] and stability_gate[name]["pass"])
    agreement={VISIT_NAMES[v]:{"WB_LD_vs_MASK_E21":corr(wb[:,v,0],pop["morph"][:,v,0,0]),"WB_SPH_vs_MASK_BBOX_FILL":corr(wb[:,v,1],pop["morph"][:,v,0,2]),"LD_zero_count":int((wb[:,v,0]==0).sum()),"LD_zero_with_valid_mask":int(((wb[:,v,0]==0)&pop["rm"][:,v]).sum())} for v in range(3)}
    probe={"raw":{},"c0_residual":{}}
    for name in residuals:
        pass
    for name in ("M1_MORPH_ONLY","M2_LOCAL_SHAPE","M3_ZERO_AWARE"):
        probe["raw"][name]={};probe["c0_residual"][name]={}
        raw,ms=candidates[name]
        for f in range(5):
            pc,mask=residuals[(name,f)];probe["raw"][name][str(f)]={};probe["c0_residual"][name][str(f)]={}
            for inp,xall in inputs.items():
                x=xall[f] if inp=="C0_STATE" else xall
                probe["raw"][name][str(f)][inp]=probe_all(x,raw,mask,frame,f,ids)
                probe["c0_residual"][name][str(f)][inp]=probe_all(x,pc,mask,frame,f,ids)
    def macro(blob,name,inp):
        return float(np.nanmean([v["spearman"] for f in range(5) for v in blob[name][str(f)][inp]]))
    summary={}
    for name in probe["raw"]:
        summary[name]={inp:{"raw":macro(probe["raw"],name,inp),"c0_residual":macro(probe["c0_residual"],name,inp)} for inp in inputs}
    eligible=[n for n in ("M1_MORPH_ONLY","M2_LOCAL_SHAPE","M3_ZERO_AWARE") if quality[n]["eligible"]]
    gains={n:summary[n]["SPATIAL_FULL"]["c0_residual"]-summary[n]["C0_STATE"]["c0_residual"] for n in eligible}
    diagnosis="TARGET_MEASUREMENT_FAILURE" if not eligible else ("SPATIAL_INFORMATION_PRESENT" if any(v>=.03 for v in gains.values()) else "SPATIAL_INFORMATION_ABSENT")
    result={"status":"PASS","diagnosis":diagnosis,"eligible_candidates":eligible,"target_quality":quality,"mask_audit":masks_report,"stability":stability,"source_agreement":agreement,"probe_summary":summary,"probe_detail":probe,"outcome_fields_read":[],"clinical_fields_read":[]}
    write_json(ROOT/"target_quality.json",quality);write_json(ROOT/"morphology_source_audit.json",{"status":"PASS","stability":stability,"stability_gate":stability_gate,"source_agreement":agreement,"diagnosis":diagnosis,"outcome_fields_read":[],"clinical_fields_read":[]});write_json(ROOT/"mask_coupling_audit.json",masks_report);write_json(ROOT/"residualization_audit.json",{"status":"PASS","target_quality":quality,"outcome_fields_read":[],"clinical_fields_read":[]});write_json(ROOT/"frozen_spatial_probe.json",result);write_json(ROOT/"phase_a_decision.json",{"diagnosis":diagnosis,"eligible_candidates":eligible,"frozen_spatial_gain":gains,"stability_gate":stability_gate,"phase_b_allowed":bool(eligible),"pCR_evaluation":"LOCKED","outcome_fields_read":[],"clinical_fields_read":[]})
    print(json.dumps({"status":"PASS","diagnosis":diagnosis,"eligible_candidates":eligible,"frozen_spatial_gain":gains},indent=2))

if __name__=="__main__":main()
