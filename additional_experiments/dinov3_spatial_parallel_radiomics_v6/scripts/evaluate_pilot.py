#!/usr/bin/env python3
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from spatial_rg.contracts import TARGET_DIR,ROOT as ER,folds,atomic_json
from spatial_rg.data import Targets
VISITS=(0,1,2); ALPHAS=(.1,1.,10.,100.)
def corr(y,p):
 k=np.isfinite(y)&np.isfinite(p)
 return float(spearmanr(y[k],p[k]).statistic) if k.sum()>10 and np.unique(y[k]).size>1 and np.unique(p[k]).size>1 else np.nan
def load_state(fold,arm):
 p=ER/"features/private/states"/f"seed2026_fold{fold}_{arm}.npz"
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def fit_probe(x,y,mask,ids,fold,visit):
 f=folds(); current=f[f.fold==fold]; tr=set(current.loc[current.split=="train","patient_id"]); va=set(current.loc[current.split=="val","patient_id"]); te=set(current.loc[current.split=="test","patient_id"]); ids=np.asarray(ids).astype(str); m=np.asarray(mask[:,visit],bool); tr=np.array([i in tr for i in ids])&m; va=np.array([i in va for i in ids])&m; te=np.array([i in te for i in ids])&m
 def xx(z):
  oh=np.zeros((len(z),3));oh[:,visit]=1;return np.c_[z,oh]
 scaler=StandardScaler().fit(xx(x[tr,visit])); best=(None,1e99)
 for a in ALPHAS:
  model=Ridge(alpha=a).fit(scaler.transform(xx(x[tr,visit])),y[tr,visit]); mse=np.mean((model.predict(scaler.transform(xx(x[va,visit]))) - y[va,visit])**2)
  if mse<best[1]-1e-12 or abs(mse-best[1])<=1e-12 and (best[0] is None or a>best[0]):best=(a,mse)
 trva=np.array([i in (set(current.loc[current.split=="train","patient_id"])|set(current.loc[current.split=="val","patient_id"])) for i in ids])&m; scaler=StandardScaler().fit(xx(x[trva,visit])); model=Ridge(alpha=best[0]).fit(scaler.transform(xx(x[trva,visit])),y[trva,visit]); return corr(y[te,visit],model.predict(scaler.transform(xx(x[te,visit]))))
def probe(x,target,fold):
 vals=[]; fam=[]
 for pc in range(16):
  for v in VISITS:vals.append(fit_probe(x,target.target[:,:,pc],target.mask,target.patient_id,fold,v)); fam.append((pc//4,vals[-1]))
 return float(np.nanmean(vals)),{str(f):float(np.nanmean([z for g,z in fam if g==f])) for f in range(4)}
def direct(pred,target,fold,ids):
 f=folds();test=set(f.loc[f.fold==fold,"patient_id"]); pos={str(p):i for i,p in enumerate(ids)};vals=[]
 for pc in range(16):
  for v in VISITS:
   keep=[i for i,p in enumerate(target.patient_id) if p in test and target.mask[i,v] and p in pos]; vals.append(corr(target.target[keep,v,pc],pred[[pos[target.patient_id[i]] for i in keep],v,pc]))
 return float(np.nanmean(vals))
def ftv(x,target,fold):
 return float(np.nanmean([fit_probe(x,target.ftv,target.ftv_mask,target.patient_id,fold,v) for v in VISITS]))
def main():
 rows=[]
 for fold in range(5):
  t=Targets(TARGET_DIR/f"fold_{fold}_targets.private.npz"); s0=load_state(fold,"S0_SUMMARY"); s1=load_state(fold,"S1_SPATIAL"); pos={str(p):i for i,p in enumerate(s0["patient_id"])}; ids=t.patient_id; ix=[pos[str(p)] for p in ids]; a={k:s0[k][ix] for k in ("rad_initial_state","rad_state","parallel_initial_state","parallel_state","radiomics_prediction")}; b={k:s1[k][ix] for k in ("rad_initial_state","rad_state","parallel_initial_state","parallel_state","radiomics_prediction")};
  if not np.array_equal(a["parallel_initial_state"],b["parallel_initial_state"]):raise AssertionError(f"initial state mismatch fold {fold}")
  pi,fi=probe(a["rad_initial_state"],t,fold); p0,f0=probe(a["rad_state"],t,fold); p1,f1=probe(b["rad_state"],t,fold); rows.append({"fold":fold,"initial_probe":pi,"summary_probe":p0,"spatial_probe":p1,"spatial_gain_vs_initial":p1-pi,"spatial_gain_vs_summary":p1-p0,"summary_direct_head":direct(a["radiomics_prediction"],t,fold,ids),"spatial_direct_head":direct(b["radiomics_prediction"],t,fold,ids),"initial_ftv":ftv(a["parallel_initial_state"],t,fold),"summary_ftv":ftv(a["parallel_state"],t,fold),"spatial_ftv":ftv(b["parallel_state"],t,fold),"initial_f0":fi["0"],"initial_f1":fi["1"],"initial_f2":fi["2"],"initial_f3":fi["3"],"summary_f0":f0["0"],"summary_f1":f0["1"],"summary_f2":f0["2"],"summary_f3":f0["3"],"spatial_f0":f1["0"],"spatial_f1":f1["1"],"spatial_f2":f1["2"],"spatial_f3":f1["3"]})
 mean={k:float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k!="fold"}; fam_abs=[mean[f"spatial_f{i}"] for i in range(4)]; fam_gain=[mean[f"spatial_f{i}"]-mean[f"summary_f{i}"] for i in range(4)]; cfg=json.loads((ROOT/"configs/protocol.json").read_text())["gates"]; checks={"direct_head":mean["spatial_direct_head"]>=cfg["direct_head"],"probe_absolute":mean["spatial_probe"]>=cfg["probe_absolute"],"gain_initial":mean["spatial_gain_vs_initial"]>=cfg["gain_vs_initial"],"gain_summary":mean["spatial_gain_vs_summary"]>=cfg["spatial_gain_vs_summary"],"positive_folds":sum(r["spatial_gain_vs_summary"]>0 for r in rows)>=cfg["positive_folds"],"morphology_absolute":fam_abs[0]>=cfg["family_absolute"],"texture_absolute":fam_abs[3]>=cfg["family_absolute"],"morphology_spatial_gain":fam_gain[0]>=cfg["family_spatial_gain"],"texture_spatial_gain":fam_gain[3]>=cfg["family_spatial_gain"],"intensity_retained":fam_gain[1]>=-.02,"kinetic_retained":fam_gain[2]>=-.02,"ftv_static_retained":mean["spatial_ftv"]-mean["initial_ftv"]>=-cfg["max_ftv_drop"]}
 result={"status":"PASS" if all(checks.values()) else "FAIL","folds":rows,"mean":mean,"family_absolute":fam_abs,"family_gain_vs_summary":fam_gain,"checks":checks,"outcome_fields_read":[],"clinical_fields_read":[]}; atomic_json(ROOT/"metrics/pilot_gate.json",result); atomic_json(ROOT/"decision.json",{"decision":"PILOT_LOCKED" if result["status"]=="PASS" else "SPATIAL_RADIOMICS_NOT_TRANSFERRED","pilot_gate":result["status"],"pCR_evaluation":"LOCKED","outcome_fields_read":[],"clinical_fields_read":[]}); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
