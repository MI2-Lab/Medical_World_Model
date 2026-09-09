from __future__ import annotations
import numpy as np, pandas as pd
from scipy import ndimage
from .contracts import RAW,ROIS,SUMMARY,SPATIAL,V6_TARGET,V6_STATES,token,folds,VISITS,VISIT_NAMES

def _morph_descriptor(mask):
    m=np.asarray(mask,bool); out=np.full(6,np.nan,float)
    xyz=np.argwhere(m).astype(float)
    if len(xyz)<64 or np.count_nonzero(m.any(axis=(1,2)))<3:return out
    xyz[:,0]*=2.; xyz[:,1:]*=.9
    cov=np.cov(xyz,rowvar=False); eig=np.sort(np.maximum(np.linalg.eigvalsh(cov),0))[::-1]
    if eig[0]>0:
        out[0]=np.sqrt(eig[1]/eig[0]);out[1]=np.sqrt(eig[2]/eig[0])
    spans=np.ptp(xyz,axis=0)+np.asarray([2.,.9,.9])
    out[2]=float(len(xyz)/(np.prod(spans)/np.prod([2.,.9,.9])))
    areas=m.sum(axis=(1,2)); areas=areas[areas>0]
    if len(areas)>=3 and areas.mean()>0:out[3]=float(areas.std(ddof=0)/areas.mean())
    boundary=m & ~ndimage.binary_erosion(m,structure=np.ones((1,3,3),bool),border_value=0)
    out[4]=float(boundary.sum()/m.sum())
    out[5]=float(np.count_nonzero(m.any(axis=(1,2))))
    return out

def load_population():
    rows=[]; names=None
    for rp in sorted(RAW.glob("*.private.npz")):
        with np.load(rp,allow_pickle=False) as z:
            pid=str(z["patient_id"].item()); fn=tuple(z["feature_name"].astype(str)); val=np.asarray(z["value"],float); vv=np.asarray(z["variant_valid"],bool)
        with np.load(ROIS/f"{token(pid)}.private.npz",allow_pickle=False) as z:
            roi=np.asarray(z["roi_mask"],bool); valid_source=np.asarray(z["local_valid_source_mask"],bool); rm=np.asarray(z["radiomics_mask"],bool); vol=np.asarray(z["local_mask_volume_mm3"],float); ftv=np.asarray(z["ftv"],float); ftvm=np.asarray(z["ftv_mask"],bool)
        if names is None:names=fn
        if fn!=names or val.shape!=(4,3,len(fn)) or roi.shape!=(4,32,72,72):raise ValueError("V2 raw/ROI contract failed")
        variants=np.full((4,3,6),np.nan,float)
        for v in VISITS:
            base=roi[v]&valid_source[v]
            original=base; eroded=ndimage.binary_erosion(base,structure=np.ones((1,3,3),bool),border_value=0)&valid_source[v]; dilated=ndimage.binary_dilation(base,structure=np.ones((1,3,3),bool),border_value=0)&valid_source[v]
            for k,m in enumerate((original,eroded,dilated)):variants[v,k]=_morph_descriptor(m)
        rows.append((pid,val,vv,rm,vol,ftv,ftvm,variants))
    if len(rows)!=375:raise ValueError(f"expected 375 patients, got {len(rows)}")
    return {
      "ids":np.asarray([r[0] for r in rows],str),"names":names,
      "values":np.stack([r[1] for r in rows]),"variant_valid":np.stack([r[2] for r in rows]),
      "rm":np.stack([r[3] for r in rows]),"volume":np.stack([r[4] for r in rows]),"ftv":np.stack([r[5] for r in rows]),"ftvm":np.stack([r[6] for r in rows]),"morph":np.stack([r[7] for r in rows])}

def workbook(ids):
    book=pd.read_excel("/data/data/Breast_Cancer/I-SPY2/Multi-feature-MRI-NACT-Data.xlsx",sheet_name="datawith4visits")
    by={str(int(x)).zfill(6):r for x,r in zip(book["CLINICAL-TRIAL-SUBJECT-ID"],book.to_dict("records"))}
    out=np.full((len(ids),4,2),np.nan,float)
    for i,p in enumerate(ids):
        r=by.get(str(p).split("-")[-1])
        if r is None:raise ValueError(f"workbook join missing {p}")
        for v in range(4):
            for j,key in enumerate(("LD","SPHERICITY")):
                x=r[f"{key}_T{v}"];out[i,v,j]=float(x) if pd.notna(x) else np.nan
    return out

def load_c0(ids):
    out=[]
    for f in range(5):
        with np.load(V6_STATES/f"seed2026_fold{f}_S0_SUMMARY.npz",allow_pickle=False) as z:
            pos={str(p):i for i,p in enumerate(z["patient_id"].astype(str))};out.append(np.stack([z["c0_state"][pos[p]] for p in ids]))
    return np.stack(out)

def load_input_features(ids):
    glob=[];spat=[]
    for p in ids:
        with np.load(SUMMARY/f"{token(p)}.private.npz",allow_pickle=False) as z: glob.append(np.asarray(z["summary"],np.float32))
        with np.load(SPATIAL/f"{token(p)}.private.npz",allow_pickle=False) as z: spat.append(np.asarray(z["spatial_tokens"],np.float32))
    glob=np.stack(glob);spat=np.stack(spat)
    # Per visit, preserve channel/slice aggregation and patch geometry.
    g=glob.mean(axis=(2,3))
    full=spat.mean(axis=(2,3)).reshape(len(ids),4,-1)
    center=spat[:,:,:,:,2:5,2:5,:].mean(axis=(2,3,4,5)); border=np.concatenate((spat[:,:,:,:, :2,:,:].reshape(len(ids),4,-1,768),spat[:,:,:,:, 5:,:,:].reshape(len(ids),4,-1,768),spat[:,:,:,:,2:5,:2,:].reshape(len(ids),4,-1,768),spat[:,:,:,:,2:5,5:,:].reshape(len(ids),4,-1,768)),axis=2).mean(axis=2)
    quads=np.stack([spat[:,:,:,:,:4,:4,:].mean((2,3,4,5)),spat[:,:,:,:,:4,3:,:].mean((2,3,4,5)),spat[:,:,:,:,3:,:4,:].mean((2,3,4,5)),spat[:,:,:,:,3:,3:,:].mean((2,3,4,5))],axis=2)
    cb=np.concatenate((center[:,:,None,:],border[:,:,None,:],quads),axis=2).reshape(len(ids),4,-1)
    rng=np.random.default_rng(20260903); perm=spat.copy()
    for i,p in enumerate(ids):
        for v in range(4):
            q=rng.permutation(49); x=perm[i,v].reshape(7,32,49,768);perm[i,v]=x[:,:,q,:].reshape(7,32,7,7,768)
    shuffled=perm.mean(axis=(2,3)).reshape(len(ids),4,-1)
    return {"C0_STATE":load_c0(ids),"GLOBAL_SUMMARY":g,"SPATIAL_FULL":full,"SPATIAL_CENTER_BORDER":cb,"SPATIAL_PERMUTED":shuffled}

def current_masks(ids):
    out=[]
    for f in range(5):
        with np.load(V6_TARGET/f"fold_{f}_targets.private.npz",allow_pickle=False) as z:
            pos={str(p):i for i,p in enumerate(z["patient_id"].astype(str))};out.append(np.stack([z["target_mask"][pos[p]] for p in ids]).astype(bool))
    return np.stack(out)
