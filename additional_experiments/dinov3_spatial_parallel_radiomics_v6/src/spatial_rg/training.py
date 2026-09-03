from __future__ import annotations
import json,random
from pathlib import Path
import numpy as np, torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from .contracts import ROOT,TARGET_DIR,V5_STATES,c0_path,folds,protocol,sha256_file,atomic_json
from .data import Dataset,Targets
from .model import ParallelSpatialModel
from .objective import family_smooth_l1

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def ids_for(fold,split,target):
 f=folds(); ids=set(f.loc[(f.fold==fold)&(f.split==split),"patient_id"].astype(str)); return tuple(sorted(ids&set(target.patient_id)))
def loader(ids,path,spatial,shuffle,seed):return DataLoader(Dataset(ids,path,spatial),batch_size=4 if spatial else 32,shuffle=shuffle,drop_last=False,num_workers=0,pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(seed))
def dev(d):return torch.device(d if d!="cuda" or torch.cuda.is_available() else "cpu")
def run_epoch(model,dl,device,opt=None,accum=1):
 train=opt is not None; model.train(train); tot=0.; n=0; chunks=[]
 if train:opt.zero_grad(set_to_none=True)
 for bi,b in enumerate(dl):
  x=b["summary"].to(device); st=b.get("spatial_tokens"); st=st.to(device) if st is not None else None; y=b["target"].to(device); m=b["target_mask"].to(device)
  with torch.set_grad_enabled(train),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
   o=model(x,st); loss,_=family_smooth_l1(o["prediction"],y,m); scaled=loss/accum
  if train:
   scaled.backward()
   if (bi+1)%accum==0 or bi==len(dl)-1:
    params=model.rad_parameters(); clip_grad_norm_(params,5.); opt.step();opt.zero_grad(set_to_none=True)
  if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
  tot+=float(loss.detach())*len(x); n+=len(x); chunks.append(o["rad_state"].detach().float().cpu())
 state=torch.cat(chunks).reshape(-1,64); return {"loss":tot/n,"state_mean_sd":float(state.std(0,unbiased=False).mean()),"samples":n}

def train_cell(fold,seed,arm,device,checkpoint_root):
 cfg=protocol(); seed_all(seed+fold); target_path=TARGET_DIR/f"fold_{fold}_targets.private.npz"; target=Targets(target_path); train=ids_for(fold,"train",target); val=ids_for(fold,"val",target); model=ParallelSpatialModel(str(c0_path(fold)),arm).to(dev(device)); initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; physical=arm=="S1_SPATIAL"; tr=loader(train,target_path,physical,True,seed+fold); va=loader(val,target_path,physical,False,seed+fold); cell=Path(checkpoint_root)/f"seed{seed}_fold{fold}_{arm}"; cell.mkdir(parents=True,exist_ok=True); torch.save({"model_state":initial_state},cell/"initial.private.pt"); history=[]
 model.freeze_rad(); opt=torch.optim.AdamW(model.heads.parameters(),lr=cfg["training"]["head_lr"],weight_decay=cfg["training"]["weight_decay"]); best=None; score=1e30; stale=0
 for e in range(cfg["training"]["head_epochs"]):
  a=run_epoch(model,tr,model.c0.pos.device if False else dev(device),opt); b=run_epoch(model,va,dev(device)); history.append({"stage":"head","epoch":e,"train":a,"val":b});
  if b["loss"]<score and b["state_mean_sd"]>=cfg["training"]["minimum_state_sd"]:score=b["loss"];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
  else:stale+=1
  if stale>=cfg["training"]["head_patience"]:break
 if best is None:raise RuntimeError("head warmup no feasible checkpoint")
 model.load_state_dict(best); model.unfreeze_rad(); params=[{"params":model.rad_adapter.parameters(),"lr":cfg["training"]["summary_lr"]},{"params":model.rad_branch.parameters(),"lr":cfg["training"]["branch_lr"]},{"params":model.heads.parameters(),"lr":cfg["training"]["head_lr"]}]
 if model.spatial is not None:params += [{"params":model.spatial.parameters(),"lr":cfg["training"]["spatial_lr"]},{"params":model.spatial_to_adapter.parameters(),"lr":cfg["training"]["spatial_lr"]}]
 opt=torch.optim.AdamW(params,weight_decay=cfg["training"]["weight_decay"]); best=None; score=1e30; stale=0
 for e in range(cfg["training"]["epochs"]):
  a=run_epoch(model,tr,dev(device),opt, cfg["training"]["accumulation_spatial"] if physical else 1); b=run_epoch(model,va,dev(device)); history.append({"stage":"full","epoch":e,"train":a,"val":b});
  if b["loss"]<score and b["state_mean_sd"]>=cfg["training"]["minimum_state_sd"]:score=b["loss"];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
  else:stale+=1
  if e+1>=cfg["training"]["minimum_epochs"] and stale>=cfg["training"]["patience"]:break
 if best is None:raise RuntimeError("full training no feasible checkpoint")
 model.load_state_dict(best); cp=cell/"selected.private.pt"; torch.save({"model_state":best,"arm":arm,"seed":seed,"fold":fold,"c0_checkpoint_sha256":sha256_file(c0_path(fold)),"spatial_cache":arm=="S1_SPATIAL","radiomics_weight":1.0,"jepa_weight":0.,"sigreg_weight":0.,"ftv_weight":0.,"outcome_fields_read":[],"clinical_fields_read":[]},cp); atomic_json(cell/"history.private.json",{"history":history}); atomic_json(cell/"cell_complete.private.json",{"status":"COMPLETE","arm":arm,"seed":seed,"fold":fold,"c0_checkpoint_sha256":sha256_file(c0_path(fold)),"checkpoint_sha256":sha256_file(cp),"c0_trainable":False,"outcome_fields_read":[],"clinical_fields_read":[]}); return {"status":"COMPLETE","fold":fold,"seed":seed,"arm":arm}

@torch.inference_mode()
def export_cell(fold,seed,arm,device):
 target=Targets(TARGET_DIR/f"fold_{fold}_targets.private.npz"); f=folds(); ids=tuple(sorted(f.loc[f.fold==fold,"patient_id"].astype(str))); physical=arm=="S1_SPATIAL"; model=ParallelSpatialModel(str(c0_path(fold)),arm); cell=ROOT/"checkpoints/pilot"/f"seed{seed}_fold{fold}_{arm}"; payload=torch.load(cell/"selected.private.pt",map_location="cpu",weights_only=False); model.load_state_dict(payload["model_state"]); model.to(dev(device)).eval(); dl=loader(ids,TARGET_DIR/f"fold_{fold}_targets.private.npz",physical,False,0); outputs=[]; initials=[]
 initial=ParallelSpatialModel(str(c0_path(fold)),arm); initial_payload=torch.load(cell/"initial.private.pt",map_location="cpu",weights_only=False); initial.load_state_dict(initial_payload["model_state"],strict=True); initial=initial.to(dev(device)).eval();
 for b in dl:
  x=b["summary"].to(dev(device)); st=b.get("spatial_tokens"); st=st.to(dev(device)) if st is not None else None; o=model(x,st); oi=initial(x,st); outputs.append((o,oi))
 c0=np.concatenate([o["c0_state"].cpu().numpy() for o,i in outputs]); rad=np.concatenate([o["rad_state"].cpu().numpy() for o,i in outputs]); init=np.concatenate([i["rad_state"].cpu().numpy() for o,i in outputs]); pred=np.concatenate([o["prediction"].cpu().numpy() for o,i in outputs]); initial_parallel=np.concatenate([np.concatenate((i["c0_state"].cpu().numpy(),i["rad_state"].cpu().numpy()),-1) for o,i in outputs]); parallel=np.concatenate([np.concatenate((o["c0_state"].cpu().numpy(),o["rad_state"].cpu().numpy()),-1) for o,i in outputs]); out=ROOT/"features/private/states"/f"seed{seed}_fold{fold}_{arm}.npz"; out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,patient_id=np.asarray(ids),c0_state=c0,rad_initial_state=init,rad_state=rad,parallel_initial_state=initial_parallel,parallel_state=parallel,radiomics_prediction=pred,c0_checkpoint_sha256=np.asarray(sha256_file(c0_path(fold))),rad_checkpoint_sha256=np.asarray(sha256_file(cell/"selected.private.pt"))); return {"status":"COMPLETE","path":str(out),"sha256":sha256_file(out)}
