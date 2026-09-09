#!/usr/bin/env python3
"""Evaluate one frozen cell. Optional donor map is required for matched-history counterfactuals."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT.parents[0]/"radiomics_next_change"/"src")]
from lcr.data import bundle_from_config,FTVTransform,ChangeDataset,records
from lcr.model import ChangeRepairModel
from rnc.data import split_ids

def main() -> None:
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--split",choices=("val","test"),default="test");p.add_argument("--donor-map",type=Path);p.add_argument("--device",default="cuda");a=p.parse_args()
 payload=torch.load(a.checkpoint,map_location="cpu",weights_only=True); cfg=yaml.safe_load((ROOT/"configs/experiment.yaml").read_text()); bundle=bundle_from_config(cfg); splits=split_ids(bundle,int(payload["fold"]))
 transform=FTVTransform(**payload["transform"]); ds=ChangeDataset(records(bundle,splits[a.split]),bundle.raw_radiomics,transform); loader=DataLoader(ds,batch_size=16,shuffle=False)
 m=payload["model"];model=ChangeRepairModel(m["image_channels"],m["base_channels"],m["state_dim"],m["predictor_depth"],m["predictor_heads"],m["predictor_mlp_dim"],m["dropout"]);model.load_state_dict(payload["state"]);device=torch.device(a.device if torch.cuda.is_available() else "cpu");model.to(device).eval()
 donors={}
 if a.donor_map:
  dm=pd.read_csv(a.donor_map); needed={"patient_id","donor_patient_id"}
  if not needed <= set(dm):raise ValueError("donor map requires patient_id and donor_patient_id")
  if not set(dm.patient_id.astype(str)) >= set(ds.patient_ids):raise ValueError("donor map does not cover evaluation split")
  if set(dm.donor_patient_id.astype(str))-set(ds.patient_ids):raise ValueError("donor must remain in the held-out split")
  donors=dict(zip(dm.patient_id.astype(str),dm.donor_patient_id.astype(str)))
 images={x["patient_id"]:x["image"] for x in ds}
 rows=[]
 with torch.no_grad():
  for batch in loader:
   image=batch["image"].to(device); arm=str(payload["arm"]); history=arm in {"F2","F4"};out=model(image,history=history)
   pred=out.observed_change if arm.startswith("R") else out.future_change; target=batch["change"] if arm.startswith("R") else batch["change"][:,1:];mask=batch["change_mask"] if arm.startswith("R") else batch["change_mask"][:,1:]
   for i,pid in enumerate(batch["patient_id"]):
    for h in range(pred.shape[1]):
     if bool(mask[i,h]):rows.append({"patient_id":pid,"seed":payload["seed_base"],"fold":payload["fold"],"split":a.split,"arm":arm,"condition":"true","horizon":h,"target":float(target[i,h]),"prediction":float(pred[i,h])})
    if donors and not arm.startswith("R"):
     altered=image[i:i+1].clone(); altered[:,0]=images[donors[pid]][0].to(device); altered[:,1]=image[i:i+1,1] # only prior history changes
     cf=model(altered,history=True).future_change[0]
     for h in range(2):
      if bool(mask[i,h]):rows.append({"patient_id":pid,"seed":payload["seed_base"],"fold":payload["fold"],"split":a.split,"arm":arm,"condition":"matched_history","horizon":h,"target":float(target[i,h]),"prediction":float(cf[h])})
 outdir=ROOT/"private_results/predictions";outdir.mkdir(parents=True,exist_ok=True);out=outdir/f"{payload['arm']}_seed{payload['seed_base']}_fold{payload['fold']}_{a.split}.csv";pd.DataFrame(rows).to_csv(out,index=False);print(out)
if __name__=="__main__":main()
