from __future__ import annotations
import os, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from .contracts import C1B_CACHE_MANIFEST, SPATIAL_DIR, SUMMARY_DIR, atomic_json, canonical, protocol, sha256_file, token

MEAN=(.485,.456,.406); STD=(.229,.224,.225)
def load_entries():
    f=pd.read_csv(C1B_CACHE_MANIFEST,usecols=["patient_id","cache_path","cache_sha256","input_kind"],dtype=str)
    if len(f)!=947 or f.patient_id.duplicated().any() or not f.input_kind.eq("c1b").all():raise ValueError("C1B manifest contract failed")
    return {r.patient_id:(Path(r.cache_path),r.cache_sha256) for r in f.itertuples(index=False)}

def load_dino(device):
    from transformers import AutoModel
    cfg=json_load=protocol()["dino"]
    # The V2 protocol is the source of the full artifact hashes; V6 records
    # the same revision and only adds a token-preserving output contract.
    model=AutoModel.from_pretrained(cfg["repository_id"],revision=cfg["revision"],local_files_only=True).to(device).eval().requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):raise AssertionError("DINO must be frozen")
    return model

def prepare(x):
    x=x.float().clamp(-5,5).add(5).div(10)[:,None].expand(-1,3,-1,-1)
    x=F.interpolate(x,size=(224,224),mode="bicubic",align_corners=False,antialias=True)
    mean=x.new_tensor(MEAN).view(1,3,1,1); std=x.new_tensor(STD).view(1,3,1,1); return (x-mean)/std

@torch.inference_mode()
def spatial_for_image(model, image, device, batch_size=64):
    image=np.asarray(image)
    if image.shape!=(4,7,112,176,160) or image.dtype!=np.float32:raise ValueError("C1B source shape/dtype failed")
    z,y,x=40,52,44; local=image[:,:,z:z+32,y:y+72,x:x+72]; flat=torch.from_numpy(local.reshape(-1,72,72)); chunks=[]
    for start in range(0,len(flat),batch_size):
        px=prepare(flat[start:start+batch_size]).to(device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            hidden=model(pixel_values=px).last_hidden_state
        patches=hidden[:,5:]
        if patches.shape[1]!=196:raise ValueError("DINO patch grid is not 14x14")
        patches=patches.reshape(-1,14,14,768).permute(0,3,1,2)
        pooled=F.avg_pool2d(patches,kernel_size=2,stride=2).permute(0,2,3,1)
        chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(chunks).reshape(4,7,32,7,7,768).astype(np.float16)

def write(path, pid, spatial, source_sha, contract_sha):
    path.parent.mkdir(parents=True,exist_ok=True); q=path.with_name("."+path.name+".tmp.npz")
    np.savez_compressed(q,patient_token=np.asarray(token(pid)),spatial_tokens=spatial,source_cache_sha256=np.asarray(source_sha),contract_sha256=np.asarray(contract_sha)); q.replace(path)

def run(args):
    gate=Path(__file__).resolve().parents[2]/"target_feasibility.json"
    if not gate.is_file() or json_load_file(gate).get("status")!="PASS":raise SystemExit("target feasibility must PASS before spatial extraction")
    entries=load_entries(); ids=sorted(entries); device=torch.device(args.device); model=load_dino(device); contract=canonical({"revision":protocol()["dino"]["revision"],"register_tokens":4,"patch_grid":[14,14],"pooled_grid":[7,7],"source_local_shape":[32,72,72]})
    selected=ids[args.shard_index::args.num_shards]
    if args.limit is not None: selected=selected[:args.limit]
    for i,pid in enumerate(selected):
        out=SPATIAL_DIR/f"{token(pid)}.private.npz"
        if out.exists() and not args.overwrite:continue
        source,expected=entries[pid]
        if not args.skip_source_hash and sha256_file(source)!=expected:raise ValueError("C1B source hash mismatch")
        with np.load(source,allow_pickle=False) as z:
            if str(z["patient_id"].item())!=pid:raise ValueError("C1B patient binding failed")
            spatial=spatial_for_image(model,np.asarray(z["image"]),device,args.batch_size)
        write(out,pid,spatial,expected,contract); print({"patient":i+1,"total":len(selected),"status":"WRITTEN"},flush=True)
    return {"status":"SHARD_COMPLETE","patients":len(selected),"shard_index":args.shard_index,"contract_sha256":contract}

def json_load_file(p):
    import json
    return json.loads(Path(p).read_text())
