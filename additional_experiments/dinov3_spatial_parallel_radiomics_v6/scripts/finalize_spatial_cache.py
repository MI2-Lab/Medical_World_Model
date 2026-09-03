#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from spatial_rg.contracts import SPATIAL_DIR, token, atomic_json, canonical, protocol
from spatial_rg.extraction import load_entries
def main():
 p=argparse.ArgumentParser(); p.add_argument("--contract-sha256",required=True); a=p.parse_args(); entries=load_entries(); missing=[]; hashes=[]
 import numpy as np
 for pid in sorted(entries):
  q=SPATIAL_DIR/f"{token(pid)}.private.npz"
  if not q.is_file():missing.append(pid); continue
  with np.load(q,allow_pickle=False) as z:
   if set(z.files)!={"patient_token","spatial_tokens","source_cache_sha256","contract_sha256"} or str(z["patient_token"].item())!=token(pid) or tuple(z["spatial_tokens"].shape)!=(4,7,32,7,7,768) or z["spatial_tokens"].dtype!=np.float16 or not np.isfinite(z["spatial_tokens"]).all() or str(z["contract_sha256"].item())!=a.contract_sha256: raise ValueError(f"spatial cache contract failed: {pid}")
  from spatial_rg.contracts import sha256_file
  hashes.append(sha256_file(q))
 result={"status":"PASS" if not missing else "FAIL","patients":len(entries),"missing":len(missing),"spatial_shape":[4,7,32,7,7,768],"contract_sha256":a.contract_sha256,"ordered_hashes_sha256":canonical(hashes),"outcome_fields_read":[],"clinical_fields_read":[]}; atomic_json(ROOT/"spatial_cache_check.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
