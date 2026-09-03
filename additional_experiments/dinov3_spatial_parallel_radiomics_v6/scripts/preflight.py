#!/usr/bin/env python3
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from spatial_rg.contracts import SUMMARY_DIR,TARGET_DIR,V2,V4,V5,SUMMARY_SHAPE,SPATIAL_SHAPE,c0_path,folds,atomic_json,sha256_file
from spatial_rg.data import Targets,load_summary
def main():
 checks={"summary_947":len(list(SUMMARY_DIR.glob("*.private.npz")))==947,"target_5":True,"target_t3_false":True,"c0_5":all(c0_path(f).is_file() for f in range(5)),"fold_4040":len(folds())==4040,"parent_v2":V2.is_dir(),"parent_v4":V4.is_dir(),"parent_v5":V5.is_dir(),"target_shape":True,"summary_roundtrip":False}
 for f in range(5):
  p=TARGET_DIR/f"fold_{f}_targets.private.npz"
  try:
   t=Targets(p); checks["target_t3_false"] &= not t.mask[:,3].any(); checks["target_shape"] &= t.target.shape==(375,4,16)
  except Exception:checks["target_5"]=False
 try:
  pid=str(folds().iloc[0].patient_id); s=load_summary(pid); checks["summary_roundtrip"]=s.shape==SUMMARY_SHAPE
 except Exception:checks["summary_roundtrip"]=False
 result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"counts":{"summaries":len(list(SUMMARY_DIR.glob("*.private.npz"))),"targets":len(list(TARGET_DIR.glob("fold_*_targets.private.npz"))),"c0":sum(c0_path(f).is_file() for f in range(5))},"target_sha256":{str(f):sha256_file(TARGET_DIR/f"fold_{f}_targets.private.npz") for f in range(5) if (TARGET_DIR/f"fold_{f}_targets.private.npz").is_file()},"outcome_fields_read":[],"clinical_fields_read":[]};atomic_json(ROOT/"metrics/preflight.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
