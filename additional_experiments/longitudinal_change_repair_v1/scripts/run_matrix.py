#!/usr/bin/env python3
"""Run the locked cells sequentially. Stage F is refused until an explicit Stage-R gate artifact exists."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main() -> None:
 p=argparse.ArgumentParser();p.add_argument("--stage",choices=("r","f"),required=True);p.add_argument("--device",default="cuda");p.add_argument("--smoke-patients",type=int);a=p.parse_args()
 if a.stage=="f":
  gate=ROOT/"private_results/stage_r_gate.json"
  if not gate.exists() or not json.loads(gate.read_text()).get("pass"): raise RuntimeError("Stage R gate has not passed; Stage F is prohibited")
 arms=("R0","R1","R2") if a.stage=="r" else ("F1","F2","F3","F4")
 for arm in arms:
  for seed in (2026,3026):
   for fold in range(5):
    cmd=[sys.executable,str(ROOT/"scripts/train_cell.py"),"--arm",arm,"--seed",str(seed),"--fold",str(fold),"--device",a.device]
    if a.smoke_patients: cmd += ["--smoke-patients",str(a.smoke_patients)]
    subprocess.run(cmd,check=True)
if __name__=="__main__":main()
