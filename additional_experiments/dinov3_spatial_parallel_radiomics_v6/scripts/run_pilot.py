#!/usr/bin/env python3
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda");p.add_argument("--fold",type=int);p.add_argument("--arm",choices=["S0_SUMMARY","S1_SPATIAL"]);a=p.parse_args();
 if a.fold is None or a.arm is None:raise SystemExit("run one explicit fold/arm per GPU")
 cmd=[sys.executable,str(ROOT/"scripts/train_cell.py"),"--fold",str(a.fold),"--arm",a.arm,"--device",a.device];print(subprocess.run(cmd,check=True).returncode)
