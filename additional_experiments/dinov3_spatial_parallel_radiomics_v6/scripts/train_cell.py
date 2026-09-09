#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from spatial_rg.training import train_cell,export_cell
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--fold",type=int,required=True);p.add_argument("--arm",choices=["S0_SUMMARY","S1_SPATIAL"],required=True);p.add_argument("--seed",type=int,default=2026);p.add_argument("--device",default="cuda");p.add_argument("--workers",type=int,default=0);a=p.parse_args(); print(json.dumps(train_cell(a.fold,a.seed,a.arm,a.device,ROOT/"checkpoints/pilot"),indent=2)); print(json.dumps(export_cell(a.fold,a.seed,a.arm,a.device),indent=2))
