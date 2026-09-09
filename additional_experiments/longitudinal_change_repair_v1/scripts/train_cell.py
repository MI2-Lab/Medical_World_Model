#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parents[0]/"radiomics_next_change"/"src")]
from lcr.training import train_cell
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument("--arm",required=True);p.add_argument("--fold",type=int,choices=range(5),required=True);p.add_argument("--seed",type=int,choices=(2026,3026),required=True);p.add_argument("--config",type=Path,default=ROOT/"configs/experiment.yaml");p.add_argument("--device",default="cuda");p.add_argument("--smoke-patients",type=int);p.add_argument("--epochs",type=int);a=p.parse_args(); print(train_cell(a.config,a.arm,a.fold,a.seed,a.device,a.smoke_patients,a.epochs))
if __name__=="__main__":main()
