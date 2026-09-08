#!/usr/bin/env python3
"""Aggregate Stage-R validation evidence and write the hard Stage-F gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main() -> None:
 p=argparse.ArgumentParser();p.add_argument("--summary",type=Path,required=True,help="CSV/JSON produced by the evaluator with seed, arm, validation_r2, current_only_r2");a=p.parse_args()
 import pandas as pd
 frame=pd.read_csv(a.summary) if a.summary.suffix==".csv" else pd.DataFrame(json.loads(a.summary.read_text()))
 required={"seed","arm","validation_r2","current_only_r2"}
 if not required <= set(frame): raise ValueError(f"missing {required-set(frame)}")
 r2=frame[frame.arm=="R2"].groupby("seed")
 checks=[]
 for seed, group in r2:
  checks.append({"seed":int(seed),"r2_positive":bool((group.validation_r2>0).any()),"history_improves":bool((group.validation_r2-group.current_only_r2>0).all())})
 gate={"pass":len(checks)==2 and all(x["r2_positive"] and x["history_improves"] for x in checks),"checks":checks,"source":str(a.summary)}
 out=ROOT/"private_results/stage_r_gate.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(gate,ensure_ascii=False,indent=2));print(out)
if __name__=="__main__":main()
