#!/usr/bin/env python3
"""Create public aggregate tables and paired-bootstrap intervals from private evaluator output."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from lcr.analysis import paired_bootstrap_delta
def main() -> None:
 p=argparse.ArgumentParser();p.add_argument("--predictions",type=Path,required=True,help="private long CSV: patient_id,seed,fold,horizon,arm,target,prediction");a=p.parse_args()
 x=pd.read_csv(a.predictions);required={"patient_id","seed","fold","horizon","arm","target","prediction"}
 if not required <= set(x):raise ValueError(f"missing {required-set(x)}")
 rows=[];cis=[]
 for keys,g in x.groupby(["seed","horizon","arm"]):
  y=g.target.to_numpy();q=g.prediction.to_numpy();r2=1-((y-q)**2).sum()/max(((y-y.mean())**2).sum(),1e-8)
  rows.append(dict(seed=keys[0],horizon=keys[1],arm=keys[2],n=len(g),r2=r2,mae=abs(y-q).mean()))
 for seed in sorted(x.seed.unique()):
  for horizon in sorted(x.horizon.unique()):
   z=x[(x.seed==seed)&(x.horizon==horizon)].pivot(index="patient_id",columns="arm",values=["target","prediction"])
   target=z["target"].iloc[:,0].to_numpy()
   for hist,cur in (("F2","F1"),("F4","F3")):
    if {hist,cur} <= set(z["prediction"]):
     point,lo,hi=paired_bootstrap_delta(target,z["prediction"][hist].to_numpy(),z["prediction"][cur].to_numpy())
     cis.append(dict(seed=seed,horizon=horizon,comparison=f"{hist}-{cur}",delta_r2=point,ci_low=lo,ci_high=hi))
 out=ROOT/"publication";out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/"aggregate.csv",index=False);pd.DataFrame(cis).to_csv(out/"paired_bootstrap.csv",index=False)
if __name__=="__main__":main()
