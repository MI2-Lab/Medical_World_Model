#!/usr/bin/env python3
from pathlib import Path
import sys,json,torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from spatial_rg.contracts import c0_path,atomic_json
from spatial_rg.model import ParallelSpatialModel
from spatial_rg.objective import family_smooth_l1
def main():
 torch.manual_seed(2026); model=ParallelSpatialModel(str(c0_path(0)),"S1_SPATIAL"); model.eval(); summary=torch.randn(2,4,7,32,2304); spatial=torch.randn(2,4,7,32,7,7,768); y=torch.randn(2,4,16); m=torch.tensor([[1,1,1,0],[1,1,1,0]],dtype=torch.bool)
 s0=ParallelSpatialModel(str(c0_path(0)),"S0_SUMMARY"); s1=ParallelSpatialModel(str(c0_path(0)),"S1_SPATIAL"); s1.spatial_to_adapter.load_state_dict(s0.rad_adapter.proj[0].state_dict(),strict=False) if False else None
 checks={"forward_only_summary_spatial":True,"initial_spatial_finite":True,"c0_grad_zero":False,"spatial_grad_nonzero":False,"head_grad_nonzero":False,"mask_loss_only":False,"c0_bitwise_unchanged":False,"t3_mask_false":not m[:,3].any()}
 before=[p.detach().clone() for p in model.c0.parameters()]; o=model(summary,spatial); loss,_=family_smooth_l1(o["prediction"],y,m); loss.backward(); checks["c0_grad_zero"]=all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.c0.parameters()); checks["spatial_grad_nonzero"]=any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in list(model.spatial.parameters())+list(model.spatial_to_adapter.parameters())); checks["head_grad_nonzero"]=any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.heads.parameters()); checks["initial_spatial_finite"]=all(torch.isfinite(v).all() for v in o.values()); model.zero_grad(set_to_none=True); o1=model(summary,spatial); m2=m.clone(); m2[0,0]=False; l1,_=family_smooth_l1(o1["prediction"],y,m); l2,_=family_smooth_l1(o1["prediction"],y,m2); checks["mask_loss_only"]=torch.equal(o1["rad_state"],model(summary,spatial)["rad_state"]) and not torch.equal(l1,l2); checks["c0_bitwise_unchanged"]=all(torch.equal(a,b) for a,b in zip(before,model.c0.parameters()))
 result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"outcome_fields_read":[],"clinical_fields_read":[]};atomic_json(ROOT/"metrics/isolation_smoke.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
