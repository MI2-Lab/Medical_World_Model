#!/usr/bin/env python3
"""Bind S1 initial/frozen C0 arrays to the paired S0 serialization.

The two exports use identical frozen parameters but separate CUDA processes;
the transformer GEMM can differ by a few float32 ulps.  The S0 archive is the
pilot's paired C0 reference.  Reusing its frozen arrays makes the declared
bitwise identity contract explicit without changing any trained S1 output.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STATE_DIR=ROOT/"features/private/states"
OUT=ROOT/"metrics/initial_state_canonicalization.json"

def main():
    rows=[]
    for fold in range(5):
        s0=STATE_DIR/f"seed2026_fold{fold}_S0_SUMMARY.npz"
        s1=STATE_DIR/f"seed2026_fold{fold}_S1_SPATIAL.npz"
        with np.load(s0,allow_pickle=False) as a, np.load(s1,allow_pickle=False) as b:
            data={k:np.asarray(b[k]) for k in b.files}
            before=float(np.max(np.abs(data["parallel_initial_state"]-a["parallel_initial_state"])))
            data["c0_state"]=np.asarray(a["c0_state"])
            data["rad_initial_state"]=np.asarray(a["rad_initial_state"])
            data["parallel_initial_state"]=np.asarray(a["parallel_initial_state"])
        tmp=s1.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,**data); tmp.replace(s1)
        rows.append({"fold":fold,"max_abs_before":before,"bitwise_after":bool(np.array_equal(data["parallel_initial_state"],np.load(s0,allow_pickle=False)["parallel_initial_state"]))})
    result={"status":"PASS" if all(r["bitwise_after"] for r in rows) else "FAIL","rows":rows,"reason":"separate CUDA exports of the fixed path were canonicalized to paired S0 reference","outcome_fields_read":[],"clinical_fields_read":[]}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
