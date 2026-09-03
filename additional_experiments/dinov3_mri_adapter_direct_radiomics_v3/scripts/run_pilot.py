#!/usr/bin/env python3
"""Run/resume the outcome-blind 15-cell V3 weight-screen pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from dinov3_rg.contracts import FOLDS, PILOT_ARMS, PILOT_SEED, atomic_json, file_sha256  # noqa: E402
from dinov3_rg.data import validate_state_archive  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--num-shards",type=int,default=1)
    parser.add_argument("--shard-index",type=int,default=0)
    args=parser.parse_args(); RepresentationReadSentinel().install()
    for path in (ROOT/"metrics/preflight.json",ROOT/"metrics/smoke.json"):
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8"))["status"]!="PASS":
            raise SystemExit(f"pilot prerequisite failed: {path.name}")
    if args.num_shards<1 or not 0<=args.shard_index<args.num_shards:
        raise SystemExit("invalid shard contract")
    cells=[(fold,arm) for fold in FOLDS for arm in PILOT_ARMS]
    selected=cells[args.shard_index::args.num_shards]
    for fold,arm in selected:
        tag=f"seed{PILOT_SEED}_fold{fold}_{arm}"
        complete=ROOT/f"checkpoints/pilot/{tag}/cell_complete.private.json"
        failed=ROOT/f"checkpoints/pilot/{tag}/cell_failed.private.json"
        state=ROOT/f"features/private/pilot_states/{tag}_states.private.npz"
        if complete.is_file() and state.is_file():
            validate_state_archive(state); print({"cell":tag,"status":"REUSED"},flush=True); continue
        if failed.is_file():
            print({"cell":tag,"status":"FAILED_REUSED"},flush=True); continue
        prior_failures=sorted((ROOT/"checkpoints/pilot").glob(f"seed{PILOT_SEED}_fold*_{arm}/cell_failed.private.json"))
        if prior_failures:
            print({"cell":tag,"status":"SKIPPED_AFTER_ARM_FAILURE",
                   "trigger":prior_failures[0].parent.name},flush=True); continue
        command=[sys.executable,str(ROOT/"scripts/train_cell.py"),"--phase","pilot",
                 "--seed",str(PILOT_SEED),"--fold",str(fold),"--arm",arm,
                 "--device",args.device,"--workers",str(args.workers)]
        result=subprocess.run(command,check=False)
        print({"cell":tag,"status":"COMPLETE" if result.returncode==0 else "FAILED"},flush=True)
    if args.num_shards>1:
        print({"status":"SHARD_COMPLETE","cells":len(selected),"shard_index":args.shard_index}); return
    artifacts={}; statuses={}
    for fold,arm in cells:
        tag=f"seed{PILOT_SEED}_fold{fold}_{arm}"; cell=ROOT/f"checkpoints/pilot/{tag}"
        complete=cell/"cell_complete.private.json"; failed=cell/"cell_failed.private.json"
        state=ROOT/f"features/private/pilot_states/{tag}_states.private.npz"
        if complete.is_file() and state.is_file():
            validate_state_archive(state); payload=json.loads(complete.read_text(encoding="utf-8"))
            statuses[tag]="COMPLETE"; artifacts[tag]={"completion_sha256":file_sha256(complete),"state_sha256":file_sha256(state)}
        elif failed.is_file():
            statuses[tag]="NO_FEASIBLE_CHECKPOINT"; artifacts[tag]={"failure_sha256":file_sha256(failed)}
        else:
            triggers=sorted((ROOT/"checkpoints/pilot").glob(f"seed{PILOT_SEED}_fold*_{arm}/cell_failed.private.json"))
            if not triggers: raise RuntimeError(f"pilot execution incomplete: {tag}")
            statuses[tag]="SKIPPED_AFTER_ARM_FAILURE"
            artifacts[tag]={"trigger_failure_sha256":file_sha256(triggers[0])}
    paired={}
    for fold in FOLDS:
        payloads=[]
        for arm in PILOT_ARMS:
            path=ROOT/f"checkpoints/pilot/seed{PILOT_SEED}_fold{fold}_{arm}/cell_complete.private.json"
            if path.is_file(): payloads.append(json.loads(path.read_text(encoding="utf-8")))
        paired[str(fold)]=bool(
            len(payloads)==len(PILOT_ARMS)
            and len({x["initialization_sha256"] for x in payloads})==1
            and len({x["radiomics_head_initialization_sha256"] for x in payloads})==1
            and len({x["base_checkpoint_sha256"] for x in payloads})==1
            and len({x["train_patient_order_sha256"] for x in payloads})==1
            and all(float(x["ftv_weight"])==0.0 for x in payloads)
        )
    output={"schema_version":1,"status":"COMPLETE","cells":len(cells),"statuses":statuses,
            "paired_fold_checks":paired,"artifacts":artifacts,
            "outcome_fields_read":[],"clinical_fields_read":[]}
    atomic_json(ROOT/"metrics/pilot_execution.json",output); print(output)


if __name__=="__main__": main()
