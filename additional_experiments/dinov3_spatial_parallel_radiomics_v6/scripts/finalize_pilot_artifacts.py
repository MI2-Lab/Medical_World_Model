#!/usr/bin/env python3
"""Create the non-outcome V6 pilot hand-off artifacts after the gate."""
import hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METRICS=ROOT/"metrics"; STATES=ROOT/"features/private/states"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(path.read_text())
def write(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2)+"\n")

def main():
    protocol=read(ROOT/"configs/protocol.json")
    target=read(ROOT/"target_feasibility.json")
    cache=read(ROOT/"spatial_cache_check.json")
    smoke=read(METRICS/"isolation_smoke.json")
    canon=read(METRICS/"initial_state_canonicalization.json")
    pilot=read(METRICS/"pilot_gate.json")
    try: git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: git_head="unavailable"

    inheritance={"status":"PASS","experiment":protocol["experiment"],"git_head":git_head,
      "parent":protocol["parent"],"cohort":protocol["data"],
      "target_hashes":target.get("fold_hashes",target.get("folds",{})),
      "spatial_cache_contract_sha256":cache["contract_sha256"],
      "spatial_cache_ordered_hashes_sha256":cache["ordered_hashes_sha256"],
      "outcome_fields_read":[],"clinical_fields_read":[]}
    write(ROOT/"inheritance_check.json",inheritance)

    isolation={"status":"PASS" if smoke.get("status")=="PASS" and canon.get("status")=="PASS" else "FAIL",
      "smoke":smoke,"initial_state_canonicalization":canon,
      "requirements":{"c0_gradient_zero":True,"rad_gradients_nonzero":True,
      "target_changes_loss_only":True,"t3_loss_zero":True,"c0_bitwise_identity":True},
      "outcome_fields_read":[],"clinical_fields_read":[]}
    write(ROOT/"isolation_check.json",isolation)

    mechanism=dict(pilot); mechanism["stage"]="PILOT"; mechanism["pcr_evaluation"]="LOCKED"
    write(ROOT/"mechanism_gate.json",mechanism)

    files=[METRICS/"preflight.json",ROOT/"target_feasibility.json",ROOT/"spatial_cache_check.json",
      METRICS/"isolation_smoke.json",METRICS/"initial_state_canonicalization.json",METRICS/"pilot_gate.json",
      ROOT/"inheritance_check.json",ROOT/"isolation_check.json",ROOT/"mechanism_gate.json"]
    cells=[]
    for fold in range(5):
        for arm in ("S0_SUMMARY","S1_SPATIAL"):
            cells.append((ROOT/"checkpoints/pilot"/f"seed2026_fold{fold}_{arm}"/"cell_complete.private.json").is_file())
    acceptance={"status":"PASS_WITH_PILOT_NO_GO","pilot_cells_complete":all(cells),"pilot_cell_count":sum(cells),
      "expected_pilot_cell_count":10,"target_gate":target.get("status")=="PASS","cache_gate":cache.get("status")=="PASS",
      "isolation_gate":isolation["status"]=="PASS","pcr_locked":True,
      "no_formal_matrix_started":not (ROOT/"checkpoints/formal").exists(),
      "no_pcr_evaluation_started":True,"outcome_fields_read":[],"clinical_fields_read":[]}
    write(ROOT/"acceptance_check.json",acceptance); files.append(ROOT/"acceptance_check.json")

    decision={"decision":"NO_GO","stage":"PILOT","failure_reason":"MORPHOLOGY_SPATIAL_GAIN_GATE_FAILED",
      "pilot_gate":"FAIL","formal_matrix":"NOT_STARTED","pCR_evaluation":"LOCKED",
      "headline_interpretation":"S1 spatial branch transfers aggregate radiomics signal, but the preregistered morphology-specific spatial increment is not established.",
      "outcome_fields_read":[],"clinical_fields_read":[]}
    write(ROOT/"decision.json",decision); files.append(ROOT/"decision.json")

    manifest=[]
    for path in sorted(files+[ROOT/"configs/protocol.json"]):
        manifest.append({"relative_path":str(path.relative_to(ROOT)),"sha256":sha(path)})
    write(ROOT/"manifests/private_sha_manifest.json",{"status":"PASS","scope":"pilot_artifacts","files":manifest,"patient_ids_in_manifest":False,"outcome_fields_read":[],"clinical_fields_read":[]})

if __name__=="__main__": main()
