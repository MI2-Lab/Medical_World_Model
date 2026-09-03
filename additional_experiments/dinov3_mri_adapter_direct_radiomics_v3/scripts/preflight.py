#!/usr/bin/env python3
"""Validate V2 inheritance and V3 representation contracts without outcomes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.cache_io import load_c1b_manifest  # noqa: E402
from dinov3_rg.contracts import (  # noqa: E402
    FOLDS, TRAIN_ONLY_MANIFEST, V2_CHECKPOINT_ROOT, V2_ROOT, V2_STATE_ROOT,
    V2_SUMMARY_DIR, V2_TARGET_DIR, atomic_json, canonical_sha256, file_sha256,
    load_protocol, private_patient_token,
)
from dinov3_rg.data import FoldTargets, load_fold_frame, load_train_only_ids, load_summary, validate_state_archive  # noqa: E402
from dinov3_rg.model import MRIAdapterWorldModel, initialization_sha256  # noqa: E402
from dinov3_rg.objective import DirectRadiomicsObjective, masked_patient_smooth_l1  # noqa: E402
from dinov3_rg.security import scan_representation_sources  # noqa: E402
from dinov3_rg.training import reinitialize_radiomics_head  # noqa: E402


def main() -> None:
    protocol = load_protocol()
    if os.environ.get("CONDA_DEFAULT_ENV") != "bowen":
        raise SystemExit("preflight must run in Anaconda environment 'bowen'")
    parent = protocol["parent"]
    inherited_files = {
        "decision": (V2_ROOT / "decision.json", parent["v2_decision_sha256"]),
        "protocol": (V2_ROOT / "configs/protocol.json", parent["v2_protocol_sha256"]),
        "dino_manifest": (V2_ROOT / "manifests/dinov3_cache_complete.json", parent["v2_dino_manifest_sha256"]),
        "target_feasibility": (V2_ROOT / "target_feasibility.json", parent["v2_target_feasibility_sha256"]),
    }
    inherited_hashes = {name: file_sha256(path) for name, (path, _) in inherited_files.items()}
    inherited_public = all(inherited_hashes[name] == expected for name, (_, expected) in inherited_files.items())
    v2_clean = subprocess.run(
        ["git", "diff", "--quiet", parent["v2_commit"], "--", str(V2_ROOT.relative_to(ROOT.parents[1]))],
        cwd=ROOT.parents[1], check=False,
    ).returncode == 0
    dino_manifest = json.loads((V2_ROOT / "manifests/dinov3_cache_complete.json").read_text(encoding="utf-8"))
    cache_entries = load_c1b_manifest(); patient_ids = tuple(sorted(cache_entries))
    cache_hashes=[]
    for patient_id in patient_ids:
        path = V2_SUMMARY_DIR / f"{private_patient_token(patient_id)}.private.npz"
        load_summary(path, patient_id); cache_hashes.append(file_sha256(path))
    cache_complete = (
        len(cache_hashes) == 947
        and canonical_sha256(cache_hashes) == dino_manifest["ordered_cache_hashes_sha256"]
        and dino_manifest["contract_sha256"] == protocol["inheritance"]["dino_contract_sha256"]
    )
    target_checks={}; c0_checks={}
    for fold in FOLDS:
        gate_path = V2_ROOT / f"metrics/fold_{fold}_target_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        target_path = V2_TARGET_DIR / f"fold_{fold}_targets.private.npz"
        targets = FoldTargets.load(target_path)
        target_checks[str(fold)] = bool(
            file_sha256(gate_path) == protocol["inheritance"]["target_fold_gate_sha256"][str(fold)]
            and file_sha256(target_path) == gate["target_sha256"]
            and not targets.radiomics_mask[:, 3].any()
        )
        cell = V2_CHECKPOINT_ROOT / f"seed2026_fold{fold}_D1"
        complete = json.loads((cell / "cell_complete.private.json").read_text(encoding="utf-8"))
        state_path = V2_STATE_ROOT / f"seed2026_fold{fold}_D1_states.private.npz"
        validate_state_archive(state_path)
        c0_checks[str(fold)] = bool(
            complete["status"] == "COMPLETE"
            and file_sha256(cell / "selected.private.pt") == complete["checkpoint_sha256"]
        )
    folds = load_fold_frame(); train_only = load_train_only_ids(TRAIN_ONLY_MANIFEST)
    torch.manual_seed(3); model = MRIAdapterWorldModel(dropout=0.0)
    candidate_a = MRIAdapterWorldModel(dropout=0.0); candidate_a.load_state_dict(model.state_dict())
    candidate_b = MRIAdapterWorldModel(dropout=0.0); candidate_b.load_state_dict(model.state_dict())
    head_a = reinitialize_radiomics_head(candidate_a, 902026)
    head_b = reinitialize_radiomics_head(candidate_b, 902026)
    synthetic = torch.randn(2, 4, 7, 32, 2304)
    output = candidate_a(synthetic)
    target = torch.randn(2, 4, 16); mask = torch.ones(2, 4, dtype=torch.bool); mask[:, 3] = False
    radiomics_loss, _, _ = masked_patient_smooth_l1(output.radiomics_prediction, target, mask)
    radiomics_loss.backward()
    adapter_gradient = sum(float(p.grad.abs().sum()) for p in candidate_a.adapter.parameters() if p.grad is not None)
    projection_gradient = sum(float(p.grad.abs().sum()) for p in candidate_a.adapter.response_projection.parameters() if p.grad is not None)
    head_gradient = sum(float(p.grad.abs().sum()) for p in candidate_a.radiomics_head.parameters() if p.grad is not None)
    objective = DirectRadiomicsObjective(0.25)
    try:
        candidate_a(synthetic, torch.ones(1))  # type: ignore[call-arg]
    except TypeError:
        extra_input_rejected = True
    else:
        extra_input_rejected = False
    representation_paths = [
        ROOT / "src/dinov3_rg/data.py", ROOT / "src/dinov3_rg/model.py",
        ROOT / "src/dinov3_rg/objective.py", ROOT / "src/dinov3_rg/training.py",
        ROOT / "src/dinov3_rg/probes.py", ROOT / "scripts/preflight.py",
        ROOT / "scripts/run_smoke.py", ROOT / "scripts/train_cell.py",
        ROOT / "scripts/run_pilot.py", ROOT / "scripts/evaluate_pilot.py",
        ROOT / "scripts/run_formal_matrix.py", ROOT / "scripts/evaluate_mechanism.py",
    ]
    source_scan = scan_representation_sources(representation_paths)
    gates = {
        "bowen_environment": True, "v2_public_hashes": inherited_public,
        "v2_tree_immutable": v2_clean, "dino_cache_947_hash_bound": cache_complete,
        "five_target_folds_hash_bound": all(target_checks.values()),
        "five_pilot_c0_cells_available": all(c0_checks.values()),
        "folds_808x5": len(folds) == 4040, "train_only_139": len(train_only) == 139,
        "candidate_ftv_weight_zero": objective.ftv_weight == 0.0,
        "paired_head_initialization": head_a == head_b,
        "model_state_shape": tuple(output.response_state.shape) == (2,4,192),
        "radiomics_gradient_adapter": adapter_gradient > 0,
        "radiomics_gradient_response_projection": projection_gradient > 0,
        "radiomics_gradient_head": head_gradient > 0,
        "extra_forward_input_rejected": extra_input_rejected,
        "representation_source_scan": source_scan["status"] == "PASS",
    }
    inheritance = {
        "schema_version": 1, "status": "PASS" if all(gates.values()) else "FAIL",
        "v2_commit": parent["v2_commit"], "v2_decision": parent["v2_decision"],
        "inherited_public_hashes": inherited_hashes,
        "dino_contract_sha256": dino_manifest["contract_sha256"],
        "dino_ordered_cache_hashes_sha256": dino_manifest["ordered_cache_hashes_sha256"],
        "target_fold_checks": target_checks, "pilot_c0_checks": c0_checks,
        "outcome_fields_read": [], "clinical_fields_read": [],
    }
    payload = {
        "schema_version": 1, "status": inheritance["status"], "gates": gates,
        "protocol_sha256": file_sha256(ROOT / "configs/protocol.json"),
        "source_scan": source_scan, "model_architecture": model.architecture_contract(),
        "initialization_probe_sha256": initialization_sha256(model),
        "outcome_fields_read": [], "clinical_fields_read": [],
    }
    atomic_json(ROOT / "inheritance_check.json", inheritance)
    atomic_json(ROOT / "metrics/preflight.json", payload)
    print(payload)
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
