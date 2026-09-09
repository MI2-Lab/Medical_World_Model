"""Create the fail-closed exploratory pCR lock from immutable V6 states."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT.parent / "dinov3_spatial_parallel_radiomics_v6"
sys_path = ROOT / "src"
import sys
sys.path.insert(0, str(sys_path))
from dinov3_rg.contracts import atomic_json, canonical_sha256, file_sha256  # noqa: E402


def main() -> None:
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    cache_check = json.loads((V6 / "spatial_cache_check.json").read_text())
    if cache_check.get("status") != "PASS" or cache_check.get("patients") != 947:
        raise RuntimeError("V6 spatial cache contract is not valid")
    if cache_check.get("contract_sha256") != protocol["parent"]["v6_spatial_cache_contract_sha256"]:
        raise RuntimeError("V6 spatial cache contract hash drifted")
    if cache_check.get("ordered_hashes_sha256") != protocol["parent"]["v6_spatial_cache_ordered_hashes_sha256"]:
        raise RuntimeError("V6 spatial cache ordering hash drifted")

    state_root = V6 / "features/private/states"
    records = []
    reference_c0 = {}
    reference_order = {}
    for arm in ("S0_SUMMARY", "S1_SPATIAL"):
        for fold in range(5):
            path = state_root / f"seed2026_fold{fold}_{arm}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as z:
                required = {"patient_id", "c0_state", "parallel_state", "radiomics_prediction"}
                if not required.issubset(z.files):
                    raise RuntimeError(f"state contract missing in {path.name}")
                ids = tuple(z["patient_id"].astype(str).tolist())
                c0 = np.asarray(z["c0_state"], np.float32)
                state = np.asarray(z["parallel_state"], np.float32)
                pred = np.asarray(z["radiomics_prediction"], np.float32)
            if len(ids) != 808 or len(set(ids)) != 808 or not np.isfinite(c0).all() or not np.isfinite(state).all() or not np.isfinite(pred).all():
                raise RuntimeError(f"finite/order contract failed in {path.name}")
            if c0.shape != (808, 4, 192) or state.shape != (808, 4, 256) or pred.shape != (808, 4, 16):
                raise RuntimeError(f"state tensor contract failed in {path.name}")
            if fold not in reference_c0:
                reference_c0[fold], reference_order[fold] = c0.copy(), ids
            elif reference_order[fold] != ids or not np.array_equal(reference_c0[fold], c0):
                raise RuntimeError(f"C0 identity failed in {path.name}")
            records.append({"arm": arm, "fold": fold, "sha256": file_sha256(path), "shape": [808, 4, 256]})

    target_records = []
    for fold in range(5):
        path = V6 / "features/private/fold_targets" / f"fold_{fold}_targets.private.npz"
        with np.load(path, allow_pickle=False) as z:
            target = np.asarray(z["target"], np.float32)
            mask = np.asarray(z["target_mask"], bool)
        if target.shape != (375, 4, 16) or mask.shape != (375, 4) or mask[:, 3].any():
            raise RuntimeError(f"V6 target contract failed in fold {fold}")
        if not np.isfinite(target[:, :, 4:16]).all():
            raise RuntimeError(f"three-family target is non-finite in fold {fold}")
        target_records.append({"fold": fold, "sha256": file_sha256(path), "shape": [375, 4, 12], "t3_mask_false": True})

    implementation = {}
    for path in sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py")):
        implementation[str(path.relative_to(ROOT))] = file_sha256(path)
    inheritance = {
        "status": "PASS",
        "experiment": protocol["experiment"],
        "v6_commit": protocol["parent"]["v6_commit"],
        "v6_cache_contract_sha256": cache_check["contract_sha256"],
        "v6_cache_ordered_hashes_sha256": cache_check["ordered_hashes_sha256"],
        "state_files": records,
        "target_files": target_records,
        "c0_bitwise_identity_across_10_state_archives": True,
        "morphology_in_objective": False,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    atomic_json(ROOT / "inheritance_check.json", inheritance)
    lock = {
        "schema_version": 1,
        "status": "LOCKED_FOR_EXPLORATORY_PCR",
        "scope": "V6 seed-2026 state rescue only",
        "state_file_count": len(records),
        "state_files": [{"arm": x["arm"], "fold": x["fold"], "sha256": x["sha256"]} for x in records],
        "target_files": target_records,
        "v6_parent_commit": protocol["parent"]["v6_commit"],
        "v6_cache_contract_sha256": cache_check["contract_sha256"],
        "three_family_target": ["INTENSITY_PC4", "KINETIC_PC4", "TEXTURE_PC4"],
        "morphology_gate": "secondary_diagnostic_only",
        "implementation": implementation,
        "implementation_sha256": canonical_sha256(implementation),
        "outcome_fields_read_before_lock": [],
        "clinical_fields_read_before_lock": [],
        "pcr_read_authorized": True,
        "exploratory_not_confirmatory": True,
    }
    lock["lock_content_sha256"] = canonical_sha256(lock)
    atomic_json(ROOT / "EXPLORATORY_EVALUATION_LOCK.json", lock)
    print(json.dumps({"status": "PASS", "state_files": len(records), "target_files": len(target_records)}, indent=2))


if __name__ == "__main__":
    main()
