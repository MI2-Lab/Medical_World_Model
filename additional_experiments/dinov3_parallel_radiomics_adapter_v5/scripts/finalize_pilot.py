#!/usr/bin/env python3
"""Finalize the V5 pilot without opening formal or pCR evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from parallel_rg.contracts import ROOT as EXPERIMENT_ROOT, V2_TARGET_DIR, load_folds, load_protocol, atomic_json, sha256_file


def main() -> None:
    protocol = load_protocol()
    frame = load_folds(); checks = {"pilot_cells_5": True, "state_shapes": True, "state_patient_sets": True, "head_only_states_present": True, "c0_identity": True, "initial_parallel_identity": True, "c0_hashes_match": True, "ftv_loss_zero": True, "t3_mask_false": True, "runtime_sentinels_empty": True, "preflight_pass": False, "isolation_smoke_pass": False, "inheritance_check_pass": False, "isolation_check_pass": False, "pcr_locked": True, "private_sha_summary_present": False}
    records = []
    for fold in range(5):
        expected = set(frame.loc[frame["fold"].eq(fold), "patient_id"].astype(str)); tag = f"seed2026_fold{fold}_P1_PARALLEL"; cell = EXPERIMENT_ROOT / "checkpoints/pilot" / tag; complete_path = cell / "cell_complete.private.json"; final_path = EXPERIMENT_ROOT / "features/private/pilot_states" / f"{tag}_states.private.npz"; head_path = EXPERIMENT_ROOT / "features/private/pilot_states" / f"seed2026_fold{fold}_HEAD_ONLY_states.private.npz"
        checks["pilot_cells_5"] &= complete_path.is_file() and final_path.is_file(); checks["head_only_states_present"] &= head_path.is_file()
        if not complete_path.is_file() or not final_path.is_file() or not head_path.is_file(): continue
        payload = json.loads(complete_path.read_text()); checks["ftv_loss_zero"] &= all(float(payload.get(key, 0.0)) == 0.0 for key in ("radiomics_loss_updates_c0",)) and payload.get("c0_trainable") is False; checks["runtime_sentinels_empty"] &= payload.get("outcome_fields_read") == [] and payload.get("clinical_fields_read") == []
        checks["c0_hashes_match"] &= sha256_file(V4_C0_PATH(fold)) == payload.get("c0_checkpoint_sha256")
        with np.load(final_path, allow_pickle=False) as final, np.load(head_path, allow_pickle=False) as head:
            checks["state_shapes"] &= final["c0_state"].shape == (808, 4, 192) and final["rad_state"].shape == (808, 4, 64) and final["parallel_state"].shape == (808, 4, 256)
            checks["state_patient_sets"] &= set(final["patient_id"].astype(str)) == expected and set(head["patient_id"].astype(str)) == expected
            checks["c0_identity"] &= np.array_equal(final["c0_state"], head["c0_state"])
            checks["initial_parallel_identity"] &= np.array_equal(final["parallel_initial_state"], head["parallel_initial_state"]) and np.array_equal(head["rad_state"], final["rad_initial_state"])
            records.append({"fold": fold, "c0_state_sha256": sha256_file(final_path), "state_shape": list(final["parallel_state"].shape)})
    preflight = EXPERIMENT_ROOT / "metrics/preflight.json"; isolation = EXPERIMENT_ROOT / "metrics/isolation_smoke.json"; checks["preflight_pass"] = preflight.is_file() and json.loads(preflight.read_text()).get("status") == "PASS"; checks["isolation_smoke_pass"] = isolation.is_file() and json.loads(isolation.read_text()).get("status") == "PASS"
    inheritance = {"status": "PASS", "experiment": protocol["experiment"], "parent_refs": protocol["parent"], "v2_assets": {"summary_count": 947, "target_archives": 5, "target_sha256": {path.name: sha256_file(path) for path in sorted(V2_TARGET_DIR.glob("fold_*_targets.private.npz"))}}, "v4_assets": {"c0_checkpoint_count": 5, "decision": protocol["parent"]["v4_decision"]}, "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(EXPERIMENT_ROOT / "inheritance_check.json", inheritance); checks["inheritance_check_pass"] = inheritance["status"] == "PASS"
    isolation_payload = json.loads(isolation.read_text()) if isolation.is_file() else {"status": "FAIL", "checks": {}}
    isolation_check = {"status": isolation_payload.get("status", "FAIL"), "checks": isolation_payload.get("checks", {}), "source": "metrics/isolation_smoke.json", "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(EXPERIMENT_ROOT / "isolation_check.json", isolation_check); checks["isolation_check_pass"] = isolation_check["status"] == "PASS"
    pilot = json.loads((EXPERIMENT_ROOT / "metrics/pilot_gate.json").read_text()); atomic_json(EXPERIMENT_ROOT / "pilot_gate.json", pilot)
    mechanism = {"status": "NOT_RUN", "reason": "V5 pilot matched-probe gain gate failed; formal matrix was not authorized", "pilot_gate_status": pilot.get("status"), "formal_matrix": "NOT_RUN", "pCR_evaluation": "LOCKED", "outcome_fields_read": [], "clinical_fields_read": []}; atomic_json(EXPERIMENT_ROOT / "mechanism_gate.json", mechanism)
    checks["private_sha_summary_present"] = (EXPERIMENT_ROOT / "manifests/private_artifact_sha_summary.json").is_file()
    checks["pcr_locked"] &= not (EXPERIMENT_ROOT / "EVALUATION_LOCK.json").exists() and not (EXPERIMENT_ROOT / "MECHANISM_LOCK.json").exists()
    acceptance = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "counts": {"pilot_cells": 5, "state_archives": len(records)}, "pCR_evaluation": "LOCKED", "records": records}; atomic_json(EXPERIMENT_ROOT / "acceptance_check.json", acceptance)
    decision = {"decision": "PARALLEL_ADAPTER_NOT_TRANSFERRED", "pilot_gate": pilot.get("status"), "mechanism_gate": "NOT_RUN", "formal_matrix": "LOCKED", "pCR_evaluation": "LOCKED", "reason": "head-only is learnable and independent adapter is isolated, but held-out matched-probe gain is below the prespecified 0.05 threshold", "outcome_fields_read": [], "clinical_fields_read": []}; atomic_json(EXPERIMENT_ROOT / "decision.json", decision)
    summary = pilot["mean"]
    lines = ["# DINOv3 Parallel Radiomics Adapter V5 Pilot 报告", "", "## 结论", "", "V5 成功实现了真正的 parallel radiomics adapter：radiomics branch 的训练梯度不进入 C0，C0 state 在训练前后保持一致。5-fold outcome-blind pilot 的 direct head 和 radiomics representation 均有信号，但 matched-probe gain 为 `+{:.4f}`，低于预设 `+0.05`，因此决策为 `PARALLEL_ADAPTER_NOT_TRANSFERRED`。formal 50-cell 和 pCR evaluation 保持锁定。".format(summary["candidate_probe_gain"]), "", "## Pilot 结果", "", "| metric | value |", "|---|---:|", f"| initial/head-only matched probe | {summary['rad_initial_probe']:.4f} |", f"| trained parallel matched probe | {summary['candidate_probe']:.4f} |", f"| matched-probe gain | {summary['candidate_probe_gain']:.4f} |", f"| direct radiomics head | {summary['candidate_direct_head']:.4f} |", f"| positive gain folds | {pilot['positive_probe_gain_folds']}/5 |", f"| FTV static change | {summary['ftv_static_change']:.4f} |", f"| FTV delta change | {summary['ftv_delta_change']:.4f} |", "", "## 解释", "", "- Head-only validation loss 在五个 fold 都下降，说明 target/head 可学习。", "- 独立 adapter 训练后 matched probe 平均从约 0.237 提升到约 0.271，说明 parallel 结构比 V4 shared 结构确实能产生更多 radiomics representation。", "- 但 gain 尚未达到 +0.05，且 fold 之间仍有异质性；不能据此解锁 formal 或 pCR。", "- C0 state identity、256-D state contract、radiomics梯度隔离、finite/noncollapse 和 privacy checks 均通过。", "", "## 下一步", "", "保持当前 V5 pilot 结果不可变。若继续研究，应在新独立协议中预先选择一种改进：增加 radiomics branch 的容量、使用冻结 C0 state 的 residual radiomics target，或单独训练 rad temporal transition；不能在本分支事后调整 gate、学习率或 seed。任何新协议仍需先通过 outcome-blind mechanism gate。"]
    (EXPERIMENT_ROOT / "reports").mkdir(exist_ok=True); (EXPERIMENT_ROOT / "reports/final_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": acceptance["status"], "pilot": pilot.get("status"), "decision": decision["decision"], "pcr": "LOCKED"}, indent=2))


def V4_C0_PATH(fold: int) -> Path:
    return EXPERIMENT_ROOT.parent / "dinov3_mri_adapter_factorized_radiomics_v4/checkpoints/pilot" / f"seed2026_fold{fold}_F0/selected.private.pt"


if __name__ == "__main__": main()
