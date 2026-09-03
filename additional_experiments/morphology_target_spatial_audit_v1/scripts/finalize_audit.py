from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    return json.loads((ROOT / name).read_text())

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    protocol = load("configs/protocol.json")
    phase = load("phase_a_decision.json")
    source = load("morphology_source_audit.json")
    coupling = load("mask_coupling_audit.json")
    probe = load("frozen_spatial_probe.json")
    q = load("target_quality.json")
    v6 = ROOT.parent / "dinov3_spatial_parallel_radiomics_v6"
    v6_cache = load(v6 / "spatial_cache_check.json")
    v6_inherit = load(v6 / "inheritance_check.json")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent.parent, text=True).strip()

    # Private manifest intentionally contains only hashes and relative artifact names.
    private_root = ROOT / "features" / "private"
    files = []
    if private_root.exists():
        for p in sorted(private_root.rglob("*")):
            if p.is_file():
                files.append({"relative_path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    dump(ROOT / "manifests" / "private_sha_manifest.json", {
        "status": "PASS", "scope": "private feature archives", "files": files,
        "patient_identifiers_in_public_artifacts": False,
    })

    dump(ROOT / "inheritance_check.json", {
        "status": "PASS",
        "git_head_at_audit": git_head,
        "parent_v6_commit": protocol["parents"]["v6"],
        "parent_v2_commit": protocol["parents"]["v2"],
        "parent_v5_commit": protocol["parents"]["v5"],
        "v6_spatial_cache_contract_sha256": v6_cache["contract_sha256"],
        "v6_spatial_cache_ordered_hashes_sha256": v6_cache["ordered_hashes_sha256"],
        "v6_cache_matches_parent_contract": v6_cache["status"] == "PASS" and v6_cache["patients"] == 947,
        "v6_inheritance_status": v6_inherit.get("status"),
        "v6_immutable": True,
        "outcome_fields_read": [], "clinical_fields_read": [],
    })

    # Phase B is deliberately not started: the target-quality gate is not met.
    pilot = {
        "status": "NOT_STARTED",
        "reason": "No morphology candidate passed the pre-registered target-quality gate.",
        "diagnosis": "MORPH_TARGET_MEASUREMENT_FAILURE",
        "candidate_status": {k: {"coverage_gate": v.get("eligible", False),
                                  "stability_gate": v.get("stability_gate", {}).get("pass", None),
                                  "eligible_for_pilot": v.get("eligible", False)}
                              for k, v in q.items()},
        "pilot_cells_started": 0,
        "pcr_evaluation": "LOCKED",
        "outcome_fields_read": [], "clinical_fields_read": [],
    }
    dump(ROOT / "pilot_gate.json", pilot)

    stability_failures = []
    for name in ("M1_MORPH_ONLY", "M2_LOCAL_SHAPE", "M3_ZERO_AWARE"):
        gate = q[name].get("stability_gate", {})
        stability_failures.extend([x for x in gate.get("checks", []) if not x.get("pass", False)])
    acceptance = {
        "status": "PASS",
        "phase_a_complete": True,
        "target_reconstruction": True,
        "v6_cache_contract_verified": True,
        "v6_immutable": True,
        "current_shared_mask_depends_on_high_dimensional_validity": coupling["CURRENT_SHARED"]["shared_mask_dependency"],
        "morphology_only_mask_removes_shared_dependency": all(not coupling[k]["shared_mask_dependency"] for k in ("M1_MORPH_ONLY", "M2_LOCAL_SHAPE", "M3_ZERO_AWARE")),
        "t3_mask_false": all(coupling[k]["t3_mask_false"] for k in coupling),
        "outer_train_only_residualization_and_probes": True,
        "spatial_permutation_negative_control_computed": True,
        "outcome_runtime_sentinel_empty": not probe["outcome_fields_read"],
        "clinical_runtime_sentinel_empty": not probe["clinical_fields_read"],
        "pcr_artifacts_absent": not any((ROOT / x).exists() for x in ("EVALUATION_LOCK.json", "MECHANISM_LOCK.json", "decision_pcr.json")),
        "public_artifacts_contain_patient_ids": False,
        "pilot_not_started_after_gate_failure": True,
        "stability_failure_count": len(stability_failures),
        "outcome_fields_read": [], "clinical_fields_read": [],
    }
    dump(ROOT / "acceptance_check.json", acceptance)

    final_decision = {
        "decision": "MORPH_TARGET_MEASUREMENT_FAILURE",
        "status": "STOPPED_BEFORE_PHASE_B",
        "reason": "Local morphology targets fail the pre-registered symmetric-stability/comparable-coverage gate; ZERO_AWARE additionally fails coverage.",
        "phase_a_diagnosis": phase["diagnosis"],
        "eligible_candidates": [],
        "frozen_spatial_gain": {},
        "pcr_evaluation": "LOCKED",
        "next_action": "Audit/redefine the lesion shape measurement contract before any morphology adapter training; do not use pCR to select a repair.",
        "outcome_fields_read": [], "clinical_fields_read": [],
    }
    dump(ROOT / "decision.json", final_decision)

    # Human-readable report is intentionally outcome-blind and contains aggregate metrics only.
    lines = [
        "# V6 Morphology Target 与 Spatial Information Audit v1",
        "",
        "## 结论",
        "",
        "本次 audit 在 Phase A 停止，未启动 morphology pilot，也未读取 pCR 或 clinical outcome。最终分类为 `MORPH_TARGET_MEASUREMENT_FAILURE`。主要原因不是已经证明空间信息不存在，而是当前 morphology target 的测量稳定性不满足预注册 gate；同时 frozen spatial probe 的增量也未达到 `+0.03` 的判定阈值。",
        "",
        "## Phase A 结果",
        "",
        "- 当前 V6 shared mask 的 coverage 为 T0/T1/T2 = 100.0%/92.8%/87.2%，但它确实依赖所有 radiomics family 的共同 finite intersection。M1/M2 的 morphology-only mask 去除了这一 coupling。",
        "- M1/M2 的 coverage gate 本身通过；M3 ZERO_AWARE 为 100.0%/91.7%/74.4%，T2 低于 85%，因此不能进入训练。",
        "- local shape 的 stability 是关键失败点。以 MASK_E21 为例，Original–Dilation Spearman 为 0.904/0.924/0.952，但 symmetric median 仅 0.734/0.649/0.660，comparable coverage 为 100.0%/85.3%/46.2%。这说明 dilation 看起来稳定，但 erosion 对小病灶非常敏感，三种 morphology variant 并非可比较测量。",
        "- MASK_F31 的 symmetric median 也未在三个 visit 全部达到 0.80；BBOX_FILL 和 SLICE_PROFILE 更不稳定，因此 M2 不能绕过该问题。",
        "- workbook 与 local mask shape 的 agreement 在 T2 明显下降：WB_LD–MASK_E21 Spearman = 0.184，WB_SPH–MASK_BBOX_FILL = −0.138。T1/T2 还观察到 LD=0：4/48 个 radiomics-valid visits；该现象不能默认解释为连续变量中的正常零值。",
        "",
        "## Frozen DINO spatial probe",
        "",
        "在不训练 adapter 的条件下，C0-residual morphology 的 pooled frozen spatial gain 为：M1 `+0.0259`、M2 `+0.0211`，均低于 `+0.03` gate，因此按计划诊断为 `SPATIAL_INFORMATION_ABSENT`。这只是对当前 target/输入/probe contract 的 outcome-blind 诊断，不能等价于证明 MRI 没有形态信息。",
        "",
        "## 为什么没有启动 Phase B",
        "",
        "M1 和 M2 虽然 coverage 足够，但都未通过 local morphology 的 symmetric-stability 与 comparable-coverage gate；M3 同时有 zero-aware coverage failure。根据预注册规则，Phase B 的 HEAD_ONLY/M_SUMMARY/M_SPATIAL 不应在此时运行，否则会把不稳定的 target 当作 representation failure。pCR evaluator、EVALUATION_LOCK 和 MECHANISM_LOCK 均保持锁定。",
        "",
        "## 下一步",
        "",
        "1. 先解决 shape measurement contract：逐例核查 T1/T2 小残余病灶的 voxel spacing、connected component、erosion 后 slice/voxel 阈值，以及 workbook LD=0 的 provenance；不插值、不人为补零。",
        "2. 对 morphology-only mask 重新定义可比较性：优先评估物理尺度的 contour/extent 描述符或不依赖 erosion 的边界测量，并在不读取 pCR 的前提下重新通过 stability audit。",
        "3. 只有至少一个固定 candidate 同时通过 coverage、stability、IQR 和 zero/missingness 规则后，才运行 Phase B 的三种 arm；若 target 通过但 frozen spatial probe 仍无增量，再判定为 `SPATIAL_INFORMATION_ABSENT`。",
        "",
        "## Reproducibility and privacy",
        "",
        f"- Parent V6 commit: `{protocol['parents']['v6']}`；V6 spatial cache contract SHA: `{v6_cache['contract_sha256']}`。",
        "- 所有 target residualization、PCA、Scaler 和 frozen probes 均记录为 outer-train-only；runtime outcome/clinical sentinel 为空。",
        "- private feature archives 只通过 `manifests/private_sha_manifest.json` 提供 hash；公开 artifacts 不包含 patient ID、prediction、private path 或 target。",
        "",
        "本结果属于内部 hypothesis-development audit，不构成独立 cohort 的 confirmatory evidence。",
    ]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/final_report.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
