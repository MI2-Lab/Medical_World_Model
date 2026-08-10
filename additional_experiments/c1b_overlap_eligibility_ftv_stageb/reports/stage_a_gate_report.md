# Stage A：Four-Visit overlap eligibility + C1B-H model-ready gate

## 结论

`STAGE_A = GO`；15 项 hard gate 中 15/15 PASS，失败项：无。因此允许启动 Stage B。该结论属于预注册的全新 run，不追溯修改旧 `STAGE_A_NO_GO`，也不改变独立 provenance audit 的 `AUDIT-NOT-REPAIRABLE`。

## Table 1：Technical eligibility + Stage-A QC

- candidate patients：948
- eligible patients：947
- excluded patients：1
- candidate visits：3792
- eligible visits：3788
- cache completion：1.0
- formal intersection：375 patients / 1500 visits
- formal containment：0.978
- FTV retention Q05：1.0

| # | Gate | 状态 | 观测（仅聚合） | 冻结要求 |
|---:|---|---|---|---|
| 1 | `eligibility_rule_frozen_before_stage_b` | PASS | `{"eligibility_output_bound_to_plan": true, "preregistration_verified": true, "stage_b_artifacts_before_gate": 0}` | 规则在 eligibility/Stage-B 结果之前冻结，且 finalizer 前无 Stage-B 产物 |
| 2 | `eligibility_outcome_free_and_public_private` | PASS | `{"outcome_forbidden_field_lists_empty": true, "public_privacy_scan": "PASS"}` | eligibility 仅读 source/geometry/grid/overlap；公开文本无 ID、UID 或绝对路径 |
| 3 | `eligible_cohort_mechanically_determined` | PASS | `{"candidate_patients": 948, "eligible_patients": 947, "excluded_patients": 1}` | 从完整 candidate manifest 机械执行通用四访 AND；无硬编码人数或病例规则 |
| 4 | `all_eligible_visits_positive_overlap` | PASS | `{"candidate_valid_visits": 3791, "candidate_zero_overlap_visits": 1, "eligible_visits": 3788, "frozen_grid_and_geometry_digest_closure": true}` | 每名 eligible patient 的 T0-T3 均有 valid_source_voxels > 0 |
| 5 | `dicom_repair_contract` | PASS | `{"failed_visits": 0, "max_cell_error": 0.0, "repaired_visits": 146}` | 旧 raw-DICOM PixelData/geometry repair 证据 hash 不变且持续 PASS |
| 6 | `true_ras_orientation_contract` | PASS | `{"candidate_patients": 948, "candidate_visits": 3792, "canonical_ras_fraction": 1.0}` | 完整 candidate population 持续为真实 array-reordered RAS+，非 header relabel |
| 7 | `c1b_h_strategy_frozen` | PASS | `{"chosen_strategy": "H", "decision_frozen": true, "prior_stage_a_status": "NO-GO", "provenance_audit_decision": "AUDIT-NOT-REPAIRABLE"}` | 唯一正式策略仍为 C1B-H；旧决策与不可变树 hash 闭合 |
| 8 | `formal_ftv_support_containment` | PASS | `{"exact_containment_rate": 0.978, "formal_patients_after_eligibility_intersection": 375, "formal_visits_after_intersection": 1500}` | 旧 formal support 先与新 eligible IDs 取交集，再机械重聚合；rate >= 0.95 |
| 9 | `ftv_retention_q05` | PASS | `{"formal_visits_after_intersection": 1500, "physical_volume_retention_q05": 1.0}` | 交集后 formal physical FTV retention Q05 >= 0.95 |
| 10 | `grounding_observability_loss_side_only` | PASS | `{"formal_ineligible_visits": 14, "formal_observable_visits": 1486, "is_model_input": false}` | grounding_observable_mask 仅为 loss-side metadata，不筛 base training |
| 11 | `complete_dce7_cache` | PASS | `{"cache_evidence_schema": "formal_runner_v1", "completed_patients": 947, "completed_visits": 3788, "completion_fraction": 1.0, "eligible_patients": 947, "eligible_visits": 3788, "finite_fraction": 1.0, "grid_fraction": 1.0, "nonconstant_fraction": 1.0, "orientation_fraction": 1.0, "phase_fraction": 1.0, "shape_fraction": 1.0}` | eligible patient 行集精确一致；四访 DCE7 cache 完成率与全部 QC 均为 100% |
| 12 | `cache_roundtrip_and_hash` | PASS | `{"cache_and_private_assets_owner_only": true, "cache_files_have_single_link": true, "eligibility_exact_count_match_fraction": 1.0, "input_provenance_fraction": 1.0, "live_cache_file_hash_fraction": 1.0, "private_tables_sha256_bound": true, "roundtrip_fraction": 1.0}` | cache reload/round-trip、文件/content/input provenance hash 与 eligibility count 闭合 |
| 13 | `no_patient_specific_manual_correction` | PASS | `{"cache_manual_corrections": 0, "manual_transform_trials": 0, "patient_specific_eligibility_rules": 0}` | 无 patient-specific flip、translation、recenter、registration repair 或 known-case rule |
| 14 | `no_unresolved_catastrophic_resampling` | PASS | `{"eligible_zero_overlap_visits": 0, "unresolved_catastrophic_resampling_cases": 0}` | eligible cohort 无 unresolved catastrophic overlap/resampling case |
| 15 | `geometry_metadata_excluded_from_model_tensor` | PASS | `{"geometry_metadata_is_model_input": false, "model_loader_image_only_fraction": 1.0, "valid_source_mask_is_model_input": false}` | model tensor 仅含 DCE7 image；geometry/mask/support/provenance 均为 sidecar |

## 审计边界

Formal containment、retention 与 grounding evidence 均先用 technical-eligible patient set 机械取交集，再从逐访私有证据重聚合；eligibility 本身未读取这些 lesion/FTV 字段。旧 DICOM、RAS+、C1B-H、containment、retention、grounding contracts 以及两个旧实验 tracked tree 均须通过预注册 hash lock。公开表、报告和 sentinel 只含聚合计数与 SHA-256，不含 patient identifier、UID、源路径或逐病例坐标。
