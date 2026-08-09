# Stage A C1B model-ready hard gate

## 结论

`STAGE_A = NO-GO`。完整 948 人、3792 visit header audit发现1个冻结visit为 `ZERO_VALID_SOURCE_OVERLAP`（valid-source voxel = 0）。262/263个validation cache虽已原子完成，但cohort-level schema-3 contract未闭合；不改变人口、不放宽门槛，禁止启动Stage B。

| 子门 | 状态 | 观测 | 冻结要求 |
|---|---|---|---|
| repaired_dicom_pixel_geometry | PASS | `{"max_cell_error": 0.0, "repaired_visits": 146, "verified_cells": 153112}` | 146/146 singular model-input visits pass exact pixel/geometry rebuild |
| true_canonical_orientation | PASS | `{"canonical_ras_fraction": 1.0, "patients": 948, "visits": 3792}` | 948x4 true RAS+ array reorientation, not header-only relabeling |
| registration_success_and_strategy | PASS | `{"chosen_strategy": "H", "formal_pairs": 1125, "success_rate": 0.7626666666666667, "successful_pairs": 858}` | complete 1,125-pair sensitivity audit and uniquely frozen C1B-H decision |
| formal_support_containment | PASS | `{"exact_containment_rate": 0.978, "formal_visits": 1500}` | >=0.95 exact containment over all 1,500 formal visits |
| formal_ftv_retention_q05 | PASS | `{"posthoc_registration_audit_q05": 1.0, "q05": 1.0}` | >=0.95 physical FTV volume retention at Q05 |
| resampling_and_source_overlap | FAIL | `{"exact_frozen_population_coverage": true, "failed_visits": 1, "failure_code": "ZERO_VALID_SOURCE_OVERLAP", "header_audit_rows": 3792, "minimum_valid_source_voxels": 0}` | all frozen visits must have at least one valid source voxel |
| complete_dce7_builder_and_cache | FAIL | `{"atomic_caches_present": 262, "cohort_cache_contract_completed": false, "validation_patients": 263}` | 263/263 validation caches and complete schema-3 cohort contract |
| leakage_exclusion_contract | PASS | `{"clinical_or_outcome_tables_read": [], "outcome_fields_read": [], "public_identifiers": false}` | no clinical, treatment, pCR, LD, identifier, or path leakage |
| geometry_and_mask_scope_contract | PASS | `{"anchor_uses_t0_only": true, "future_support_used_for_grid": false, "grounding_mask_is_model_input": false, "grounding_observable_visits": 1486}` | T0-only anchor; future support excluded; grounding mask is loss-only |
| stage_a_model_ready | FAIL | `{"frozen_patients": 948, "frozen_visits": 3792}` | every Stage-A hard gate must pass without eligibility changes |

公开产物只含聚合计数和私有证据SHA-256，不含patient identifier或路径。
