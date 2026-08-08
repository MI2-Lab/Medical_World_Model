# Response-Observable Multiscale Crop 最终报告

## 最终判断：INPUT-CONTRACT PARTIAL

available-support crop observability子门通过，但上游acquisition边界、geometry、temporal、resampling或model-input pipeline仍有未解决项。预注册的world-model主要方向为`C1B`（5 mm margin）：available-support sufficient=97.9%、exact=97.9%、FTV Q05=1.000。完整候选门控未通过项为：`upstream_acquisition_large_ld_sensitivity、no_extreme_downsampling、temporal_frame_validated、orientation_contract_validated、geometry_model_ready、model_input_pipeline_validated`。

这里的`PARTIAL`严格限定为 **available-support crop containment显著改善**，不等于end-to-end clinical whole-lesion observability已经成立。14/1500个visit的FTV inclusion support接触原始image face；将其作为上游censoring sensitivity后，C1B最坏visit的large-LD Q75/Q90疑似不可完整观察率为8.2%/10.5%，上游敏感性门控未通过。该敏感性不事后改写预注册crop-specific指标，但会阻止GO。

Stage B状态：**未授权、未执行**。无论本轮Decision如何，FTV+LD dual grounding、transition、pCR/clinical/treatment supervision均未执行。

独立raw-DICOM header复核覆盖72/72个异常visit：header geometry=PASS，但pixel rebuild=false、pixel order verified=false、model-ready=false。详见 [DICOM几何修复审计](dicom_geometry_repair_audit.md)。

## 逐条回答

1. **current crop physical FOV variability**：固定`32×96×96 voxel`的pooled median为X/Y/Z=63.8/63.8/64.0 mm，范围为30.0–127.5/30.0–127.5/25.6–80.0 mm；37.1%患者四访X spacing变化。
2. **C1 adaptive能否解决truncation**：在现有acquisition内可用support口径下，visit-adaptive C1A sufficient/exact为100.0%/100.0%，保留T0坐标的C1B为97.9%/97.9%，均相对C0的18.7%/20.5%显著改善；计入source-face uncertainty后，C1B source-uncensored且exact为97.1%。C1A依赖逐visit bbox重定位，不能忽略temporal normalization风险。
3. **C2是否提供额外context**：C2B context的有效source context/lesion volume ratio中位数由detail的705.3增至905.5，但padding中位数也由36.8%增至49.5%；95.1%的context visit存在任一轴downsampling factor `>2`（max=4.80）。所以它提供更多几何context，却尚未证明是额外“有效信号”，当前不优先于C1B。
4. **large lesion是否仍系统性截断**：预注册的available-support crop-specific最坏visit Q75/Q90为5.0%/5.3%，通过5%/10%阈值，已远低于legacy T0/T1的99–100%。但上游censoring sensitivity为8.2%/10.5%并未通过；所以不能声称large lesion的端到端observability已经可靠。
5. **FTV support是否完整可观察**：对available support，C1B overall Q05=1.000、exact=97.9%、minimum=0.163；source-uncensored且exact为97.1%。oracle-union也只给出available-support exact=100.0%，不能消除原始image边缘的不确定性。
6. **morphology/surface readiness**：C1B surface Q05=1.000，但minimum=0.168、fully-observable surface=97.1%、cut/missed-component visit比例为1.8%/1.9%。它达到crop-level群体门槛，但proxy surface不能提升为真实tumor morphology真值。
7. **visit-adaptive是否破坏longitudinal geometry**：是。C1A crop-center max drift median=31.75 mm，window跟随每访support bbox center，从而删除bbox平移；实际voxel centroid相对bbox center的变化仍保留并单独报告，但C1A不能作为唯一world-model coordinate view。
8. **T0-anchored是否更适合world model**：crop-induced geometry方面更合适；C1B window drift Q95=0.000000mm、FOV relative change Q95=0.000000。但header frame未做image-only rigid registration，patient repositioning仍是限制。
9. **observable且causally deployable candidate**：当前没有通过完整门控的candidate。C1B是最佳causal方向，只通过available-support crop containment与direct-leakage子门；上游acquisition sensitivity、DICOM pixel rebuild、orientation contract、resampling、temporal-frame validation和完整3-D model-input pipeline仍阻止GO。C2B的context分支重采样更严重；C1C明确禁止部署。
10. **是否有资格进入FTV-only retraining**：否；Stage B未授权。必须先完成raw-DICOM pixel rebuild/order验收、统一orientation策略、image-only registration sensitivity、极端downsampling处置以及3-D DCE7 phase selection/resampling/normalization/cache round-trip验证。
11. **observability是否改善FTV R²/optimization stability**：本轮Stage B未执行，不能声称改善或不改善。
12. **是否有资格重新测试LD grounding**：否。本Goal明确不自动执行Stage C；即使input GO，也应先完成FTV-only representation sanity。

## Candidate gate

```json
{
  "C1A": {
    "checks": {
      "anisotropy": true,
      "causally_deployable": true,
      "ftv_retention_q05": true,
      "geometry_model_ready": false,
      "large_ld_top_10pct": true,
      "large_ld_top_quartile": true,
      "model_input_pipeline_validated": false,
      "morphology_surface_q05": true,
      "no_direct_geometry_input": true,
      "no_extreme_downsampling": false,
      "orientation_contract_validated": false,
      "overall_exact_full_support": true,
      "overall_sufficient_containment": true,
      "source_boundary_observability": true,
      "temporal_frame_validated": false,
      "temporal_no_recentering_normalization": false,
      "upstream_acquisition_large_ld_sensitivity": false
    },
    "contract": "C1A",
    "evaluated_views": [
      "detail"
    ],
    "failed_checks": [
      "upstream_acquisition_large_ld_sensitivity",
      "no_extreme_downsampling",
      "temporal_no_recentering_normalization",
      "temporal_frame_validated",
      "orientation_contract_validated",
      "geometry_model_ready",
      "model_input_pipeline_validated"
    ],
    "large_ld_gate_scope": "WORST_VISIT_AND_VIEW",
    "large_ld_primary_definition": "PRE_REGISTERED_CROP_SUSPECTED_TRUNCATION_ON_AVAILABLE_SUPPORT",
    "observed": {
      "ftv_retention_q05": 1.0,
      "geometry_model_ready_fraction": 0.952,
      "max_cardinal_obliquity_deg": 12.500000923929592,
      "max_resize_anisotropy_ratio": 3.4722177725756236,
      "max_resize_factor": 2.88,
      "overall_exact_full_support_rate": 1.0,
      "overall_sufficient_containment_rate": 1.0,
      "source_uncensored_and_exact_rate": 0.9906666666666667,
      "surface_retention_q05": 1.0,
      "top_10pct_suspected_truncation_rate": 0.0,
      "top_10pct_upstream_censoring_adjusted_suspected_rate": 0.07894736842105263,
      "top_quartile_suspected_truncation_rate": 0.0,
      "top_quartile_upstream_censoring_adjusted_suspected_rate": 0.05102040816326531
    },
    "passed": false,
    "source_boundary_sensitivity_does_not_redefine_primary_metrics": true,
    "view": "detail"
  },
  "C1B": {
    "checks": {
      "anisotropy": true,
      "causally_deployable": true,
      "ftv_retention_q05": true,
      "geometry_model_ready": false,
      "large_ld_top_10pct": true,
      "large_ld_top_quartile": true,
      "model_input_pipeline_validated": false,
      "morphology_surface_q05": true,
      "no_direct_geometry_input": true,
      "no_extreme_downsampling": false,
      "orientation_contract_validated": false,
      "overall_exact_full_support": true,
      "overall_sufficient_containment": true,
      "source_boundary_observability": true,
      "temporal_frame_validated": false,
      "temporal_no_recentering_normalization": true,
      "upstream_acquisition_large_ld_sensitivity": false
    },
    "contract": "C1B",
    "evaluated_views": [
      "detail"
    ],
    "failed_checks": [
      "upstream_acquisition_large_ld_sensitivity",
      "no_extreme_downsampling",
      "temporal_frame_validated",
      "orientation_contract_validated",
      "geometry_model_ready",
      "model_input_pipeline_validated"
    ],
    "large_ld_gate_scope": "WORST_VISIT_AND_VIEW",
    "large_ld_primary_definition": "PRE_REGISTERED_CROP_SUSPECTED_TRUNCATION_ON_AVAILABLE_SUPPORT",
    "observed": {
      "ftv_retention_q05": 1.0,
      "geometry_model_ready_fraction": 0.952,
      "max_cardinal_obliquity_deg": 12.500000923929592,
      "max_resize_anisotropy_ratio": 3.4722177725756236,
      "max_resize_factor": 2.88,
      "overall_exact_full_support_rate": 0.9793333333333333,
      "overall_sufficient_containment_rate": 0.9793333333333333,
      "source_uncensored_and_exact_rate": 0.9706666666666667,
      "surface_retention_q05": 1.0,
      "top_10pct_suspected_truncation_rate": 0.05263157894736842,
      "top_10pct_upstream_censoring_adjusted_suspected_rate": 0.10526315789473684,
      "top_quartile_suspected_truncation_rate": 0.04950495049504951,
      "top_quartile_upstream_censoring_adjusted_suspected_rate": 0.08163265306122448
    },
    "passed": false,
    "source_boundary_sensitivity_does_not_redefine_primary_metrics": true,
    "view": "detail"
  },
  "C2A": {
    "checks": {
      "anisotropy": true,
      "causally_deployable": true,
      "ftv_retention_q05": true,
      "geometry_model_ready": false,
      "large_ld_top_10pct": true,
      "large_ld_top_quartile": true,
      "model_input_pipeline_validated": false,
      "morphology_surface_q05": true,
      "no_direct_geometry_input": true,
      "no_extreme_downsampling": false,
      "orientation_contract_validated": false,
      "overall_exact_full_support": true,
      "overall_sufficient_containment": true,
      "source_boundary_observability": true,
      "temporal_frame_validated": false,
      "temporal_no_recentering_normalization": false,
      "upstream_acquisition_large_ld_sensitivity": false
    },
    "contract": "C2A",
    "evaluated_views": [
      "detail",
      "context"
    ],
    "failed_checks": [
      "upstream_acquisition_large_ld_sensitivity",
      "no_extreme_downsampling",
      "temporal_no_recentering_normalization",
      "temporal_frame_validated",
      "orientation_contract_validated",
      "geometry_model_ready",
      "model_input_pipeline_validated"
    ],
    "large_ld_gate_scope": "WORST_VISIT_AND_VIEW",
    "large_ld_primary_definition": "PRE_REGISTERED_CROP_SUSPECTED_TRUNCATION_ON_AVAILABLE_SUPPORT",
    "observed": {
      "ftv_retention_q05": 1.0,
      "geometry_model_ready_fraction": 0.952,
      "max_cardinal_obliquity_deg": 12.500000923929592,
      "max_resize_anisotropy_ratio": 3.4722177725756236,
      "max_resize_factor": 4.8,
      "overall_exact_full_support_rate": 1.0,
      "overall_sufficient_containment_rate": 1.0,
      "source_uncensored_and_exact_rate": 0.9906666666666667,
      "surface_retention_q05": 1.0,
      "top_10pct_suspected_truncation_rate": 0.0,
      "top_10pct_upstream_censoring_adjusted_suspected_rate": 0.07894736842105263,
      "top_quartile_suspected_truncation_rate": 0.0,
      "top_quartile_upstream_censoring_adjusted_suspected_rate": 0.05102040816326531
    },
    "passed": false,
    "source_boundary_sensitivity_does_not_redefine_primary_metrics": true,
    "view": "detail+context"
  },
  "C2B": {
    "checks": {
      "anisotropy": true,
      "causally_deployable": true,
      "ftv_retention_q05": true,
      "geometry_model_ready": false,
      "large_ld_top_10pct": true,
      "large_ld_top_quartile": true,
      "model_input_pipeline_validated": false,
      "morphology_surface_q05": true,
      "no_direct_geometry_input": true,
      "no_extreme_downsampling": false,
      "orientation_contract_validated": false,
      "overall_exact_full_support": true,
      "overall_sufficient_containment": true,
      "source_boundary_observability": true,
      "temporal_frame_validated": false,
      "temporal_no_recentering_normalization": true,
      "upstream_acquisition_large_ld_sensitivity": false
    },
    "contract": "C2B",
    "evaluated_views": [
      "detail",
      "context"
    ],
    "failed_checks": [
      "upstream_acquisition_large_ld_sensitivity",
      "no_extreme_downsampling",
      "temporal_frame_validated",
      "orientation_contract_validated",
      "geometry_model_ready",
      "model_input_pipeline_validated"
    ],
    "large_ld_gate_scope": "WORST_VISIT_AND_VIEW",
    "large_ld_primary_definition": "PRE_REGISTERED_CROP_SUSPECTED_TRUNCATION_ON_AVAILABLE_SUPPORT",
    "observed": {
      "ftv_retention_q05": 1.0,
      "geometry_model_ready_fraction": 0.952,
      "max_cardinal_obliquity_deg": 12.500000923929592,
      "max_resize_anisotropy_ratio": 3.4722177725756236,
      "max_resize_factor": 4.8,
      "overall_exact_full_support_rate": 0.9793333333333333,
      "overall_sufficient_containment_rate": 0.9793333333333333,
      "source_uncensored_and_exact_rate": 0.9706666666666667,
      "surface_retention_q05": 1.0,
      "top_10pct_suspected_truncation_rate": 0.05263157894736842,
      "top_10pct_upstream_censoring_adjusted_suspected_rate": 0.10526315789473684,
      "top_quartile_suspected_truncation_rate": 0.04950495049504951,
      "top_quartile_upstream_censoring_adjusted_suspected_rate": 0.08163265306122448
    },
    "passed": false,
    "source_boundary_sensitivity_does_not_redefine_primary_metrics": true,
    "view": "detail+context"
  }
}
```

## 最终input recommendation

- 主要deployable方向：T0-anchored fixed physical detail（C1B），但状态只是设计候选，不是model-ready输入。真实5-case二维preview通过finite/non-constant sanity；它不替代完整3-D DCE7验收。
- audit upper bound：C1C ORACLE-UNION，永不进入训练loader。
- 资源边界：C1B四访DCE7 float32 input约353 MB/患者（legacy的10.7倍）；C2B约500 MB（15.1倍），均未计activations/optimizer。C2B context的高padding与重采样风险不支持升级为首选。
- 必须先完成：72个奇异DCE sform visit的raw-DICOM pixel rebuild验收、source-edge病例复核、T0 frame的image-only rigid-registration sensitivity，以及完整3-D DCE7 builder/normalization/cache验收；不得用future lesion mask修补C1B miss。
- model batch只含DCE7 image tensor；所有geometry metadata均留在sidecar。

本实验的结论不是“为了LD把crop变大”，而是：current fixed voxel-space crop不保证clinically meaningful response target可从图像观察，必须先建立Response-Observable Image State。
