# Mask-Free Region-Aware Representation Audit 最终报告

## 结论先行

本 frozen-feature diagnostic 的唯一科学分类是 **REGION_SIGNAL_EXISTS_BUT_NOT_BEYOND_FTV**。四个预注册门控如下：

- Gate A：通过
- Gate B：未通过
- Gate C：未通过
- Gate D：未通过

本报告中的 “mask-free” 是严格限定语：**mask-free-at-readout-only**。新增区域 readout 不读取 lesion mask、bbox、FTV、clinical、pCR、phenotype 或 future visit；但上游输入是 **T0 localization-centered C1B** crops，因此它不是 acquisition-centered、端到端 mask-independent deployment 验证。

## 逐项回答 12 个问题

### Q1 — 哪个 mask-free region 最好？

按注册 MRI-only pCR cells 的描述性平均 R0 增量，最好的是 **R3**（平均 ΔAUROC `0.002`）。这是内部 OOF 汇总，不是事后选择新 primary，也不改变预注册 gates。

### Q2 — Central / Inner / Outer 哪个更有用？

三者的描述性最佳项是 **R3**（平均 ΔAUROC `0.002`；R1=Central、R2=Inner、R3=Outer）。该排序只描述冻结 probe，不把 shell 解码性解释为组织来源。

### Q3 — Three-region representation 是否优于 Full Local？

R5 相对 R0 的注册 cells 平均 ΔAUROC 为 `-0.017`；因此描述性答案为 **否**。正式支持仍由 Gate A（当前通过）决定，而非单个平均值。

### Q4 — pCR 是否改善？

最佳 mask-free candidate 的 MRI-only pCR 平均增量为 `0.002`；Gate A **通过**。这仅说明 frozen representation 中的可解码信息变化，不等于治疗反应机制或外部临床效用。

### Q5 — Clinical+FTV 后是否仍有增量？

C+F+Rk 相对 C+F+R0 的最佳描述性项为 **R3**（平均 ΔAUROC `-0.001`）；Gate B **未通过**。Gate B 同时约束了相对 C+F baseline 不系统性为负，不能用单个正 cell 替代该判断。

### Q6 — HR/HER2/subtype 是否改善？

相对 R0 的 phenotype 汇总中最佳项为 **R1**（平均 ΔAUROC `-0.002`）；Gate D **未通过**。Gate D 未通过时，pCR regional signal 不得称为 molecular phenotype；即使通过，也只能称 profile-associated decoding，不能称分子机制。

### Q7 — FTV / ΔFTV 是否改善？

在 `primary_measurement_valid` 汇总和 `r2` 口径下，FTV 的 R5=`0.108`、R0=`-0.128`、改善量=`0.237`，因此 **改善**；ΔFTV 的 R5=`0.104`、R0=`0.106`、改善量=`-0.002`，因此 **未改善**。两项必须分开回答，不能用 FTV 的正结果代替 ΔFTV。它们是 burden/response 的 Ridge-probe diagnostics，不是影像测量替代品，也不证明生物学过程。

### Q8 — 是否部分恢复 Goal 5 PERI20 Oracle gain？

最佳注册 candidate 是 **R2**，平均 recovery ratio 为 `-0.127`；Gate C **未通过**。ratio 只在 matched、Oracle uplift 为正的 LOCAL0/T0-T1 cells 定义。

### Q9 — Oracle gain 是否必须依赖 lesion-relative geometry？

固定坐标 readout 未达到部分恢复门槛，结果与 lesion-relative localization 可能重要一致；但失败不能证明它是必要条件，learned localization 仍只是待检验假设。

必须突出 **Oracle denominator representation mismatch**：新 numerator 是 `Rk(raw regional means) - R0(raw full-local mean)`，冻结 denominator 是 Goal 5 `PERI20(mean+std) - FIXED_P3(full-local mean+std)`。因此 recovery ratio 是预注册诊断桥接量，不是同构表示之间的因果分解。

### Q10 — 下一步应保持 Full Local、fixed region-aware、learned localization 还是 lesion-relative region？

当前唯一选择是 **保持 Full Local**。该选择由预注册 classification 映射产生，不根据 test set 重选半径、region 或模型容量。

### Q11 — 是否值得训练正式 Region-Aware Response State？

不值得据此启动正式 Region-Aware Response State 训练；保留 Full Local，等待独立证据。 禁止从本 Goal 自动训练 region-token JEPA、attention、MIL、segmentation-guided branch 或任何主 encoder/JEPA。

### Q12 — 哪些结论必须保持 diagnostic，不能写成 biological claim？

这是 **diagnostic/non-biological boundary**：AUROC、FTV/ΔFTV 可解码性、central/shell 排序和 Oracle recovery 都只是 frozen、内部 OOF representation diagnostics。不得写成统计显著性、外部泛化、因果疗效、peritumoral biology、molecular mechanism 或端到端 mask-independent deployment。限制包括粗 Z spacing、大 receptive field、complete-four-visit selection，以及上游 T0-centered localization。`T0-T3` 必须始终解释为 **T3 late/pre-surgery**，不能与 early/mid timing 合并宣传。

## 时间、几何与比较口径

- Primary physical partition 固定为 32/48/64 mm；secondary 24/40/64 mm 不参与 primary classification。
- 所有区域使用 fractional feature-cell occupancy weighted pooling，而非 nearest-voxel binary assignment。
- pCR causal prefixes 为 T0、T0-T1、T0-T2、T0-T3；最后一项永久标记 late/pre-surgery。
- patient-level bootstrap 是 outer-fold 内 paired resampling；公开文件只保留 aggregate。
- Goal 5 absolute AUROC 只在同一 matched population 内比较，不跨 variant-specific population。

## 必附图表

### Region schematic

[区域示意图](../figures/01_region_schematic.png)

### Region occupancy statistics

[完整 CSV](../metrics/region_occupancy.csv) · [图](../figures/02_region_occupancy.png)

| geometry | region | variant | definition | inner_boundary_mm | outer_boundary_mm | mean_effective_cells | weight_sum_cells | physical_volume_mm3 | expected_physical_volume_mm3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_and_secondary | R0 | R0 | full_local_cube | 0 | 64 | 316.05 | 316.05 | 2.6214e+05 | 2.6214e+05 |
| primary | R1 | R1 | central_cube | 0 | 32 | 39.506 | 39.506 | 32768 | 32768 |
| primary | R2 | R2 | inner_shell | 32 | 48 | 93.827 | 93.827 | 77824 | 77824 |
| primary | R3 | R3 | outer_shell | 48 | 64 | 182.72 | 182.72 | 1.5155e+05 | 1.5155e+05 |
| secondary | S1 | S1 | central_cube | 0 | 24 | 16.667 | 16.667 | 13824 | 13824 |
| secondary | S2 | S2 | inner_shell | 24 | 40 | 60.494 | 60.494 | 50176 | 50176 |
| secondary | S3 | S3 | outer_shell | 40 | 64 | 238.89 | 238.89 | 1.9814e+05 | 1.9814e+05 |

（预览 7/7 行、10/13 列；完整表见链接。）

### MRI-only pCR

[完整 CSV](../metrics/table_mri_only_pcr.csv) · [图](../figures/03_mri_only_pcr.png)

| seed | arm | analysis | context | view | timing_label | target | variant | model | population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R0 | R0 | ftv_complete_375 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R0 | R0 | full_808 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R1 | R1 | ftv_complete_375 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R1 | R1 | full_808 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R2 | R2 | ftv_complete_375 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R2 | R2 | full_808 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R3 | R3 | ftv_complete_375 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R3 | R3 | full_808 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R4 | R4 | ftv_complete_375 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R4 | R4 | full_808 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R5 | R5 | ftv_complete_375 |
| 2026 | LOCAL0 | mri_only_pcr | MRI_ONLY | T0 | NA | pCR | R5 | R5 | full_808 |

（预览 12/384 行、10/23 列；完整表见链接。）

### Phenotype probes

[完整 CSV](../metrics/table_phenotype.csv) · [图](../figures/04_phenotype_probes.png)

| seed | arm | analysis | context | view | timing_label | target | variant | model | population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | R0 | R0 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | R1 | R1 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | R2 | R2 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | R3 | R3 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | R4 | R4 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | R5 | R5 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | R5_RP192 | R5_RP192 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | S1 | S1 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | S2 | S2 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | S3 | S3 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | S4 | S4 | full_808 |
| 2026 | LOCAL0 | phenotype | MRI_ONLY_STATIC | T0 | NA | HER2 | S5 | S5 | full_808 |

（预览 12/576 行、10/22 列；完整表见链接。）

### C+F incremental

[完整 CSV](../metrics/table_clinical_ftv_incremental.csv) · [图](../figures/05_clinical_ftv_incremental.png)

| seed | arm | analysis | context | view | timing_label | target | variant | model | population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | NONE | C+F | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | R0 | C+F+R0 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | R1 | C+F+R1 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | R2 | C+F+R2 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | R3 | C+F+R3 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | R4 | C+F+R4 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | R5 | C+F+R5 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | R5_RP192 | C+F+R5_RP192 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | S1 | C+F+S1 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | S2 | C+F+S2 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | S3 | C+F+S3 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_ftv_pcr | C_PLUS_F | T0 | NA | pCR | S4 | C+F+S4 | ftv_complete_375 |

（预览 12/208 行、10/28 列；完整表见链接。）

### FTV / delta-FTV

[完整 CSV](../metrics/table_ftv.csv) · [图](../figures/06_ftv_response_control.png)

| seed | arm | variant | feature_dim | task | endpoint | view | target | analysis_scope | target_semantics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | LOCAL0 | R0 | 128 | delta | T0_to_T1 | T0_to_T1 | delta_FTV | observable_only | literal_ftv_end_minus_ftv_start |
| 2026 | LOCAL0 | R0 | 128 | delta | T0_to_T1 | T0_to_T1 | delta_FTV | primary_measurement_valid | literal_ftv_end_minus_ftv_start |
| 2026 | LOCAL0 | R0 | 128 | delta | T1_to_T2 | T1_to_T2 | delta_FTV | observable_only | literal_ftv_end_minus_ftv_start |
| 2026 | LOCAL0 | R0 | 128 | delta | T1_to_T2 | T1_to_T2 | delta_FTV | primary_measurement_valid | literal_ftv_end_minus_ftv_start |
| 2026 | LOCAL0 | R0 | 128 | delta | T2_to_T3 | T2_to_T3 | delta_FTV | observable_only | literal_ftv_end_minus_ftv_start |
| 2026 | LOCAL0 | R0 | 128 | delta | T2_to_T3 | T2_to_T3 | delta_FTV | primary_measurement_valid | literal_ftv_end_minus_ftv_start |
| 2026 | LOCAL0 | R0 | 128 | static | T0 | T0 | FTV | observable_only | static_ftv_log_winsor_median_iqr_inverse_natural |
| 2026 | LOCAL0 | R0 | 128 | static | T0 | T0 | FTV | primary_measurement_valid | static_ftv_log_winsor_median_iqr_inverse_natural |
| 2026 | LOCAL0 | R0 | 128 | static | T1 | T1 | FTV | observable_only | static_ftv_log_winsor_median_iqr_inverse_natural |
| 2026 | LOCAL0 | R0 | 128 | static | T1 | T1 | FTV | primary_measurement_valid | static_ftv_log_winsor_median_iqr_inverse_natural |
| 2026 | LOCAL0 | R0 | 128 | static | T2 | T2 | FTV | observable_only | static_ftv_log_winsor_median_iqr_inverse_natural |
| 2026 | LOCAL0 | R0 | 128 | static | T2 | T2 | FTV | primary_measurement_valid | static_ftv_log_winsor_median_iqr_inverse_natural |

（预览 12/864 行、10/24 列；完整表见链接。）

### Oracle recovery

[完整 CSV](../metrics/table_oracle_recovery.csv) · [图](../figures/07_oracle_recovery.png)

| row_type | seed | arm | view | target | population | source | variant | candidate | reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_metric | 2026 | LOCAL0 | T0-T1 | pCR | oracle_pair_PERI20 | Goal5_Oracle | FIXED_P3 | NA | NA |
| matched_metric | 2026 | LOCAL0 | T0-T1 | pCR | oracle_pair_PERI10 | Goal5_Oracle | PERI10 | NA | NA |
| matched_metric | 2026 | LOCAL0 | T0-T1 | pCR | oracle_pair_PERI20 | Goal5_Oracle | PERI20 | NA | NA |
| matched_metric | 2026 | LOCAL0 | T0-T1 | pCR | oracle_pair_CORE | Goal5_Oracle | CORE | NA | NA |
| matched_metric | 2026 | LOCAL0 | T0-T1 | pCR | oracle_pair_CORE_PERI | Goal5_Oracle | CORE_PERI | NA | NA |
| matched_metric | 3026 | LOCAL0 | T0-T1 | pCR | oracle_pair_PERI20 | Goal5_Oracle | FIXED_P3 | NA | NA |
| matched_metric | 3026 | LOCAL0 | T0-T1 | pCR | oracle_pair_PERI10 | Goal5_Oracle | PERI10 | NA | NA |
| matched_metric | 3026 | LOCAL0 | T0-T1 | pCR | oracle_pair_PERI20 | Goal5_Oracle | PERI20 | NA | NA |
| matched_metric | 3026 | LOCAL0 | T0-T1 | pCR | oracle_pair_CORE | Goal5_Oracle | CORE | NA | NA |
| matched_metric | 3026 | LOCAL0 | T0-T1 | pCR | oracle_pair_CORE_PERI | Goal5_Oracle | CORE_PERI | NA | NA |
| matched_metric | 2026 | LOCAL0 | T0-T1 | pCR | ftv_complete_375 | MaskFree | R0 | NA | NA |
| matched_metric | 2026 | LOCAL0 | T0-T1 | pCR | ftv_complete_375 | MaskFree | R1 | NA | NA |

（预览 12/30 行、10/27 列；完整表见链接。）

### Patient-level bootstrap

[完整 CSV](../metrics/table_bootstrap.csv) · [图](../figures/08_patient_bootstrap.png)

| seed | arm | context | view | timing | target | population | candidate | reference_model | comparison_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | pCR | full_808 | R2 | R0 | R2 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | pCR | full_808 | R2 | R0 | R2 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | pCR | full_808 | R2 | R0 | R2 |
| 2026 | LOCAL0 | C_PLUS_F | T0 | T0 | pCR | ftv_complete_375 | R2 | C+F+R0 | C+F+R2 |
| 2026 | LOCAL0 | C_PLUS_F | T0 | T0 | pCR | ftv_complete_375 | R2 | C+F+R0 | C+F+R2 |
| 2026 | LOCAL0 | C_PLUS_F | T0 | T0 | pCR | ftv_complete_375 | R2 | C+F+R0 | C+F+R2 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | pCR | full_808 | R3 | R0 | R3 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | pCR | full_808 | R3 | R0 | R3 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | pCR | full_808 | R3 | R0 | R3 |
| 2026 | LOCAL0 | C_PLUS_F | T0 | T0 | pCR | ftv_complete_375 | R3 | C+F+R0 | C+F+R3 |
| 2026 | LOCAL0 | C_PLUS_F | T0 | T0 | pCR | ftv_complete_375 | R3 | C+F+R0 | C+F+R3 |
| 2026 | LOCAL0 | C_PLUS_F | T0 | T0 | pCR | ftv_complete_375 | R3 | C+F+R0 | C+F+R3 |

（预览 12/222 行、10/27 列；完整表见链接。）

### Seed consistency

[完整 CSV](../metrics/table_seed_consistency.csv) · [图](../figures/09_seed_consistency.png)

| context | arm | view | timing | target | population | candidate | reference | seed_2026_delta_auroc | seed_3026_delta_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MRI_ONLY | LOCAL0 | T0 | T0 | pCR | full_808 | R2 | R0 | -0.017063 | -0.0012963 |
| MRI_ONLY | LOCAL0 | T0-T1 | T0-T1 | pCR | full_808 | R2 | R0 | 0.0069657 | -0.0010302 |
| MRI_ONLY | LOCAL0 | T0-T2 | T0-T2 | pCR | full_808 | R2 | R0 | -0.011666 | 0.0049872 |
| MRI_ONLY | LOCAL3 | T0 | T0 | pCR | full_808 | R2 | R0 | -0.0023128 | 0.0086577 |
| MRI_ONLY | LOCAL3 | T0-T1 | T0-T1 | pCR | full_808 | R2 | R0 | -0.0088965 | -6.8224e-06 |
| MRI_ONLY | LOCAL3 | T0-T2 | T0-T2 | pCR | full_808 | R2 | R0 | -0.0013781 | -0.017329 |
| MRI_ONLY | LOCAL0 | T0 | T0 | pCR | full_808 | R3 | R0 | 0.03839 | -0.010391 |
| MRI_ONLY | LOCAL0 | T0-T1 | T0-T1 | pCR | full_808 | R3 | R0 | 0.031527 | -0.0038479 |
| MRI_ONLY | LOCAL0 | T0-T2 | T0-T2 | pCR | full_808 | R3 | R0 | 0.027556 | -0.021129 |
| MRI_ONLY | LOCAL3 | T0 | T0 | pCR | full_808 | R3 | R0 | 0.026887 | -0.012069 |
| MRI_ONLY | LOCAL3 | T0-T1 | T0-T1 | pCR | full_808 | R3 | R0 | 0.017575 | -0.013379 |
| MRI_ONLY | LOCAL3 | T0-T2 | T0-T2 | pCR | full_808 | R3 | R0 | 0.041999 | -0.01531 |

（预览 12/36 行、10/12 列；完整表见链接。）

### Timing sensitivity

[完整 CSV](../metrics/table_timing_sensitivity.csv) · [图](../figures/10_timing_sensitivity.png)

| seed | arm | context | view | timing | timing_label | target | population | variant | reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | ftv_complete_375 | R0 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | ftv_complete_375 | R1 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | ftv_complete_375 | R2 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | ftv_complete_375 | R3 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | ftv_complete_375 | R4 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | ftv_complete_375 | R5 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | full_808 | R0 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | full_808 | R1 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | full_808 | R2 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | full_808 | R3 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | full_808 | R4 | R0 |
| 2026 | LOCAL0 | MRI_ONLY | T0 | T0 | NA | pCR | full_808 | R5 | R0 |

（预览 12/288 行、10/13 列；完整表见链接。）

### Clinical pCR 辅助表

[完整 CSV](../metrics/table_clinical_pcr.csv)

| seed | arm | analysis | context | view | timing_label | target | variant | model | population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | NONE | C | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | R0 | C+R0 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | R1 | C+R1 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | R2 | C+R2 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | R3 | C+R3 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | R4 | C+R4 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | R5 | C+R5 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | R5_RP192 | C+R5_RP192 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | S1 | C+S1 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | S2 | C+S2 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | S3 | C+S3 | ftv_complete_375 |
| 2026 | LOCAL0 | clinical_pcr | C_PLUS_R | T0 | NA | pCR | S4 | C+S4 | ftv_complete_375 |

（预览 12/208 行、10/26 列；完整表见链接。）

## 执行与 Git 记录

- Branch：`feature/mask-free-region-aware-audit`
- Commit SHA：`177c976e511bce7684fa1f5b0c744656fb3675f8`
- Push status：`PUSHED`
- Push error：`null`
- Formal elapsed seconds：`1344.6029386520386`
- Run status：`COMPLETED`

如果 push 失败，正式 run summary 必须把状态写为 `GITHUB_PUSH_FAILED` 并保留真实错误；禁止 force push。报告与图只读取公开 aggregate metrics、gates 和 run summary，没有读取 private predictions 或 labels。
