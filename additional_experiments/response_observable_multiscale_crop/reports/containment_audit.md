# 病灶包含性审计

## 口径

新contract在source physical domain检查完整voxel footprint，不用resampled mask voxel count替代full-support retention。FTV inclusion support只作为observability proxy；它不是radiologist target lesion的dense segmentation。LD保持source raw unit，只按每个visit分别计算Q75/Q90并用`>=`包含ties。

预注册的`boundary_touch/suspected/sufficient/exact`只衡量**现有acquisition内可用support相对crop**的保留，不在看到结果后改写。另行加入上游acquisition-boundary sensitivity：14/1500个visit的support接触source image face。它不被悄悄并入primary crop rate，但作为完整GO的独立blocker；因此“available-support crop通过”不能被表述为clinical whole lesion已经确定完整可观察。

## Overall

| contract | view | n | boundary_touch_rate | suspected_truncation_rate | severe_truncation_rate | sufficient_containment_rate | exact_full_support_containment_rate | source_boundary_touch_rate | source_uncensored_and_exact_rate | retained_ftv_fraction_q05 | minimum_margin_mm_q05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | 1500 | 0.7767 | 0.8133 | 0.5460 | 0.1867 | 0.2047 | 0.0093 | 0.2047 | 0.1562 | -42.0000 |
| C1A | detail | 1500 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 1.0000 | 20.1126 |
| C1B | detail | 1500 | 0.0200 | 0.0207 | 0.0040 | 0.9793 | 0.9793 | 0.0093 | 0.9707 | 1.0000 | 8.6935 |
| C1C | detail | 1500 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 1.0000 | 14.1886 |
| C2A | detail | 1500 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 1.0000 | 20.1126 |
| C2A | context | 1500 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 1.0000 | 36.6147 |
| C2B | detail | 1500 | 0.0200 | 0.0207 | 0.0040 | 0.9793 | 0.9793 | 0.0093 | 0.9707 | 1.0000 | 8.6935 |
| C2B | context | 1500 | 0.0087 | 0.0087 | 0.0020 | 0.9913 | 0.9907 | 0.0093 | 0.9820 | 1.0000 | 23.2748 |

## 按 visit

| contract | view | scope | n | suspected_truncation_rate | severe_truncation_rate | sufficient_containment_rate | exact_full_support_containment_rate | retained_ftv_fraction_q05 | minimum_margin_mm_q05 |
|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | T0 | 375 | 0.7067 | 0.3707 | 0.2933 | 0.3200 | 0.5315 | -30.0000 |
| C0 | legacy | T1 | 375 | 0.8427 | 0.5360 | 0.1573 | 0.1707 | 0.1985 | -42.5066 |
| C0 | legacy | T2 | 375 | 0.8427 | 0.6160 | 0.1573 | 0.1653 | 0.1207 | -42.9009 |
| C0 | legacy | T3 | 375 | 0.8613 | 0.6613 | 0.1387 | 0.1627 | 0.0851 | -45.7601 |
| C1A | detail | T0 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 20.8660 |
| C1A | detail | T1 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 20.1338 |
| C1A | detail | T2 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 19.3782 |
| C1A | detail | T3 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 20.0206 |
| C1B | detail | T0 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 20.8660 |
| C1B | detail | T1 | 375 | 0.0213 | 0.0027 | 0.9787 | 0.9787 | 1.0000 | 11.0555 |
| C1B | detail | T2 | 375 | 0.0320 | 0.0027 | 0.9680 | 0.9680 | 1.0000 | 3.4168 |
| C1B | detail | T3 | 375 | 0.0293 | 0.0107 | 0.9707 | 0.9707 | 1.0000 | 4.2591 |
| C1C | detail | T0 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 13.9227 |
| C1C | detail | T1 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 15.1808 |
| C1C | detail | T2 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 14.6379 |
| C1C | detail | T3 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 13.8543 |
| C2B | detail | T0 | 375 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 20.8660 |
| C2B | detail | T1 | 375 | 0.0213 | 0.0027 | 0.9787 | 0.9787 | 1.0000 | 11.0555 |
| C2B | detail | T2 | 375 | 0.0320 | 0.0027 | 0.9680 | 0.9680 | 1.0000 | 3.4168 |
| C2B | detail | T3 | 375 | 0.0293 | 0.0107 | 0.9707 | 0.9707 | 1.0000 | 4.2591 |

## Large-LD subgroup

| contract | view | subgroup | n | suspected_truncation_rate | upstream_censoring_adjusted_suspected_rate | source_boundary_censored_rate | severe_truncation_rate | sufficient_containment_rate | exact_full_support_containment_rate | retained_ftv_fraction_q05 | minimum_margin_mm_median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | LD_TOP_QUARTILE | 394 | 0.9569 | 0.9569 | 0.0228 | 0.7183 | 0.0431 | 0.0457 | 0.2212 | -21.6000 |
| C0 | legacy | LD_TOP_10PCT | 154 | 1.0000 | 1.0000 | 0.0390 | 0.8506 | 0.0000 | 0.0000 | 0.1937 | -26.4000 |
| C1A | detail | LD_TOP_QUARTILE | 394 | 0.0000 | 0.0228 | 0.0228 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 35.7769 |
| C1A | detail | LD_TOP_10PCT | 154 | 0.0000 | 0.0390 | 0.0390 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 31.7094 |
| C1B | detail | LD_TOP_QUARTILE | 394 | 0.0228 | 0.0431 | 0.0228 | 0.0051 | 0.9772 | 0.9772 | 1.0000 | 30.4435 |
| C1B | detail | LD_TOP_10PCT | 154 | 0.0195 | 0.0519 | 0.0390 | 0.0000 | 0.9805 | 0.9805 | 1.0000 | 24.8859 |
| C1C | detail | LD_TOP_QUARTILE | 394 | 0.0000 | 0.0228 | 0.0228 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 31.2824 |
| C1C | detail | LD_TOP_10PCT | 154 | 0.0000 | 0.0390 | 0.0390 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 26.3992 |
| C2A | detail | LD_TOP_QUARTILE | 394 | 0.0000 | 0.0228 | 0.0228 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 35.7769 |
| C2A | detail | LD_TOP_10PCT | 154 | 0.0000 | 0.0390 | 0.0390 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 31.7094 |
| C2B | detail | LD_TOP_QUARTILE | 394 | 0.0228 | 0.0431 | 0.0228 | 0.0051 | 0.9772 | 0.9772 | 1.0000 | 30.4435 |
| C2B | detail | LD_TOP_10PCT | 154 | 0.0195 | 0.0519 | 0.0390 | 0.0000 | 0.9805 | 0.9805 | 1.0000 | 24.8859 |

`suspected_truncation_rate`是预注册的crop-specific primary；`upstream_censoring_adjusted_suspected_rate`是将source-face touch并入后的保守敏感性。后者用于阻止过强的end-to-end observability声明，不追溯改写primary metric或margin选择。

## FTV-specific observability

voxel retention与physical-volume retention均在source domain计算；extent retention为保留support physical extent相对完整support的最小轴比例。

| contract | view | n | median | q05 | q25 | minimum | physical_volume_retention_median | physical_volume_retention_q05 | physical_volume_retention_q25 | physical_volume_retention_minimum | extent_retention_min_axis_median | extent_retention_min_axis_q05 | extent_retention_min_axis_q25 | extent_retention_min_axis_minimum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | 1500 | 0.8612 | 0.1562 | 0.5802 | 0.0000 | 0.8612 | 0.1562 | 0.5802 | 0.0000 | 0.6848 | 0.3200 | 0.5367 | 0.0133 |
| C1A | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C1B | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 0.1631 | 1.0000 | 1.0000 | 1.0000 | 0.1631 | 1.0000 | 1.0000 | 1.0000 | 0.4592 |
| C1C | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C2A | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C2A | context | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C2B | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 0.1631 | 1.0000 | 1.0000 | 1.0000 | 0.1631 | 1.0000 | 1.0000 | 1.0000 | 0.4592 |
| C2B | context | 1500 | 1.0000 | 1.0000 | 1.0000 | 0.2232 | 1.0000 | 1.0000 | 1.0000 | 0.2232 | 1.0000 | 1.0000 | 1.0000 | 0.5306 |

## LD rank sanity

reported LD保持source raw unit，不做mm换算。这里同时报告完整FTV inclusion support physical bbox最大轴与largest-component approximate extent；前者受远端碎片影响，后者忽略其他component，二者都只作rank proxy，均不等同radiologist target lesion。

| scope | n | ld_zero_fraction | ld_full_support_bbox_extent_spearman | ld_full_support_bbox_extent_pvalue | ld_full_support_bbox_extent_n | ld_largest_component_extent_spearman | ld_largest_component_extent_pvalue | ld_largest_component_extent_n | full_support_bbox_extent_median_mm | full_support_bbox_extent_q95_mm | largest_component_extent_median_mm | largest_component_extent_q95_mm | extent_proxy | ld_unit_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OVERALL | 1500 | 0.1260 | 0.3538 | 0.0000 | 1500 | 0.5932 | 0.0000 | 1500 | 63.8066 | 120.9998 | 47.0669 | 123.2333 | FULL_SUPPORT_BBOX_AND_LARGEST_COMPONENT_APPROXIMATE_EXTENT_NOT_RADIOLOGIST_TARGET | LD_SOURCE_UNIT_NOT_EXPLICIT_NO_CONVERSION |
| T0_T1 | 750 | 0.0067 | 0.5049 | 0.0000 | 750 | 0.5994 | 0.0000 | 750 | 65.0818 | 120.9999 | 62.5817 | 131.7069 | FULL_SUPPORT_BBOX_AND_LARGEST_COMPONENT_APPROXIMATE_EXTENT_NOT_RADIOLOGIST_TARGET | LD_SOURCE_UNIT_NOT_EXPLICIT_NO_CONVERSION |
| T0 | 375 | 0.0000 | 0.5782 | 0.0000 | 375 | 0.5948 | 0.0000 | 375 | 65.0818 | 121.7893 | 67.6419 | 134.9872 | FULL_SUPPORT_BBOX_AND_LARGEST_COMPONENT_APPROXIMATE_EXTENT_NOT_RADIOLOGIST_TARGET | LD_SOURCE_UNIT_NOT_EXPLICIT_NO_CONVERSION |
| T1 | 375 | 0.0133 | 0.4466 | 0.0000 | 375 | 0.5652 | 0.0000 | 375 | 65.0818 | 119.1797 | 56.1306 | 124.2328 | FULL_SUPPORT_BBOX_AND_LARGEST_COMPONENT_APPROXIMATE_EXTENT_NOT_RADIOLOGIST_TARGET | LD_SOURCE_UNIT_NOT_EXPLICIT_NO_CONVERSION |
| T2 | 375 | 0.1653 | 0.3050 | 0.0000 | 375 | 0.4322 | 0.0000 | 375 | 62.5781 | 121.1484 | 36.8317 | 115.5104 | FULL_SUPPORT_BBOX_AND_LARGEST_COMPONENT_APPROXIMATE_EXTENT_NOT_RADIOLOGIST_TARGET | LD_SOURCE_UNIT_NOT_EXPLICIT_NO_CONVERSION |
| T3 | 375 | 0.3253 | 0.1329 | 0.0100 | 375 | 0.2509 | 0.0000 | 375 | 59.4000 | 120.0703 | 26.1148 | 105.8400 | FULL_SUPPORT_BBOX_AND_LARGEST_COMPONENT_APPROXIMATE_EXTENT_NOT_RADIOLOGIST_TARGET | LD_SOURCE_UNIT_NOT_EXPLICIT_NO_CONVERSION |

LD与minimum crop margin的overall秩相关如下；T0–T3逐visit结果保存在公开`containment_summary.csv`。

| contract | view | scope | ld_margin_spearman | ld_margin_pvalue | ld_margin_n |
|---|---|---|---|---|---|
| C0 | legacy | OVERALL | -0.1290 | 0.0000 | 1488 |
| C1A | detail | OVERALL | -0.3751 | 0.0000 | 1500 |
| C1B | detail | OVERALL | -0.1880 | 0.0000 | 1500 |
| C1C | detail | OVERALL | -0.3202 | 0.0000 | 1500 |
| C2B | detail | OVERALL | -0.1880 | 0.0000 | 1500 |

## Margin sensitivity

margin候选在任何pCR/model结果之前冻结。四档均实际审计C1A与C1B；选择规则是在所有observability gate通过后，取使overflow/scale change最少的最小候选，因此主分析使用5mm。该选择不声称5mm是唯一最优margin。

| strategy | margin_mm | n | n_patients | exact_full_support_containment_rate | source_boundary_uncensored_rate | sufficient_containment_rate | suspected_truncation_rate | large_ld_top_quartile_worst_visit_suspected_rate | large_ld_top_10pct_worst_visit_suspected_rate | retained_ftv_fraction_q05 | minimum_margin_q05_mm | expanded_from_nominal_rate | expanded_patient_count_public | expanded_count_status | extreme_axis_factor_gt2_rate | resize_anisotropy_max | fov_x_mm_median | fov_y_mm_median | fov_z_mm_median | fov_volume_liter_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1A | 5.0000 | 1500 | 375 | 1.0000 | 0.9907 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20.1126 | NA | <5 | SUPPRESSED_LT_5 | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |
| C1A | 10.0000 | 1500 | 375 | 1.0000 | 0.9907 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20.1126 | 0.0120 | 7 | REPORTED | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |
| C1A | 15.0000 | 1500 | 375 | 1.0000 | 0.9907 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20.1126 | 0.0267 | 17 | REPORTED | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |
| C1A | 20.0000 | 1500 | 375 | 1.0000 | 0.9907 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20.1188 | 0.0500 | 24 | REPORTED | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |
| C1B | 5.0000 | 1500 | 375 | 0.9793 | 0.9907 | 0.9793 | 0.0207 | 0.0495 | 0.0526 | 1.0000 | 8.6935 | NA | <5 | SUPPRESSED_LT_5 | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |
| C1B | 10.0000 | 1500 | 375 | 0.9813 | 0.9907 | 0.9813 | 0.0187 | 0.0495 | 0.0526 | 1.0000 | 10.0000 | 0.0133 | 5 | REPORTED | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |
| C1B | 15.0000 | 1500 | 375 | 0.9820 | 0.9907 | 0.9820 | 0.0180 | 0.0495 | 0.0526 | 1.0000 | 11.2747 | 0.0293 | 11 | REPORTED | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |
| C1B | 20.0000 | 1500 | 375 | 0.9840 | 0.9907 | 0.9840 | 0.0160 | 0.0495 | 0.0526 | 1.0000 | 12.1972 | 0.0480 | 18 | REPORTED | 0.0260 | 3.4722 | 144.0000 | 158.4000 | 224.0000 | 5.1094 |

## 解释边界

`C1C`使用T0–T3 support union，只是offline available-support observability upper bound。它即使达到100% crop containment，也不能消除source-boundary uncertainty，更不能作为T0 causal input。`C1A-tight`通过tight bbox resize获得containment，但其occupancy/blur与lesion geometry耦合，只作size-normalization sensitivity。
