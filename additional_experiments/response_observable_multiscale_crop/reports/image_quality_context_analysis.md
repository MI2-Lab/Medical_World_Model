# 图像质量与上下文分析

## Grid选择的数据依据

| axis | native_spacing_median_mm | native_spacing_q90_mm | native_spacing_q95_mm | detail_target_spacing_mm | context_target_spacing_mm | visit_adaptive_bbox_plus_margin_q95_mm | visit_adaptive_bbox_extent_median_mm | visit_adaptive_bbox_extent_q95_mm | visit_adaptive_bbox_extent_q99_mm | visit_adaptive_bbox_extent_max_mm | visit_adaptive_bbox_plus_margin_q99_mm | visit_adaptive_bbox_plus_margin_max_mm | t0_anchored_all_visit_exact_q95_mm | t0_anchored_all_visit_exact_q99_mm | t0_anchored_all_visit_margin_q95_mm | t0_anchored_all_visit_margin_q99_mm | acquisition_fov_q05_mm | acquisition_fov_median_mm | acquisition_fov_q95_mm | detail_nominal_fov_mm | context_nominal_fov_mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X | 0.6641 | 0.8854 | 0.9375 | 0.9000 | 1.3750 | 106.6736 | 47.0792 | 96.6736 | 122.1412 | 169.5313 | 132.1412 | 179.5313 | 118.6379 | 148.6690 | 128.6379 | 158.6690 | 144.9984 | 170.6667 | 340.0000 | 144.0000 | 176.0000 |
| Y | 0.6641 | 0.8854 | 0.9375 | 0.9000 | 1.5000 | 118.9131 | 57.1621 | 108.9131 | 130.0056 | 177.1875 | 140.0056 | 187.1875 | 138.2035 | 171.5373 | 148.2035 | 181.5373 | 144.9984 | 170.6667 | 340.0000 | 158.4000 | 192.0000 |
| Z | 2.0000 | 2.4000 | 2.5000 | 2.0000 | 3.0000 | 120.0000 | 54.0000 | 110.0000 | 134.0000 | 151.1996 | 144.0000 | 161.1996 | 219.8677 | 267.7500 | 229.8677 | 277.7500 | 158.4000 | 160.0000 | 193.6000 | 224.0000 | 240.0000 |

detail spacing的X/Y=0.9mm接近native P90（0.885/0.885mm），Z=2.0mm等于native median；它比context的1.375/1.5/3.0mm更细。detail nominal FOV的X/Y=144.0/158.4mm，高于P95的T0-anchored四访最坏bbox+5mm margin需求（128.6/148.2mm）；Z=224.0mm位于P95 exact需求219.9mm与P95加margin需求229.9mm之间。这是outcome-free coverage/detail折中，不是已证明最优的model grid；其padding与downsampling必须继续作为PARTIAL blocker审计。

未来production预处理必须仅用training split或外部protocol冻结spacing/FOV，再原样应用于validation/test；本轮全375人统计只用于outcome-free Stage A设计审计。

## Context与padding

`valid_context_to_lesion_volume_ratio`只把acquisition内有效source视为context，不把crop超出acquisition的zero padding算作额外context。

| contract | view | n | minimum_context_margin_median_mm | minimum_context_margin_q05_mm | minimum_context_margin_min_mm | median_context_margin_median_mm | median_context_margin_q05_mm | median_context_margin_min_mm | valid_context_to_lesion_volume_ratio_median | valid_context_to_lesion_volume_ratio_q05 | valid_source_fraction_median | padding_fraction_median | padding_fraction_q95 | padding_fraction_gt_50pct_rate | support_occupancy_median | support_occupancy_q05 | fov_x_mm_median | fov_y_mm_median | fov_z_mm_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | 1500 | -12.8450 | -42.0000 | -70.5469 | 4.5262 | -19.5607 | -43.8999 | 47.5892 | 2.7184 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0202 | 0.0005 | 63.7536 | 63.7536 | 64.0000 |
| C1A | detail | 1500 | 46.3434 | 20.1126 | 5.0000 | 52.2673 | 30.3670 | 5.0000 | 713.5367 | 56.6963 | 0.6519 | 0.3481 | 0.4975 | 0.0460 | 0.0009 | 0.0000 | 144.0000 | 158.4000 | 224.0000 |
| C1A-tight | detail | 1500 | 5.0000 | 5.0000 | 5.0000 | 5.0000 | 5.0000 | 5.0000 | 47.6051 | 8.4241 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0206 | 0.0011 | 57.0792 | 67.1621 | 64.0000 |
| C1B | detail | 1500 | 39.0108 | 8.6935 | -62.4626 | 53.5877 | 29.4624 | 5.0000 | 705.3159 | 56.6963 | 0.6325 | 0.3675 | 0.5447 | 0.1020 | 0.0009 | 0.0000 | 144.0000 | 158.4000 | 224.0000 |
| C1C | detail | 1500 | 40.5344 | 14.1886 | 5.0000 | 53.9238 | 30.8686 | 7.3033 | 714.3318 | 55.0841 | 0.6462 | 0.3538 | 0.5091 | 0.0647 | 0.0009 | 0.0000 | 144.0000 | 158.4000 | 224.0000 |
| C2A | detail | 1500 | 46.3434 | 20.1126 | 5.0000 | 52.2673 | 30.3670 | 5.0000 | 713.5367 | 56.6963 | 0.6519 | 0.3481 | 0.4975 | 0.0460 | 0.0009 | 0.0000 | 144.0000 | 158.4000 | 224.0000 |
| C2A | context | 1500 | 62.5271 | 36.6147 | 5.0000 | 68.6647 | 46.8643 | 10.6875 | 910.0920 | 68.8369 | 0.5149 | 0.4851 | 0.6172 | 0.4267 | 0.0006 | 0.0000 | 176.0000 | 192.0000 | 240.0000 |
| C2B | detail | 1500 | 39.0108 | 8.6935 | -62.4626 | 53.5877 | 29.4624 | 5.0000 | 705.3159 | 56.6963 | 0.6325 | 0.3675 | 0.5447 | 0.1020 | 0.0009 | 0.0000 | 144.0000 | 158.4000 | 224.0000 |
| C2B | context | 1500 | 54.7630 | 23.2748 | -54.4626 | 69.1560 | 44.8789 | 7.1448 | 905.4518 | 69.2291 | 0.5049 | 0.4951 | 0.6354 | 0.4813 | 0.0006 | 0.0000 | 176.0000 | 192.0000 | 240.0000 |

zero padding即使不作为显式mask输入，也会在图像中形成可见边界，并可能成为visit/geometry cue；因此`no direct geometry metadata`不等于没有间接geometry cue。

## Resampling distortion

resize factor定义为`effective output spacing / native spacing`；大于1为downsampling，小于1为upsampling。`resize_anisotropy`是三轴resize factor的max/min，才用于distortion gate；`output_spacing_anisotropy`只描述输出voxel spacing。`C1A-tight`的effective spacing随lesion bbox改变，可能标准化absolute size；fixed-FOV策略仅在明确标记的overflow病例改变scale。

| contract | view | n | expanded_from_nominal_rate | direct_bbox_resize | resize_anisotropy_median | resize_anisotropy_q95 | resize_anisotropy_max | output_spacing_anisotropy_median | output_spacing_anisotropy_q95 | output_spacing_anisotropy_max | extreme_axis_factor_gt2_rate | extreme_axis_factor_lt0_5_rate | support_occupancy_cv | resize_factor_x_min | resize_factor_x_q05 | resize_factor_x_median | resize_factor_x_q95 | resize_factor_x_max | resize_factor_y_min | resize_factor_y_q05 | resize_factor_y_median | resize_factor_y_q95 | resize_factor_y_max | resize_factor_z_min | resize_factor_z_q05 | resize_factor_z_median | resize_factor_z_q95 | resize_factor_z_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | 1500 | 0.0000 | False | 1.0000 | 1.0000 | 1.0000 | 2.9257 | 4.0000 | 5.1195 | 0.0000 | 0.0000 | 2.2032 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C1A | detail | 1500 | 0.0033 | False | 1.4323 | 1.9748 | 3.4722 | 2.2222 | 2.2222 | 2.2222 | 0.0260 | 0.0000 | 2.2015 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.8000 | 0.8000 | 1.0000 | 2.0000 | 2.5000 |
| C1A-tight | detail | 1500 | 0.0000 | True | 1.8311 | 3.1095 | 7.3589 | 1.7320 | 2.5733 | 5.4598 | 0.0000 | 0.8113 | 1.1229 | 0.0875 | 0.2550 | 0.5254 | 0.9942 | 1.4736 | 0.0966 | 0.2700 | 0.5467 | 1.0174 | 1.3182 | 0.0674 | 0.1607 | 0.3594 | 0.8068 | 1.2768 |
| C1B | detail | 1500 | 0.0053 | False | 1.4323 | 1.9748 | 3.4722 | 2.2222 | 2.2222 | 2.2222 | 0.0260 | 0.0000 | 2.1957 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.8000 | 0.8000 | 1.0000 | 2.0000 | 2.5000 |
| C1C | detail | 1500 | 0.0187 | False | 1.4323 | 1.9748 | 3.4722 | 2.2222 | 2.2222 | 2.5026 | 0.0260 | 0.0000 | 2.1678 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.8000 | 0.8000 | 1.0000 | 2.0000 | 2.5000 |
| C2A | detail | 1500 | 0.0033 | False | 1.4323 | 1.9748 | 3.4722 | 2.2222 | 2.2222 | 2.2222 | 0.0260 | 0.0000 | 2.2015 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.8000 | 0.8000 | 1.0000 | 2.0000 | 2.5000 |
| C2A | context | 1500 | 0.0020 | False | 1.5645 | 2.0481 | 3.4091 | 2.1818 | 2.1818 | 2.1818 | 0.9507 | 0.0000 | 2.3338 | 1.0353 | 1.4667 | 2.0705 | 2.4279 | 4.4000 | 1.1294 | 1.6000 | 2.2587 | 2.6486 | 4.8000 | 1.2000 | 1.2000 | 1.5000 | 3.0000 | 3.7501 |
| C2B | detail | 1500 | 0.0053 | False | 1.4323 | 1.9748 | 3.4722 | 2.2222 | 2.2222 | 2.2222 | 0.0260 | 0.0000 | 2.1957 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.6776 | 0.9600 | 1.3552 | 1.5891 | 2.8800 | 0.8000 | 0.8000 | 1.0000 | 2.0000 | 2.5000 |
| C2B | context | 1500 | 0.0027 | False | 1.5645 | 2.0481 | 3.4091 | 2.1818 | 2.1818 | 2.1818 | 0.9507 | 0.0000 | 2.3313 | 1.0353 | 1.4667 | 2.0705 | 2.4279 | 4.4000 | 1.1294 | 1.6000 | 2.2587 | 2.6486 | 4.8000 | 1.2000 | 1.2000 | 1.5000 | 3.0000 | 3.7501 |

## Temporal consistency

| contract | view | n_patients | valid_temporal_geometry_fraction | crop_center_drift_median_mm | crop_center_drift_q95_mm | crop_center_drift_max_mm | lesion_center_drift_median_mm | lesion_center_drift_q95_mm | relative_position_change_median_mm | crop_follow_lesion_ratio_median | fov_max_relative_change_q95 | effective_spacing_max_relative_change_q95 | resize_factor_max_relative_change_median | resize_factor_max_relative_change_q95 | resize_factor_max_relative_change_max | valid_source_fraction_range_median | valid_source_fraction_range_q95 | valid_context_ratio_relative_change_q95 | support_occupancy_relative_change_q95 | temporal_interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | 375 | 0.9787 | 34.7860 | 70.3134 | 124.6788 | 34.0878 | 67.9038 | 24.2999 | 1.0571 | 0.1667 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 239.7751 | 0.9967 | AUDIT_ONLY_OR_LEGACY |
| C1A | detail | 375 | 1.0000 | 31.7451 | 68.9840 | 116.5254 | 34.0878 | 67.9038 | 10.3584 | 0.9672 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.4667 | 0.0644 | 0.1823 | 206.4548 | 0.9967 | RECENTERING_REMOVES_LESION_MOTION |
| C1A-tight | detail | 375 | 1.0000 | 31.7451 | 68.9840 | 116.5254 | 34.0878 | 67.9038 | 10.3584 | 0.9672 | 0.6129 | 0.6129 | 0.2311 | 0.6127 | 0.9589 | 0.0000 | 0.0334 | 121.7634 | 0.9924 | RECENTERING_REMOVES_LESION_MOTION |
| C1B | detail | 375 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 34.0878 | 67.9038 | 34.0878 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.4667 | 0.0885 | 0.2658 | 201.6832 | 0.9967 | FIXED_T0_WINDOW_PRESERVES_HEADER_FRAME_MOTION |
| C1C | detail | 375 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 34.0878 | 67.9038 | 34.0878 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.4667 | 0.0745 | 0.2086 | 206.2049 | 0.9967 | AUDIT_ONLY_OR_LEGACY |
| C2A | detail | 375 | 1.0000 | 31.7451 | 68.9840 | 116.5254 | 34.0878 | 67.9038 | 10.3584 | 0.9672 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.4667 | 0.0644 | 0.1823 | 206.4548 | 0.9967 | RECENTERING_REMOVES_LESION_MOTION |
| C2A | context | 375 | 1.0000 | 31.7451 | 68.9840 | 116.5254 | 34.0878 | 67.9038 | 10.3584 | 0.9672 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.4667 | 0.0523 | 0.1791 | 208.0434 | 0.9967 | RECENTERING_REMOVES_LESION_MOTION |
| C2B | detail | 375 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 34.0878 | 67.9038 | 34.0878 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.4667 | 0.0885 | 0.2658 | 201.6832 | 0.9967 | FIXED_T0_WINDOW_PRESERVES_HEADER_FRAME_MOTION |
| C2B | context | 375 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 34.0878 | 67.9038 | 34.0878 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.4667 | 0.0653 | 0.2106 | 197.3197 | 0.9967 | FIXED_T0_WINDOW_PRESERVES_HEADER_FRAME_MOTION |

Visit-adaptive方案使crop center跟随每访support bbox center，因而删除bbox平移；表中的voxel-centroid relative change在共同world frame计算，仍保留centroid相对bbox center的变化。T0-anchored方案的window center/FOV在四访保持不变，但尚未完成image-only rigid registration，仍混有patient repositioning。

## Morphology readiness

| contract | view | n | surface_available_fraction | surface_retention_median | surface_retention_q05 | surface_retention_minimum | bbox_containment_rate | source_boundary_censored_rate | fully_observable_surface_rate | any_cut_component_rate | any_missed_component_rate | cut_components_total | missed_components_total | component_count_median | proxy_semantics |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | legacy | 1500 | 0.9920 | 0.8540 | 0.1847 | 0.0011 | 0.2047 | 0.0093 | 0.2047 | 0.7327 | 0.7560 | 7955 | 59611 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C1A | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 0.0000 | 0.0000 | 0 | 0 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C1A-tight | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 0.0000 | 0.0000 | 0 | 0 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C1B | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 0.1677 | 0.9793 | 0.0093 | 0.9707 | 0.0180 | 0.0193 | 130 | 609 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C1C | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 0.0000 | 0.0000 | 0 | 0 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C2A | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 0.0000 | 0.0000 | 0 | 0 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C2A | context | 1500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0093 | 0.9907 | 0.0000 | 0.0000 | 0 | 0 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C2B | detail | 1500 | 1.0000 | 1.0000 | 1.0000 | 0.1677 | 0.9793 | 0.0093 | 0.9707 | 0.0180 | 0.0193 | 130 | 609 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |
| C2B | context | 1500 | 1.0000 | 1.0000 | 1.0000 | 0.2295 | 0.9907 | 0.0093 | 0.9820 | 0.0087 | 0.0093 | 76 | 268 | 48.0000 | FTV_INCLUSION_SUPPORT_NOT_DENSE_TUMOR_SEGMENTATION |

surface/component统计针对高度碎片化的FTV inclusion proxy，只回答“support surface是否被crop切断”，不把proxy surface解释为真实tumor boundary irregularity或正式sphericity target。

## 真实图像与normalization sensitivity

| preview_scope | contract | view | n_images | n_cases | n_visits | source_image | image_mode | sample_mode | image_interpolation | mask_usage | normalization_sensitivity | valid_source_fraction_mean | valid_source_fraction_min | finite_fraction_within_source_mean | finite_fraction_within_source_min | robust_dynamic_range_mean | robust_dynamic_range_min | nonconstant_fraction | mask_center_plane_intersection_fraction | legacy_norm_median_mean | legacy_norm_scale_mean | legacy_norm_source_mean_mean | legacy_norm_source_std_mean | legacy_norm_padding_value_mean | legacy_norm_saturation_fraction_mean | contains_patient_identifiers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| selected_T0_strata_plus_middle_stratum_T0_T3 | C0 | legacy | 8 | 5 | 4 | raw_DCE_NIfTI | first_post_minus_pre | physical_center_plane | scipy_order1_single_pass | nearest_neighbor_audit_overlay_only_not_image_channel | 2D_ENHANCEMENT_PLANE_LEGACY_P01_P99_MEDIAN_IQR_WITH_ZERO_PADDING | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 474.9781 | 78.5000 | 1.0000 | 1.0000 | 58.9688 | 72.5537 | 0.2689 | 1.3370 | NA | 0.0242 | False |
| selected_T0_strata_plus_middle_stratum_T0_T3 | C1B | detail | 8 | 5 | 4 | raw_DCE_NIfTI | first_post_minus_pre | physical_center_plane | scipy_order1_single_pass | nearest_neighbor_audit_overlay_only_not_image_channel | 2D_ENHANCEMENT_PLANE_LEGACY_P01_P99_MEDIAN_IQR_WITH_ZERO_PADDING | 0.9432 | 0.7775 | 1.0000 | 1.0000 | 471.6896 | 143.4154 | 1.0000 | 0.8750 | 8.5333 | 37.7446 | 0.6615 | 1.6685 | -0.1571 | 0.0583 | False |
| selected_T0_strata_plus_middle_stratum_T0_T3 | C2B | context | 8 | 5 | 4 | raw_DCE_NIfTI | first_post_minus_pre | physical_center_plane | scipy_order1_single_pass | nearest_neighbor_audit_overlay_only_not_image_channel | 2D_ENHANCEMENT_PLANE_LEGACY_P01_P99_MEDIAN_IQR_WITH_ZERO_PADDING | 0.8551 | 0.6094 | 1.0000 | 1.0000 | 488.8783 | 144.0363 | 1.0000 | 0.8750 | 1.9455 | 26.8517 | 0.9360 | 1.9105 | -0.0775 | 0.0953 | False |
| selected_T0_strata_plus_middle_stratum_T0_T3 | C2B | detail | 5 | 5 | 1 | raw_DCE_NIfTI | first_post_minus_pre | physical_center_plane | scipy_order1_single_pass | nearest_neighbor_audit_overlay_only_not_image_channel | 2D_ENHANCEMENT_PLANE_LEGACY_P01_P99_MEDIAN_IQR_WITH_ZERO_PADDING | 0.9091 | 0.7775 | 1.0000 | 1.0000 | 653.1588 | 177.2368 | 1.0000 | 1.0000 | 11.0003 | 51.9279 | 0.7706 | 1.7280 | -0.1571 | 0.0651 | False |

该表覆盖5个去标识case的真实raw-DCE二维物理中心平面，并对first-post-minus-pre执行legacy P01/P99 clipping + median/IQR normalization（zero padding计入统计）。它是代表性sensitivity，不是完整3-D DCE7验收；variable raw phase到production DCE7的phase selection、统一anatomical orientation、全部7通道3-D单次重采样、anti-alias、归一化与cache round-trip仍未实现，继续作为model-ready blocker。

## Tensor footprint与实现边界

| contract | views | visits | dce_channels | spatial_voxels_per_visit | float32_values_per_patient | float32_megabytes_per_patient | relative_to_legacy | excludes_activations_optimizer_and_padding_mask |
|---|---|---|---|---|---|---|---|---|
| C0 | legacy | 4 | 7 | 294912 | 8257536 | 33.0301 | 1.0000 | True |
| C1B | detail | 4 | 7 | 3153920 | 88309760 | 353.2390 | 10.6944 | True |
| C2B | detail+context | 4 | 7 | 4464640 | 125009920 | 500.0397 | 15.1389 | True |

本轮已实现source-domain physical geometry、C0/C1/C2 window、完整cohort audit与真实2-D preview；尚未生成可训练的3-D DCE7 volume/cache。表中仅是float32 input本体，未计activation、gradient或optimizer，因此不能把geometry contract误称为已验证的same-encoder训练可行性。

## 强度归一化与合规边界

legacy pipeline在crop后做每通道P01/P99 clipping与median/IQR normalization。改变FOV/context会改变这些统计，因此model-ready builder必须冻结“先physical resample/crop，再按view独立沿用legacy normalization”的顺序，并在matched control中保持一致。preview不参与gate或candidate选择。

三张代表图不含ID、路径或敏感PNG metadata，并已对本轮12张公开图做人工视觉复核。自动验证器不对任意未来PNG执行像素OCR，因此这仍只是本轮技术性去标识与人工图审结论。若把derived MRI图发布到仓库外，仍须按I-SPY2数据使用协议单独确认再分发权限。
