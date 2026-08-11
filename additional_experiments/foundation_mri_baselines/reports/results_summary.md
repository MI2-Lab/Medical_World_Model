# Foundation MRI 结果汇总（描述性）

正式 foundation 候选为：dino_vitb16_imagenet1k、medicalnet_resnet50_3dseg8。本页逐一保留全部预注册候选，没有按 test 指标筛选“最佳模型”。

推断边界：descriptive paired outer-fold OOF patient bootstrap; not confirmatory and not used for model or checkpoint selection。配对区间采用同一患者非参数 bootstrap，固定 seed=2026、5000 次、percentile 95% CI。所有差值均为“候选减参照”；AUROC/AUPRC 越高越好，Brier 越低越好。区间仅用于描述不确定性，不作确认性显著性检验。

## pCR pooled OOF（完整 808 人）

| 模型 | 空间 | 时点 | n | AUROC | AUPRC | Brier | 校准斜率 | ECE10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAP0_mri_clinical | GLOBAL | T0 | 808 | 0.720 | 0.566 | 0.218 | 1.236 | 0.151 |
| GAP0_mri_clinical | GLOBAL | T0-T1 | 808 | 0.715 | 0.561 | 0.218 | 0.975 | 0.147 |
| GAP0_mri_clinical | GLOBAL | T0-T2 | 808 | 0.696 | 0.541 | 0.223 | 0.664 | 0.145 |
| GAP0_mri_only | GLOBAL | T0 | 808 | 0.495 | 0.333 | 0.259 | -0.109 | 0.168 |
| GAP0_mri_only | GLOBAL | T0-T1 | 808 | 0.514 | 0.352 | 0.266 | 2.272 | 0.179 |
| GAP0_mri_only | GLOBAL | T0-T2 | 808 | 0.552 | 0.394 | 0.274 | 2.924 | 0.195 |
| LOCAL0_mri_clinical | LOCAL | T0 | 808 | 0.688 | 0.529 | 0.225 | 0.803 | 0.143 |
| LOCAL0_mri_clinical | LOCAL | T0-T1 | 808 | 0.684 | 0.522 | 0.228 | 0.729 | 0.150 |
| LOCAL0_mri_clinical | LOCAL | T0-T2 | 808 | 0.675 | 0.508 | 0.233 | 0.410 | 0.154 |
| LOCAL0_mri_only | LOCAL | T0 | 808 | 0.510 | 0.358 | 0.289 | 0.041 | 0.213 |
| LOCAL0_mri_only | LOCAL | T0-T1 | 808 | 0.560 | 0.396 | 0.265 | 0.136 | 0.181 |
| LOCAL0_mri_only | LOCAL | T0-T2 | 808 | 0.547 | 0.372 | 0.272 | 0.094 | 0.186 |
| clinical_only | NONE | T0 | 808 | 0.709 | 0.558 | 0.224 | 1.071 | 0.148 |
| clinical_only | NONE | T0-T1 | 808 | 0.709 | 0.558 | 0.224 | 1.071 | 0.148 |
| clinical_only | NONE | T0-T2 | 808 | 0.709 | 0.558 | 0.224 | 1.071 | 0.148 |
| dino_vitb16_imagenet1k_mri_clinical | GLOBAL | T0 | 808 | 0.703 | 0.529 | 0.228 | 1.440 | 0.160 |
| dino_vitb16_imagenet1k_mri_clinical | GLOBAL | T0-T1 | 808 | 0.711 | 0.544 | 0.228 | 2.197 | 0.160 |
| dino_vitb16_imagenet1k_mri_clinical | GLOBAL | T0-T2 | 808 | 0.707 | 0.542 | 0.229 | 2.157 | 0.160 |
| dino_vitb16_imagenet1k_mri_clinical | LOCAL | T0 | 808 | 0.669 | 0.498 | 0.237 | 0.414 | 0.164 |
| dino_vitb16_imagenet1k_mri_clinical | LOCAL | T0-T1 | 808 | 0.680 | 0.488 | 0.232 | 0.702 | 0.148 |
| dino_vitb16_imagenet1k_mri_clinical | LOCAL | T0-T2 | 808 | 0.672 | 0.498 | 0.231 | 0.642 | 0.150 |
| dino_vitb16_imagenet1k_mri_only | GLOBAL | T0 | 808 | 0.536 | 0.380 | 0.280 | 2.544 | 0.206 |
| dino_vitb16_imagenet1k_mri_only | GLOBAL | T0-T1 | 808 | 0.550 | 0.390 | 0.288 | 2.215 | 0.213 |
| dino_vitb16_imagenet1k_mri_only | GLOBAL | T0-T2 | 808 | 0.539 | 0.397 | 0.309 | 1.000 | 0.265 |
| dino_vitb16_imagenet1k_mri_only | LOCAL | T0 | 808 | 0.528 | 0.373 | 0.304 | 1.457 | 0.237 |
| dino_vitb16_imagenet1k_mri_only | LOCAL | T0-T1 | 808 | 0.557 | 0.387 | 0.264 | 0.172 | 0.176 |
| dino_vitb16_imagenet1k_mri_only | LOCAL | T0-T2 | 808 | 0.578 | 0.418 | 0.263 | 0.262 | 0.189 |
| medicalnet_resnet50_3dseg8_mri_clinical | GLOBAL | T0 | 808 | 0.663 | 0.482 | 0.248 | 1.000 | 0.184 |
| medicalnet_resnet50_3dseg8_mri_clinical | GLOBAL | T0-T1 | 808 | 0.706 | 0.519 | 0.229 | 2.126 | 0.160 |
| medicalnet_resnet50_3dseg8_mri_clinical | GLOBAL | T0-T2 | 808 | 0.674 | 0.488 | 0.233 | 0.340 | 0.154 |
| medicalnet_resnet50_3dseg8_mri_clinical | LOCAL | T0 | 808 | 0.703 | 0.509 | 0.231 | 0.588 | 0.165 |
| medicalnet_resnet50_3dseg8_mri_clinical | LOCAL | T0-T1 | 808 | 0.710 | 0.524 | 0.229 | 2.028 | 0.160 |
| medicalnet_resnet50_3dseg8_mri_clinical | LOCAL | T0-T2 | 808 | 0.703 | 0.518 | 0.230 | 1.913 | 0.160 |
| medicalnet_resnet50_3dseg8_mri_only | GLOBAL | T0 | 808 | 0.539 | 0.378 | 0.302 | 1.000 | 0.239 |
| medicalnet_resnet50_3dseg8_mri_only | GLOBAL | T0-T1 | 808 | 0.537 | 0.376 | 0.287 | 1.887 | 0.213 |
| medicalnet_resnet50_3dseg8_mri_only | GLOBAL | T0-T2 | 808 | 0.521 | 0.368 | 0.315 | 1.000 | 0.254 |
| medicalnet_resnet50_3dseg8_mri_only | LOCAL | T0 | 808 | 0.527 | 0.366 | 0.292 | 1.000 | 0.217 |
| medicalnet_resnet50_3dseg8_mri_only | LOCAL | T0-T1 | 808 | 0.508 | 0.348 | 0.295 | 1.000 | 0.237 |
| medicalnet_resnet50_3dseg8_mri_only | LOCAL | T0-T2 | 808 | 0.527 | 0.354 | 0.288 | 1.000 | 0.216 |

## 预注册配对比较（完整 808 人与 complete-case 375 人）

| 比较族 | 人群 | 时点 | 参照 | 候选 | n | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔBrier [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beyond_ftv | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@LOCAL | 375 | -0.204 [-0.281, -0.127] | -0.227 [-0.326, -0.132] | 0.086 [0.051, 0.122] |
| beyond_ftv | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | 375 | -0.260 [-0.338, -0.183] | -0.219 [-0.321, -0.129] | 0.159 [0.116, 0.203] |
| beyond_ftv | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | 375 | -0.173 [-0.246, -0.103] | -0.217 [-0.305, -0.132] | 0.088 [0.054, 0.123] |
| beyond_ftv | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@GLOBAL | 375 | -0.092 [-0.155, -0.029] | -0.088 [-0.176, -0.002] | 0.034 [0.008, 0.061] |
| beyond_ftv | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@GLOBAL | 375 | -0.058 [-0.120, 0.007] | -0.072 [-0.169, 0.019] | 0.010 [-0.010, 0.030] |
| beyond_ftv | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@LOCAL | 375 | -0.032 [-0.091, 0.029] | -0.060 [-0.143, 0.028] | 0.011 [-0.006, 0.028] |
| beyond_ftv | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@LOCAL | 375 | -0.145 [-0.222, -0.066] | -0.184 [-0.280, -0.084] | 0.047 [0.020, 0.075] |
| beyond_ftv | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | 375 | -0.103 [-0.171, -0.034] | -0.153 [-0.240, -0.066] | 0.041 [0.014, 0.070] |
| beyond_ftv | radiomics_complete_case_375 | T0 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | 375 | -0.074 [-0.141, -0.004] | -0.097 [-0.187, 0.004] | 0.027 [0.003, 0.051] |
| beyond_ftv | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | dino_vitb16_imagenet1k_mri_clinical_ftv@LOCAL | 375 | -0.150 [-0.225, -0.076] | -0.154 [-0.255, -0.054] | 0.054 [0.023, 0.084] |
| beyond_ftv | radiomics_complete_case_375 | T0-T2 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@GLOBAL | 375 | -0.201 [-0.273, -0.129] | -0.169 [-0.272, -0.071] | 0.057 [0.029, 0.085] |
| beyond_ftv | radiomics_complete_case_375 | T0-T1 | clinical_ftv@TABULAR | medicalnet_resnet50_3dseg8_mri_clinical_ftv@GLOBAL | 375 | -0.171 [-0.247, -0.093] | -0.175 [-0.273, -0.074] | 0.044 [0.019, 0.070] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.067 [-0.134, 0.002] | -0.088 [-0.176, -0.001] | 0.018 [-0.009, 0.046] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.175 [-0.254, -0.097] | -0.120 [-0.214, -0.039] | 0.120 [0.080, 0.161] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.082 [-0.153, -0.010] | -0.085 [-0.179, 0.010] | 0.040 [0.010, 0.071] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.144 [-0.218, -0.073] | -0.158 [-0.241, -0.080] | 0.070 [0.038, 0.102] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.037 [-0.073, -0.003] | -0.059 [-0.115, -0.005] | 0.007 [-0.003, 0.018] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.066 [-0.142, 0.011] | -0.061 [-0.154, 0.027] | 0.011 [-0.014, 0.037] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | GAP0_mri_clinical_paired@GLOBAL | 375 | -0.038 [-0.092, 0.014] | -0.021 [-0.088, 0.045] | 0.004 [-0.016, 0.025] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.084 [-0.151, -0.014] | -0.098 [-0.188, -0.002] | 0.023 [-0.001, 0.048] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.068 [-0.130, -0.003] | -0.077 [-0.170, 0.013] | 0.007 [-0.013, 0.028] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.002 [-0.031, 0.026] | -0.016 [-0.064, 0.032] | 0.005 [-0.002, 0.012] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.001 [-0.026, 0.027] | -0.034 [-0.080, 0.019] | 0.005 [-0.001, 0.012] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | GAP0_mri_clinical_paired@GLOBAL | 375 | -0.046 [-0.110, 0.015] | -0.028 [-0.108, 0.047] | 0.011 [-0.014, 0.037] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | 0.002 [-0.026, 0.029] | -0.013 [-0.059, 0.033] | 0.005 [-0.002, 0.011] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.102 [-0.165, -0.039] | -0.090 [-0.178, -0.005] | 0.032 [0.006, 0.059] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.041 [-0.075, -0.009] | -0.060 [-0.113, -0.009] | 0.013 [0.002, 0.025] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | LOCAL0_mri_clinical@LOCAL | 808 | -0.025 [-0.055, 0.005] | -0.035 [-0.081, 0.013] | 0.005 [-0.004, 0.013] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.029 [-0.063, 0.004] | -0.070 [-0.124, -0.019] | 0.008 [-0.002, 0.018] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.140 [-0.216, -0.063] | -0.112 [-0.205, -0.016] | 0.022 [-0.001, 0.046] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | -0.006 [-0.036, 0.024] | -0.049 [-0.094, -0.001] | 0.008 [0.000, 0.015] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.003 [-0.028, 0.023] | -0.039 [-0.083, 0.009] | 0.006 [-0.001, 0.012] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.042 [-0.100, 0.018] | -0.061 [-0.142, 0.024] | 0.008 [-0.010, 0.026] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.046 [-0.085, -0.010] | -0.076 [-0.132, -0.016] | 0.024 [0.010, 0.040] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.019 [-0.058, 0.021] | -0.022 [-0.075, 0.031] | -0.000 [-0.011, 0.010] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | GAP0_mri_clinical@GLOBAL | 808 | 0.011 [-0.013, 0.036] | 0.008 [-0.041, 0.057] | -0.006 [-0.013, 0.002] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.144 [-0.223, -0.065] | -0.147 [-0.238, -0.056] | 0.035 [0.011, 0.060] |
| clinical_gain | radiomics_complete_case_375 | T0-T1 | clinical_only_paired@NONE | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.050 [-0.111, 0.009] | -0.042 [-0.127, 0.037] | 0.010 [-0.011, 0.031] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.006 [-0.033, 0.020] | -0.028 [-0.072, 0.015] | 0.004 [-0.003, 0.010] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.109 [-0.185, -0.034] | -0.067 [-0.163, 0.032] | 0.012 [-0.013, 0.037] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | LOCAL0_mri_clinical@LOCAL | 808 | -0.034 [-0.069, -0.001] | -0.050 [-0.102, 0.002] | 0.010 [-0.003, 0.022] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | -0.006 [-0.033, 0.021] | -0.039 [-0.087, 0.012] | 0.006 [-0.001, 0.013] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.035 [-0.071, -0.001] | -0.069 [-0.126, -0.008] | 0.009 [-0.002, 0.021] |
| clinical_gain | radiomics_complete_case_375 | T0 | clinical_only_paired@NONE | GAP0_mri_clinical_paired@GLOBAL | 375 | -0.034 [-0.080, 0.013] | -0.021 [-0.083, 0.036] | -0.006 [-0.022, 0.011] |
| clinical_gain | full_808 | T0-T1 | clinical_only@NONE | GAP0_mri_clinical@GLOBAL | 808 | 0.006 [-0.017, 0.029] | 0.003 [-0.042, 0.049] | -0.006 [-0.014, 0.001] |
| clinical_gain | radiomics_complete_case_375 | T0-T2 | clinical_only_paired@NONE | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.138 [-0.212, -0.060] | -0.146 [-0.237, -0.055] | 0.051 [0.020, 0.081] |
| clinical_gain | full_808 | T0-T2 | clinical_only@NONE | GAP0_mri_clinical@GLOBAL | 808 | -0.013 [-0.042, 0.018] | -0.016 [-0.061, 0.035] | -0.001 [-0.011, 0.009] |
| clinical_gain | full_808 | T0 | clinical_only@NONE | LOCAL0_mri_clinical@LOCAL | 808 | -0.021 [-0.050, 0.008] | -0.029 [-0.081, 0.024] | 0.002 [-0.007, 0.011] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | GAP0_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.129 [-0.213, -0.048] | -0.092 [-0.187, -0.004] | 0.109 [0.064, 0.156] |
| foundation_vs_current_cnn | full_808 | T0-T1 | GAP0_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | 808 | 0.023 [-0.030, 0.077] | 0.023 [-0.025, 0.074] | 0.020 [0.001, 0.040] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | GAP0_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | 375 | 0.044 [-0.040, 0.124] | 0.012 [-0.048, 0.068] | -0.007 [-0.045, 0.031] |
| foundation_vs_current_cnn | full_808 | T0 | GAP0_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.057 [-0.095, -0.020] | -0.084 [-0.139, -0.029] | 0.030 [0.016, 0.045] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | GAP0_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.106 [-0.186, -0.029] | -0.137 [-0.216, -0.058] | 0.066 [0.031, 0.102] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_only_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.000 [-0.085, 0.086] | 0.007 [-0.060, 0.085] | -0.039 [-0.086, 0.006] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | GAP0_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | 375 | 0.094 [0.001, 0.186] | 0.074 [-0.005, 0.157] | -0.043 [-0.080, -0.005] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | GAP0_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | 375 | -0.001 [-0.088, 0.086] | 0.057 [-0.018, 0.126] | 0.052 [0.006, 0.099] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_clinical_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.055 [-0.141, 0.034] | -0.061 [-0.149, 0.032] | 0.011 [-0.032, 0.052] |
| foundation_vs_current_cnn | full_808 | T0-T2 | GAP0_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.022 [-0.057, 0.011] | -0.053 [-0.110, 0.002] | 0.010 [-0.001, 0.023] |
| foundation_vs_current_cnn | full_808 | T0 | LOCAL0_mri_only@LOCAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | 0.017 [-0.037, 0.069] | 0.008 [-0.040, 0.054] | 0.003 [-0.018, 0.025] |
| foundation_vs_current_cnn | full_808 | T0-T2 | LOCAL0_mri_only@LOCAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.020 [-0.075, 0.035] | -0.018 [-0.066, 0.027] | 0.016 [-0.006, 0.039] |
| foundation_vs_current_cnn | full_808 | T0 | GAP0_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | 808 | 0.044 [-0.014, 0.102] | 0.045 [-0.000, 0.095] | 0.042 [0.022, 0.063] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | GAP0_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | 375 | 0.000 [-0.091, 0.089] | 0.031 [-0.039, 0.107] | -0.039 [-0.076, -0.002] |
| foundation_vs_current_cnn | full_808 | T0 | LOCAL0_mri_only@LOCAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.018 [-0.038, 0.071] | 0.015 [-0.037, 0.067] | 0.015 [-0.009, 0.040] |
| foundation_vs_current_cnn | full_808 | T0 | GAP0_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.017 [-0.046, 0.012] | -0.037 [-0.091, 0.018] | 0.009 [0.002, 0.017] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | GAP0_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | 375 | -0.068 [-0.133, -0.001] | -0.069 [-0.150, 0.011] | 0.038 [0.011, 0.065] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_clinical_paired@LOCAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.017 [-0.088, 0.056] | -0.045 [-0.129, 0.041] | 0.008 [-0.022, 0.039] |
| foundation_vs_current_cnn | full_808 | T0 | LOCAL0_mri_clinical@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.015 [-0.017, 0.048] | -0.020 [-0.072, 0.037] | 0.006 [-0.004, 0.015] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | GAP0_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | 375 | 0.032 [-0.050, 0.109] | 0.040 [-0.019, 0.103] | 0.026 [0.001, 0.051] |
| foundation_vs_current_cnn | full_808 | T0-T1 | GAP0_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | -0.004 [-0.028, 0.020] | -0.017 [-0.060, 0.027] | 0.011 [0.003, 0.018] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | LOCAL0_mri_only_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.004 [-0.087, 0.082] | 0.011 [-0.054, 0.081] | -0.014 [-0.044, 0.016] |
| foundation_vs_current_cnn | full_808 | T0-T1 | GAP0_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | 808 | -0.008 [-0.032, 0.014] | -0.042 [-0.085, 0.001] | 0.012 [0.004, 0.019] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | LOCAL0_mri_only_paired@LOCAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.038 [-0.051, 0.124] | 0.027 [-0.047, 0.099] | -0.024 [-0.056, 0.008] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_clinical_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.094 [-0.174, -0.011] | -0.105 [-0.193, -0.011] | 0.026 [-0.004, 0.055] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | GAP0_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | 375 | 0.033 [-0.048, 0.112] | 0.036 [-0.021, 0.091] | 0.012 [-0.013, 0.037] |
| foundation_vs_current_cnn | full_808 | T0 | LOCAL0_mri_clinical@LOCAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.019 [-0.053, 0.012] | -0.031 [-0.083, 0.021] | 0.011 [0.000, 0.024] |
| foundation_vs_current_cnn | full_808 | T0-T2 | LOCAL0_mri_clinical@LOCAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.002 [-0.040, 0.035] | -0.009 [-0.063, 0.046] | -0.002 [-0.016, 0.012] |
| foundation_vs_current_cnn | full_808 | T0-T2 | LOCAL0_mri_clinical@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.029 [-0.005, 0.064] | 0.011 [-0.041, 0.069] | -0.004 [-0.017, 0.009] |
| foundation_vs_current_cnn | full_808 | T0-T1 | LOCAL0_mri_only@LOCAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | -0.003 [-0.058, 0.049] | -0.009 [-0.063, 0.045] | -0.001 [-0.017, 0.016] |
| foundation_vs_current_cnn | full_808 | T0-T1 | LOCAL0_mri_clinical@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.027 [-0.002, 0.056] | 0.002 [-0.044, 0.052] | 0.001 [-0.008, 0.010] |
| foundation_vs_current_cnn | full_808 | T0-T2 | LOCAL0_mri_only@LOCAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.031 [-0.023, 0.082] | 0.046 [-0.006, 0.098] | -0.009 [-0.026, 0.009] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_only_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | 0.013 [-0.071, 0.099] | 0.003 [-0.066, 0.077] | 0.016 [-0.009, 0.043] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_clinical_paired@LOCAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.016 [-0.065, 0.097] | 0.023 [-0.074, 0.128] | -0.029 [-0.065, 0.007] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | GAP0_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.063 [-0.144, 0.017] | -0.039 [-0.136, 0.067] | 0.001 [-0.031, 0.034] |
| foundation_vs_current_cnn | full_808 | T0-T2 | GAP0_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | 808 | 0.011 [-0.020, 0.041] | 0.000 [-0.050, 0.050] | 0.006 [-0.004, 0.016] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | LOCAL0_mri_only_paired@LOCAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.069 [-0.020, 0.156] | 0.042 [-0.038, 0.127] | 0.012 [-0.019, 0.044] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T2 | LOCAL0_mri_only_paired@LOCAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.035 [-0.050, 0.120] | 0.053 [-0.025, 0.132] | -0.068 [-0.112, -0.025] |
| foundation_vs_current_cnn | full_808 | T0-T1 | LOCAL0_mri_clinical@LOCAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.004 [-0.036, 0.027] | -0.035 [-0.083, 0.010] | 0.003 [-0.007, 0.014] |
| foundation_vs_current_cnn | full_808 | T0-T2 | GAP0_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | 808 | -0.031 [-0.087, 0.023] | -0.026 [-0.080, 0.028] | 0.041 [0.016, 0.067] |
| foundation_vs_current_cnn | full_808 | T0 | GAP0_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@GLOBAL | 808 | 0.041 [-0.014, 0.097] | 0.047 [0.001, 0.092] | 0.021 [0.005, 0.038] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | LOCAL0_mri_clinical_paired@LOCAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | -0.065 [-0.125, -0.003] | -0.075 [-0.157, 0.012] | 0.024 [0.003, 0.046] |
| foundation_vs_current_cnn | full_808 | T0-T2 | GAP0_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@GLOBAL | 808 | -0.013 [-0.067, 0.042] | 0.003 [-0.055, 0.060] | 0.035 [0.010, 0.060] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0-T1 | GAP0_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.102 [-0.182, -0.019] | -0.092 [-0.185, 0.009] | 0.018 [-0.011, 0.047] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | GAP0_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | 375 | -0.034 [-0.096, 0.030] | -0.056 [-0.143, 0.027] | 0.013 [-0.009, 0.035] |
| foundation_vs_current_cnn | full_808 | T0-T1 | GAP0_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@GLOBAL | 808 | 0.037 [-0.020, 0.094] | 0.038 [-0.014, 0.092] | 0.022 [0.001, 0.044] |
| foundation_vs_current_cnn | radiomics_complete_case_375 | T0 | LOCAL0_mri_clinical_paired@LOCAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.023 [-0.071, 0.027] | -0.039 [-0.113, 0.044] | 0.008 [-0.006, 0.023] |
| foundation_vs_current_cnn | full_808 | T0-T1 | LOCAL0_mri_only@LOCAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.052 [-0.106, 0.001] | -0.047 [-0.093, 0.001] | 0.031 [0.010, 0.052] |
| local_vs_global | full_808 | T0 | GAP0_mri_only@GLOBAL | LOCAL0_mri_only@LOCAL | 808 | 0.015 [-0.040, 0.070] | 0.025 [-0.017, 0.070] | 0.030 [0.013, 0.046] |
| local_vs_global | full_808 | T0-T2 | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | 0.006 [-0.046, 0.058] | -0.015 [-0.060, 0.029] | -0.027 [-0.052, -0.003] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | 0.033 [-0.043, 0.110] | 0.001 [-0.066, 0.068] | 0.030 [-0.003, 0.063] |
| local_vs_global | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.007 [-0.045, 0.060] | -0.004 [-0.054, 0.046] | -0.025 [-0.046, -0.004] |
| local_vs_global | radiomics_complete_case_375 | T0 | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | 0.026 [-0.026, 0.079] | 0.016 [-0.052, 0.095] | 0.000 [-0.017, 0.017] |
| local_vs_global | radiomics_complete_case_375 | T0 | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.016 [-0.086, 0.058] | -0.006 [-0.067, 0.060] | -0.006 [-0.031, 0.017] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | medicalnet_resnet50_3dseg8_mri_only_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_only_paired@LOCAL | 375 | -0.039 [-0.113, 0.037] | -0.040 [-0.109, 0.027] | 0.009 [-0.019, 0.035] |
| local_vs_global | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.018 [-0.047, 0.085] | -0.008 [-0.084, 0.076] | -0.009 [-0.039, 0.021] |
| local_vs_global | full_808 | T0 | GAP0_mri_clinical@GLOBAL | LOCAL0_mri_clinical@LOCAL | 808 | -0.032 [-0.060, -0.005] | -0.037 [-0.083, 0.009] | 0.007 [-0.001, 0.015] |
| local_vs_global | radiomics_complete_case_375 | T0 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.024 [-0.052, 0.104] | 0.015 [-0.053, 0.085] | -0.002 [-0.031, 0.026] |
| local_vs_global | full_808 | T0-T2 | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.029 [0.000, 0.058] | 0.030 [-0.018, 0.078] | -0.004 [-0.013, 0.006] |
| local_vs_global | full_808 | T0 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | -0.008 [-0.062, 0.044] | -0.007 [-0.058, 0.047] | 0.024 [0.000, 0.048] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | GAP0_mri_clinical_paired@GLOBAL | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.036 [-0.108, 0.038] | -0.057 [-0.139, 0.026] | 0.030 [-0.004, 0.063] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | GAP0_mri_only_paired@GLOBAL | LOCAL0_mri_only_paired@LOCAL | 375 | 0.033 [-0.053, 0.119] | 0.025 [-0.035, 0.084] | 0.030 [-0.014, 0.075] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.070 [-0.018, 0.160] | 0.021 [-0.067, 0.114] | -0.091 [-0.135, -0.047] |
| local_vs_global | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.035 [-0.068, -0.003] | -0.043 [-0.095, 0.007] | 0.002 [-0.007, 0.012] |
| local_vs_global | radiomics_complete_case_375 | T0 | GAP0_mri_only_paired@GLOBAL | LOCAL0_mri_only_paired@LOCAL | 375 | 0.020 [-0.063, 0.103] | 0.023 [-0.033, 0.087] | 0.034 [0.007, 0.062] |
| local_vs_global | full_808 | T0 | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.012 [-0.063, 0.040] | -0.012 [-0.061, 0.036] | -0.010 [-0.033, 0.014] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_only_paired@GLOBAL | dino_vitb16_imagenet1k_mri_only_paired@LOCAL | 375 | 0.067 [-0.017, 0.149] | 0.061 [-0.004, 0.135] | -0.031 [-0.069, 0.008] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.028 [-0.099, 0.042] | -0.079 [-0.164, 0.003] | 0.039 [0.007, 0.071] |
| local_vs_global | full_808 | T0-T2 | GAP0_mri_clinical@GLOBAL | LOCAL0_mri_clinical@LOCAL | 808 | -0.022 [-0.062, 0.017] | -0.034 [-0.091, 0.018] | 0.011 [-0.004, 0.025] |
| local_vs_global | full_808 | T0-T1 | medicalnet_resnet50_3dseg8_mri_only@GLOBAL | medicalnet_resnet50_3dseg8_mri_only@LOCAL | 808 | -0.029 [-0.081, 0.022] | -0.027 [-0.076, 0.018] | 0.009 [-0.011, 0.029] |
| local_vs_global | full_808 | T0 | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.040 [0.005, 0.079] | 0.027 [-0.027, 0.082] | -0.017 [-0.032, -0.002] |
| local_vs_global | full_808 | T0-T2 | GAP0_mri_only@GLOBAL | LOCAL0_mri_only@LOCAL | 808 | -0.005 [-0.062, 0.051] | -0.023 [-0.074, 0.031] | -0.002 [-0.023, 0.019] |
| local_vs_global | radiomics_complete_case_375 | T0 | GAP0_mri_clinical_paired@GLOBAL | LOCAL0_mri_clinical_paired@LOCAL | 375 | 0.015 [-0.033, 0.066] | -0.001 [-0.059, 0.061] | 0.005 [-0.010, 0.020] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.077 [0.001, 0.154] | 0.070 [-0.005, 0.145] | -0.052 [-0.090, -0.014] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | GAP0_mri_clinical_paired@GLOBAL | LOCAL0_mri_clinical_paired@LOCAL | 375 | -0.012 [-0.080, 0.059] | -0.021 [-0.109, 0.062] | 0.006 [-0.020, 0.031] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | medicalnet_resnet50_3dseg8_mri_clinical_paired@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical_paired@LOCAL | 375 | -0.004 [-0.087, 0.081] | -0.035 [-0.122, 0.050] | 0.013 [-0.015, 0.042] |
| local_vs_global | full_808 | T0-T1 | GAP0_mri_clinical@GLOBAL | LOCAL0_mri_clinical@LOCAL | 808 | -0.031 [-0.062, 0.000] | -0.038 [-0.090, 0.013] | 0.011 [0.001, 0.021] |
| local_vs_global | full_808 | T0-T1 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.031 [-0.062, -0.003] | -0.056 [-0.104, -0.012] | 0.003 [-0.005, 0.012] |
| local_vs_global | full_808 | T0 | dino_vitb16_imagenet1k_mri_clinical@GLOBAL | dino_vitb16_imagenet1k_mri_clinical@LOCAL | 808 | -0.034 [-0.065, -0.004] | -0.031 [-0.079, 0.014] | 0.009 [-0.001, 0.020] |
| local_vs_global | full_808 | T0-T2 | dino_vitb16_imagenet1k_mri_only@GLOBAL | dino_vitb16_imagenet1k_mri_only@LOCAL | 808 | 0.038 [-0.016, 0.094] | 0.021 [-0.036, 0.082] | -0.046 [-0.070, -0.022] |
| local_vs_global | full_808 | T0-T1 | medicalnet_resnet50_3dseg8_mri_clinical@GLOBAL | medicalnet_resnet50_3dseg8_mri_clinical@LOCAL | 808 | 0.004 [-0.005, 0.013] | 0.005 [-0.018, 0.027] | -0.000 [-0.002, 0.002] |
| local_vs_global | full_808 | T0-T1 | GAP0_mri_only@GLOBAL | LOCAL0_mri_only@LOCAL | 808 | 0.046 [-0.012, 0.104] | 0.043 [-0.009, 0.096] | -0.002 [-0.020, 0.016] |
| local_vs_global | radiomics_complete_case_375 | T0-T2 | dino_vitb16_imagenet1k_mri_clinical_paired@GLOBAL | dino_vitb16_imagenet1k_mri_clinical_paired@LOCAL | 375 | 0.109 [0.023, 0.195] | 0.058 [-0.031, 0.160] | -0.109 [-0.154, -0.065] |
| local_vs_global | radiomics_complete_case_375 | T0-T1 | GAP0_mri_only_paired@GLOBAL | LOCAL0_mri_only_paired@LOCAL | 375 | 0.042 [-0.048, 0.130] | 0.031 [-0.037, 0.093] | -0.050 [-0.083, -0.018] |

其中 `foundation_vs_current_cnn` 中 LOCAL 比较直接对应每个 foundation LOCAL 与 LOCAL0 的同患者差值；这些结果只用于描述 current World Model 是否可能 underuse MRI，不用于事后过滤 foundation 候选。

## HR/HER2 phenotype probes

| 任务 | 模型 | 空间 | 时点 | AUROC | AUPRC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| HER2 | GAP0 | GLOBAL | T0 | 0.547 | 0.284 | 0.255 |
| HER2 | LOCAL0 | LOCAL | T0 | 0.525 | 0.264 | 0.274 |
| HER2 | dino_vitb16_imagenet1k | GLOBAL | T0 | 0.523 | 0.263 | 0.269 |
| HER2 | dino_vitb16_imagenet1k | LOCAL | T0 | 0.557 | 0.299 | 0.253 |
| HER2 | medicalnet_resnet50_3dseg8 | GLOBAL | T0 | 0.519 | 0.274 | 0.250 |
| HER2 | medicalnet_resnet50_3dseg8 | LOCAL | T0 | 0.464 | 0.227 | 0.279 |
| HR | GAP0 | GLOBAL | T0 | 0.538 | 0.581 | 0.254 |
| HR | LOCAL0 | LOCAL | T0 | 0.500 | 0.564 | 0.268 |
| HR | dino_vitb16_imagenet1k | GLOBAL | T0 | 0.522 | 0.583 | 0.283 |
| HR | dino_vitb16_imagenet1k | LOCAL | T0 | 0.596 | 0.629 | 0.274 |
| HR | medicalnet_resnet50_3dseg8 | GLOBAL | T0 | 0.488 | 0.539 | 0.296 |
| HR | medicalnet_resnet50_3dseg8 | LOCAL | T0 | 0.535 | 0.583 | 0.260 |

## HR/HER2 subtype probe

| 模型 | 空间 | 时点 | macro AUROC | macro AUPRC | Brier | 准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| GAP0 | GLOBAL | T0 | 0.536 | 0.279 | 0.749 | 0.303 |
| LOCAL0 | LOCAL | T0 | 0.505 | 0.248 | 0.794 | 0.271 |
| dino_vitb16_imagenet1k | GLOBAL | T0 | 0.541 | 0.282 | 0.784 | 0.345 |
| dino_vitb16_imagenet1k | LOCAL | T0 | 0.564 | 0.301 | 0.813 | 0.339 |
| medicalnet_resnet50_3dseg8 | GLOBAL | T0 | 0.504 | 0.259 | 0.880 | 0.256 |
| medicalnet_resnet50_3dseg8 | LOCAL | T0 | 0.497 | 0.250 | 0.856 | 0.291 |

## FTV / ΔFTV decodability

| 模型 | 空间 | 任务 | 终点 | Spearman | R² | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GAP0 | GLOBAL | delta | T0-T1 | 0.030 | -0.021 | 25.599 | 13.863 |
| GAP0 | GLOBAL | delta | T1-T2 | 0.157 | -0.102 | 18.048 | 10.301 |
| GAP0 | GLOBAL | delta | T2-T3 | 0.037 | -0.060 | 28.198 | 10.061 |
| GAP0 | GLOBAL | static | T0 | 0.305 | 0.018 | 42.280 | 20.312 |
| GAP0 | GLOBAL | static | T1 | 0.409 | 0.064 | 36.446 | 13.750 |
| GAP0 | GLOBAL | static | T2 | 0.169 | -0.026 | 31.500 | 8.326 |
| GAP0 | GLOBAL | static | T3 | 0.061 | -0.029 | 12.589 | 3.357 |
| LOCAL0 | LOCAL | delta | T0-T1 | 0.394 | 0.038 | 24.840 | 13.224 |
| LOCAL0 | LOCAL | delta | T1-T2 | 0.354 | 0.148 | 15.873 | 9.031 |
| LOCAL0 | LOCAL | delta | T2-T3 | 0.171 | -0.012 | 27.553 | 9.721 |
| LOCAL0 | LOCAL | static | T0 | 0.759 | 0.302 | 35.651 | 15.503 |
| LOCAL0 | LOCAL | static | T1 | 0.553 | 0.094 | 35.868 | 11.812 |
| LOCAL0 | LOCAL | static | T2 | 0.405 | 0.002 | 31.075 | 7.972 |
| LOCAL0 | LOCAL | static | T3 | 0.317 | 0.058 | 12.045 | 3.181 |
| dino_vitb16_imagenet1k | GLOBAL | delta | T0-T1 | 0.336 | 0.117 | 23.806 | 15.001 |
| dino_vitb16_imagenet1k | GLOBAL | delta | T1-T2 | 0.297 | 0.079 | 16.497 | 10.304 |
| dino_vitb16_imagenet1k | GLOBAL | delta | T2-T3 | 0.267 | 0.083 | 26.229 | 12.215 |
| dino_vitb16_imagenet1k | GLOBAL | static | T0 | 0.620 | 0.395 | 33.176 | 15.996 |
| dino_vitb16_imagenet1k | GLOBAL | static | T1 | 0.574 | 0.399 | 29.215 | 10.631 |
| dino_vitb16_imagenet1k | GLOBAL | static | T2 | 0.523 | 0.326 | 25.543 | 6.915 |
| dino_vitb16_imagenet1k | GLOBAL | static | T3 | 0.347 | 0.065 | 11.997 | 3.213 |
| dino_vitb16_imagenet1k | LOCAL | delta | T0-T1 | 0.508 | 0.225 | 22.301 | 12.451 |
| dino_vitb16_imagenet1k | LOCAL | delta | T1-T2 | 0.488 | 0.225 | 15.138 | 8.847 |
| dino_vitb16_imagenet1k | LOCAL | delta | T2-T3 | 0.288 | 0.152 | 25.220 | 11.725 |
| dino_vitb16_imagenet1k | LOCAL | static | T0 | 0.834 | 0.623 | 26.184 | 10.549 |
| dino_vitb16_imagenet1k | LOCAL | static | T1 | 0.726 | 0.323 | 31.005 | 9.618 |
| dino_vitb16_imagenet1k | LOCAL | static | T2 | 0.576 | 0.155 | 28.598 | 6.955 |
| dino_vitb16_imagenet1k | LOCAL | static | T3 | 0.373 | 0.113 | 11.687 | 3.169 |
| medicalnet_resnet50_3dseg8 | GLOBAL | delta | T0-T1 | 0.237 | -8.560 | 78.322 | 25.156 |
| medicalnet_resnet50_3dseg8 | GLOBAL | delta | T1-T2 | 0.327 | -10.684 | 58.769 | 16.915 |
| medicalnet_resnet50_3dseg8 | GLOBAL | delta | T2-T3 | 0.207 | -5.920 | 72.053 | 22.548 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T0 | 0.515 | -1.116 | 62.060 | 23.639 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T1 | 0.497 | -1.087 | 54.429 | 16.434 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T2 | 0.253 | -0.974 | 43.695 | 12.040 |
| medicalnet_resnet50_3dseg8 | GLOBAL | static | T3 | 0.135 | -1.736 | 20.524 | 5.225 |
| medicalnet_resnet50_3dseg8 | LOCAL | delta | T0-T1 | 0.140 | -516.792 | 576.414 | 90.319 |
| medicalnet_resnet50_3dseg8 | LOCAL | delta | T1-T2 | 0.360 | -271882.474 | 8965.015 | 494.354 |
| medicalnet_resnet50_3dseg8 | LOCAL | delta | T2-T3 | 0.165 | -82687.432 | 7876.058 | 457.864 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T0 | 0.564 | -3.456 | 90.070 | 33.147 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T1 | 0.342 | -3.686 | 81.563 | 25.629 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T2 | 0.141 | -3.493 | 65.928 | 18.070 |
| medicalnet_resnet50_3dseg8 | LOCAL | static | T3 | 0.153 | -3.009 | 24.844 | 6.594 |
